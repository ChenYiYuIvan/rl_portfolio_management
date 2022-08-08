import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.cuda import amp

from ..utils.torch_utils import USE_CUDA, FLOAT, copy_params, update_params

from .actor import Actor
from .critic import Critic
from .replay_buffer import ReplayBuffer
from .noise import OrnsteinUhlenbeckActionNoise


class DDPG():

    def __init__(self, state_dim, action_dim, args):
        
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.actor = Actor()
        self.actor_target = Actor()
        self.actor_optim = Adam(self.actor.parameters(), lr=args.lr_actor)

        self.critic = Critic()
        self.critic_target = Critic()
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr_critic)

        self.loss = nn.MSELoss()

        # target networks must have same initial weights as originals
        copy_params(self.actor_target, self.actor)
        copy_params(self.critic_target, self.critic)

        self.buffer = ReplayBuffer(args.buffer_size)

        # noise for exploration
        self.actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(self.action_dim))

        # hyper-parameters
        self.batch_size = args.batch_size
        self.tau = args.tau
        self.discount = args.discount
        self.depsilon = 1.0 / args.epsilon

        # 
        self.epsilon = 1.0
        self.s_t = None # Most recent state
        self.a_t = None # Most recent action
        self.is_training = True

        # 
        if USE_CUDA:
            self.cuda()


    def update_policy(self):

        scaler = amp.GradScaler()

        # sample batch
        s_batch, a_batch, r_batch, t_batch, s2_batch = self.buffer.sample_batch(self.batch_size)

        # prepare target q batch
        next_q_values = self.critic_target([s2_batch, self.actor_target(s2_batch)])
        target_q_batch = r_batch + self.discount * t_batch * next_q_values

        # set gradients to 0
        self.critic_optim.zero_grad()
        self.actor_optim.zero_grad()

        # update critic
        with amp.autocast():
            q_batch = self.critic([s_batch, a_batch])  #maybe torch.cat instead of []
            value_loss = self.loss(q_batch, target_q_batch)

        # backward pass for critic
        scaler.scale(value_loss).backward()
        scaler.step(self.critic_optim)

        # update actor
        with amp.autocast():
            policy_loss = -self.critic([s_batch, self.actor(s_batch)])
            policy_loss = policy_loss.mean()

        # backward pass for actor
        scaler.scale(policy_loss).backward()
        scaler.step(self.actor_optim)

        # update target netweoks
        update_params(self.actor_target, self.actor, self.tau)
        update_params(self.critic_target, self.critic, self.tau)


    def eval(self):
        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()


    def cuda(self):
        self.actor.cuda()
        self.actor_target.cuda()
        self.critic.cuda()
        self.critic_target.cuda()