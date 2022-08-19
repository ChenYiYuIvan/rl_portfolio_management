import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from src.agents.base_ac_agent import BaseACAgent

from src.utils.torch_utils import USE_CUDA, FLOAT, copy_params, update_params

from src.models.actor import Actor
from src.models.critic import Critic
from src.models.noise import OrnsteinUhlenbeckActionNoise


class DDPGAgent(BaseACAgent):

    def __init__(self, name, env, seed, args):
        super().__init__(name, env, seed, args)

        # noise for exploration
        self.actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(self.action_dim))


    def define_actors_critics(self, args):
        self.actor = Actor(self.state_dim[2], self.state_dim[1])
        self.actor_target = Actor(self.state_dim[2], self.state_dim[1])
        self.actor_optim = Adam(self.actor.parameters(), lr=args.lr_actor)

        self.critic = Critic(self.state_dim[2], self.action_dim, self.state_dim[1])
        self.critic_target = Critic(self.state_dim[2], self.action_dim, self.state_dim[1])
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr_critic)


    def copy_params_to_target(self):
        copy_params(self.actor_target, self.actor)
        copy_params(self.critic_target, self.critic)


    def update_target_params(self):
        update_params(self.actor_target, self.actor, self.tau)
        update_params(self.critic_target, self.critic, self.tau)

    def set_networks_grad(self, networks, requires_grad):
        # networks = {'actor', 'critic', 'target'}
        # requires_grad = {True, False}
        assert networks in ('actor', 'critic', 'target')
        assert requires_grad in (True, False)

        if networks == 'actor':
            self.actor.requires_grad(requires_grad)
        elif networks == 'critic':
            self.critic.requires_grad(requires_grad)
        elif networks == 'target':
            self.actor_target.requires_grad(requires_grad)
            self.critic_target.requires_grad(requires_grad)


    def compute_value_loss(self, s_batch, a_batch, r_batch, t_batch, s2_batch):
        mse_loss = nn.MSELoss()

        # prepare target q batch
        next_q_values = self.critic_target(*s2_batch, self.actor_target(*s2_batch))
        target_q_batch = r_batch + self.gamma * (1 - t_batch) * next_q_values

        # update critic
        q_batch = self.critic(*s_batch, a_batch)
        value_loss = mse_loss(q_batch, target_q_batch)

        return value_loss


    def compute_policy_loss(self, s_batch):
        policy_loss = self.critic(*s_batch, self.actor(*s_batch))
        policy_loss = -torch.mean(policy_loss)

        return policy_loss


    def predict_action(self, obs, exploration=False):
        a,b = obs
        a = torch.tensor(a, dtype=FLOAT, device=self.device)
        b = torch.tensor(b, dtype=FLOAT, device=self.device)
        action = self.actor(a, b)
        if USE_CUDA:
            action = action.detach().cpu().numpy()
        else:
            action = action.detach().numpy()

        if exploration: # add exploration
            action += self.actor_noise()
        
        action = np.clip(action, 0, 1)
        action /= action.sum()

        return action


    def set_train(self):
        self.actor.train()
        self.actor_target.train()
        self.critic.train()
        self.critic_target.train()


    def set_eval(self):
        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()


    def cuda(self):
        self.actor.cuda()
        self.actor_target.cuda()
        self.critic.cuda()
        self.critic_target.cuda()