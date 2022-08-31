import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.cuda import amp

from src.agents.base_ac_agent import BaseACAgent

from src.utils.torch_utils import USE_CUDA, FLOAT, copy_params, update_params

from src.models.actor import Actor
from src.models.critic import Critic
from src.models.lstm_models import DeterministicLSTMActor, LSTMCritic
from src.models.gru_models import DeterministicGRUActor, GRUCritic
from src.models.noise import OrnsteinUhlenbeckActionNoise


class DDPGAgent(BaseACAgent):

    def __init__(self, name, env, seed, args):
        super().__init__(name, env, seed, args)

        # noise for exploration
        self.actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(self.action_dim))


    def define_actors_critics(self, args):
        
        num_price_features = self.state_dim[2]
        window_length = self.state_dim[1]
        num_stocks = self.state_dim[0]
        
        if self.network_type == 'cnn':
            self.actor = Actor(num_price_features, window_length)
            self.actor_target = Actor(num_price_features, window_length)
            
            self.critic = Critic(num_price_features, self.action_dim, window_length)
            self.critic_target = Critic(num_price_features, self.action_dim, window_length)
            
        elif self.network_type == 'lstm':
            # add 1 to output_size for cash dimension
            self.actor = DeterministicLSTMActor(num_price_features * num_stocks, self.action_dim)
            self.actor_target = DeterministicLSTMActor(num_price_features * num_stocks, self.action_dim)

            self.critic = LSTMCritic(num_price_features * num_stocks, self.action_dim)
            self.critic_target = LSTMCritic(num_price_features * num_stocks, self.action_dim)

        elif self.network_type == 'gru':
            # add 1 to output_size for cash dimension
            self.actor = DeterministicGRUActor(num_price_features * num_stocks, self.action_dim)
            self.actor_target = DeterministicGRUActor(num_price_features * num_stocks, self.action_dim)

            self.critic = GRUCritic(num_price_features * num_stocks, self.action_dim)
            self.critic_target = GRUCritic(num_price_features * num_stocks, self.action_dim)
            
        self.actor_optim = Adam(self.actor.parameters(), lr=args.lr_actor)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr_critic, weight_decay=1e-2)


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


    def update(self, scaler, wandb_inst, step):

        # sample batch
        s_batch, a_batch, r_batch, t_batch, s2_batch = self.buffer.sample_batch(self.batch_size)

        # set gradients to 0
        self.critic_optim.zero_grad()
        self.actor_optim.zero_grad()

        self.set_networks_grad('actor', False)

        # compute loss for critic
        with amp.autocast():
            value_loss = self.compute_value_loss(s_batch, a_batch, r_batch, t_batch, s2_batch)

        # backward pass for critic
        scaler.scale(value_loss).backward()
        scaler.step(self.critic_optim)

        self.set_networks_grad('actor', True)
        self.set_networks_grad('critic', False)

        # compute loss for actor
        with amp.autocast():
            policy_loss = self.compute_policy_loss(s_batch)

        # backward pass for actor
        scaler.scale(policy_loss).backward()
        scaler.step(self.actor_optim)

        # update
        scaler.update()

        self.set_networks_grad('critic', True)

        # update target netweoks
        self.update_target_params()

        # log to wandb
        wandb_inst.log({'value_loss': value_loss, 'policy_loss': policy_loss}, step=step)


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
        if action.sum() == 0:
            action[0] = 1
        else:
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