import torch
import torch.nn as nn
from torch.optim import Adam
from torch.cuda import amp

from src.utils.torch_utils import FLOAT, USE_CUDA

from agents.ddpg_agent import DDPGAgent

from src.models.msm_models import DeterministicMSMActor, DoubleMSMCritic
from src.models.transformer_model import DeterministicTransformerActor, DoubleTransformerCritic
from src.models.noise import NormalActionNoise


class TD3Agent(DDPGAgent):

    def __init__(self, name, env, seed, args):
        super().__init__(name, env, seed, args)

        self.policy_delay = args.policy_delay
        self.update_iters = 0

        # noise for exploration
        self.policy_noise = args.policy_noise
        self.actor_noise = NormalActionNoise(self.action_dim, sigma=self.policy_noise)

    def define_actors_critics(self, args):
        num_price_features = self.state_dim[2]
        window_length = self.state_dim[1]
        num_stocks = self.state_dim[0]

        if self.preprocess == 'log_return':
            window_length -= 1 # because log returns instead of actual prices
        
        if self.network_type == 'msm':
            self.actor = DeterministicMSMActor(num_price_features, num_stocks, window_length)
            self.actor_target = DeterministicMSMActor(num_price_features, num_stocks, window_length)
            
            self.critic = DoubleMSMCritic(num_price_features, num_stocks, window_length)
            self.critic_target = DoubleMSMCritic(num_price_features, num_stocks, window_length)

        elif self.network_type == 'trans':
            # close - high - low - volume
            num_price_features = 4

            self.actor = DeterministicTransformerActor(num_price_features, num_stocks, window_length, d_model=64, num_heads=8, num_layers=3)
            self.actor_target = DeterministicTransformerActor(num_price_features, num_stocks, window_length, d_model=64, num_heads=8, num_layers=3)

            self.critic = DoubleTransformerCritic(num_price_features, num_stocks, window_length, d_model=64, num_heads=8, num_layers=3)
            self.critic_target = DoubleTransformerCritic(num_price_features, num_stocks, window_length, d_model=64, num_heads=8, num_layers=3)

        self.actor_optim = Adam(self.actor.parameters(), lr=args.lr_actor)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr_critic, weight_decay=1e-2)

    def compute_value_loss(self, s_batch, a_batch, r_batch, t_batch, s2_batch):
        mse_loss = nn.MSELoss()

        # target actor action + noise
        noise = torch.randn_like(a_batch) * self.policy_noise
        a2_batch = (self.actor_target(*s2_batch) + noise).clamp(0, 1)
        sum_weights = torch.sum(a2_batch, dim=-1)[:,None] # for correct shape
        a2_batch = torch.div(a2_batch, sum_weights) # make sure weights sum up to 1

        # target q value
        next_q_values1, next_q_values2 = self.critic_target(*s2_batch, a2_batch)
        next_q_values = torch.min(next_q_values1, next_q_values2)
        target_q_batch = r_batch + self.gamma * (1 - t_batch) * next_q_values

        # current q value
        q_batch1, q_batch2 = self.critic(*s_batch, a_batch)

        # value loss
        value_loss1 = mse_loss(q_batch1, target_q_batch)
        value_loss2 = mse_loss(q_batch2, target_q_batch)

        return value_loss1 + value_loss2

    def compute_policy_loss(self, s_batch):
        policy_loss, _ = self.critic(*s_batch, self.actor(*s_batch))
        policy_loss = -torch.mean(policy_loss)

        return policy_loss

    def update(self, scaler, wandb_inst, step, pretrained_path=None):
        
        self.update_iters += 1
        pretrained = pretrained_path is not None

        # sample batch
        s_batch, a_batch, r_batch, t_batch, s2_batch = self.buffer.sample_batch(self.batch_size)

        # set gradients to 0
        self.critic_optim.zero_grad()
        self.actor_optim.zero_grad()

        self.set_networks_grad('actor', False, pretrained)

        # compute loss for critic
        with amp.autocast():
            value_loss = self.compute_value_loss(s_batch, a_batch, r_batch, t_batch, s2_batch)

        # backward pass for critic
        scaler.scale(value_loss).backward()
        scaler.step(self.critic_optim)

        # log to wandb
        wandb_inst.log({'value_loss': value_loss}, step=step)

        self.set_networks_grad('actor', True, pretrained)

        if self.update_iters % self.policy_delay == 0:

            self.set_networks_grad('critic', False, pretrained)

            # compute loss for actor
            with amp.autocast():
                policy_loss = self.compute_policy_loss(s_batch)

            # backward pass for actor
            scaler.scale(policy_loss).backward()
            scaler.step(self.actor_optim)

            # log to wandb
            wandb_inst.log({'policy_loss': policy_loss}, step=step)

            self.set_networks_grad('critic', True, pretrained)

            # update target netweoks
            self.update_target_params()

        # update
        scaler.update()
