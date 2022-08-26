import itertools
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.cuda import amp

from src.agents.base_ac_agent import BaseACAgent

from src.utils.torch_utils import USE_CUDA, FLOAT, copy_params, update_params

from src.models.gaussian_actor import GaussianActor
from src.models.critic import Critic
from src.models.lstm_models import GaussianLSTMActor, LSTMCritic
from src.models.gru_models import GaussianGRUActor, GRUCritic


class SACAgent(BaseACAgent):

    def __init__(self, name, env, seed, args):
        super().__init__(name, env, seed, args)

        # temperature and its optimizer
        # for continuous action spaces, target entropy is -dim(action space)
        self.target_entropy = -torch.prod(torch.tensor(self.env.action_space.shape, dtype=FLOAT, device=self.device))
        
        self.alpha_tuning = args.alpha_tuning
        if self.alpha_tuning: # automatic tuning of temperature parameter
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp()
            self.alpha_optim = Adam([self.log_alpha], lr=args.lr_alpha)
        else: # fixed temperature
            self.alpha = args.alpha

        self.reward_scale = args.reward_scale


    def define_actors_critics(self, args):

        num_price_features = self.state_dim[2]
        window_length = self.state_dim[1]
        num_stocks = self.state_dim[0]

        if self.network_type == 'cnn':
            # policy function
            self.actor = GaussianActor(num_price_features, window_length)

            # state-value function (soft q-function)
            self.critic1 = Critic(num_price_features, self.action_dim, window_length)
            self.critic2 = Critic(num_price_features, self.action_dim, window_length)

            # target networks for soft q-functions
            self.critic1_target = Critic(num_price_features, self.action_dim, window_length)
            self.critic2_target = Critic(num_price_features, self.action_dim, window_length)

        elif self.network_type == 'lstm':
            # policy function
            self.actor = GaussianLSTMActor(num_price_features * num_stocks, self.action_dim)

            # state-value function (soft q-function)
            self.critic1 = LSTMCritic(num_price_features * num_stocks, self.action_dim)
            self.critic2 = LSTMCritic(num_price_features * num_stocks, self.action_dim)

            # target networks for soft q-functions
            self.critic1_target = LSTMCritic(num_price_features * num_stocks, self.action_dim)
            self.critic2_target = LSTMCritic(num_price_features * num_stocks, self.action_dim)

        elif self.network_type == 'gru':
            # policy function
            self.actor = GaussianGRUActor(num_price_features * num_stocks, self.action_dim)

            # state-value function (soft q-function)
            self.critic1 = GRUCritic(num_price_features * num_stocks, self.action_dim)
            self.critic2 = GRUCritic(num_price_features * num_stocks, self.action_dim)

            # target networks for soft q-functions
            self.critic1_target = GRUCritic(num_price_features * num_stocks, self.action_dim)
            self.critic2_target = GRUCritic(num_price_features * num_stocks, self.action_dim)

        # optimizers
        self.actor_optim = Adam(self.actor.parameters(), lr=args.lr_actor)
        self.critic1_optim = Adam(self.critic1.parameters(), lr=args.lr_critic)
        self.critic2_optim = Adam(self.critic2.parameters(), lr=args.lr_critic)


    def copy_params_to_target(self):
        copy_params(self.critic1_target, self.critic1)
        copy_params(self.critic2_target, self.critic2)


    def update_target_params(self):
        update_params(self.critic1_target, self.critic1, self.tau)
        update_params(self.critic2_target, self.critic2, self.tau)


    def set_networks_grad(self, networks, requires_grad):
        # networks = {'actor', 'critic', 'target'}
        # requires_grad = {True, False}
        assert networks in ('actor', 'critic', 'target')
        assert requires_grad in (True, False)

        if networks == 'actor':
            self.actor.requires_grad(requires_grad)
        elif networks == 'critic':
            self.critic1.requires_grad(requires_grad)
            self.critic2.requires_grad(requires_grad)
        elif networks == 'target':
            self.critic1_target.requires_grad(requires_grad)
            self.critic2_target.requires_grad(requires_grad)


    def compute_value_loss(self, s_batch, a_batch, r_batch, t_batch, s2_batch):
        mse_loss = nn.MSELoss()

        # soft q values
        q1_batch = self.critic1(*s_batch, a_batch)
        q2_batch = self.critic2(*s_batch, a_batch)

        # bellman backup for q functions

        # target actions come from "current" policy
        a2_batch, logprob_a2_batch = self.actor(*s2_batch)

        # target q-values
        next_q_values1 = self.critic1_target(*s2_batch, a2_batch)
        next_q_values2 = self.critic2_target(*s2_batch, a2_batch)
        next_q_values = torch.min(next_q_values1, next_q_values2)
        if self.alpha_tuning:
            alpha = self.alpha.detach()
        else:
            alpha = self.alpha
        target_q_batch = r_batch + self.gamma * (1 - t_batch) * (next_q_values - alpha * logprob_a2_batch)

        # mse loss against bellman backup
        loss_q1 = mse_loss(q1_batch, target_q_batch)
        loss_q2 = mse_loss(q2_batch, target_q_batch)

        return loss_q1, loss_q2


    def compute_policy_loss(self, s_batch):
        a_pred_batch, logprob_a_pred_batch = self.actor(*s_batch)
        q1_pi = self.critic1(*s_batch, a_pred_batch)
        q2_pi = self.critic2(*s_batch, a_pred_batch)
        q_pi = torch.min(q1_pi, q2_pi)

        # entropy-regularized policy loss
        if self.alpha_tuning:
            alpha = self.alpha.detach()
        else:
            alpha = self.alpha
        policy_loss = torch.mean(alpha * logprob_a_pred_batch - q_pi)

        # entropy = -logprob_a_pred_batch
        return policy_loss, logprob_a_pred_batch


    def compute_entropy_loss(self, logprob_a_pred_batch):
        entropy_loss = -torch.mean(self.alpha * (logprob_a_pred_batch + self.target_entropy).detach())

        return entropy_loss


    def update(self, scaler, wandb_inst, step):

        # sample batch
        s_batch, a_batch, r_batch, t_batch, s2_batch = self.buffer.sample_batch(self.batch_size)

        # reward scaling
        r_batch = self.reward_scale * r_batch

        # set gradients to 0
        self.critic1_optim.zero_grad()
        self.critic2_optim.zero_grad()
        self.actor_optim.zero_grad()
        if self.alpha_tuning:
            self.alpha_optim.zero_grad()

        self.set_networks_grad('actor', False)

        # compute loss for critic
        with amp.autocast():
            loss_q1, loss_q2 = self.compute_value_loss(s_batch, a_batch, r_batch, t_batch, s2_batch)

        # backward pass for critics
        scaler.scale(loss_q1).backward()
        scaler.step(self.critic1_optim)

        scaler.scale(loss_q2).backward()
        scaler.step(self.critic2_optim)

        # freeze critics, unfreze actor
        self.set_networks_grad('actor', True)
        self.set_networks_grad('critic', False)

        # compute loss for actor and for temperature alpha
        with amp.autocast():
            policy_loss, logprob_a_pred_batch = self.compute_policy_loss(s_batch)

        # backward pass for actor
        scaler.scale(policy_loss).backward()
        scaler.step(self.actor_optim)

        if self.alpha_tuning:
            # compute loss for temperature alpha
            with amp.autocast():
                entropy_loss = self.compute_entropy_loss(logprob_a_pred_batch)

            # backward pass for alpha
            scaler.scale(entropy_loss).backward()
            scaler.step(self.alpha_optim)
            self.alpha = self.log_alpha.exp() # update value of alpha to respect changes to log_alpha

        # update
        scaler.update()

        self.set_networks_grad('critic', True)

        # update target networks with polyak averaging
        self.update_target_params()

        # log to wandb
        wandb_inst.log({'loss_q1': loss_q1, 'loss_q2': loss_q1, 'policy_loss': policy_loss,
                        'alpha': self.alpha}, step=step)


    def predict_action(self, obs, exploration=False):
        a,b = obs
        a = torch.tensor(a, dtype=FLOAT, device=self.device)
        b = torch.tensor(b, dtype=FLOAT, device=self.device)
        action, _ = self.actor(a, b, exploration)
        if USE_CUDA:
            action = action.detach().cpu().numpy()
        else:
            action = action.detach().numpy()

        return action

    def load_actor_model(self, path):
        self.actor.load_state_dict(torch.load(path))


    def set_train(self):
        self.actor.train()
        self.critic1.train()
        self.critic1_target.train()
        self.critic2.train()
        self.critic2_target.train()


    def set_eval(self):
        self.actor.eval()
        self.critic1.eval()
        self.critic1_target.eval()
        self.critic2.eval()
        self.critic2_target.eval()

    def cuda(self):
        self.actor.cuda()
        self.critic1.cuda()
        self.critic1_target.cuda()
        self.critic2.cuda()
        self.critic2_target.cuda()