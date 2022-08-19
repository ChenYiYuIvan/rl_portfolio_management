import itertools
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from src.agents.base_ac_agent import BaseACAgent

from src.utils.torch_utils import USE_CUDA, FLOAT, copy_params, update_params

from src.models.gaussian_actor import GaussianActor
from src.models.critic import Critic


class SACAgent(BaseACAgent):

    def __init__(self, name, env, seed, args):
        super().__init__(name, env, seed, args)

        self.alpha = args.alpha


    def define_actors_critics(self, args):
        # policy function
        self.actor = GaussianActor(self.state_dim[2], self.state_dim[1])

        # state-value function (soft q-function)
        self.critic1 = Critic(self.state_dim[2], self.action_dim, self.state_dim[1])
        self.critic2 = Critic(self.state_dim[2], self.action_dim, self.state_dim[1])

        # target networks for soft q-functions
        self.critic1_target = Critic(self.state_dim[2], self.action_dim, self.state_dim[1])
        self.critic2_target = Critic(self.state_dim[2], self.action_dim, self.state_dim[1])

        # optimizers
        critic_params = itertools.chain(self.critic1.parameters(), self.critic2.parameters())
        self.actor_optim = Adam(self.actor.parameters(), lr=args.lr_actor)
        self.critic_optim = Adam(critic_params, lr=args.lr_critic)


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
        a2_batch, logp_a2_pred_batch = self.actor(*s2_batch)

        # target q-values
        q1_pi_targ = self.critic1_target(*s2_batch, a2_batch)
        q2_pi_targ = self.critic2_target(*s2_batch, a2_batch)
        q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
        backup = r_batch + self.gamma * (1 - t_batch) * (q_pi_targ - self.alpha * logp_a2_pred_batch)

        # mse loss against bellman backup
        loss_q1 = mse_loss(q1_batch, backup)
        loss_q2 = mse_loss(q2_batch, backup)
        loss_q = loss_q1 + loss_q2

        return loss_q


    def compute_policy_loss(self, s_batch):
        pred_batch, log_prob_pred_batch = self.actor(*s_batch)
        q1_pi = self.critic1(*s_batch, pred_batch)
        q2_pi = self.critic2(*s_batch, pred_batch)
        q_pi = torch.min(q1_pi, q2_pi)

        # entropy-regularized policy loss
        loss_pi = (self.alpha * log_prob_pred_batch - q_pi).mean()

        return loss_pi


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