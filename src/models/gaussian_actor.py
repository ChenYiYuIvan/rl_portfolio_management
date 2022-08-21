import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from src.utils.torch_utils import FLOAT


# TODO: improve code

LOG_STD_MAX = 2
LOG_STD_MIN = -20

class GaussianActor(nn.Module):
    # represents the policy function

    def __init__(self, in_channels, window_length):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=2, kernel_size=(1,3))
        self.conv2 = nn.Conv2d(in_channels=2, out_channels=20, kernel_size=(1,window_length-2))
        self.conv3 = nn.Conv2d(in_channels=20+1, out_channels=1, kernel_size=1)

        self.softmax = lambda x: nn.Softmax(dim=x)
        self.relu = nn.ReLU()

        self.mu_layer = nn.Linear(17,17)
        self.log_std_layer = nn.Linear(17,17)


    def forward(self, x, w, exploration=True, with_logprob=True):
        
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)

        if len(w.shape) > 1:  # for batches
            w = w[:, 1:]
            w_reshape = torch.reshape(w, (w.shape[0], 1, w.shape[1], 1))
            dim_cat = 1
            cash_bias = torch.zeros(w.shape[0], 1, dtype=FLOAT, device=x.get_device())
        else:
            w = w[1:]
            w_reshape = torch.reshape(w, (1, w.shape[0], 1))
            dim_cat = 0
            cash_bias = torch.zeros(1, dtype=FLOAT, device=x.get_device())
        x = self.conv3(torch.cat((x, w_reshape), dim=dim_cat)).squeeze()

        x = torch.cat((cash_bias, x), dim=dim_cat)

        x = self.softmax(dim_cat)(x)

        mu = self.mu_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if not exploration:
            # Only used for evaluating policy at test time.
            xs = mu
        else:
            xs = pi_distribution.rsample()

        pi_action = torch.tanh(xs)
        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.

            #logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1, keepdim=True)
            #logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=-1, keepdim=True)
            
            logp_pi = pi_distribution.log_prob(xs) - torch.log(1 - pi_action.pow(2) + 1e-8)
            logp_pi = logp_pi.sum(dim=-1, keepdim=True)
        else:
            logp_pi = None

        #pi_action = torch.tanh(pi_action) # in [-1, 1]

        #pi_action = self.act_limit * pi_action # in [-lim, lim]
        pi_action = (pi_action + 1)
        div = pi_action.sum(axis=-1)
        if dim_cat == 1:
            pi_action = pi_action.transpose(0,1)
            pi_action = torch.div(pi_action, div)
            pi_action = pi_action.transpose(0,1)

            #logp_pi = torch.reshape(logp_pi, (logp_pi.shape[0], 1))
        else:
            pi_action /= div

        return pi_action, logp_pi


    def requires_grad(self, req):
        for param in self.parameters():
            param.requires_grad = req