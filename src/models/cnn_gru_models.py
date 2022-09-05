import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from src.utils.data_utils import EPS


class DeterministicCNNGRUActor(nn.Module):
    # represents the policy function

    def __init__(self, in_channels, num_stocks):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels*2, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels*2, 1, kernel_size=3)

        self.gru = nn.GRU(num_stocks-4, num_stocks+1, batch_first=True)

        self.fc = nn.Linear((num_stocks+1)*2, num_stocks+1)

        self.leakyrelu = nn.LeakyReLU()
        self.softmax = nn.Softmax(-1)

        self.init_weights()


    def forward(self, x, w):
        # x = state
        # w = current portfolio weights

        if len(x.shape) == 3:
            # single data point - no batches
            x = x[None,:,:,:]
            w = w[None,:]

        x = self.conv1(x)
        x = self.leakyrelu(x)
        x = self.conv2(x)
        x = self.leakyrelu(x)
        _, x = self.gru(x.squeeze(1))
        x = self.leakyrelu(x.squeeze(0))

        # concatenate current portfolio weights
        x = torch.cat((x,w), dim=-1)
        x = self.fc(x)
        x = self.leakyrelu(x)

        x = self.softmax(x)

        return x.squeeze()

    
    def init_weights(self):
        for _, module in self.named_modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, a=0.01)
            elif isinstance(module, nn.GRU):
                for name, m in module.named_modules():
                    if 'bias' in name:
                        nn.init.constant_(m, 0.0)
                    elif 'weight' in name:
                        nn.init.kaiming_normal_(m.weight, a=0.01)


    def requires_grad(self, req):
        for param in self.parameters():
            param.requires_grad = req


LOG_STD_MAX = 2
LOG_STD_MIN = -20

class GaussianCNNGRUActor(nn.Module):
    # represents the policy function

    def __init__(self, in_channels, num_stocks):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels*2, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels*2, 1, kernel_size=3)

        self.gru = nn.GRU(num_stocks-4, num_stocks+1, batch_first=True)

        self.fc_mu = nn.Linear((num_stocks+1)*2, num_stocks+1)
        self.fc_logstd = nn.Linear((num_stocks+1)*2, num_stocks+1)

        self.leakyrelu = nn.LeakyReLU()
        self.softmax = nn.Softmax(-1)

        self.init_weights()


    def forward(self, x, w, exploration=True, with_logprob=True):
        # x = state
        # w = current portfolio weights

        if len(x.shape) == 3:
            # single data point - no batches
            x = x[None,:,:,:]
            w = w[None,:]

        x = self.conv1(x)
        x = self.leakyrelu(x)
        x = self.conv2(x)
        x = self.leakyrelu(x)
        _, x = self.gru(x.squeeze(1))
        x = self.leakyrelu(x.squeeze(0))

        # concatenate current portfolio weights
        x = torch.cat((x,w), dim=-1)

        mu = self.fc_mu(x)
        log_std = self.fc_logstd(x)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # action distribution
        pi_distribution = Normal(mu, std)
        if exploration:
            xs = pi_distribution.sample()
        else:
            # only when testing policy
            xs = mu

        # tanh squashing
        pi_action = torch.tanh(xs)
        if with_logprob: # compute log_probability of pi_action
            log_pi = pi_distribution.log_prob(xs) - torch.log(1 - pi_action.pow(2) + EPS)
            log_pi = log_pi.sum(dim=-1, keepdim=True)
        else:
            log_pi = None

        # transform action to satisfy portfolio constraints
        pi_action = pi_action + 1
        total = pi_action.sum(dim=-1)
        pi_action = pi_action / total[:,None]

        return pi_action.squeeze(), log_pi


    def init_weights(self):
        for _, module in self.named_modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, a=0.01)
            elif isinstance(module, nn.GRU):
                for name, m in module.named_modules():
                    if 'bias' in name:
                        nn.init.constant_(m, 0.0)
                    elif 'weight' in name:
                        nn.init.kaiming_normal_(m.weight, a=0.01)


    def requires_grad(self, req):
        for param in self.parameters():
            param.requires_grad = req


class CNNGRUCritic(nn.Module):
    # represents the policy function

    def __init__(self, in_channels, num_stocks):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels*2, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels*2, 1, kernel_size=3)

        self.gru = nn.GRU(num_stocks-4, num_stocks+1, batch_first=True)

        self.fc1 = nn.Linear((num_stocks+1)*2, num_stocks+1)
        self.fc2 = nn.Linear((num_stocks+1)*2, 1)

        self.leakyrelu = nn.LeakyReLU()

        self.init_weights()


    def forward(self, x, w, action):
        # x = state
        # w = past action

        if len(x.shape) == 3:
            # single data point - no batches
            x = x[None,:,:,:]
            w = w[None,:]
            action = action[None,:]

        x = self.conv1(x)
        x = self.leakyrelu(x)
        x = self.conv2(x)
        x = self.leakyrelu(x)
        _, x = self.gru(x.squeeze(1))
        x = self.leakyrelu(x.squeeze(0))

        # concatenate current portfolio weights
        x = torch.cat((x,w), dim=-1)
        x = self.fc1(x)
        x = self.leakyrelu(x)

        # concatenate action
        x = torch.cat((x,action), dim=-1)
        x = self.fc2(x)
        x = self.leakyrelu(x)

        return x


    def init_weights(self):
        for _, module in self.named_modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, a=0.01)
            elif isinstance(module, nn.GRU):
                for name, m in module.named_modules():
                    if 'bias' in name:
                        nn.init.constant_(m, 0.0)
                    elif 'weight' in name:
                        nn.init.kaiming_normal_(m.weight, a=0.01)


    def requires_grad(self, req):
        for param in self.parameters():
            param.requires_grad = req


class DoubleCNNGRUCritic(nn.Module):
    # double critics for sac

    def __init__(self, in_channels, num_stocks):
        super().__init__()

        self.Q1 = CNNGRUCritic(in_channels, num_stocks)
        self.Q2 = CNNGRUCritic(in_channels, num_stocks)


    def forward(self, x, w, action):
        # x = state
        # w = current portfolio weights

        q1 = self.Q1(x, w, action)
        q2 = self.Q2(x, w, action)

        return q1, q2


    def requires_grad(self, req):
        self.Q1.requires_grad(req)
        self.Q2.requires_grad(req)