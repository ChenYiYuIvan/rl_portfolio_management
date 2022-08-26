import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from src.utils.data_utils import EPS


class DeterministicLSTMActor(nn.Module):
    # represents the policy function

    def __init__(self, input_size, output_size):
        super().__init__()

        self.lstm = nn.LSTM(input_size, output_size, num_layers=2, batch_first=True)
        
        self.fc1 = nn.Linear(2*output_size,128) # input = state + past action
        self.fc2 = nn.Linear(128,64)
        self.fc3 = nn.Linear(64,32)
        self.fc4 = nn.Linear(32,output_size)

        self.leaky_relu = nn.LeakyReLU()
        self.softmax = nn.Softmax(-1)


    def forward(self, x, w):
        # x = state
        # w = past action

        if len(x.shape) == 3:
            # single data point - no batches
            x = x[None,:,:,:]
            w = w[None,:]

        #x = torch.transpose(x, 1, 3) # so as to have [batch, time window, stock, high/low/close]
        x = x.flatten(2) # so as to have [batch, time window, features]

        _, (x,_) = self.lstm(x)
        x = self.leaky_relu(x[-1,:,:]) # take only hidden state of last layer

        # concatenate past action to state
        x = torch.cat((x,w), dim=-1)

        x = self.fc1(x)
        x = self.leaky_relu(x)
        x = self.fc2(x)
        x = self.leaky_relu(x)
        x = self.fc3(x)
        x = self.leaky_relu(x)
        x = self.fc4(x)
        x = self.softmax(x)

        return x.squeeze()


    def requires_grad(self, req):
        for param in self.parameters():
            param.requires_grad = req


LOG_STD_MAX = 2
LOG_STD_MIN = -20

class GaussianLSTMActor(nn.Module):
    # represents the policy function

    def __init__(self, input_size, output_size):
        super().__init__()

        self.lstm = nn.LSTM(input_size, output_size, num_layers=2, batch_first=True)
        
        self.fc1 = nn.Linear(2*output_size,128) # input = state + past action
        self.fc2 = nn.Linear(128,64)
        self.fc3 = nn.Linear(64,32)

        self.fc_mu = nn.Linear(32, output_size)
        self.fc_logstd = nn.Linear(32, output_size)

        self.leaky_relu = nn.LeakyReLU()
        self.softmax = nn.Softmax(-1)


    def forward(self, x, w, exploration=True, with_logprob=True):
        # x = state
        # w = past action

        if len(x.shape) == 3:
            # single data point - no batches
            x = x[None,:,:,:]
            w = w[None,:]

        #x = torch.transpose(x, 1, 3) # so as to have [batch, time window, stock, high/low/close]
        x = x.flatten(2) # so as to have [batch, time window, features]

        _, (x,_) = self.lstm(x)
        x = self.leaky_relu(x[-1,:,:]) # take only hidden state of last layer

        # concatenate past action to state
        x = torch.cat((x,w), dim=-1)

        x = self.fc1(x)
        x = self.leaky_relu(x)
        x = self.fc2(x)
        x = self.leaky_relu(x)
        x = self.fc3(x)
        x = self.leaky_relu(x)

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


    def requires_grad(self, req):
        for param in self.parameters():
            param.requires_grad = req


class LSTMCritic(nn.Module):
    # represents the policy function

    def __init__(self, input_size, output_size):
        super().__init__()

        self.lstm = nn.LSTM(input_size, output_size, num_layers=2, batch_first=True)
        
        self.fc1 = nn.Linear(3*output_size,128) # input = state + past action + action
        self.fc2 = nn.Linear(128,64)
        self.fc3 = nn.Linear(64,32)
        self.fc4 = nn.Linear(32,16)
        self.fc5 = nn.Linear(16,1)

        self.leaky_relu = nn.LeakyReLU()


    def forward(self, x, w, action):
        # x = state
        # w = past action

        if len(x.shape) == 3:
            # single data point - no batches
            x = x[None,:,:,:]
            w = w[None,:]
            action = action[None,:]

        #x = torch.transpose(x, 1, 3) # so as to have [batch, time window, stock, high/low/close]
        x = x.flatten(2) # so as to have [batch, time window, features]
        
        _, (x,_) = self.lstm(x)
        x = self.leaky_relu(x[-1,:,:]) # take only hidden state of last layer

        # concatenate past action to state
        x = torch.cat((x,w,action), dim=-1)

        x = self.fc1(x)
        x = self.leaky_relu(x)
        x = self.fc2(x)
        x = self.leaky_relu(x)
        x = self.fc3(x)
        x = self.leaky_relu(x)
        x = self.fc4(x)
        x = self.leaky_relu(x)
        x = self.fc5(x)

        return x


    def requires_grad(self, req):
        for param in self.parameters():
            param.requires_grad = req


class DoubleLSTMCritic(nn.Module):
    # double critics for sac

    def __init__(self, input_size, output_size):
        super().__init__()

        self.Q1 = LSTMCritic(input_size, output_size)
        self.Q2 = LSTMCritic(input_size, output_size)


    def forward(self, x, w, action):
        # x = state
        # w = past action

        q1 = self.Q1(x, w, action)
        q2 = self.Q2(x, w, action)

        return q1, q2


    def requires_grad(self, req):
        self.Q1.requires_grad(req)
        self.Q2.requires_grad(req)