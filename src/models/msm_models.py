from turtle import forward
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from src.models.base_models import BaseModel
from src.utils.data_utils import EPS


class FirstSM(BaseModel):

    def __init__(self, num_price_featues, num_hidden_features):
        super().__init__()

        kernel_size = 3

        self.conv1 = nn.Conv1d(num_price_featues, num_hidden_features, kernel_size)
        self.conv2 = nn.Conv1d(num_hidden_features, num_hidden_features, kernel_size)

        self.gru = nn.GRU(num_hidden_features, num_hidden_features, batch_first=True)

        self.fc = nn.Linear(num_hidden_features, 1)

        self.max = nn.MaxPool1d(kernel_size)

        self.relu = nn.ReLU()

    def forward(self, x):

        if len(x.shape) == 2:
            # single data point - no batches
            x = x[None,:,:]
        # shape = [batch, 3, time window]

        x = self.conv1(x)
        x = self.relu(x)
        # shape = [batch, num_hidden_features, time window - 2]

        x = self.conv2(x)
        x = self.relu(x)
        # shape = [batch, num_hidden_features, time window - 4]

        x = self.max(x)
        # time_steps = int((window_length - 2*(kernel_size-1)) / 3)
        # shape = [batch, num_hidden_features, time_steps]

        x = torch.transpose(x, 1, 2)
        # shape = [batch, time_steps, num_hidden_features]

        _, x = self.gru(x)
        x = self.relu(x.squeeze(0))
        # shape = [batch, num_hidden_features]

        x = self.fc(x)
        # shape = [batch, 1]

        return x


class SecondSM(BaseModel):

    def __init__(self, num_price_featues, num_hidden_features):
        super().__init__()

        self.conv1 = nn.Conv2d(num_price_featues, num_hidden_features, kernel_size=(2,3))
        self.conv2 = nn.Conv2d(num_hidden_features, num_hidden_features, kernel_size=(1,3))

        self.gru = nn.GRU(num_hidden_features, num_hidden_features, batch_first=True)

        self.fc = nn.Linear(num_hidden_features, 1)

        self.max = nn.MaxPool2d(kernel_size=(1,3))

        self.relu = nn.ReLU()

    def forward(self, x):
        # x = bivariate time series of a specific stock

        if len(x.shape) == 3:
            # single data point - no batches
            x = x[None,:,:,:]
        # shape = [batch, 3, 2, time window]

        x = self.conv1(x)
        x = self.relu(x)
        # shape = [batch, num_hidden_features, 1, time window - 2]

        x = self.conv2(x)
        x = self.relu(x)
        # shape = [batch, num_hidden_features, 1, time window - 4]

        x = self.max(x)
        # time_steps = int((window_length - 2*(kernel_size-1)) / 3)
        # shape = [batch, num_hidden_features, 1, time_steps]

        x = torch.transpose(x.squeeze(2), 1, 2)
        # shape = [batch, time_steps, num_hidden_features]

        _, x = self.gru(x)
        x = self.relu(x.squeeze(0))
        # shape = [batch, num_hidden_features]

        x = self.fc(x)
        # shape = [batch, 1]

        return x


class BaseMSM(BaseModel):

    def __init__(self, price_features, num_stocks, window_length):
        super().__init__()

        self.first_sm = FirstSM(price_features, num_stocks)
        self.second_sm = SecondSM(price_features, num_stocks)

        num_scores = int(num_stocks * (1 + (num_stocks-1) / 2))
        num_assets = num_stocks + 1

        self.fc1 = nn.Linear(num_scores, num_assets)
        self.fc2 = nn.Linear(2*num_assets, 4*num_assets)
        self.fc3 = nn.Linear(4*num_assets, 2*num_assets)

        self.relu = nn.ReLU()

    def forward(self, x, w):
        # x = state
        # w = past action

        if len(x.shape) == 3:
            # single data point - no batches
            x = x[None,:,:,:]
            w = w[None,:]
        # x shape = [batch, price features, num stocks, window length]
        # w shape = [batch, stock&cash weights]

        scores = None

        num_stocks = x.shape[2]
        for stock_id in range(num_stocks):
            score1 = self.first_sm(x[:,:,stock_id,:])
            if scores is None:
                scores = score1
            else:
                scores = torch.cat((scores, score1), dim=-1)

            for stock_id2 in range(stock_id+1, num_stocks):
                score_ids = [stock_id, stock_id2]
                score2 = self.second_sm(x[:,:,score_ids,:])
                scores = torch.cat((scores, score2), dim=-1)
        # scores shape = [batch, num_scores]

        x = self.fc1(scores)
        x = self.relu(x)
        # shape = [batch, num_assets]

        # concatenate past action to state
        x = torch.cat((x,w), dim=-1)
        x = self.fc2(x)
        x = self.relu(x)
        # shape = [batch, 4*num_assets]

        x = self.fc3(x)
        x = self.relu(x)
        # shape = [batch, 2*num_assets]

        return x


class DeterministicMSMActor(BaseModel):
    # represents the policy function

    def __init__(self, price_features, num_stocks, window_length):
        super().__init__()

        self.common = BaseMSM(price_features, num_stocks, window_length)

        num_assets = num_stocks + 1

        self.fc = nn.Linear(2*num_assets, num_assets)

        self.relu = nn.ReLU()

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, w):
        
        x = self.common(x, w)

        x = self.fc(x)
        x = self.softmax(x)

        return x.squeeze()


class GaussianMSMActor(BaseModel):

    def __init__(self, price_features, num_stocks, window_length):
        super().__init__()

        self.common = BaseMSM(price_features, num_stocks, window_length)

        num_assets = num_stocks + 1

        self.fc_mu = nn.Linear(2*num_assets, num_assets)
        self.fc_logstd = nn.Linear(2*num_assets, num_assets)

        self.relu = nn.ReLU()

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, w, exploration=True):
        
        x = self.common(x, w)

        mu = self.fc_mu(x)
        log_std = self.fc_logstd(x)
        #log_std = torch.clamp(log_std, -20, 2)
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
        
        # compute log_probability of pi_action
        log_pi = pi_distribution.log_prob(xs) - torch.log(1 - pi_action.pow(2) + EPS)
        log_pi = log_pi.sum(dim=-1, keepdim=True)

        #x = self.softmax(pi_action)
        x = (pi_action + 1) / 2
        total = x.sum(dim=-1)
        x /= total[:,None]

        return x.squeeze(), log_pi


class MSMCritic(BaseModel):
    # represents the value function

    def __init__(self, price_features, num_stocks, window_length):
        super().__init__()

        self.common = BaseMSM(price_features, num_stocks, window_length)

        num_assets = num_stocks + 1
        self.fc = nn.Linear(3*num_assets, 1)

    def forward(self, x, w, action):
        
        x = self.common(x, w)
        # shape = [batch, 2*num_assets]

        x = torch.cat((x,action), dim=-1)
        # shape = [batch, 3*num_assets]

        x = self.fc(x)
        # shape = [batch, 1]

        return x


class DoubleMSMCritic(BaseModel):

    def __init__(self, price_features, num_stocks, window_length):
        super().__init__()

        self.Q1 = MSMCritic(price_features, num_stocks, window_length)
        self.Q2 = MSMCritic(price_features, num_stocks, window_length)

    def forward(self, x, w, action):
        # x = state
        # w = past action

        q1 = self.Q1(x, w, action)
        q2 = self.Q2(x, w, action)

        return q1, q2