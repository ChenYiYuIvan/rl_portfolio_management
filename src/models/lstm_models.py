import torch
from torch import nn
from torch.distributions.normal import Normal
from src.models.base_models import BaseModel
from src.utils.data_utils import EPS
import math


class BaseLSTM(BaseModel):
    
    def __init__(self, price_features, num_stocks, window_length, d_model, num_layers):
        super().__init__()
        
        num_input_features = price_features * num_stocks
        
        self.input_fc = nn.Linear(num_input_features, d_model)

        self.lstm = nn.LSTM(input_size=d_model, hidden_size=d_model, num_layers=num_layers, batch_first=True)

        self.lstm_fc1 = nn.Linear(d_model, 1)
        self.lstm_fc2 = nn.Linear(window_length, 1)

        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        
        if len(x.shape) == 3:
            x = x[None,:,:,:]
        # shape = [batch, window_length, num_stocks, price_features]

        x = torch.flatten(x, 2)
        # shape = [batch, window_length, num_stocks * price_features]

        x = self.leaky_relu(self.input_fc(x))
        # shape = [batch, window_length, d_model]

        x, _ = self.lstm(x)
        # shape = [batch, window_length, d_model]

        x = self.leaky_relu(self.lstm_fc1(x)).squeeze(2)
        # shape = [batch, window_length]

        x = self.leaky_relu(self.lstm_fc2(x))
        # shape = [batch, 1]

        return x

class BaseActionLSTM(BaseModel):

    def __init__(self, price_features, num_stocks, window_length, d_model, num_layers):
        super().__init__()

        num_assets = num_stocks + 1

        self.base = BaseLSTM(price_features, num_stocks, window_length, d_model, num_layers)

        self.weights_fc = nn.Linear(num_assets, d_model)

        self.fc = nn.Linear(2*d_model, d_model)

        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x, w):

        if len(w.shape) == 1:
            w = w[None,:]

        x = self.base(x)

        w = self.leaky_relu(self.weights_fc(w))
        x = torch.cat((x, w), dim=-1)
        # shape = [batch, 2*d_model]
        
        x = self.leaky_relu(self.fc(x))
        # shape = [batch, d_model]

        return x


class DeterministicLSTMActor(BaseModel):

    def __init__(self, price_features, num_stocks, window_length, d_model, num_layers):
        super().__init__()

        self.common = BaseActionLSTM(price_features, num_stocks, window_length, d_model, num_layers)

        num_assets = num_stocks + 1
        self.fc = nn.Linear(d_model, num_assets)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, w):

        x = self.common(x, w)

        x = self.fc(x)
        x = self.softmax(x)

        return x.squeeze()


class GaussianLSTMActor(BaseModel):

    def __init__(self, price_features, num_stocks, window_length, d_model, num_layers):
        super().__init__()

        self.common = BaseActionLSTM(price_features, num_stocks, window_length, d_model, num_layers)

        num_assets = num_stocks + 1

        self.fc_mu = nn.Linear(d_model, num_assets)
        self.fc_logstd = nn.Linear(d_model, num_assets)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, w, exploration=True):

        x = self.common(x, w)

        mu = self.fc_mu(x)

        log_std = self.fc_logstd(x)
        std = torch.exp(log_std)

        pi_distribution = Normal(mu, std)
        if exploration:
            xs = pi_distribution.sample()
        else:
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


class LSTMCritic(BaseModel):

    def __init__(self, price_features, num_stocks, window_length, d_model, num_layers):
        super().__init__()

        self.common = BaseActionLSTM(price_features, num_stocks, window_length, d_model, num_layers)

        num_assets = num_stocks + 1

        self.action_fc = nn.Linear(num_assets, d_model)

        self.fc = nn.Linear(2*d_model, 1)
        
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x, w, action):

        if len(x.shape) == 3:
            action = action[None,:]

        x = self.common(x, w)

        action = self.action_fc(action)
        x = torch.cat((x,action), dim=-1)

        x = self.fc(x)

        return x.squeeze(0)


class DoubleLSTMCritic(BaseModel):

    def __init__(self, price_features, num_stocks, window_length, d_model, num_layers):
        super().__init__()

        self.Q1 = LSTMCritic(price_features, num_stocks, window_length, d_model, num_layers)
        self.Q2 = LSTMCritic(price_features, num_stocks, window_length, d_model, num_layers)

    def forward(self, x, w, action):
        # x = state
        # w = past action

        q1 = self.Q1(x, w, action)
        q2 = self.Q2(x, w, action)

        return q1, q2
    
    
class LSTMForecaster(BaseModel):

    def __init__(self, price_features, num_stocks, window_length, d_model, num_layers):
        super().__init__()

        self.base = BaseLSTM(price_features, num_stocks, window_length, d_model, num_layers)

        self.fc1 = nn.Linear(d_model, d_model)
        self.fc2 = nn.Linear(d_model, num_stocks)

        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x):

        x = self.base(x)

        x = self.leaky_relu(self.fc1(x))

        x = self.fc2(x)

        return x.squeeze(0)