import torch
from torch import nn
from torch.distributions.normal import Normal
from src.models.base_models import BaseModel
from src.utils.data_utils import EPS
import math


class PositionalEncoder(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=50):
        """
        Taken from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        """

        super().__init__()

        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]

        return self.dropout(x)


class BaseTransformer(BaseModel):

    def __init__(self, price_features, num_stocks, window_length, d_model, num_heads, num_layers):
        super().__init__()

        num_input_features = price_features * num_stocks

        self.input_fc = nn.Linear(num_input_features, d_model)
        self.positional_encoder = PositionalEncoder(d_model=d_model, max_len=window_length)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers)
        
        self.encoder_conv = nn.Conv1d(d_model, 1, kernel_size=1)
        self.encoder_fc = nn.Linear(window_length, d_model)

        self.relu = nn.ReLU()

    def forward(self, x):

        if len(x.shape) == 3:
            x = x[None,:,:,:]
        # shape = [batch, window_length, num_stocks, price_features]

        x = torch.flatten(x, 2)
        # shape = [batch, window_length, num_stocks * price_features]

        x = self.relu(self.input_fc(x))
        # shape = [batch, window_length, d_model]

        x = self.positional_encoder(x)
        # shape = [batch, window_length, d_model]

        x = self.encoder(x)
        # shape = [batch, window_length, d_model]

        x = torch.transpose(x, 1, 2)
        # shape = [batch, d_model, window_length]

        x = self.encoder_conv(x).squeeze(1)
        # shape = [batch, window_length]

        x = self.relu(self.encoder_fc(x))
        # shape = [batch, d_model]

        return x


class BaseActionTransformer(BaseModel):

    def __init__(self, price_features, num_stocks, window_length, d_model, num_heads, num_layers):
        super().__init__()

        num_assets = num_stocks + 1

        self.base = BaseTransformer(price_features, num_stocks, window_length, d_model, num_heads, num_layers)

        self.weights_fc = nn.Linear(num_assets, d_model)

        self.fc = nn.Linear(2*d_model, d_model)

        self.relu = nn.ReLU()

    def forward(self, x, w):

        if len(w.shape) == 1:
            w = w[None,:]

        x = self.base(x)

        w = self.relu(self.weights_fc(w))
        x = torch.cat((x, w), dim=-1)
        # shape = [batch, 2*d_model]
        
        x = self.relu(self.fc(x))
        # shape = [batch, d_model]

        return x


class DeterministicTransformerActor(BaseModel):

    def __init__(self, price_features, num_stocks, window_length, d_model, num_heads, num_layers):
        super().__init__()

        self.common = BaseActionTransformer(price_features, num_stocks, window_length, d_model, num_heads, num_layers)

        num_assets = num_stocks + 1
        self.fc = nn.Linear(d_model, num_assets)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, w):

        x = self.common(x, w)

        x = self.fc(x)
        x = self.softmax(x)

        return x.squeeze()


class GaussianTransformerActor(BaseModel):

    def __init__(self, price_features, num_stocks, window_length, d_model, num_heads, num_layers):
        super().__init__()

        self.common = BaseActionTransformer(price_features, num_stocks, window_length, d_model, num_heads, num_layers)

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


class TransformerCritic(BaseModel):

    def __init__(self, price_features, num_stocks, window_length, d_model, num_heads, num_layers):
        super().__init__()

        self.common = BaseActionTransformer(price_features, num_stocks, window_length, d_model, num_heads, num_layers)

        num_assets = num_stocks + 1

        self.action_fc = nn.Linear(num_assets, d_model)

        self.fc = nn.Linear(2*d_model, 1)

    def forward(self, x, w, action):

        if len(x.shape) == 3:
            action = action[None,:]

        x = self.common(x, w)

        action = self.action_fc(action)
        x = torch.cat((x,action), dim=-1)

        x = self.fc(x)

        return x.squeeze(0)


class DoubleTransformerCritic(BaseModel):

    def __init__(self, price_features, num_stocks, window_length, d_model, num_heads, num_layers):
        super().__init__()

        self.Q1 = TransformerCritic(price_features, num_stocks, window_length, d_model, num_heads, num_layers)
        self.Q2 = TransformerCritic(price_features, num_stocks, window_length, d_model, num_heads, num_layers)

    def forward(self, x, w, action):
        # x = state
        # w = past action

        q1 = self.Q1(x, w, action)
        q2 = self.Q2(x, w, action)

        return q1, q2


class TransformerForecaster(BaseModel):

    def __init__(self, price_features, num_stocks, window_length, d_model, num_heads, num_layers):
        super().__init__()

        self.base = BaseTransformer(price_features, num_stocks, window_length, d_model, num_heads, num_layers)

        self.fc1 = nn.Linear(d_model, d_model)
        self.fc2 = nn.Linear(d_model, num_stocks)

        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.base(x)

        x = self.relu(self.fc1(x))

        x = self.fc2(x)

        return x.squeeze(0)
