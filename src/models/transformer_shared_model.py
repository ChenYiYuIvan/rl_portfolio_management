from turtle import forward
import torch
from torch import nn
from torch.distributions.normal import Normal
from src.models.base_models import BaseModel
from src.utils.data_utils import EPS
from src.models.transformer_model import PositionalEncoder, DeterministicTransformerActor, GaussianTransformerActor, TransformerCritic, DoubleTransformerCritic, TransformerForecaster
import math


class BaseTransformerShared(BaseModel):

    def __init__(self, price_features, window_length, d_model, num_heads, num_layers):
        super().__init__()

        self.input_fc = nn.Linear(price_features, d_model)
        self.positional_encoder = PositionalEncoder(d_model=d_model, max_len=window_length)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers)
        
        self.encoder_fc1 = nn.Linear(d_model, 1)
        self.encoder_fc2 = nn.Linear(window_length, 1)

        self.dropout = nn.Dropout(0.1)

        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        # shape = [batch, window_length, price_features]

        x = self.dropout(self.leaky_relu(self.input_fc(x)))
        # shape = [batch, window_length, d_model]

        x = self.positional_encoder(x)
        # shape = [batch, window_length, d_model]

        x = self.encoder(x)
        # shape = [batch, window_length, d_model]

        x = self.dropout(self.leaky_relu(self.encoder_fc1(x))).squeeze(2)
        # shape = [batch, window_length]

        x = self.dropout(self.leaky_relu(self.encoder_fc2(x)))
        # shape = [batch, 1]

        return x


class BaseActionTransformerShared(BaseModel):

    def __init__(self, price_features, num_stocks, window_length, d_model, num_heads, num_layers):
        super().__init__()

        num_assets = num_stocks + 1

        self.base = BaseTransformerShared(price_features, window_length, d_model, num_heads, num_layers)

        self.base_fc = nn.Linear(num_stocks, d_model)

        self.weights_fc = nn.Linear(num_assets, d_model)

        self.fc = nn.Linear(2*d_model, d_model)

        self.dropout = nn.Dropout(0.1)

        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x, w):

        if len(x.shape) == 3:
            x = x[None,:,:,:]
            w = w[None,:]
        # shape = [batch, window_length, num_stocks, price_features]

        x_stocks = []
        for stock in range(x.shape[2]):
            # shape = [batch, window_length, price_features]
            x_stock = self.base(x[:,:,stock,:])
            # shape = [batch, 1]

            x_stocks.append(x_stock)
        x = torch.cat(x_stocks, dim=1)
        # shape = [batch, num_stocks]

        x = self.dropout(self.leaky_relu(self.base_fc(x)))
        w = self.dropout(self.leaky_relu(self.weights_fc(w)))
        x = torch.cat((x, w), dim=-1)
        # shape = [batch, 2*d_model]
        
        x = self.dropout(self.leaky_relu(self.fc(x)))
        # shape = [batch, d_model]

        return x


class DeterministicTransformerSharedActor(DeterministicTransformerActor):

    def __init__(self, price_features, num_stocks, window_length, d_model, num_heads, num_layers):
        super().__init__(price_features, num_stocks, window_length, d_model, num_heads, num_layers)

        self.common = BaseActionTransformerShared(price_features, num_stocks, window_length, d_model, num_heads, num_layers)


class GaussianTransformerSharedActor(GaussianTransformerActor):

    def __init__(self, price_features, num_stocks, window_length, d_model, num_heads, num_layers):
        super().__init__(price_features, num_stocks, window_length, d_model, num_heads, num_layers)

        self.common = BaseActionTransformerShared(price_features, num_stocks, window_length, d_model, num_heads, num_layers)


class TransformerSharedCritic(TransformerCritic):

    def __init__(self, price_features, num_stocks, window_length, d_model, num_heads, num_layers):
        super().__init__()

        self.common = BaseActionTransformerShared(price_features, num_stocks, window_length, d_model, num_heads, num_layers)


class DoubleTransformerSharedCritic(DoubleTransformerCritic):

    def __init__(self, price_features, num_stocks, window_length, d_model, num_heads, num_layers):
        super().__init__()

        self.Q1 = TransformerSharedCritic(price_features, num_stocks, window_length, d_model, num_heads, num_layers)
        self.Q2 = TransformerSharedCritic(price_features, num_stocks, window_length, d_model, num_heads, num_layers)


class TransformerSharedForecaster(BaseModel):

    def __init__(self, price_features, num_stocks, window_length, d_model, num_heads, num_layers):
        super().__init__()

        self.base = BaseTransformerShared(price_features, window_length, d_model, num_heads, num_layers)

        self.fc1 = nn.Linear(num_stocks, d_model)
        self.fc2 = nn.Linear(d_model, num_stocks)

        self.dropout = nn.Dropout(0.1)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x):

        if len(x.shape) == 3:
            x = x[None,:,:,:]
        # shape = [batch, window_length, num_stocks, price_features]

        x_stocks = []
        for stock in range(x.shape[2]):
            # shape = [batch, window_length, price_features]
            x_stock = self.base(x[:,:,stock,:])
            # shape = [batch, 1]

            x_stocks.append(x_stock)
        x = torch.cat(x_stocks, dim=1)
        # shape = [batch, num_stocks]

        x = self.dropout(self.leaky_relu(self.fc1(x)))

        x = self.fc2(x)

        return x.squeeze(0)
