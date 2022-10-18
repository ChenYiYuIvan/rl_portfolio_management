import torch
from torch import nn
from src.models.base_models import BaseModel
from src.models.transformer_model import PositionalEncoder, DeterministicTransformerActor, GaussianTransformerActor, TransformerCritic, DoubleTransformerCritic, TransformerForecaster


class BaseTransformerShared(BaseModel):

    def __init__(self, price_features, window_length, d_model, num_heads, num_layers):
        super().__init__()

        self.input_fc = nn.Linear(price_features, d_model)
        self.positional_encoder = PositionalEncoder(d_model=d_model, max_len=window_length)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers)
        
        self.encoder_fc1 = nn.Linear(d_model, 1)
        self.encoder_fc2 = nn.Linear(window_length, 1)

        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        # shape = [batch, window_length, price_features]

        x = self.leaky_relu(self.input_fc(x))
        # shape = [batch, window_length, d_model]

        x = self.positional_encoder(x)
        # shape = [batch, window_length, d_model]

        x = self.encoder(x)
        # shape = [batch, window_length, d_model]

        x = self.leaky_relu(self.encoder_fc1(x)).squeeze(2)
        # shape = [batch, window_length]

        x = self.leaky_relu(self.encoder_fc2(x))
        # shape = [batch, 1]

        return x


class BaseActionTransformerShared(BaseModel):

    def __init__(self, price_features, num_stocks, window_length, d_model, num_heads, num_layers):
        super().__init__()

        num_assets = num_stocks + 1

        self.base = BaseTransformerShared(price_features, window_length, d_model, num_heads, num_layers)

        self.base_fc = nn.Linear(num_assets, d_model)

        self.weights_fc = nn.Linear(num_assets, d_model)

        self.fc = nn.Linear(2*d_model, d_model)

        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x, w):

        if len(x.shape) == 3:
            x = x[None,:,:,:]
            w = w[None,:]
        # shape = [batch, num_assets, window_length, price_features]

        batch_size = x.shape[0]
        num_assets = x.shape[1]
        x = torch.reshape(x, (batch_size*num_assets, x.shape[2], x.shape[3]))
        # shape = [batch * num_assets, window_length, price_features]

        x = self.base(x)
        # shape = [batch * num_assets]

        x = torch.reshape(x, (batch_size, num_assets))
        # shape = [batch, num_assets]

        x = self.leaky_relu(self.base_fc(x))
        w = self.leaky_relu(self.weights_fc(w))
        x = torch.cat((x, w), dim=-1)
        # shape = [batch, 2*d_model]
        
        x = self.leaky_relu(self.fc(x))
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
        super().__init__(price_features, num_stocks, window_length, d_model, num_heads, num_layers)

        self.common = BaseActionTransformerShared(price_features, num_stocks, window_length, d_model, num_heads, num_layers)


class DoubleTransformerSharedCritic(DoubleTransformerCritic):

    def __init__(self, price_features, num_stocks, window_length, d_model, num_heads, num_layers):
        super().__init__(price_features, num_stocks, window_length, d_model, num_heads, num_layers)

        self.Q1 = TransformerSharedCritic(price_features, num_stocks, window_length, d_model, num_heads, num_layers)
        self.Q2 = TransformerSharedCritic(price_features, num_stocks, window_length, d_model, num_heads, num_layers)


class TransformerSharedForecaster(BaseModel):

    def __init__(self, price_features, num_stocks, window_length, d_model, num_heads, num_layers):
        super().__init__()

        num_assets = num_stocks + 1

        self.base = BaseTransformerShared(price_features, window_length, d_model, num_heads, num_layers)

        self.fc1 = nn.Linear(num_assets, d_model)
        self.fc2 = nn.Linear(d_model, num_stocks)

        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x):

        if len(x.shape) == 3:
            x = x[None,:,:,:]
        # shape = [batch, num_assets, window_length, price_features]

        batch_size = x.shape[0]
        num_assets = x.shape[1]
        x = torch.reshape(x, (batch_size*num_assets, x.shape[2], x.shape[3]))
        # shape = [batch * num_assets, window_length, price_features]

        x = self.base(x)
        # shape = [batch * num_assets]

        x = torch.reshape(x, (batch_size, num_assets))
        # shape = [batch, num_assets]

        x = self.leaky_relu(self.fc1(x))

        x = self.fc2(x)

        return x.squeeze(0)
