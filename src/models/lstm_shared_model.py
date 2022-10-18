import torch
from torch import nn
from src.models.base_models import BaseModel


class BaseLSTMShared(BaseModel):

    def __init__(self, price_features, window_length, d_model, num_layers):
        super().__init__()

        self.input_fc = nn.Linear(price_features, d_model)

        self.lstm = nn.LSTM(input_size=d_model, hidden_size=d_model, num_layers=num_layers, batch_first=True)

        self.lstm_fc1 = nn.Linear(d_model, 1)
        self.lstm_fc2 = nn.Linear(window_length, 1)

        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        # shape = [batch, window_length, price_features]

        x = self.leaky_relu(self.input_fc(x))
        # shape = [batch, window_length, d_model]

        x, _ = self.lstm(x)
        # shape = [batch, window_length, d_model]

        x = self.leaky_relu(self.lstm_fc1(x)).squeeze(2)
        # shape = [batch, window_length]

        x = self.leaky_relu(self.lstm_fc2(x))
        # shape = [batch, 1]

        return x


class LSTMSharedForecaster(BaseModel):

    def __init__(self, price_features, num_stocks, window_length, d_model, num_layers):
        super().__init__()

        num_assets = num_stocks + 1

        self.base = BaseLSTMShared(price_features, window_length, d_model, num_layers)

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
