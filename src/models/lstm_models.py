import torch
import torch.nn as nn


class LSTMActor(nn.Module):
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

        x = torch.transpose(x, 1, 3) # so as to have [batch, time window, stock, high/low/close]
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

        x = torch.transpose(x, 1, 3) # so as to have [batch, time window, stock, high/low/close]
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