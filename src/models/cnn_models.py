from tkinter.messagebox import NO
import torch
import torch.nn as nn
from src.utils.torch_utils import FLOAT


class DeterministicCNNActor(nn.Module):
    # represents the policy function

    def __init__(self, price_features, num_stocks, window_length):
        super().__init__()

        self.conv1 = nn.Conv2d(price_features, 2, kernel_size=(1,3))
        self.conv2 = nn.Conv2d(in_channels=2, out_channels=20, kernel_size=(1,window_length-2))
        self.conv3 = nn.Conv2d(in_channels=20+1, out_channels=1, kernel_size=1)

        self.fc1 = nn.Linear(num_stocks, num_stocks+1)
        self.fc2 = nn.Linear(num_stocks+1, num_stocks+1)

        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU()


    def forward(self, x, w):
        # x = state
        # w = past action

        if len(x.shape) == 3:
            # single data point - no batches
            x = x[None,:,:,:]
            w = w[None,None,1:,None] # remove cash value
        else:
            w = w[:,None,1:,None]
        
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)

        x = torch.cat((x, w), dim=-3)
        x = self.conv3(x)

        # remove unused dimensions
        x = x.squeeze(3)
        x = x.squeeze(1)
        
        # cash_bias = torch.ones(x.shape[0], 1, dtype=FLOAT)
        # if x.is_cuda:
        #     cash_bias = cash_bias.cuda()
        
        # x = torch.cat((x,cash_bias), dim=-1)

        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.softmax(x)

        return x.squeeze()


    def requires_grad(self, req):
        for param in self.parameters():
            param.requires_grad = req


class CNNCritic(nn.Module):
    # represents the policy function

    def __init__(self, price_features, num_stocks, window_length):
        super().__init__()

        self.conv1 = nn.Conv2d(price_features, 2, kernel_size=(1,3))
        self.conv2 = nn.Conv2d(in_channels=2, out_channels=20, kernel_size=(1,window_length-2))
        self.conv3 = nn.Conv2d(in_channels=20+1, out_channels=1, kernel_size=1)

        self.fc1 = nn.Linear(num_stocks, num_stocks+1)
        self.fc2 = nn.Linear((num_stocks+1)*2, 1)

        self.relu = nn.ReLU()


    def forward(self, x, w, action):
        # x = state
        # w = past action

        if len(x.shape) == 3:
            # single data point - no batches
            x = x[None,:,:,:]
            w = w[None,None,1:,None] # remove cash value
            action = action[None,:]
        else:
            w = w[:,None,1:,None]
        
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)

        x = torch.cat((x, w), dim=-3)
        x = self.conv3(x)

        # remove unused dimensions
        x = x.squeeze(3)
        x = x.squeeze(1)
        
        # cash_bias = torch.ones(x.shape[0], 1, dtype=FLOAT)
        # if x.is_cuda:
        #     cash_bias = cash_bias.cuda()
        
        # x = torch.cat((x,cash_bias), dim=-1)

        x = self.fc1(x)
        x = self.relu(x)

        x = torch.cat((x,action), dim=-1)
        x = self.fc2(x)

        return x


    def requires_grad(self, req):
        for param in self.parameters():
            param.requires_grad = req