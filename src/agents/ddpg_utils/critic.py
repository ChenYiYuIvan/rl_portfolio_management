import torch
import torch.nn as nn


class Critic(nn.Module):
    # represents the value function

    def __init__(self, in_channels, num_assets, window_length):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=2, kernel_size=(1,3))
        self.conv2 = nn.Conv2d(in_channels=2, out_channels=20, kernel_size=(1,window_length-2))
        self.conv3 = nn.Conv2d(in_channels=20+1, out_channels=1, kernel_size=1)

        self.fc1 = nn.Linear(num_assets, num_assets)
        self.fc2 = nn.Linear(2 * num_assets, 1)

        self.relu = nn.ReLU()


    def forward(self, s, w, action):
        
        x = self.conv1(s)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        
        if len(w.shape) > 1:  # for batches
            w_reshape = torch.reshape(w, (w.shape[0], 1, w.shape[1], 1))
            dim_cat = 1
        else:
            w_reshape = torch.reshape(w, (1, w.shape[0], 1))
            dim_cat = 0
        x = self.conv3(torch.cat((x, w_reshape), dim=dim_cat)).squeeze()
        
        a = self.fc1(action)

        x = torch.cat((x, a), dim=dim_cat)
        x = self.relu(x)
        x = self.fc2(x)

        return x


    def requires_grad(self, req):
        for param in self.parameters():
            param.requires_grad = req