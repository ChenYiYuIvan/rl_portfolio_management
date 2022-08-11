import torch
import torch.nn as nn


class Actor(nn.Module):
    # represents the policy function

    def __init__(self, in_channels, window_length):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=2, kernel_size=(1,3))
        self.conv2 = nn.Conv2d(in_channels=2, out_channels=20, kernel_size=(1,window_length-2))
        self.conv3 = nn.Conv2d(in_channels=20+1, out_channels=1, kernel_size=1)

        self.softmax = lambda x: nn.Softmax(dim=x)
        self.relu = nn.ReLU()


    def forward(self, x, w):
        
        x = self.conv1(x)
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

        x = self.softmax(dim_cat)(x)

        return x


    def requires_grad(self, req):
        for param in self.parameters():
            param.requires_grad = req