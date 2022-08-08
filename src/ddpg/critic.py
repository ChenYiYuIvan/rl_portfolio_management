import numpy as np
import torch
import torch.nn as nn


class Critic(nn.Module):
    # represents the value function

    def __init__(self):
        super().__init__()