import numpy as np
import torch
import torch.nn as nn


class Actor(nn.Module):
    # represents the policy function

    def __init__(self):
        super().__init__()