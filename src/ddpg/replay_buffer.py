"""
Source: https://github.com/vermouth1992/deep-learning-playground/blob/master/tensorflow/ddpg/replay_buffer.py
"""
from collections import deque
import random
from tkinter import N
import numpy as np
import warnings
import torch
from ..utils.torch_utils import USE_CUDA, FLOAT


class ReplayBuffer(object):
    def __init__(self, buffer_size, random_seed=123):
        """
        The right side of the deque contains the most recent experiences
        """
        self.device = torch.device("cuda:0" if USE_CUDA else "cpu")

        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()
        random.seed(random_seed)

    def add(self, s, a, r, t, s2):
        experience = (s, a, r, t, s2)
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        if self.count < batch_size:
            warnings.warn('Not enough entries to sample without replacement. Consider increasing your warm-up phase to avoid oversampling!')
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        # torch.stack -> np.array

        s_b = [_[0] for _ in batch]  # current observation
        s_batch = (torch.tensor(np.array([_[0] for _ in s_b]), dtype=FLOAT, device=self.device),
            torch.tensor(np.array([_[1] for _ in s_b]), dtype=FLOAT, device=self.device))

        a_batch = torch.tensor(np.array([_[1] for _ in batch]), dtype=FLOAT, device=self.device)  # action
        r_batch = torch.tensor(np.array([_[2] for _ in batch]), dtype=FLOAT, device=self.device)  # reward
        t_batch = torch.tensor(np.array([_[3] for _ in batch]), dtype=torch.bool, device=self.device)  # done

        # for consistent shape
        r_batch = r_batch[:, None]
        t_batch = t_batch[:, None]

        s2_b = [_[4] for _ in batch] # next observation
        s2_batch = (torch.tensor(np.array([_[0] for _ in s2_b]), dtype=FLOAT, device=self.device),
            torch.tensor(np.array([_[1] for _ in s2_b]), dtype=FLOAT, device=self.device))

        return s_batch, a_batch, r_batch, t_batch, s2_batch

    def clear(self):
        self.buffer.clear()
        self.count = 0
