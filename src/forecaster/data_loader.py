import torch
from torch.utils.data import Dataset
import numpy as np
from src.environments.market import Market
from src.utils.torch_utils import USE_CUDA, FLOAT


class StockDataset(Dataset):

    def __init__(self, agent, start_date, end_date, window_length, stock_names, mode):
        super().__init__()

        self.device = torch.device("cuda:0" if USE_CUDA else "cpu")

        # parameters
        self.window_length = window_length
        self.stock_names = stock_names[1:]

        # market environment
        self.market = Market(start_date, end_date, window_length, self.stock_names)

        # rl agent - for preprocessing
        self.agent = agent

        # train / eval
        self.mode = mode

    def __len__(self):
        return self.market.tot_steps

    def __getitem__(self, idx):
        self.market.next_step = idx
        curr_obs, next_obs, _ = self.market.step()

        (curr_obs, _) = self.agent.preprocess_data((curr_obs, None))
        (next_obs, _) = self.agent.preprocess_data((next_obs, None))

        obs = torch.tensor(curr_obs, dtype=FLOAT, device=self.device)
        truth = torch.tensor(next_obs[1:,-1,2], dtype=FLOAT, device=self.device)
        if self.agent.normalize:
            # have to use non-normalized truth value
            mean_close = torch.tensor(self.agent.mean[:,3], dtype=FLOAT, device=self.device)
            std_close = torch.tensor(self.agent.std[:,3], dtype=FLOAT, device=self.device)
            truth = truth * std_close + mean_close

        return obs, truth