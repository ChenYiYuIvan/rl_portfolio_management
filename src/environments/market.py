from src.utils import data_utils
from copy import deepcopy

class Market():
    
    def __init__(self, start_date, end_date, window_length = 30, stock_names = None):
        """
        Environment to provide historical market data for each episode.
        Params:
            start_date - first day of time window to be considered
            end_date - last day of time window to be considered
            window_length - observation window
            stock_names - names of stocks to be considered
        """

        self.start_date = start_date
        self.end_date = end_date
        self.window_length = window_length

        self.date_list = data_utils.date_list
        self.start_idx = self.date_list.index(self.start_date)
        self.end_idx = self.date_list.index(self.end_date)

        assert self.start_idx >= self.window_length - 1, "Invalid starting date: not enough preceding observations"
        assert self.start_idx <= self.end_idx, "Starting date must be before ending date"
        # assert self.end_idx < len(self.date_list), "Invalid ending date: no observations past 2018-02-07 in the dataset"

        self.stock_names = stock_names

        # store only the required data
        stock_idxs = [data_utils.stock_names.index(stock) for stock in self.stock_names]
        stock_idxs = [0, *stock_idxs]
        self.data = data_utils.hist_data[stock_idxs, self.start_idx - self.window_length + 1 : self.end_idx + 2, :]

        # S&P500 ETF values
        self.snp = data_utils.snp[self.start_idx - self.window_length + 1 : self.end_idx + 1, :]
        
        # set current step to 0
        self.reset()

    def step(self):
        # execute 1 time step within the environment
        curr_obs = deepcopy(self.data[:, self.next_step : self.next_step + self.window_length, :])
        next_obs = deepcopy(self.data[:, self.next_step + 1 : self.next_step + self.window_length + 1, :])

        self.next_step += 1
        done = self.next_step >= self.end_idx - self.start_idx + 1  # if true, it means simulation has reached end date

        return curr_obs, next_obs, done


    def reset(self):
        # reset environment to initial state
        self.next_step = 0

        curr_obs = deepcopy(self.data[:, self.next_step : self.next_step + self.window_length, :])

        return curr_obs


    def step_to_date(self, step = None):
        # return date of given step

        if step is None:
            step = self.next_step - 1

        return self.date_list[step + self.start_idx]