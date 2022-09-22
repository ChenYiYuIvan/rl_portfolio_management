#from src.utils import data_utils
from copy import deepcopy
import numpy as np
import yfinance as yf
import datetime as dt

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

        self.stock_names = stock_names

        # stocks data
        _date_fmt = '%Y-%m-%d'
        self.data = np.stack([
            yf.download(stock,
                start=(dt.datetime.strptime(self.start_date, _date_fmt) +
                        dt.timedelta(1)).strftime(_date_fmt),
                end=(dt.datetime.strptime(self.end_date, _date_fmt) + dt.timedelta(1)).strftime(_date_fmt)
                ).drop(columns=['Adj Close']).to_numpy()
            for stock in self.stock_names
        ])

        # add cash data
        _cash_data = np.ones((1, self.data.shape[1], self.data.shape[2]))
        self.data = np.concatenate((_cash_data, self.data), axis=0)
        self.stock_names = ['CASH', *self.stock_names]

        # market data
        self.snp = yf.download('SPY',
                start=(dt.datetime.strptime(self.start_date, _date_fmt) +
                        dt.timedelta(1)).strftime(_date_fmt),
                end=(dt.datetime.strptime(self.end_date, _date_fmt) + dt.timedelta(1)).strftime(_date_fmt))

        self.date_list = [str(time).split(' ')[0] for time in self.snp.index]
        self.snp = self.snp.drop(columns=['Adj Close']).to_numpy()

        # set current step to 0
        self.tot_steps = len(self.date_list) - self.window_length
        self.reset()

    def step(self):
        # execute 1 time step within the environment
        curr_obs = deepcopy(self.data[:, self.next_step : self.next_step + self.window_length, :])
        next_obs = deepcopy(self.data[:, self.next_step + 1 : self.next_step + self.window_length + 1, :])

        self.next_step += 1
        done = self.next_step - self.start_step >= self.max_steps  # if true, it means simulation has reached end date

        return curr_obs, next_obs, done


    def reset(self, random_start=False, max_steps=None):
        # reset environment to initial state
        if random_start and max_steps is not None:
            self.start_step = np.random.randint(0, self.tot_steps - max_steps + 1)
            self.max_steps = max_steps
        else:
            self.start_step = 0
            self.max_steps = self.tot_steps
        self.next_step = self.start_step

        curr_obs = deepcopy(self.data[:, self.next_step : self.next_step + self.window_length, :])

        return curr_obs


    def step_to_date(self, step = None):
        # return date of given step

        if step is None:
            step = self.next_step - 1

        return self.date_list[step + self.window_length - 1]