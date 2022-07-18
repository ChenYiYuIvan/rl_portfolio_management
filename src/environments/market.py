from src.utils.data_utils import get_date_list
import numpy as np
import csv

class Market():
    
    def __init__(self,
                 start_date,
                 end_date,
                 window_length = 30,
                 stock_names = None
                 ) -> None:
        """
        Environment to provide historical market data.
        Params:
            start_date - first day of time window to be considered
            end_date - last day of time window to be considered
            window_length - observation window
            stock_names - names of stocks to be considered
        """

        self.start_date = start_date
        self.end_date = end_date
        self.window_length = window_length

        self.date_list = get_date_list()
        self.start_idx = self.date_list.index(self.start_date)
        self.end_idx = self.date_list.index(self.end_date)

        assert self.start_idx >= self.window_length, "Invalid starting date: not enough preceding observations"
        assert self.start_idx <= self.end_idx, "Starting date must be before ending date"
        assert self.end_idx < len(self.date_list), "Invalid ending date: no observations past 2018-02-07 in the dataset"

        if stock_names is None:
            self.stock_names = ["AAPL", "ATVI", "CMCSA", "COST", "CSX", "DISH", "EA", "EBAY", "FB", "GOOGL", "HAS", "ILMN", "INTC", "MAR", "REGN", "SBUX"]
        else:
            self.stock_names = stock_names

        hist_data = [] # [num_obs, 7, num_stocks]
        for stock in self.stock_names:
            file_path = f'data/{stock}_data.csv'

            stock_data = []
            with open(file_path, 'r') as f:
                data = csv.reader(f)
                header = next(data)
                for row in data:
                    # memorize only necessary observations
                    date_idx = self.date_list.index(row[0])
                    if date_idx >= self.start_idx - self.window_length and date_idx <= self.end_idx:
                        row[1:5] = [float(num) for num in row[1:5]] # open, high, low, close
                        row[5] = int(row[5]) # volume
                        stock_data.append(row[1:6])

            hist_data.append(stock_data)

        self.data = np.array(hist_data)
        
        # set current step to 0
        self.reset()

    def step(self):
        self.current_step += 1

        obs = self.data[:, self.current_step : self.current_step + self.window_length, :]
        done = self.current_step >= self.end_idx - self.start_idx + 1 # if true, it means simulation has reached end date
        return obs, done


    def reset(self):
        self.current_step = 0

        obs = self.data[:, self.current_step : self.current_step + self.window_length, :]
        return obs


    def step_to_date(self, step = None):
        # return date of given step

        if step is None:
            step = self.current_step

        return self.date_list[step + self.start_idx]