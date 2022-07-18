from src.utils import data_utils

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

        assert self.start_idx >= self.window_length, "Invalid starting date: not enough preceding observations"
        assert self.start_idx <= self.end_idx, "Starting date must be before ending date"
        assert self.end_idx < len(self.date_list), "Invalid ending date: no observations past 2018-02-07 in the dataset"

        if stock_names is None:
            self.stock_names = ["AAPL", "ATVI", "CMCSA", "COST", "CSX", "DISH", "EA", "EBAY", "FB", "GOOGL", "HAS", "ILMN", "INTC", "MAR", "REGN", "SBUX"]
        else:
            self.stock_names = stock_names

        # to avoid data leakage during training, store only the required data
        stock_idxs = [data_utils.stock_names.index(stock) for stock in self.stock_names]
        self.data = data_utils.hist_data[stock_idxs, self.start_idx - self.window_length : self.end_idx, :]
        
        # set current step to 0
        self.reset()

    def step(self):
        self.current_step += 1

        obs = self.data[:, self.current_step : self.current_step + self.window_length, :]
        done = self.current_step >= self.end_idx - self.start_idx  # if true, it means simulation has reached end date
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