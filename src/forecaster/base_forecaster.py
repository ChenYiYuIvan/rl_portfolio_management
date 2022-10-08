import numpy as np
import matplotlib.pyplot as plt
from src.utils.data_utils import EPS


class BaseForecaster:

    def __init__(self, name, preprocess, market_train):
        self.name = name
        self.preprocess = preprocess

        # market environments
        self.market_train = market_train

    def _forecast(self, obs):
        raise NotImplementedError

    def forecast_all(self, market):

        curr_obs = market.reset()

        pred_vec, truth_vec = [], []
        pred_price_vec, truth_price_vec = [curr_obs[1:,-1,3]], [curr_obs[1:,-1,3]]

        done = False
        while not done:
            curr_obs, next_obs, done = market.step()

            pred = self._forecast(curr_obs)
            pred_price = self._price_from_value(pred, truth_price_vec[-1], next_obs[1:,:,3])
            truth_price = next_obs[1:,-1,3]
            truth = self._value_from_price(truth_price, truth_price_vec[-1], next_obs[1:,:,3])
            
            pred_vec.append(pred)
            pred_price_vec.append(pred_price)
            truth_price_vec.append(truth_price)
            truth_vec.append(truth)

        fig, arr = plt.subplots(len(market.stock_names[1:]), 2, squeeze=False)
        for i, stock in enumerate(market.stock_names[1:]):

            predictions = [el[i] for el in pred_vec]
            truths = [el[i] for el in truth_vec]
            arr[i,0].plot(truths)
            arr[i,0].plot(predictions)
            arr[i,0].title.set_text(f'{stock} - {self.preprocess}')

            predictions_price = [el[i] for el in pred_price_vec]
            truths_price = [el[i] for el in truth_price_vec]
            arr[i,1].plot(truths_price[1:])
            arr[i,1].plot(predictions_price[1:])
            arr[i,1].title.set_text(f'{stock} - price')

        fig.legend(['truth', 'pred'])
        plt.show()

    def _price_from_value(self, value, past_price, close_prices):
        
        if self.preprocess == 'log_return':
            price = (past_price + EPS) * np.exp(value) - EPS
        elif self.preprocess == 'simple_return':
            price = past_price * (value + 1)
        elif self.preprocess == 'minmax':
            max_val = np.max(close_prices, axis=1)
            min_val = np.min(close_prices, axis=1)
            price = value * (max_val - min_val + EPS) + min_val

        return price

    def _value_from_price(self, price, past_price, close_prices):

        if self.preprocess == 'log_return':
            value = np.log(price + EPS) - np.log(past_price + EPS)
        elif self.preprocess == 'simple_return':
            value = np.divide(price, past_price) - 1
        elif self.preprocess == 'minmax':
            max_price = np.max(close_prices, axis=1)
            min_price = np.min(close_prices, axis=1)
            value = (price - min_price) - (max_price - min_price + EPS)

        return value