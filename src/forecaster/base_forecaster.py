import numpy as np
import matplotlib.pyplot as plt
from src.utils.data_utils import EPS


class BaseForecaster:

    def __init__(self, name, preprocess, market_train):
        self.name = name
        self.preprocess = preprocess

        # market environments
        self.market_train = market_train
        self.model = None

    def _forecast(self, obs):
        raise NotImplementedError

    def forecast_all(self, market, render=False):

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

        # mse
        pred_vec = np.array(pred_vec)
        truth_vec = np.array(truth_vec)
        mse = ((pred_vec - truth_vec)**2).mean()

        if render:
            # plot prediction vs actual values
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

        return mse

    def plot_all(self, pred_vec_train, truth_vec_train, pred_vec_test, truth_vec_test):

        num_train = pred_vec_train.shape[0]
        num_test = pred_vec_test.shape[0]

        mse_train = self._calculate_mse(pred_vec_train, truth_vec_train)
        mse_test = self._calculate_mse(pred_vec_test, truth_vec_test)

        pred_vec = np.concatenate((pred_vec_train, pred_vec_test), axis=0)
        truth_vec = np.concatenate((truth_vec_train, truth_vec_test), axis=0)
        fig, arr = plt.subplots(len(self.market_train.stock_names[1:]), 1, squeeze=False)
        for i, stock in enumerate(self.market_train.stock_names[1:]):

            predictions = [el[i] for el in pred_vec]
            truths = [el[i] for el in truth_vec]
            arr[i,0].plot(truths)
            arr[i,0].plot(predictions)

            arr[i,0].axvspan(0, num_train, facecolor='b', alpha=0.3)
            arr[i,0].axvspan(num_train, num_train+num_test, facecolor='r', alpha=0.3)
            arr[i,0].title.set_text(f'{stock}')

        fig.suptitle(f'{self.preprocess} - mse_train = {mse_train:.6f} - mse_test = {mse_test:.6f}')
        fig.legend(['truth', 'pred'])
        plt.show()

    def _calculate_mse(self, pred_vec, truth_vec):
        return np.mean((pred_vec - truth_vec)**2)

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