import numpy as np
import matplotlib.pyplot as plt


class BaseForecaster:

    def __init__(self, name, preprocess, market_train):
        self.name = name
        self.preprocess = preprocess

        # market environments
        self.market_train = market_train
        self.model = None

    def _forecast(self, obs):
        raise NotImplementedError

    def plot_all(self, pred_vec_train, truth_vec_train, pred_vec_test, truth_vec_test):

        num_train = pred_vec_train.shape[0]
        num_test = pred_vec_test.shape[0]

        rmse_train = self._calculate_rmse(pred_vec_train, truth_vec_train)
        rmse_test = self._calculate_rmse(pred_vec_test, truth_vec_test)

        pred_vec = np.concatenate((pred_vec_train, pred_vec_test), axis=0)
        truth_vec = np.concatenate((truth_vec_train, truth_vec_test), axis=0)
        fig, arr = plt.subplots(len(self.market_train.stock_names[1:]), 1, squeeze=False)
        for i, stock in enumerate(self.market_train.stock_names[1:]):

            predictions = [el[i] for el in pred_vec]
            truths = [el[i] for el in truth_vec]
            arr[i,0].plot(truths)
            arr[i,0].plot(predictions)

            arr[i,0].axvspan(0, num_train, facecolor='g', alpha=0.3)
            arr[i,0].axvspan(num_train, num_train+num_test, facecolor='r', alpha=0.3)
            arr[i,0].set_xlim(left=0, right=num_train+num_test)
            arr[i,0].title.set_text(f'{stock}')

        fig.suptitle(f'{self.preprocess} - rmse_train = {rmse_train:.6f} - rmse_test = {rmse_test:.6f}')
        fig.legend(['truth', 'pred'])
        plt.show()

    def _calculate_mse(self, pred_vec, truth_vec):
        return np.mean((pred_vec - truth_vec)**2)

    def _calculate_rmse(self, pred_vec, truth_vec):
        return np.sqrt(self._calculate_mse(pred_vec, truth_vec))