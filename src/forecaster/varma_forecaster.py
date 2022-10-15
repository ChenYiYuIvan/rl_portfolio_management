from src.forecaster.base_forecaster import BaseForecaster
from src.utils.data_utils import prices_to_logreturns
from statsmodels.tsa.statespace.varmax import VARMAX
from src.utils.data_utils import EPS
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import pickle


class VARMAForecaster(BaseForecaster):

    def __init__(self, name, preprocess, market_train, market_test):
        super().__init__(name, preprocess, market_train)

        # from previous tests: only high prices granger-cause close prices
        self.data = self._process_data(market_test)

    def train(self, save_path, maxlag=5, numfolds=5):
        import warnings
        warnings.filterwarnings("ignore")

        print('Start train')

        # creating directory to store models if it doesn't exist
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        mse_results = np.zeros((maxlag, maxlag, numfolds))

        # create train-val-test splits for cross-validation
        data_train = self.data[:self.id_sep]
        data_test = self.data[self.id_sep:]
        k, m = divmod(data_train.shape[0], numfolds)
        #splits = [data_train[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(numfolds)]
        splits = [(i+1)*k+min(i+1, m) for i in range(5)]

        for fold in range(numfolds-1):
            split_train = data_train[0:splits[fold]]
            split_test = data_train[splits[fold]:splits[fold+1]]

            for p in range(maxlag):
                for q in range(maxlag):
                    if p + q > 0:
                        print(f'Currently: fold = {fold}, p = {p}, q = {q}', end=' - ')
                        model = self._fit_model(split_train, p, q)
                        print('Start val', end=' - ')
                        mse = self._eval(model, split_test)
                        mse_results[p,q,fold] = mse
                        print(f'Loss = {mse}')

        mse_mean = mse_results.mean(axis=-1)

        best_idx = mse_mean.argmax()
        best_p = best_idx // maxlag - 1
        best_q = best_idx % maxlag

        print(f'Best model found: p = {best_p}, q = {best_q}')
        self.model = self._fit_model(data_train, best_p, best_q)

        # mse on train data
        obj_cols = [f'{stock}_close' for stock in self.market_train.stock_names[1:]]
        pred_vec_train = self.model.predict()[obj_cols].values
        truth_vec_train = data_train[obj_cols].values
        mse_train = self._calculate_mse(pred_vec_train, truth_vec_train)

        # mse on test data
        mse_test, pred_vec_test, truth_vec_test = self._eval(self.model, data_test)

        # save model
        with open(f'{save_path}/varma_p{best_p}_q{best_q}.pkl', 'wb') as outp:
            pickle.dump(self.__dict__, outp, pickle.HIGHEST_PROTOCOL)

        # plot results
        print(f'Train loss = {mse_train} - Test loss = {mse_test}')
        self.plot_all(pred_vec_train, truth_vec_train, pred_vec_test, truth_vec_test)

        return (best_p, best_q), mse_train, mse_test

    def load_model(self, path):
        self.__dict__.clear()
        with open(path, 'rb') as inp:
            self.__dict__.update(pickle.load(inp))

    def _fit_model(self, data, p, q):
        model = VARMAX(data, order=(p,q))
        model_fit = model.fit(disp=False)
        return model_fit

    def _eval(self, model, data):
        # assuming eval data directly follows train data
        model_new = model
        obj_cols = [f'{stock}_close' for stock in self.market_train.stock_names[1:]]

        # rolling window setup for out of sample forecasting
        num_obs = data.shape[0]
        model_new = model_new.append(data)
        pred_vec = model_new.predict(-num_obs)
        truth_vec = data[obj_cols].values
        pred_vec = pred_vec[obj_cols].values

        mse = self._calculate_mse(pred_vec, truth_vec)
        return mse, pred_vec, truth_vec

    def _process_data(self, market_test):

        data_train = self.market_train.data
        data_test = market_test.data
        data = np.concatenate((data_train, data_test), axis=1)

        #dates_train = self.market_train.date_list
        #dates_test = market_test.date_list
        #date_list = dates_train + dates_test

        self.id_sep = data_train.shape[1] - 1
        # id of start of testing data

        data = data[1:,:,[1,3]] # take only high and close
        data = np.transpose(data, [1,0,2]) # shape = [window_length, stocks, features]
        data = np.reshape(data, [data.shape[0], data.shape[1] * data.shape[2]])

        if self.preprocess == 'log_return':
            new = data[1:,:]
            old = data[:-1,:]
            data = np.log(new + EPS) - np.log(old + EPS)

        stock_names = self.market_train.stock_names[1:]
        features_used = ['high','close']
        all_features = [f'{stock}_{feature}' for stock in stock_names for feature in features_used]

        data = pd.DataFrame(data, columns=all_features)
        #data['dates'] = pd.to_datetime(date_list[1:])
        #data.set_index('dates', inplace=True)

        return data