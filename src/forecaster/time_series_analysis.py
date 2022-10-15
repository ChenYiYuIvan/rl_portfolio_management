from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, grangercausalitytests, kpss
from pmdarima.arima import auto_arima
from src.environments.market import Market
from src.environments.portfolio import Portfolio
from src.utils.data_utils import prices_to_logreturns
from src.utils.file_utils import read_yaml_config
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def analyze_time_series(data, threshold=0.05, maxlag=5, plot=False):
    # input shape: [time, features]

    features = ['open', 'high', 'low', 'close', 'volume']
    # target feature: close
    
    if plot:
        fig, ax = plt.subplots(2)

        ax[0].plot(data['close'])
        ax[0].grid()

        # pacf - close returns
        plot_pacf(data['close'], ax=ax[1], lags=maxlag, method='ywm')

        plt.show()

    print('Augumented Dickey Fuller test for unit roots')
    for feature in features:
        print(f'{feature}:')
        print(adfuller(data[feature]))
        print('------')
    print('------------------------------------------')

    print('KPSS test for stationarity')
    for feature in features:
        print(f'{feature}:')
        print(kpss(data[feature]))
        print('------')
    print('------------------------------------------')

    print('Granger causality test')
    for feature in features:
        if feature != 'close':
            print(f'{feature}:')
            granger_results = grangercausalitytests(data[['close',feature]], maxlag, verbose=False)
            for lag, results in granger_results.items():
                (results, _) = results
                passed = False
                for test, test_results in results.items():
                    if test_results[1] < threshold:
                        passed = True
                if passed:
                    print(f'lag {lag} - {feature} returns granger-causes close returns')
            print('------')
    print('------------------------------------------')


def main():

    preprocess = 'log_return'

    env_config_train = read_yaml_config('default/env_small_train')
    env_train = Portfolio(env_config_train)
    market_train = env_train.market
    data_train_o = market_train.data
    dates_train = market_train.date_list

    env_config_test = read_yaml_config('default/env_small_test')
    env_test = Portfolio(env_config_test)
    market_test = env_test.market
    data_test_o = market_test.data
    dates_test = market_test.date_list

    if preprocess == 'log_return':
        # shape = [stock, window length, price features]
        data_train = prices_to_logreturns(data_train_o)[1:] # to remove cash data
        data_test = prices_to_logreturns(data_test_o)[1:]
        
        dates_train = dates_train[1:]
        dates_test = dates_test[1:]
    

    for (stock_data_train, stock_data_test) in zip(data_train[:], data_test[:]):
        data_train = pd.DataFrame(data=stock_data_train, columns=['open','high','low','close','volume'])
        data_train['dates'] = pd.to_datetime(dates_train)
        data_train.set_index('dates', inplace=True)

        analyze_time_series(data_train, plot=True)


if __name__ == '__main__':
    main()
