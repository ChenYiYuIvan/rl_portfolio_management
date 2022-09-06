import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from empyrical import simple_returns, sharpe_ratio, sortino_ratio, max_drawdown, value_at_risk, conditional_value_at_risk


EPS = 1e-6


def plot_stocks_info(env, num_cols=4, print_metrics=True, plot_log=True):
    # 

    stock_names = env.stock_names

    data = env.market.data[1:, :, 3]  # only closing prices and no cash data
    stock_metrics = []

    num_rows = int(np.ceil(data.shape[0] / num_cols))

    fig, axarr = plt.subplots(num_rows, num_cols, squeeze=False)
    for stock_id in range(data.shape[0]):
        row = int(stock_id / num_cols) # current row in figure
        col = stock_id % num_cols # current col in figure

        stock_values = data[stock_id,:]
        log_rets = np.log(stock_values[1:] + EPS) - np.log(stock_values[:-1] + EPS)
        if plot_log:
            values = log_rets
        else:
            values = stock_values / stock_values[0] # assuming initial value = 1

        df = [{'date': env.market.date_list[day_id], 'value': values[day_id]}
                for day_id in range(len(values))]
        name = stock_names[stock_id]

        df = pd.DataFrame(df)
        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
        df.set_index('date', inplace=True)

        df.plot(ax=axarr[row, col], title=name, rot=30, legend=False)

        # stock metrics
        stock_rets = simple_returns(stock_values)
        stock_metrics.append({
            'stock': name,
            'final_value': stock_values[-1] / stock_values[0], # assuming initial value = 1
            'mean_log_rets': np.mean(log_rets),
            'sharpe_ratio': sharpe_ratio(stock_rets, annualization=len(stock_rets)),
            'sortino_ratio': sortino_ratio(stock_rets, annualization=len(stock_rets)),
            'max_drawdown': max_drawdown(stock_rets),
            'var_95': value_at_risk(stock_rets),
            'cvar_95': conditional_value_at_risk(stock_rets),
        })

    if print_metrics:
        stock_metrics = pd.DataFrame(stock_metrics)
        stock_metrics.set_index('stock', inplace=True)
        print(stock_metrics)

    plt.show()


def prices_to_logreturns(prices):
    # shape: [num_stocks, window_length, price_features]

    new = prices[:, 1:, :]
    old = prices[:, :-1, :]

    log_rets = np.log(new + EPS) - np.log(old + EPS)

    return log_rets


def prices_to_simplereturns(prices):
    # shape: [num_stocks, window_length, price_features]
    # output shape: [num_stocks, window_length-1, price_features]

    new = prices[:, 1:, :]
    old = prices[:, :-1, :]

    simple_rets = np.divide(new, old) - 1

    return simple_rets


def remove_not_used(prices, cash=True, volume=True, open=True, high=False, low=False):
    # shape: [num_stocks, window_length, price_features]
    # True to remove
    # False to keep

    dims_to_keep = [*range(5)]
    if volume:
        dims_to_keep.remove(4)
    if open:
        dims_to_keep.remove(0)
    if high:
        dims_to_keep.remove(1)
    if low:
        dims_to_keep.remove(2)

    if cash:
        return prices[1:, :, dims_to_keep]
    else:
        return prices[:, :, dims_to_keep]


def rnn_transpose(prices):
    # use last, after all other transformations
    # used to get correct shape for lstm based networks

    # from [num_stocks, window_length, price_features]
    # to [window_length, num_stocks, price_features]

    prices = np.transpose(prices, (1, 0, 2))
    return prices


def cnn_transpose(prices):
    # use last, after all other transformations
    # used to get correct shape for cnn based networks

    # from [num_stocks, window_length, price_features]
    # to [price_features, num_stocks, window_length]

    prices = np.transpose(prices, (2, 0, 1))
    return prices


def cnn_rnn_transpose(prices):
    # use last, after all other transformations
    # used to get correct shape for cnn based networks

    # from [num_stocks, window_length, price_features]
    # to [price_features, window_length, num_stocks]

    prices = np.transpose(prices, (2, 1, 0))
    return prices
