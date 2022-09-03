import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


EPS = 1e-6


def plot_stock_values(env, num_cols=4):

    stock_names = env.stock_names

    data = env.market.data[1:, :, 3]  # only closing prices and no cash data

    num_rows = int(np.ceil(data.shape[0] / num_cols))

    fig, axarr = plt.subplots(num_rows, num_cols, squeeze=False)
    for row in range(num_rows):
        for col in range(num_cols):
            stock_id = col + row*num_cols
            df = [{'date': env.market.date_list[day_id], 'value': data[stock_id, day_id]}
                  for day_id in range(data.shape[1])]
            name = stock_names[stock_id]

            df = pd.DataFrame(df)
            df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
            df.set_index('date', inplace=True)

            df.plot(ax=axarr[row, col], title=name, rot=30, legend=False)
    plt.show()


def prices_to_logreturns(prices):
    # shape: [num_stocks, window_length, price_features]

    new = prices[:, 1:, :]
    old = prices[:, :-1, :]

    log_rets = np.log(new + EPS) - np.log(old + EPS)

    return log_rets


def prices_to_simplereturns(prices):
    # shape: [num_stocks, window_length, price_features]

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
