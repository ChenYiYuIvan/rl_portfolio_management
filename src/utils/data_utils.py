import csv
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import datetime as dt


# to load all the data into a script, simply import this file
start_date = '2013-02-08'
end_date = '2018-02-07'

date_fmt = '%Y-%m-%d'
snp = yf.download('SPY',
                  start=(dt.datetime.strptime(start_date, date_fmt) +
                         dt.timedelta(1)).strftime(date_fmt),
                  end=(dt.datetime.strptime(end_date, date_fmt) + dt.timedelta(1)).strftime(date_fmt))

date_list = snp.index.astype(str).tolist()

snp = snp.drop(columns=['Adj Close']).to_numpy()


# in caso ci siano problemi con yfinance
if snp.shape[0] == 0:
    date_list = []
    with open("data/AAPL_data.csv", 'r') as f:
        data = csv.reader(f)
        header = next(data)
        for row in data:
            date_list.append(row[0])

    snp = np.ones((1000, 5))


stock_names = os.listdir('data')
stock_names = [name.split('_')[0] for name in stock_names]


hist_data = []
for stock in stock_names:
    file_path = f'data/{stock}_data.csv'

    stock_data = []
    with open(file_path, 'r') as f:
        data = csv.reader(f)
        header = next(data)
        for row in data:
            # memorize only necessary observations
            date_idx = date_list.index(row[0])
            # open, high, low, close, volume
            row[1:6] = [float(num) for num in row[1:6]]
            stock_data.append(row[1:6])

    hist_data.append(stock_data)

hist_data = np.array(hist_data)

# add cash info
cash_data = np.ones((1, hist_data.shape[1], hist_data.shape[2]))
hist_data = np.concatenate((cash_data, hist_data), axis=0)
stock_names = ['CASH', *stock_names]


def plot_stock_values(env, num_cols = 4):

    start_date = env.start_date
    end_date = env.end_date
    start_id = date_list.index(start_date)
    end_id = date_list.index(end_date)

    stocks = env.stock_names
    stocks_id = [stock_names.index(stock) for stock in stocks]

    data = hist_data[stocks_id, start_id:end_id+1, 3] # only closing prices
    
    num_rows = int(np.ceil(data.shape[0] / num_cols))

    fig, axarr = plt.subplots(num_rows, num_cols, squeeze=False)
    for row in range(num_rows):
        for col in range(num_cols):
            stock_id = col + row*num_cols
            df = [{'date': date_list[start_id+day_id], 'value': data[stock_id, day_id]} for day_id in range(data.shape[1])]
            name = stock_names[stock_id + 1]

            df = pd.DataFrame(df)
            df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
            df.set_index('date', inplace=True)

            df.plot(ax=axarr[row,col], title=name, rot=30, legend=False)
    plt.show()

