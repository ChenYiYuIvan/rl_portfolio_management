import csv
import os
import numpy as np
import yfinance as yf
import datetime as dt

# to load all the data into a script, simply import this file
start_date = '2013-02-08'
end_date = '2018-02-07'

date_fmt = '%Y-%m-%d'
snp = yf.download('SPY',
    start=(dt.datetime.strptime(start_date, date_fmt) + dt.timedelta(1)).strftime(date_fmt),
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

    snp = np.ones((1000,5))


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
            row[1:6] = [float(num) for num in row[1:6]] # open, high, low, close, volume
            stock_data.append(row[1:6])

    hist_data.append(stock_data)

hist_data = np.array(hist_data)

# add cash info
cash_data = np.ones((1, hist_data.shape[1], hist_data.shape[2]))
hist_data = np.concatenate((cash_data, hist_data), axis=0)
