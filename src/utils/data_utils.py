import csv
import os
import numpy as np

# to load all the data into a script, simply import this file

date_list = []
with open("data/AAPL_data.csv", 'r') as f:
    data = csv.reader(f)
    header = next(data)
    for row in data:
        date_list.append(row[0])


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
            row[1:5] = [float(num) for num in row[1:5]] # open, high, low, close
            row[5] = int(row[5]) # volume
            stock_data.append(row[1:6])

    hist_data.append(stock_data)

hist_data = np.array(hist_data)
