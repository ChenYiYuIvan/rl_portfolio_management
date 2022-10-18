from src.environments.portfolio import Portfolio
from src.utils.file_utils import read_yaml_config
from src.utils.data_utils import prices_to_logreturns
import numpy as np


env_config = read_yaml_config('experiments/env_train_1')
env = Portfolio(env_config)
data = env.market.data
logdata = prices_to_logreturns(data)

# mean and std of stocks (no cash)
mean = logdata[1:].mean(axis=1)[:,None,:]
std = logdata[1:].std(axis=1)[:,None,:]

print(f'mean = {mean}')
print(f'std = {std}')

# to avoid normalizing cash data
logdata_norm = logdata
logdata_norm[1:] = (logdata[1:] - mean) / std

# check correctness
mean_new = logdata_norm[1:].mean(axis=1)
std_new = logdata_norm[1:].std(axis=1)

print(f'mean_new = {mean_new}')
print(f'std_new = {std_new}')
