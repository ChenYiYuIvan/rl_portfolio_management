import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.cuda import amp
from tqdm import tqdm
import wandb

from src.forecaster.data_loader import StockDataset
from src.models.transformer_model import TransformerForecaster
from src.models.transformer_shared_model import TransformerSharedForecaster
from src.utils.file_utils import read_yaml_config
from src.agents.sac_agent import SACAgent
from src.environments.portfolio import Portfolio
from src.utils.torch_utils import FLOAT, USE_CUDA
from src.utils.data_utils import EPS
from src.forecaster.train_forecaster import plot_result


# Turns a dictionary into a class
class Dict2Class(object):
      
    def __init__(self, my_dict):
          
        for key in my_dict:
            setattr(self, key, my_dict[key])


def main():

    config = {
        'seed': 42,
        'env_train': 'experiments/env_train_1',
        'env_test': 'experiments/env_test_1',
        'agent': 'experiments/sac_11',
        'model': 'transformer_shared', # transformed / transformed_shared
        'num_price_features': 4,
        'window_length': 49,
        'num_stocks': 1,
        'd_model': 64,
        'num_heads': 8,
        'num_layers': 3,
        'save_model_path': './checkpoints_forecaster/trans_shared_log_return',
        'model_name': 'trans_forecaster',
        'episode': 339,
    }
    
    config = Dict2Class(config)

    seed = 42

    env_config_train = read_yaml_config(config.env_train)
    env_train = Portfolio(env_config_train)

    env_config_test = read_yaml_config(config.env_test)
    env_test = Portfolio(env_config_test)

    agent_config = read_yaml_config(config.agent)
    agent = SACAgent('sac', env_train, seed, agent_config)

    if config.model == 'transformer':
        model = TransformerForecaster(config.num_price_features, config.num_stocks, config.window_length, config.d_model, config.num_heads, config.num_layers)
    elif config.model == 'transformer_shared':
        model = TransformerSharedForecaster(config.num_price_features, config.num_stocks, config.window_length, config.d_model, config.num_heads, config.num_layers)
    model = model.cuda()
    model.eval()

    model.load_state_dict(torch.load(f'{config.save_model_path}/ep{config.episode}_{config.model_name}.pth'))
    plot_result(model, env_train.market, agent, value=False)


if __name__ == '__main__':
    main()