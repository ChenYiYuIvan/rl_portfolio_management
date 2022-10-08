import torch
import wandb

from src.forecaster.nn_forecaster import NNForecaster
from src.utils.file_utils import read_yaml_config
from src.agents.sac_agent import SACAgent
from src.environments.portfolio import Portfolio


if __name__ == '__main__':

    config = {
        'seed': 42,
        'env_train': 'experiments/env_train_1',
        'env_test': 'experiments/env_test_1',
        'agent': 'experiments/sac_11',
        'model': 'transformer_shared', # transformed / transformed_shared
        'batch_size': 64,
        'num_epochs': 10000,
        'learning_rate': 1e-4,
        'weight_decay': 1e-2,
        'eval_steps': 10,
        'num_price_features': 4,
        'window_length': 49,
        'num_stocks': 1,
        'd_model': 64,
        'num_heads': 8,
        'num_layers': 3,
        'save_model_path': './checkpoints_forecaster/trans_shared_log_return',
        'model_name': 'trans_forecaster',
        'checkpoint_ep': 0,
    }

    wandb.login()
    with wandb.init(project="forecasting", entity="mldlproj1gr2", config=config, mode="disabled") as run:
        config = wandb.config

        seed = config.seed

        env_config_train = read_yaml_config(config.env_train)
        env_train = Portfolio(env_config_train)

        env_config_test = read_yaml_config(config.env_test)
        env_test = Portfolio(env_config_test)

        agent_config = read_yaml_config(config.agent)
        agent = SACAgent('sac', env_train, seed, agent_config)

        loss = torch.nn.MSELoss()

        forecaster = NNForecaster('transformer', agent, env_train.market)

        forecaster.train(env_test.market, config, run)

        forecaster.load_model(f'{config.save_model_path}/ep9_{config.model_name}.pth')
        forecaster.forecast_all(env_train.market)