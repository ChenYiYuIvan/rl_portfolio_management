import torch
import wandb

from src.forecaster.nn_forecaster import NNForecaster
from src.forecaster.varma_forecaster import VARMAForecaster
from src.utils.file_utils import read_yaml_config
from src.agents.sac_agent import SACAgent
from src.environments.portfolio import Portfolio


def main(method):

    # environment to test on
    env_num = 1

    if method == 'varma':
        
        env_config_train = read_yaml_config(f'experiments/env_train_{env_num}')
        # to make comparison fair, shift start date of train data ahead
        # by length of rolling window used for ML method
        env_train = Portfolio(env_config_train)
        env_config_train.start_date = env_train.market.date_list[env_train.market.window_length-1]
        env_train = Portfolio(env_config_train) # update portfolio
        market_train = env_train.market

        env_config_test = read_yaml_config(f'experiments/env_test_{env_num}')
        env_test = Portfolio(env_config_test)
        market_test = env_test.market

        model = VARMAForecaster('varma', 'log_return', market_train, market_test)
        model.train(f'./checkpoints_forecaster/varma_log_return_env{env_num}')

    elif method == 'sac':

        config = {
            'seed': 42,
            'env_train': f'experiments/env_train_{env_num}',
            'env_test': f'experiments/env_test_{env_num}',
            'agent': 'experiments/sac_11',
            'model': 'transformer_shared', # transformed / transformed_shared
            'batch_size': 64,
            'num_epochs': 10000,
            'learning_rate': 1e-4,
            'weight_decay': 1e-2,
            'eval_steps': 10,
            'save_model_path': f'./checkpoints_forecaster/trans_shared_log_return_env{env_num}',
            'model_name': 'trans_forecaster',
            'checkpoint_ep': 0,
        }

        wandb.login()
        with wandb.init(project="forecasting", entity="mldlproj1gr2", config=config, mode="online") as run:
            config = wandb.config

            seed = config.seed

            env_config_train = read_yaml_config(config.env_train)
            env_train = Portfolio(env_config_train)

            env_config_test = read_yaml_config(config.env_test)
            # assuming test directly follows train -> have to modify test start date so that
            # the first forecast is first day of the year using obs from prev year
            env_config_test.start_date = env_train.market.date_list[-env_config_test.window_length]
            env_test = Portfolio(env_config_test)

            agent_config = read_yaml_config(config.agent)
            agent = SACAgent('sac', env_train, seed, agent_config)

            forecaster = NNForecaster('transformer', agent, env_train.market)

            forecaster.train(env_test.market, config, run)

            #forecaster.load_model(f'{config.save_model_path}/ep9_{config.model_name}.pth')
            forecaster.forecast_all(env_train.market)


if __name__ == '__main__':

    main('varma')
    #main('sac')