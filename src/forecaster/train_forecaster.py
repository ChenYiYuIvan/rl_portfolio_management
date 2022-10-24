import torch
import wandb

from src.forecaster.nn_forecaster import NNForecaster
from src.forecaster.varma_forecaster import VARMAForecaster
from src.utils.file_utils import read_yaml_config
from src.agents.sac_agent import SACAgent
from src.environments.portfolio import Portfolio


def main(method):

    # environment to train on
    env_num = 2

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
        model.train(f'./checkpoints_forecaster/varma_log_return_env{env_num}', maxiter=1000)

    elif method == 'sac':

        model = 'lstm' # trans / lstm

        config = {
            'seed': 0,
            'env_train': f'experiments/env_train_{env_num}',
            'env_test': f'experiments/env_test_{env_num}',
            #'env_train': f'default/env_small_train',
            #'env_test': f'default/env_small_test',
            'agent': 'experiments/sac_12',
            'model': f'{model}_shared',
            'batch_size': 256,
            'num_epochs': 2000,
            'learning_rate': 1e-4,
            'weight_decay': 0,
            'eval_steps': 10,
            'save_model_path': f'./checkpoints_forecaster/{model}_shared_log_return_env{env_num}pre',
            'model_name': f'{model}_forecaster',
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

            forecaster = NNForecaster(config.model, agent)
            #forecaster.model.init_weights(nonlinearity='leaky_relu', a=0.1)

            forecaster.train(env_test.market, config, run)

            #forecaster.load_model(f'{config.save_model_path}/ep9_{config.model_name}.pth')
            forecaster.load_pretrained(f'./checkpoints_forecaster/{model}_shared_log_return_envsmall/ep999_{config.model_name}.pth')
            forecaster.forecast_all(env_train.market)


if __name__ == '__main__':

    #main('varma')
    main('sac')