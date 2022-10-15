from src.forecaster.nn_forecaster import NNForecaster
from src.forecaster.varma_forecaster import VARMAForecaster
from src.utils.file_utils import read_yaml_config
from src.agents.sac_agent import SACAgent
from src.environments.portfolio import Portfolio
from src.utils.file_utils import Dict2Class


def main(method):

    # env to test on
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
        model.fit_test_model_given_par(p=0, q=2, render=True)

    elif method == 'sac':

        config = {
            'seed': 42,
            'env_train': f'experiments/env_train_{env_num}',
            'env_test': f'experiments/env_test_{env_num}',
            'agent': 'experiments/sac_11',
            'model': 'transformer_shared', # transformed / transformed_shared
            'save_model_path': f'./checkpoints_forecaster/trans_shared_log_return_env{env_num}',
            'model_name': 'trans_forecaster',
            'episode': 9,
        }
        
        config = Dict2Class(config)

        seed = 42

        env_config_train = read_yaml_config(f'experiments/env_train_{env_num}')
        env_train = Portfolio(env_config_train)

        env_config_test = read_yaml_config(f'experiments/env_test_{env_num}')
        # assuming test directly follows train -> have to modify test start date so that
        # the first forecast is first day of the year using obs from prev year
        env_config_test.start_date = env_train.market.date_list[-env_config_test.window_length]
        env_test = Portfolio(env_config_test)

        agent_config = read_yaml_config(config.agent)
        agent = SACAgent('sac', env_train, seed, agent_config)

        forecaster = NNForecaster('transformer', agent, env_train.market)

        forecaster.load_model(f'{config.save_model_path}/ep{config.episode}_{config.model_name}.pth')
        forecaster.forecast_all(env_test.market, render=True)


if __name__ == '__main__':

    #main('varma')
    main('sac')