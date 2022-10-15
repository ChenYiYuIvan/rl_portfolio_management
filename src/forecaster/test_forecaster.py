from src.forecaster.nn_forecaster import NNForecaster
from src.utils.file_utils import read_yaml_config
from src.agents.sac_agent import SACAgent
from src.environments.portfolio import Portfolio
from src.utils.file_utils import Dict2Class


def main():

    config = {
        'seed': 42,
        'env': 'experiments/env_train_1',
        'agent': 'experiments/sac_11',
        'model': 'transformer_shared', # transformed / transformed_shared
        'save_model_path': './checkpoints_forecaster/trans_shared_log_return',
        'model_name': 'trans_forecaster',
        'episode': 9,
    }
    
    config = Dict2Class(config)

    seed = 42

    env_config = read_yaml_config(config.env)
    env = Portfolio(env_config)

    agent_config = read_yaml_config(config.agent)
    agent = SACAgent('sac', env, seed, agent_config)

    forecaster = NNForecaster('transformer', agent, env.market)

    forecaster.load_model(f'{config.save_model_path}/ep{config.episode}_{config.model_name}.pth')
    mse = forecaster.forecast_all(env.market, render=False)
    print(mse)


if __name__ == '__main__':
    main()