from statsmodels.tsa.statespace.varmax import VARMAX
from src.environments.market import Market
from src.environments.portfolio import Portfolio
from src.agents.sac_agent import SACAgent
from src.utils.file_utils import read_yaml_config


def main():

    env_config_train = read_yaml_config('experiments/env_train_1')
    env_train = Portfolio(env_config_train)
    market_train = env_train.market
    data_train_o = market_train.data

    env_config_test = read_yaml_config('experiments/env_test_1')
    env_test = Portfolio(env_config_test)
    market_test = env_test.market
    data_test_o = market_test.data

    agent_config = read_yaml_config('experiments/sac_8')
    agent = SACAgent('sac', env_train, 42, agent_config)

    (data_train, _) = agent.preprocess_data((data_train_o, None))

    print(data_train.squeeze())


if __name__ == '__main__':
    main()

