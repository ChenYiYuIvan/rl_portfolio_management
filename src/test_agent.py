from src.agents.agents_evaluator import AgentsEvaluator
from src.environments.portfolio import Portfolio
from src.agents.ddpg_agent import DDPGAgent
from src.agents.td3_agent import TD3Agent
from src.agents.sac_agent import SACAgent
from src.agents.crp_agent import CRPAgent
from src.agents.mpt_agent import MPTAgent
from src.agents.random_agent import RandomAgent
from src.utils.file_utils import read_yaml_config

import pandas as pd
pd.options.display.float_format = '{:,.6f}'.format


def main():

    seed = 0
    env_num = 2

    env_config_train = read_yaml_config(f'experiments/env_train_{env_num}')
    env_train = Portfolio(env_config_train)
    
    env_config_test = read_yaml_config(f'experiments/env_test_{env_num}')
    env_test = Portfolio(env_config_test)

    agents_list = []

    #ddpg_config = read_yaml_config('experiments/ddpg_11')
    #ddpg = DDPGAgent('ddpg', env_train, seed, ddpg_config)
    #ddpg.load_models(68)
    #agents_list.append(ddpg)

    #td3_config = read_yaml_config('experiments/td3_1')
    #td3 = TD3Agent('td3', env_train, seed, td3_config)
    #td3.load_models(35)
    #agents_list.append(td3)

    sac_config = read_yaml_config('experiments/sac_17')
    sac = SACAgent('sac', env_train, seed, sac_config)
    sac.load_models(2)
    agents_list.append(sac)
    
    sac_config = read_yaml_config('experiments/sac_16')
    sac = SACAgent('sac_pretraining', env_train, seed, sac_config)
    sac.load_models(21)
    agents_list.append(sac)

    #crp = CRPAgent('crp', env_train, seed)
    #agents_list.append(crp)

    mpt = MPTAgent('mpt', env_train, seed, 'sharpe_ratio')
    agents_list.append(mpt)

    #rng = RandomAgent('rng', env_train, seed)
    #agents_list.append(rng)

    evaluator = AgentsEvaluator(env_test, agents_list)
    evaluator.evaluate_all(num_cols=4, exploration=False, plot_stocks=True, plot_log=False)


if __name__ == '__main__':

    main()
