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

    seed = 42

    #env_config = read_yaml_config('default/env_small_train')
    env_config = read_yaml_config('experiments/env_train_1')
    env = Portfolio(env_config)

    agents_list = []

    #ddpg_config = read_yaml_config('experiments/ddpg_8')
    #ddpg = DDPGAgent('ddpg', env, seed, ddpg_config)
    #ddpg.load_models(0)
    ##ddpg.load_actor_from_path('./checkpoints_pretrained/msm_real_7_49/real_epoch99.pth')
    #agents_list.append(ddpg)

    #td3_config = read_yaml_config('experiments/td3_1')
    #td3 = TD3Agent('td3', env, seed, td3_config)
    #td3.load_models(35)
    #agents_list.append(td3)

    sac_config = read_yaml_config('experiments/sac_7')
    sac = SACAgent('sac', env, seed, sac_config)
    sac.load_models(20)
    agents_list.append(sac)

    crp = CRPAgent('crp', env, seed)
    agents_list.append(crp)

    #mpt = MPTAgent('mpt', env, seed, 'sharpe_ratio')
    #agents_list.append(mpt)

    #rng = RandomAgent('rng', env, seed)
    #agents_list.append(rng)

    evaluator = AgentsEvaluator(env, agents_list)
    evaluator.evaluate_all(num_cols=4, exploration=False, plot_stocks=True, plot_log=False)


if __name__ == '__main__':

    main()
