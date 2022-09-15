from src.agents.agents_evaluator import AgentsEvaluator
from src.environments.portfolio import Portfolio
from src.agents.ddpg_agent import DDPGAgent
from src.agents.sac_agent import SACAgent
from src.agents.crp_agent import CRPAgent
from src.agents.mpt_agent import MPTAgent
from src.agents.random_agent import RandomAgent
from src.utils.file_utils import read_yaml_config, get_checkpoint_folder

import pandas as pd
pd.options.display.float_format = '{:,.6f}'.format


def main():

    seed = 42

    env_config = read_yaml_config('env_small_test')
    #env_config = read_yaml_config('env_default_test')
    env = Portfolio(env_config)

    agents_list = []

    ddpg_config = read_yaml_config('ddpg_1')
    ddpg = DDPGAgent('ddpg', env, seed, ddpg_config)
    #ddpg.load_actor_model('./checkpoints_pretrained/cnn_real_7_49/real_epoch89.pth')
    ddpg.load_actor_model(get_checkpoint_folder(ddpg, env, False) + '/ddpg_ep50.pth')
    agents_list.append(ddpg)

    ddpg_config2 = read_yaml_config('ddpg_2')
    ddpg2 = DDPGAgent('ddpg', env, seed, ddpg_config2)
    #ddpg.load_actor_model('./checkpoints_pretrained/cnn_real_7_49/real_epoch89.pth')
    ddpg2.load_actor_model(get_checkpoint_folder(ddpg2, env, False) + '/ddpg_ep11.pth')
    agents_list.append(ddpg2)

    #sac_config = read_yaml_config('sac_default')
    #sac = SACAgent('SAC', env, seed, sac_config)
    #sac.load_actor_model(get_checkpoint_folder(sac, env) + '/sac_ep2.pth')
    #agents_list.append(sac)

    #crp = CRPAgent('crp', env, seed, 'diff_sharpe_ratio')
    #agents_list.append(crp)

    #mpt = MPTAgent('mpt', env, seed, 'diff_sharpe_ratio', 'sharpe_ratio')
    #agents_list.append(mpt)

    #rng = RandomAgent('rng', env, seed, 'diff_sharpe_ratio')
    #agents_list.append(rng)

    evaluator = AgentsEvaluator(env, agents_list)
    evaluator.evaluate_all(num_cols=4, exploration=False, plot_stocks=True, plot_log=False)


if __name__ == '__main__':

    main()
