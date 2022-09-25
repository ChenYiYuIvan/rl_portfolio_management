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

    #env_config = read_yaml_config('default/env_small_train')
    env_config = read_yaml_config('experiments/env_test_0')
    env = Portfolio(env_config)

    agents_list = []

    ddpg_config = read_yaml_config('experiments/ddpg_7')
    ddpg = DDPGAgent('ddpg', env, seed, ddpg_config)
    model_folder = get_checkpoint_folder(ddpg, env, ddpg.imitation_learning == 'passive')
    print(model_folder)
    ddpg.load_actor_model(model_folder + '/ddpg_ep1.pth')
    #ddpg.load_actor_model('./checkpoints_pretrained/msm_real_7_49/real_epoch99.pth')
    agents_list.append(ddpg)

    #sac_config = read_yaml_config('experiments/sac_2')
    #sac = SACAgent('SAC', env, seed, sac_config)
    #model_folder = get_checkpoint_folder(sac, env, sac.imitation_learning == 'passive')
    #sac.load_actor_model(model_folder + '/sac_ep99.pth')
    #agents_list.append(sac)

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
