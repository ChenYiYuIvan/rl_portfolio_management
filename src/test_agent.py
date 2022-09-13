from src.agents.agents_evaluator import AgentsEvaluator
from src.environments.portfolio_end import PortfolioEnd
from src.agents.ddpg_agent import DDPGAgent
from src.agents.sac_agent import SACAgent
from src.agents.crp_agent import CRPAgent
from src.agents.mpt_agent import MPTAgent
from src.agents.random_agent import RandomAgent
from src.utils.file_utils import read_yaml_config, get_checkpoint_folder


def main():

    seed = 42

    env_config = read_yaml_config('env_default_train')
    #env_config = read_yaml_config('env_default_test')
    env = PortfolioEnd(env_config)

    agents_list = []

    ddpg_config = read_yaml_config('ddpg_default')
    ddpg = DDPGAgent('exact', env, seed, ddpg_config)
    #ddpg.load_actor_model('./checkpoints_pretrained/cnn_real_7_49_noise/real_epoch59.pth')
    ddpg.load_actor_model(get_checkpoint_folder(ddpg, env, False) + '/ddpg_ep45.pth')
    agents_list.append(ddpg)

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
