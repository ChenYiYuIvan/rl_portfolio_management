from src.agents.agents_evaluator import AgentsEvaluator
from src.environments.portfolio_end import PortfolioEnd
from src.agents.ddpg_agent import DDPGAgent
from src.agents.sac_agent import SACAgent
from src.agents.crp_agent import CRPAgent
from src.agents.mpt_agent import MPTAgent
from src.agents.random_agent import RandomAgent
from src.utils.file_utils import read_yaml_config


def main():

    seed = 42

    env_config = read_yaml_config('env_default_test')
    ddpg_config = read_yaml_config('ddpg_default')
    sac_config = read_yaml_config('sac_default')

    env = PortfolioEnd(env_config)

    ddpg = DDPGAgent('ddpg', env, seed, ddpg_config)
    ddpg.load_actor_model('./checkpoints/DDPG_gru_stocks16_batch256/ddpg_ep1.pth')

    sac = SACAgent('sac', env, seed, sac_config)
    sac.load_actor_model('./checkpoints/SAC_gru_stocks16_batch64_scale100/sac_ep1.pth')

    crp = CRPAgent('crp', env, seed)

    mpt = MPTAgent('mpt', env, seed, 'sharpe_ratio')

    #rng = RandomAgent('rng', env, seed)

    evaluator = AgentsEvaluator(env, [ddpg, sac, crp, mpt])
    evaluator.evaluate_all(num_cols=4, exploration=False, plot_stocks=True)


if __name__ == '__main__':

    main()
