from src.agents.agents_evaluator import AgentsEvaluator
from src.environments.portfolio_end import PortfolioEnd
from src.agents.ddpg_agent import DDPGAgent
from src.agents.crp_agent import CRPAgent
from src.agents.random_agent import RandomAgent
from src.utils.file_utils import read_yaml_config


def main():

    seed = 42

    env_config = read_yaml_config('env_default_test')
    ddpg_config = read_yaml_config('ddpg_default')

    env = PortfolioEnd(env_config)

    ddpg = DDPGAgent('ddpg', env, seed, ddpg_config)
    ddpg.load_actor_model('./checkpoints/checkpoints_ddpg/ddpg_ep499.pth')

    crp = CRPAgent('crp', env, seed)

    #rng = RandomAgent('rng', env, seed)

    evaluator = AgentsEvaluator(env, [ddpg, crp])
    evaluator.evaluate_all()


if __name__ == '__main__':

    main()