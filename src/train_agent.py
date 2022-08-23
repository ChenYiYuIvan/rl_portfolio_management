import wandb
from src.utils.file_utils import read_yaml_config
from src.environments.portfolio_end import PortfolioEnd
from src.agents.ddpg_agent import DDPGAgent
from src.agents.sac_agent import SACAgent


def main():

    seed = 42

    env_config_train = read_yaml_config('env_default_train')
    env_config_test = read_yaml_config('env_default_test')

    ddpg_config = read_yaml_config('ddpg_default')
    sac_config = read_yaml_config('sac_default')

    #agent_config = ddpg_config
    agent_config = sac_config

    config = {'env_train':vars(env_config_train), 'env_test':vars(env_config_test), 'agent': vars(agent_config)}

    wandb.login()
    with wandb.init(project="thesis", entity="mldlproj1gr2", config=config, mode="online") as run:

        env_train = PortfolioEnd(env_config_train)
        env_test = PortfolioEnd(env_config_test)

        #agent = DDPGAgent('ddpg', env_train, seed, ddpg_config)
        agent = SACAgent('sac', env_train, seed, sac_config)

        agent.train(run, env_test)
        agent.eval(env_test, render=True)


if __name__ == '__main__':

    main()