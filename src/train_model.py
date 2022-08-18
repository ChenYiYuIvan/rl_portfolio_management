import wandb
from src.utils.file_utils import read_yaml_config
from src.environments.portfolio_end import PortfolioEnd
from src.agents.ddpg_agent import DDPGAgent


def main():

    seed = 42

    env_config_train = read_yaml_config('env_default_train')
    env_config_test = read_yaml_config('env_default_test')
    ddpg_config = read_yaml_config('ddpg_default')

    config = {'env_train':vars(env_config_train), 'env_test':vars(env_config_test), 'ddpg': vars(ddpg_config)}

    wandb.login()
    with wandb.init(project="thesis", entity="mldlproj1gr2", config=config, mode="disabled") as run:

        env_train = PortfolioEnd(env_config_train)
        env_test = PortfolioEnd(env_config_test)

        ddpg = DDPGAgent('ddpg', env_train, seed, ddpg_config)

        ddpg.train(run, env_test)
        ddpg.eval(env_test, render=True)


if __name__ == '__main__':

    main()