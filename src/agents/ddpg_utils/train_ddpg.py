import wandb
from src.utils.file_utils import read_config
from src.environments.portfolio_end import PortfolioEnd
from src.agents.ddpg_agent import DDPGAgent


def main():

    args = read_config('ddpg_default')

    wandb.login()
    with wandb.init(project="thesis", entity="mldlproj1gr2", config=vars(args), mode="disabled") as run:
        config = wandb.config

        env_train = PortfolioEnd(config.start_train, config.end_train, config.window_length, config.stock_names, config.trading_cost, config.continuous, config.normalize)
        env_test = PortfolioEnd(config.start_test, config.end_test, config.window_length, config.stock_names, config.trading_cost, config.continuous, config.normalize)

        ddpg = DDPGAgent('ddpg', env_train, config)

        ddpg.train(run, env_test)
        ddpg.eval(env_test, render=True)


if __name__ == '__main__':

    main()