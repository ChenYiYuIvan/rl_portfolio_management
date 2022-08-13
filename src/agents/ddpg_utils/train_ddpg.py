import argparse
import wandb
from src.environments.portfolio_end import PortfolioEnd
from src.agents.ddpg_agent import DDPGAgent


def main(params):

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=42, type=int, help='')
    parser.add_argument('--save_model_path', type=str, default=None, help='path to save model')
    parser.add_argument('--model_name', type=str, default=None, help='name of the model')

    parser.add_argument('--start_train', type=str, default="2013-03-22", help='starting date of training data')
    parser.add_argument('--end_train', type=str, default="2016-02-05", help='end date date of training data')
    parser.add_argument('--start_test', type=str, default="2016-03-22", help='starting date of testing data')
    parser.add_argument('--end_test', type=str, default="2018-02-07", help='end date of testing data')
    parser.add_argument('--window_length', default=30, type=int, help='window length')
    parser.add_argument('--stock_names', type=str, default=None, help='name of stocks in the market')
    parser.add_argument('--trading_cost', default=0.002, type=float, help='trading cost')
    parser.add_argument('--continuous', dest='continuous', default=False, action='store_true',
                        help='True to include continuous market assumption, False otherwise')
    parser.add_argument('--normalize', dest='normalize', default=False, action='store_true',
                        help='True to normalize data, False otherwise')

    parser.add_argument('--num_episodes', default=500, type=int, help='number of episodes to train for')
    parser.add_argument('--eval_steps', default=20, type=int, help='how many episodes for every evaluation step')
    parser.add_argument('--batch_size', default=64, type=int, help='minibatch size')
    parser.add_argument('--tau', default=0.001, type=float, help='moving average for target network')
    parser.add_argument('--gamma', default=0.99, type=float, help='')
    parser.add_argument('--buffer_size', default=100000, type=int, help='buffer size')
    parser.add_argument('--lr_actor', default=0.0001, type=float, help='actor learning rate')
    parser.add_argument('--lr_critic', default=0.001, type=float, help='critic learning rate')

    args = parser.parse_args(params)

    wandb.login()
    with wandb.init(project="thesis", entity="mldlproj1gr2", config=vars(args), mode="disabled") as run:
        config = wandb.config

        env_train = PortfolioEnd(config.start_train, config.end_train, config.window_length, config.stock_names, config.trading_cost, config.continuous, config.normalize)
        env_test = PortfolioEnd(config.start_test, config.end_test, config.window_length, config.stock_names, config.trading_cost, config.continuous, config.normalize)

        ddpg = DDPGAgent('ddpg', env_train, config)

        ddpg.train(run, env_test)
        ddpg.eval(env_test, render=True)


if __name__ == '__main__':
    params = [
        '--save_model_path', './checkpoints_ddpg',
        '--model_name', 'ddpg',
    ]
    main(params)