import argparse
import wandb
from src.environments.portfolio_end import PortfolioEnd
from src.ddpg.ddpg import DDPG


def main(params):

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=-1, type=int, help='')
    parser.add_argument('--save_model_path', type=str, default=None, help='path to save model')
    parser.add_argument('--model_name', type=str, default=None, help='name of the model')

    parser.add_argument('--window_length', default=30, type=int, help='window length')
    parser.add_argument('--num_episodes', default=50, type=int, help='number of episodes to train for')
    parser.add_argument('--eval_steps', default=64, type=int, help='how many episodes for every evaluation step')
    parser.add_argument('--batch_size', default=64, type=int, help='minibatch size')
    parser.add_argument('--tau', default=0.001, type=float, help='moving average for target network')
    parser.add_argument('--gamma', default=0.99, type=float, help='')
    parser.add_argument('--buffer_size', default=6000000, type=int, help='buffer size')
    parser.add_argument('--lr_actor', default=0.001, type=float, help='actor learning rate')
    parser.add_argument('--lr_critic', default=0.001, type=float, help='critic learning rate')

    args = parser.parse_args(params)

    wandb.login()
    with wandb.init(project="thesis", entity="mldlproj1gr2", config=vars(args), mode="online") as run:
        config = wandb.config

        start_date = "2013-03-22"
        end_date = "2016-02-05"

        env = PortfolioEnd(start_date, end_date, window_length=args.window_length)
        ddpg = DDPG(env, config)

        ddpg.train(run)


if __name__ == '__main__':
    params = [
        '--save_model_path', './checkpoints_ddpg',
        '--model_name', 'ddpg',
    ]
    main(params)