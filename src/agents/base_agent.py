import numpy as np
import torch

from src.utils.torch_utils import USE_CUDA

# TODO: implement uniform buy and hold, best stock, best constant rebalanced portfolio

class BaseAgent:

    def __init__(self, name, env, seed, reward_type=None):

        if seed > 0:
            self.seed(seed)

        self.name = name
        self.env = env

        # reward signal
        if reward_type is None:
            self.reward_type = 'log_return'
        else:
            assert reward_type in ('log_return', 'simple_return', 'diff_sharpe_ratio', 'diff_sortino_ratio')
            self.reward_type = reward_type

        self.state_dim = self.env.observation_space.shape # [price features, stocks, time window]
        self.action_dim = self.env.action_space.shape[0] # cash + num stocks

        # initialization of parameters for differential sharpe ratio and differential downside deviation ratio
        self.reset()


    def eval(self, env, exploration=False, render=False):
        # mode: exploration vs no exploration
        # render: plot
        # episode: current training episode (if eval is called during training)ù
        # table: to store eval results on wandb

        # initial state of environment
        self.reset()
        curr_obs = env.reset(test_mode=True)
        curr_obs = self.preprocess_data(curr_obs)

        # initialize values
        infos = []
        ep_reward = 0

        # keep sampling until done
        done = False
        while not done:
            # select action
            action = self.predict_action(curr_obs, exploration)

            # step forward environment
            next_obs, done, info = env.step(action)
            reward = self.get_reward(info)
            next_obs = self.preprocess_data(next_obs)

            infos.append(info)

            ep_reward += reward
            curr_obs = next_obs

        end_port_value = info['port_value_new']

        if render:
            env.render()

        return ep_reward, infos, end_port_value


    def random_action(self):
        action = self.env.action_space.sample()
        action /= action.sum()

        return action


    def predict_action(self, obs, exploration=False):
        raise NotImplementedError


    def preprocess_data(self, obs):
        return obs


    def get_reward(self, info):
        simple_ret = info['simple_return']
        if self.reward_type in ('log_return, simple_return'):
            reward = info[self.reward_type]

        elif self.reward_type == 'diff_sharpe_ratio':
            deltaA = simple_ret - self.A
            deltaB = simple_ret**2 - self.B

            num = self.B * deltaA - self.A * deltaB / 2
            denom = (self.B - self.A**2)**(3/2)
            reward = num / (denom + 1e-12)

            self.A += self.eta * deltaA
            self.B += self.eta * deltaB

        elif self.reward_type == 'diff_sortino_ratio':
            # also called differential downside deviation ratio
            if simple_ret > 0:
                num = simple_ret - self.A / 2
                denom = self.DD
            else:
                num = self.DD**2 * (simple_ret - self.A / 2) - self.A * simple_ret**2 / 2
                denom = self.DD**3
            reward = num / (denom + 1e-12)

            self.A = self.A + self.eta * (simple_ret - self.A)
            self.DD = np.sqrt(self.DD**2 + self.eta * (np.minimum(simple_ret, 0)**2 - self.DD**2))

        if hasattr(self, 'reward_scale'):
            reward *= self.reward_scale

        return reward


    def reset(self):
        self.eta = 1e-5
        self.A = 0
        self.B = 0
        self.DD = 0


    def seed(self,s):
        np.random.seed(s)
        torch.manual_seed(s)
        if USE_CUDA:
            torch.cuda.manual_seed(s)