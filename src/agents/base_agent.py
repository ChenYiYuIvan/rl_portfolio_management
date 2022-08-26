import numpy as np
import torch

from src.utils.torch_utils import USE_CUDA

# TODO: implement uniform buy and hold, best stock, best constant rebalanced portfolio

class BaseAgent:

    def __init__(self, name, env, seed):

        if seed > 0:
            self.seed(seed)

        self.name = name
        self.env = env

        self.state_dim = self.env.observation_space.shape # [price features, stocks, time window]
        self.action_dim = self.env.action_space.shape[0] # cash + num stocks


    def eval(self, env, render = False):
        # mode: exploration vs no exploration
        # render: plot
        # episode: current training episode (if eval is called during training)ù
        # table: to store eval results on wandb

        print("Begin eval!")

        # initial state of environment
        curr_obs = env.reset()
        curr_obs = self.preprocess_data(curr_obs)

        # initialize values
        infos = []
        ep_reward = 0

        # keep sampling until done
        done = False
        while not done:
            # select action
            action = self.predict_action(curr_obs)

            # step forward environment
            next_obs, reward, done, info = env.step(action)
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


    def predict_action(self, obs):
        raise NotImplementedError


    def preprocess_data(self, obs):
        return obs


    def seed(self,s):
        np.random.seed(s)
        torch.manual_seed(s)
        if USE_CUDA:
            torch.cuda.manual_seed(s)