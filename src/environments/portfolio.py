import gym
import numpy as np
from src.environments.market import Market
from scipy.special import softmax
import pandas as pd
import matplotlib.pyplot as plt

class Portfolio(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, start_date, end_date, window_length = 30, stock_names = None, trading_cost = 0.002):
        super().__init__()

        self.start_date = start_date
        self.end_date = end_date
        self.window_length = window_length

        self.init_port_value = 1000

        if stock_names is None:
            self.stock_names = ["AAPL", "ATVI", "CMCSA", "COST", "CSX", "DISH", "EA", "EBAY", "FB", "GOOGL", "HAS", "ILMN", "INTC", "MAR", "REGN", "SBUX"]
        else:
            self.stock_names = stock_names

        self.trading_cost = trading_cost

        self.market = Market(self.start_date, self.end_date, self.window_length, self.stock_names)

        # action space = new portfolio weights
        # no short selling allowed
        self.action_space = gym.spaces.Box(low=-np.inf, high=np.inf,
            shape=(len(self.stock_names) + 1,), dtype=np.float32)  # include cash

        # observation space = past values of asset prices
        # open, high, low, close, volume -> 5 values
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf,
            shape=(len(self.stock_names), self.window_length, 5), dtype=np.float32)

        self.reset()


    def step(self, action):
        # execute 1 time step within the environment
        # action = new portfolio weights

        obs, done, _ = self.market.step()  # observe open/.../close price of stocks
        new_weights = softmax(action)  # new portfolio weights (sum = 1)
        reward, reward_info = self.get_reward(obs, new_weights)  # compute reward of action

        # update info
        info = self.parse_info(obs, reward, reward_info, done)
        self.infos.append(info)
        self.weights = new_weights
        self.port_value = info['port_value_new']
        
        return obs, reward, info, done


    def reset(self):
        # reset environment to initial state

        self.infos = []

        # assume that starting portfolio is all cash
        self.weights = np.array([1] + [0 for i in range(len(self.stock_names))])
        self.port_value = self.init_port_value # initial value of portfolio

        self.market.reset()



    def render(self, mode='human', close=False):
        # plot value of portfolio and compare with value of market (~ S&P500 ETF)
        df = [{'date': info['date'], 'portfolio': info['port_value_old'], 'market': snp} 
            for info, snp in zip(self.infos, self.market.snp[:,3])]
        df = pd.DataFrame(df)

        # scale value of market to have same initial value as portfolio
        # only use closing values
        df['market'] = df['market'].apply(lambda x: x / self.market.snp[0,3] * self.init_port_value)
        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
        df.set_index('date', inplace=True)
        df.plot(fig=plt.gcf(), rot=30)


    def get_reward(self, obs, action):
        # observe open, low, high, close, volume data of day t
        # compute optimal weight for next day (time period t+1)
        # compute cost of rebalancing and reward

        close_price = obs[:, -1, 3]
        open_price = obs[:, -1, 0]
        relative_price = close_price / open_price

        weights_old = self.weights
        weights_old_end = (relative_price * weights_old) / np.dot(relative_price, weights_old)
        weights_new = action

        port_value_old = self.port_value # value of portfolio immediately after last rebalance
        port_value_old_end = port_value_old * np.dot(relative_price, weights_old)

        # TODO: wrong, fix (fixed point iteration) 
        trans_cost = port_value_old_end * self.trading_cost * np.abs(weights_new - weights_old_end).sum()
        assert trans_cost < port_value_old_end, 'Transaction cost is bigger than current portfolio value'

        port_value_new = port_value_old_end - trans_cost

        log_return = np.log(port_value_new / port_value_old)
        simple_return = port_value_new / port_value_old - 1


        reward_info = {
            'weights_old': weights_old,
            'weights_old_end': weights_old_end,
            'weights_new': weights_new,
            'log_return': log_return,
            'simple_return': simple_return,
            'cost': trans_cost,
            'port_value_old': port_value_old,
            'port_value_old_end': port_value_old_end,
            'port_value_new': port_value_new,
            's&p500': self.market.snp[self.market.next_step - 1, 3] / self.market.snp[0, 3] * self.init_port_value,
        }

        return log_return, reward_info


    def parse_info(self, obs, reward, reward_info, done):
        info = {
            'time_period': self.market.next_step - 1,
            'date': self.market.step_to_date(),
            'obs': obs,
            'reward': reward,
            'done': done,
        }

        for key in reward_info:
            info[key] = reward_info[key]

        return info