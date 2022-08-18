import gym
import numpy as np
from src.environments.market import Market
from empyrical import sharpe_ratio, max_drawdown
import pandas as pd
import matplotlib.pyplot as plt


class Portfolio(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, config):
        super().__init__()

        # no slippage assumption: transaction happens immediately
        # -> stock prices are the same as when the order was put

        self.eps = 1e-8

        self.continuous = config.continuous  # bool to use continuous market assumption
        self.normalize = config.normalize  # divide price matrix by close price of time t

        self.start_date = config.start_date
        self.end_date = config.end_date
        self.window_length = config.window_length

        self.init_port_value = config.init_portfolio_value

        self.stock_names = config.stock_names

        self.trading_cost = config.trading_cost

        self.market = Market(self.start_date, self.end_date, self.window_length, self.stock_names)

        # action space = new portfolio weights
        # no short selling allowed
        self.action_space = gym.spaces.Box(low=0, high=1,
                                           shape=(len(self.stock_names) + 1,), dtype=np.float32)  # include cash

        # observation space = past values of asset prices
        # open, high, low, close, volume -> 5 values
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf,
                                                shape=(len(self.stock_names), self.window_length, 5), dtype=np.float32)

        self.reset()


    def step(self, action):
        # execute 1 time step within the environment
        # action = new portfolio weights

        # observe open/.../close price of stocks
        curr_obs, next_obs, done = self.market.step()

        reward, reward_info = self._take_action(curr_obs, action, next_obs)

        return next_obs, reward, done, reward_info


    def _take_action(self, curr_obs, action, next_obs):

        # observe open, low, high, close, volume data of day t
        close_price = curr_obs[:, -1, 3]
        open_price = curr_obs[:, -1, 0]
        relative_price = close_price / open_price

        # 1: start of day t (due to decision taken at end of day t-1)
        weights1 = self.weights
        port_value1 = self.port_value

        # 2: end of day t - just before rebalancing (due to diff between close and open)
        weights2 = (relative_price * weights1) / np.dot(relative_price, weights1)
        port_value2 = port_value1 * np.dot(relative_price, weights1)

        # 3: end of day t - just after rebalancing (due to action taken)
        # weights3 = softmax(action)  # new portfolio weights (sum = 1)
        weights3 = action
        remaining_value = self._get_remaining_value(weights2, weights3)
        assert remaining_value >= 0, 'Transaction cost is bigger than current portfolio value'
        assert remaining_value <= 1, 'Transaction cost less than 0'
        port_value3 = remaining_value * port_value2

        if not self.continuous:
            # 4: start of day t+1 (due to diff between next open and close)
            next_open = next_obs[:, -1, 0]
            relative_price2 = next_open / close_price
            weights4 = (relative_price2 * weights3) / np.dot(relative_price2, weights3)
            port_value4 = port_value3 * np.dot(relative_price2, weights3)

            weights_end = weights4
            port_value_end = port_value4

        else:
            # price at end of day t == price at start of day t+1
            weights_end = weights3
            port_value_end = port_value3

        # reward
        log_return = np.log(port_value_end / port_value1)
        simple_return = port_value_end / port_value1 - 1

        reward = log_return

        reward_info = {
            'time_period': self.market.next_step - 1,
            'date': self.market.step_to_date(),
            'curr_obs': curr_obs,
            'next_obs': next_obs,
            'action': action,
            'weights_old': weights1,
            'weights_new': weights_end,
            'cost': 1 - remaining_value,
            'port_value_old': port_value1,
            'port_value_new': port_value_end,
            'log_return': log_return,
            'simple_return': simple_return,
            'reward': reward,
            's&p500': self.market.snp[self.market.next_step - 1, 3] / self.market.snp[0, 3] * self.init_port_value,
        }

        # update values
        self.weights = weights_end
        self.port_value = port_value_end
        self.infos.append(reward_info)

        return reward, reward_info

    def _get_remaining_value(self, weights_old, weights_new, iters=10):
        # initial value
        remaining_value = self.trading_cost * np.abs(weights_new - weights_old).sum()

        # fixed point iteration
        def cost(mu): return (1 - self.trading_cost*weights_old[0] - (2*self.trading_cost - self.trading_cost**2)*np.sum(
            np.maximum(0, weights_old[1:] - mu*weights_new[1:]))) / (1 - self.trading_cost*weights_new[0])
        for _ in range(iters):
            remaining_value = cost(remaining_value)
        
        # barely above 1
        diff_up = np.maximum(0, remaining_value - 1)
        if diff_up > 0 and diff_up < self.eps:
            remaining_value = 1
        # barely below 0
        diff_down = np.minimum(0, remaining_value)
        if diff_down < 0 and diff_down > -self.eps:
            remaining_value = 0
        
        return remaining_value


    def reset(self):
        # reset environment to initial state

        self.infos = []

        # assume that starting portfolio is all cash
        self.weights = np.array([1] + [0 for _ in range(len(self.stock_names))])
        self.port_value = self.init_port_value  # initial value of portfolio

        curr_obs = self.market.reset()

        return curr_obs


    def render(self, mode='human', close=False):
        # plot value of portfolio and compare with value of market (~ S&P500 ETF)
        df = [{'date': info['date'], 'market': info['s&p500'], 'portfolio': info['port_value_new'], 'rate_of_return': info['simple_return']}
              for info in self.infos]
        df = pd.DataFrame(df)

        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
        df.set_index('date', inplace=True)

        mdd = max_drawdown(df.rate_of_return)
        sharpe = sharpe_ratio(df.rate_of_return)
        title = 'max_drawdown={: 2.2%} sharpe_ratio={: 2.4f}'.format(mdd, sharpe)

        df[['market', 'portfolio']].plot(title=title, rot=30)
        plt.show()
