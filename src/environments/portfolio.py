import gym
import numpy as np
from src.environments.market import Market
from empyrical import sharpe_ratio, sortino_ratio, max_drawdown, value_at_risk, conditional_value_at_risk
import pandas as pd
import matplotlib.pyplot as plt
from src.utils.data_utils import EPS


class Portfolio(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, config):
        super().__init__()

        # no slippage assumption: transaction happens immediately
        # -> stock prices are the same as when the order was put

        self.continuous = config.continuous  # bool to use continuous market assumption

        self.window_length = config.window_length
        self.start_date = config.start_date
        self.end_date = config.end_date

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
        # remove open and volume -> 3 values
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf,
                                                shape=(len(self.stock_names), self.window_length, 3), dtype=np.float32)

        # Portfolio environment in which everything is done at end of day
        # -> state s_t == weights and value of portfolio at end of day t before rebalance
        # -> action a_t == rebalanced portfolio
        # -> state s_{t+1} = f(s_t, a_t)

        # 1: price at end of day t before rebalancing
        # 2: rebalancing at end of day t
        # 3: price at end of day t+1 before rebalancing

        self.reset()

    def step(self, action):
        # execute 1 time step within the environment
        # action = new portfolio weights

        # observe open/.../close price of stocks
        curr_obs, next_obs, done = self.market.step()
        # obs shape: [stocks, time window, price features]

        action_info = self._take_action(curr_obs, action, next_obs)

        next_state = (next_obs, self.weights)

        return next_state, done, action_info

    def _take_action(self, curr_obs, action, next_obs):

        # 1
        weights1 = self.weights
        port_value1 = self.port_value

        # 2
        weights2 = action
        remaining_value = self._get_remaining_value(weights1, weights2)
        if remaining_value < 0 or remaining_value > 1:
            print(remaining_value)
            raise ValueError
        port_value2 = remaining_value * port_value1

        # 3
        curr_close = curr_obs[:, -1, 3]
        next_close = next_obs[:, -1, 3]
        relative_price = next_close / curr_close
        weights3 = (relative_price * weights2) / (np.dot(relative_price, weights2) + EPS)
        port_value3 = port_value2 * np.dot(relative_price, weights2)

        # possible reward signals
        log_return = np.log((port_value3 + EPS) / (port_value1 + EPS))
        simple_return = port_value3 / port_value1 - 1

        if len(self.simple_ret_vec) == 0:
            self.simple_ret_vec = np.array([simple_return])
            curr_sharpe_ratio = 0
            curr_sortino_ratio = 0
        else:
            self.simple_ret_vec = np.append(self.simple_ret_vec, simple_return)
            curr_sharpe_ratio = sharpe_ratio(self.simple_ret_vec, annualization=1)
            curr_sortino_ratio = sortino_ratio(self.simple_ret_vec, annualization=1)
        
        action_info = {
            'time_period': self.market.next_step - 1,
            'date': self.market.step_to_date(),
            'curr_obs': curr_obs,
            'next_obs': next_obs,
            'action': action,
            'weights_old': weights1,
            'weights_new': weights3,
            'cost': 1 - remaining_value,
            'port_value_old': port_value1,
            'port_value_new': port_value3,
            'log_return': log_return,
            'simple_return': simple_return,
            'sharpe_ratio': curr_sharpe_ratio,
            'sortino_ratio': curr_sortino_ratio,
            'max_drawdown': max_drawdown(self.simple_ret_vec),
            'var_95': value_at_risk(self.simple_ret_vec),
            'cvar_95': conditional_value_at_risk(self.simple_ret_vec),
            's&p500': self.market.snp[self.market.next_step - 1, 3] / self.market.snp[0, 3] * self.init_port_value,
        }

        # update values
        self.weights = weights3
        self.port_value = port_value3
        self.infos.append(action_info)

        return action_info

    def _get_remaining_value(self, weights_old, weights_new, iters=10):
        # initial value
        remaining_value = 1 - self.trading_cost * np.abs(weights_new - weights_old).sum()

        # fixed point iteration
        def cost(mu): return (1 - self.trading_cost*weights_old[0] - (2*self.trading_cost - self.trading_cost**2)*np.sum(
            np.maximum(0, weights_old[1:] - mu*weights_new[1:]))) / (1 - self.trading_cost*weights_new[0])
        for _ in range(iters):
            remaining_value = cost(remaining_value)
        
        # barely above 1
        diff_up = np.maximum(0, remaining_value - 1)
        if diff_up > 0 and diff_up < EPS:
            remaining_value = 1
        # barely below 0
        diff_down = np.minimum(0, remaining_value)
        if diff_down < 0 and diff_down > -EPS:
            remaining_value = 0
        
        return remaining_value

    def reset(self):
        # reset environment to initial state

        self.infos = []
        self.simple_ret_vec = [] # vector of simple returns

        # assume that starting portfolio is all cash
        self.weights = np.array([1] + [0 for _ in range(len(self.stock_names))])
        self.port_value = self.init_port_value  # initial value of portfolio

        # open, low, high, close, volume data of day 0
        # curr_obs shape: [stocks, time window, price features]
        curr_obs = self.market.reset()

        # initialize values needed for differential sharpe ratio (A_t and B_t)
        # A_t = first moment of returns
        # B_t = second momento of returns
        # initial portfolio is all cash 
        


        # end of day -> nothing has changed because only cash

        # s_t = (X_t, w_{t-1}^{end}})
        curr_state = (curr_obs, self.weights)

        return curr_state

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

    def copy_env(self, env2):
        # copy portfolio values from env2
        self.infos = env2.infos
        self.simple_ret_vec = env2.simple_ret_vec
        self.weights = env2.weights
        self.port_value = env2.port_value