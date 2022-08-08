from subprocess import SW_HIDE
from .portfolio import Portfolio
import numpy as np


class PortfolioEnd(Portfolio):

    def __init__(self, start_date, end_date, window_length = 30, stock_names = None, trading_cost = 0.002, continuous = False):
        super().__init__(start_date, end_date, window_length, stock_names, trading_cost, continuous)

        # Portfolio environment in which everything is done at end of day
        # -> state s_t == weights and value of portfolio at end of day t before rebalance
        # -> action a_t == rebalanced portfolio
        # -> state s_{t+1} = f(s_t, a_t)

        # 1: price at end of day t before rebalancing
        # 2: rebalancing at end of dat t
        # 3: price at end of day t+1 before rebalancing


    def step(self, action):
        # execute 1 time step within the environment
        # action = new portfolio weights

        curr_obs, next_obs, done = self.market.step()  # observe open/.../close price of stocks

        reward, reward_info = self._take_action(curr_obs, action, next_obs)

        next_state = (next_obs[:, :, 1:], self.weights)

        return next_state, reward, done, reward_info


    def _take_action(self, curr_obs, action, next_obs):
        
        # 1
        weights1 = self.weights
        port_value1 = self.port_value

        # 2
        weights2 = action
        trans_cost = self._get_trans_cost(port_value1, weights1, weights2)
        assert trans_cost < port_value1, 'Transaction cost is bigger than current portfolio value'
        port_value2 = port_value1 - trans_cost

        # 3
        curr_close = curr_obs[:, -1, 3]
        next_close = next_obs[:, -1, 3]
        relative_price = next_close / curr_close
        weights3 = (relative_price * weights2) / np.dot(relative_price, weights2)
        port_value3 = port_value2 * np.dot(relative_price, weights2)

        # reward
        log_return = np.log(port_value3 / port_value1)
        simple_return = port_value3 / port_value1 - 1

        reward = log_return

        reward_info = {
            'time_period': self.market.next_step - 1,
            'date': self.market.step_to_date(),
            'curr_obs': curr_obs,
            'next_obs': next_obs,
            'action': action,
            'weights_old': weights1,
            'weights_new': weights3,
            'cost': trans_cost,
            'port_value_old': port_value1,
            'port_value_new': port_value3,
            'log_return': log_return,
            'simple_return': simple_return,
            'reward': reward,
            's&p500': self.market.snp[self.market.next_step - 1, 3] / self.market.snp[0, 3] * self.init_port_value,
        }

        # update values
        self.weights = weights3
        self.port_value = port_value3
        self.infos.append(reward_info)

        return reward, reward_info



    def reset(self):
        # reset environment to initial state

        self.infos = []

        # assume that starting portfolio is all cash
        weights = np.array([1] + [0 for _ in range(len(self.stock_names))])
        port_value = self.init_port_value # initial value of portfolio

        # open, low, high, close, volume data of day 0
        curr_obs = self.market.reset()
        close_price = curr_obs[:, -1, 3]
        open_price = curr_obs[:, -1, 0]
        relative_price = close_price / open_price

        # end of day -> values have changed
        self.weights = (relative_price * weights) / np.dot(relative_price, weights)
        self.port_value = port_value * np.dot(relative_price, weights)

        # s_t = (X_t, w_{t-1}^{end}})
        curr_state = (curr_obs[:,:,1:], self.weights)

        return curr_state