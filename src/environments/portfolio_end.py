from src.environments.portfolio import Portfolio
from src.utils.data_utils import EPS, prices_to_simplereturns
from src.utils.portfolio_utils import get_sharpe_ratio
from empyrical import sharpe_ratio, sortino_ratio, max_drawdown, value_at_risk, conditional_value_at_risk
import numpy as np
import gym


class PortfolioEnd(Portfolio):

    def __init__(self, config):
        super().__init__(config)

        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf,
                                                shape=(len(self.stock_names), self.window_length, 3), dtype=np.float32)

        # Portfolio environment in which everything is done at end of day
        # -> state s_t == weights and value of portfolio at end of day t before rebalance
        # -> action a_t == rebalanced portfolio
        # -> state s_{t+1} = f(s_t, a_t)

        # 1: price at end of day t before rebalancing
        # 2: rebalancing at end of day t
        # 3: price at end of day t+1 before rebalancing


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
            curr_sharpe_ratio = sharpe_ratio(self.simple_ret_vec, annualization=len(self.simple_ret_vec))
            curr_sortino_ratio = sortino_ratio(self.simple_ret_vec, annualization=len(self.simple_ret_vec))
        
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
