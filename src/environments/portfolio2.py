from src.environments.portfolio import Portfolio
from scipy.special import softmax
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Portfolio2(Portfolio):

    def __init__(self, start_date, end_date, window_length = 30, stock_names = None, trading_cost = 0.002):
        # portfolio without assuming that opening prices of next day are equal
        # to closing prices of current day

        super().__init__(start_date, end_date, window_length, stock_names, trading_cost)


    def step(self, action):
        # 1: starting portfolio value and weights for timestep t-1 (start of period t)
        # 2: portfolio value and weights given stock value at end of period t
        # 3: just after rebalancing at end of period t (weights == action)
        # 4: value and weights fiven stock value at start of period t+1

        # 1->2: due to change in value between start and end of period t
        # 2->3: rebalancing + transaction costs
        # 3->4: due to change in value between end of period t and start of period t+1

        obs, done, future = self.market.step()  # observe open/.../close price of stocks
        action = softmax(action)  # new portfolio weights (sum = 1)
        reward, weights_new, value_new, reward_info = self.get_reward(obs, action, future)  # compute reward of action

        # update info
        info = self.parse_info(obs, reward, reward_info, done)
        self.infos.append(info)
        self.weights = weights_new
        self.port_value = value_new
        
        return obs, reward, info, done


    def render(self, mode='human', close=False):
        # plot value of portfolio and compare with value of market (~ S&P500 ETF)
        df = [{'date': info['date'], 'portfolio': info['port_value1'], 'market': snp} 
            for info, snp in zip(self.infos, self.market.snp[:,3])]
        df = pd.DataFrame(df)

        # scale value of market to have same initial value as portfolio
        # only use closing values
        df['market'] = df['market'].apply(lambda x: x / self.market.snp[0,3] * self.init_port_value)
        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
        df.set_index('date', inplace=True)
        df.plot(fig=plt.gcf(), rot=30)


    def get_reward(self, obs, action, next_open = None):
        # observe open, low, high, close, volume data of day t
        # compute optimal weight for next day (time period t+1)
        # compute cost of rebalancing and reward

        close_price = obs[:, -1, 3]
        open_price = obs[:, -1, 0]

        relative_price = close_price / open_price
        relative_price2 = next_open / close_price

        weights1 = self.weights
        port_value1 = self.port_value

        weights2 = (relative_price * weights1) / np.dot(relative_price, weights1)
        port_value2 = port_value1 * np.dot(relative_price, weights1)

        weights3 = action
        # TODO: wrong, fix (fixed point iteration) 
        trans_cost = port_value2 * self.trading_cost * np.abs(weights3 - weights2).sum()
        assert trans_cost < port_value2, 'Transaction cost is bigger than current portfolio value'
        port_value3 = port_value2 - trans_cost

        weights4 = (relative_price2 * weights3) / np.dot(relative_price2, weights3)
        port_value4 = port_value3 * np.dot(relative_price2, weights3)
        

        log_return = np.log(port_value4 / port_value1)
        simple_return = port_value4 / port_value1 - 1

        reward = log_return

        reward_info = {
            'weights1': weights1,
            'weights2': weights2,
            'weights3': weights3,
            'weights4': weights4,
            'cost': trans_cost,
            'port_value1': port_value1,
            'port_value2': port_value2,
            'port_value3': port_value3,
            'port_value4': port_value4,
            'log_return': log_return,
            'simple_return': simple_return,
            's&p500': self.market.snp[self.market.next_step - 1, 3] / self.market.snp[0, 3] * self.init_port_value,
        }

        return reward, weights4, port_value4, reward_info