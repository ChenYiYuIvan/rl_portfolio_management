import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.utils.data_utils import plot_stocks_info, EPS
from empyrical import simple_returns
from empyrical import sharpe_ratio, sortino_ratio, max_drawdown, value_at_risk, conditional_value_at_risk


class AgentsEvaluator:

    def __init__(self, env, agents_list):

        self.env = env
        self.agents_list = agents_list

    def evaluate_all(self, exploration=False, market=True, plot_stocks=False, plot_values=True, plot_weights=True, plot_log=True, num_cols=5):
        # market: True to plot SPY values with other agents
        # plot_values: True to plot agents' values
        # plot_weights: True to plot agents' actions over time
        # num_cols: for plot_weights

        date_format = '%Y-%m-%d'

        agent_metrics = []
        # for value plot legend
        agent_names = [agent.name for agent in self.agents_list]

        if plot_values:
            fig1, ax1 = plt.subplots()
        if plot_weights:
            stock_names = ['CASH', *self.env.stock_names]
            num_rows = int(np.ceil(len(stock_names) / num_cols))
            fig2, ax2 = plt.subplots(num_rows, num_cols, squeeze=False)

        for agent in self.agents_list:  # current agent to plot
            reward, infos, end_port_value = agent.eval(
                self.env, exploration=exploration)

            info = infos[-1]
            # agent statistics
            agent_metrics.append({
                'agent': agent.name,
                'final_value': end_port_value,
                'mean_log_rets': np.array([a['log_return'] for a in infos]).mean(),
                'sharpe_ratio': info['sharpe_ratio'],
                'sortino_ratio': info['sortino_ratio'],
                'max_drawdown': info['max_drawdown'],
                'var_95': info['var_95'],
                'cvar_95': info['cvar_95'],
            })

            if plot_values:
                df = pd.DataFrame(infos)
                df = df[['date', 'port_value_old']]
                df['date'] = pd.to_datetime(df['date'], format=date_format)
                df.set_index('date', inplace=True)
                df.plot(ax=ax1, rot=30)

            if plot_weights:
                for stock_id in range(len(stock_names)):
                    row = int(stock_id / num_cols)  # current row in figure
                    col = stock_id % num_cols  # current col in figure
                    stock_name = stock_names[stock_id]  # current stock to plot
                    info_stock = [
                        {'date': item['date'], agent.name: item['action'][stock_id]} for item in infos]
                    df2 = pd.DataFrame(info_stock)
                    df2['date'] = pd.to_datetime(
                        df2['date'], format=date_format)
                    df2.set_index('date', inplace=True)
                    df2.plot(ax=ax2[row, col], title=stock_name,
                             rot=30, legend=False)

        if market:  # also compare agents to market index
            agent_names.append('market')
            
            # closing prices only
            market_values = self.env.market.snp[self.env.window_length-1:, 3]
            market_values = market_values / \
                market_values[0]  # starting value = 1
            market_log_rets = np.log(
                market_values[1:] + EPS) - np.log(market_values[:-1] + EPS)
            market_returns = simple_returns(market_values)  # prices -> returns

            # market statistics
            agent_metrics.append({
                'agent': 'market',
                'final_value': market_values[-1],
                'mean_log_rets': np.mean(market_log_rets),
                'sharpe_ratio': sharpe_ratio(market_returns, annualization=1),
                'sortino_ratio': sortino_ratio(market_returns, annualization=1),
                'max_drawdown': max_drawdown(market_returns),
                'var_95': value_at_risk(market_returns),
                'cvar_95': conditional_value_at_risk(market_returns)
            })

            if plot_values:
                df = {
                    'date': self.env.market.date_list[self.env.window_length-1:], 'value': market_values}
                df = pd.DataFrame(df)
                df['date'] = pd.to_datetime(df['date'], format=date_format)
                df.set_index('date', inplace=True)
                df.plot(ax=ax1, rot=30)

        # print portfolio metrics
        agent_metrics = pd.DataFrame(agent_metrics)
        agent_metrics.set_index('agent', inplace=True)
        print(agent_metrics)

        if plot_values:  # plot agent generated portfolio values
            ax1.legend(agent_names)

        if plot_weights:  # plot agents' action for each stock
            if market:
                fig2.legend(agent_names[:-1])
            else:
                fig2.legend(agent_names)

        if plot_stocks:  # plot stock values
            plot_stocks_info(self.env, num_cols=num_cols, plot_log=plot_log)

        plt.show()
