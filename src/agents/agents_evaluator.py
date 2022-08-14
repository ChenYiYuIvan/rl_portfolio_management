import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.environments.portfolio_end import PortfolioEnd
from src.agents.ddpg_agent import DDPGAgent
from src.agents.crp_agent import CRPAgent
from src.utils.file_utils import read_yaml_config
from src.utils.data_utils import date_list, snp
from empyrical import simple_returns
from empyrical import sharpe_ratio, sortino_ratio, max_drawdown, value_at_risk, conditional_value_at_risk


class AgentsEvaluator:

    def __init__(self, env, agents_list):
        
        self.env = env
        self.agents_list = agents_list


    def evaluate_all(self, market=True, plot_values=True, plot_weights=True, num_cols=5):
        # market: True to plot SPY values with other agents
        # plot_values: True to plot agents' values
        # plot_weights: True to plot agents' actions over time
        # num_cols: for plot_weights

        date_format = '%Y-%m-%d'

        agent_metrics = []
        agent_names = [agent.name for agent in self.agents_list] # for value plot legend
        
        if plot_values:
            fig1, ax1 = plt.subplots()
        if plot_weights:
            stock_names = ['CASH', *self.env.stock_names]
            num_rows = int(np.ceil(len(stock_names) / num_cols))
            fig2, ax2 = plt.subplots(num_rows, num_cols)

        if market: # also compare agents to market index
            start_id = date_list.index(self.env.start_date)
            end_id = date_list.index(self.env.end_date) + 1 # have to include day after last day
            dates = date_list[start_id : end_id]
            market_values = snp[start_id : end_id + 1, 3] # closing prices only
            market_values = market_values / market_values[0] # starting value = 1
            market_returns = simple_returns(market_values) # prices -> returns

            # market statistics
            agent_metrics.append({'agent': 'market',
                                'sharpe_ratio': sharpe_ratio(market_returns),
                                'sortino_ratio': sortino_ratio(market_returns),
                                'max_drawdown': max_drawdown(market_returns),
                                'var_95': value_at_risk(market_returns),
                                'cvar_95': conditional_value_at_risk(market_returns)
                                })

            if plot_values:
                df = {'date': dates, 'value': market_values[:-1]}
                df = pd.DataFrame(df)
                df['date'] = pd.to_datetime(df['date'], format=date_format)
                df.set_index('date', inplace=True)
                df.plot(ax=ax1, rot=30)
                agent_names.insert(0, 'market')

        for agent in self.agents_list: # current agent to plot
            reward, info = agent.eval(self.env)

            rate_of_return = np.array([el['simple_return'] for el in info])
            agent_metrics.append({'agent': agent.name,
                                  'sharpe_ratio': sharpe_ratio(rate_of_return),
                                  'sortino_ratio': sortino_ratio(rate_of_return),
                                  'max_drawdown': max_drawdown(rate_of_return),
                                  'var_95': value_at_risk(rate_of_return),
                                  'cvar_95': conditional_value_at_risk(rate_of_return)
                                  })

            if plot_values:
                df = pd.DataFrame(info)
                df = df[['date', 'port_value_old']]
                df['date'] = pd.to_datetime(df['date'], format=date_format)
                df.set_index('date', inplace=True)
                df.plot(ax=ax1, rot=30)

            if plot_weights:
                for stock_id in range(len(stock_names)):
                    row = int(stock_id / num_cols) # current row in figure
                    col = stock_id % num_cols # current col in figure
                    stock_name = stock_names[stock_id] # current stock to plot
                    info_stock = [{'date': item['date'], agent.name: item['action'][stock_id]} for item in info]
                    df2 = pd.DataFrame(info_stock)
                    df2['date'] = pd.to_datetime(df2['date'], format=date_format)
                    df2.set_index('date', inplace=True)
                    df2.plot(ax=ax2[row,col], title=stock_name, rot=30)

        # print portfolio metrics
        agent_metrics = pd.DataFrame(agent_metrics)
        agent_metrics.set_index('agent', inplace=True)
        print(agent_metrics)

        if plot_values: # plot agent generated portfolio values
            ax1.legend(agent_names)
            plt.show()

        if plot_weights: # plot agents' action for each stock
            fig2.legend(stock_names)
            plt.show()


def main():

    seed = 42

    env_config = read_yaml_config('env_default_train')
    ddpg_config = read_yaml_config('ddpg_default')

    env = PortfolioEnd(env_config)

    ddpg = DDPGAgent('ddpg', env, seed, ddpg_config)
    ddpg.load_actor_model('./checkpoints/checkpoints_ddpg/ddpg_ep499.pth')

    crp = CRPAgent('crp', env, seed)


    evaluator = AgentsEvaluator(env, [ddpg, crp])
    evaluator.evaluate_all()


if __name__ == '__main__':

    main()