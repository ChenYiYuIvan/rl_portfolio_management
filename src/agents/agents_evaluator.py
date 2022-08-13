from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.environments.portfolio_end import PortfolioEnd
from src.agents.ddpg_agent import DDPGAgent
from src.agents.crp_agent import CRPAgent
from src.utils.file_utils import read_json_config, read_yaml_config
from empyrical import sharpe_ratio, sortino_ratio, max_drawdown, value_at_risk, conditional_value_at_risk


class AgentsEvaluator:

    def __init__(self, env, agents_list):
        
        self.env = env
        self.agents_list = agents_list


    def evaluate_all(self, market=True):

        df = defaultdict(dict)

        info_agents = []
        metric_agents = [] # for every agent: sharpe ratio, drawdown
        for agent in self.agents_list: # for every agent
            reward, info = agent.eval(self.env)

            rate_of_return = np.array([el['simple_return'] for el in info])
            metric_agents.append({'agent': agent.name,
                                  'sharpe_ratio': sharpe_ratio(rate_of_return),
                                  'sortino_ratio': sortino_ratio(rate_of_return),
                                  'max_drawdown': max_drawdown(rate_of_return),
                                  'var_95': value_at_risk(rate_of_return),
                                  'cvar_95': conditional_value_at_risk(rate_of_return)
                                  })

            info_modified = []
            for item in info: # for every timestep
                if market: # add market data
                    info_modified.append({'date': item['date'], 'market': item['s&p500']})
                # add agent data
                info_modified.append({'date': item['date'], agent.name: item['port_value_old']})

            info_agents += info_modified
            market = False # add market data only once

        for item in info_agents:
            df[item['date']].update(item)

        # print portfolio metrics
        metric_agents = pd.DataFrame(metric_agents)
        metric_agents.set_index('agent', inplace=True)
        print(metric_agents)

        # plot portfolio value over time
        df = list(df.values())
        df = pd.DataFrame(df)
        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
        df.set_index('date', inplace=True)
        df.plot(rot=30)
        plt.show()


def main():

    args = read_yaml_config('ddpg_default')

    env = PortfolioEnd(args.start_test, args.end_test, args.window_length, args.stock_names, args.trading_cost, args.continuous, args.normalize)

    ddpg = DDPGAgent('ddpg', env, args)
    ddpg.load_actor_model('./checkpoints_ddpg/ddpg_ep499.pth')

    crp = CRPAgent('crp', env, args)


    evaluator = AgentsEvaluator(env, [ddpg, crp])
    evaluator.evaluate_all()


if __name__ == '__main__':

    main()