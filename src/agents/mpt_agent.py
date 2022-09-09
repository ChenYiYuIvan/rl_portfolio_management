import numpy as np
from src.agents.base_agent import BaseAgent
from src.utils.portfolio_utils import get_opt_portfolio


class MPTAgent(BaseAgent):

    def __init__(self, name, env, seed, reward_type, objective):
        super().__init__(name, env, seed, reward_type)

        self.objective = objective


    def predict_action(self, obs, exploration=False):
        action, message = get_opt_portfolio(obs, self.objective, self.env.trading_cost)
        if message is not None:
            print(self.env.market.step_to_date(), message, sep=' - ')

        return action