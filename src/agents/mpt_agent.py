import numpy as np
from src.agents.base_agent import BaseAgent
from src.utils.portfolio_utils import get_opt_portfolio


class MPTAgent(BaseAgent):

    def __init__(self, name, env, seed, objective):
        super().__init__(name, env, seed)

        self.objective = objective


    def predict_action(self, obs, exploration=False, verbose=True):
        action, message = get_opt_portfolio(obs, self.objective, self.env.trading_cost)
        if message is not None and verbose:
            print(self.env.market.step_to_date(), message, sep=' - ')

        return action