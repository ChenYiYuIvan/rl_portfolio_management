import numpy as np
from src.agents.base_agent import BaseAgent
from src.utils.markowitz_utils import get_opt_portfolio


class MPTAgent(BaseAgent):

    def __init__(self, name, env, seed, objective):
        super().__init__(name, env, seed)

        self.objective = objective


    def predict_action(self, obs, exploration=False):
        return get_opt_portfolio(obs, self.objective, self.env.trading_cost)