from src.agents.base_agent import BaseAgent
import numpy as np


class CRPAgent(BaseAgent):
    # CRP = constantly rebalanced portfolio
    # at each period the portfolio is rebalanced to the initial wealth distribution among
    # all the assets (including the cash)

    def __init__(self, name, env, seed):
        super().__init__(name, env, seed)


    def predict_action(self, obs):
        return np.ones(self.action_dim) / self.action_dim
