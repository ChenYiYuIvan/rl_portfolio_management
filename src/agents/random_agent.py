from src.agents.base_agent import BaseAgent
import numpy as np


class RandomAgent(BaseAgent):
    # at each period the portfolio is rebalanced with random weights

    def __init__(self, name, env, seed, reward_type):
        super().__init__(name, env, seed, reward_type)


    def predict_action(self, obs):
        return self.random_action()
