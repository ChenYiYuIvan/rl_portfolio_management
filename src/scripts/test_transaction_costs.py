from src.environments.portfolio_end import PortfolioEnd
from src.utils.portfolio_utils import get_approx_trans_cost
from src.utils.file_utils import read_yaml_config
from src.agents.random_agent import RandomAgent
import numpy as np
import matplotlib.pyplot as plt



env_config = read_yaml_config('env_default_train')
env = PortfolioEnd(env_config)

agent = RandomAgent('rng', env, 42, 'log_return')

curr_rets, curr_weights = env.reset()
done = False

exact_vec = []
approx_vec = []
while not done:
    action = agent.predict_action((curr_rets, curr_weights))
    (next_rets, next_weights), done, info = env.step(action)

    exact = info['cost'] * info['port_value_old']
    approx = get_approx_trans_cost(curr_weights, next_weights, env.trading_cost) * info['port_value_old']

    curr_rets, curr_weights = next_rets, next_weights

    exact_vec.append(exact)
    approx_vec.append(approx)

exact_vec = np.array(exact_vec)
approx_vec = np.array(approx_vec)

fig, ax = plt.subplots()
ax.plot(np.divide(exact_vec - approx_vec, exact_vec))
plt.show()