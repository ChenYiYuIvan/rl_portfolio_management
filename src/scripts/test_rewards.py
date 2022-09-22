from src.agents.random_agent import RandomAgent
from src.environments.portfolio import Portfolio
from src.utils.file_utils import read_yaml_config
import numpy as np
import matplotlib.pyplot as plt

reward_type = 'sortino_ratio'

seed = 42

env_config = read_yaml_config('default/env_small_train')
env = Portfolio(env_config)

agent = RandomAgent('crp', env, 42)
agent.reward_type = f'diff_{reward_type}'
agent.reward_scale = 1e-4

agent.reset()
curr_obs = env.reset(test_mode=True)
curr_obs = agent.preprocess_data(curr_obs)

diff_reward_approx = []
reward = []

done = False
while not done:
    action = agent.predict_action(curr_obs)

    next_obs, done, info = env.step(action)
    diff_reward = agent.get_reward(info)
    next_obs = agent.preprocess_data(next_obs)

    diff_reward_approx.append(diff_reward)
    reward.append(info[reward_type])

    curr_obs = next_obs

reward_approx = np.cumsum(diff_reward_approx)
diff_reward = np.diff(reward)

fig, ax = plt.subplots(2)
ax[0].plot(reward_approx)
ax[0].plot(reward)
ax[0].grid()

ax[1].plot(diff_reward_approx[1:])
ax[1].plot(diff_reward)
ax[1].grid()

fig.legend(['approx', 'actual'])
plt.show()

print(f'approx = {reward_approx[-1]} - actual = {reward[-1]}')