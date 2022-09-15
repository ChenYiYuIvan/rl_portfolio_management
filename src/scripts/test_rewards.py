from src.agents.crp_agent import CRPAgent
from src.agents.ddpg_agent import DDPGAgent
from src.environments.portfolio import Portfolio
from src.utils.file_utils import read_yaml_config, get_checkpoint_folder
import numpy as np
import matplotlib.pyplot as plt


seed = 42

env_config = read_yaml_config('env_small_train')
env = Portfolio(env_config)

agent_config = read_yaml_config('ddpg_2')
agent = DDPGAgent('ddpg', env, seed, agent_config)
agent.load_actor_model(get_checkpoint_folder(agent, env) + '/ddpg_ep3.pth')

#agent = CRPAgent('crp', env, 42, 'diff_sharpe_ratio')

agent.reset()
curr_obs = env.reset()
curr_obs = agent.preprocess_data(curr_obs)

diff_sharpe_ratio_approx = []
sharpe_ratio = []

done = False
while not done:
    action = agent.predict_action(curr_obs)

    next_obs, done, info = env.step(action)
    diff_sharpe = agent.get_reward(info)
    next_obs = agent.preprocess_data(next_obs)

    diff_sharpe_ratio_approx.append(diff_sharpe)
    sharpe_ratio.append(info['sharpe_ratio'])

    curr_obs = next_obs

sharpe_ratio_approx = np.cumsum(diff_sharpe_ratio_approx)
diff_sharpe_ratio = np.diff(sharpe_ratio)

fig, ax = plt.subplots(2)
#ax[0].plot(sharpe_ratio_approx[10:] - sharpe_ratio[10:])
ax[0].plot(sharpe_ratio_approx)
ax[0].plot(sharpe_ratio)

ax[0].grid()

#ax[1].plot(diff_sharpe_ratio_approx[501:] - diff_sharpe_ratio[500:])
ax[1].plot(diff_sharpe_ratio_approx[1:])
ax[1].plot(diff_sharpe_ratio)

ax[1].grid()

fig.legend(['approx', 'actual'])
plt.show()

print(f'approx = {sharpe_ratio_approx[-1]} - actual = {sharpe_ratio[-1]}')