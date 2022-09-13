from src.agents.ddpg_agent import DDPGAgent
from src.environments.portfolio import Portfolio
from src.utils.file_utils import get_checkpoint_folder, read_yaml_config

import numpy as np
import matplotlib.pyplot as plt

env_config = read_yaml_config('env_default_train')
env = Portfolio(env_config)

agent_config = read_yaml_config('ddpg_default')
agent = DDPGAgent('ddpg', env, 42, agent_config)

reward_vec = []
sharpe_vec = []
val_vec = []
for i in range(100):
    agent.load_actor_model(get_checkpoint_folder(agent, env) + f'/ddpg_ep{i}.pth')

    reward, info, val = agent.eval(env)
    reward_vec.append(reward)
    sharpe_vec.append(info[-1]['sharpe_ratio'])
    val_vec.append(val)


reward_vec = np.array(reward_vec)
sharpe_vec = np.array(sharpe_vec)
val_vec = np.array(val_vec)

fig, ax = plt.subplots(3)

ax[0].plot(reward_vec)
ax[1].plot(sharpe_vec)
ax[2].plot(val_vec)
plt.show()