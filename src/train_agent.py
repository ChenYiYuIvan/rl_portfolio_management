import wandb
from src.utils.file_utils import read_yaml_config
from src.environments.portfolio import Portfolio
from src.agents.ddpg_agent import DDPGAgent
from src.agents.sac_agent import SACAgent
from src.agents.td3_agent import TD3Agent


def main(agent_name):

    seed = 0
    env_num = 2

    env_config_train = read_yaml_config(f'experiments/env_train_{env_num}')
    env_config_test = read_yaml_config(f'experiments/env_test_{env_num}')
    env_vers = env_config_train.name

    env_train = Portfolio(env_config_train)
    env_test = Portfolio(env_config_test)

    if agent_name == 'ddpg':
        agent_config = read_yaml_config('experiments/ddpg_12')
        agent = DDPGAgent('ddpg', env_train, seed, agent_config)

    elif agent_name == 'td3':
        agent_config = read_yaml_config('experiments/td3_2')
        agent = TD3Agent('td3', env_train, seed, agent_config)

    elif agent_name == 'sac':
        agent_config = read_yaml_config('experiments/sac_18')
        agent = SACAgent('sac', env_train, seed, agent_config)

    agent_vers = agent_config.name

    config = {'env_train':vars(env_config_train), 'env_test':vars(env_config_test), 'agent': vars(agent_config)}

    wandb.login()
    with wandb.init(project="thesis", entity="mldlproj1gr2", name=f"{agent_vers}_{env_vers}",
                    config=config, mode="online") as run:

        pretrained_path = f'./checkpoints_forecaster/{agent_config.network_type}_{agent_config.preprocess}_env{env_num}/ep999_lstm_forecaster.pth'
        agent.train(run, env_test, pretrained_path)
        #agent.eval(env_test, render=False)


if __name__ == '__main__':

    #main('ddpg')
    #main('td3')
    main('sac')