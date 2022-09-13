import wandb
from src.utils.file_utils import read_yaml_config
from src.environments.portfolio import Portfolio
from src.agents.ddpg_agent import DDPGAgent
from src.agents.sac_agent import SACAgent


def main(agent_name):

    seed = 42

    env_config_train = read_yaml_config('env_default_train')
    env_config_test = read_yaml_config('env_default_test')

    if agent_name == 'ddpg':
        agent_config = read_yaml_config('ddpg_1')
        config = {'env_train':vars(env_config_train), 'env_test':vars(env_config_test), 'agent': vars(agent_config)}

        wandb.login()
        with wandb.init(project="thesis", entity="mldlproj1gr2", config=config, mode="online") as run:

            env_train = Portfolio(env_config_train)
            env_test = Portfolio(env_config_test)

            agent = DDPGAgent('ddpg', env_train, seed, agent_config)

            pretrained_path = './checkpoints_pretrained/cnn_real_7_49_noise/real_epoch59.pth'
            agent.train(run, env_test, None)
            #agent.eval(env_test, render=False)

    elif agent_name == 'sac':
        agent_config = read_yaml_config('sac_default')
        config = {'env_train':vars(env_config_train), 'env_test':vars(env_config_test), 'agent': vars(agent_config)}

        wandb.login()
        with wandb.init(project="thesis", entity="mldlproj1gr2", config=config, mode="online") as run:

            env_train = Portfolio(env_config_train)
            env_test = Portfolio(env_config_test)

            agent = SACAgent('sac', env_train, seed, agent_config)

            agent.train(run, env_test)
            #agent.eval(env_test, render=False)


if __name__ == '__main__':

    main('ddpg')
    #main('sac')