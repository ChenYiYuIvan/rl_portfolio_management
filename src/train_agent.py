import wandb
from src.utils.file_utils import read_yaml_config
from src.environments.portfolio import Portfolio
from src.agents.ddpg_agent import DDPGAgent
from src.agents.sac_agent import SACAgent


def main(agent_name):

    seed = 42

    env_config_train = read_yaml_config('experiments/env_train_0')
    env_config_test = read_yaml_config('experiments/env_test_0')

    if agent_name == 'ddpg':
        agent_config = read_yaml_config('experiments/ddpg_8')
        config = {'env_train':vars(env_config_train), 'env_test':vars(env_config_test), 'agent': vars(agent_config)}

        wandb.login()
        with wandb.init(project="thesis", entity="mldlproj1gr2", config=config, mode="online") as run:

            env_train = Portfolio(env_config_train)
            env_test = Portfolio(env_config_test)

            agent = DDPGAgent('ddpg', env_train, seed, agent_config)

            #pretrained_path = './checkpoints_pretrained/msm_real_7_49/real_epoch99.pth'
            agent.train(run, env_test, pretrained_path=None)
            #agent.eval(env_test, render=False)

    elif agent_name == 'sac':
        agent_config = read_yaml_config('experiments/sac_2')
        config = {'env_train':vars(env_config_train), 'env_test':vars(env_config_test), 'agent': vars(agent_config)}

        wandb.login()
        with wandb.init(project="thesis", entity="mldlproj1gr2", config=config, mode="online") as run:

            env_train = Portfolio(env_config_train)
            env_test = Portfolio(env_config_test)

            agent = SACAgent('sac', env_train, seed, agent_config)

            #model_folder = get_checkpoint_folder(agent, env_train, agent.imitation_learning == 'passive')
            #print('Starting from checkpoint at epoch 43')
            #agent.load_actor_model(model_folder + '/sac_ep43.pth')

            agent.train(run, env_test)
            #agent.eval(env_test, render=False)


if __name__ == '__main__':

    main('ddpg')
    #main('sac')