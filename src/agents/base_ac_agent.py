import numpy as np
import torch
from torch.cuda import amp

from src.agents.base_agent import BaseAgent

from src.utils.torch_utils import USE_CUDA

from src.models.replay_buffer import ReplayBuffer

from tqdm import tqdm
import wandb
import os


class BaseACAgent(BaseAgent):

    def __init__(self, name, env, seed, args):
        super().__init__(name, env, seed)
        
        # define actors and critics networks and optimizers
        self.define_actors_critics(args)

        # target networks must have same initial weights as originals
        self.copy_params_to_target()

        # define replay buffer
        self.buffer = ReplayBuffer(args.buffer_size)

        # hyper-parameters
        self.num_episodes = args.num_episodes
        self.batch_size = args.batch_size
        self.tau = args.tau # for polyak averaging
        self.gamma = args.gamma # for bellman equation

        # evaluate model every few episodes
        self.eval_steps = args.eval_steps

        # cpu or gpu
        self.device = torch.device("cuda:0" if USE_CUDA else "cpu")
        if USE_CUDA:
            self.cuda()

        # freeze target networks, only update manually
        self.set_networks_grad('target', False)

        # define checkpoint folder
        self.checkpoint_folder = f'./checkpoints/{args.checkpoint_folder}'


    def define_actors_critics(self, args):
        raise NotImplementedError


    def copy_params_to_target(self):
        raise NotImplementedError


    def update_target_params(self):
        raise NotImplementedError


    def set_networks_grad(self, networks, requires_grad):
        # networks = {'actor', 'critic', 'target'}
        # requires_grad = {True, False}
        raise NotImplementedError


    def compute_value_loss(self, s_batch, a_batch, r_batch, t_batch, s2_batch):
        raise NotImplementedError


    def compute_policy_loss(self, s_batch):
        raise NotImplementedError


    def update_policy(self, scaler):

        # sample batch
        s_batch, a_batch, r_batch, t_batch, s2_batch = self.buffer.sample_batch(self.batch_size)

        # set gradients to 0
        self.critic_optim.zero_grad()
        self.actor_optim.zero_grad()

        self.set_networks_grad('actor', False)

        # compute loss for critic
        with amp.autocast():
            value_loss = self.compute_value_loss(s_batch, a_batch, r_batch, t_batch, s2_batch)

        # backward pass for critic
        scaler.scale(value_loss).backward()
        scaler.step(self.critic_optim)

        self.set_networks_grad('actor', True)
        self.set_networks_grad('critic', False)

        # compute loss for actor
        with amp.autocast():
            policy_loss = self.compute_policy_loss(s_batch)

        # backward pass for actor
        scaler.scale(policy_loss).backward()
        scaler.step(self.actor_optim)

        # update
        scaler.update()

        self.set_networks_grad('critic', True)

        # update target netweoks
        self.update_target_params()

        return value_loss, policy_loss


    def train(self, wandb_inst, env_test):
        # env_test: environment used to test agent during training (usually different from training environment)
        print("Begin train!")
        wandb_inst.watch(self.actor)
        artifact = wandb.Artifact(name=self.name, type='model')

        # create table to store actions for wandb
        table_train = []
        table_eval = []
        columns = ['episode', 'date', 'value_before', 'trans_cost', 'reward', 'CASH', *self.env.stock_names]


        # creating directory to store models if it doesn't exist
        if not os.path.isdir(self.checkpoint_folder):
            os.makedirs(self.checkpoint_folder)

        scaler = amp.GradScaler()

        step = 0
        max_ep_reward_val = -np.inf
        # loop over episodes
        for episode in range(self.num_episodes):

            # logging
            num_steps = self.env.market.end_idx - self.env.market.start_idx + 1
            tq = tqdm(total=num_steps)
            tq.set_description('episode %d' % (episode))

            # set models in trainig mode
            self.set_train()

            # initial state of environment
            curr_obs = self.env.reset()  # may need to normalize obs

            # initialize values
            ep_reward = 0

            # keep sampling until done
            done = False
            while not done:
                # select action
                if self.buffer.size() >= self.batch_size:
                    action = self.predict_action(curr_obs, True)
                else:
                    action = self.random_action()

                # step forward environment
                next_obs, reward, done, info_train = self.env.step(action)  # may need to normalize obs
                
                # add to buffer
                self.buffer.add(curr_obs, action, reward, done, next_obs)

                # if replay buffer has enough observations
                if self.buffer.size() >= self.batch_size:
                    value_loss, policy_loss = self.update_policy(scaler)
                    wandb_inst.log({'value_loss': value_loss, 'policy_loss': policy_loss}, step=step)

                ep_reward += reward
                curr_obs = next_obs

                tq.update(1)
                tq.set_postfix(ep_reward='%.6f' % ep_reward)

                wandb_inst.log({'episode': episode, 'reward_train': reward}, step=step)
                step += 1

            tq.close()
            print(f"Episode reward: {ep_reward}")
            wandb_inst.log({'episode_reward_train': ep_reward}, step=step)

            # evaluate model every few episodes
            if episode % self.eval_steps == self.eval_steps - 1:
                ep_reward_val, infos_eval = self.eval(env_test, render=False)
                print(f"Episode reward - eval: {ep_reward_val}")
                wandb_inst.log({'episode_reward_eval': ep_reward_val}, step=step)

                # store info data in table (both train and eval)
                # store train data only every few steps to reduce memory consuption
                table_train.append([episode, info_train['date'], info_train['port_value_old'], info_train['cost'],
                    info_train['reward'], *info_train['action']])
                for info_eval in infos_eval:
                    table_eval.append([episode, info_eval['date'], info_eval['port_value_old'], info_eval['cost'],
                        info_eval['reward'], *info_eval['action']])

                if ep_reward_val > max_ep_reward_val:
                    max_ep_reward_val = ep_reward_val
                    wandb_inst.summary['max_ep_reward_val'] = max_ep_reward_val

                # save trained model
                actor_path_name = os.path.join(self.checkpoint_folder,
                    f'{self.name}_ep{episode}.pth')
                torch.save(self.actor.state_dict(), actor_path_name)
                artifact.add_file(actor_path_name, name=f'{self.name}_ep{episode}.pth')

        wandb_inst.log({'table_train': wandb.Table(columns=columns, data=table_train)})
        wandb_inst.log({'table_eval': wandb.Table(columns=columns, data=table_eval)})
        wandb_inst.log_artifact(artifact)


    def eval(self, env, render = False):

        self.set_eval()
        return super().eval(env, render)


    def predict_action(self, obs, exploration=False):
        raise NotImplementedError


    def load_actor_model(self, path):
        self.actor.load_state_dict(torch.load(path))


    def set_train(self):
        raise NotImplementedError


    def set_eval(self):
        raise NotImplementedError

    def cuda(self):
        raise NotImplementedError