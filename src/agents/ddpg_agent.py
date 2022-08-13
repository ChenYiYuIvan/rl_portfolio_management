import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.cuda import amp

from src.agents.base_agent import BaseAgent

from src.utils.torch_utils import USE_CUDA, FLOAT, copy_params, update_params

from src.agents.ddpg_utils.actor import Actor
from src.agents.ddpg_utils.critic import Critic
from src.agents.ddpg_utils.replay_buffer import ReplayBuffer
from src.agents.ddpg_utils.noise import OrnsteinUhlenbeckActionNoise

from tqdm import tqdm
import wandb
import os


class DDPGAgent(BaseAgent):

    def __init__(self, name, env, seed, args):
        super().__init__(name, env, seed)
        
        self.actor = Actor(self.state_dim[2], self.state_dim[1])
        self.actor_target = Actor(self.state_dim[2], self.state_dim[1])
        self.actor_optim = Adam(self.actor.parameters(), lr=args.lr_actor)

        self.critic = Critic(self.state_dim[2], self.action_dim, self.state_dim[1])
        self.critic_target = Critic(self.state_dim[2], self.action_dim, self.state_dim[1])
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr_critic)

        self.loss = nn.MSELoss()

        # target networks must have same initial weights as originals
        copy_params(self.actor_target, self.actor)
        copy_params(self.critic_target, self.critic)

        self.buffer = ReplayBuffer(args.buffer_size)

        # noise for exploration
        self.actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(self.action_dim))

        # hyper-parameters
        self.num_episodes = args.num_episodes
        self.batch_size = args.batch_size
        self.tau = args.tau
        self.gamma = args.gamma

        # 
        self.eval_steps = args.eval_steps

        # 
        self.device = torch.device("cuda:0" if USE_CUDA else "cpu")
        if USE_CUDA:
            self.cuda()

        # freeze target networks, only update manually
        for param in self.actor_target.parameters():
            param.requires_grad = False

        for param in self.critic_target.parameters():
            param.requires_grad = False

        #
        self.checkpoint_folder = f'./checkpoints/{args.checkpoint_folder}'


    def update_policy(self, scaler):

        # sample batch
        s_batch, a_batch, r_batch, t_batch, s2_batch = self.buffer.sample_batch(self.batch_size)

        # set gradients to 0
        self.critic_optim.zero_grad()
        self.actor_optim.zero_grad()

        self.actor.requires_grad(False)
      
        with amp.autocast():
            # prepare target q batch
            next_q_values = self.critic_target(*s2_batch, self.actor_target(*s2_batch))
            target_q_batch = r_batch + self.gamma * t_batch * next_q_values

            # update critic
            q_batch = self.critic(*s_batch, a_batch)
            value_loss = self.loss(q_batch, target_q_batch)

        # backward pass for critic
        scaler.scale(value_loss).backward(retain_graph=True)
        scaler.step(self.critic_optim)

        self.actor.requires_grad(True)
        self.critic.requires_grad(False)

        # update actor
        with amp.autocast():
            policy_loss = -self.critic(*s_batch, self.actor(*s_batch))
            policy_loss = torch.mean(policy_loss)

        # backward pass for actor
        scaler.scale(policy_loss).backward(retain_graph=True)
        scaler.step(self.actor_optim)

        # update
        scaler.update()

        self.critic.requires_grad(True)

        # update target netweoks
        update_params(self.actor_target, self.actor, self.tau)
        update_params(self.critic_target, self.critic, self.tau)

        return value_loss, policy_loss


    def train(self, wandb_inst, env_test):
        # env_test: environment used to test agent during training (usually different from training environment)
        print("Begin train!")
        wandb_inst.watch((self.actor, self.critic))
        artifact = wandb.Artifact(name='ddpg', type='model')

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
        a,b = obs
        a = torch.tensor(a, dtype=FLOAT, device=self.device)
        b = torch.tensor(b, dtype=FLOAT, device=self.device)
        action = self.actor(a, b)
        if USE_CUDA:
            action = action.detach().cpu().numpy()
        else:
            action = action.detach().numpy()

        if exploration: # add exploration
            action += self.actor_noise()
        
        action = np.clip(action, 0, 1)
        action /= action.sum()

        return action


    def load_actor_model(self, path):
        self.actor.load_state_dict(torch.load(path))


    def set_train(self):
        self.actor.train()
        self.actor_target.train()
        self.critic.train()
        self.critic_target.train()


    def set_eval(self):
        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()


    def cuda(self):
        self.actor.cuda()
        self.actor_target.cuda()
        self.critic.cuda()
        self.critic_target.cuda()