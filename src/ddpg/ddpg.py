import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.cuda import amp

from ..utils.torch_utils import USE_CUDA, FLOAT, copy_params, update_params

from .actor import Actor
from .critic import Critic
from .replay_buffer import ReplayBuffer
from .noise import OrnsteinUhlenbeckActionNoise

from tqdm import tqdm
import wandb
import os


class DDPG():

    def __init__(self, env, args):
        
        if args.seed > 0:
            self.seed(args.seed)

        self.env = env
        self.device = self.env.device

        self.state_dim = self.env.observation_space.shape
        self.action_dim = self.env.action_space.shape[0]

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


    def train(self, wandb_inst):
        print("Begin train!")
        wandb_inst.watch((self.actor, self.critic))
        artifact = wandb.Artifact(name='ddpg', type='model')

        scaler = amp.GradScaler()

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
                    action = self.predict_action(curr_obs)
                else:
                    action = self.random_action()

                # step forward environment
                next_obs, reward, done, info = self.env.step(action)  # may need to normalize obs
                
                # add to buffer
                self.buffer.add(curr_obs, action, reward, done, next_obs)

                # if replay buffer has enough observations
                if self.buffer.size() >= self.batch_size:
                    self.update_policy(scaler)

                ep_reward += reward
                curr_obs = next_obs

                tq.update(1)
                tq.set_postfix(ep_reward='%.6f' % ep_reward)

            tq.close()
            print(f"Episode reward: {ep_reward}")
            wandb_inst.log({'episode_reward_train': ep_reward}, step=episode)

            # evaluate model every few episodes
            if episode % self.eval_steps == self.eval_steps - 1:
                ep_reward_val = self.eval()
                print(f"Episode reward - eval: {ep_reward}")
                wandb_inst.log({'episode_reward_eval': ep_reward_val}, step=episode)

                # save trained model
                actor_path_name = os.path.join(wandb_inst.config.save_model_path,
                    f'{wandb_inst.config.model_name}_ep{episode}.pth')
                torch.save(self.actor)
                artifact.add_file(actor_path_name, name=f'{wandb_inst.config.model_name}_ep{episode}.pth')


    def eval(self):
        print("Begin eval!")
        self.set_eval()

        # initial state of environment
        curr_obs = self.env.reset()  # may need to normalize obs

        # initialize values
        ep_reward = 0

        # keep sampling until done
        done = False
        while not done:
            # select action
            action = self.predict_action(curr_obs)

            # step forward environment
            next_obs, reward, done, info = self.env.step(action)  # may need to normalize obs

            ep_reward += reward
            curr_obs = next_obs

        return ep_reward


    def random_action(self):
        action = np.random.rand(17)
        action = action / action.sum()

        return action


    def predict_action(self, obs):
        a,b = obs
        a = torch.tensor(a, dtype=FLOAT, device=self.device)
        b = torch.tensor(b, dtype=FLOAT, device=self.device)
        action = self.actor(a, b)
        if USE_CUDA:
            action = action.detach().cpu().numpy()
        else:
            action = action.detach().numpy()
        action += self.actor_noise()
        
        action = np.clip(action, 0, 1)
        action = action / action.sum()

        return action


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


    def seed(self,s):
        torch.manual_seed(s)
        if USE_CUDA:
            torch.cuda.manual_seed(s)