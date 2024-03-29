from copy import deepcopy
import numpy as np
import torch
from torch.cuda import amp

from src.agents.base_agent import BaseAgent

from src.utils.file_utils import get_checkpoint_folder
from src.utils.torch_utils import USE_CUDA
from src.utils.data_utils import cnn_rnn_transpose, prices_to_logreturns, prices_to_norm, prices_to_simplereturns, remove_not_used, rnn_transpose, cnn_transpose, prices_to_range

from src.models.replay_buffer import ReplayBuffer

from src.utils.portfolio_utils import get_opt_portfolio

from tqdm import tqdm
import wandb
import os
import copy


class BaseACAgent(BaseAgent):

    def __init__(self, name, env, seed, args):
        super().__init__(name, env, seed, args.reward_type)

        # network type
        self.network_type = args.network_type

        self.preprocess = args.preprocess
        self.normalize = args.normalize
        if self.normalize:
            data = env.market.data
            if self.preprocess == 'log_return':
                data = prices_to_logreturns(data)
            elif self.preprocess == 'divide_close':
                data = prices_to_norm(data)
            elif self.preprocess == 'simple_return':
                data = prices_to_simplereturns(data)
            elif self.preprocess == 'minmax':
                data = prices_to_range(data)

            # mean and std of stocks (no cash)
            self.mean = data[1:].mean(axis=1)
            self.std = data[1:].std(axis=1)

        # hyper-parameters
        self.num_episodes = args.num_episodes
        self.warmup_steps = args.warmup_steps
        self.batch_size = args.batch_size
        self.tau = args.tau # for polyak averaging
        self.gamma = args.gamma # for bellman equation
        self.reward_scale = args.reward_scale

        # exploration during training
        self.exploration_training = args.exploration_training

        # evaluate model every few episodes
        self.eval_steps = args.eval_steps

        # define actors and critics networks and optimizers
        self.define_actors_critics(args)

        # target networks must have same initial weights as originals
        self.copy_params_to_target()

        # define replay buffer
        self.buffer = ReplayBuffer(args.buffer_size)

        # imitation learning
        self.imitation_learning = args.imitation_learning
        if self.imitation_learning == 'active':
            self.env_copy = deepcopy(self.env)

        # cpu or gpu
        self.device = torch.device("cuda:0" if USE_CUDA else "cpu")
        if USE_CUDA:
            self.cuda()

        # freeze target networks, only update manually
        self.set_networks_grad('target', False)

        # define checkpoint folder
        self.checkpoint_folder = get_checkpoint_folder(args.name, env.name)


    def define_actors_critics(self, args):
        raise NotImplementedError


    def copy_params_to_target(self):
        raise NotImplementedError


    def update_target_params(self):
        raise NotImplementedError

    
    def load_pretrained(self, path, config):
        raise NotImplementedError


    def set_networks_grad(self, networks, requires_grad, pretrained=False):
        # networks = {'actor', 'critic', 'target'}
        # requires_grad = {True, False}
        raise NotImplementedError


    def compute_value_loss(self, s_batch, a_batch, r_batch, t_batch, s2_batch):
        raise NotImplementedError


    def compute_policy_loss(self, s_batch):
        raise NotImplementedError


    def update(self, scaler, wandb_inst, step, pretrained_path=None):
        raise NotImplementedError


    def train(self, wandb_inst, env_test, pretrained_path=None):
        # env_test: environment used to test agent during training (usually different from training environment)
        print("Begin train!")

        if pretrained_path is not None:
            assert self.imitation_learning == 'passive', 'Provided path for pretrained model but not using passive imitation learning'
            self.load_pretrained(pretrained_path, wandb_inst)
        elif pretrained_path is None:
            assert self.imitation_learning != 'passive', 'Using passive imitation learning but didn\'t provide path of pretrained model'

        wandb_inst.watch((self.actor, self.critic), log='all', log_freq=100)
        artifact = wandb.Artifact(name=self.name, type='model')

        # creating directory to store models if it doesn't exist
        if not os.path.isdir(self.checkpoint_folder):
            os.makedirs(self.checkpoint_folder)

        scaler = amp.GradScaler()

        step = 0
        max_ep_reward_val_train = -np.inf
        max_ep_reward_val = -np.inf
        # loop over episodes
        for episode in range(self.num_episodes):

            # logging
            num_steps = self.env.max_steps
            tq = tqdm(total=num_steps)
            tq.set_description('episode %d' % (episode))

            # set models in trainig mode
            self.set_train()

            # initial state of environment
            self.reset()
            curr_obs_original = self.env.reset()
            curr_obs = self.preprocess_data(curr_obs_original)
            if self.imitation_learning == 'active':
                self.env_copy.reset()

            # initialize values
            ep_reward_train = 0

            # keep sampling until done
            old_info = None
            old_info_copy = None
            done = False
            while not done:
                step += 1
                
                # select action
                if self.buffer.size() >= self.batch_size and step >= self.warmup_steps:
                    action = self.predict_action(curr_obs, self.exploration_training)
                else:
                    action = self.random_action()

                # step forward environment
                next_obs_original, done, info_action = self.env.step(action)
                reward = self.get_reward(info_action, old_info)
                next_obs = self.preprocess_data(next_obs_original)
                old_info = info_action

                # add to buffer
                self.buffer.add(curr_obs, action, reward, done, next_obs)

                # active imitation learning
                if self.imitation_learning == 'active' and self.buffer.size() >= self.batch_size and step >= self.warmup_steps:
                    # MPT action
                    action_copy, _ = get_opt_portfolio(curr_obs_original, 'sharpe_ratio', self.env_copy.trading_cost)
                    
                    next_obs_copy_original, done_copy, info_action_copy = self.env_copy.step(action)
                    reward_copy = self.get_reward(info_action_copy, old_info_copy)
                    next_obs_copy = self.preprocess_data(next_obs_copy_original)
                    old_info_copy = info_action_copy

                    self.buffer.add(curr_obs, action_copy, reward_copy, done_copy, next_obs_copy)

                    # copy values from original env
                    curr_obs_original = next_obs_original
                    self.env_copy.copy_env(self.env)

                    _, val = curr_obs_original
                    assert np.array_equal(val, self.env_copy.weights), 'Error in copying envs'

                # if replay buffer has enough observations
                if self.buffer.size() >= self.batch_size and step >= self.warmup_steps:
                    self.update(scaler, wandb_inst, step, pretrained_path)

                ep_reward_train += reward
                curr_obs = next_obs

                tq.update(1)
                tq.set_postfix(ep_reward='%.6f' % ep_reward_train)

                # logging
                stocks_names = ['CASH', *self.env.stock_names]
                stocks_names = [f'step_{name}' for name in stocks_names]
                data_to_log = dict(zip(stocks_names, action))
                data_to_log.update({
                    'episode': episode,
                    'step_reward_train': reward,
                    'step_port_value': info_action['port_value_old'],
                    'step_trans_cost': info_action['cost'],
                    'step_log_return': info_action['log_return'],
                    'step_simple_return': info_action['simple_return'],
                    'step_sharpe_ratio': info_action['sharpe_ratio'],
                    'step_sortino_ratio': info_action['sortino_ratio'],
                    'step_max_drawdown': info_action['max_drawdown'],
                    'step_var_95': info_action['var_95'],
                    'step_cvar_95': info_action['cvar_95'],
                })

                wandb_inst.log(data_to_log, step=step)

            tq.close()
            print(f"Train - Episode final portfolio value: {info_action['port_value_old']} | Episode total reward: {ep_reward_train}")
            wandb_inst.log({'ep_reward_train': ep_reward_train,
                            'ep_port_value_train': info_action['port_value_old']}, step=step)

            # evaluate model every few episodes
            if episode % self.eval_steps == self.eval_steps - 1 or episode == 0:
                
                # evaluate on training data
                ep_reward_val_train, infos_eval_train, ep_port_value_val_train = self.eval(self.env)
                print(f"Eval on train - Episode final portfolio value: {ep_port_value_val_train} | Episode total reward: {ep_reward_val_train}")
                wandb_inst.log({'ep_reward_eval_train': ep_reward_val_train,
                                'ep_port_value_eval_train': ep_port_value_val_train}, step=step)

                if ep_reward_val_train > max_ep_reward_val_train:
                    max_ep_reward_val_train = ep_reward_val_train
                    wandb_inst.summary['max_ep_reward_val_train'] = max_ep_reward_val_train

                # evaluate on test data
                ep_reward_val, infos_eval, ep_port_value_val = self.eval(env_test)
                print(f"Eval - Episode final portfolio value: {ep_port_value_val} | Episode total reward: {ep_reward_val}")
                wandb_inst.log({'ep_reward_eval': ep_reward_val,
                                'ep_port_value_eval': ep_port_value_val}, step=step)

                if ep_reward_val > max_ep_reward_val:
                    max_ep_reward_val = ep_reward_val
                    wandb_inst.summary['max_ep_reward_val'] = max_ep_reward_val

                # save trained models
                path_name = os.path.join(self.checkpoint_folder, f'ep{episode}_')

                torch.save(self.actor.state_dict(), path_name + 'actor.pth')
                artifact.add_file(path_name + 'actor.pth', name=f'ep{episode}_actor.pth')

                torch.save(self.actor_optim.state_dict(), path_name + 'actor_optim.pth')
                artifact.add_file(path_name + 'actor_optim.pth', name=f'ep{episode}_actor_optim.pth')

                torch.save(self.critic.state_dict(), path_name + 'critic.pth')
                artifact.add_file(path_name + 'critic.pth', name=f'ep{episode}_critic.pth')

                torch.save(self.critic_optim.state_dict(), path_name + 'critic_optim.pth')
                artifact.add_file(path_name + 'critic_optim.pth', name=f'ep{episode}_critic_optim.pth')

        wandb_inst.log_artifact(artifact)


    def eval(self, env, exploration=False, render=False):
        self.set_eval()
        return super().eval(env, exploration=exploration, render=render)


    def predict_action(self, obs, exploration=False):
        raise NotImplementedError


    def preprocess_data(self, obs):
        prices, weights = obs
        if self.preprocess == 'log_return':
            prices = prices_to_logreturns(prices)
        elif self.preprocess == 'divide_close':
            prices = prices_to_norm(prices)
        elif self.preprocess == 'simple_return':
            prices = prices_to_simplereturns(prices)
        elif self.preprocess == 'minmax':
            prices = prices_to_range(prices)

        if self.normalize:
            # don't normalize cash data
            prices[1:] = (prices[1:] - self.mean[:,None,:]) / self.std[:,None,:]

        if self.network_type == 'trans':
            prices = remove_not_used(prices, volume=False)
        elif self.network_type == 'trans_shared' or self.network_type == 'lstm_shared':
            prices = remove_not_used(prices, cash=False, volume=False)
        else:
            prices = remove_not_used(prices)

        #if self.network_type == 'cnn' or self.network_type == 'msm':
        #    prices = cnn_transpose(prices)
        #elif self.network_type == 'trans' or self.network_type == 'trans_shared' or self.network_type == 'lstm' or self.network_type == 'gru':
        #    prices = rnn_transpose(prices)
        #elif self.network_type == 'cnn_gru':
        #    prices = cnn_rnn_transpose(prices)

        return (prices, weights)


    def load_models(self, episode):
        path = os.path.join(self.checkpoint_folder, f'ep{episode}_')

        self.actor.load_state_dict(torch.load(path + 'actor.pth'))
        self.critic.load_state_dict(torch.load(path + 'critic.pth'))

        self.actor_optim.load_state_dict(torch.load(path + 'actor_optim.pth'))
        self.critic_optim.load_state_dict(torch.load(path + 'critic_optim.pth'))

        if hasattr(self, 'actor_target'):
            self.actor_target = copy.deepcopy(self.actor)
        if hasattr(self, 'critic_target'):
            self.critic_target = copy.deepcopy(self.critic)


    def load_actor_from_path(self, path):
        self.actor.load_state_dict(torch.load(path))


    def set_train(self):
        raise NotImplementedError


    def set_eval(self):
        raise NotImplementedError


    def cuda(self):
        raise NotImplementedError