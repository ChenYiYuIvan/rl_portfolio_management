import numpy as np
import torch
from torch.cuda import amp

from src.agents.base_agent import BaseAgent

from src.utils.file_utils import get_checkpoint_folder
from src.utils.torch_utils import USE_CUDA
from src.utils.data_utils import prices_to_logreturns, remove_not_used, rnn_transpose, cnn_transpose

from src.models.replay_buffer import ReplayBuffer

from tqdm import tqdm
import wandb
import os


class BaseACAgent(BaseAgent):

    def __init__(self, name, env, seed, args):
        super().__init__(name, env, seed)
        
        # network type
        assert args.network_type in ('cnn', 'lstm', 'gru')
        self.network_type = args.network_type

        # reward signal
        assert args.reward_type in ('log_return', 'simple_return', 'diff_sharpe_ratio', 'diff_sortino_ratio')
        self.reward_type = args.reward_type

        # hyper-parameters
        self.num_episodes = args.num_episodes
        self.warmup_steps = args.warmup_steps
        self.batch_size = args.batch_size
        self.tau = args.tau # for polyak averaging
        self.gamma = args.gamma # for bellman equation

        # evaluate model every few episodes
        self.eval_steps = args.eval_steps

        # define actors and critics networks and optimizers
        self.define_actors_critics(args)

        # target networks must have same initial weights as originals
        self.copy_params_to_target()

        # define replay buffer
        self.buffer = ReplayBuffer(args.buffer_size)

        # cpu or gpu
        self.device = torch.device("cuda:0" if USE_CUDA else "cpu")
        if USE_CUDA:
            self.cuda()

        # freeze target networks, only update manually
        self.set_networks_grad('target', False)

        # define checkpoint folder
        self.checkpoint_folder = get_checkpoint_folder(self, self.env)

        # initialization of parameters for differential sharpe ratio and differential downside deviation ratio
        self.initialize_differential()


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


    def initialize_differential(self):
        self.eta = 1e-3
        self.A = 0
        self.B = 1
        self.DD = 1


    def compute_value_loss(self, s_batch, a_batch, r_batch, t_batch, s2_batch):
        raise NotImplementedError


    def compute_policy_loss(self, s_batch):
        raise NotImplementedError


    def update(self, scaler, wandb_inst, step):
        raise NotImplementedError


    def train(self, wandb_inst, env_test):
        # env_test: environment used to test agent during training (usually different from training environment)
        print("Begin train!")
        wandb_inst.watch(self.actor)
        artifact = wandb.Artifact(name=self.name, type='model')

        # creating directory to store models if it doesn't exist
        if not os.path.isdir(self.checkpoint_folder):
            os.makedirs(self.checkpoint_folder)

        scaler = amp.GradScaler()

        step = 0
        max_ep_reward_val = -np.inf
        # loop over episodes
        for episode in range(self.num_episodes):

            # logging
            num_steps = self.env.market.tot_steps
            tq = tqdm(total=num_steps)
            tq.set_description('episode %d' % (episode))

            # set models in trainig mode
            self.set_train()

            # initial state of environment
            curr_obs = self.env.reset()
            curr_obs = self.preprocess_data(curr_obs)

            # initialize values
            ep_reward_train = 0
            self.initialize_differential()

            # keep sampling until done
            done = False
            while not done:
                step += 1
                
                # select action
                if self.buffer.size() >= self.batch_size and step >= self.warmup_steps:
                    action = self.predict_action(curr_obs, True)
                else:
                    action = self.random_action()

                # step forward environment
                next_obs, done, info_action = self.env.step(action)
                reward = self.get_reward(info_action)
                next_obs = self.preprocess_data(next_obs)

                # add to buffer
                self.buffer.add(curr_obs, action, reward, done, next_obs)

                # if replay buffer has enough observations
                if self.buffer.size() >= self.batch_size:
                    self.update(scaler, wandb_inst, step)

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
                ep_reward_val, infos_eval, ep_port_value_val = self.eval(env_test, render=False)
                print(f"Eval - Episode final portfolio value: {ep_port_value_val} | Episode total reward: {ep_reward_val}")
                wandb_inst.log({'ep_reward_eval': ep_reward_val,
                                'ep_port_value_eval': ep_port_value_val}, step=step)

                if ep_reward_val > max_ep_reward_val:
                    max_ep_reward_val = ep_reward_val
                    wandb_inst.summary['max_ep_reward_val'] = max_ep_reward_val

                # save trained model
                actor_path_name = os.path.join(self.checkpoint_folder,
                    f'{self.name}_ep{episode}.pth')
                torch.save(self.actor.state_dict(), actor_path_name)
                artifact.add_file(actor_path_name, name=f'{self.name}_ep{episode}.pth')

        wandb_inst.log_artifact(artifact)


    def eval(self, env, exploration=False, render=False):
        self.set_eval()
        return super().eval(env, exploration=exploration, render=render)


    def predict_action(self, obs, exploration=False):
        raise NotImplementedError


    def preprocess_data(self, obs):
        prices, weights = obs
        prices = remove_not_used(prices)
        prices = prices_to_logreturns(prices)
        if self.network_type == 'cnn':
            prices = cnn_transpose(prices)
        elif self.network_type == 'lstm' or self.network_type == 'gru':
            prices = rnn_transpose(prices)

        return (prices, weights)


    def get_reward(self, info):
        if self.reward_type in ('log_return, simple_return'):
            return super().get_reward(info)
        elif self.reward_type == 'diff_sharpe_ratio':
            deltaA = info['simple_return'] - self.A
            deltaB = info['simple_return']**2 - self.B
            reward = (self.B * deltaA - self.A * deltaB / 2) / (self.B - self.A**2)**(3/2)

            self.A = self.A + self.eta * deltaA
            self.B = self.B + self.eta * deltaB

        elif self.reward_type == 'diff_sortino_ratio':
            # also called differential downside deviation ratio
            if info['simple_return'] > 0:
                num = info['simple_return'] - self.A / 2
                denom = self.DD
            else:
                num = self.DD**2 * (info['simple_return'] - self.A / 2) - self.A * info['simple_return']**2 / 2
                denom = self.DD**3
            reward = num / denom

            self.A = self.A + self.eta * (info['simple_return'] - self.A)
            self.DD = np.sqrt(self.DD**2 + self.eta * (np.minimum(info['simple_return'], 0)**2 - self.DD**2))

        return reward


    def load_actor_model(self, path):
        self.actor.load_state_dict(torch.load(path))


    def set_train(self):
        raise NotImplementedError


    def set_eval(self):
        raise NotImplementedError


    def cuda(self):
        raise NotImplementedError