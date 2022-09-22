import numpy as np
from src.models.cnn_models import DeterministicCNNActor
from src.models.msm_models import DeterministicMSMActor
from torch.optim import Adam
import torch.nn as nn
from tqdm import tqdm
import wandb
from torch.cuda import amp
import torch
from src.utils.torch_utils import FLOAT, USE_CUDA
import os
from src.utils.data_utils import cnn_transpose, remove_not_used, prices_to_logreturns
from src.agents.mpt_agent import MPTAgent
import matplotlib.pyplot as plt
import pandas as pd
from src.utils.file_utils import read_yaml_config
from src.environments.portfolio import Portfolio


class PreTrainer:

    def __init__(self, args, env_train=None, env_test=None):

        window_length = args.window_length
        if args.log_return:
            window_length -= 1

        # path for synthetic data
        data_path = f'./data/synthetic_data/data{args.num_stocks}_{args.window_length}'

        self.network_type = args.network_type

        if self.network_type == 'cnn':
            self.actor = DeterministicCNNActor(args.num_price_features, args.num_stocks, window_length)
        elif self.network_type == 'msm':
            self.actor = DeterministicMSMActor(args.num_price_features, args.num_stocks, window_length)

        if env_train is None:
            self.X_obs_train = np.load(f'{data_path}/X_obs_train.npy')
            self.X_weight_train = np.load(f'{data_path}/X_weight_train.npy')
            self.y_train = np.load(f'{data_path}/y_train.npy')

        if env_test is None:
            self.X_obs_test = np.load(f'{data_path}/X_obs_test.npy')
            self.X_weight_test = np.load(f'{data_path}/X_weight_test.npy')
            self.y_test = np.load(f'{data_path}/y_test.npy')

        self.eval_steps = args.eval_steps

        self.num_stocks = args.num_stocks

        if env_train is not None:
            self.agent_train = MPTAgent('mpt_train', env_train, args.seed, 'sharpe_ratio')

        if env_test is not None:
            self.agent_test = MPTAgent('mpt_test', env_test, args.seed, 'sharpe_ratio')

        # hyper-parameters
        self.lr = 1e-3
        self.num_epochs = 100

        self.noise = args.noise

        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.loss = nn.MSELoss()

        # cpu or gpu
        self.device = torch.device("cuda:0" if USE_CUDA else "cpu")
        if USE_CUDA:
            self.actor.cuda()

        self.seed(args.seed)

        # folder
        self.folder = './checkpoints_pretrained/'
        if env_train is None:
            self.folder += f'{self.network_type}_synthetic_{self.num_stocks}_{window_length}/'
        else:
            self.folder += f'{self.network_type}_real_{self.num_stocks}_{window_length}'
            if self.noise:
                self.folder += '_noise'
        if not os.path.isdir(self.folder):
            os.makedirs(self.folder)

    def train(self, wandb_inst):

        print("Begin train!")
        #wandb_inst.watch((self.actor), log='none', log_freq=100)
        artifact = wandb.Artifact(name='pretrained', type='model')

        scaler = amp.GradScaler()
        step = 0

        best_loss_train = np.inf
        best_loss_eval = np.inf
        for epoch in range(self.num_epochs):
            self.actor.train()

            if not hasattr(self, 'agent_train'):
                actor_name = f'synthetic_epoch{epoch}.pth'
                num_obs = self.X_obs_train.shape[0]
            else:
                actor_name = f'real_epoch{epoch}.pth'
                num_obs = self.agent_train.env.market.tot_steps

            tq = tqdm(total=num_obs)
            tq.set_description('epoch %d' % (epoch))
        
            loss_train_vec = []

            if not hasattr(self, 'agent_train'): # use simulated data
            
                for i in range(num_obs):
                    step += 1

                    obs, weight, y_true = self.get_data_point(self.X_obs_train, self.X_weight_train, self.y_train, i)
                    y_pred = self.actor(obs, weight)

                    with amp.autocast():
                        loss_train = self.loss(y_pred, y_true)

                    scaler.scale(loss_train).backward()
                    scaler.step(self.actor_optim)
                    scaler.update()

                    loss_train_vec.append(loss_train.item())
                    wandb_inst.log({'loss_train': loss_train, 'epoch': epoch}, step=step)

                    tq.update(1)
                    tq.set_postfix(loss='%.6f' % loss_train)

            else: # use training environment
                
                self.agent_train.reset()
                obs, weight = self.agent_train.env.reset()
                #weight = self.agent_train.random_action()

                done = False
                while not done:
                    step += 1

                    y_true = self.agent_train.predict_action((obs, weight), verbose=False)

                    # preprocess data for actor network
                    obs, weight, y_true = self.process_real_data(obs, weight, y_true)

                    y_pred = self.actor(obs, weight)

                    with amp.autocast():
                        loss_train = self.loss(y_pred, y_true)

                    scaler.scale(loss_train).backward()
                    scaler.step(self.actor_optim)
                    scaler.update()

                    loss_train_vec.append(loss_train.item())
                    wandb_inst.log({'loss_train': loss_train, 'epoch': epoch}, step=step)

                    tq.update(1)
                    tq.set_postfix(loss='%.6f' % loss_train)

                    # step forward environment
                    if USE_CUDA:
                        y_pred = y_pred.detach().cpu().numpy()
                        y_true = y_true.detach().cpu().numpy()
                    else:
                        y_pred = y_pred.detach().numpy()
                        y_true = y_true.detach().numpy()

                    #action = (y_pred + y_true) / 2
                    action = y_true
                    if self.noise:
                        action = self.add_action_noise(action)
                    (obs, weight), done, _ = self.agent_train.env.step(action)
                    #weight = self.agent_train.random_action()

            tq.close()

            loss_train_mean = np.mean(loss_train_vec)
            print(f'Train - Mean loss = {loss_train_mean}')
            wandb_inst.log({"loss_epoch_train": loss_train_mean}, step=step)
            if loss_train_mean < best_loss_train:
                wandb_inst.summary['best_loss_train'] = loss_train_mean
                best_loss_train = loss_train_mean

            # save model
            actor_path_name = os.path.join(self.folder, actor_name)
            torch.save(self.actor.state_dict(), actor_path_name)
            artifact.add_file(actor_path_name, name=actor_name)

            if epoch % self.eval_steps == self.eval_steps - 1 or epoch == 0:
                loss_eval_mean = self.eval()
                print(f'Eval - Mean loss = {loss_eval_mean}')
                wandb_inst.log({"loss_epoch_eval": loss_eval_mean}, step=step)
                if loss_eval_mean < best_loss_eval:
                    wandb_inst.summary['best_loss_eval'] = loss_eval_mean
                    best_loss_eval = loss_eval_mean

        wandb_inst.log_artifact(artifact)
            
    def eval(self, render=False, num_cols=4):
        self.actor.eval()

        loss_eval_vec = []
        y_true_vec = []
        y_pred_vec = []
        if not hasattr(self, 'agent_test'): # use synthetic data
            num_obs = self.X_obs_test.shape[0]

            for i in range(num_obs):
                obs, weight, y_true = self.get_data_point(self.X_obs_test, self.X_weight_test, self.y_test, i)
                y_pred = self.actor(obs, weight)

                loss_eval = self.loss(y_pred, y_true)
                loss_eval_vec.append(loss_eval.item())

                if USE_CUDA:
                    y_true = y_true.detach().cpu().numpy()
                    y_pred = y_pred.detach().cpu().numpy()
                else:
                    y_true = y_true.detach().numpy()
                    y_pred = y_pred.detach().numpy()

                y_true_vec.append(y_true)
                y_pred_vec.append(y_pred)

        else:

            self.agent_test.reset()
            obs, weight = self.agent_test.env.reset(test_mode=True)

            done = False
            while not done:

                y_true = self.agent_test.predict_action((obs, weight), verbose=False)

                # preprocess data for actor network
                obs, weight, y_true = self.process_real_data(obs, weight, y_true)

                y_pred = self.actor(obs, weight)

                loss_eval = self.loss(y_pred, y_true)
                loss_eval_vec.append(loss_eval.item())

                if USE_CUDA:
                    y_true = y_true.detach().cpu().numpy()
                    y_pred = y_pred.detach().cpu().numpy()
                else:
                    y_true = y_true.detach().numpy()
                    y_pred = y_pred.detach().numpy()

                y_true_vec.append(y_true)
                y_pred_vec.append(y_pred)

                # step forward environment
                action = y_true
                (obs, weight), done, _ = self.agent_test.env.step(action)

        loss_eval_mean = np.mean(loss_eval_vec)

        if render:
            num_assets = self.num_stocks + 1 # cash
            num_rows = int(np.ceil(num_assets / num_cols))

            fig, ax = plt.subplots(num_rows, num_cols, squeeze=False)

            for asset_id in range(num_assets):
                row = int(asset_id / num_cols)  # current row in figure
                col = asset_id % num_cols  # current col in figure

                exact = np.array([el[asset_id] for el in y_true_vec])
                pred = np.array([el[asset_id] for el in y_pred_vec])

                df = pd.DataFrame({'exact': exact, 'pred': pred})
                df.plot(ax=ax[row,col])

            plt.show()

        return loss_eval_mean

    def get_data_point(self, rets, weights_old, weights_new, step):

        rets = rets[step,:,:,:]
        weights_old = weights_old[step,:]
        weights_new = weights_new[step,:]

        rets = remove_not_used(rets)
        if self.network_type == 'cnn':
            rets = cnn_transpose(rets)

        rets = torch.tensor(rets, dtype=FLOAT, device=self.device)
        weights_old = torch.tensor(weights_old, dtype=FLOAT, device=self.device)
        weights_new = torch.tensor(weights_new, dtype=FLOAT, device=self.device)

        return rets, weights_old, weights_new

    def process_real_data(self, rets, weights_old, weights_new):
        rets = prices_to_logreturns(rets)
        rets = remove_not_used(rets)
        if self.network_type == 'cnn':
            rets = cnn_transpose(rets)

        rets = torch.tensor(rets, dtype=FLOAT, device=self.device)
        weights_old = torch.tensor(weights_old, dtype=FLOAT, device=self.device)
        weights_new = torch.tensor(weights_new, dtype=FLOAT, device=self.device)

        return rets, weights_old, weights_new

    def add_action_noise(self, action, range=0.1):
        # apply uniform noise to action
        noise = np.random.uniform(-range/2, range/2, action.shape)
        action += noise
        action = np.clip(action, 0, 1)
        action /= action.sum()

        return action

    def load_actor_model(self, path):
        self.actor.load_state_dict(torch.load(path))

    def seed(self,s):
        np.random.seed(s)
        torch.manual_seed(s)
        if USE_CUDA:
            torch.cuda.manual_seed(s)


if __name__ == '__main__':

    args = {
        'seed': 42,
        'eval_steps': 5,
        'network_type': 'msm',
        'num_price_features': 3,
        'num_stocks': 7,
        'window_length': 50,
        'log_return': True,
        'lr': 1e-3,
        'noise': False,
    }
    
    wandb.login()
    with wandb.init(project="pretraining", entity="mldlproj1gr2", config=args, mode="online") as run:
        config = wandb.config

        env_config_train = read_yaml_config('default/env_small_train')
        env_config_test = read_yaml_config('default/env_small_test')

        env_train = Portfolio(env_config_train)
        env_test = Portfolio(env_config_test)

        pretrainer = PreTrainer(config, env_train, env_test)
        pretrainer.train(run)

        #pretrainer.load_actor_model('./checkpoints_pretrained/cnn_real_7_49_noise/real_epoch69.pth')
        #pretrainer.eval(True)