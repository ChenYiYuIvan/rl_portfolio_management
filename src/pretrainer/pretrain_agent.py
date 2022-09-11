import numpy as np
from src.models.cnn_models import DeterministicCNNActor, CNNCritic
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
from src.environments.portfolio_end import PortfolioEnd


class PreTrainer:

    def __init__(self, args, env_train=None, env_test=None):

        window_length = args.window_length
        if args.log_return:
            window_length -= 1

        data_path = f'./data/synthetic_data/data{args.num_stocks}_{args.window_length}'

        self.network_type = args.network_type

        if self.network_type == 'cnn':
            self.actor = DeterministicCNNActor(args.num_price_features, args.num_stocks, window_length)
            self.critic = CNNCritic(args.num_price_features, args.num_stocks, window_length)
        
        self.X_obs_train = np.load(f'{data_path}/X_obs_train.npy')
        self.X_weight_train = np.load(f'{data_path}/X_weight_train.npy')
        self.y_train = np.load(f'{data_path}/y_train.npy')

        self.X_obs_test = np.load(f'{data_path}/X_obs_test.npy')
        self.X_weight_test = np.load(f'{data_path}/X_weight_test.npy')
        self.y_test = np.load(f'{data_path}/y_test.npy')

        self.num_stocks = args.num_stocks

        if env_train is not None:
            self.agent_train = MPTAgent('mpt_train', env_train, args.seed, 'diff_sharpe_ratio', 'sharpe_ratio')

        if env_test is not None:
            self.agent_test = MPTAgent('mpt_test', env_test, args.seed, 'diff_sharpe_ratio', 'sharpe_ratio')

        # hyper-parameters
        self.lr = 1e-3
        self.num_epochs = 100

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
            self.folder += f'{self.network_type}_synthetic/'
        else:
            self.folder += f'{self.network_type}_real/'
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
                        y_true = y_true.detach().cpu().numpy()
                    else:
                        y_true = y_true.detach().numpy()
                    (obs, weight), done, _ = self.agent_train.env.step(y_true)

            tq.close()

            loss_train_mean = np.mean(loss_train_vec)
            print(f'Train - Mean loss = {loss_train_mean}')
            wandb_inst.log({"loss_epoch_train": loss_train_mean}, step=step)
            if loss_train_mean < best_loss_train:
                wandb_inst.summary['best_loss_train'] = loss_train_mean
                best_loss_train = loss_train_mean

            # save model
            torch.save(self.actor.state_dict(), self.folder + actor_name)
            artifact.add_file(self.folder + actor_name, name=actor_name)

            loss_eval_mean = self.eval()
            print(f'Eval - Mean loss = {loss_eval_mean}')
            wandb_inst.log({"loss_epoch_eval": loss_eval_mean}, step=step)
            if loss_eval_mean < best_loss_eval:
                wandb_inst.summary['best_loss_eval'] = loss_eval_mean
                best_loss_eval = loss_eval_mean

        wandb_inst.log_artifact(artifact)
            
    def eval(self, render=False, num_cols=5):
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
            obs, weight = self.agent_test.env.reset()

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
                (obs, weight), done, _ = self.agent_test.env.step(y_true)

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
        'network_type': 'cnn',
        'num_price_features': 3,
        'num_stocks': 16,
        'window_length': 50,
        'log_return': True,
        'lr': 1e-3,
    }
    
    wandb.login()
    with wandb.init(project="pretraining", entity="mldlproj1gr2", config=args, mode="disabled") as run:
        config = wandb.config

        env_config_train = read_yaml_config('env_default_train')
        env_config_eval = read_yaml_config('env_default_test')

        env_train = PortfolioEnd(env_config_train)
        env_test = PortfolioEnd(env_config_train)

        pretrainer = PreTrainer(config, env_train, env_test)
        pretrainer.train(run)

        #pretrainer.load_actor_model('./checkpoints_pretrained/cnn_synthetic/synthetic_epoch99.pth')
        #pretrainer.load_actor_model('./src/pretrainer/models/real_epoch24.pth')
        #pretrainer.eval(True)