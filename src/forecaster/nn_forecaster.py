from src.forecaster.base_forecaster import BaseForecaster
from src.utils.torch_utils import USE_CUDA, FLOAT
from src.utils.data_utils import EPS
from src.models.transformer_shared_model import TransformerSharedForecaster
from src.models.lstm_shared_model import LSTMSharedForecaster
from src.forecaster.data_loader import StockDataset

import os
import wandb
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.cuda import amp
from tqdm import tqdm


class NNForecaster(BaseForecaster):

    def __init__(self, name, agent):
        super().__init__(name, agent.preprocess, agent.env.market)

        self.device = torch.device("cuda:0" if USE_CUDA else "cpu")
        self.agent = agent

        window_length = self.market_train.window_length
        if self.preprocess in ('log_return', 'simple_return'):
            window_length -= 1

        if agent.network_type == 'trans_shared':
            self.model = TransformerSharedForecaster(price_features=4, num_stocks=len(self.market_train.stock_names[1:]), window_length=window_length, d_model=64, num_heads=8, num_layers=3)
        elif agent.network_type == 'lstm_shared':
            self.model = LSTMSharedForecaster(price_features=4, num_stocks=len(self.market_train.stock_names[1:]), window_length=window_length, d_model=64, num_layers=3)

        if USE_CUDA:
            self.model = self.model.cuda()

    def _forecast(self, obs):

        (obs, _) = self.agent.preprocess_data((obs, None))
        obs = torch.tensor(obs, dtype=FLOAT, device=self.device)

        pred = self.model(obs)
        if USE_CUDA:
            pred = pred.detach().cpu().numpy()
        else:
            pred = pred.detach().numpy()

        if self.agent.normalize:
            # have to denormalize prediction
            mean_close = self.agent.mean[:,3]
            std_close = self.agent.std[:,3]
            pred = pred * std_close + mean_close

        return pred

    def forecast_all(self, market_test, render=False):
        # equivalent to fit_model_given_par of varma_forecaster
        self.model.eval()

        # results on train
        self.market_train.reset()

        pred_vec_train, truth_vec_train = [], []

        done = False
        while not done:
            curr_obs, next_obs, done = self.market_train.step()

            pred = self._forecast(curr_obs)
            truth = self._value_from_price(next_obs[1:,-1,3], curr_obs[1:,-1,3], next_obs[1:,:,3])

            pred_vec_train.append(pred)
            truth_vec_train.append(truth)

        pred_vec_train = np.array(pred_vec_train)
        truth_vec_train = np.array(truth_vec_train)
        rmse_train = self._calculate_rmse(pred_vec_train, truth_vec_train)

        # results on test
        market_test.reset()

        pred_vec_test, truth_vec_test = [], []

        done = False
        while not done:
            curr_obs, next_obs, done = market_test.step()

            pred = self._forecast(curr_obs)
            truth = self._value_from_price(next_obs[1:,-1,3], curr_obs[1:,-1,3], next_obs[1:,:,3])

            pred_vec_test.append(pred)
            truth_vec_test.append(truth)

        pred_vec_test = np.array(pred_vec_test)
        truth_vec_test = np.array(truth_vec_test)
        rmse_test = self._calculate_rmse(pred_vec_test, truth_vec_test)

        if render:
            self.plot_all(pred_vec_train, truth_vec_train, pred_vec_test, truth_vec_test)

        return rmse_train, rmse_test

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path))
        
    def load_pretrained(self, model_path):
         # initialize common part of actors and critics to values obtained from training forecaster
        num_price_features = 4
        window_length = self.market_train.window_length
        num_stocks = len(self.market_train.stock_names[1:])
        if self.preprocess == 'log_return':
            window_length -= 1 # because log returns instead of actual prices

        if self.network_type == 'lstm_shared':
            forecaster = LSTMSharedForecaster(num_price_features, num_stocks, window_length, d_model=64, num_layers=3)
            
        forecaster.load_state_dict(torch.load(model_path))
        forecaster_state_dict = forecaster.state_dict()
        
        # keep only base part
        forecaster_state_dict = {k.replace('base', 'common.base'): v for k, v in forecaster_state_dict.items() if k.startswith('base')}
        
        state_dict = self.model.state_dict()
        state_dict.update(forecaster_state_dict)
        self.actor.load_state_dict(state_dict)
        
        # freeze pretrained params
        for name, param in self.model.named_parameters():
            if 'base.' in name:
                param.requires_grad = False

    def train(self, market_test, config, wandb_inst):

        # setup
        optimizer = torch.optim.Adam(self.model.parameters(), config.learning_rate, weight_decay=config.weight_decay)

        train_data = StockDataset(self.agent, self.market_train.start_date, self.market_train.end_date, self.market_train.window_length, self.market_train.stock_names, 'train')
        test_data = StockDataset(self.agent, market_test.start_date, market_test.end_date, market_test.window_length, market_test.stock_names, 'eval')
       
        dataloader_train = DataLoader(train_data, config.batch_size, shuffle=True)
        dataloader_test = DataLoader(test_data, 1, shuffle=False)

        if config.checkpoint_ep > 0:
            self.model.load_state_dict(torch.load(f'{config.save_model_path}/ep{config.checkpoint_ep}_{config.model_name}.pth'))

        # rmse
        loss = lambda pred, truth: torch.sqrt(torch.nn.MSELoss()(pred, truth))

        wandb_inst.watch(self.model, loss, log='all', log_freq=config.batch_size)
        artifact = wandb.Artifact(name='forecaster', type='model', metadata=dict(config))

        # creating directory to store models if it doesn't exist
        if not os.path.isdir(config.save_model_path):
            os.makedirs(config.save_model_path)

        scaler = amp.GradScaler()

        step = 0
        for epoch in range(config.checkpoint_ep + 1, config.num_epochs):
            
            self.model.train()
            tq = tqdm(total=len(dataloader_train) * config.batch_size)
            tq.set_description('epoch %d' % (epoch))

            loss_record = []
            for i, (obs, truth) in enumerate(dataloader_train):
                obs = obs.cuda()
                truth = truth.cuda()

                with amp.autocast():
                    pred = self.model(obs)
                    if self.agent.normalize:
                        # have to denormalize prediction
                        mean_close = torch.tensor(self.agent.mean[:,3], dtype=FLOAT, device=self.device)
                        std_close = torch.tensor(self.agent.std[:,3], dtype=FLOAT, device=self.device)
                        pred = pred * std_close + mean_close
                    loss_value = loss(pred, truth)

                scaler.scale(loss_value).backward()
                scaler.step(optimizer)
                scaler.update()

                tq.update(config.batch_size)
                tq.set_postfix(loss='%.6f' % loss_value)
                step += 1

                wandb_inst.log({"epoch": epoch, "train_loss": loss_value}, step=step)

                loss_record.append(loss_value.item())

            loss_train_mean = np.mean(loss_record)

            wandb_inst.log({"loss_epoch_train": loss_train_mean}, step=step)

            print('loss for train : %f' % loss_train_mean)
            if epoch % config.eval_steps == config.eval_steps - 1:

                model_path_name = os.path.join(config.save_model_path, f'ep{epoch}_{config.model_name}.pth')
                torch.save(self.model.state_dict(), model_path_name)
                artifact.add_file(model_path_name, name=f'ep{epoch}_{config.model_name}.pth')

                loss_test_mean = self._eval(loss, dataloader_test)
                wandb_inst.log({"test_loss": loss_test_mean}, step=step)

        wandb_inst.log_artifact(artifact)

        # plot results
        rmse_train, rmse_test = self.forecast_all(market_test, render=True)
        print(f'Train loss = {rmse_train} - Test loss = {rmse_test}')

        return rmse_train, rmse_test

    def _eval(self, loss, dataloader_test):

        print('start val!')

        loss_test_mean = []
        with torch.no_grad():
            self.model.eval()

            for i, (obs, truth) in enumerate(dataloader_test):
                obs = obs.cuda()
                truth = truth.cuda()

                pred = self.model(obs)
                if self.agent.normalize:
                    # have to denormalize prediction
                    mean_close = torch.tensor(self.agent.mean[:,3], dtype=FLOAT, device=self.device)
                    std_close = torch.tensor(self.agent.std[:,3], dtype=FLOAT, device=self.device)
                    pred = pred * std_close + mean_close
                loss_value = loss(pred, truth.squeeze(0))
                loss_test_mean.append(loss_value.item())

        return np.mean(loss_test_mean)

    def _value_from_price(self, price, past_price, close_prices):

        if self.preprocess == 'log_return':
            value = np.log(price + EPS) - np.log(past_price + EPS)
        elif self.preprocess == 'simple_return':
            value = np.divide(price, past_price) - 1
        elif self.preprocess == 'minmax':
            max_price = np.max(close_prices, axis=1)
            min_price = np.min(close_prices, axis=1)
            value = (price - min_price) - (max_price - min_price + EPS)

        return value