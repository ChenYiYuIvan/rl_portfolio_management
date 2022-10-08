from src.forecaster.base_forecaster import BaseForecaster
from src.utils.torch_utils import USE_CUDA, FLOAT
from src.models.transformer_shared_model import TransformerSharedForecaster
from src.forecaster.data_loader import StockDataset

import os
import wandb
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.cuda import amp
from tqdm import tqdm


class NNForecaster(BaseForecaster):

    def __init__(self, name, agent, market_train):
        super().__init__(name, agent.preprocess, market_train)

        self.device = torch.device("cuda:0" if USE_CUDA else "cpu")
        self.agent = agent

        if agent.network_type == 'trans_shared':
            window_length = market_train.window_length
            if self.preprocess in ('log_return', 'simple_return'):
                window_length -= 1
            self.model = TransformerSharedForecaster(price_features=4, num_stocks=len(market_train.stock_names[1:]), window_length=window_length, d_model=64, num_heads=8, num_layers=3)

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

        return pred

    def forecast_all(self, market):
        self.model.eval()
        super().forecast_all(market)

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path))

    def train(self, market_test, config, wandb_inst):

        # setup
        optimizer = torch.optim.Adam(self.model.parameters(), config.learning_rate, weight_decay=config.weight_decay)

        train_data = StockDataset(self.agent, self.market_train.start_date, self.market_train.end_date, self.market_train.window_length, self.market_train.stock_names, 'train')
        test_data = StockDataset(self.agent, market_test.start_date, market_test.end_date, market_test.window_length, market_test.stock_names, 'eval')
       
        dataloader_train = DataLoader(train_data, config.batch_size, shuffle=True)
        dataloader_test = DataLoader(test_data, 1, shuffle=False)

        if config.checkpoint_ep > 0:
            self.model.load_state_dict(torch.load(f'{config.save_model_path}/ep{config.checkpoint_ep}_trans_forecaster.pth'))

        loss = torch.nn.MSELoss()

        wandb_inst.watch(self.model, loss, log_freq=config.batch_size)
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

    def _eval(self, loss, dataloader_test):

        print('start val!')

        loss_test_mean = []
        with torch.no_grad():
            self.model.eval()

            for i, (obs, truth) in enumerate(dataloader_test):
                obs = obs.cuda()
                truth = truth.cuda()

                pred = self.model(obs)
                loss_value = loss(pred, truth.squeeze(0))
                loss_test_mean.append(loss_value.item())

        return np.mean(loss_test_mean)