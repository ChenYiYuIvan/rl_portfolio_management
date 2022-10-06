import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.cuda import amp
from tqdm import tqdm
import wandb

from src.forecaster.data_loader import StockDataset
from src.models.transformer_model import TransformerForecaster
from src.models.transformer_shared_model import TransformerSharedForecaster
from src.utils.file_utils import read_yaml_config
from src.agents.sac_agent import SACAgent
from src.environments.portfolio import Portfolio
from src.utils.torch_utils import FLOAT, USE_CUDA
from src.utils.data_utils import EPS


def train(model, optimizer, loss, dataloader_train, dataloader_test, config, wandb_inst):
    wandb_inst.watch(model, loss, log_freq=config.batch_size)
    artifact = wandb.Artifact(name='forecaster', type='model', metadata=dict(config))

    # creating directory to store models if it doesn't exist
    if not os.path.isdir(config.save_model_path):
        os.makedirs(config.save_model_path)

    scaler = amp.GradScaler()

    step = 0
    for epoch in range(config.checkpoint_ep + 1, config.num_epochs):
        
        model.train()
        tq = tqdm(total=len(dataloader_train) * config.batch_size)
        tq.set_description('epoch %d' % (epoch))

        loss_record = []
        for i, (obs, truth) in enumerate(dataloader_train):
            obs = obs.cuda()
            truth = truth.cuda()

            with amp.autocast():
                pred = model(obs)
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
            torch.save(model.state_dict(), model_path_name)
            artifact.add_file(model_path_name, name=f'ep{epoch}_{config.model_name}.pth')

            loss_test_mean = eval(model, loss, dataloader_test)
            wandb_inst.log({"test_loss": loss_test_mean}, step=step)

    wandb_inst.log_artifact(artifact)

def eval(model, loss, dataloader_test):

    print('start val!')

    loss_test_mean = []
    with torch.no_grad():
        model.eval()

        for i, (obs, truth) in enumerate(dataloader_test):
            obs = obs.cuda()
            truth = truth.cuda()

            pred = model(obs)
            loss_value = loss(pred, truth.squeeze(0))
            loss_test_mean.append(loss_value.item())

    return np.mean(loss_test_mean)

def plot_result(model, market, agent, value):

    device = torch.device("cuda:0" if USE_CUDA else "cpu")
    model.eval()

    curr_obs = market.reset()
    pred_vec = [curr_obs[1:,-1,3]]
    truth_vec = [curr_obs[1:,-1,3]]

    done = False
    while not done:
        curr_obs, next_obs, done = market.step()
        
        (curr_obs_p, _) = agent.preprocess_data((curr_obs, None))
        (next_obs_p, _) = agent.preprocess_data((next_obs, None))

        obs = torch.tensor(curr_obs_p, dtype=FLOAT, device=device)
        truth = next_obs_p[-1,:,2]

        pred = model(obs)
        if USE_CUDA:
            pred = pred.detach().cpu().numpy()
        else:
            pred = pred.detach().numpy()

        #print(pred, truth)

        if agent.preprocess == 'log_return':
            pred_price = (truth_vec[-1] + EPS) * np.exp(pred) - EPS
            true_price = (truth_vec[-1] + EPS) * np.exp(truth) - EPS
        elif agent.preprocess == 'simple_return':
            pred_price = truth_vec[-1] * (pred + 1)
            true_price = truth_vec[-1] * (truth + 1)
        elif agent.preprocess == 'minmax':
            close_values = next_obs[1:,:,3]
            max_val = np.max(close_values, axis=1)
            min_val = np.min(close_values, axis=1)

            pred_price = pred * (max_val - min_val + EPS) + min_val
            true_price = truth * (max_val - min_val + EPS) + min_val

        #assert true_price == next_obs[1:,-1,3]

        if value:
            pred_vec.append(pred_price)
            truth_vec.append(true_price)
        else:
            pred_vec.append(pred)
            truth_vec.append(truth)

    fig, arr = plt.subplots(len(market.stock_names[1:]), 1, squeeze=False)
    for i, stock in enumerate(market.stock_names[1:]):
        predictions = [el[i] for el in pred_vec]
        truths = [el[i] for el in truth_vec]

        arr[i,0].plot(truths[1:])
        arr[i,0].plot(predictions[1:])

        arr[i,0].title.set_text(stock)
        #print(f'{stock} - pred: {np.mean(predictions)}, truth: {np.mean(truths)}')

    plt.legend(['truth', 'pred'])
    plt.show()


def main():

    config = {
        'seed': 42,
        'env_train': 'experiments/env_train_1',
        'env_test': 'experiments/env_test_1',
        'agent': 'experiments/sac_8',
        'model': 'transformer_shared', # transformed / transformed_shared
        'batch_size': 64,
        'num_epochs': 10000,
        'learning_rate': 1e-4,
        'eval_steps': 10,
        'num_price_features': 4,
        'window_length': 49,
        'num_stocks': 1,
        'd_model': 64,
        'num_heads': 8,
        'num_layers': 3,
        'save_model_path': './checkpoints_forecaster/trans_shared_log_return',
        'model_name': 'trans_forecaster',
        'checkpoint_ep': 0,
    }

    wandb.login()
    with wandb.init(project="forecasting", entity="mldlproj1gr2", config=config, mode="online") as run:
        config = wandb.config

        seed = config.seed

        env_config_train = read_yaml_config(config.env_train)
        env_train = Portfolio(env_config_train)

        env_config_test = read_yaml_config(config.env_test)
        env_test = Portfolio(env_config_test)

        agent_config = read_yaml_config(config.agent)
        agent = SACAgent('sac', env_train, seed, agent_config)

        if config.model == 'transformer':
            model = TransformerForecaster(config.num_price_features, config.num_stocks, config.window_length, config.d_model, config.num_heads, config.num_layers)
        elif config.model == 'transformer_shared':
            model = TransformerSharedForecaster(config.num_price_features, config.num_stocks, config.window_length, config.d_model, config.num_heads, config.num_layers)
        model = model.cuda()

        optimizer = torch.optim.Adam(model.parameters(), config.learning_rate)

        train_data = StockDataset(agent, env_train.start_date, env_train.end_date, env_train.window_length, env_train.stock_names, 'train')
        test_data = StockDataset(agent, env_test.start_date, env_test.end_date, env_test.window_length, env_test.stock_names, 'eval')
       
        dataloader_train = DataLoader(train_data, config.batch_size, shuffle=True)
        dataloader_test = DataLoader(test_data, 1, shuffle=False)

        if config.checkpoint_ep > 0:
            model.load_state_dict(torch.load(f'{config.save_model_path}/ep{config.checkpoint_ep}_trans_forecaster.pth'))

        loss = torch.nn.MSELoss()

        train(model, optimizer, loss, dataloader_train, dataloader_test, config, run)
        
        #model.load_state_dict(torch.load(f'{config.save_model_path}/ep9_trans_forecaster.pth'))
        plot_result(model, env_train.market, agent, value=False)


if __name__ == '__main__':
    main()