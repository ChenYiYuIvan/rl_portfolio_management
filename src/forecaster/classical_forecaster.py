from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from torch import mode
from src.environments.market import Market
from src.environments.portfolio import Portfolio
from src.agents.sac_agent import SACAgent
from src.utils.file_utils import read_yaml_config
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr


def main():

    env_config_train = read_yaml_config('experiments/env_train_1')
    env_train = Portfolio(env_config_train)
    market_train = env_train.market
    data_train_o = market_train.data

    env_config_test = read_yaml_config('experiments/env_test_1')
    env_test = Portfolio(env_config_test)
    market_test = env_test.market
    data_test_o = market_test.data

    agent_config = read_yaml_config('experiments/sac_8')
    agent = SACAgent('sac', env_train, 42, agent_config)

    (data_train, _) = agent.preprocess_data((data_train_o, None))
    (data_test, _) = agent.preprocess_data((data_test_o, None))
    

    fig, ax = plt.subplots(2)

    ax[0].plot(data_train[:,0,2])
    ax[0].grid()

    # pacf - close returns
    plot_pacf(data_train[:,0,2], ax=ax[1], lags=50)

    for i, val in zip([0,1,3],['high','low','close']):
        print(f'{val}:')
        for lag in range(1,51):
            close_rets = data_train[lag:,0,2]
            high_rets = data_train[:-lag,0,i]
            corr, p_val = pearsonr(close_rets, high_rets)
            if p_val < 0.05:
                print(f'lag: {lag} - corr: {corr}')
        print('------')

    data_train_df = pd.DataFrame(data=data_train.squeeze(), columns=['high','low','close','volume'])
    model = VAR(data_train_df)
    model_fit = model.fit(maxlags=10)
    print(model_fit.summary())

    plt.show()


if __name__ == '__main__':
    main()

