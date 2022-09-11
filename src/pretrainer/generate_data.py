import numpy as np
from src.utils.portfolio_utils import get_opt_portfolio


def generate_data(num_obs, num_assets, window_length, objective, trans_coef):

    X_obs = []
    X_weight = []
    y = []

    while len(X_obs) < num_obs:

        weights = np.random.random_sample(num_assets+1)
        weights /= weights.sum()

        L = np.random.random_sample((num_assets, num_assets)) / 100
        L = np.tril(L)

        mu = np.ones(num_assets) / 100
        var = np.matmul(L, L.T)

        a = (var < 0).sum()
        if a > 0:
            print('a')

        log_rets1 = np.random.multivariate_normal(mu, var, window_length-1).T
        log_rets2 = np.random.multivariate_normal(mu, var, window_length-1).T

        log_rets_max = np.maximum(log_rets1, log_rets2)
        log_rets_min = np.minimum(log_rets1, log_rets2)
        log_rets_close = np.zeros(log_rets_max.shape)
        for i in range(log_rets_max.shape[0]):
            for j in range(log_rets_min.shape[1]):
                log_rets_close[i,j] = np.random.uniform(log_rets_min[i,j], log_rets_max[i,j])

        # open / high / low / close / volume
        # open and volume are set to close because they're not used anyway
        log_rets = np.array((log_rets_close, log_rets_max, log_rets_min, log_rets_close, log_rets_close))
        log_rets = np.transpose(log_rets, (1,2,0))

        # add cash data
        cash = np.zeros((1,log_rets.shape[1],log_rets.shape[2]))
        log_rets = np.concatenate((log_rets, cash), axis=0)

        state = (log_rets, weights)
        action, message = get_opt_portfolio(state, objective, trans_coef, to_logret=False)

        if message is None: # = success
            X_obs.append(log_rets)
            X_weight.append(weights)
            y.append(action)

    return X_obs, X_weight, y


if __name__ == '__main__':

    num_stocks = 4
    window_length = 50

    X_obs_train, X_weight_train, y_train = generate_data(2000, num_stocks, window_length, 'sharpe_ratio', 0.002)
    X_obs_test, X_weight_test, y_test = generate_data(500, num_stocks, window_length, 'sharpe_ratio', 0.002)

    path = f'src/pretrainer/data{num_stocks}_{window_length}/'

    np.save(f'{path}X_obs_train.npy', X_obs_train)
    np.save(f'{path}X_weight_train.npy', X_weight_train)
    np.save(f'{path}y_train.npy', y_train)
    np.save(f'{path}X_obs_test.npy', X_obs_test)
    np.save(f'{path}X_weight_test.npy', X_weight_test)
    np.save(f'{path}y_test.npy', y_test)


# a = np.load('src/pretrainer/data/X_obs_train.npy')
# print(a.shape)