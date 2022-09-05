import numpy as np
import scipy.optimize as spo

from src.utils.data_utils import EPS, remove_not_used, prices_to_simplereturns


def get_opt_portfolio(state, objective, trans_coef):
    # get optimal portfolio weights given current portfolio and past stock prices
    # according to Modern Portfolio Theory
    
    assert objective in ('sharpe_ratio')
    if objective == 'sharpe_ratio':
        objective_func = get_sharpe_ratio

    # obs = price matrix, weights = current portfolio weights
    obs, weights = state
    obs = remove_not_used(obs, cash=False, high=True, low=True)
    rets = prices_to_simplereturns(obs).squeeze()
    # obs shape: [num stocks, window length]

    bounds = [(0,None) for _ in range(obs.shape[0])]
    constraints = {
        'type': 'eq',
        'fun': lambda weights: np.sum(weights) - 1
    }

    optim = spo.minimize(objective_func, weights, (weights, rets, trans_coef),
        bounds=bounds,
        constraints=constraints,
        method='SLSQP'
    )

    if optim.success:
        return optim.x, None
    else:
        # print('Problem with MPT, keep portfolio unchanged')
        return weights, optim.message


def get_mean_portfolio_ret(mu, weights):
    return np.dot(weights, mu)


def get_var_portfolio_ret(sigma, weights):
    return np.dot(np.dot(weights, sigma), weights)


def get_approx_trans_cost(weights_old, weights_new, trans_coef):
    return np.sum(np.abs(weights_old - weights_new)) * trans_coef


def get_sharpe_ratio(weights_new, weights_old, rets, trans_coef):
    mu = np.mean(rets, axis=1)
    sigma = np.cov(rets)

    num = get_mean_portfolio_ret(mu, weights_new) - get_approx_trans_cost(weights_old, weights_new, trans_coef)
    denom = get_var_portfolio_ret(sigma, weights_new)

    # minus because it's a max problem solved as a min problem
    return - num / (denom + EPS)