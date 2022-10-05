# Thesis Structure

## 0. Introduction / Summary

- Introduce the problem of portfolio optimization and give motivation for this thesis
- Explain structure of next chapters

## 1. Background

### Portfolio optmization

- Definition of portfolio
- Return + Log-return
- Markowitz + Efficient frontier + Sharpe ratio (tangency portfolio) + Sortino ratio
- Transaction cost

### Reinforcement Learning

- Mathematical formalism (MDP) -> Bellman equation, policy, value function, ...
- Dynamic programming (model-based)
- Classical approaches: policy / value iteration
- Reinforcement learning (model-free)
- Reward signal
- Exploration vs exploitation
- Classical approaches: temporal differences, SARSA, q-learning

### Deep learning

- Neural network
- Gradient descent, backpropagation
- Deep learning approaches to reinforcement learning: overview + most popular methods

### Time series analisys

- Stock prices / returns as multivariate time series
- Forecasting
- Classical methods (ARIMA) vs deep learning approaches

## 2. Contribution (methods employed)

### Problem statement + framework definition

- Define problem of having to dynamically rebalance a portfolio of stocks
- Assumptions made on the market (fractional transactions, zero slippage, zero market impact, ...)
- action and observation spaces
- Reward signals used (log-return, differential sharpe ratio, differential downside deviation ratio, ...)

### Network architectures and methods used

- CNN, RNN (LSTM / GRU), Transformers
- DDPG, TD3, SAC + modifications (adding imitation learning / pretraining / forecasting)

## 3. Experiments

- Data used (S&P500)
- Results obtained on different sets of stocks
- Comparison with various baselines (constantly rebalanced portfolio, Markowitz portfolio, ...)

## 4. Conclusion

- Comments on results obtained and the whole work
- Possible extensions as future work