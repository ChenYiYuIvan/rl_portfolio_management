# Market info

Market composed by $n=7$ risky assets (stocks chosen randomly from S&P500) and a risk-free asset (cash) with $r_f = 0$

Transaction cost: $0.2\%$ of transaction value for both buying and selling

Market data: daily OHLCV -> use only daily high ($p_t^h$), low ($p_t^l$) and close ($p_t$) prices

Period under analysis:
- from 2007-01-01 to 2016-12-31 for training RL agents
- from 2017-01-01 to 2020-12-31 for testing

# Portfolio constraints

- $w = [w_0, \dots, w_n]$
- $\sum_{i=0}^{n}w_i = 1$
- $w_i \geq 0 \space \forall i$

# Assumpitons

- Trading volume of portfolio is not large enough to influence the price of the stocks
- 0 slippage: expected price of trade == actual price of trade
- Portfolio is rebalanced at the end of every trading day
- Starting portfolio is 1$ in cash

# Portfolio dynamics:
![image](images/dynamics.png)
where:
- $w_{t-1}$ are the portfolio weights at the start of period $t$
- $w_t^{'}$ are the portfolio weights at the end of period $t$ before rebalancing
- $y_t$ is the vector of stock price changes
- $\mu_t$ is the multiplicative cost of transaction to pass from $w_t^{'}$ to $w_{t-1}$

# RL framework:

- input data: 