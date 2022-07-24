# Hypothesis

- Trading volume of portfolio is not large enough to influence the price of the stocks
- Transaction cost of 0.2% or 0.25% for both buy and sell

# Training a Reinforcement Learning algorithm

Episode = keep policy / value function fixed and apply actions on environment

Loop for episodes:
- select [subsection of / all of] historical data
- use them as observation to make actions
- store rewards
- update policy / value function


# Data used

16 stocks from S&P500 ("AAPL", "ATVI", "CMCSA", "COST", "CSX", "DISH", "EA", "EBAY", "FB", "GOOGL", "HAS", "ILMN", "INTC", "MAR", "REGN", "SBUX")

Data taken from https://www.kaggle.com/datasets/camnugent/sandp500/versions/4?resource=download

Data from 2013-02-08 to 2018-02-07:
- training set: 2013-02-08 to 2016-02-07
- testing set: 2016-02-08 to 2018-02-07

# Implementation

Time step = 1 day

Observation = {start, end, lowest, highest, volume} for single day

Price relative vector = End value / Start value

Assumption: end price == start price of next day

Portfolio weights = fraction of portfolio value invested in a given stock

Action = new portfolio weights (calculated end of time period, valid for next time period)

Also include cash as a stock -> price relative vector = 1

Action depends on the past N observations (from today - N days to yesterday)

Rebalancing for time interval t+1 is done at closing of time interval t, which means the 

Idea -> Try to also use opening value of today

Flow $(t-1 \rightarrow t) ==$ period t:
- have portfolio with weights $w_{t-1}$ and value v
- see open/high/low/close values of timestep $t-1$
- weights $w_{t-1}$ change to new weights $a_{t-1}$
- update weights to $w_t$, compute trading cost $\mu_t$, new portfolio value v and obtain reward $r_t$
- $t \leftarrow t+1$