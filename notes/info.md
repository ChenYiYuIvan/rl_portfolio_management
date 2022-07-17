# Hypothesis

- Trading volume of portfolio is not large enough to influence the price of the stocks
- Transaction cost of 0.2% or 0.25% for both buy and sell

# Training a Reinforcement Learning algorithm

Episode = keep policy / value function fixed and apply actions on environment

Loop for episodes:
    - select subsection of historical data
    - use them as observation to make actions
    - store rewards
    - update policy / value function


# Data used

16 stocks from S&P500

# Implementation

Time step = 1 day

Observation = {start, end, lowest, highest, volume} for single day

Price relative vector = End value / Start value

Portfolio weights = fraction of portfolio value invested in a given stock

Action = new portfolio weights

Also include cash as a stock -> price relative vector = 1