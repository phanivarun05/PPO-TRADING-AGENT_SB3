# PPO-Based Algorithmic Trading Agent

This project implements an **algorithmic trading agent** using **Proximal Policy Optimization (PPO)** with **Stable-Baselines3**.  
The agent learns to trade a financial asset from historical market data by interacting with a **custom Gymnasium trading environment**.

The project follows **industry-correct reinforcement learning practices**, including:
- Time-based train/test split
- Proper baselines (Buy & Hold)
- Realistic trading constraints
- Clean separation of training and evaluation

---

## 🚀 Key Features

- Custom **Gymnasium trading environment**
- Continuous action space (position sizing)
- PPO with Generalized Advantage Estimation (GAE)
- Cash and position constraints (no unlimited leverage)
- GPU training support (Lightning AI / cloud)
- Buy-and-Hold baseline comparison
- Equity curve visualization
- Modular, reproducible project structure

---

## 🧠 Reinforcement Learning Formulation

### Observation Space
At each time step, the agent observes:
- A rolling window of OHLCV price data
- Current account balance
- Current asset position

### Action Space
Continuous action ∈ **[-1, 1]**
- -1 → decrease position (sell)
-  0 → hold
- +1 → increase position (buy)

Position size is **bounded** to prevent unrealistic leverage.

### Reward Function
Step-wise change in portfolio net worth:

