import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd 
from stable_baselines3 import PPO
from env.trading_env import TradingEnv
import matplotlib.pyplot as plt

df = pd.read_csv("data/raw_data.csv")

split_idx = int(0.7 * len(df))
test_df = df.iloc[split_idx:]

env = TradingEnv(test_df)

model = PPO.load("models/ppo_trading_model")

obs, _ = env.reset()
done = False

equity_curve = []

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, _ = env.step(action)
    equity_curve.append(env.net_worth)

buy_price = test_df["Close"].iloc[0]
sell_price = test_df["Close"].iloc[-1]

buy_and_hold_networth = (sell_price/buy_price) * env.initial_balance

print(f"PPO Final Net worth: {env.net_worth:.2f}")
print(f"Buy and Hold Net worth: {buy_and_hold_networth:.2f}")

plt.figure(figsize=(10, 5))
plt.plot(equity_curve, label="PPO Agent")
plt.axhline(
    y=buy_and_hold_networth,
    color="r",
    linestyle="--",
    label="Buy & Hold"
)
plt.legend()
plt.title("Equity Curve Comparison")
plt.xlabel("Time Step")
plt.ylabel("Net Worth")
plt.savefig("results/equity_curve.png")
plt.show()