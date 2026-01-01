import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd 
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from env.trading_env import TradingEnv

df = pd.read_csv("data/raw_data.csv")

split_idx = int(0.7 * len(df))
train_df = df.iloc[:split_idx]

env = DummyVecEnv([
    lambda: TradingEnv(train_df)
])

model = PPO(
    "MlpPolicy",
    env,
    verbose = 1,
    tensorboard_log = "./logs/tensorboard/",
    device = "cuda"
)

model.learn(total_timesteps=300_000)

model.save("models/ppo_trading_model") 