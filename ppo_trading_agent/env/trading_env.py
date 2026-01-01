import gymnasium as gym
from gymnasium import spaces
import numpy as np 
import pandas as pd 

class TradingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, df, window_size=10, initial_balance=10000):
        super().__init__()

        self.df = df
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.max_position = 10

        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=np.inf, high=np.inf, shape=(window_size * 5 + 2,), dtype=np.float32)

        self.reset()

    def reset(self, seed=None):
        self.balance = self.initial_balance
        self.position = 0
        self.current_step = self.window_size
        self.net_worth = self.initial_balance
        self.prev_net_worth = self.initial_balance

        return self._get_observation(), {}

    def _get_observation(self):
        window = self.df.iloc[self.current_step - self.window_size : self.current_step][["Open", "High", "Low", "Close", "Volume"]].values.flatten()

        obs = np.concatenate([window, [self.balance, self.position]])

        return obs 

    def step(self, action):
        action = float(action[0])
        action = np.clip(action, -1, 1)

        price = self.df.iloc[self.current_step]["Close"]

        new_position = np.clip(
            self.position + action,
            -self.max_position,
            self.max_position
        )

        delta_position = new_position - self.position
        cost = delta_position * price

        if self.balance - cost >= 0:
            self.position = new_position
            self.balance -= cost

        self.net_worth = self.balance + self.position * price

        reward = self.net_worth - self.prev_net_worth 
        self.prev_net_worth = self.net_worth

        self.current_step += 1
        done = self.current_step >= len(self.df) - 1

        return self._get_observation(), reward, done, False, {}

    def render(self):
        print(f"Step: {self.current_step}, Net Worth: {self.net_worth}")

