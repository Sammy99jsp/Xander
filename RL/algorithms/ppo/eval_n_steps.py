from typing import Callable

import numpy as np
import gymnasium as gym
from stable_baselines3.common.callbacks import EventCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv

class EvalForNSteps(EventCallback):
    eval_env: gym.Env

    def __init__(self, env_fn: Callable[[], gym.Env], eval_steps = 1000):
        super().__init__()
        self.eval_steps = eval_steps
        self.eval_env = env_fn() # type: ignore[list-item, return-value]

    def _on_step(self) -> bool:
        ep_reward: float = 0.0
        ep_length = 0

        ep_lengths: list[float] = []
        ep_rewards: list[float] = []
        wins = 0
        losses = 0
        illegal_actions = 0
        ep_hp_at_end: list[float] = []
        attacks: int = 0

        print(f"TESTING! For {self.eval_steps}")

        obs, info = self.eval_env.reset()
        for _ in range(self.eval_steps):
            actions, _ = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = self.eval_env.step(actions)
            ep_reward += reward # type: ignore
            ep_length += 1

            if "illegal" in info.keys():
                illegal_actions += 1

            if "won" in info.keys():
                wins += 1
            
            if "lost" in info.keys():
                losses += 1

            if "hp_left" in info.keys():
                ep_hp_at_end.append(info["hp_left"])

            if "attack" in info.keys():
                attacks += 1

            if truncated or terminated:
                ep_lengths.append(ep_length)
                ep_rewards.append(ep_reward)
                ep_reward = 0.0
                ep_length = 0

                obs, info = self.eval_env.reset()

        if len(ep_lengths) > 0:
            self.logger.record("test/mean_ep_length", np.mean(ep_lengths))
            self.logger.record("test/mean_ep_reward", np.mean(ep_rewards))
            self.logger.record("test/wins", wins / len(ep_lengths))
            self.logger.record("test/losses", losses / len(ep_lengths))
            self.logger.record("test/hp_left_mean", np.mean(ep_hp_at_end))
        
        self.logger.record("test/illegal_actions", illegal_actions / self.eval_steps)
        self.logger.record("test/attacks", attacks / self.eval_steps)

        return True

