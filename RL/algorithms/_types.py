import typing
import gymnasium as gym
from typing import Generic, Protocol, TypeVar, Union

from stable_baselines3.common.vec_env import VecEnv

import numpy as np
from wandb.sdk.wandb_run import Run

if typing.TYPE_CHECKING:
    from RL.env2.duel import XanderDuelEnv


Hyperparameters = TypeVar("Hyperparameters", covariant=True)
WrappedEnv = TypeVar("WrappedEnv", bound=Union[gym.Env, VecEnv], covariant=True)
class Agent(Protocol, Generic[Hyperparameters, WrappedEnv]):
    def __init__(self, env: "XanderDuelEnv", hyperparameters: Hyperparameters, run: Run) -> None: ...

    @property
    def env(self) -> WrappedEnv: ...

    @property
    def run(self) -> Run: ...

    @property
    def num_envs(self) -> int: ...

    @property
    def is_training_divisible(self) -> bool: ...

    def train(self, steps: int) -> None: ...

    def predict(self, obs: np.ndarray) -> np.ndarray: ...

    def load(self, path: str) -> None: ...

    def save(self, path: str) -> None: ...
