import random
from typing import Any, Callable, Literal, cast
import typing

import numpy as np
import torch as T
import torch.nn as nn
import wandb

from RL.algorithms.pdqn import PolicyNet, ValueNet, train

import gymnasium as gym


Action = Literal["left", "right"]
class Policy(PolicyNet[Action], nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        self.actions: dict[Action, nn.Module] = {
            "left": nn.Linear(128, 1),
            "right": nn.Linear(128, 1),
        }

    def forward(self, state) -> dict[Action, T.Tensor]:
        x: T.Tensor = self.shared.forward(state)
        return {action: T.clamp(layer(x), 0.0, 1.0) for action, layer in self.actions.items()}

class Value(ValueNet[Action], nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        self.actions = {
            "left": nn.Sequential(nn.Linear(128 + 1, 128), nn.ReLU(), nn.Linear(128, 1)),
            "right": nn.Linear(128 + 1, 1),
        }
    

    def forward(self, state, action_ty, action_param) -> T.Tensor:
        """
        Parameters
        ----------
        state: T.Tensor | list[T.Tensor; batch_size] (batch_size, input_dim)
            The state of the environment.
        action_ty: Action | list[Action; batch_size]
            The type of action taken.
        action_param: T.Tensor | list[T.Tensor; batch_size]
            The parameters of the action taken.

        Returns
        -------
        T.Tensor (batch_size, 1)
            The value of the state-action pair.
        """
        if isinstance(state, list) and isinstance(action_ty, list) and isinstance(action_param, list):
            assert len(state) == len(action_ty) == len(action_param)

            return T.stack([self.actions[k](T.concat([self.shared(s), x_k]))\
                        for s, k, x_k in zip(state, action_ty, action_param)])
        return self.actions[action_ty](T.concat([self.shared(state), action_param], dim=-1))


class EnvWrapper(gym.Env[T.Tensor, tuple[Action, T.Tensor]]):
    def __init__(self, env: gym.Env):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
    
    def reset(self):
        state, info = self.env.reset()
        return T.tensor(state, dtype=T.float32), info
    
    def step(self, raw_action: tuple[Action, T.Tensor]) -> tuple[T.Tensor, float, bool, bool, dict[str, Any]]:
        ty, param = raw_action
        action = np.array([ACTION[ty] * param.item()])
        state, reward, terminated, truncated, info = self.env.step(action)
        # self.env.render()
        return T.tensor(state, dtype=T.float32), cast(float, reward), terminated, truncated, info

ACTION: dict[Action, float] = {"left": -1.0, "right": 1.0}

random_action: Callable[[T.Tensor], tuple[Action, T.Tensor]] = lambda _state: (random.choice(["left", "right"]), T.rand(1) * 1.0)
    
env = gym.make("MountainCarContinuous-v0", max_episode_steps=1000)
policy = Policy(env.observation_space.shape[0]).to("cuda") # type: ignore
value = Value(env.observation_space.shape[0]).to("cuda") # type: ignore

run = wandb.init(project="pdqn-test", entity="sammy99jsp_wandb", mode="online")
train(["left", "right"], EnvWrapper(env), random_action, policy, value, device="cuda", run=run, eps_decay=100_000,  MAX_STEPS=1_000_000)