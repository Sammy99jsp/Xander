from itertools import count
import random
from typing import Any, Literal, cast
import numpy as np
import torch as T
import torch.nn as nn
import gymnasium as gym
from RL.algorithms.pdqn import train, PolicyNet, ValueNet, Transition, RandomAction


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
        x = self.shared(state)
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
        if not isinstance(action_ty, list):
            assert isinstance(action_param, T.Tensor)

    
class ActionGenerator(RandomAction[Action]):
    def __call__(self, state) -> tuple[Action, T.Tensor]:
        return random.choice(["left", "right"]), T.rand(1) * 1.0
    

ACTION: dict[Action, float] = {"left": -1.0, "right": 1.0}

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
        self.env.render()
        return T.tensor(state, dtype=T.float32), cast(float, reward), terminated, truncated, info

env = gym.make("MountainCarContinuous-v0", max_episode_steps=250, render_mode="human")
policy = Policy(env.observation_space.shape[0])
value = Value(env.observation_space.shape[0])

train(
    action_types=["left", "right"],
    random_action=ActionGenerator(),
    policy=policy,
    value=value,
    env=EnvWrapper(env),
    gamma=0.99,
    alpha=1e-6,
    beta=1e-3,
    epsilon_start=0.99,
    epsilon_decay=10_000,
    epsilon_end=0.05,
    memory_size=10000,
    batch_size=32,
    device=T.device("cuda")
)