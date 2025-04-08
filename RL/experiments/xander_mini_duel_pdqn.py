import argparse
from copy import deepcopy
import math
import os
import random
from typing import Any, Literal, cast
import typing
import numpy as np
import wandb
import gymnasium as gym

import torch as T
import torch.nn as nn

from RL.algorithms import pdqn
from RL.algorithms.pdqn import PolicyNet, ValueNet
from RL.env.env import Duel, DuelConfig, parse_config

Action = Literal["end", "move", "attack0"]
ACTIONS: list[Action] = ["end", "move", "attack0"]

class XanderDuelPolicy(PolicyNet[Action], nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        self.actions: dict[Action, nn.Module] = {
            "end": nn.Linear(128, 0),
            "move": nn.Linear(128, 8),
            "attack0": nn.Linear(128, 8),
        }
    
    def to(self, device: T.device) -> typing.Self:
        self.shared.to(device)
        for layer in self.actions.values():
            layer.to(device)

        return self

    def forward(self, state) -> dict[Action, T.Tensor]:
        x: T.Tensor = self.shared.forward(state)
        return {action: T.clamp(layer(x), 0.0, 1.0) for action, layer in self.actions.items()}

class XanderDuelValue(ValueNet[Action], nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        self.actions = {
            "end": nn.Sequential(nn.Linear(128 + 0, 1),),
            "move": nn.Sequential(nn.Linear(128 + 8, 128), nn.ReLU(), nn.Linear(128, 1)),
            "attack0": nn.Sequential(nn.Linear(128 + 8, 128), nn.ReLU(), nn.Linear(128, 1)),
        }
    
    def to(self, device: T.device) -> typing.Self:
        self.shared.to(device)
        for layer in self.actions.values():
            layer.to(device)

        return self
    
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

def maximal(arr: np.ndarray) -> np.ndarray:
    out = np.zeros_like(arr)
    out[np.argmax(arr)] = 1.0
    return out

def maximal_t(arr: T.Tensor) -> T.Tensor:
    out = T.zeros_like(arr)
    out[T.argmax(arr)] = 1.0
    return out

class XanderDuelWrapper(gym.Env[T.Tensor, tuple[Action, np.ndarray]]):
    def __init__(self, env: Duel):
        self.env = env
        self.observation_space = env.observation_space # type: ignore
        self.action_space = env.action_space # type: ignore
    
    def reset(self):
        state, info = self.env.reset()
        return T.tensor(state, dtype=T.float32), info
    
    def step(self, raw_action: tuple[Action, np.ndarray]) -> tuple[T.Tensor, float, bool, bool, dict[str, Any]]:
        match raw_action:
            case ("end", _):
                action = np.concatenate([np.array([1.0, 0.0, 0.0]), np.zeros(8, dtype=np.float32)])
            case ("move", param):
                action = np.concatenate([np.array([0.0, 1.0, 0.0]), maximal(param)])
            case ("attack0", param):
                action = np.concatenate([np.array([0.0, 0.0, 1.0]), maximal(param)])
            case (ty, _):
                raise ValueError(f"Unknown action ty: {ty}")

        state, reward, terminated, truncated, info = self.env.step(action)
        # self.env.render()
        return T.tensor(state, dtype=T.float32), cast(float, reward), terminated, truncated, info
    
def random_action(_: T.Tensor) -> tuple[Action, T.Tensor]:
    action = np.random.choice(ACTIONS)
    if action == "end":
        return action, T.zeros(0)
    else:
        return action, maximal_t(T.rand(8) * 1.0)

def main(args: argparse.Namespace) -> None:
    config = parse_config(args.config)

    if args.threads > 1:
        import multiprocessing as mp
        
        for i in range(args.threads):
            config = deepcopy(config)
            config["seed"] = random.randint(0, 2**32 - 1)
            mp.Process(target=launch, args=(config, os.path.join("out", "pdqn", f"run{i}"))).start()
    else:
        launch(config, os.path.join("out", "pdqn", "run0"))

def launch(config: DuelConfig, out_path: str) -> None:
    device = T.device("cuda" if T.cuda.is_available() else "cpu")
    env = XanderDuelWrapper(Duel(config))
    policy = XanderDuelPolicy(env.observation_space.shape[0]).to(device) # type: ignore
    value = XanderDuelValue(env.observation_space.shape[0]).to(device) # type: ignore

    pdqn.train(
        action_types=ACTIONS, 
        env=env, 
        random_action=random_action, 
        policy=policy, 
        value=value,
        device=device,
        run=wandb.init(config=config, **config["wandb"]), # type: ignore
        out_dir=out_path,
        **config["hyperparameters"],
    )
    

if __name__ == "__main__":

    p = argparse.ArgumentParser()
    
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--threads", type=int, default=1)

    args = p.parse_args()
    main(args)