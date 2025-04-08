# Code under MIT License
# Author: Sammy99jsp
# Implementation of Parameterized Deep Q-Network (PDQN) algorithm
# Paper: https://arxiv.org/pdf/1810.06394

from collections import deque
from itertools import count
import math
import os
import random
import typing
import numpy as np
import torch as T
import torch.nn as nn
import gymnasium as gym
import tqdm
import wandb
import wandb.wandb_run

class PolicyNet[K](nn.Module):
    """
    Represents the policy network `x_k(•, θ_t)`.

    Each head of the network corresponds to a different action type, `k`.

    Parameters
    ----------
    state: T.Tensor
        The state of the environment.
        Shape: `(batch_size, input_dim)`
    
    Returns
    -------
    dict[K, T.Tensor]
        A dictionary mapping action type `k` -> tensor of shape `(batch_size, |x_k|)` (`x_k` in `X_k`).
    """
    def forward(self, state: T.Tensor) -> dict[K, T.Tensor]:
        raise NotImplementedError
    
    if typing.TYPE_CHECKING:
        @typing.overload # type: ignore
        def __call__(self, state: T.Tensor) -> dict[K, T.Tensor]: ... 

class ValueNet[K](nn.Module):
    def forward(self, state: T.Tensor, k: K | list[K], x_k: T.Tensor | list[T.Tensor]) -> T.Tensor:
        """
        Parameters
        ----------
        state: T.Tensor | list[T.Tensor; batch_size]
            The state of the environment.
            Shape: `(batch_size, input_dim)`
        
        k: K | list[K]
            The action types taken.
            Shape: `(batch_size,)`
        
        x_k: list[T.Tensor] | T.Tensor
            The parameters of the actions taken.
            Shape: `(batch_size, |x_k|)`

        Returns
        -------
        T.Tensor
            The value of the state-action pair.
            Shape: `(batch_size, 1)`
        """
        raise NotImplementedError
    
    if typing.TYPE_CHECKING:
        @typing.overload # type: ignore
        def __call__(self, s_t: T.Tensor, k: K, x_k: T.Tensor) -> T.Tensor: ...
        @typing.overload
        def __call__(self, s_t: list[T.Tensor], k: list[K], x_k: list[T.Tensor]) -> T.Tensor: ...

class Transition[K](typing.NamedTuple):
    state: T.Tensor
    action: tuple[K, T.Tensor]
    reward: float
    next_state: T.Tensor | None

class ReplayMemory[K]:
    memory: deque[Transition[K]]

    def __init__(self, capacity: int):
        self.memory = deque([], maxlen=capacity)

    def store(self, transition: Transition[K]):
        self.memory.append(transition)
    
    def sample(self, batch_size: int, clip = True) -> list[Transition[K]]:
        return random.sample(self.memory, batch_size if not clip else min(batch_size, len(self.memory)))
    
    def __len__(self):
        return len(self.memory)

def optimize[K](
    step: int,
    memory: ReplayMemory[K],
    batch_size: int,
    policy: PolicyNet[K],
    value: ValueNet[K],
    value_optimizer: T.optim.Optimizer,
    policy_optimizer: T.optim.Optimizer,
    gamma: float,
    device: T.device,
    run: wandb.wandb_run.Run
):
    if len(memory) < batch_size:
        return
    
    # Sample a batch of transitions [s_b, a_b, r_b, s_{b+1}] from 𝓓.
    batch: tuple[list[T.Tensor], list[tuple[K, T.Tensor]], list[float], list[T.Tensor | None]] = tuple(map(list, zip(*memory.sample(batch_size)))) # type: ignore
    s_b, s_b_next = batch[0], batch[3]
    r_b = T.tensor(batch[2], device=device) # (batch_size,)
    assert r_b.shape == T.Size([len(s_b)]), f"Reward shape: {r_b.shape}"

    non_final_mask = T.tensor([s is not None for s in s_b_next], device=device)
    non_terminal_states_next = T.stack([s for s in s_b_next if s is not None])
    y_b = r_b
    y_b[non_final_mask] = T.max(gamma * T.stack([
        value(non_terminal_states_next, k, x_k) for k, x_k in policy(non_terminal_states_next).items()
    ]).squeeze(), dim=0).values

    __t: tuple[list[K], list[T.Tensor]] = tuple(map(list, zip(*batch[1]))) # type: ignore
    b_k, b_x_k = __t

    # Compute the loss for the value network
    value_loss = 0.5 * (value(s_b, b_k, b_x_k) - y_b) ** 2
    value_optimizer.zero_grad()
    value_loss_sum = value_loss.sum()
    value_loss_sum.backward()
    value_optimizer.step()

    # Compute the loss for the policy network
    s_b_tensor = T.stack(s_b)
    policy_loss = -T.sum(T.cat([
        value(s_b_tensor, k, x_k) for k, x_k in policy(s_b_tensor).items()
    ], dim=1), dim=1) # (batch_size,)
    assert policy_loss.shape == T.Size([len(s_b)]), f"Policy loss shape: {policy_loss.shape}, expected: {T.Size([len(s_b)])}"

    policy_optimizer.zero_grad()
    policy_loss_sum  = policy_loss.sum()
    policy_loss_sum.backward()
    policy_optimizer.step()
    
    run.log({"policy_loss": policy_loss_sum.item(), "value_loss": value_loss.sum().item()}, step=step)

def train[K](
    action_types: list[K],
    env: gym.Env[T.Tensor, tuple[K, np.ndarray]],
    random_action: typing.Callable[[T.Tensor], tuple[K, T.Tensor]],
    policy: PolicyNet[K],
    value: ValueNet[K],
    device: T.device,
    run: wandb.wandb_run.Run,
    alpha: float = 0.001,
    beta: float = 0.1,
    eps_start: float = 0.9,
    eps_end: float = 0.05,
    eps_decay: float = 1000,
    gamma: float = 0.99,
    memory_capacity: int = 10_000,
    batch_size: int = 32,
    MAX_STEPS: int = 100_000,
    save_interval: int = 20_000,
    out_dir: str = "out"
):
    policy_save_path = os.path.join(out_dir, "policy")
    value_save_path = os.path.join(out_dir, "value")
    os.makedirs(policy_save_path, exist_ok=True)
    os.makedirs(value_save_path, exist_ok=True)
    
    policy = policy.to(device)
    value = value.to(device)
    memory: ReplayMemory[K] = ReplayMemory(memory_capacity)
    value_optimizer = T.optim.SGD(value.parameters(), lr=alpha)
    policy_optimizer = T.optim.SGD(policy.parameters(), lr=beta,)

    steps = 0
    p_bar = tqdm.tqdm(range(MAX_STEPS))
    p_bar.display()
    for episode in count():
        ep_reward = 0
        s_t, _ = env.reset()
        for t in count(start=1):
            if steps > MAX_STEPS: break
            if steps % save_interval == 0:
                T.save(policy.state_dict(), os.path.join(policy_save_path, f"{steps}.pt"))
                T.save(value.state_dict(), os.path.join(value_save_path, f"{steps}.pt"))

            with T.no_grad():
                # Compute action parameters x_k ← x_k(s_t, θ_t)
                s_t = s_t.to(device)
                x = policy(s_t)

                # Select action a_t = (k, x_k_t) according to the ε-greedy policy
                a_t: tuple[K, T.Tensor]
                epsilon = eps_end + (eps_start - eps_end) * \
                    math.exp(-1. * steps / eps_decay)
                match T.rand(1):
                    case p if p < epsilon:
                        a_t = random_action(s_t)
                    case _:
                        a_t = max(x.items(), key=lambda item: value(s_t, item[0], item[1]).item())

                a_t = (a_t[0], a_t[1].to(device))
                # Take action a_t and observe reward r_t and the next state s_{t+1}.
                s_t_next, r_t, terminated, truncated, _ = env.step((a_t[0], a_t[1].cpu().numpy())) # type: ignore
                s_t_next = s_t_next.to(device)
                done = terminated or truncated

                # Store transition [s_t, a_t, r_t, s_{t+1}] into replay memory 𝓓.
                memory.store(Transition(s_t, a_t, r_t, s_t_next if not done else None)) # type: ignore
                ep_reward += r_t # type: ignore

            optimize(steps, memory, batch_size, policy, value, value_optimizer, policy_optimizer, gamma, device, run)

            steps += 1
            s_t = s_t_next
            p_bar.update(1)
            if done: break

        run.log({"sum_rew": ep_reward, "episode_length": t, "episode": episode}, step=steps)

    T.save(policy.state_dict(), os.path.join(policy_save_path, f"{steps}-final.pt"))
    T.save(value.state_dict(), os.path.join(value_save_path, f"{steps}-final.pt"))