from collections import deque, namedtuple
from itertools import count
import math
import random
from typing import Optional, TypedDict
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
import wandb
import wandb.wandb_run

from rl.env import Duel, DuelConfig


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    memory: deque[Transition]

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size, clip=False) -> list[Transition]:
        return random.sample(self.memory, batch_size if not clip else min(batch_size, len(self.memory)))

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, n_observations, n_actions, n_dirs):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3_a = nn.Linear(128, n_actions)
        self.layer3_b = nn.Linear(128, n_dirs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return torch.concat((self.layer3_a(x), self.layer3_b(x)), dim=1)
    
def select_action_zero(tensor: torch.Tensor) -> torch.Tensor:
    # print(f"select_action_zero: {tensor.shape}")
    max_action = torch.argmax(tensor[:, :, :3], dim=-1)
    max_dir = torch.argmax(tensor[:, :, 3:], dim=-1)

    hot = torch.zeros(tensor.shape, device=tensor.device, dtype=torch.float32)

    # print(f"max_action: {max_action.shape} -> {max_action}")
    hot[:, :, :3][:, :, max_action.flatten()] = 1.0
    hot[:, :, 3:][:, :, max_dir.flatten()] = 1.0
    # print(f"hot: {hot.shape}, hot: {hot[:, :, :]}")
    return hot

def mean(l):
    if len(l) == 0:
        return 0
    
    return sum(l) / len(l)

class Hyperparameters(TypedDict):
    BATCH_SIZE: int
    """Number of transitions sampled from the replay buffer."""

    GAMMA: float
    """Discount factor."""

    EPS_START: float
    """Starting value of epsilon."""
    
    EPS_END: float
    """Final value of epsilon."""

    EPS_DECAY: float
    """Controls the rate of exponential decay of epsilon."""
    
    TAU: float
    """Update rate of the target network."""

    LR: float
    """Learning rate of the ``AdamW`` optimizer."""

    REPLAY_BUFFER_LEN: int
    """Length of the replay buffer."""

def train(env: "Duel", config: DuelConfig, device: torch.device) -> DQN:
    with wandb.init(project=config["wandb"]["project"], entity=config["wandb"]["entity"], config=config["hyperparameters"],) as run:
        hyperparameters = config["hyperparameters"]

        BATCH_SIZE = hyperparameters["BATCH_SIZE"]
        GAMMA = hyperparameters["GAMMA"]
        EPS_START = hyperparameters["EPS_START"]
        EPS_END = hyperparameters["EPS_END"]
        EPS_DECAY = hyperparameters["EPS_DECAY"]
        TAU = hyperparameters["TAU"]
        LR = hyperparameters["LR"]
        MEMORY_SIZE = hyperparameters["REPLAY_BUFFER_LEN"]

        n_actions = 8
        n_dirs = 3

        # Get the number of state observations
        state, info = env.reset(seed=config["seed"])
        n_observations = len(state)

        policy_net = DQN(n_observations, n_actions, n_dirs).to(device)
        target_net = DQN(n_observations, n_actions, n_dirs).to(device)
        target_net.load_state_dict(policy_net.state_dict())

        optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
        memory = ReplayMemory(MEMORY_SIZE)

        def select_action(state, steps_done=[0]):
            sample = random.random()
            eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                math.exp(-1. * steps_done[0] / EPS_DECAY)
            steps_done[0] += 1
            if sample > eps_threshold:
                with torch.no_grad():
                    # t.max(1) will return the largest column value of each row.
                    # second column on max result is index of where max element was
                    # found, so we pick action with the larger expected reward.
                    return select_action_zero(policy_net(state).unsqueeze(0))
            else:
                return select_action_zero(torch.tensor(np.array([[env.action_space.sample()]]), device=device, dtype=torch.float32))


        episode_durations = []

        def optimize_model(run: wandb.wandb_run.Run):
            if len(memory) < BATCH_SIZE:
                return
            transitions = memory.sample(BATCH_SIZE)
            
            # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
            # detailed explanation). This converts batch-array of Transitions
            # to Transition of batch-arrays.
            batch = Transition(*zip(*transitions))

            # Compute a mask of non-final states and concatenate the batch elements
            # (a final state would've been the one after which simulation ended)
            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=device, dtype=torch.bool)
            non_final_next_states = torch.cat([s for s in batch.next_state
                                                        if s is not None])
            state_batch = torch.cat(batch.state)
            action_batch = torch.cat(batch.action)
            reward_batch = torch.cat(batch.reward)

            # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
            # columns of actions taken. These are the actions which would've been taken
            # for each batch state according to policy_net

            # TODO: We don't really present this as 'fixed' actions per se, but rather as a
            # (action, direction) pair
            # Fix this!
            tmp: torch.Tensor = policy_net(state_batch).reshape(BATCH_SIZE, 1, -1)
            # print(f"tmp: {tmp.shape}, action_batch: {action_batch.shape}")
            # state_action_values = tmp.gather(1, action_batch)
            state_action_values = tmp

            # Compute V(s_{t+1}) for all next states.
            # Expected values of actions for non_final_next_states are computed based
            # on the "older" target_net; selecting their best reward with max(1).values
            # This is merged based on the mask, such that we'll have either the expected
            # state value or 0 in case the state was final.
            next_state_values = torch.zeros(BATCH_SIZE, device=device)
            with torch.no_grad():
                next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
            # Compute the expected Q values
            expected_state_action_values = (next_state_values * GAMMA) + reward_batch

            # Compute Huber loss
            criterion = nn.SmoothL1Loss()
            loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

            # Optimize the model
            optimizer.zero_grad()
            loss.backward()

            # Log loss

            # In-place gradient clipping
            torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
            optimizer.step()

            return loss

        if torch.cuda.is_available() or torch.backends.mps.is_available():
            num_episodes = 600
        else:
            num_episodes = 50

        # Initialize the environment and get its state
        
        
        initial_state, info = env.reset(seed=config["seed"])
        state: Optional[torch.Tensor] = torch.tensor(initial_state, dtype=torch.float32, device=device).unsqueeze(0)
        start_step = 0
        for step in tqdm.tqdm(count() if env.max_steps == -1 else range(env.max_steps), total=env.max_steps if env.max_steps != -1 else None, unit=" step"):
            action = select_action(state)

            observation, reward, terminated, truncated, _ = env.step(action)
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            next_state: Optional[torch.Tensor] = None
            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            assert state is not None
            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            loss = optimize_model(run)

            if step % config["wandb"]["log_every"] == 0:
                run.log({ "loss": loss, }, step=step,)

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            target_net.load_state_dict(target_net_state_dict)

            if done:
                ep_len = step - start_step
                episode_durations.append(ep_len)
                start_step = step

                sum_reward = sum([memory.memory[i].reward.item() for i in range(-1, max(-ep_len-1, -len(memory.memory)), -1)])
                
                run.log({ "episode_length": ep_len, "sum_reward": sum_reward, "episode": len(episode_durations) - 1,})
                # Reset the environment
                initial_state, info = env.reset(seed=config["seed"])
                state: Optional[torch.Tensor] = torch.tensor(initial_state, dtype=torch.float32, device=device).unsqueeze(0)

            if step % 5_000 == 0 and step > 0:
                # Checkpoint the model
                torch.save(policy_net.state_dict(), f"out/{step}_policy.pt")
                torch.save(target_net.state_dict(), f"out/{step}_target.pt")
                # print(f"Sum Rew. Latest: {mean([m.reward.item() for m in memory.sample(100, clip=True)])}")
                # print(f"Ep. Len (10): {mean(episode_durations[-10:]):.3f}\n")

    return policy_net