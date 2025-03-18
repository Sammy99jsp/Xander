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


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

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
    return sum(l) / len(l)

def train(env: gym.Env[np.ndarray, np.ndarray], device: torch.device) -> DQN:
    # BATCH_SIZE is the number of transitions sampled from the replay buffer
    # GAMMA is the discount factor as mentioned in the previous section
    # EPS_START is the starting value of epsilon
    # EPS_END is the final value of epsilon
    # EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
    # TAU is the update rate of the target network
    # LR is the learning rate of the ``AdamW`` optimizer
    BATCH_SIZE = 128
    GAMMA = 0.99
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 1000
    TAU = 0.005
    LR = 1e-4

    n_actions = 8
    n_dirs = 3

    # Get the number of state observations
    state, info = env.reset()
    n_observations = len(state)

    policy_net = DQN(n_observations, n_actions, n_dirs).to(device)
    target_net = DQN(n_observations, n_actions, n_dirs).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(10_000)

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
            return select_action_zero(torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.float32))


    episode_durations = []

    def optimize_model():
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
        tmp: torch.Tensor = policy_net(state_batch)
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
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
        optimizer.step()

    if torch.cuda.is_available() or torch.backends.mps.is_available():
        num_episodes = 600
    else:
        num_episodes = 50

    for i_episode in tqdm.tqdm(range(num_episodes), unit="ep"):
        # Initialize the environment and get its state
        initial_state, info = env.reset()
        state: Optional[torch.Tensor] = torch.tensor(initial_state, dtype=torch.float32, device=device).unsqueeze(0)
        for t in count():
            action = select_action(state)
            observation, reward, terminated, truncated, _ = env.step(action)
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            next_state: Optional[torch.Tensor] = None
            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            optimize_model()

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            target_net.load_state_dict(target_net_state_dict)

            if done:
                episode_durations.append(t + 1)
                print(f"Sum Rew. Latest: {mean([m.reward.item() for m in memory.sample(100, clip=True)])}")
                print(f"Ep. Len (10): {mean(episode_durations[-10:]):.3f}")
                # plot_durations()

                if i_episode % 50 == 0 and i_episode > 0:
                    # Checkpoint the model
                    torch.save(policy_net.state_dict(), f"out/{i_episode}_policy.pt")
                    torch.save(target_net.state_dict(), f"out/{i_episode}_target.pt")


                break

    return policy_net