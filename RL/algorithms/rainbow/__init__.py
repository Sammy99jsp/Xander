from typing import Annotated
import typing
import gymnasium as gym
import numpy as np
from pydantic import BaseModel, Field
import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import tqdm # type: ignore

from pydantic.types import PositiveInt
from wandb.sdk.wandb_run import Run

from RL.algorithms._types import Agent
from RL.algorithms.rainbow.network import Network
from RL.algorithms.rainbow.prioritised_replay_buffer import PrioritizedReplayBuffer
from RL.algorithms.rainbow.replay_buffer import ReplayBuffer
from RL.env2.wrappers import DiscreteOneHot

if typing.TYPE_CHECKING:
    from RL.env2.duel import XanderDuelEnv

BetweenZeroAndOne = Annotated[float, Field(ge=0.0, le=1.0)]
PositiveFloat = Annotated[float, Field(gt=0.0)]

class RainbowHyperparameters(BaseModel):
    memory_size: PositiveInt = Field(default=100_000, description="length of memory")
    batch_size: PositiveInt = Field(default=32, description="batch size for sampling")
    target_update: PositiveInt = Field(default=1000, description="period for target model's hard update")
    gamma: BetweenZeroAndOne = Field(default=0.99, description="discount factor")
    # PER parameters
    alpha: PositiveFloat = Field(default=0.2, description="determines how much prioritization is used")
    beta: PositiveFloat = Field(default=0.6, description="determines how much importance sampling is used")
    prior_eps: float = Field(default=1e-6, description="guarantees every transition can be sampled")
    # Categorical DQN parameters
    v_min: float = Field(default=-10.0, description="min value of support")
    v_max: float = Field(default=20.0, description="max value of support")
    atom_size: PositiveInt = Field(default=51, description="the unit number of support")
    # N-step Learning
    n_step: PositiveInt = Field(default=3, description="step number to calculate n-step td error")
    device: str = Field(default="cuda", description="PyTorch device to use for the model")


class RainbowAgent(Agent[RainbowHyperparameters, gym.wrappers.FlattenObservation]):
    """Rainbow Agent interacting with environment.
    
    Attribute:
        env (gym.Env): openAI Gym environment
        memory (PrioritizedReplayBuffer): replay memory to store transitions

        batch_size (int): batch size for sampling
        target_update (int): period for target model's hard update
        gamma (float): discount factor
        dqn (Network): model to train and select actions
        dqn_target (Network): target model to update
        optimizer (torch.optim): optimizer for training dqn
        transition (list): transition information including 
                           state, action, reward, next_state, done
        v_min (float): min value of support
        v_max (float): max value of support
        atom_size (int): the unit number of support
        support (torch.Tensor): support for categorical dqn
        use_n_step (bool): whether to use n_step memory
        n_step (int): step number to calculate n-step td error
        memory_n (ReplayBuffer): n-step replay buffer
    """

    _env: gym.wrappers.FlattenObservation


    def __init__(
        self, 
        env: "XanderDuelEnv",
        H: RainbowHyperparameters,
        run: Run,
        /,
    ):
        """Initialization.
        
        Args:
            env (gym.Env): openAI Gym environment
            H (RainbowHyperparameters): hyperparameters for Rainbow agent
        """
        
        self._run = run

        self._env = gym.wrappers.FlattenObservation(DiscreteOneHot(env))
        obs_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.n

        self.cfg = env.config

        self.batch_size = H.batch_size
        self.target_update = H.target_update
        self.seed = env.config.seed
        self.gamma = H.gamma
        # NoisyNet: All attributes related to epsilon are removed
        
        # device: cpu / gpu
        self.device = torch.device(H.device)
        
        # PER
        # memory for 1-step Learning
        self.beta = H.beta
        self.prior_eps = H.prior_eps
        self.memory = PrioritizedReplayBuffer(
            obs_dim, H.memory_size, H.batch_size, alpha=H.alpha, gamma=H.gamma
        )
        
        # memory for N-step Learning
        self.use_n_step = True if H.n_step > 1 else False
        if self.use_n_step:
            self.n_step = H.n_step
            self.memory_n = ReplayBuffer(
                obs_dim, H.memory_size, H.batch_size, n_step=H.n_step, gamma=H.gamma
            )
            
        # Categorical DQN parameters
        self.v_min = H.v_min
        self.v_max = H.v_max
        self.atom_size = H.atom_size
        self.support = torch.linspace(
            self.v_min, self.v_max, self.atom_size
        ).to(self.device)

        # networks: dqn, dqn_target
        self.dqn = Network(
            obs_dim, action_dim, self.atom_size, self.support
        ).to(self.device)
        self.dqn_target = Network(
            obs_dim, action_dim, self.atom_size, self.support
        ).to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()
        
        # optimizer
        self.optimizer = optim.Adam(self.dqn.parameters())

        # transition to store in memory
        self.transition = list()
        
        # mode: train / test
        self.is_test = False

    @property
    def env(self) -> gym.wrappers.FlattenObservation:
        """Return the environment."""
        return self._env
    
    @property
    def run(self) -> Run:
        return self._run
    
    @property
    def num_envs(self) -> int:
        return 1
    
    @property
    def is_training_divisible(self) -> bool:
        return True

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        # NoisyNet: no epsilon greedy action selection
        selected_action = self.dqn(
            torch.FloatTensor(state).to(self.device)
        ).argmax()
        selected_action = selected_action.detach().cpu().numpy()
        
        if not self.is_test:
            self.transition = [state, selected_action]
        
        return selected_action

    def step(self, action: np.ndarray) -> tuple[np.ndarray, np.float64, bool, dict[str, typing.Any]]:
        """Take an action and return the response of the env."""
        next_state, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        
        if not self.is_test:
            self.transition += [reward, next_state, done]
            
            # N-step transition
            if self.use_n_step:
                one_step_transition = self.memory_n.store(*self.transition)
            # 1-step transition
            else:
                one_step_transition = self.transition

            # add a single step transition
            if one_step_transition:
                self.memory.store(*one_step_transition)
    
        return next_state, reward, done, info

    def update_model(self) -> torch.Tensor:
        """Update the model by gradient descent."""
        # PER needs beta to calculate weights
        samples = self.memory.sample_batch(self.beta)
        weights = torch.FloatTensor(
            samples["weights"].reshape(-1, 1)
        ).to(self.device)
        indices = samples["indices"]
        
        # 1-step Learning loss
        elementwise_loss = self._compute_dqn_loss(samples, self.gamma)
        
        # PER: importance sampling before average
        loss = torch.mean(elementwise_loss * weights)
        
        # N-step Learning loss
        # we are gonna combine 1-step loss and n-step loss so as to
        # prevent high-variance. The original rainbow employs n-step loss only.
        if self.use_n_step:
            gamma = self.gamma ** self.n_step
            samples = self.memory_n.sample_batch_from_idxs(indices)
            elementwise_loss_n_loss = self._compute_dqn_loss(samples, gamma)
            elementwise_loss += elementwise_loss_n_loss
            
            # PER: importance sampling before average
            loss = torch.mean(elementwise_loss * weights)

        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.dqn.parameters(), 10.0)
        self.optimizer.step()
        
        # PER: update priorities
        loss_for_prior = elementwise_loss.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.prior_eps
        self.memory.update_priorities(indices, new_priorities)
        
        # NoisyNet: reset noise
        self.dqn.reset_noise()
        self.dqn_target.reset_noise()

        return loss.item()
        
    def train(self, steps: int):
        """Train the agent."""
        self.is_test = False
        
        state, _ = self.env.reset(seed=self.seed)
        update_cnt = 0
        scores = np.zeros(steps)
        score = 0

        for step in range(steps):
            action = self.select_action(state)
            next_state, reward, done, info = self.step(action)

            state = next_state
            scores[step] = reward
            
            # NoisyNet: removed decrease of epsilon
            
            # PER: increase beta
            fraction = min(step / steps, 1.0)
            self.beta = self.beta + fraction * (1.0 - self.beta)

            # if episode ends
            if done:
                state, _ = self.env.reset(seed=self.seed)

            # if training is ready
            if len(self.memory) >= self.batch_size:
                loss = self.update_model()
                self._run.log({ "loss": loss })

                update_cnt += 1
                
                # if hard update is needed
                if update_cnt % self.target_update == 0:
                    self._target_hard_update()

        # log the average sum of rewards
        self._run.log({"sum_ep_rew": np.mean(scores)})
    

    def _compute_dqn_loss(self, samples: dict[str, np.ndarray], gamma: float) -> torch.Tensor:
        """Return categorical dqn loss."""
        device = self.device  # for shortening the following lines
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.LongTensor(samples["acts"]).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)
        
        # Categorical DQN algorithm
        delta_z = float(self.v_max - self.v_min) / (self.atom_size - 1)

        with torch.no_grad():
            # Double DQN
            next_action = self.dqn(next_state).argmax(1)
            next_dist = self.dqn_target.dist(next_state)
            next_dist = next_dist[range(self.batch_size), next_action]

            t_z = reward + (1 - done) * gamma * self.support
            t_z = t_z.clamp(min=self.v_min, max=self.v_max)
            b = (t_z - self.v_min) / delta_z
            l = b.floor().long()
            u = b.floor().long() + 1

            offset = (
                torch.linspace(
                    0, (self.batch_size - 1) * self.atom_size, self.batch_size
                ).long()
                .unsqueeze(1)
                .expand(self.batch_size, self.atom_size)
                .to(self.device)
            )

            proj_dist = torch.zeros(next_dist.size(), device=self.device)
            proj_dist.view(-1).index_add_(
                0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
            )
            proj_dist.view(-1).index_add_(
                0, (u.clamp(max=self.atom_size - 1) + offset).view(-1), (next_dist * (b - l.float())).view(-1)
            )

        dist = self.dqn.dist(state)
        log_p = torch.log(dist[range(self.batch_size), action])
        elementwise_loss = -(proj_dist * log_p).sum(1)

        return elementwise_loss

    def _target_hard_update(self):
        """Hard update: target <- local."""
        self.dqn_target.load_state_dict(self.dqn.state_dict())

    def predict(self, obs: np.ndarray) -> np.ndarray:
        return self.select_action(obs)

    def load(self, path: str):
        """Load model parameters."""
        checkpoint = torch.load(path)
        self.dqn.load_state_dict(checkpoint["model"])
        self.dqn_target.load_state_dict(checkpoint["target"])
    
    def save(self, path: str):
        """Save model parameters."""
        torch.save({
            "model": self.dqn.state_dict(),
            "target": self.dqn_target.state_dict(),
        }, path)