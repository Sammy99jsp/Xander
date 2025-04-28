from typing import Any, Callable, Optional
import typing
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation

from pydantic import BaseModel, Field, PositiveInt
from wandb.sdk.wandb_run import Run
from wandb.integration.sb3 import WandbCallback
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import CallbackList, EveryNTimesteps, CheckpointCallback

from RL.algorithms._types import Agent
from RL.algorithms.ppo.eval_n_steps import EvalForNSteps
from RL.env2.wrappers import DiscreteInt

if typing.TYPE_CHECKING:
    from RL.env2.duel import XanderDuelEnv

class PPOHyperparameters(BaseModel):
    n_envs: PositiveInt = Field(default=4, description="""The number of environments to run in parallel""")
    learning_rate: float = Field(default=3e-4, description="""The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)""")
    n_steps: int = Field(default=2048, description="""The number of steps to run for each environment per update
        (i.e. rollout buffer size is n_steps * n_envs where n_envs is number of environment copies running in parallel)
        NOTE: n_steps * n_envs must be greater than 1 (because of the advantage normalization)
        See https://github.com/pytorch/pytorch/issues/29372""")
    batch_size: int = Field(default=64, description="""Minibatch size""")
    n_epochs: int = Field(default=10, description="""Number of epoch when optimizing the surrogate loss""")
    gamma: float = Field(default=0.99, description="""Discount factor""")
    gae_lambda: float = Field(default=0.95,description="""Factor for trade-off of bias vs variance for Generalized Advantage Estimator""")
    clip_range: float = Field(default=0.2,description="""Clipping parameter, it can be a function of the current progress
        remaining (from 1 to 0).""")
    clip_range_vf: Optional[float] = Field(default=None,description="""Clipping parameter for the value function,
        it can be a function of the current progress remaining (from 1 to 0).
        This is a parameter specific to the OpenAI implementation. If None is passed (default),
        no clipping will be done on the value function.
        IMPORTANT: this clipping depends on the reward scaling.""")
    normalize_advantage: bool = Field(default=True,description="""Whether to normalize or not the advantage""")
    ent_coef: float = Field(default=0.0,description="""Entropy coefficient for the loss calculation""")
    vf_coef: float = Field(default=0.5,description="""Value function coefficient for the loss calculation""")
    max_grad_norm: float = Field(default=0.5,description="""The maximum value for the gradient clipping""")
    use_sde: bool = Field(default=False,description="""Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)""")
    sde_sample_freq: int = Field(default=-1,description="""Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)""")
    # rollout_buffer_class: Optional[type[RolloutBuffer]] = Field(default=None,description="""""")
    # rollout_buffer_kwargs: Optional[dict[str, Any]] = Field(default=None,description="""""")
    target_kl: Optional[float] = Field(default=None,description="""Limit the KL divergence between updates,
        because the clipping is not enough to prevent large update
        see issue #213 (cf https://github.com/hill-a/stable-baselines/issues/213)
        By default, there is no limit on the kl div.""")
    stats_window_size: int = Field(default=100,description="""Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over""")
    # tensorboard_log: Optional[str] = Field(default=None,description="""""")
    policy_kwargs: Optional[dict[str, Any]] = Field(default=None,description="""additional arguments to be passed to the policy on creation.""")
    verbose: int = Field(default=0,description="""Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages""")
    # seed: Optional[int] = Field(default=None,description="""""")
    device: str = Field(default="auto",description="""Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.""")

class PPOAgent(Agent[PPOHyperparameters, VecEnv]):
    def __init__(self, env: "XanderDuelEnv", hyperparameters: PPOHyperparameters, run: Run):
        from RL.env2.duel import XanderDuelEnv
        # Inelegant :(
        cfg = env.config
        fn = lambda: Monitor(FlattenObservation(DiscreteInt(XanderDuelEnv(cfg.model_copy(deep=True)))))
        vec_env = make_vec_env(
            fn, 
            n_envs=hyperparameters.n_envs
        )

        self.callbacks = CallbackList([
            EveryNTimesteps(cfg.training.test_every, EvalForNSteps(fn, cfg.training.test_steps)),
            CheckpointCallback(save_freq=cfg.training.save_every, save_path=cfg.training.save_path, name_prefix="ppo"),
            WandbCallback(verbose=2)
        ])

        self._run = run

        del env
        tmp = hyperparameters.model_dump()
        tmp.pop("n_envs")

        self.model = PPO("MlpPolicy", vec_env, tensorboard_log=cfg.training.save_path, **tmp)

    @property
    def env(self) -> VecEnv:
        return self.model.get_env()
    
    @property
    def num_envs(self) -> int:
        return self.model.env.num_envs
    
    @property
    def run(self) -> Run:
        return self._run
    
    @property
    def is_training_divisible(self) -> bool:
        return False

    def train(self, steps: int):
        # Account for the number of environments
        self.model.learn(total_timesteps=steps * self.env.num_envs, callback=self.callbacks, progress_bar=True)

    def save(self, path: str):
        self.model.save(path)

    def load(self, path: str):
        self.model = PPO.load(path, env=self.env)

    def predict(self, obs: np.ndarray) -> np.ndarray:
        action, _ = self.model.predict(obs)
        return action
