# pyright: reportUnusedImport=false
import abc
from enum import Enum
import json
import random
import gymnasium as gym
import numpy as np
from pydantic import BaseModel, NonNegativeInt, PositiveInt, computed_field, create_model
from pydantic.fields import Field
import typing
from typing import ClassVar, Literal, Literal as L, Annotated, TypedDict
import functools
import wandb
import os

# These are imported for WandB Pydantic
from wandb.sdk.lib.paths import StrPath
from wandb.sdk import Settings
from typing import Sequence, Any

from RL.algorithms import ALGORITHMS
from RL.algorithms.ppo import PPOHyperparameters
from RL.algorithms.rainbow import RainbowHyperparameters
from RL.env.agents import CallbackAgent, RandomAgent
from RL.env.util import GRID_SQ, set_seed
import xander.engine as X
from xander.engine.actors import Stats # type: ignore
from xander.engine.combat.arena import Simple


type P3 = tuple[int, int, int]
type P2 = tuple[int, int]


@functools.cache
def wandb_config_model() -> type[BaseModel]:
    import inspect
    sig = inspect.signature(wandb.init)
    
    params = { param.name: (param.annotation, param.default) for param in sig.parameters.values() }
    return create_model("WandbConfig", **params) # type: ignore


GridInt = Annotated[int, Field(multiple_of=GRID_SQ, gt=0)]
WandbConfig = wandb_config_model()

class XanderRewards(BaseModel):
    """
    Configuration for the rewards.
    """

    win: float = Field(default=1.0, description="""Reward for winning.""")
    lose: float = Field(default=-1.0, description="""Reward for losing (dying).""")
    step: float = Field(default=-0.1, description="""Reward given for each step taken.""")
    illegal: float = Field(default=-0.5, description="""Reward for taking an illegal action.""")
    

class ArenaConfig(BaseModel):
    """
    Configuration for the arena.
    """
    width: GridInt
    height: GridInt

    def grid_size(self) -> tuple[int, int]:
        """
        Get the grid size of the arena.
        """
        return self.width // GRID_SQ, self.height // GRID_SQ


class CombatMemberConfig(BaseModel):
    type: str
    stats_file: str = Field(description="Path to the stats block (JSON file) for this combatant.")
    name: typing.Optional[str] = Field(default=None, description="Name for this combatant in the initiative.")


class CombatRandomMemberConfig(CombatMemberConfig):
    type: L["random"]


class CombatFixedStrategyMemberConfig(CombatMemberConfig):
    type: L["fixed_strategy"]
    strategy: str


class CombatLearnerMemberConfig(CombatMemberConfig):
    type: L["learner"]
    algorithm: str
    hyperparameters: dict[str, typing.Any] = Field(description="Hyperparameters for the algorithm.")
    pretrained_model_path: typing.Optional[str] = Field(default=None, description="Path to a pretrained model file.")
    train: bool = Field(description="Whether to train this model, or only use it for inference.")

class RainbowConfig(CombatLearnerMemberConfig):
    algorithm: L["rainbow"]
    hyperparameters: RainbowHyperparameters

class PPOConfig(CombatLearnerMemberConfig):
    algorithm: L["ppo"]
    hyperparameters: PPOHyperparameters


class TrainingConfig(BaseModel):
    max_steps: NonNegativeInt = Field(description="Maximum number of steps to train for. Set to 0 to train indefinitely.")
    test_every: NonNegativeInt = Field(default=10_000, description="Test the model every N steps.")
    save_every: NonNegativeInt = Field(default=10_000, description="Take a checkpoint every N steps.")
    test_steps: PositiveInt = Field(default=1_000, description="Number of steps to test for.")
    save_path: str = Field(description="Folder path to save the model to.")
    log_every: NonNegativeInt = Field(default=100, description="Log every N steps.")

class XanderEnvConfig(BaseModel, abc.ABC):
    """
    Configuration for the Xander environment.
    """
    type: str
    seed: NonNegativeInt = Field(description="Global Seed for the environment/training script.")
    arena: ArenaConfig = Field(description="Arena configuration.")
    rewards: XanderRewards = Field(description="Rewards configuration.")
    members: list[CombatRandomMemberConfig | CombatFixedStrategyMemberConfig | RainbowConfig | PPOConfig] = Field(description="Combatants in the arena.")
    training: TrainingConfig = Field(description="Training configuration.")
    wandb: WandbConfig = Field(description="Configuration for Weights & Biases.") # type: ignore

    def build(self, rebuild=False) -> tuple[X.combat.Combat, CallbackAgent]:
        """
        Build the Xander environment.
        """
        
        # Set the random seed
        if not rebuild:
            set_seed(self.seed)

        arena = Simple(self.arena.width, self.arena.height)
        combat = X.combat.Combat(arena)

        prev_positions: list[P2] = []
        
        learner = None
        agent: RandomAgent | CallbackAgent
        for i, member in enumerate(self.members):
            stats = Stats.from_json(member.stats_file)
            match member.type:
                case "random":
                    agent = RandomAgent()
                case "learner":
                    agent = CallbackAgent()
                    if not rebuild:
                        agent.algorithm = member.algorithm # type: ignore
                        agent.hyperparameters = member.hyperparameters # type: ignore
                    learner = agent
                case "fixed_strategy":
                    raise NotImplementedError
            agent.combatant = combat.join(
                stats, member.name or f"R{i}",
                (*random_position(self, prev_positions), 0),
                hook=agent.act
            )

        assert learner is not None, "No learner agent found in the configuration."

        return combat, learner

def random_position(cfg: XanderEnvConfig, prev: list[tuple[int, int]]) -> tuple[int, int]:
    def rand_pos():
        return random.randint(0, cfg.arena.width // GRID_SQ), random.randint(0, cfg.arena.height // GRID_SQ)
   
    x, y = rand_pos()

    while (x, y) in prev:
        x, y = rand_pos()

    prev.append((x, y))

    return x, y
