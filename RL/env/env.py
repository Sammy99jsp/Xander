import abc
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Literal, Optional, TypedDict, List, Callable, cast
import json
import random
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from RL.env.utils import DELTA, M
from xander.engine import dice
from xander.engine.combat.arena import Simple
from xander.engine.combat.turn import Turn
from xander.engine.actors import Stats
from xander.engine.combat import Combatant, Combat

from .agents import CallbackAgent, RandomAgent


class AdversaryConfig(TypedDict):
    stats_file: str
    type: str


class DuelConfig(TypedDict):
    seed: int
    arena_width: int
    arena_height: int
    max_steps: int
    agent_stats: str
    adversaries: List[AdversaryConfig]
    hyperparameters: dict[str, Any]
    wandb: dict[str, Any]
    
    
def random_position(cfg: DuelConfig, prev: list[tuple[int, int]]) -> tuple[int, int]:
    def rand_pos():
        return random.randint(0, cfg["arena_width"] // M), random.randint(0, cfg["arena_height"] // M)
   
    x, y = rand_pos()

    while (x, y) in prev:
        x, y = rand_pos()

    prev.append((x, y))

    return x, y


def parse_config(path: str) -> DuelConfig:
    with open(Path(path), 'r') as f:
        config = json.load(f)
    
    return config

def observation_size(cfg: DuelConfig) -> int:
    w, h = cfg["arena_width"], cfg["arena_height"]
    return 3 + 8 + 8 + (w // M) * (h // M)

def make_combat(cfg: DuelConfig) -> tuple[Combat, CallbackAgent]:
    """Create a Combat instance from a configuration.
    
    Args:
        config: The duel configuration
        
    Returns:
        A Combat instance with all combatants added
    """
    # Set random seed for reproducibility
    random.seed(cfg["seed"])
    
    # Create the arena
    arena = Simple(cfg["arena_width"], cfg["arena_height"])
    
    # Create the combat
    combat = Combat(arena)

    prev_positions: list[tuple[int, int]] = []
    
    agent_stats = Stats.from_json(cfg["agent_stats"])
    agent = CallbackAgent()
    agent.combatant = combat.join(agent_stats, "Agent", (*random_position(cfg, prev_positions), 0), hook=agent.act)
    
    # Add adversaries with random positions
    for i, adv_config in enumerate(cfg["adversaries"]):
        # Load adversary stats
        adv_stats = Stats.from_json(adv_config["stats_file"])
        
        match adv_config["type"]:
            case "random":
                adv = RandomAgent()
            case _:
                # Default to random if type is unknown
                adv = RandomAgent()

        adv.combatant = combat.join(adv_stats, f"Adversary {i+1}", (*random_position(cfg, prev_positions), 0), adv.act)
        
    
    return combat, agent


class ActionType(Enum):
    END = 0
    MOVE = 1
    ATTACK = 2


type TargetSquare = int # Literal[0, 1, 2, 3, 4, 5, 6, 7, 8]


class XanderAction(abc.ABC):
    """An action in the Xander environment."""

    type: ClassVar[ActionType]

    @staticmethod
    def decode(r_action: np.ndarray) -> "XanderAction":
        """
        Args:
            action: The action tensor
                    - [0:3] action type
                    - [3:] target square (one-hot encoded)
        """

        action = r_action.flatten()
        assert action.shape == (3 + 8, ), action
        
        match np.argmax(action[:3]):
            case ActionType.MOVE.value:
                return MoveAction(np.argmax(action[3:]).item()) # type: ignore
            case ActionType.ATTACK.value:
                return AttackAction(np.argmax(action[3:]).item()) # type: ignore
            case ActionType.END.value | _:
                return EndAction()

# N, NE, E, SE, S, SW, W, NW

class EndAction(XanderAction):
    type = ActionType.END

    def __repr__(self):
        return "EndAction()"


class MoveAction(XanderAction):
    type = ActionType.MOVE
    delta: tuple[float, float, float]

    __match_args__ = ("target", )
    def __init__(self, target: TargetSquare):
        self.delta = DELTA[target]

    def __repr__(self):
        return f"MoveAction({self.delta})"


class AttackAction(XanderAction):
    __match_args__ = ("target", )
    type = ActionType.ATTACK
    target: tuple[float, float, float]

    def __init__(self, target: TargetSquare):
        self.target = DELTA[target]

    def __repr__(self):
        return f"AttackAction({self.target})"


class XanderObs:
    """Observation space for the Xander environment.
    
    Parts:
        - Agent Stats ([hp_left %, speed_left %, actions left %])
        - Available Move directions (one-hot encoded)
        - Available Attack directions (one-hot encoded)
        - Surrounding squares (0 for empty, 1 for occupied)
    """
    
    @staticmethod
    def observe(combatant: Combatant, turn: Turn) -> np.ndarray:
        return np.concatenate((
            np.array([
                combatant.stats.hp / combatant.stats.max_hp,
                turn.movement_left() / combatant.stats.speed(),
                turn.actions_left / turn.max_actions
            ]),
            np.array(turn.movement_directions_one_hot()),
            np.array(turn.attack_directions_one_hot(combatant.stats.actions[0].as_attack(), filter=lambda c: not c.stats.dead)),
            np.array(combatant.observe())
        ), dtype=np.float32)


ILLEGAL_REW = -0.5
SURVIVE_REW = -0.01 # Small, encourage it to be quick.
WIN_REW = 1.0
LOSE_REW = -1.0


class Duel(gym.Env[np.ndarray, np.ndarray]):
    def __init__(self, cfg: DuelConfig):
        dice.set_seed(cfg["seed"])
        self.cfg = cfg
        self.combat, self.agent = make_combat(self.cfg)

        self.action_space = gym.spaces.Box(
            shape=(3 + 8,), dtype=np.float32, low=0.0, high=1.0
        )

        self.observation_space = gym.spaces.Box(
            shape=(observation_size(cfg),), dtype=np.float32, low=0.0, high=1.0
        )

        self.max_steps = cfg["max_steps"]
    
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            dice.set_seed(seed)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        while True:
            self.combat, self.agent = make_combat(self.cfg)
            while True:
                payload = self._fast_forward()
                match payload:
                    case "lost":
                        # self.cfg["seed"] += 1
                        break
                    case (combatant, turn):
                        return XanderObs.observe(combatant, turn), {}
        

    def step(self, raw_action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        payload = self._fast_forward()

        match payload:
            case "timeout" | "won" | "lost":
                raise NotImplementedError("Shouldn't happen. Reset the environment!")
            case _:
                pass
        
        combatant, turn = payload
        
        action = XanderAction.decode(raw_action)
        rew = 0.0
        match action:
            case EndAction():
                turn.end()
            case MoveAction(target):
                l = turn.move(target)

                if not l.is_legal():
                    rew += ILLEGAL_REW

            case AttackAction(target):
                l = turn.attack(self.agent.combatant.stats.actions[0].as_attack(), target)

                if not l.is_legal():
                    rew += ILLEGAL_REW


        obs = XanderObs.observe(combatant, turn)
        payload = self._fast_forward()

        match payload:
            case "timeout":
                return obs, rew, False, True, {}
            case "won":
                rew += WIN_REW
                return obs, rew, True, False, {}
            case "lost":
                rew += LOSE_REW
                return obs, rew, True, False, {}
            case _:
                rew += SURVIVE_REW
                return obs, rew, False, False, {}


    def _fast_forward(self) -> tuple[Combatant, Turn] | Literal["won", "lost"]:
        while True:
            with self.agent as payload:
                if self.agent.combatant.stats.dead:
                    return "lost" 
                if all(
                    c.stats.dead for c in self.combat.combatants \
                        if c.name != self.agent.combatant.name
                ):
                    return "won" # terminate the episode
                if payload is not None:
                    return payload
                
                self.combat.step()
  