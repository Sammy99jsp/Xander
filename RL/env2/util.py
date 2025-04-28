# pyright: reportUnusedImport=false
from enum import Enum
import random
import gymnasium as gym
import numpy as np
import typing
from typing import ClassVar, Literal as L, Annotated, TypedDict
import torch


from RL.env2.space import OneHot

import xander.engine as X
from xander.engine.combat import Combatant
from xander.engine.combat.turn import Turn
from xander.engine.combat.action.attack import Attack
import xander.engine.dice as D
from xander.engine.actors import Stats
from xander.engine.combat.arena import Simple

if typing.TYPE_CHECKING:
    from RL.env2.config import XanderEnvConfig


def set_seed(seed: int) -> None:
    D.set_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


GRID_SQ = 5
DIRECTIONS = [(0.0, GRID_SQ, 0.0), (GRID_SQ, GRID_SQ, 0.0), (GRID_SQ, 0.0, 0.0), (GRID_SQ, -GRID_SQ, 0.0), (0.0, -GRID_SQ, 0.0), (-GRID_SQ, -GRID_SQ, 0.0), (-GRID_SQ, 0.0, 0.0), (-GRID_SQ, GRID_SQ, 0.0)]

ActionParamType = L["none", "reach", "melee", "ranged"]


def actions_available(combat: X.combat.Combat, cfg: "XanderEnvConfig", name: str) -> dict[str, ActionParamType]:
    agent_name = next((m.name for m in cfg.members if m.name == name), None)
    assert agent_name is not None # (1.) `name` should be a member of the combat.
    agent = next((c.stats for c in combat.combatants if c.name == agent_name), None)
    assert agent is not None # Agent should be a combatant!

    actions = {
        "end": "none",
        "move": "reach",
    }

    actions.update({
        f"attack.{attack.name}": attack.type for action in agent.actions \
            if (attack := action.as_attack()) is not None
    })

    return actions # type: ignore


class XanderObs(TypedDict):
    hp_percent: np.ndarray
    speed_percent: np.ndarray
    actions_left_percent: np.ndarray

    movement_directions_one_hot: np.ndarray
    """Available Move directions (one-hot encoded)"""

    surroundings: np.ndarray
    """Surrounding squares (0 for empty, 1 for occupied)"""

def observation_space(cfg: "XanderEnvConfig") -> gym.Space[XanderObs]:
    return gym.spaces.Dict({
        "hp_percent": gym.spaces.Box(low=0.0, high=1.0, shape=(), dtype=np.float32),
        "speed_percent": gym.spaces.Box(low=0.0, high=1.0, shape=(), dtype=np.float32),
        "actions_left_percent": gym.spaces.Box(low=0.0, high=1.0, shape=(), dtype=np.float32),
        "movement_directions_one_hot":  gym.spaces.MultiBinary(n=(8,)),
        "surroundings": gym.spaces.MultiBinary(n=(cfg.arena.width // GRID_SQ, cfg.arena.height // GRID_SQ)),
    })

def observe(cfg: "XanderEnvConfig", combatant: Combatant, turn: Turn) -> XanderObs:
    return XanderObs(
        hp_percent=np.array(combatant.stats.hp / combatant.stats.max_hp, dtype=np.float32),
        speed_percent=np.array(turn.movement_left() / combatant.stats.speed(), dtype=np.float32),
        actions_left_percent=np.array(turn.actions_left / turn.max_actions, dtype=np.float32),
        movement_directions_one_hot=np.array(turn.movement_directions_one_hot(), dtype=np.int8),
        surroundings=np.array(combatant.observe(), dtype=np.int8).reshape(cfg.arena.grid_size()),
    )
    # return np.concatenate((
    #     np.array([
    #         combatant.stats.hp / combatant.stats.max_hp,
    #         turn.movement_left() / combatant.stats.speed(),
    #         turn.actions_left / turn.max_actions
    #     ]),
    #     np.array(turn.movement_directions_one_hot()),
    #     np.array(turn.attack_directions_one_hot(combatant.stats.actions[0].as_attack(), filter=lambda c: not c.stats.dead)),
    #     np.array(combatant.observe())
    # ), dtype=np.float32)

type CompassDirection = int # L[0, 1, 2, 3, 4, 5, 6, 7] (Clockwise from North)
type TargetSquare = tuple[float, float, float]


class ActionType(Enum):
    END = 0
    MOVE = 1
    ATTACK = 2

class XanderAction:
    pass


class EndAction(XanderAction):
    type = ActionType.END

    def __repr__(self):
        return "EndAction()"


class MoveAction(XanderAction):
    type = ActionType.MOVE
    direction: tuple[float, float, float]

    __match_args__ = ("direction", )
    def __init__(self, direction: CompassDirection):
        self.direction = DIRECTIONS[direction]

    def __repr__(self):
        return f"MoveAction({self.direction})"


class AttackAction(XanderAction):
    __match_args__ = ("attack", "target", )
    type = ActionType.ATTACK
    attack: Attack
    target: tuple[float, float, float]

    def __init__(self, attack: Attack, target: TargetSquare):
        self.attack = attack
        self.target = target

    def __repr__(self):
        return f"AttackAction({self.attack}, {self.target})"


def pos_add(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])
