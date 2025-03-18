from dataclasses import dataclass
from enum import Enum
import json
from random import randrange
from typing import Callable, ClassVar, Literal, Optional, TypedDict
import typing
import gymnasium as gym
import numpy as np

from torch import nn
import torch

from argparse import ArgumentParser

from tests.python.rl.utils import XanderConfig
from xander.engine.combat import Combat, Combatant
from xander.engine import dice
from xander.engine.actors import Stats
from xander.engine.combat.action.attack import Attack
from xander.engine.combat.arena import Simple as Arena
from xander.engine.combat.turn import Turn
from xander.engine.legality import Legality


GRID_LENGTH = 5



class RandomAgent:
    """
    An agent that randomly selects either a move or attack.
    This should only act legally.
    """

    combatant: Combatant
    attacks: list[Attack]

    def __init__(self, combatant: Combatant):
        self.combatant = combatant
        self.attacks = [tmp_attack for action in combatant.stats.actions if (tmp_attack := action.as_attack()) is not None]

    def hook(self, turn: Turn) -> "Legality[typing.Any]":
        # This shouldn't really occur, but there's a bug somewhere.
        if self.combatant.stats.dead:
            return turn.end()
        

        # TODO: Add attacking.
        movement_directions: list[tuple[float, float, float]] = [] if (tmp := turn.movement_directions()).is_illegal() else tmp.inner() # type: ignore
        l_attack = turn.attack_directions(self.attacks[0], lambda x: not x.stats.dead)

        attack_directions: list[tuple[float, float, float]] = []
        if l_attack.is_legal():
            attack_directions = l_attack.inner() # type: ignore
        
        opt = randrange(len(movement_directions) + len(attack_directions) + 1)

        l: "Legality[typing.Any]"

        if opt < len(movement_directions):
            l = turn.move(movement_directions[opt])
        elif opt < len(movement_directions) + len(attack_directions):
            dir = attack_directions[opt - len(movement_directions)]
            l = turn.attack(self.attacks[0], dir)
        else:
            l = turn.end()

        # This branch ideally shouldn't ideally be taken.
        if l.is_illegal():
            return turn.end()

        return l


P3 = tuple[float, float, float]
"""Position in 3D space."""

def random_position(cfg: XanderConfig) -> P3:
    return (randrange(start=0, stop=cfg["arena_width"], step=GRID_LENGTH), randrange(0, stop=cfg["arena_height"], step=GRID_LENGTH), 0)

class CallbackAgent:
    _combatant: Combatant
    _attacks: list[Attack]
    hook: Callable[[Turn], None]

    def __init__(self, hook: Callable[[Turn], None]):
        self.hook = hook

    def assign_combatant(self, combatant: Combatant):
        self._combatant = combatant
        self._attacks = [tmp for action in combatant.stats.actions if (tmp := action.as_attack()) is not None]

Step = np.ndarray
"""
A step for the agent to take.

[0:1]: 0 to end, 1 to move, 2 to attack
[1:9]: one-hot encoding for target square (cardinal + intercardinal directions, starting clockwise from North) (N-NE-E-...-NW).
"""

@dataclass
class XanderObs:
    LEN: ClassVar[int] = 20

    hp: int
    max_hp: int
    movement_left: int
    attacks_left: int
    move_dirs: torch.Tensor
    attack_dirs: torch.Tensor
    # observation: Observation

    def encode(self) -> torch.Tensor:
        return torch.concatenate([
            torch.tensor([self.hp, self.max_hp, self.movement_left, self.attacks_left]),
            self.move_dirs,
            self.attack_dirs,
        ])


@dataclass
class XanderTurn:
    TYPES: ClassVar[list[str]] = ["end", "move", "attack"]
    TYPES_LEN: ClassVar[int] = len(TYPES)
    HOT_POSITION_LEN: ClassVar[int] = 8
    DIRECTIONS: ClassVar[list[P3]] = [(0, GRID_LENGTH, 0), (GRID_LENGTH, GRID_LENGTH, 0), (GRID_LENGTH, 0, 0), (GRID_LENGTH, -GRID_LENGTH, 0), (0, -GRID_LENGTH, 0), (-GRID_LENGTH, -GRID_LENGTH, 0), (-GRID_LENGTH, 0, 0), (-GRID_LENGTH, GRID_LENGTH, 0)]

    type: Literal["end"] | Literal["move"] | Literal["attack"]
    direction: Optional[P3]

    @staticmethod
    def decode(step_type: torch.Tensor, position: torch.Tensor) -> "XanderTurn":
        step_type = step_type.argmax()
        position = position.argmax()

        match XanderTurn.TYPES[step_type]:
            case "end":
                return XanderTurn(type="end", direction=None)
            case "move":
                return XanderTurn(type="move", direction=XanderTurn.DIRECTIONS[position])
            case "attack":
                return XanderTurn(type="attack", direction=XanderTurn.DIRECTIONS[position])
        
        assert False

    @staticmethod
    def end():
        return XanderTurn(type="end", direction=None)

def random_init(m: nn.Module):
    if isinstance(m, nn.Linear):
        torch.nn.init.normal_(m.weight, mean=0, std=1)
        torch.nn.init.zeros_(m.bias)

class XanderNet(nn.Module):
    main_pass: nn.Sequential

    action: nn.Sequential
    position: nn.Sequential

    def __init__(self):
        super(XanderNet, self).__init__()
        self.main_pass = nn.Sequential(
            nn.Linear(XanderObs.LEN, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, XanderTurn.HOT_POSITION_LEN + XanderTurn.TYPES_LEN),
        )

        self.action = nn.Sequential(
            nn.Linear(11, XanderTurn.TYPES_LEN),
            nn.Softmax(),
        )

        self.position = nn.Sequential(
            nn.Linear(11, XanderTurn.HOT_POSITION_LEN),
            nn.Softmax(),
        )

    def init_weights(self):
        self.apply(self.main_pass, random_init)
        self.apply(self.action, random_init)
        self.apply(self.position, random_init)

    def forward(self, obs: XanderObs) -> XanderTurn:
        inp = obs.encode()

        main = self.main_pass(inp)
        action = self.action(main)
        position = self.position(main)

        return XanderTurn.decode(action, position)


Timestep = tuple[XanderObs, float, bool, dict]

class TerminalCondition(Enum):
    NONE = 0
    DEAD = 1
    WIN = 2

    def is_terminal(self) -> bool:
        return self != TerminalCondition.NONE
    
    def reward(self) -> float:
        match self:
            case TerminalCondition.DEAD:
                return -10
            case TerminalCondition.WIN:
                return 10
            case TerminalCondition.NONE:
                return 0

type Config = typing.Any

class Duel(gym.Env):
    combat: Combat
    turn: Turn

    len_adversaries: int
    
    _cfg: Config
    _cb: CallbackAgent

    _reward: float = 0.0

    def __init__(self, cfg: Config):
        # super().__init__(cfg)
        self._cfg = cfg
        self._cb = CallbackAgent(lambda t: self._xander_turn(t))
        self._reward = 0
        self.len_adversaries = len(cfg["adversaries"])
        np.random.seed(cfg["seed"])
        dice.set_seed(cfg["seed"])

        self.combat, _ = make_combat(self._cfg, self._cb)
    


    @property
    def combatant(self) -> Combatant:
        return self._cb._combatant
    
    @property
    def attacks(self) -> list[Attack]:
        return self._cb._attacks

    def seed(self, seed):
        np.random.seed(seed)
        dice.set_seed(seed)

    def reset(self) -> Timestep:
        del self.combat
        self.combat, self._cb = make_combat(self._cfg, self._cb)
        self._reward = 0.0

        return self._my_next_turn()
    
    def __repr__(self):
        return "<Duel>"

    def close(self):
        del self.turn
        del self._cb
        del self.combat

    def step(self, action: XanderTurn) -> Timestep:
        l : "Legality[typing.Any]"
        match action.type:
            case "end":
                l = self.turn.end()
            case "move":
                l = self.turn.move(action.direction) # type: ignore
            case "attack":
                l = self.turn.attack(self.attacks[0], action.direction) # type: ignore
        # TODO: Do something else (like rewarding for damage dealt).

        self._reward = 0
        if l.is_illegal():
            self._reward = -1
        
        return self._my_next_turn()

    def _obs(self) -> XanderObs:
        return XanderObs(
            move_dirs=torch.tensor(self.turn.movement_directions_one_hot()),
            attack_dirs=torch.tensor(self.turn.attack_directions_one_hot(self.attacks[0], filter=lambda x: not x.stats.dead)),
            hp=self.combatant.stats.hp,
            max_hp=self.combatant.stats.max_hp,
            movement_left=self.turn.movement_left(),
            attacks_left=self.turn.actions_left,
        )

    def _terminal_condition(self) -> TerminalCondition:
        if self.combat.current.stats.dead:
            return TerminalCondition.DEAD

        if sum([1 if c.stats.dead else 0 for c in self.combat.combatants]) >= (self.len_adversaries - 1):
            return TerminalCondition.WIN
        
        return TerminalCondition.NONE
    

    def _my_next_turn(self) -> Timestep:
        while True:
            self.combat.step()
            if self.combat.current.name == self.combatant.name or self._terminal_condition():
                break

        t = self._terminal_condition()
        obs: Timestep = (self._obs(), self._reward + t.reward(), t.is_terminal(), {})
        self._reward = 0.0

        return obs

    def _xander_turn(self, turn: Turn):
        # Renew the turn.
        self.turn = turn
        
        pass


def make_combat(cfg: Config, cb: CallbackAgent) -> tuple[Combat, CallbackAgent]:
    combat = Combat(Arena(width=cfg["arena_width"], height=cfg["arena_height"]))
    
    positions: list[P3] = []

    pos = random_position(cfg)
    positions.append(pos)
    cb.assign_combatant(combat.join(Stats.from_json(cfg["agent_stats"]), "Agent", pos))
    combat.set_combatant_hook(cb._combatant, cb.hook)

    for i, adversary in enumerate(cfg["adversaries"]):
        match adversary:
            case "random":

                pos = random_position(cfg)
                while pos in positions:
                    pos = random_position(cfg)

                c = combat.join(Stats.from_json(adversary["stats_file"]), f"RA{i+1}", pos)
                combat.set_combatant_hook(c, RandomAgent(c).hook)
                positions.append(pos)

    return combat, cb

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to the combat config file.")

    args = parser.parse_args()
    # Read config from file.
    cfg: Config = json.load(open(args.config))
    
    env = Duel(cfg)

    model = XanderNet()

    step = env.reset()
    for i in range(10):
        obs, reward, done, info = step
        a = model(obs)
        print(a)

        step = env.step(XanderTurn(type="move", direction=(0, GRID_LENGTH, 0)))
# D4RN