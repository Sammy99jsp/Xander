from typing import Callable
import typing
import gymnasium as gym
import numpy as np

import RL.env.space as XS
from RL.env.util import AttackAction, EndAction, MoveAction, XanderAction, XanderObs, DIRECTIONS
from xander.engine.combat.action.attack import Attack

if typing.TYPE_CHECKING:
    from RL.env.duel import XanderDuelEnv


def for_attack(attack: Attack, arena_size: tuple[int, int]) -> tuple[Callable[[int], AttackAction], int]:
    match attack.type:
        case "melee":
            return lambda dir: AttackAction(attack, DIRECTIONS[dir]), len(DIRECTIONS)
        case "ranged":
            return lambda i: AttackAction(attack, (float(i % arena_size[0]), float(i // arena_size[0]), 0.0)), arena_size[0] * arena_size[1]
    
def discrete_action_table(env: "XanderDuelEnv") -> tuple[dict[int, Callable[[int], XanderAction]], int]:
    attacks: list[Attack] = list(__attack \
        for action in env.learner.combatant.stats.actions \
            if (__attack := action.as_attack()) is not None
    )

    directions = len(DIRECTIONS) # 8

    actions = {
        0: lambda _: EndAction(),
        1: lambda dir: MoveAction(dir),
    }

    start_i = 1 + directions
    for attack in attacks:
        func, l = for_attack(attack, env.config.arena.grid_size())
        actions[start_i] = func
        start_i += l


    actions = dict(reversed(actions.items()))
    return actions, start_i

def lookup_discrete_action(actions: dict[int, Callable[[int], XanderAction]], i: int) -> XanderAction:
    return next(
        (func(i - start_i) for start_i, func in actions.items() if i >= start_i),
    )

class DiscreteOneHot(gym.Wrapper[XanderObs, XS.MaskNDArray, XanderObs, XanderAction]):
    def __init__(self, env: "XanderDuelEnv"):
        super().__init__(env)

        self._actions, discrete_len = discrete_action_table(env)
        self._action_space = XS.OneHot(discrete_len)

    def step(self, action):
        i = action.argmax()
        action = lookup_discrete_action(self._actions, i)
        return super().step(action)

class DiscreteInt(gym.Wrapper[XanderObs, np.int64, XanderObs, XanderAction]):
    def __init__(self, env: "XanderDuelEnv"):
        super().__init__(env)

        self._actions, discrete_len = discrete_action_table(env)
        self._action_space = gym.spaces.Discrete(discrete_len)
    
    def step(self, action: np.int64):
        action = lookup_discrete_action(self._actions, action)
        return super().step(action)

    