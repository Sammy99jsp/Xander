import json
import random
from typing import Any, Literal as L
import numpy as np
import torch as T
import gymnasium as gym

from gymnasium import spaces as S
from RL.env2.agents import CallbackAgent
from RL.env2.config import XanderEnvConfig
from RL.env2.util import AttackAction, EndAction, MoveAction, XanderAction, XanderObs, observation_space, observe, pos_add
from xander.engine import dice
from xander.engine.combat import Combat, Combatant
from xander.engine.combat.turn import Turn
from xander.engine.legality import Legality


class XanderDuelEnvConfig(XanderEnvConfig):
    type: L["duel"]

class XanderDuelEnv(gym.Env[XanderObs, XanderAction]):
    """
    Xander Duel Environment for reinforcement learning.
    """

    config: XanderDuelEnvConfig
    learner: CallbackAgent
    combat: Combat

    def __init__(self, config: XanderDuelEnvConfig):
        """
        Constructor for the XanderDuelEnv class.

        Args:
            config: Configuration dictionary for the environment.
        """
        super().__init__()
        self.config = config
        self.combat, self.learner = config.build()


        # self.action_space = S.Discrete(3)
        self.observation_space = observation_space(self.config)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None
    ) -> tuple[XanderObs, dict[str, Any]]:
        if seed is not None:
            dice.set_seed(seed)
            random.seed(seed)
            np.random.seed(seed)
            T.manual_seed(seed)
            T.cuda.random.manual_seed(seed)
        
        while True:
            self.combat, self.learner = self.config.build()
            while True:
                payload = self._fast_forward()
                match payload:
                    case "lost":
                        # self.cfg["seed"] += 1
                        break
                    case (combatant, turn):
                        return observe(self.config, combatant, turn), {}
                    case _:
                        raise NotImplementedError
        

    def step(self, action: XanderAction) -> tuple[XanderObs, float, bool, bool, dict]:
        payload = self._fast_forward()

        match payload:
            case "timeout" | "won" | "lost":
                raise NotImplementedError("Shouldn't happen. Reset the environment!")
            case _:
                pass
        
        combatant, turn = payload
        
        rew = 0.0
        info = {}

        match action:
            case EndAction():
                turn.end()
            case MoveAction(direction=dir):
                l1 = turn.move(dir)

                if not l1.is_legal():
                    rew += self.config.rewards.illegal
                    info["illegal"] = True

            case AttackAction(attack=attack, target=target):
                l2 = turn.attack(
                    attack, 
                    target if attack.type == "ranged" \
                    else pos_add(self.learner.combatant.position, target)
                )
                
                if l2.is_legal():
                    _ = l2.inner()
                    _dmg = _.damage
                    if _dmg is not None:
                        info["damage"] = _dmg.sum()
                    info["attack"] = repr(_.to_hit)
                else:
                    rew += self.config.rewards.illegal
                    info["illegal"] = True


        obs = observe(self.config, combatant, turn)
        payload = self._fast_forward()

        match payload:
            case "timeout":
                return obs, rew, False, True, info
            case "won":
                rew += self.config.rewards.win
                return obs, rew, True, False, {
                    "won": True,
                    "hp_left": self.learner.combatant.stats.hp / self.learner.combatant.stats.max_hp,
                    **info
                }
            case "lost":
                rew += self.config.rewards.lose
                return obs, rew, True, False, {
                    "lost": True, 
                    "hp_left": self.learner.combatant.stats.hp / self.learner.combatant.stats.max_hp,
                    **info
                }
            case _:
                rew += self.config.rewards.step
                return obs, rew, False, False, info


    def _fast_forward(self) -> tuple[Combatant, Turn] | L["won", "lost"]:
        while True:
            with self.learner as payload:
                if self.learner.combatant.stats.dead:
                    return "lost" 
                if all(
                    c.stats.dead for c in self.combat.combatants \
                        if c.name != self.learner.combatant.name
                ):
                    return "won" # terminate the episode
                if payload is not None:
                    return payload
                
                self.combat.step()


if __name__ == "__main__":
    print(json.dumps(XanderDuelEnvConfig.model_json_schema()))
