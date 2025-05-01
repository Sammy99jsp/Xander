import functools
from typing import Any, Optional
import numpy as np
from RL.env.util import DIRECTIONS
from xander.engine.combat import Combatant
from xander.engine.combat.action.attack import Attack
from xander.engine.combat.turn import Turn


class RandomAgent:
    combatant: Combatant

    """An agent that takes random actions during combat."""
    def __init__(self):
        pass

    @functools.cached_property
    def attacks(self) -> list[Attack]:
        """Get the list of attacks available to the agent."""
        return [attack for a in self.combatant.stats.actions if (attack := a.as_attack()) is not None]
    
    def act(self, turn: Turn) -> None:
        """Take random actions until the turn is over."""
        
        attack_dirs: np.ndarray = np.array(turn.attack_directions_one_hot(self.combatant.stats.actions[0].as_attack(), filter=lambda c: not c.stats.dead))
        move_dirs: np.ndarray = np.array(turn.movement_directions_one_hot())

        action = np.random.choice(np.concatenate((
            move_dirs.nonzero()[0],
            attack_dirs.nonzero()[0] + 8,
            np.array([16])
        )))

        match action:
            case 16:
                turn.end()
            case i if i >= 8:
                turn.attack(self.combatant.stats.actions[0].as_attack(), DIRECTIONS[i - 8])
            case i:
                turn.move(DIRECTIONS[i])

class CallbackAgent:
    combatant: Combatant
    algorithm: str
    hyperparameters: Any
    
    _payload: Optional[tuple[Combatant, Turn]]
    
    """An agent that uses a callback function to decide on actions."""
    def __init__(self):
        self._payload = None
    
    def act(self, turn: Turn) -> None:
        """Use the callback to take actions."""
        self._payload = (self.combatant, turn)

    def __enter__(self) -> Optional[tuple[Combatant, Turn]]:
        l = self._payload
        self._payload = None
        return l
    
    def __exit__(self, exc_type, exc_value, traceback) -> None:
        pass
        # self._payload = None
