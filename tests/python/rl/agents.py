from typing import Callable, Optional

import numpy as np
from tests.python.rl.utils import DELTA
from xander.engine.combat import Combatant
from xander.engine.combat.turn import Turn


class RandomAgent:
    """An agent that takes random actions during combat."""
    def __init__(self, combatant: Combatant):
        self.combatant = combatant
    
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
                turn.attack(self.combatant.stats.actions[0].as_attack(), DELTA[i - 8])
            case i:
                turn.move(DELTA[i])


class CallbackAgent:
    combatant: Combatant
    
    _payload: Optional[tuple[Combatant, Turn]]
    
    """An agent that uses a callback function to decide on actions."""
    def __init__(self, combatant: Combatant):
        self.combatant = combatant
        self._payload = None
    
    def act(self, turn: Turn) -> None:
        """Use the callback to take actions."""
        self._payload = (self.combatant, turn)

    def __enter__(self) -> Optional[tuple[Combatant, Turn]]:
        return self._payload
    
    def __exit__(self, exc_type, exc_value, traceback) -> None:
        pass
        # self._payload = None