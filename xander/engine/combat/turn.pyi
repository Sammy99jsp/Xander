from typing import Callable
from xander.engine.combat import Combatant, speed
from xander.engine.combat.action.attack import Attack, AttackResult
from xander.engine.legality import Legality

type P3 = tuple[float, float, float]
type OneHotDirection = list[int]

class Turn:
    def move(self, delta: P3, mode = speed.Walking) -> Legality[None]: ...

    def attack(self, attack: Attack, target: P3) -> Legality[AttackResult]: ...

    def end(self) -> Legality[None]: ...

    def is_combat_active(self) -> bool: ...

    def movement_directions(self, mode = speed.Walking) -> Legality[list[tuple[float, float, float]]]: ...
    def movement_directions_one_hot(self, mode = speed.Walking) -> OneHotDirection:
        """One-hot encoding of the eight movement directions (N-NE-...-NW)."""

    def attack_directions(self, attack: Attack, filter: Callable[[Combatant], bool]) -> Legality[list[tuple[float, float, float]]]: ...
    def attack_directions_one_hot(self, attack: Attack, filter: Callable[[Combatant], bool]) -> OneHotDirection:
        """One-hot encoding of the eight attack directions (N-NE-...-NW)."""

    def movement_left(self, mode = speed.Walking) -> int: ...    
    
    @property
    def actions_left(self) -> int: ...

    @property
    def max_actions(self) -> int: ...

    def __repr__(self) -> str: ...