from xander.engine.combat.action import Action
from xander.engine.combat.speed import SpeedType, Walking

class Stats:
    @staticmethod
    def from_json(file: str) -> Stats:
        """
        Loads a monster stat block from a JSON stat block file.

        Examples
        --------

        Load a rat:
        >>> from xander.engine.actors import Monster
        >>> rat = Monster.from_json("rat.json")
        >>> rat
        Rat <2/2 HP>
        """
        ...

    def __repr__(self) -> str: ...

    def speed(self, ty: SpeedType = Walking) -> int: ...

    @property
    def hp(self) -> int: ...
    
    @property
    def max_hp(self) -> int: ...
    
    @property
    def temp_hp(self) -> int | None: ...

    @property
    def actions(self) -> list[Action]: ...

    @property
    def dead(self) -> bool: ...