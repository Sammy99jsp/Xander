import typing

class Arena:
    """An active arena."""
    @property
    def grid_dimensions(self) -> tuple[int, int]: ...

    def _repr_html_(self) -> str:
        ...

    def save_image(self, path: str):
        """Save the arena's current state to an image."""
        ...

class ProtoArena(typing.Protocol):
    pass

class Simple(ProtoArena):
    """A simple rectangular arena."""
    
    def __init__(self, width: int, height: int):
        ...