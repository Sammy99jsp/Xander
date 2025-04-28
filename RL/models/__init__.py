import typing
import torch as T
import torch.nn as nn
import torch.nn.functional as F

from RL.models.noisy import NoisyLinear

__all__ = ["ActionHeads", "UnitAction", "DirectionalAction", "SpatialAction", "NoisyLinear"]

from xander.engine.combat.action.attack import Attack

class UnitAction(nn.Module):
    def __init__(self, shared_size: int, ):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(shared_size, 0),
        )

    def forward(self, x: T.Tensor) -> T.Tensor:
        return self.network(x)


class DirectionalAction(nn.Module):
    def __init__(self, shared_size: int, hidden_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(shared_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 8),
        )

    def forward(self, x: T.Tensor) -> T.Tensor:
        return self.network(x)


class SpatialAction(nn.Module):
    def __init__(self, shared_size: int, map_size: tuple[int, int]):
        super().__init__()

        self.map_size = map_size

        self.network = nn.Sequential(
            nn.Linear(shared_size, 512),
            nn.ReLU(),
            nn.Unflatten(1, (32, 4, 4)),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 1, kernel_size=1)
        )

    def forward(self, x: T.Tensor) -> T.Tensor:
        logits = self.network(x) # (..., 32, 32)
        if (logits.shape[-2], logits.shape[-1]) != self.map_size:
            logits = F.interpolate(
                logits,
                size=self.map_size,
                mode='bilinear',
                align_corners=False
            )
        return logits


def net_for_attack(attack: Attack, shared_size: int, map_size: tuple[int, int]) -> nn.Module:
    match attack.type:
        case "melee":
            return DirectionalAction(shared_size=shared_size, hidden_dim=512)
        case "ranged":
            return SpatialAction(shared_size=shared_size, map_size=map_size)
        case _:
            raise ValueError(f"Unknown attack type: {attack.type}")


class ActionTypeBlock[T](typing.TypedDict):
    end: T
    move: T
    attack: dict[str, T]


def map_action_block[T, U](a_block: ActionTypeBlock[T], map_fn: typing.Callable[[T], U]) -> ActionTypeBlock[U]:
    return {
        "end": map_fn(a_block["end"]),
        "move": map_fn(a_block["move"]),
        "attack": {
            name: map_fn(action) for name, action in a_block["attack"].items()
        }
    }


class ActionHeads(nn.Module):
    network: ActionTypeBlock[nn.Module]
    def __init__(self, shared_size: int, map_size: tuple[int, int], attacks: list[Attack]):
        super().__init__()
        self.network = {
            "end": UnitAction(shared_size=shared_size),
            "move": DirectionalAction(shared_size=shared_size, hidden_dim=512),
            "attack": {
                attack.name: net_for_attack(attack=attack, shared_size=shared_size, map_size=map_size) \
                    for attack in attacks
            },
        }

    def forward(self, x: T.Tensor) -> ActionTypeBlock[T.Tensor]:
        return {
            "end": self.network["end"](x),
            "move": self.network["move"](x),
            "attack": {
                name: action(x) for name, action in self.network["attack"].items()
            }
        }
    
    if typing.TYPE_CHECKING:
        def __call__(self, x: T.Tensor) -> ActionTypeBlock[T.Tensor]: ...