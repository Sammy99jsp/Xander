import typing

import numpy as np


class Transition(typing.NamedTuple):
    obs: np.ndarray
    action: np.ndarray
    reward: float
    next_obs: np.ndarray
    done: bool