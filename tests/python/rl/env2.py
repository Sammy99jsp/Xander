from typing import Any, Optional, Sequence
import gymnasium as gym
import numpy as np

class OneHot(gym.Space):
    def __init__(
        self,
        n : np.ndarray | Sequence[int] | int,
        seed: Optional[int | np.random.Generator] = None
    ):
        if isinstance(n, (Sequence, np.ndarray)):
            self.n = tuple(int(i) for i in n)
        else:
            self.n = (int(n),)
        assert (np.asarray(self.n) > 0).all() # all axes must have positive sizes
        super().__init__(self.n, np.int8, seed)

    @property
    def shape(self):
        return self._shape
    
    @property
    def is_np_flattenable(self):
        return True

    def sample(self, mask: Any | None = None):
        if mask is not None:
            assert isinstance(mask, np.ndarray), f"The expected type of the mask is np.ndarray, actual type: {type(mask)}"
            assert mask.dtype == np.int8, f"The expected dtype of the mask is np.int8, actual type: {mask.dtype}"
            assert mask.shape == self.shape, f"The expected shape of the mask is {self.shape}, actual shape: {mask.shape}"
            assert mask.min() >= 0, f"The mask should have non-negative values, actual values: {mask}"
    def contains(self, x):
        return isinstance(x, int) and x >= 0 and x < self.n

    def __repr__(self):
        return "OneHot(%d)" % self.n