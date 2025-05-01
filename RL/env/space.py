# mypy: disable-error-code="override"

from typing import Any, Sequence
import gymnasium as gym
import numpy as np
import numpy.typing as npt

MaskNDArray = npt.NDArray[np.int8]

class OneHot(gym.Space[MaskNDArray]):
    """One-hot encoding action space."""
    n: tuple[int, ...] | int
    
    """One-hot encoding space."""
    def __init__(
        self,
        n: npt.NDArray[np.integer[Any]] | Sequence[int] | int,
        seed: int | np.random.Generator | None = None,
    ):
        """
        Constructor of :class:`OneHotSpace` space.

        Based off the [gymnasium.spaces.MultiBinary] space.
        """
        if isinstance(n, (Sequence, np.ndarray)):
            self.n = input_n = tuple(int(i) for i in n)
            assert (np.asarray(input_n) > 0).all()  # n (counts) have to be positive
        else:
            self.n = n = int(n)
            input_n = (n,)
            assert (np.asarray(input_n) > 0).all()  # n (counts) have to be positive

        super().__init__(input_n, np.int8, seed)

    @property
    def shape(self) -> tuple[int, ...]:
        """Has stricter type than gym.Space - never None."""
        return self._shape  # type: ignore

    @property
    def is_np_flattenable(self):
        """Checks whether this space can be flattened to a :class:`spaces.Box`."""
        return True
    
    def sample(self, mask: MaskNDArray | None = None) -> npt.NDArray[np.int8]:
        """Generates a single random sample from this space.

        Args:
            mask: An optional np.ndarray to mask samples with expected shape of ``space.shape``.
                FOr ``mask == 0`` then the bit will be excluded and ``mask == 1` then the bit is included.
                The expected mask shape is the space shape and mask dtype is ``np.int8``.

        Returns:
            Sampled on-hot-encoded-value from space
        """
        if mask is not None:
            assert isinstance(
                mask, np.ndarray
            ), f"The expected type of the mask is np.ndarray, actual type: {type(mask)}"
            assert (
                mask.dtype == np.int8
            ), f"The expected dtype of the mask is np.int8, actual dtype: {mask.dtype}"
            assert (
                mask.shape == self.shape
            ), f"The expected shape of the mask is {self.shape}, actual shape: {mask.shape}"
            assert np.all(
                (mask == 0) | (mask == 1)
            ), f"All values of a mask should be 0, or 1, actual values: {mask}"

            non_zero_indices = np.where(mask == 1)[0]
            idx = self._np_random.choice(non_zero_indices, size=1, replace=False) # type: ignore
            sample = np.zeros(self.n, dtype=self.dtype)
            sample[idx] = 1
            return sample
        
        else:
            sample = np.zeros(self.n, dtype=self.dtype)
            idx = self._np_random.integers(0, self.n) # type: ignore
            sample[idx] = 1
            return sample
        
    def to_jsonable(self, sample_n: Sequence[npt.NDArray[np.int8]]) -> list[Sequence[int]]:
        """Convert a batch of samples from this space to a JSONable data type."""
        return np.array(sample_n).tolist()

    def from_jsonable(self, sample_n: list[Sequence[int]]) -> list[npt.NDArray[np.int8]]:
        """Convert a JSONable data type to a batch of samples from this space."""
        return [np.asarray(sample, self.dtype) for sample in sample_n]

    def __repr__(self) -> str:
        """Gives a string representation of this space."""
        return f"OneHot({self.n})"

    def __eq__(self, other: Any) -> bool:
        """Check whether `other` is equivalent to this instance."""
        return isinstance(other, OneHot) and self.n == other.n
