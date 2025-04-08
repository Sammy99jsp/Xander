from typing import Generator, TypeVar
from contextlib import contextmanager

import torch
import torch.nn as nn

K = TypeVar('K', bound=nn.Module)

@contextmanager
def frozen_weights(model: K) -> Generator[K, None, None]:
    """
    Temporarily freeze the layers of a model.
    This is expressed as a context manager -- you use it in a `with` block.

    Arguments
    ---------
    model : nn.Module
        The model to freeze.

    Example
    -------
    >>> import torch
    >>> import torch.nn as nn
    >>>
    >>> model = nn.Linear(10, 10)
    >>> with frozen(model) as model:
    >>>    # model is frozen here, so gradients of `model` will not be included
    >>>    # in the autograd graph.
    >>>    y = model(torch.tensor([1.0, 2.0, 3.0]))
    """
    try:
        for param in model.parameters():
            param.requires_grad = False
        yield model
    finally:
        for param in model.parameters():
            param.requires_grad = True


def default_device() -> torch.device:
    """
    Return the a default device for PyTorch.

    Preference is 'CUDA' > 'MPS' > 'CPU'
    """
    return \
        torch.device('cpu') if not torch.mps.is_available() \
            else torch.device('mps') \
        if not torch.cuda.is_available() \
            else torch.device('cuda')