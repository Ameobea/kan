from typing import Callable, Optional

from tinygrad import Tensor
import numpy as np


def param_count(t: Tensor) -> int:
    return int(np.prod(t.shape))


def build_activation(activation_fn: Optional[str]) -> Callable[[Tensor], Tensor]:
    if activation_fn == "tanh":
        return lambda x: x.tanh()
    if activation_fn == "relu":
        return lambda x: x.relu()
    if activation_fn == "sigmoid":
        return lambda x: x.sigmoid()
    if activation_fn is None:
        return lambda x: x
    raise ValueError(f"Unknown activation function: {activation_fn}")
