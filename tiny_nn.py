from typing import List, Set, Callable
import numpy as np

from tinygrad import Tensor, nn, dtypes

from shape_checker import check_shapes


def param_count(t: Tensor) -> int:
    return int(np.prod(t.shape))


def build_activation(activation_fn: str) -> Callable[[Tensor], Tensor]:
    if activation_fn == "tanh":
        return lambda x: x.tanh()
    if activation_fn == "relu":
        return lambda x: x.relu()
    if activation_fn == "sigmoid":
        return lambda x: x.sigmoid()
    raise ValueError(f"Unknown activation function: {activation_fn}")


class TinyNN:
    def __init__(
        self,
        in_count: int,
        out_count: int,
        hidden_layer_sizes: List[int],
        activation_fn="tanh",
    ):
        if len(hidden_layer_sizes) == 0:
            self.layers = [nn.Linear(in_count, out_count)]
            return

        self.layers = [nn.Linear(in_count, hidden_layer_sizes[0])]
        for i in range(1, len(hidden_layer_sizes)):
            self.layers.append(
                nn.Linear(hidden_layer_sizes[i - 1], hidden_layer_sizes[i])
            )
        self.layers.append(nn.Linear(hidden_layer_sizes[-1], out_count))

        self.activation_fn = build_activation(activation_fn)

    @check_shapes([(None, None), dtypes.float32], ret=[(None, None), dtypes.float32])
    def __call__(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
            x = self.activation_fn(x)

        return x

    def get_learnable_params(self) -> Set[Tensor]:
        params = set()
        for layer in self.layers:
            params.add(layer.weight)
            if layer.bias is not None:
                params.add(layer.bias)

        return params

    def param_count(self):
        return sum(param_count(p) for p in self.get_learnable_params())
