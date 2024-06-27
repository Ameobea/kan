from typing import List, Set, Callable
import numpy as np

from tinygrad import Tensor, nn, dtypes

from shape_checker import check_shapes
from util import build_activation, param_count


class TinyNN:
    def __init__(
        self,
        in_count: int,
        out_count: int,
        hidden_layer_sizes: List[int],
        activation_fn="tanh",
        post_activation_fn="tanh",
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
        self.post_activation_fn = build_activation(post_activation_fn)

    @check_shapes([(None, None), dtypes.float32], ret=[(None, None), dtypes.float32])
    def __call__(self, x: Tensor) -> Tensor:
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.activation_fn(x)

        x = self.layers[-1](x)
        x = build_activation(self.post_activation_fn)(x)

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
