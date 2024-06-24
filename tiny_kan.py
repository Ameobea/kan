import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from tinygrad import Tensor, nn, TinyJit
from dataclasses import dataclass
import numpy as np
from typing import Callable, List
import matplotlib.pyplot as plt

from b_spline import coef2curve, plot_random_spline


def init(size) -> Tensor:
    return Tensor.uniform(size, low=-0.2, high=0.2)


class BatchKANCubicLayer:
    def __init__(self, in_count: int, out_count: int, **kwargs):
        self.a = init((out_count, in_count))
        self.b = init((out_count, in_count))
        self.c = init((out_count, in_count))

        self.bias = Tensor.uniform((out_count,), low=-0.2, high=0.2)

    def __call__(self, x: Tensor):
        # x is of shape (batch_size, in_count)
        x = x.unsqueeze(1)
        # x is now of shape (batch_size, 1, in_count)

        y = self.a * x.pow(3) + self.b * x.pow(2) + self.c * x
        y = y.sum(axis=-1)

        y = y + self.bias

        return y

    def get_learnable_params(self):
        params = [
            self.a,
            self.b,
            self.c,
            self.bias,
        ]
        return params


class BatchKANQuadraticLayer:
    def __init__(self, in_count: int, out_count: int, use_tanh=True, **kwargs):
        self.a = init((out_count, in_count))
        self.b = init((out_count, in_count))

        self.tanh_weights = (
            Tensor.uniform((out_count,), low=0.4, high=0.6) if use_tanh else None
        )
        self.bias = Tensor.uniform((out_count,), low=-0.2, high=0.2)

    def __call__(self, x: Tensor):
        # x is of shape (batch_size, in_count)
        x = x.unsqueeze(1)
        # x is now of shape (batch_size, 1, in_count)

        y = self.a * x.pow(2) + self.b * x
        y = y.sum(axis=-1)

        if self.tanh_weights is not None:
            y = (y * self.tanh_weights).tanh() + (1 - self.tanh_weights) * y
        y = y + self.bias

        return y

    def get_learnable_params(self):
        params = [
            self.a,
            self.b,
            self.bias,
        ]
        if self.tanh_weights is not None:
            params.append(self.tanh_weights)
        return params


class BatchKANLinearLayer:
    def __init__(self, in_count: int, out_count: int, **kwargs):
        self.a = init((out_count, in_count))

        self.tan_weights = Tensor.uniform((out_count,), low=0.4, high=0.6)
        self.bias = Tensor.uniform((out_count,), low=-0.2, high=0.2)

    def __call__(self, x: Tensor):
        # x is of shape (batch_size, in_count)
        x = x.unsqueeze(1)
        # x is now of shape (batch_size, 1, in_count)

        y = self.a * x
        y = y.sum(axis=-1)

        y = (y * self.tan_weights).tanh() + (1 - self.tan_weights) * y
        y = y + self.bias

        return y

    def get_learnable_params(self):
        params = [
            self.a,
            self.bias,
            self.tan_weights,
        ]
        return params


class BatchKANCubicBSplineLayer:
    def __init__(
        self,
        in_count: int,
        out_count: int,
        num_knots: int = 2,
        spline_order: int = 3,
        use_bias=True,
    ):
        self.in_count = in_count
        self.out_count = out_count
        self.num_knots = num_knots
        self.order = spline_order
        self.num_splines = in_count * out_count
        num_grid_intervals = num_knots - 1

        self.coefficients = Tensor.uniform(
            (self.num_splines, num_grid_intervals + spline_order),
            low=-0.2,
            high=0.2,
        )

        self.bias = (
            Tensor.uniform((out_count,), low=-0.2, high=0.2) if use_bias else None
        )

        domain = (-1, 1)
        self.grid = Tensor.einsum(
            "i,j->ij",
            Tensor.ones(self.num_splines),
            Tensor(np.linspace(domain[0], domain[1], num_grid_intervals + 1)),
        )
        assert self.grid.shape == (self.num_splines, num_grid_intervals + 1)

    def __call__(self, x: Tensor, use_sigmoid_trick=True):
        batch_size = x.shape[0]
        # x is of shape (batch_size, in_count)
        assert x.shape == (batch_size, self.in_count)
        x = x.unsqueeze(1).expand(-1, self.out_count, -1)
        # x is now of shape (batch_size, out_count, in_count)
        x = x.permute(2, 1, 0).reshape(self.num_splines, batch_size)
        assert x.shape == (self.num_splines, batch_size)

        y = coef2curve(
            x,
            self.grid,
            self.coefficients,
            self.order,
            use_sigmoid_trick=use_sigmoid_trick,
        )
        # y is of shape (num_splines, batch_size)
        assert y.shape == (self.num_splines, batch_size)
        y = y.permute(1, 0)
        # y is now of shape (batch_size, num_splines)
        assert y.shape == (batch_size, self.num_splines)
        y = y.reshape(batch_size, self.in_count, self.out_count)
        # y is now of shape (batch_size, out_count, in_count)
        assert y.shape == (batch_size, self.in_count, self.out_count)
        y = y.sum(axis=1)
        # y is now of shape (batch_size, out_count)
        assert y.shape == (batch_size, self.out_count)

        if self.bias is not None:
            y = y + self.bias

        return y

    def get_learnable_params(self):
        params = [self.coefficients]
        if self.bias is not None:
            params.append(self.bias)
        return params

    def plot_response(self, input_range=(-1, 1), resolution=100):
        x = np.linspace(input_range[0], input_range[1], resolution)
        x = Tensor(x).unsqueeze(1)
        x = x.expand(resolution, self.in_count)

        y_pred_sigmoid_trick = self(x, use_sigmoid_trick=True).numpy()
        y_pred_no_trick = self(x, use_sigmoid_trick=False).numpy()

        for i in range(self.out_count):
            plt.plot(x.numpy(), y_pred_sigmoid_trick[:, i], label=f"sigmoid_trick_{i}")
            plt.plot(x.numpy(), y_pred_no_trick[:, i], label=f"no_trick_{i}")

        plt.legend()
        plt.show()


@dataclass
class HiddenLayerDef:
    out_count: int
    num_knots: int = 2
    spline_order: int = 3


class KAN:
    def __init__(
        self,
        in_count: int,
        out_count: int,
        hidden_layer_defs: List[HiddenLayerDef],
        Layer=BatchKANCubicLayer,
        layer_params: dict = {},
    ):
        self.layers = [Layer(in_count, hidden_layer_defs[0].out_count)]
        for i in range(1, len(hidden_layer_defs)):
            self.layers.append(
                Layer(
                    hidden_layer_defs[i - 1].out_count,
                    hidden_layer_defs[i].out_count,
                    num_knots=hidden_layer_defs[i].num_knots,
                    spline_order=hidden_layer_defs[i].spline_order,
                    **layer_params,
                )
            )
        self.layers.append(Layer(hidden_layer_defs[-1].out_count, out_count))

    def __call__(self, x: Tensor):
        for layer in self.layers:
            x = layer(x)
        return x

    def get_learnable_params(self):
        params = []
        for layer in self.layers:
            params += layer.get_learnable_params()
        return set(params)

    def plot_response(
        self,
        target_fn: Callable[[Tensor], Tensor],
        input_domain=(-1, 1),
        resolution=100,
    ):
        x = np.linspace(input_domain[0], input_domain[1], resolution)
        y_true = [target_fn(Tensor([i])).numpy()[0] for i in x]
        y_pred = self(Tensor(x).reshape(resolution, 1)).reshape(resolution).numpy()

        plt.plot(x, y_true, label="Actual")
        plt.plot(x, y_pred, label="Predicted")

        plt.legend()
        plt.show()

    def plot_neuron_responses(self, input_domain=(-1, 1), resolution=100):
        """
        Plots the response of each neuron in the network with subplots
        """

        x = np.linspace(input_domain[0], input_domain[1], resolution)
        x = Tensor(x).unsqueeze(1)
        x = x.expand(resolution, 1)

        prev_layer_output = x
        for i, layer in enumerate(self.layers):
            y = layer(prev_layer_output).numpy()
            for j in range(y.shape[1]):
                plt.subplot(len(self.layers), y.shape[1], i * y.shape[1] + j + 1)
                plt.plot(x.numpy(), y[:, j])
            prev_layer_output = Tensor(y)

        plt.show()


def run_model():
    model = KAN(
        1,
        1,
        [HiddenLayerDef(4), HiddenLayerDef(4), HiddenLayerDef(4)],
        Layer=BatchKANQuadraticLayer,
        layer_params={"use_tanh": False},
    )

    all_params = model.get_learnable_params()

    opt = nn.optim.Adam(list(all_params), lr=0.01)

    def target_fn(x: Tensor):
        # x_abs = (x * 1.5).abs()
        # return x + x_abs - x_abs.trunc()

        return ((x * 8).sin() * 80).tanh()

    input_range = (-2, 2)

    def generate_input(batch_size) -> float:
        return np.random.uniform(input_range[0], input_range[1], (batch_size, 1))

    @TinyJit
    def train_step() -> Tensor:
        with Tensor.train():
            opt.zero_grad()

    batch_size = 32

    @TinyJit
    def train_step(x: Tensor) -> Tensor:
        with Tensor.train():
            opt.zero_grad()

            y_pred = model(x)
            y_actual = target_fn(x)
            loss = (y_pred - y_actual).pow(2).mean()
            loss = loss.backward()
            opt.step()
            return loss

    with Tensor.train():
        for step in range(30000):
            x = Tensor(generate_input(batch_size))
            loss = train_step(x)
            print(f"step: {step}, loss: {loss.numpy()}")

    model.plot_neuron_responses(input_domain=input_range)
    model.plot_response(target_fn, input_domain=input_range)


run_model()
