import sys
import os

from util import build_activation, param_count

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from tinygrad import Tensor, nn, TinyJit, dtypes
import numpy as np
from typing import Callable, List, Optional
import matplotlib.pyplot as plt

from b_spline import coef2curve
from shape_checker import check_shapes, check_shape


def init(shape) -> Tensor:
    return Tensor.uniform(shape, low=-0.25, high=0.25, dtype=dtypes.float32)


class BatchKANCubicLayer:
    def __init__(
        self,
        in_count: int,
        out_count: int,
        use_tanh=False,
        use_pre_tanh_post_weights=False,
        use_pre_tanh_bias=False,
        use_post_tanh_post_weights=False,
        use_post_tanh_bias=False,
        **kwargs,
    ):
        self.in_count = in_count
        self.out_count = out_count

        self.a = init((out_count, in_count))
        self.b = init((out_count, in_count))
        self.c = init((out_count, in_count))

        self.use_tanh = use_tanh

        self.pre_tanh_post_weights = (
            init((out_count,)) if use_pre_tanh_post_weights else None
        )
        self.pre_tanh_bias = init((out_count,)) if use_pre_tanh_bias else None
        self.post_tanh_post_weights = (
            init((out_count,)) if use_post_tanh_post_weights else None
        )
        self.post_tanh_bias = init((out_count,)) if use_post_tanh_bias else None

    @check_shapes(ret=[(None, None), dtypes.float32])
    def __call__(self, x: Tensor):
        check_shape(x, [(None, self.in_count), dtypes.float32])
        x = x.unsqueeze(1)

        y = self.a * x.pow(3) + self.b * x.pow(2) + self.c * x
        y = y.sum(axis=-1).contiguous()

        if self.pre_tanh_post_weights is not None:
            y = y * self.pre_tanh_post_weights
        if self.pre_tanh_bias is not None:
            y = y + self.pre_tanh_bias

        elif self.use_tanh:
            y = y.tanh()

        if self.post_tanh_post_weights is not None:
            y = y * self.post_tanh_post_weights
        if self.post_tanh_bias is not None:
            y = y + self.post_tanh_bias

        return y

    def __repr__(self):
        return f"BatchKANQuadraticLayer(in_count={self.in_count}, out_count={self.out_count}, use_tanh={self.use_tanh}, use_pre_tanh_post_weights={self.pre_tanh_post_weights is not None}, use_pre_tanh_bias={self.pre_tanh_bias is not None}, use_post_tanh_post_weights={self.post_tanh_post_weights is not None}, use_post_tanh_bias={self.post_tanh_bias is not None})"

    def get_learnable_params(self) -> List[Tensor]:
        params = [self.a, self.b, self.c]

        if self.pre_tanh_post_weights is not None:
            params.append(self.pre_tanh_post_weights)
        if self.pre_tanh_bias is not None:
            params.append(self.pre_tanh_bias)
        if self.post_tanh_post_weights is not None:
            params.append(self.post_tanh_post_weights)
        if self.post_tanh_bias is not None:
            params.append(self.post_tanh_bias)
        return params

    def param_count(self) -> int:
        return sum(param_count(p) for p in self.get_learnable_params())


class BatchKANQuadraticLayer:
    def __init__(
        self,
        in_count: int,
        out_count: int,
        use_tanh=False,
        use_pre_tanh_post_weights=False,
        use_pre_tanh_bias=False,
        use_post_tanh_post_weights=False,
        use_post_tanh_bias=False,
        use_base_fn=False,
        **kwargs,
    ):
        self.in_count = in_count
        self.out_count = out_count

        self.a = init((out_count, in_count))
        self.b = init((out_count, in_count))

        self.use_tanh = use_tanh

        self.pre_tanh_post_weights = (
            init((out_count,)) if use_pre_tanh_post_weights else None
        )
        self.pre_tanh_bias = init((out_count,)) if use_pre_tanh_bias else None
        self.post_tanh_post_weights = (
            init((out_count,)) if use_post_tanh_post_weights else None
        )
        self.post_tanh_bias = init((out_count,)) if use_post_tanh_bias else None

        self.base_weights = init((out_count, in_count)) if use_base_fn else None

    @check_shapes(ret=[(None, None), dtypes.float32])
    def __call__(self, x: Tensor):
        check_shape(x, [(None, self.in_count), dtypes.float32])
        x = x.unsqueeze(1)

        y = self.a * x.pow(2) + self.b * x

        if self.base_weights is not None:
            y = y + self.base_weights * x.gelu()

        y = y.sum(axis=-1).contiguous()

        if self.pre_tanh_post_weights is not None:
            y = y * self.pre_tanh_post_weights
        if self.pre_tanh_bias is not None:
            y = y + self.pre_tanh_bias

        elif self.use_tanh:
            y = y.tanh()

        if self.post_tanh_post_weights is not None:
            y = y * self.post_tanh_post_weights
        if self.post_tanh_bias is not None:
            y = y + self.post_tanh_bias

        return y

    def __repr__(self):
        return f"BatchKANQuadraticLayer(in_count={self.in_count}, out_count={self.out_count}, use_tanh={self.use_tanh}, use_pre_tanh_post_weights={self.pre_tanh_post_weights is not None}, use_pre_tanh_bias={self.pre_tanh_bias is not None}, use_post_tanh_post_weights={self.post_tanh_post_weights is not None}, use_post_tanh_bias={self.post_tanh_bias is not None}, use_base_fn={self.base_weights is not None})"

    def get_learnable_params(self) -> List[Tensor]:
        params = [self.a, self.b]

        if self.pre_tanh_post_weights is not None:
            params.append(self.pre_tanh_post_weights)
        if self.pre_tanh_bias is not None:
            params.append(self.pre_tanh_bias)
        if self.post_tanh_post_weights is not None:
            params.append(self.post_tanh_post_weights)
        if self.post_tanh_bias is not None:
            params.append(self.post_tanh_bias)
        if self.base_weights is not None:
            params.append(self.base_weights)
        return params

    def param_count(self) -> int:
        return sum(param_count(p) for p in self.get_learnable_params())


class BatchKANBSplineLayer:
    def __init__(
        self,
        in_count: int,
        out_count: int,
        num_knots: int = 4,
        spline_order: int = 3,
        use_post_weights=True,
        use_bias=True,
        use_skip_conn_weights=False,
        **kwargs,
    ):
        self.in_count = in_count
        self.out_count = out_count
        self.num_knots = num_knots
        self.order = spline_order
        self.num_splines = in_count * out_count
        num_grid_intervals = num_knots - 1

        self.coefficients = init((self.num_splines, num_grid_intervals + spline_order))

        self.skip_conn_weights = (
            init((self.num_splines,)) if use_skip_conn_weights else None
        )
        self.post_weights = init((out_count,)) if use_post_weights else None
        self.bias = init((out_count,)) if use_bias else None

        domain = (-1, 1)
        self.grid = Tensor.einsum(
            "i,j->ij",
            Tensor.ones(self.num_splines, dtype=dtypes.float32),
            Tensor(
                np.linspace(domain[0], domain[1], num_grid_intervals + 1),
                dtype=dtypes.float32,
            ),
        )
        check_shape(
            self.grid, [(self.num_splines, num_grid_intervals + 1), dtypes.float32]
        )

    @check_shapes(ret=[(None, None), dtypes.float32])
    def __call__(self, x: Tensor):
        batch_size = x.shape[0]
        check_shape(x, [(batch_size, self.in_count), dtypes.float32])
        x = x.unsqueeze(1).expand(-1, self.out_count, -1)
        check_shape(x, [(batch_size, self.out_count, self.in_count), dtypes.float32])
        coef_x = x.permute(2, 1, 0).reshape(self.num_splines, batch_size)
        check_shape(coef_x, [(self.num_splines, batch_size), dtypes.float32])

        y = coef2curve(coef_x, self.grid, self.coefficients, self.order)
        check_shape(y, [(self.num_splines, batch_size), dtypes.float32])
        y = y.permute(1, 0)
        check_shape(y, [(batch_size, self.num_splines), dtypes.float32])

        if self.skip_conn_weights is not None:
            y = (
                y
                + (x.reshape(batch_size, self.num_splines) * self.skip_conn_weights)
                .contiguous()
                .gelu()
            )

        y = y.reshape(batch_size, self.in_count, self.out_count)
        check_shape(y, [(batch_size, self.in_count, self.out_count), dtypes.float32])
        y = y.sum(axis=1)
        check_shape(y, [(batch_size, self.out_count), dtypes.float32])

        if self.post_weights is not None:
            y = y * self.post_weights
        if self.bias is not None:
            y = y + self.bias

        return y

    def __repr__(self):
        return f"BatchKANBSplineLayer(in_count={self.in_count}, out_count={self.out_count}, num_knots={self.num_knots}, spline_order={self.order}, use_post_weights={self.post_weights is not None}, use_bias={self.bias is not None}, use_skip_conn_weights={self.skip_conn_weights is not None})"

    def get_learnable_params(self):
        params = [self.coefficients]
        if self.skip_conn_weights is not None:
            params.append(self.skip_conn_weights)
        if self.post_weights is not None:
            params.append(self.post_weights)
        if self.bias is not None:
            params.append(self.bias)
        return params

    def param_count(self) -> int:
        return sum(param_count(p) for p in self.get_learnable_params())

    def plot_response(self, input_range=(-1, 1), resolution=100):
        x = np.linspace(input_range[0], input_range[1], resolution, dtype=np.float32)
        x = Tensor(x).unsqueeze(1)
        x = x.expand(resolution, self.in_count)

        y_pred = self(x).numpy()

        for i in range(self.out_count):
            plt.plot(x.numpy(), y_pred[:, i], label=f"y{i}")

        plt.legend()
        plt.show()


class NNLayer:
    """
    Traditional linear/dense layer from normal neural networks
    """

    def __init__(self, in_count: int, out_count: int, activation_fn="tanh", **kwargs):
        self.in_count = in_count
        self.out_count = out_count

        self.linear = nn.Linear(in_count, out_count)
        self.activation_fn_name = activation_fn
        self.activation_fn = build_activation(activation_fn)

    @check_shapes([(None, None), dtypes.float32], ret=[(None, None), dtypes.float32])
    def __call__(self, x: Tensor):
        return self.activation_fn(self.linear(x))

    def __repr__(self):
        return f"NNLayer(in_count={self.in_count}, out_count={self.out_count}, activation_fn={self.activation_fn_name})"

    def get_learnable_params(self):
        return [self.linear.weight, self.linear.bias]


class HiddenLayerDef:
    def __init__(self, out_count: int, **kwargs):
        self.out_count = out_count
        self.kwargs = kwargs


class KAN:
    def __init__(
        self,
        in_count: int,
        out_count: int,
        hidden_layer_defs: List[HiddenLayerDef],
        Layer=BatchKANCubicLayer,
        FirstLayer=None,
        LastLayer=None,
        layer_params: dict = {},
        post_activation_fn: Optional[str] = None,
    ):
        FirstLayer = FirstLayer or Layer
        LastLayer = LastLayer or Layer

        if len(hidden_layer_defs) == 0:
            self.layers = [FirstLayer(in_count, out_count, **layer_params)]
            return

        self.layers = [
            FirstLayer(in_count, hidden_layer_defs[0].out_count, **layer_params)
        ]
        for i in range(1, len(hidden_layer_defs)):
            params = {**layer_params, **hidden_layer_defs[i].kwargs}
            self.layers.append(
                Layer(
                    hidden_layer_defs[i - 1].out_count,
                    hidden_layer_defs[i].out_count,
                    **params,
                )
            )
        last_layer_params = {
            **layer_params,
            "use_tanh": False,
            "use_pre_tanh_post_weights": False,
            "use_pre_tanh_bias": False,
            "use_post_tanh_post_weights": False,
            "use_post_tanh_bias": False,
            "activation_fn": "linear",
        }
        self.layers.append(
            LastLayer(hidden_layer_defs[-1].out_count, out_count, **last_layer_params)
        )

        self.post_activation_fn_name = post_activation_fn
        self.post_activation_fn = build_activation(post_activation_fn)

    @check_shapes((None, None))
    def __call__(self, x: Tensor):
        for layer in self.layers:
            x = layer(x)

        x = self.post_activation_fn(x)
        return x

    def __repr__(self):
        return f"KAN(layers={[repr(layer) for layer in self.layers]}, post_activation_fn={self.post_activation_fn_name})"

    def get_learnable_params(self):
        params = []
        for layer in self.layers:
            params += layer.get_learnable_params()
        return set(params)

    def param_count(self) -> int:
        return sum(param_count(p) for p in self.get_learnable_params())

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


def run_model(quiet=False):
    model = KAN(
        1,
        1,
        [HiddenLayerDef(4)],
        Layer=BatchKANQuadraticLayer,
    )

    all_params = model.get_learnable_params()

    opt = nn.optim.Adam(list(all_params), lr=0.01)

    def target_fn(x: Tensor):
        # x_abs = (x * 1.5).abs()
        # return x + x_abs - x_abs.trunc()

        # return ((x * 8).sin() * 80).tanh()
        # return 1 if x < 0 else -1
        return (x < 0).where(1, -1)

    input_range = (-2, 2)

    def generate_input(batch_size: int) -> np.ndarray:
        return np.random.uniform(input_range[0], input_range[1], (batch_size, 1))

    batch_size = 32

    @TinyJit
    @check_shapes([(batch_size, 1), dtypes.float32], ret=[(), dtypes.float32])
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
            if not quiet:
                print(f"step: {step}, loss: {loss.numpy()}")

    model.plot_neuron_responses(input_domain=input_range)
    model.plot_response(target_fn, input_domain=input_range)


def plot_spline_layer_domain_and_range():
    layer = BatchKANBSplineLayer(1, 1, num_knots=8, spline_order=3, use_bias=True)
    layer.plot_response(input_range=(-8, 8), resolution=100)


if __name__ == "__main__":
    # run_model()
    plot_spline_layer_domain_and_range()
