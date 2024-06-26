import sys
import os
import numpy as np


from tinygrad import Tensor, nn, TinyJit, dtypes

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from image_parameterization.fourier_encoding import encode_inputs
from image_parameterization.load_image import load_image
from image_parameterization.plotting import plot_target_fn


from tiny_kan import (
    KAN,
    BatchKANQuadraticLayer,
    BatchKANCubicLayer,
    BatchKANLinearLayer,
    HiddenLayerDef,
)
from shape_checker import check_shapes, check_shape
from tiny_nn import TinyNN


@check_shapes(ret=[(None, 2), dtypes.float32])
def get_random_unencoded_inputs(batch_size=1, input_range=(-1, 1)) -> Tensor:
    return Tensor.uniform((batch_size, 2), low=input_range[0], high=input_range[1])


@check_shapes([(None, 2), dtypes.float32], ret=[(None, 1), dtypes.float32])
def bottom_left_quadrant(coords: Tensor) -> Tensor:
    # expects `coords` to be a tensor of shape (batch_size, 2)
    #
    # binarized so that points inside the circle are 1, outside are -1

    if len(coords.shape) == 1:
        coords = coords.unsqueeze(0)
    elif len(coords.shape) == 0:
        coords = coords.unsqueeze(0).unsqueeze(0)

    return ((coords[:, 0] < 0) * (coords[:, 1] < 0)).where(1, -1).unsqueeze(1)


@check_shapes([(None, 2), dtypes.float32], ret=[(None, 1), dtypes.float32])
def circle(coords: Tensor) -> Tensor:
    # SDF-style.  We want to learn a circle centered at the origin with radius 0.5
    #
    # expects `coords` to be a tensor of shape (batch_size, 2)
    #
    # binarized so that points inside the circle are 1, outside are -1

    if len(coords.shape) == 1:
        coords = coords.unsqueeze(0)
    elif len(coords.shape) == 0:
        coords = coords.unsqueeze(0).unsqueeze(0)

    dist = (coords**2).sum(axis=1).sqrt()
    out = (dist < 0.5).where(1, -1).unsqueeze(1)
    return out


@check_shapes([(None, 2), dtypes.float32], ret=[(None, 1), dtypes.float32])
def ridges(coords: Tensor) -> Tensor:
    if len(coords.shape) == 1:
        coords = coords.unsqueeze(0)
    elif len(coords.shape) == 0:
        coords = coords.unsqueeze(0).unsqueeze(0)

    scaled = coords[:, 0].abs() * 5.0
    return (scaled - scaled.trunc() > 0.5).unsqueeze(1)


@check_shapes([(None, 2), dtypes.float32], ret=[(None, 1), dtypes.float32])
def stitches(coords: Tensor) -> Tensor:
    if len(coords.shape) == 1:
        coords = coords.unsqueeze(0)
    elif len(coords.shape) == 0:
        coords = coords.unsqueeze(0).unsqueeze(0)

    scaled = coords[:, 0].abs() * 5.0
    out = scaled - scaled.trunc() > 0.5
    out = out * (coords[:, 1] < 0.5) * (coords[:, 1] > -0.5)
    return out.unsqueeze(1).where(1.0, -1.0)


img_tensor = load_image("/Users/casey/Downloads/smaller.png")


@check_shapes([(None, 2), dtypes.float32], ret=[(None, 1), dtypes.float32])
def real_image_test(coords: Tensor) -> Tensor:
    # convert from [-1, 1] to [0, img_tensor.shape[0] - 1]
    coords = (coords + 1) / 2 * (img_tensor.shape[0] - 1)

    return img_tensor[
        coords[:, 1].cast(dtypes.int32), coords[:, 0].cast(dtypes.int32)
    ].unsqueeze(1)


target_fn = real_image_test


def plot_model_response(
    model: KAN,
    channels_per_dim=4,
    input_range=(-1, 1),
    output_range=(-1, 1),
    resolution=100,
):
    import matplotlib.pyplot as plt

    x = np.linspace(input_range[0], input_range[1], resolution, dtype=np.float32)
    y = np.linspace(input_range[0], input_range[1], resolution, dtype=np.float32)
    X, Y = np.meshgrid(x, y)
    coords = np.stack([X.ravel(), Y.ravel()], axis=1)
    coords = encode_inputs(coords, channels_per_dim=channels_per_dim)
    Z = model(Tensor(coords)).reshape(X.shape).numpy()

    plt.contourf(
        X, Y, Z, levels=2, cmap="coolwarm", vmin=output_range[0], vmax=output_range[1]
    )
    plt.colorbar()
    plt.show()


def train_model():
    channels_per_dim = 6
    input_range = (-1, 1)
    # \/ worked for the stitches example; KAN seemed to have a small advantage
    # to the NN when comparing by param count, but it's possible that different
    # architectures or hyperparams would have changed that
    # model = KAN(
    #     2 * channels_per_dim,
    #     1,
    #     [
    #         HiddenLayerDef(6),
    #         HiddenLayerDef(2),
    #     ],
    #     Layer=BatchKANQuadraticLayer,
    #     post_activation_fn="tanh",
    # )

    # model = TinyNN(2 * channels_per_dim, 1, [4, 4, 4, 4, 4], "tanh")

    model = KAN(
        2 * channels_per_dim,
        1,
        [
            HiddenLayerDef(64),
            HiddenLayerDef(32),
            HiddenLayerDef(32),
            HiddenLayerDef(16),
        ],
        Layer=BatchKANQuadraticLayer,
    )

    all_params = model.get_learnable_params()
    print("PARAM COUNT: ", model.param_count())
    # raise 1

    opt = nn.optim.Adam(list(all_params), lr=0.005)
    batch_size = 512

    @TinyJit
    @check_shapes(
        [(batch_size, 2), dtypes.float32],
        [(batch_size, None), dtypes.float32],
        ret=[(), dtypes.float32],
    )
    def train_step(unencoded_x: Tensor, encoded_x: Tensor) -> Tensor:
        with Tensor.train():
            opt.zero_grad()

            y_pred = model(encoded_x)
            check_shape(y_pred, [(batch_size, 1), dtypes.float32])
            y_actual = target_fn(unencoded_x)
            check_shape(y_actual, [(batch_size, 1), dtypes.float32])

            loss = (y_pred - y_actual).pow(2).mean()
            loss = loss.backward()
            opt.step()
            return loss

    with Tensor.train():
        for step in range(20000):
            x = get_random_unencoded_inputs(batch_size, input_range=input_range)
            encoded_x = encode_inputs(
                x.numpy(), channels_per_dim=channels_per_dim, input_range=input_range
            )

            loss = train_step(x, Tensor(encoded_x))
            print(f"step: {step}, loss: {loss.numpy()}")

    plot_target_fn(input_range=input_range)
    plot_model_response(
        model,
        channels_per_dim=channels_per_dim,
        resolution=100,
        input_range=input_range,
    )


if __name__ == "__main__":
    train_model()
