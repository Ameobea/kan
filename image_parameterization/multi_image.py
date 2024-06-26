from typing import List, Tuple, Union
import os
import sys

import matplotlib.pyplot as plt
from tinygrad import Tensor, dtypes, TinyJit, nn
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from image_parameterization.load_image import get_pixel_value_np, load_image
from image_parameterization.fourier_encoding import encode_coord, encode_coords
from shape_checker import check_shape, check_shapes
from tiny_kan import KAN, BatchKANQuadraticLayer, HiddenLayerDef
from tiny_nn import TinyNN


def load_training_data(
    fname: str = "/Users/casey/Downloads/full.png",
    size_px: int = 64,
    slice_count: int = 8,
    rng_seed: int = 0,
) -> List[np.ndarray]:
    full_img_tensor = load_image(fname)
    img_width, img_height = full_img_tensor.shape[0], full_img_tensor.shape[1]

    rng = np.random.default_rng(rng_seed)
    slice_x_coords = rng.uniform(
        low=0,
        high=img_width - size_px - 1,
        size=slice_count,
    )
    slice_y_coords = rng.uniform(
        low=0,
        high=img_height - size_px - 1,
        size=slice_count,
    )

    slices = []
    for x, y in zip(slice_x_coords, slice_y_coords):
        img_slice = full_img_tensor[
            int(x) : int(x) + size_px,
            int(y) : int(y) + size_px,
        ]
        check_shape(img_slice, [(size_px, size_px), dtypes.float32])
        slices.append(img_slice.numpy())

    return slices


# Acts as a sort of psuedo-embedding for slices
def encode_slice_ix(
    slice_ix: Union[int, float], slice_count: int, channels_per_dim=3
) -> np.ndarray:
    # first, map slice ix from [0, slice_count) to [-1, 1)
    slice_ix = 2 * slice_ix / slice_count - 1

    # then, apply the same fourier encoding as we use for coordinates
    return encode_coord(slice_ix, channels_per_dim=channels_per_dim)


rng = np.random.default_rng(0)


@check_shapes(ret=[(None, None), dtypes.float32])
def encode_input(
    slice_count: int,
    slice_ix: int,
    x: float,
    y: float,
    slice_ix_channels_per_dim: int = 3,
    coords_channels_per_dim: int = 4,
) -> np.ndarray:
    encoded_slice_ix = encode_slice_ix(
        slice_ix, slice_count, channels_per_dim=slice_ix_channels_per_dim
    )
    coords = encode_coords(x, y, channels_per_dim=coords_channels_per_dim)
    return np.concatenate([encoded_slice_ix, coords], axis=0)


@check_shapes(ret=([(None, None), dtypes.float32]))
def build_training_inputs(
    training_data: List[np.ndarray],
    batch_size: int,
    slice_ix_channels_per_dim: int = 3,
    coords_channels_per_dim: int = 4,
) -> Tuple[Tensor, Tensor, Tensor]:
    input_size = slice_ix_channels_per_dim + coords_channels_per_dim * 2
    raw_inputs = np.zeros((batch_size, 3), dtype=np.float32)
    encoded_inputs = np.zeros((batch_size, input_size), dtype=np.float32)
    expected_ys = np.zeros((batch_size, 1), dtype=np.float32)

    slice_indices = rng.integers(0, len(training_data), size=batch_size)

    for i in range(batch_size):
        x, y = rng.uniform(low=-1, high=1, size=2)

        encoded_inputs[i] = encode_input(
            len(training_data),
            slice_indices[i],
            x,
            y,
            slice_ix_channels_per_dim=slice_ix_channels_per_dim,
            coords_channels_per_dim=coords_channels_per_dim,
        )

        slice_ix = slice_indices[i]
        raw_inputs[i, 0] = slice_ix
        raw_inputs[i, 1] = x
        raw_inputs[i, 2] = y

        img_slice = training_data[slice_ix]
        expected_y = get_pixel_value_np(img_slice, x, y, interpolation="bilinear")
        expected_ys[i] = float(expected_y)

    return Tensor(raw_inputs), Tensor(encoded_inputs), Tensor(expected_ys)


def plot_slice(ax, slice: np.ndarray):
    ax.imshow(slice, cmap="gray")
    ax.set_title("Expected Output")
    ax.axis("off")


def plot_model_response(
    ax,
    slice_count: int,
    slice_ix: int,
    model: KAN,
    slice_ix_channels_per_dim: int = 3,
    coords_channels_per_dim: int = 4,
    resolution=100,
):
    inputs = []
    for y_ix in range(resolution):
        y_coord = 2 * y_ix / resolution - 1
        for x_ix in range(resolution):
            x_coord = 2 * x_ix / resolution - 1
            model_input = encode_input(
                slice_count,
                slice_ix,
                x_coord,
                y_coord,
                slice_ix_channels_per_dim,
                coords_channels_per_dim,
            )
            inputs.append(model_input)

    inputs = Tensor(np.array(inputs, dtype=np.float32))
    check_shape(
        inputs,
        [
            (resolution**2, slice_ix_channels_per_dim + coords_channels_per_dim * 2),
            dtypes.float32,
        ],
    )

    outputs = model(inputs).numpy().reshape((resolution, resolution))
    im = ax.imshow(outputs, cmap="gray")
    ax.set_title("Model Output")
    ax.axis("off")
    plt.colorbar(im, ax=ax)


def plot_expected_vs_actual_for_slice(
    training_data: List[np.ndarray],
    slice_count: int,
    slice_ix: int,
    model: KAN,
    slice_ix_channels_per_dim: int,
    coord_channels_per_dim: int,
    resolution=100,
):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    expected_slice = training_data[slice_ix]
    plot_slice(axs[0], expected_slice)
    plot_model_response(
        axs[1],
        slice_count,
        slice_ix,
        model,
        resolution=resolution,
        slice_ix_channels_per_dim=slice_ix_channels_per_dim,
        coords_channels_per_dim=coord_channels_per_dim,
    )

    plt.show()


if __name__ == "__main__":
    slice_count = 8
    training_data = load_training_data(slice_count=slice_count)

    slice_ix_channels_per_dim = 3
    coord_channels_per_dim = 8
    input_size = slice_ix_channels_per_dim + coord_channels_per_dim * 2
    model = KAN(
        input_size,
        1,
        [
            HiddenLayerDef(1024),
            HiddenLayerDef(512),
            HiddenLayerDef(256),
            HiddenLayerDef(256),
            HiddenLayerDef(256),
            HiddenLayerDef(256),
        ],
        Layer=BatchKANQuadraticLayer,
        post_activation_fn="tanh",
    )

    # model = TinyNN(input_size, 1, [1024, 1024, 512, 512, 512, 512])

    all_params = model.get_learnable_params()
    print("PARAM COUNT: ", model.param_count())
    # raise 1

    opt = nn.optim.Adam(list(all_params), lr=0.005)
    batch_size = 1024

    @TinyJit
    @check_shapes(
        [(batch_size, 3), dtypes.float32],
        [(batch_size, None), dtypes.float32],
        [(batch_size, 1), dtypes.float32],
        ret=[(), dtypes.float32],
    )
    def train_step(
        unencoded_x: Tensor, encoded_x: Tensor, y_expected: Tensor
    ) -> Tensor:
        with Tensor.train():
            opt.zero_grad()

            y_pred = model(encoded_x)
            check_shape(y_pred, [(batch_size, 1), dtypes.float32])

            loss = (y_pred - y_expected).pow(2).mean()
            loss = loss.backward()
            opt.step()
            return loss

    with Tensor.train():
        for step in range(2000):
            raw_x, encoded_x, expected_y = build_training_inputs(
                training_data,
                batch_size,
                slice_ix_channels_per_dim=slice_ix_channels_per_dim,
                coords_channels_per_dim=coord_channels_per_dim,
            )

            loss = train_step(
                raw_x.realize(), encoded_x.realize(), expected_y.realize()
            )
            print(f"step: {step}, loss: {loss.numpy()}")

    plot_expected_vs_actual_for_slice(
        training_data,
        slice_count,
        0,
        model,
        slice_ix_channels_per_dim,
        coord_channels_per_dim,
    )

    plot_expected_vs_actual_for_slice(
        training_data,
        slice_count,
        1,
        model,
        slice_ix_channels_per_dim,
        coord_channels_per_dim,
    )

    # plot something in the middle/out of distribution
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    plot_model_response(
        ax,
        slice_count,
        0.5,
        model,
        resolution=100,
        slice_ix_channels_per_dim=slice_ix_channels_per_dim,
        coords_channels_per_dim=coord_channels_per_dim,
    )
    plt.show()
