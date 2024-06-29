from typing import List, Optional, Tuple, Union
import os
import sys

import matplotlib.pyplot as plt
from tinygrad import Tensor, dtypes, TinyJit, nn
import numpy as np
from numba import njit

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from image_parameterization.grid_search import grid_search
from image_parameterization.load_image import get_pixel_value_np, load_image
from image_parameterization.fourier_encoding import encode_coord, encode_coords
from image_parameterization.image_embedding import embed_images
from shape_checker import check_shape, check_shapes
from tiny_kan import (
    KAN,
    BatchKANQuadraticLayer,
    HiddenLayerDef,
    BatchKANCubicBSplineLayer,
    NNLayer,
)

# set numpy rng seed for reproducibility
np.random.seed(0)
Tensor.manual_seed(0)


def load_training_data(
    fname: str = "/home/casey/Downloads/full.png",
    size_px: int = 64,
    slice_count: int = 8,
) -> List[np.ndarray]:
    full_img_tensor = load_image(fname)
    img_width, img_height = full_img_tensor.shape[0], full_img_tensor.shape[1]

    slice_x_coords = np.random.uniform(
        low=0,
        high=img_width - size_px - 1,
        size=slice_count,
    )
    slice_y_coords = np.random.uniform(
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
@njit
def encode_slice_ix(
    slice_ix: Union[int, float], slice_count: int, channels_per_dim=3
) -> np.ndarray:
    # first, map slice ix from [0, slice_count) to [-1, 1)
    slice_ix = 2 * slice_ix / slice_count - 1

    # then, apply the same fourier encoding as we use for coordinates
    return encode_coord(slice_ix, channels_per_dim=channels_per_dim)


# @check_shapes(ret=[(None, None), dtypes.float32])
@njit("float32[:](int32, float32, float32[:,:], float32, float32, int32, int32)")
def encode_input(
    slice_count: int,
    slice_ix: int,
    embedding: np.ndarray,
    x: float,
    y: float,
    slice_ix_channels_per_dim: int = 3,
    coords_channels_per_dim: int = 4,
) -> np.ndarray:
    encoded = np.zeros(
        (slice_ix_channels_per_dim + coords_channels_per_dim * 2,), dtype=np.float32
    )
    if embedding.shape[0] > 0:
        # linear interpolation between embeddings
        slice_ix = min(slice_ix, embedding.shape[0] - 2)
        slice_ix_frac = slice_ix % 1
        encoded[:slice_ix_channels_per_dim] = (1 - slice_ix_frac) * embedding[
            int(slice_ix)
        ] + slice_ix_frac * embedding[int(slice_ix) + 1]
    else:
        encoded[:slice_ix_channels_per_dim] = encode_slice_ix(
            slice_ix, slice_count, channels_per_dim=slice_ix_channels_per_dim
        )
    encoded[slice_ix_channels_per_dim:] = encode_coords(
        x, y, channels_per_dim=coords_channels_per_dim
    )
    # TODO: TEMP WHILE TESTING KAN SPLINE LAYER DOMAIN
    # encoded *= 2
    return encoded


@check_shapes(ret=([(None, None), dtypes.float32]))
@njit
def build_training_inputs(
    training_data: List[np.ndarray],
    batch_size: int,
    embedding: Optional[np.ndarray],
    slice_ix_channels_per_dim: int = 3,
    coords_channels_per_dim: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    input_size = slice_ix_channels_per_dim + coords_channels_per_dim * 2
    encoded_inputs = np.zeros((batch_size, input_size), dtype=np.float32)
    expected_ys = np.zeros((batch_size, 1), dtype=np.float32)

    slice_indices = np.random.randint(0, len(training_data), size=batch_size)

    for i in range(batch_size):
        x, y = np.random.uniform(low=-1, high=1, size=2)

        encoded = encode_input(
            len(training_data),
            slice_indices[i],
            np.zeros((0, 0), dtype=np.float32) if embedding is None else embedding,
            x,
            y,
            slice_ix_channels_per_dim=slice_ix_channels_per_dim,
            coords_channels_per_dim=coords_channels_per_dim,
        )
        encoded_inputs[i] = encoded

        slice_ix = slice_indices[i]
        img_slice = training_data[slice_ix]
        expected_y = get_pixel_value_np(img_slice, x, y, interpolation="bilinear")
        expected_ys[i] = expected_y

    return encoded_inputs, expected_ys


def plot_slice(ax, slice: np.ndarray):
    ax.imshow(slice, cmap="gray")
    ax.set_title("Expected Output")
    ax.axis("off")


def build_slice_eval_inputs(
    slice_count: int,
    slice_ix: int,
    embedding: Optional[np.ndarray],
    slice_ix_channels_per_dim: int = 3,
    coords_channels_per_dim: int = 4,
    resolution=100,
) -> Tuple[Tensor, Tensor]:
    inputs = []
    raw_coords = []
    for y_ix in range(resolution):
        y_coord = 2 * y_ix / resolution - 1
        for x_ix in range(resolution):
            x_coord = 2 * x_ix / resolution - 1
            model_input = encode_input(
                slice_count,
                slice_ix,
                np.zeros((0, 0), dtype=np.float32) if embedding is None else embedding,
                x_coord,
                y_coord,
                slice_ix_channels_per_dim,
                coords_channels_per_dim,
            )
            inputs.append(model_input)
            raw_coords.append((x_coord, y_coord))

    inputs = Tensor(np.array(inputs, dtype=np.float32))
    check_shape(
        inputs,
        [
            (resolution**2, slice_ix_channels_per_dim + coords_channels_per_dim * 2),
            dtypes.float32,
        ],
    )

    raw_coords = Tensor(np.array(raw_coords, dtype=np.float32))
    check_shape(raw_coords, [(resolution**2, 2), dtypes.float32])

    return inputs, raw_coords


def plot_model_response(
    ax,
    slice_count: int,
    slice_ix: int,
    embedding: Optional[np.ndarray],
    model: KAN,
    slice_ix_channels_per_dim: int = 3,
    coords_channels_per_dim: int = 4,
    resolution=100,
):
    inputs, _raw_coords = build_slice_eval_inputs(
        slice_count,
        slice_ix,
        embedding,
        slice_ix_channels_per_dim=slice_ix_channels_per_dim,
        coords_channels_per_dim=coords_channels_per_dim,
        resolution=resolution,
    )

    outputs = model(inputs).numpy().reshape((resolution, resolution))
    im = ax.imshow(outputs, cmap="gray")
    ax.set_title("Model Output")
    ax.axis("off")
    plt.colorbar(im, ax=ax)


def eval_model_full(
    model: KAN,
    embedding: Optional[np.ndarray],
    training_data: List[np.ndarray],
    slice_count: int,
    slice_ix_channels_per_dim: int,
    coords_channels_per_dim: int,
    resolution: int = 100,
) -> float:
    """
    Evaluates the model for `resolution * resolution` points for each slice, returning the average loss.
    """

    total_loss = 0
    for slice_ix in range(slice_count):
        inputs, raw_coords = build_slice_eval_inputs(
            slice_count,
            slice_ix,
            embedding,
            slice_ix_channels_per_dim=slice_ix_channels_per_dim,
            coords_channels_per_dim=coords_channels_per_dim,
            resolution=resolution,
        )

        expected_y = []
        img_slice = training_data[slice_ix]
        for x, y in raw_coords.numpy():
            expected_y.append(
                get_pixel_value_np(img_slice, x, y, interpolation="bilinear")
            )
        expected_y = Tensor(np.array(expected_y, dtype=np.float32))
        check_shape(expected_y, [(resolution**2,), dtypes.float32])
        expected_y = expected_y.unsqueeze(1)
        check_shape(expected_y, [(resolution**2, 1), dtypes.float32])

        y_pred = model(inputs)
        check_shape(y_pred, [(resolution**2, 1), dtypes.float32])

        loss = (y_pred - expected_y).pow(2).mean()
        total_loss += float(loss.numpy())

    return total_loss / slice_count


def plot_expected_vs_actual_for_slice(
    training_data: List[np.ndarray],
    slice_count: int,
    slice_ix: int,
    embedding: Optional[np.ndarray],
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
        embedding,
        model,
        resolution=resolution,
        slice_ix_channels_per_dim=slice_ix_channels_per_dim,
        coords_channels_per_dim=coord_channels_per_dim,
    )

    plt.show()


def build_model(
    training_data: List[np.ndarray],
    use_embedding: bool,
    slice_ix_channels_per_dim: int,
    coord_channels_per_dim: int,
    FirstLayer,
    Layer,
    hidden_layer_defs: List[HiddenLayerDef],
) -> Tuple[KAN, Optional[np.ndarray]]:
    embedding = None
    if use_embedding:
        embedding = embed_images(training_data, n_dims=slice_ix_channels_per_dim)
        # scale embeddings to [-1, 1]
        embedding = (
            2 * (embedding - embedding.min()) / (embedding.max() - embedding.min()) - 1
        )

    input_size = slice_ix_channels_per_dim + coord_channels_per_dim * 2
    model = KAN(
        input_size,
        1,
        hidden_layer_defs,
        FirstLayer=FirstLayer,
        Layer=Layer,
        post_activation_fn=None,
    )

    return model, embedding


def train_model(
    training_data: List[np.ndarray],
    model: KAN,
    embedding: Optional[np.ndarray],
    slice_ix_channels_per_dim: int = 3,
    coord_channels_per_dim: int = 8,
    batch_size: int = 1024 * 4,
    learning_rate: float = 0.005,
    epochs: int = 20_000,
    quiet=False,
) -> Tuple[KAN, Optional[np.ndarray]]:
    all_params = model.get_learnable_params()
    opt = nn.optim.Adam(list(all_params), lr=learning_rate)

    @TinyJit
    @check_shapes(
        [(batch_size, None), dtypes.float32],
        [(batch_size, 1), dtypes.float32],
        ret=[(), dtypes.float32],
    )
    def train_step(encoded_x: Tensor, y_expected: Tensor) -> Tensor:
        with Tensor.train():
            opt.zero_grad()

            y_pred = model(encoded_x)
            check_shape(y_pred, [(batch_size, 1), dtypes.float32])

            loss = (y_pred - y_expected).pow(2).mean()
            loss = loss.backward()
            opt.step()
            return loss

    with Tensor.train():
        for step in range(epochs):
            encoded_x, expected_y = build_training_inputs(
                training_data,
                batch_size,
                embedding=embedding,
                slice_ix_channels_per_dim=slice_ix_channels_per_dim,
                coords_channels_per_dim=coord_channels_per_dim,
            )

            loss = train_step(Tensor(encoded_x), Tensor(expected_y))
            if not quiet:
                print(f"step: {step}, loss: {loss.numpy()}")

    return model, embedding


def train_and_plot_model():
    slice_count = 256
    use_embedding = True
    slice_ix_channels_per_dim = 3
    coord_channels_per_dim = 8
    batch_size = 1024 * 1
    learning_rate = 0.005
    epochs = 200

    hidden_layer_defs = [
        HiddenLayerDef(512),
        HiddenLayerDef(256),
        HiddenLayerDef(128),
        HiddenLayerDef(64),
        HiddenLayerDef(48),
        HiddenLayerDef(32),
    ]

    training_data = load_training_data(slice_count=slice_count)

    model, embedding = train_model(
        training_data,
        use_embedding=use_embedding,
        slice_ix_channels_per_dim=slice_ix_channels_per_dim,
        coord_channels_per_dim=coord_channels_per_dim,
        batch_size=batch_size,
        learning_rate=learning_rate,
        hidden_layer_defs=hidden_layer_defs,
        epochs=epochs,
        quiet=False,
    )

    eval_loss = eval_model_full(
        model, embedding, slice_count, slice_ix_channels_per_dim, coord_channels_per_dim
    )
    print(f"Eval loss: {eval_loss}")

    plot_expected_vs_actual_for_slice(
        training_data,
        slice_count,
        0,
        embedding,
        model,
        slice_ix_channels_per_dim,
        coord_channels_per_dim,
    )

    plot_expected_vs_actual_for_slice(
        training_data,
        slice_count,
        1,
        embedding,
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
        embedding,
        model,
        resolution=100,
        slice_ix_channels_per_dim=slice_ix_channels_per_dim,
        coords_channels_per_dim=coord_channels_per_dim,
    )
    plt.show()


def run_grid_search():
    static_params = {
        "slice_count": 16,
        "use_embedding": True,
        "slice_ix_channels_per_dim": 3,
        "coord_channels_per_dim": 8,
        "batch_size": 512,
        "hidden_layer_defs": [
            # HiddenLayerDef(512),
            # HiddenLayerDef(256),
            HiddenLayerDef(128),
            HiddenLayerDef(64),
            HiddenLayerDef(48),
            HiddenLayerDef(32),
        ],
        "FirstLayer": BatchKANQuadraticLayer,
        "Layer": BatchKANQuadraticLayer,
        "learning_rate": 0.005,
        "epochs": 20_000,
    }

    dynamic_param_bounds = {
        "learning_rate": [0.001, 0.01],
        "epochs": [100, 300, 300],
    }

    grid_search(
        build_model,
        train_model,
        eval_model_full,
        load_training_data,
        "hyperparameter_results.db",
        static_params,
        dynamic_param_bounds,
    )


if __name__ == "__main__":
    # train_and_plot_model()

    run_grid_search()
