from typing import List, Tuple
import os
import sys


from tinygrad import Tensor, dtypes, TinyJit, nn
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from image_parameterization.load_image import get_pixel_value, load_image
from image_parameterization.fourier_encoding import encode_coord, encode_coords
from shape_checker import check_shape, check_shapes
from tiny_kan import KAN, BatchKANQuadraticLayer, HiddenLayerDef


def load_training_data(
    fname: str = "/Users/casey/Downloads/full.png",
    size_px: int = 64,
    slice_count: int = 8,
    rng_seed: int = 0,
) -> List[Tensor]:
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
        ].realize()
        check_shape(img_slice, [(size_px, size_px), dtypes.float32])
        slices.append(img_slice)

    return slices


# Acts as a sort of psuedo-embedding for slices
def encode_slice_ix(slice_ix: int, slice_count: int, channels_per_dim=3) -> np.ndarray:
    # first, map slice ix from [0, slice_count) to [-1, 1)
    slice_ix = 2 * slice_ix / slice_count - 1

    # then, apply the same fourier encoding as we use for coordinates
    return encode_coord(slice_ix, channels_per_dim=channels_per_dim)


@check_shapes(ret=([(None, None), dtypes.float32]))
def build_training_inputs(
    training_data: List[Tensor],
    batch_size: int,
    slice_ix_channels_per_dim: int = 3,
    coords_channels_per_dim: int = 4,
    rng_seed: int = 0,
) -> Tuple[Tensor, Tensor, Tensor]:
    rng = np.random.default_rng(rng_seed)

    slice_indices = rng.integers(0, len(training_data), size=batch_size)
    encoded_slice_indices = [
        encode_slice_ix(ix, len(training_data)) for ix in slice_indices
    ]
    input_size = slice_ix_channels_per_dim + coords_channels_per_dim * 2

    raw_inputs = np.zeros((batch_size, 3), dtype=np.float32)
    encoded_inputs = np.zeros((batch_size, input_size), dtype=np.float32)
    expected_ys = np.zeros((batch_size, 1), dtype=np.float32)

    for i in range(batch_size):
        encoded_slice_ix = encoded_slice_indices[i]
        encoded_inputs[i, :slice_ix_channels_per_dim] = encoded_slice_ix

        x, y = rng.uniform(low=-1, high=1, size=2)
        coords = encode_coords(x, y, channels_per_dim=coords_channels_per_dim)
        encoded_inputs[i, slice_ix_channels_per_dim:] = coords

        slice_ix = slice_indices[i]
        raw_inputs[i, 0] = slice_ix
        raw_inputs[i, 1] = x
        raw_inputs[i, 2] = y

        img_slice = training_data[slice_ix]
        expected_y = get_pixel_value(img_slice, x, y, interpolation="bilinear")
        expected_ys[i] = float(expected_y.numpy())

    return Tensor(raw_inputs), Tensor(encoded_inputs), Tensor(expected_ys)


if __name__ == "__main__":
    training_data = load_training_data()
    print(training_data)

    slice_ix_channels_per_dim = 3
    coord_channels_per_dim = 4
    input_size = slice_ix_channels_per_dim + coord_channels_per_dim * 2
    model = KAN(
        input_size,
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

    training_data = load_training_data()

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
        for step in range(20000):
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

    # plot_target_fn(input_range=input_range)
    # plot_model_response(
    #     model,
    #     channels_per_dim=channels_per_dim,
    #     resolution=100,
    #     input_range=input_range,
    # )
