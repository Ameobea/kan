import numpy as np
from numba import njit
from tinygrad import Tensor, dtypes
from functools import lru_cache

from shape_checker import check_shapes


@njit
def encode_coords(
    x: float, y: float, channels_per_dim=4, input_range=(-1, 1)
) -> np.ndarray:
    """
    Uses Fourier feature encoding to expand a 2D coordinate into multiple channels
    to make it easier for models to learn high-frequency patterns.
    """

    x = (x - input_range[0]) / (input_range[1] - input_range[0]) * 2 * np.pi
    y = (y - input_range[0]) / (input_range[1] - input_range[0]) * 2 * np.pi

    channels = []
    for coord in [x, y]:
        channels.append(coord)
        for i in range(channels_per_dim - 1):
            channels.append(np.sin((2**i) * coord))

    return np.array(channels, dtype=np.float32)


@njit
def encode_coord(x: float, channels_per_dim=4, input_range=(-1, 1)) -> np.ndarray:
    if channels_per_dim < 2:
        return np.array([x], dtype=np.float32)

    x = (x - input_range[0]) / (input_range[1] - input_range[0]) * 2 * np.pi

    channels = [x]
    for i in range(channels_per_dim - 1):
        channels.append(np.sin((2**i) * x))

    return np.array(channels, dtype=np.float32)


@lru_cache
# @check_shapes(ret=[(None,), dtypes.float32])
def get_pows_tensor(channels_per_dim: int) -> Tensor:
    return Tensor(
        np.array([2**i for i in range(channels_per_dim - 1)], dtype=np.float32)
    )


@check_shapes([(None, 2), dtypes.float32], ret=[(None, None), dtypes.float32])
def encode_coords_tensor(
    coords: Tensor, channels_per_dim=4, input_range=(-1, 1)
) -> Tensor:
    if channels_per_dim < 2:
        return coords

    # raw_xs = (
    #     (coords[:, 0] - input_range[0]) / (input_range[1] - input_range[0]) * 2 * np.pi
    # )
    # xs = raw_xs.unsqueeze(1).expand(-1, channels_per_dim - 1)
    # raw_ys = (
    #     (coords[:, 1] - input_range[0]) / (input_range[1] - input_range[0]) * 2 * np.pi
    # )
    # ys = raw_ys.unsqueeze(1).expand(-1, channels_per_dim - 1)

    scaled_coords = (
        (coords - input_range[0]) / (input_range[1] - input_range[0]) * 2 * np.pi
    )
    raw_xs = scaled_coords[:, 0]
    raw_ys = scaled_coords[:, 1]
    xs = raw_xs.unsqueeze(1).expand(-1, channels_per_dim - 1)
    ys = raw_ys.unsqueeze(1).expand(-1, channels_per_dim - 1)

    pows = get_pows_tensor(channels_per_dim).unsqueeze(0).expand(xs.shape[0], -1)
    powd_xs = (xs * pows).sin()
    powd_ys = (ys * pows).sin()

    # stack x, pow'd xs, y, pow'd ys along last dimension
    return raw_xs.unsqueeze(1).cat(powd_xs, raw_ys.unsqueeze(1), powd_ys, dim=1)


@njit
def encode_inputs(
    coords: np.ndarray, channels_per_dim=4, input_range=(-1, 1)
) -> np.ndarray:
    if channels_per_dim < 2:
        return coords

    inputs = [
        encode_coords(x, y, channels_per_dim=channels_per_dim, input_range=input_range)
        for x, y in coords
    ]
    return np.array(inputs, dtype=np.float32)


if __name__ == "__main__":
    coords = Tensor(np.array([[-1.0, 0.2], [0.3, 0.4], [0.5, 1]], dtype=np.float32))
    old_v = encode_coords_tensor(coords, channels_per_dim=4).numpy()

    new_v = encode_inputs(coords.numpy())

    np.set_printoptions(suppress=True)
    print(old_v - new_v)
