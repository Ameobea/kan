import numpy as np


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


def encode_coord(x: float, channels_per_dim=4, input_range=(-1, 1)) -> np.ndarray:
    if channels_per_dim < 2:
        return x

    x = (x - input_range[0]) / (input_range[1] - input_range[0]) * 2 * np.pi

    channels = [x]
    for i in range(channels_per_dim - 1):
        channels.append(np.sin((2**i) * x))

    return np.array(channels, dtype=np.float32)


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
