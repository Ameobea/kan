from typing import Callable

import numpy as np
from tinygrad import Tensor, dtypes
import matplotlib.pyplot as plt

from shape_checker import check_shape


def plot_target_fn(
    target_fn: Callable[[Tensor], Tensor],
    resolution=100,
    input_range=(-1, 1),
    output_range=(-1, 1),
):
    x = np.linspace(input_range[0], input_range[1], resolution, dtype=np.float32)
    y = np.linspace(input_range[0], input_range[1], resolution, dtype=np.float32)
    X, Y = np.meshgrid(x, y)
    coords = np.stack([X.ravel(), Y.ravel()], axis=1)
    coords = Tensor(coords)
    check_shape(coords, [(resolution**2, 2), dtypes.float32])
    Z = target_fn(coords).reshape(X.shape)
    check_shape(Z, [(resolution, resolution), dtypes.float32])
    Z = Z.numpy()

    plt.contourf(
        X, Y, Z, levels=2, cmap="coolwarm", vmin=output_range[0], vmax=output_range[1]
    )
    plt.colorbar()
    plt.show()
