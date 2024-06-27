from PIL import Image
import numpy as np
from tinygrad import Tensor, dtypes
from typing import List, Tuple
from numba import njit

from shape_checker import check_shapes


@check_shapes(ret=[(None, None), dtypes.float32])
def load_image(image_path: str) -> Tensor:
    """
    Loads an image from the given path and converts it to grayscale.

    Converts the data into a Tensor with values in the range [-1, 1].
    """

    image = Image.open(image_path).convert("L")

    image_np = np.array(image)
    image_np = (image_np / 255.0) * 2 - 1

    image_tensor = Tensor(image_np, dtype=dtypes.float32)

    return image_tensor


@check_shapes([(None, None), dtypes.float32], ret=[(), dtypes.float32])
def get_pixel_value(
    image_tensor: Tensor, x: float, y: float, interpolation="nearest"
) -> Tensor:
    height, width = image_tensor.shape
    x_idx = (x + 1) / 2 * (width - 1)
    y_idx = (y + 1) / 2 * (height - 1)

    if interpolation == "nearest":
        x_idx = int(round(x_idx))
        y_idx = int(round(y_idx))
        pixel_value = image_tensor[y_idx, x_idx]
    elif interpolation == "bilinear":
        x0 = int(np.floor(x_idx))
        x1 = min(x0 + 1, width - 1)
        y0 = int(np.floor(y_idx))
        y1 = min(y0 + 1, height - 1)

        wa = (x1 - x_idx) * (y1 - y_idx)
        wb = (x_idx - x0) * (y1 - y_idx)
        wc = (x1 - x_idx) * (y_idx - y0)
        wd = (x_idx - x0) * (y_idx - y0)

        pixel_value = (
            wa * image_tensor[y0, x0]
            + wb * image_tensor[y0, x1]
            + wc * image_tensor[y1, x0]
            + wd * image_tensor[y1, x1]
        )
    else:
        raise ValueError(f"Unsupported interpolation type: {interpolation}")

    return pixel_value


@njit
def get_pixel_value_np(
    img_data: np.ndarray,
    x: float,
    y: float,
    interpolation="nearest",
) -> float:
    height, width = img_data.shape
    x_idx = (x + 1) / 2 * (width - 1)
    y_idx = (y + 1) / 2 * (height - 1)

    if interpolation == "nearest":
        x_idx = int(round(x_idx))
        y_idx = int(round(y_idx))
        pixel_value = img_data[y_idx, x_idx]
    elif interpolation == "bilinear":
        x0 = int(np.floor(x_idx))
        x1 = min(x0 + 1, width - 1)
        y0 = int(np.floor(y_idx))
        y1 = min(y0 + 1, height - 1)

        wa = (x1 - x_idx) * (y1 - y_idx)
        wb = (x_idx - x0) * (y1 - y_idx)
        wc = (x1 - x_idx) * (y_idx - y0)
        wd = (x_idx - x0) * (y_idx - y0)

        pixel_value = (
            wa * img_data[y0, x0]
            + wb * img_data[y0, x1]
            + wc * img_data[y1, x0]
            + wd * img_data[y1, x1]
        )
    else:
        raise ValueError(f"Unsupported interpolation type: {interpolation}")

    return pixel_value


@check_shapes(
    [(None, None), dtypes.float32],
    [(2,), dtypes.float32],
    ret=[(None,), dtypes.float32],
)
def get_pixel_values(
    image_tensor: Tensor, coords: List[Tuple[float, float]], interpolation="nearest"
) -> Tensor:
    vals = []
    for x, y in coords:
        px_val = get_pixel_value(image_tensor, x, y, interpolation)
        vals.append(px_val)

    return Tensor(vals)


if __name__ == "__main__":
    fname = "/Users/casey/Downloads/smaller.png"
    img = load_image(fname)
    print(img)

    coords = [(0.5, 0.5), (0.25, 0.75), (-0.5, 0.5)]
    vals = get_pixel_values(img, coords, interpolation="bilinear")
    print(vals.numpy())
