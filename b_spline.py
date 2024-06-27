# Adapted from https://github.com/KindXiaoming/pykan/blob/master/kan/spline.py

from shape_checker import check_shapes
from tinygrad import Tensor, dtypes


def sigmoid(x: Tensor, k: float = 10.0) -> Tensor:
    return 1 / (1 + (-k * x).exp())


def smooth_step(x: Tensor, grid: Tensor, k: float = 10.0) -> Tensor:
    return sigmoid(x - grid[:, :-1], k) * (1 - sigmoid(x - grid[:, 1:], k))


@check_shapes(
    [(None, None), dtypes.float32],
    [(None, None), dtypes.float32],
    ret=[(None, None, None), dtypes.float32],
)
def B_batch(
    x: Tensor, grid: Tensor, order: int = 0, extend: bool = True, use_sigmoid_trick=True
) -> Tensor:
    """
    evaluate x on B-spline bases

    Args:
    -----
        x : 2D torch.tensor
            inputs, shape (number of splines, number of samples)
        grid : 2D torch.tensor
            grids, shape (number of splines, number of grid points)
        k : int
            the piecewise polynomial order of splines.
        extend : bool
            If True, k points are extended on both ends. If False, no extension (zero boundary condition). Default: True
        use_sigmoid_trick : bool
            If True, work around the non-differentiability of the step function by using the sigmoid function. Default: True

    Returns:
    --------
        spline values : 3D torch.tensor
            shape (number of splines, number of B-spline bases (coeffcients), number of samples). The numbef of B-spline bases = number of grid points + k - 1.

    Example
    -------
    >>> num_spline = 5
    >>> num_sample = 100
    >>> num_grid_interval = 10
    >>> k = 3
    >>> x = torch.normal(0,1,size=(num_spline, num_sample))
    >>> grids = torch.einsum('i,j->ij', torch.ones(num_spline,), torch.linspace(-1,1,steps=num_grid_interval+1))
    >>> B_batch(x, grids, k=k).shape
    """

    # x shape: (size, x); grid shape: (size, grid)
    def extend_grid(grid, k_extend=0):
        # pad k to left and right
        # grid shape: (batch, grid)
        h = (grid[:, [-1]] - grid[:, [0]]) / (grid.shape[1] - 1)

        for _ in range(k_extend):
            grid = (grid[:, [0]] - h).cat(grid, dim=1)
            grid = grid.cat(grid[:, [-1]] + h, dim=1)
        return grid

    if extend == True:
        grid = extend_grid(grid, k_extend=order)

    grid = grid.unsqueeze(dim=2)
    x = x.unsqueeze(dim=1)

    if order == 0:
        if use_sigmoid_trick:
            value = smooth_step(x, grid, k=4)
        else:
            value = (x >= grid[:, :-1]) * (x < grid[:, 1:])
            # ^ this is not differentiable by tinygrad
    else:
        B_km1 = B_batch(
            x[:, 0],
            grid=grid[:, :, 0],
            order=order - 1,
            extend=False,
            use_sigmoid_trick=use_sigmoid_trick,
        )
        value = (x - grid[:, : -(order + 1)]) / (
            grid[:, order:-1] - grid[:, : -(order + 1)]
        ) * B_km1[:, :-1] + (grid[:, order + 1 :] - x) / (
            grid[:, order + 1 :] - grid[:, 1:(-order)]
        ) * B_km1[
            :, 1:
        ]
    return value


@check_shapes(
    [(None, None), dtypes.float32],
    [(None, None), dtypes.float32],
    [(None, None), dtypes.float32],
    ret=[(None, None), dtypes.float32],
)
def coef2curve(
    x_eval: Tensor, grid: Tensor, coef: Tensor, order: int, use_sigmoid_trick=True
) -> Tensor:
    """
    converting B-spline coefficients to B-spline curves. Evaluate x on B-spline curves (summing up B_batch results over B-spline basis).

    Args:
    -----
        x_eval : 2D torch.tensor)
            shape (number of splines, number of samples)
        grid : 2D torch.tensor)
            shape (number of splines, number of grid points)
        coef : 2D torch.tensor)
            shape (number of splines, number of coef params). number of coef params = number of grid intervals + order
        order : int
            the piecewise polynomial order of splines.
        use_sigmoid_trick : bool
            If True, work around the non-differentiability of the step function by using the sigmoid function. Default: True

    Returns:
    --------
        y_eval : 2D torch.tensor
            shape (number of splines, number of samples)

    Example
    -------
    >>> num_spline = 5
    >>> num_sample = 100
    >>> num_grid_interval = 10
    >>> k = 3
    >>> x_eval = torch.normal(0,1,size=(num_spline, num_sample))
    >>> grids = torch.einsum('i,j->ij', torch.ones(num_spline,), torch.linspace(-1,1,steps=num_grid_interval+1))
    >>> coef = torch.normal(0,1,size=(num_spline, num_grid_interval+k))
    >>> coef2curve(x_eval, grids, coef, k=k).shape
    torch.Size([5, 100])
    """
    # x_eval: (size, batch), grid: (size, grid), coef: (size, coef)
    # coef: (size, coef), B_batch: (size, coef, batch), summer over coef

    return Tensor.einsum(
        "ij,ijk->ik",
        coef,
        B_batch(x_eval, grid, order, use_sigmoid_trick=use_sigmoid_trick),
    )


def plot_random_spline():
    import matplotlib.pyplot as plt
    import numpy as np

    num_spline = 1
    num_sample = 1000
    num_grid_interval = 10
    spline_order = 1
    domain = (-1, 1)
    domain_width = domain[1] - domain[0]
    x_eval = Tensor(
        np.linspace(domain[0] - domain_width, domain[1] + domain_width, num_sample)
    ).reshape(num_spline, num_sample)

    grids = Tensor.einsum(
        "i,j->ij",
        Tensor.ones(num_spline),
        Tensor(np.linspace(domain[0], domain[1], num_grid_interval + 1)),
    )
    coef = Tensor.normal(num_spline, num_grid_interval + spline_order, mean=0, std=1)
    y_eval = coef2curve(x_eval, grids, coef, order=spline_order)

    x_eval = x_eval[0].numpy()
    y_eval = y_eval[0].numpy()
    plt.plot(x_eval, y_eval, "o")
    plt.show()
