import sys
import os
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from tinygrad import Tensor, nn, TinyJit

from tiny_kan import KAN, BatchKANQuadraticLayer, BatchKANCubicLayer, HiddenLayerDef


def encode_coord(
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
        for i in range(channels_per_dim):
            channels.append(np.sin((2**i) * coord))

    return np.array(channels)


def get_random_unencoded_inputs(batch_size=1, input_range=(-1, 1)) -> Tensor:
    return Tensor.uniform((batch_size, 2), low=input_range[0], high=input_range[1])


def encode_inputs(
    coords: np.ndarray, channels_per_dim=4, input_range=(-1, 1)
) -> np.ndarray:
    if channels_per_dim < 2:
        return coords

    inputs = [
        encode_coord(x, y, channels_per_dim=channels_per_dim, input_range=input_range)
        for x, y in coords
    ]
    return np.array(inputs)


def bottom_left_quadrant(coords: Tensor) -> Tensor:
    # expects `coords` to be a tensor of shape (batch_size, 2)
    #
    # binarized so that points inside the circle are 1, outside are -1

    assert len(coords.shape) == 2
    assert coords.shape[1] == 2

    if len(coords.shape) == 1:
        coords = coords.unsqueeze(0)
    elif len(coords.shape) == 0:
        coords = coords.unsqueeze(0).unsqueeze(0)

    # TODO TEMP
    out = ((coords[:, 0] < 0) * (coords[:, 1] < 0)).where(1, -1).unsqueeze(1)
    return out


def circle(coords: Tensor) -> Tensor:
    # SDF-style.  We want to learn a circle centered at the origin with radius 0.5
    #
    # expects `coords` to be a tensor of shape (batch_size, 2)
    #
    # binarized so that points inside the circle are 1, outside are -1

    assert len(coords.shape) == 2
    assert coords.shape[1] == 2

    if len(coords.shape) == 1:
        coords = coords.unsqueeze(0)
    elif len(coords.shape) == 0:
        coords = coords.unsqueeze(0).unsqueeze(0)

    dist = (coords**2).sum(axis=1).sqrt()
    out = (dist < 0.5).where(1, -1).unsqueeze(1)
    return out


target_fn = circle


def plot_target_fn(resolution=100, input_range=(-1, 1)):
    import matplotlib.pyplot as plt

    x = np.linspace(input_range[0], input_range[1], resolution)
    y = np.linspace(input_range[0], input_range[1], resolution)
    X, Y = np.meshgrid(x, y)
    coords = np.stack([X.ravel(), Y.ravel()], axis=1)
    Z = target_fn(Tensor(coords)).reshape(X.shape).numpy()

    plt.contourf(X, Y, Z, levels=2, cmap="coolwarm")
    plt.colorbar()
    plt.show()


def plot_model_response(
    model: KAN, channels_per_dim=4, input_range=(-1, 1), resolution=100
):
    import matplotlib.pyplot as plt

    x = np.linspace(input_range[0], input_range[1], resolution)
    y = np.linspace(input_range[0], input_range[1], resolution)
    X, Y = np.meshgrid(x, y)
    coords = np.stack([X.ravel(), Y.ravel()], axis=1)
    coords = encode_inputs(coords, channels_per_dim=channels_per_dim)
    Z = model(Tensor(coords)).reshape(X.shape).tanh().numpy()

    plt.contourf(X, Y, Z, levels=2, cmap="coolwarm")
    plt.colorbar()
    plt.show()


def train_model():
    channels_per_dim = 8
    input_range = (-1, 1)
    model = KAN(
        2 * channels_per_dim,
        1,
        [HiddenLayerDef(16), HiddenLayerDef(16), HiddenLayerDef(16), HiddenLayerDef(8)],
        Layer=BatchKANCubicLayer,
        layer_params={"use_tanh": False},
    )
    # plot_model_response(model)
    all_params = model.get_learnable_params()

    opt = nn.optim.Adam(list(all_params), lr=0.001)
    batch_size = 256

    @TinyJit
    def train_step(unencoded_x: Tensor, encoded_x: Tensor) -> Tensor:
        with Tensor.train():
            opt.zero_grad()

            y_pred = model(encoded_x).tanh()
            assert y_pred.shape == (batch_size, 1)
            y_actual = target_fn(unencoded_x)
            assert y_actual.shape == (batch_size, 1)

            loss = (y_pred - y_actual).pow(2).mean()
            loss = loss.backward()
            opt.step()
            return loss

    with Tensor.train():
        for step in range(50000):
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
