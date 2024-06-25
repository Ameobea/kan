import functools
import inspect

from tinygrad import Tensor, dtypes


def check_shape(t: Tensor, shape):
    dtype = None
    if isinstance(shape, list):
        shape, dtype = shape

    assert len(t.shape) == len(
        shape
    ), f"Expected shape {shape} for tensor, got {t.shape}"

    for a, b in zip(t.shape, shape):
        if b is None or b < 0:
            continue

        assert a == b, f"Expected shape {shape} for tensor, got {t.shape}"

    if dtype is not None:
        assert t.dtype == dtype, f"Expected dtype {dtype} for tensor, got {t.dtype}"


def check_shapes(*dec_args, **dec_kwargs):
    """
    A decorator that verifies that the shapes of all arguments that are tensors
    match the shapes provided.

    `None` is used for dimensions that can be of any size.

    If a special kwarg called `ret` is provided, that shape will be used to check
    the return value of the function.

    Example:
    ```
    @check_shape((None, 2), [(None,), dtypes.float32], my_kwarg=(None, 3))
    def my_function(coords, non_tensor_arg, values, my_kwarg=None):
      pass
    ```
    """

    dec_arg_count = len(dec_args)
    return_shape = dec_kwargs.get("ret")

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            sig = inspect.signature(fn)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            arg_ix = 0
            for arg in args:
                if not isinstance(arg, Tensor):
                    continue

                expected_shape = dec_args[arg_ix]

                check_shape(arg, expected_shape)
                arg_ix += 1
                if arg_ix >= dec_arg_count:
                    break

            for param_name, arg in kwargs.items():
                if param_name not in bound.arguments:
                    continue

                expected_shape = dec_kwargs.get(param_name)
                if expected_shape is None:
                    continue

                if isinstance(arg, Tensor):
                    check_shape(arg, expected_shape)

            out = fn(*args, **kwargs)
            if return_shape is None:
                return out

            if isinstance(out, Tensor):
                check_shape(out, return_shape)

            return out

        return wrapper

    return decorator


if __name__ == "__main__":

    @check_shapes((None, 2), [(None, 3), dtypes.int32], kwtensor=(None, 2, None))
    def test_fn(t1: Tensor, whatever: int, t2: Tensor, kwtensor: Tensor = None):
        pass

    test_fn(Tensor.zeros(3, 2), 1, Tensor.zeros(3, 3), kwtensor=Tensor.zeros(1, 2, 4))
