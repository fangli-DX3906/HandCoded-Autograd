import numpy as np

from core import Tensor


def sum_to(x: Tensor, shape: tuple) -> Tensor:
    ndim = len(shape)
    lead = x.ndim - ndim
    lead_axis = tuple(range(lead))

    axis = tuple([i + lead for i, ax in enumerate(shape) if ax == 1])
    y = np.sum(x.data, lead_axis + axis, keepdims=True)
    if lead > 0:
        y = np.squeeze(y.data, lead_axis)

    return Tensor(y)


if __name__ == '__main__':
    a = Tensor(
        np.array([
            [1, 2, 3], [3, 4, 5]
        ])
    )
    y = sum_to(a, (3, 1, 1))
    print(y)
