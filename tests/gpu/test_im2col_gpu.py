import cupy as np  # !! CUPY !!
import pytest

import dezero.functions as F
from dezero.utils import array_equal, gradient_check


def test_forward1():
    n, c, h, w = 1, 1, 3, 3
    x = np.arange(n * c * h * w).reshape((n, c, h, w))
    y = F.im2col(x, 3, 3, 0, to_matrix=True)
    expected = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8]])

    res = array_equal(y.data, expected)
    assert res


def test_backward1():
    n, c, h, w = 1, 1, 3, 3
    x = np.arange(n * c * h * w).reshape((n, c, h, w))
    f = lambda x: F.im2col(x, 3, 3, 0, to_matrix=True)
    assert gradient_check(f, x)


def test_backward2():
    n, c, h, w = 1, 1, 3, 3
    x = np.arange(n * c * h * w).reshape((n, c, h, w))
    f = lambda x: F.im2col(x, 3, 3, 0, to_matrix=False)
    assert gradient_check(f, x)


if __name__ == "__main__":
    pytest.main()
