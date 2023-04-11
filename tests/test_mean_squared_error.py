import numpy as np
import pytest

import dezero.functions as F
from dezero.utils import array_allclose, gradient_check


def test_forward1():
    x0 = np.array([0.0, 1.0, 2.0])
    x1 = np.array([0.0, 1.0, 2.0])
    expected = ((x0 - x1) ** 2).sum() / x0.size
    y = F.mean_squared_error(x0, x1)
    assert array_allclose(y.data, expected)


def test_backward1():
    x0 = np.random.rand(10)
    x1 = np.random.rand(10)
    f = lambda x0: F.mean_squared_error(x0, x1)
    assert gradient_check(f, x0)


def test_backward2():
    x0 = np.random.rand(100)
    x1 = np.random.rand(100)
    f = lambda x0: F.mean_squared_error(x0, x1)
    assert gradient_check(f, x0)


if __name__ == "__main__":
    pytest.main()
