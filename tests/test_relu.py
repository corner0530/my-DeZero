import chainer.functions as CF
import numpy as np
import pytest

import dezero.functions as F
from dezero.utils import array_allclose, gradient_check


def test_forward1():
    x = np.array([[-1, 0], [2, -3], [-2, 1]], np.float32)
    res = F.relu(x)
    ans = np.array([[0, 0], [2, 0], [0, 1]], np.float32)
    assert array_allclose(res, ans)


def test_backward1():
    x_data = np.array([[-1, 1, 2], [-1, 2, 4]])
    assert gradient_check(F.relu, x_data)


def test_backward2():
    np.random.seed(0)
    x_data = np.random.rand(10, 10) * 100
    assert gradient_check(F.relu, x_data)


def test_backward3():
    np.random.seed(0)
    x_data = np.random.rand(10, 10, 10) * 100
    assert gradient_check(F.relu, x_data)


if __name__ == "__main__":
    pytest.main()
