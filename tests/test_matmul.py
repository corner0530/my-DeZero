import numpy as np
import pytest

import dezero.functions as F
from dezero import Variable
from dezero.utils import array_allclose, gradient_check


def test_forward1():
    x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
    w = Variable(x.data.T)
    y = F.matmul(x, w)
    res = y.data
    expected = np.array([[14, 32], [32, 77]])
    assert array_allclose(res, expected)


def test_backward1():
    x = np.random.randn(3, 2)
    w = np.random.randn(2, 3)
    f = lambda x: F.matmul(x, Variable(w))
    assert gradient_check(f, x)


def test_backward2():
    x_data = np.random.randn(10, 1)
    w_data = np.random.randn(1, 5)
    f = lambda w: F.matmul(Variable(x_data), w)
    assert gradient_check(f, w_data)


if __name__ == "__main__":
    pytest.main()
