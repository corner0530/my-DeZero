import numpy as np
import pytest

import dezero.functions as F
from dezero import Variable
from dezero.utils import array_equal, gradient_check


def test_forward1():
    x0 = np.array([1, 2, 3])
    x1 = Variable(np.array([1, 2, 3]))
    y = x0 + x1
    res = y.data
    expected = np.array([2, 4, 6])
    assert array_equal(res, expected)


def test_datatype():
    """np.float64ではなく、0次元のndarrayを返すかどうか"""
    x = Variable(np.array(2.0))
    y = x**2
    assert not np.isscalar(y)


def test_backward1():
    x = Variable(np.random.randn(3, 3))
    y = np.random.randn(3, 3)
    f = lambda x: x + y
    assert gradient_check(f, x)


def test_backward2():
    x = Variable(np.random.randn(3, 3))
    y = np.random.randn(3, 1)
    f = lambda x: x + y
    assert gradient_check(f, x)


def test_backward3():
    x = np.random.randn(3, 3)
    y = np.random.randn(3, 1)
    assert gradient_check(F.add, x, y)


if __name__ == "__main__":
    pytest.main()
