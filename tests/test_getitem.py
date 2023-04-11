import numpy as np
import pytest

import dezero.functions as F
from dezero import Variable
from dezero.utils import array_allclose, gradient_check


def test_forward1():
    x_data = np.arange(12).reshape((2, 2, 3))
    x = Variable(x_data)
    y = F.get_item(x, 0)
    assert array_allclose(y.data, x_data[0])


def test_forward1a():
    x_data = np.arange(12).reshape((2, 2, 3))
    x = Variable(x_data)
    y = x[0]
    assert array_allclose(y.data, x_data[0])


def test_forward2():
    x_data = np.arange(12).reshape((2, 2, 3))
    x = Variable(x_data)
    y = F.get_item(x, (0, 0, slice(0, 2, 1)))
    assert array_allclose(y.data, x_data[0, 0, 0:2:1])


def test_forward3():
    x_data = np.arange(12).reshape((2, 2, 3))
    x = Variable(x_data)
    y = F.get_item(x, (Ellipsis, 2))
    assert array_allclose(y.data, x_data[..., 2])


def test_backward1():
    x_data = np.array([[1, 2, 3], [4, 5, 6]])
    slices = 1
    f = lambda x: F.get_item(x, slices)
    assert gradient_check(f, x_data)


def test_backward2():
    x_data = np.arange(12).reshape(4, 3)
    slices = slice(1, 3)
    f = lambda x: F.get_item(x, slices)
    assert gradient_check(f, x_data)


if __name__ == "__main__":
    pytest.main()
