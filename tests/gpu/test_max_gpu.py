import cupy as np  # !! CUPY !!
import pytest

import dezero.functions as F
from dezero import Variable
from dezero.utils import array_allclose, gradient_check


def test_forward1():
    x = Variable(np.random.rand(10))
    y = F.max(x)
    expected = np.max(x.data)
    assert array_allclose(y.data, expected)


def test_forward2():
    shape = (10, 20, 30)
    axis = 1
    x = Variable(np.random.rand(*shape))
    y = F.max(x, axis=axis)
    expected = np.max(x.data, axis=axis)
    assert array_allclose(y.data, expected)


def test_forward3():
    shape = (10, 20, 30)
    axis = (0, 1)
    x = Variable(np.random.rand(*shape))
    y = F.max(x, axis=axis)
    expected = np.max(x.data, axis=axis)
    assert array_allclose(y.data, expected)


def test_forward4():
    shape = (10, 20, 30)
    axis = (0, 1)
    x = Variable(np.random.rand(*shape))
    y = F.max(x, axis=axis, keepdims=True)
    expected = np.max(x.data, axis=axis, keepdims=True)
    assert array_allclose(y.data, expected)


def test_backward1():
    x_data = np.random.rand(10)
    f = lambda x: F.max(x)
    assert gradient_check(f, x_data)


def test_backward2():
    x_data = np.random.rand(10, 10) * 100
    f = lambda x: F.max(x, axis=1)
    assert gradient_check(f, x_data)


def test_backward3():
    x_data = np.random.rand(10, 20, 30) * 100
    f = lambda x: F.max(x, axis=(1, 2))
    assert gradient_check(f, x_data)


def test_backward4():
    x_data = np.random.rand(10, 20, 20) * 100
    f = lambda x: F.sum(x, axis=None)
    assert gradient_check(f, x_data)


def test_backward5():
    x_data = np.random.rand(10, 20, 20) * 100
    f = lambda x: F.sum(x, axis=None, keepdims=True)
    assert gradient_check(f, x_data)


if __name__ == "__main__":
    pytest.main()
