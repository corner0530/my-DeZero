import cupy as np  # !! CUPY !!
import pytest

import dezero.functions as F
from dezero import Variable
from dezero.utils import array_allclose, gradient_check


def test_datatype():
    x = Variable(np.random.rand(10))
    y = F.sum(x)
    # np.float64ではなく0次元のnp.ndarrayを返す
    assert not np.isscalar(y)


def test_forward1():
    x = Variable(np.array(2.0))
    y = F.sum(x)
    expected = np.sum(x.data)
    assert array_allclose(y.data, expected)


def test_forward2():
    x = Variable(np.random.rand(10, 20, 30))
    y = F.sum(x, axis=1)
    expected = np.sum(x.data, axis=1)
    assert array_allclose(y.data, expected)


def test_forward3():
    x = Variable(np.random.rand(10, 20, 30))
    y = F.sum(x, axis=1, keepdims=True)
    expected = np.sum(x.data, axis=1, keepdims=True)
    assert array_allclose(y.data, expected)


def test_backward1():
    x_data = np.random.rand(10)
    f = lambda x: F.sum(x)
    assert gradient_check(f, x_data)


def test_backward2():
    x_data = np.random.rand(10, 10)
    f = lambda x: F.sum(x, axis=1)
    assert gradient_check(f, x_data)


def test_backward3():
    x_data = np.random.rand(10, 20, 20)
    f = lambda x: F.sum(x, axis=2)
    assert gradient_check(f, x_data)


def test_backward4():
    x_data = np.random.rand(10, 20, 20)
    f = lambda x: F.sum(x, axis=None)
    assert gradient_check(f, x_data)


if __name__ == "__main__":
    pytest.main()
