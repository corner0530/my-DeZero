import numpy as np
import pytest

import dezero.functions as F
from dezero import Variable
from dezero.utils import array_allclose, gradient_check


def test_forward1():
    x = Variable(np.random.rand(10))
    y = F.sum_to(x, (1,))
    expected = np.sum(x.data)
    assert array_allclose(y.data, expected)


def test_forward2():
    x = Variable(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    y = F.sum_to(x, (1, 3))
    expected = np.sum(x.data, axis=0, keepdims=True)
    assert array_allclose(y.data, expected)


def test_forward3():
    x = Variable(np.random.rand(10))
    y = F.sum_to(x, (10,))
    expected = x.data  # 同じ形状なので何もしない
    assert array_allclose(y.data, expected)


def test_backward1():
    x_data = np.random.rand(10)
    f = lambda x: F.sum_to(x, (1,))
    assert gradient_check(f, x_data)


def test_backward2():
    x_data = np.random.rand(10, 10) * 10
    f = lambda x: F.sum_to(x, (10,))
    assert gradient_check(f, x_data)


def test_backward3():
    x_data = np.random.rand(10, 20, 20) * 100
    f = lambda x: F.sum_to(x, (10,))
    assert gradient_check(f, x_data)


def test_backward4():
    x_data = np.random.rand(10)
    f = lambda x: F.sum_to(x, (10,)) + 1
    assert gradient_check(f, x_data)


if __name__ == "__main__":
    pytest.main()
