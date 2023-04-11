import numpy as np
import pytest

import dezero
import dezero.functions as F
from dezero import Variable
from dezero.utils import array_equal, gradient_check


def test_forward1():
    x = np.random.randn(100, 100)
    y = F.dropout(Variable(x), dropout_ratio=0.0)
    res = array_equal(y.data, x)
    assert res


def test_forward2():
    x = np.random.randn(100, 100)
    with dezero.test_mode():
        y = F.dropout(x)
    res = array_equal(y.data, x)
    assert res


def test_backward1():
    x_data = np.random.randn(10, 10)

    def f(x):
        np.random.seed(0)
        return F.dropout(x, 0.5)

    assert gradient_check(f, x_data)


def test_backward2():
    x_data = np.random.randn(10, 20)

    def f(x):
        np.random.seed(0)
        return F.dropout(x, 0.99)

    assert gradient_check(f, x_data)


def test_backward3():
    x_data = np.random.randn(10, 10)

    def f(x):
        np.random.seed(0)
        return F.dropout(x, 0.0)

    assert gradient_check(f, x_data)


if __name__ == "__main__":
    pytest.main()
