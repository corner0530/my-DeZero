import numpy as np
import pytest

import dezero.functions as F
from dezero import Variable
from dezero.utils import gradient_check


def test_forward1():
    x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
    y = F.transpose(x)
    assert y.shape == (3, 2)


def test_backward1():
    x = np.array([[1, 2, 3], [4, 5, 6]])
    assert gradient_check(F.transpose, x)


def test_backward2():
    x = np.array([1, 2, 3])
    assert gradient_check(F.transpose, x)


def test_backward3():
    x = np.random.randn(10, 5)
    assert gradient_check(F.transpose, x)


def test_backward4():
    x = np.array([1, 2])
    assert gradient_check(F.transpose, x)


if __name__ == "__main__":
    pytest.main()
