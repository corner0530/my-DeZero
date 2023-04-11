import chainer
import numpy as np
import pytest

import dezero.functions as F
from dezero import Variable
from dezero.utils import array_allclose, gradient_check


def test_forward1():
    x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
    w = Variable(x.data.T)
    b = None
    y = F.linear(x, w, b)

    res = y.data
    expected = np.array([[14, 32], [32, 77]])
    assert array_allclose(res, expected)


def test_forward2():
    x = np.array([[1, 2, 3], [4, 5, 6]]).astype("f")
    W = x.T
    b = None
    y = F.linear(x, W, b)

    cy = chainer.functions.linear(x, W.T)
    assert array_allclose(y.data, cy.data)


def test_forward3():
    layer = chainer.links.Linear(3, 2)
    x = np.array([[1, 2, 3], [4, 5, 6]]).astype("f")
    W = layer.W.data.T
    b = layer.b.data
    y = F.linear(x, W, b)

    cy = layer(x)
    assert array_allclose(y.data, cy.data)


def test_backward1():
    x = np.random.randn(3, 2)
    W = np.random.randn(2, 3)
    b = np.random.randn(3)
    f = lambda x: F.linear(x, W, b)
    assert gradient_check(f, x)


def test_backward2():
    x = np.random.randn(100, 200)
    W = np.random.randn(200, 300)
    b = None
    f = lambda x: F.linear(x, W, b)
    assert gradient_check(f, x)


if __name__ == "__main__":
    pytest.main()
