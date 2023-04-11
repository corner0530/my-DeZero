import chainer.functions as CF
import cupy as np  # !! CUPY !!
import pytest

import dezero.functions as F
from dezero.utils import array_allclose, gradient_check


def test_forward1():
    n, c, h, w = 1, 5, 15, 15
    o, k, s, p = 8, (3, 3), (1, 1), (1, 1)
    x = np.random.randn(n, c, h, w).astype("f")
    W = np.random.randn(o, c, k[0], k[1]).astype("f")
    b = None
    y = F.conv2d_simple(x, W, b, s, p)
    expected = CF.convolution_2d(x, W, b, s, p)
    assert array_allclose(expected.data, y.data)


def test_forward2():
    n, c, h, w = 1, 5, 15, 15
    o, k, s, p = 8, (3, 3), (3, 1), (2, 1)
    x = np.random.randn(n, c, h, w).astype("f")
    W = np.random.randn(o, c, k[0], k[1]).astype("f")
    b = None
    y = F.conv2d_simple(x, W, b, s, p)
    expected = CF.convolution_2d(x, W, b, s, p)
    assert array_allclose(expected.data, y.data)


def test_forward3():
    n, c, h, w = 1, 5, 20, 15
    o, k, s, p = 3, (5, 3), 1, 3
    x = np.random.randn(n, c, h, w).astype("f")
    W = np.random.randn(o, c, k[0], k[1]).astype("f")
    b = None
    y = F.conv2d_simple(x, W, b, s, p)
    expected = CF.convolution_2d(x, W, b, s, p)
    assert array_allclose(expected.data, y.data)


def test_forward4():
    n, c, h, w = 1, 5, 20, 15
    o, k, s, p = 3, (5, 3), 1, 3
    x = np.random.randn(n, c, h, w).astype("f")
    W = np.random.randn(o, c, k[0], k[1]).astype("f")
    b = np.random.randn(o).astype("f")
    y = F.conv2d_simple(x, W, b, s, p)
    expected = CF.convolution_2d(x, W, b, s, p)
    assert array_allclose(expected.data, y.data)


def test_backward1():
    n, c, h, w = 1, 5, 20, 15
    o, k, s, p = 3, (5, 3), 1, 3
    x = np.random.randn(n, c, h, w)
    W = np.random.randn(o, c, k[0], k[1])
    b = np.random.randn(o)
    f = lambda x: F.conv2d_simple(x, W, b, s, p)
    assert gradient_check(f, x)


def test_backward2():
    n, c, h, w = 1, 5, 20, 15
    o, k, s, p = 3, (5, 3), 1, 3
    x = np.random.randn(n, c, h, w)
    W = np.random.randn(o, c, k[0], k[1])
    b = np.random.randn(o)
    f = lambda b: F.conv2d_simple(x, W, b, s, p)
    assert gradient_check(f, b)


def test_backward3():
    n, c, h, w = 1, 5, 20, 15
    o, k, s, p = 3, (5, 3), 1, 3
    x = np.random.randn(n, c, h, w)
    W = np.random.randn(o, c, k[0], k[1])
    b = np.random.randn(o)
    f = lambda W: F.conv2d_simple(x, W, b, s, p)
    assert gradient_check(f, W)


if __name__ == "__main__":
    pytest.main()
