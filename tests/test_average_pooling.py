import chainer.functions as CF
import numpy as np
import pytest

import dezero.functions as F
from dezero.utils import array_allclose, gradient_check


def test_forward1():
    n, c, h, w = 1, 5, 16, 16
    ksize, stride, pad = 2, 2, 0
    x = np.random.randn(n, c, h, w).astype("f")

    y = F.average_pooling(x, ksize, stride, pad)
    expected = CF.average_pooling_2d(x, ksize, stride, pad)
    assert array_allclose(expected.data, y.data)


def test_forward2():
    n, c, h, w = 1, 5, 15, 15
    ksize, stride, pad = 2, 2, 0
    x = np.random.randn(n, c, h, w).astype("f")

    y = F.average_pooling(x, ksize, stride, pad)
    expected = CF.average_pooling_2d(x, ksize, stride, pad)
    assert array_allclose(expected.data, y.data)


def test_backward1():
    n, c, h, w = 1, 5, 16, 16
    ksize, stride, pad = 2, 2, 0
    x = np.random.randn(n, c, h, w).astype("f") * 1000
    f = lambda x: F.average_pooling(x, ksize, stride, pad)
    assert gradient_check(f, x)


if __name__ == "__main__":
    pytest.main()
