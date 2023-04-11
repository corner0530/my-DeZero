import chainer.functions as CF
import cupy as np  # !! CUPY !!
import pytest

import dezero.functions as F
from dezero.utils import array_allclose, gradient_check


def test_forward1():
    x = np.array([[-1, 0, 1, 2], [2, 0, 1, -1]], np.float32)
    y = F.log_softmax(x)
    y2 = CF.log_softmax(x)
    res = array_allclose(y.data, y2.data)
    assert res


def test_backward1():
    x = np.array([[-1, 0, 1, 2], [2, 0, 1, -1]])
    f = lambda x: F.log_softmax(x)
    assert gradient_check(f, x)


def test_backward2():
    x = np.random.randn(10, 10)
    f = lambda x: F.log_softmax(x)
    assert gradient_check(f, x)


if __name__ == "__main__":
    pytest.main()
