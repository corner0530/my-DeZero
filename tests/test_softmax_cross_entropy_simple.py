import chainer.functions as CF
import numpy as np
import pytest

import dezero.functions as F
from dezero import Variable
from dezero.utils import array_allclose, gradient_check


def test_forward1():
    x = np.array([[-1, 0, 1, 2], [2, 0, 1, -1]], np.float32)
    t = np.array([3, 0]).astype(np.int32)
    y = F.softmax_cross_entropy_simple(x, t)
    y2 = CF.softmax_cross_entropy(x, t)
    res = array_allclose(y.data, y2.data)
    assert res


def test_backward1():
    x = np.array([[-1, 0, 1, 2], [2, 0, 1, -1]], np.float32)
    t = np.array([3, 0]).astype(np.int32)
    f = lambda x: F.softmax_cross_entropy_simple(x, Variable(t))
    assert gradient_check(f, x)


def test_backward2():
    N, CLS_NUM = 10, 10
    x = np.random.randn(N, CLS_NUM)
    t = np.random.randint(0, CLS_NUM, (N,))
    f = lambda x: F.softmax_cross_entropy_simple(x, t)
    assert gradient_check(f, x)


def test_backward3():
    N, CLS_NUM = 100, 10
    x = np.random.randn(N, CLS_NUM)
    t = np.random.randint(0, CLS_NUM, (N,))
    f = lambda x: F.softmax_cross_entropy_simple(x, t)
    assert gradient_check(f, x)


if __name__ == "__main__":
    pytest.main()
