import chainer.functions as CF
import cupy as np  # !! CUPY !!
import pytest

import dezero.functions as F
from dezero.utils import array_allclose, gradient_check


def test_forward1():
    n, c_i, c_o = 10, 1, 3
    h_i, w_i = 5, 10
    h_k, w_k = 10, 10
    h_p, w_p = 5, 5
    s_y, s_x = 5, 5
    x = np.random.uniform(0, 1, (n, c_i, h_i, w_i)).astype(np.float32)
    W = np.random.uniform(0, 1, (c_i, c_o, h_k, w_k)).astype(np.float32)
    b = np.random.uniform(0, 1, c_o).astype(np.float32)

    expected = CF.deconvolution_2d(x, W, b, stride=(s_y, s_x), pad=(h_p, w_p))
    y = F.deconv2d(x, W, b, stride=(s_y, s_x), pad=(h_p, w_p))
    assert array_allclose(expected.data, y.data)


def test_forward2():
    n, c_i, c_o = 10, 1, 3
    h_i, w_i = 5, 10
    h_k, w_k = 10, 10
    h_p, w_p = 5, 5
    s_y, s_x = 5, 5
    x = np.random.uniform(0, 1, (n, c_i, h_i, w_i)).astype(np.float32)
    W = np.random.uniform(0, 1, (c_i, c_o, h_k, w_k)).astype(np.float32)
    b = None
    expected = CF.deconvolution_2d(x, W, b, stride=(s_y, s_x), pad=(h_p, w_p))
    y = F.deconv2d(x, W, b, stride=(s_y, s_x), pad=(h_p, w_p))
    assert array_allclose(expected.data, y.data)


def test_backward1():
    n, c_i, c_o = 10, 1, 3
    h_i, w_i = 5, 10
    h_k, w_k = 10, 10
    h_p, w_p = 5, 5
    s_y, s_x = 5, 5
    x = np.random.uniform(0, 1, (n, c_i, h_i, w_i))
    W = np.random.uniform(0, 1, (c_i, c_o, h_k, w_k))
    b = None  # np.random.uniform(0, 1, c_o).astype(np.float32)
    f = lambda x: F.deconv2d(x, W, b, stride=(s_y, s_x), pad=(h_p, w_p))
    assert gradient_check(f, x)


def test_backward2():
    n, c_i, c_o = 10, 1, 3
    h_i, w_i = 5, 10
    h_k, w_k = 10, 10
    h_p, w_p = 5, 5
    s_y, s_x = 5, 5
    x = np.random.uniform(0, 1, (n, c_i, h_i, w_i))
    W = np.random.uniform(0, 1, (c_i, c_o, h_k, w_k))
    b = np.random.uniform(0, 1, c_o)
    f = lambda W: F.deconv2d(x, W, b, stride=(s_y, s_x), pad=(h_p, w_p))
    assert gradient_check(f, W)


def test_backward3():
    n, c_i, c_o = 10, 1, 3
    h_i, w_i = 5, 10
    h_k, w_k = 10, 10
    h_p, w_p = 5, 5
    s_y, s_x = 5, 5
    x = np.random.uniform(0, 1, (n, c_i, h_i, w_i))
    W = np.random.uniform(0, 1, (c_i, c_o, h_k, w_k))
    b = np.random.uniform(0, 1, c_o)
    f = lambda b: F.deconv2d(x, W, b, stride=(s_y, s_x), pad=(h_p, w_p))
    assert gradient_check(f, b)


if __name__ == "__main__":
    pytest.main()
