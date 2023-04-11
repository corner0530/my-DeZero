import chainer
import chainer.functions as CF
import numpy as np
import pytest

import dezero
import dezero.functions as F
from dezero.utils import array_allclose, gradient_check


def get_params(N, C, H=None, W=None, dtype="f"):
    if H is not None:
        x = np.random.randn(N, C, H, W).astype(dtype)
    else:
        x = np.random.randn(N, C).astype(dtype)
    gamma = np.random.randn(C).astype(dtype)
    beta = np.random.randn(C).astype(dtype)
    mean = np.random.randn(C).astype(dtype)
    var = np.abs(np.random.randn(C).astype(dtype))
    return x, gamma, beta, mean, var


def test_type1():
    N, C = 8, 3
    x, gamma, beta, mean, var = get_params(N, C)
    y = F.batch_norm(x, gamma, beta, mean, var)
    assert y.data.dtype == np.float32


def test_forward1():
    N, C = 8, 1
    x, gamma, beta, mean, var = get_params(N, C)
    cy = CF.batch_normalization(x, gamma, beta, running_mean=mean, running_var=var)
    y = F.batch_norm(x, gamma, beta, mean, var)
    assert array_allclose(y.data, cy.data)


def test_forward2():
    N, C = 1, 10
    x, gamma, beta, mean, var = get_params(N, C)
    cy = CF.batch_normalization(x, gamma, beta)
    y = F.batch_norm(x, gamma, beta, mean, var)
    assert array_allclose(y.data, cy.data)


def test_forward3():
    N, C = 20, 10
    x, gamma, beta, mean, var = get_params(N, C)
    cy = CF.batch_normalization(x, gamma, beta)
    y = F.batch_norm(x, gamma, beta, mean, var)
    assert array_allclose(y.data, cy.data)


def test_forward4():
    N, C, H, W = 20, 10, 5, 5
    x, gamma, beta, mean, var = get_params(N, C, H, W)
    cy = CF.batch_normalization(x, gamma, beta)
    y = F.batch_norm(x, gamma, beta, mean, var)
    assert array_allclose(y.data, cy.data)


def test_forward5():
    N, C = 20, 10
    cl = chainer.links.BatchNormalization(C)
    l = dezero.layers.BatchNorm()

    for i in range(10):
        x = np.random.randn(N, C).astype("f")
        cy = cl(x)
        y = l(x)
        assert array_allclose(y.data, cy.data)
    assert array_allclose(cl.avg_mean.data, l.avg_mean.data)
    assert array_allclose(cl.avg_var.data, l.avg_var.data)


def test_forward6():
    N, C, H, W = 20, 10, 5, 5
    cl = chainer.links.BatchNormalization(C)
    l = dezero.layers.BatchNorm()

    for i in range(10):
        x = np.random.randn(N, C, H, W).astype("f")
        cy = cl(x)
        y = l(x)
        assert array_allclose(y.data, cy.data)
    assert array_allclose(cl.avg_mean.data, l.avg_mean.data)
    assert array_allclose(cl.avg_var.data, l.avg_var.data)


def test_backward1():
    N, C = 8, 3
    x, gamma, beta, mean, var = get_params(N, C, dtype=np.float64)
    f = lambda x: F.batch_norm(x, gamma, beta, mean, var)
    assert gradient_check(f, x)


def test_backward2():
    N, C = 8, 3
    x, gamma, beta, mean, var = get_params(N, C, dtype=np.float64)
    f = lambda gamma: F.batch_norm(x, gamma, beta, mean, var)
    assert gradient_check(f, gamma)


def test_backward3():
    N, C = 8, 3
    x, gamma, beta, mean, var = get_params(N, C, dtype=np.float64)
    f = lambda beta: F.batch_norm(x, gamma, beta, mean, var)
    assert gradient_check(f, beta)


def test_backward4():
    params = 10, 20, 5, 5
    x, gamma, beta, mean, var = get_params(*params, dtype=np.float64)
    f = lambda x: F.batch_norm(x, gamma, beta, mean, var)
    assert gradient_check(f, x)


def test_backward5():
    params = 10, 20, 5, 5
    x, gamma, beta, mean, var = get_params(*params, dtype=np.float64)
    f = lambda gamma: F.batch_norm(x, gamma, beta, mean, var)
    assert gradient_check(f, gamma)


def test_backward6():
    params = 10, 20, 5, 5
    x, gamma, beta, mean, var = get_params(*params, dtype=np.float64)
    f = lambda beta: F.batch_norm(x, gamma, beta, mean, var)
    assert gradient_check(f, beta)


if __name__ == "__main__":
    pytest.main()
