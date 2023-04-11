import chainer.functions as CF
import numpy as np
import pytest

import dezero
import dezero.functions as F
from dezero.utils import array_allclose


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
    with dezero.test_mode():
        y = F.batch_norm(x, gamma, beta, mean, var)
    assert y.data.dtype == np.float32


def test_forward1():
    N, C = 8, 1
    x, gamma, beta, mean, var = get_params(N, C)
    cy = CF.fixed_batch_normalization(x, gamma, beta, mean, var)
    with dezero.test_mode():
        y = F.batch_norm(x, gamma, beta, mean, var)
    assert array_allclose(y.data, cy.data)


def test_forward2():
    N, C = 1, 10
    x, gamma, beta, mean, var = get_params(N, C)
    cy = CF.fixed_batch_normalization(x, gamma, beta, mean, var)
    with dezero.test_mode():
        y = F.batch_norm(x, gamma, beta, mean, var)
    assert array_allclose(y.data, cy.data)


def test_forward3():
    N, C = 20, 10
    x, gamma, beta, mean, var = get_params(N, C)
    cy = CF.fixed_batch_normalization(x, gamma, beta, mean, var)
    with dezero.test_mode():
        y = F.batch_norm(x, gamma, beta, mean, var)
    assert array_allclose(y.data, cy.data)


def test_forward4():
    N, C, H, W = 20, 10, 5, 5
    x, gamma, beta, mean, var = get_params(N, C, H, W)
    cy = CF.fixed_batch_normalization(x, gamma, beta, mean, var)
    with dezero.test_mode():
        y = F.batch_norm(x, gamma, beta, mean, var)
    assert array_allclose(y.data, cy.data)


if __name__ == "__main__":
    pytest.main()
