import chainer
import numpy as np
import pytest

import dezero
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


def test_forward1():
    N, C = 8, 3
    x, gamma, beta, mean, var = get_params(N, C)
    cy = chainer.links.BatchNormalization(3)(x)
    y = dezero.layers.BatchNorm()(x)
    assert array_allclose(y.data, cy.data)


def test_forward2():
    N, C = 8, 3
    cl = chainer.links.BatchNormalization(C)
    l = dezero.layers.BatchNorm()
    for i in range(10):
        x, gamma, beta, mean, var = get_params(N, C)
        cy = cl(x)
        y = l(x)
    assert array_allclose(cl.avg_mean, l.avg_mean.data)
    assert array_allclose(cl.avg_var, l.avg_var.data)

    with dezero.test_mode():
        y = l(x)
    with chainer.using_config("train", False):
        cy = cl(x)
    assert array_allclose(cy.data, y.data)


if __name__ == "__main__":
    pytest.main()
