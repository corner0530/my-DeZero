import cupy as np  # !! CUPY !!
import pytest

import dezero.functions as F
from dezero import Variable


def test_shape_check():
    x = Variable(np.random.randn(1, 10))
    b = Variable(np.random.randn(10))
    y = x + b
    loss = F.sum(y)
    loss.backward()
    assert b.grad.shape == b.shape


if __name__ == "__main__":
    pytest.main()
