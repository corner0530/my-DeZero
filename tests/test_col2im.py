import numpy as np
import pytest

import dezero.functions as F
from dezero.utils import gradient_check


def test_backward1():
    n, c, h, w = 1, 1, 3, 3
    x = np.random.rand(1, 9)
    f = lambda x: F.col2im(x, (n, c, h, w), 3, 3, 0, to_matrix=True)
    assert gradient_check(f, x)


def test_backward2():
    n, c, h, w = 1, 1, 3, 3
    x = np.random.rand(1, 1, 3, 3, 1, 1)
    f = lambda x: F.col2im(x, (n, c, h, w), 3, 3, 0, to_matrix=False)
    assert gradient_check(f, x)


if __name__ == "__main__":
    pytest.main()
