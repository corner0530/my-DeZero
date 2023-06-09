# 24. 複雑な関数の微分
import numpy as np

from dezero import Variable


def sphere(x: Variable, y: Variable) -> Variable:
    z = x**2 + y**2
    return z


def matyas(x: Variable, y: Variable) -> Variable:
    z = 0.26 * (x**2 + y**2) - 0.48 * x * y
    return z


def goldstein(x: Variable, y: Variable) -> Variable:
    z = (
        1
        + (x + y + 1) ** 2
        * (19 - 14 * x + 3 * x**2 - 14 * y + 6 * x * y + 3 * y**2)
    ) * (
        30
        + (2 * x - 3 * y) ** 2
        * (18 - 32 * x + 12 * x**2 + 48 * y - 36 * x * y + 27 * y**2)
    )
    return z


if __name__ == "__main__":
    x = Variable(np.array(1.0))
    y = Variable(np.array(1.0))
    # z = sphere(x, y)
    # z = matyas(x, y)
    z = goldstein(x, y)
    z.backward()
    print(x.grad, y.grad)
