# 27. テイラー展開の微分
import math

import numpy as np

from dezero import Function, Variable
from dezero.utils import plot_dot_graph


class Sin(Function):
    """sin関数のクラス"""

    def forward(self, x: np.ndarray) -> np.ndarray:
        """順伝播

        Args:
            x: 入力

        Returns:
            y: 出力
        """
        y = np.sin(x)
        return y

    def backward(self, gy: np.ndarray) -> np.ndarray:
        """逆伝播

        Args:
            gy: 出力側から伝わる微分

        Returns:
            gx: 入力側に伝わる微分
        """
        x = self.inputs[0].data
        gx = gy * np.cos(x)
        return gx


def sin(x: Variable) -> Variable:
    """sin関数

    Args:
        x: 入力

    Returns:
        y: 出力
    """
    return Sin()(x)


def my_sin(x: Variable, threshold: float = 0.0001) -> Variable:
    """sin関数のテイラー展開

    Args:
        x: 入力
        threshold: 許容誤差

    Returns:
        y: 出力
    """
    y = 0
    for i in range(100000):
        c = (-1) ** i / math.factorial(2 * i + 1)
        t = c * x ** (2 * i + 1)
        y = y + t
        if abs(t.data) < threshold:
            break
    return y


if __name__ == "__main__":
    x = Variable(np.array(np.pi / 4))
    y = sin(x)
    y.backward()
    print("--- original sin ---")
    print(y.data)
    print(x.grad)

    x = Variable(np.array(np.pi / 4))
    y = my_sin(x, threshold=1e-150)
    y.backward()
    print("--- approximate sin ---")
    print(y.data)
    print(x.grad)

    x.name = "x"
    y.name = "y"
    plot_dot_graph(y, verbose=False, to_file="my_sin.png")
