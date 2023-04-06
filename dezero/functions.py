import numpy as np

from dezero.core import Function, Variable


class Sin(Function):
    """sin関数を表すクラス"""

    def forward(self, x: np.ndarray) -> np.ndarray:
        """順伝播

        Args:
            x: 入力

        Returns:
            y: 出力
        """
        y = np.sin(x)
        return y

    def backward(self, gy: Variable) -> Variable:
        """逆伝播

        Args:
            gy: 出力側から伝わる微分

        Returns:
            gx: 入力側に伝わる微分
        """
        x = self.inputs[0]
        gx = gy * cos(x)
        return gx


def sin(x: Variable) -> Variable:
    """sin関数

    Args:
        x: 入力

    Returns:
        y: 出力
    """
    return Sin()(x)


class Cos(Function):
    """cos関数を表すクラス"""

    def forward(self, x: np.ndarray) -> np.ndarray:
        """順伝播

        Args:
            x: 入力

        Returns:
            y: 出力
        """
        y = np.cos(x)
        return y

    def backward(self, gy: Variable) -> Variable:
        """逆伝播

        Args:
            gy: 出力側から伝わる微分

        Returns:
            gx: 入力側に伝わる微分
        """
        x = self.inputs[0]
        gx = gy * -sin(x)
        return gx


def cos(x: Variable) -> Variable:
    """cos関数

    Args:
        x: 入力

    Returns:
        y: 出力
    """
    return Cos()(x)


class Tanh(Function):
    """tanh関数を表すクラス"""

    def forward(self, x: np.ndarray) -> np.ndarray:
        """順伝播

        Args:
            x: 入力

        Returns:
            y: 出力
        """
        y = np.tanh(x)
        return y

    def backward(self, gy: Variable) -> Variable:
        """逆伝播

        Args:
            gy: 出力側から伝わる微分

        Returns:
            gx: 入力側に伝わる微分
        """
        y = self.outputs[0]()
        gx = gy * (1 - y * y)
        return gx


def tanh(x: Variable) -> Variable:
    """tanh関数

    Args:
        x: 入力

    Returns:
        y: 出力
    """
    return Tanh()(x)
