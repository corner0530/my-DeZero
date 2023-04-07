import numpy as np

from dezero.core import Function, Variable, as_variable


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


class Reshape(Function):
    """テンソルを整形する関数を表すクラス

    Attributes:
        shape (tuple): 出力の形状
        x_shape (tuple): 入力の形状
    """

    def __init__(self, shape: tuple[int]) -> None:
        """コンストラクタ

        Args:
            shape: 出力の形状
        """
        self.shape = shape

    def forward(self, x: np.ndarray) -> np.ndarray:
        """順伝播

        Args:
            x: 入力

        Returns:
            y: 出力
        """
        self.x_shape = x.shape
        y = x.reshape(self.shape)
        return y

    def backward(self, gy: Variable) -> Variable:
        """逆伝播

        Args:
            gy: 出力側から伝わる微分

        Returns:
            gx: 入力側に伝わる微分
        """
        gx = reshape(gy, self.x_shape)
        return gx


def reshape(x: Variable, shape: tuple[int]) -> Variable:
    """テンソルを整形する関数

    Args:
        x: 入力
        shape: 出力の形状

    Returns:
        y: 出力
    """
    if x.shape == shape:
        return as_variable(x)
    return Reshape(shape)(x)


class Transpose(Function):
    """転置を表すクラス

    Attributes:
        axes (tuple): 軸の順番
    """

    def __init__(self, axes: tuple[int] = None) -> None:
        """コンストラクタ

        Args:
            axes: 軸の順番
        """
        self.axes = axes

    def forward(self, x: np.ndarray) -> np.ndarray:
        """順伝播

        Args:
            x: 入力

        Returns:
            y: 出力
        """
        y = x.transpose(self.axes)
        return y

    def backward(self, gy: Variable) -> Variable:
        """逆伝播

        Args:
            gy: 出力側から伝わる微分

        Returns:
            gx: 入力側に伝わる微分
        """
        if self.axes is None:
            return transpose(gy)

        axes_len = len(self.axes)
        inv_axes = tuple(np.argsort([ax % axes_len for ax in self.axes]))
        return transpose(gy, inv_axes)


def transpose(x: Variable, axes: tuple[int] = None) -> Variable:
    """転置を表す関数

    Args:
        x: 入力
        axes: 軸の順番

    Returns:
        y: 出力
    """
    return Transpose(axes)(x)
