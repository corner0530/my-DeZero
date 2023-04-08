import numpy as np

from dezero import utils
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


class Sum(Function):
    """和を計算する関数を表すクラス

    Attributes:
        axis (int): 和をとる軸
        keepdims (bool): 出力の形状を入力の形状に合わせるかどうか
        x_shape (tuple): 入力の形状
    """

    def __init__(self, axis: int | tuple[int] = None, keepdims: bool = False) -> None:
        """コンストラクタ

        Args:
            axis: 和をとる軸
            keepdims: 出力の形状を入力の形状に合わせるかどうか
        """
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x: np.ndarray) -> np.ndarray:
        """順伝播

        Args:
            x: 入力

        Returns:
            y: 出力
        """
        self.x_shape = x.shape
        y = x.sum(axis=self.axis, keepdims=self.keepdims)
        return y

    def backward(self, gy: Variable) -> Variable:
        """逆伝播

        Args:
            gy: 出力側から伝わる微分

        Returns:
            gx: 入力側に伝わる微分
        """
        gy = utils.reshape_sum_backward(gy, self.x_shape, self.axis, self.keepdims)
        gx = broadcast_to(gy, self.x_shape)
        return gx


def sum(x: Variable, axis: int | tuple[int] = None, keepdims: bool = False) -> Variable:
    """和を計算する関数

    Args:
        x: 入力
        axis: 和をとる軸
        keepdims: 出力の形状を入力の形状に合わせるかどうか

    Returns:
        y: 出力
    """
    return Sum(axis, keepdims)(x)


class SumTo(Function):
    """要素の和を求めて形状を変える関数を表すクラス

    Attributes:
        shape (tuple): 出力の形状
        x_shape (tuple): 入力の形状
    """

    def __init__(self, shape: tuple) -> None:
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
        y = utils.sum_to(x, self.shape)
        return y

    def backward(self, gy: Variable) -> Variable:
        """逆伝播

        Args:
            gy: 出力側から伝わる微分

        Returns:
            gx: 入力側に伝わる微分
        """
        gx = broadcast_to(gy, self.x_shape)
        return gx


def sum_to(x: Variable, shape: tuple) -> Variable:
    """要素の和を求めて形状を変える関数

    Args:
        x: 入力
        shape: 出力の形状

    Returns:
        y: 出力
    """
    if x.shape == shape:
        return as_variable(x)
    return SumTo(shape)(x)


class BroadcastTo(Function):
    """形状を変える関数を表すクラス

    Attributes:
        shape (tuple): 出力の形状
        x_shape (tuple): 入力の形状
    """

    def __init__(self, shape: tuple) -> None:
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
        y = np.broadcast_to(x, self.shape)
        return y

    def backward(self, gy: Variable) -> Variable:
        """逆伝播

        Args:
            gy: 出力側から伝わる微分

        Returns:
            gx: 入力側に伝わる微分
        """
        gx = sum_to(gy, self.x_shape)
        return gx


def broadcast_to(x: Variable, shape: tuple) -> Variable:
    """形状を変える関数

    Args:
        x: 入力
        shape: 出力の形状

    Returns:
        y: 出力
    """
    if x.shape == shape:
        return as_variable(x)
    return BroadcastTo(shape)(x)


class MatMul(Function):
    """行列積を計算する関数を表すクラス"""

    def forward(self, x: np.ndarray, w: np.ndarray) -> np.ndarray:
        """順伝播

        Args:
            x: 入力
            w: 入力

        Returns:
            y: 出力
        """
        y = x.dot(w)
        return y

    def backward(self, gy: Variable) -> tuple[Variable, Variable]:
        """逆伝播

        Args:
            gy: 出力側から伝わる微分

        Returns:
            gx: 入力x側に伝わる微分
            gw: 入力w側に伝わる微分
        """
        x, w = self.inputs
        gx = matmul(gy, w.T)
        gw = matmul(x.T, gy)
        return gx, gw


def matmul(x: Variable, w: Variable) -> Variable:
    """行列積を計算する関数

    Args:
        x: 入力
        w: 入力

    Returns:
        y: 出力
    """
    return MatMul()(x, w)


def mean_squared_error_simple(x0: Variable, x1: Variable) -> Variable:
    """平均二乗誤差を計算する関数

    Args:
        x0: 入力
        x1: 入力

    Returns:
        y: 出力
    """
    x0, x1 = as_variable(x0), as_variable(x1)
    diff = x0 - x1
    y = sum(diff**2) / len(diff)
    return y


class MeanSquaredError(Function):
    """平均二乗誤差を計算する関数を表すクラス"""

    def forward(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        """順伝播

        Args:
            x0: 入力
            x1: 入力

        Returns:
            y: 出力
        """
        diff = x0 - x1
        y = (diff**2).sum() / len(diff)
        return y

    def backward(self, gy: Variable) -> tuple[Variable, Variable]:
        """逆伝播

        Args:
            gy: 出力側から伝わる微分

        Returns:
            gx0: 入力x0側に伝わる微分
            gx1: 入力x1側に伝わる微分
        """
        x0, x1 = self.inputs
        diff = x0 - x1
        gx0 = gy * diff * (2.0 / len(diff))
        gx1 = -gx0
        return gx0, gx1


def mean_squared_error(x0: Variable, x1: Variable) -> Variable:
    """平均二乗誤差を計算する関数

    Args:
        x0: 入力
        x1: 入力

    Returns:
        y: 出力
    """
    return MeanSquaredError()(x0, x1)
