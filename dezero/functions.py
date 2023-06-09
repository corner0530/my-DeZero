import numpy as np

import dezero
from dezero import cuda, utils
from dezero.core import Function, Variable, as_array, as_variable


# =============================================================================
# Basic functions: sin / cos / tanh / exp / log
# =============================================================================
class Sin(Function):
    """sin関数を表すクラス"""

    def forward(self, x: np.ndarray) -> np.ndarray:
        """順伝播

        Args:
            x: 入力

        Returns:
            y: 出力
        """
        xp = cuda.get_array_module(x)
        y = xp.sin(x)
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
        xp = cuda.get_array_module(x)
        y = xp.cos(x)
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
        xp = cuda.get_array_module(x)
        y = xp.tanh(x)
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


class Exp(Function):
    """exp関数を表すクラス"""

    def forward(self, x: np.ndarray) -> np.ndarray:
        """順伝播

        Args:
            x: 入力

        Returns:
            y: 出力
        """
        xp = cuda.get_array_module(x)
        y = xp.exp(x)
        return y

    def backward(self, gy: Variable) -> Variable:
        """逆伝播

        Args:
            gy: 出力側から伝わる微分

        Returns:
            gx: 入力側に伝わる微分
        """
        y = self.outputs[0]()
        gx = gy * y
        return gx


def exp(x: Variable) -> Variable:
    """exp関数

    Args:
        x: 入力

    Returns:
        y: 出力
    """
    return Exp()(x)


class Log(Function):
    """log関数を表すクラス"""

    def forward(self, x: np.ndarray) -> np.ndarray:
        """順伝播

        Args:
            x: 入力

        Returns:
            y: 出力
        """
        xp = cuda.get_array_module(x)
        y = xp.log(x)
        return y

    def backward(self, gy: Variable) -> Variable:
        """逆伝播

        Args:
            gy: 出力側から伝わる微分

        Returns:
            gx: 入力側に伝わる微分
        """
        x = self.inputs[0]
        gx = gy / x
        return gx


def log(x: Variable) -> Variable:
    """log関数

    Args:
        x: 入力

    Returns:
        y: 出力
    """
    return Log()(x)


# =============================================================================
# Tensor operations: reshape / transpose / get_item / expand_dims / flatten
# =============================================================================
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


class GetItem(Function):
    """インデックスを指定して要素を取り出す関数を表すクラス

    Attributes:
        slices (tuple): スライス
    """

    def __init__(self, slices: tuple) -> None:
        """コンストラクタ

        Args:
            slices: スライス
        """
        self.slices = slices

    def forward(self, x: np.ndarray) -> np.ndarray:
        """順伝播

        Args:
            x: 入力

        Returns:
            y: 出力
        """
        y = x[self.slices]
        return y

    def backward(self, gy: Variable) -> Variable:
        """逆伝播

        Args:
            gy: 出力側から伝わる微分

        Returns:
            gx: 入力側に伝わる微分
        """
        x = self.inputs[0]
        f = GetItemGrad(self.slices, x.shape)
        return f(gy)


class GetItemGrad(Function):
    """インデックスを指定して要素を取り出す関数の逆伝播を表すクラス

    Attributes:
        slices (tuple): スライス
        in_shape (tuple): 入力の形状
    """

    def __init__(self, slices: tuple, in_shape: tuple) -> None:
        """コンストラクタ

        Args:
            slices: スライス
            in_shape: 入力の形状
        """
        self.slices = slices
        self.in_shape = in_shape

    def forward(self, gy: np.ndarray) -> np.ndarray:
        """順伝播

        Args:
            gy: 出力側から伝わる微分

        Returns:
            gx: 入力側に伝わる微分
        """
        xp = cuda.get_array_module(gy)
        gx = xp.zeros(self.in_shape)
        if xp is np:
            np.add.at(gx, self.slices, gy)
        else:
            xp.scatter_add(gx, self.slices, gy)
        return gx

    def backward(self, ggx: Variable) -> Variable:
        """逆伝播

        Args:
            ggx: 入力側から伝わる微分

        Returns:
            ggy: 出力側に伝わる微分
        """
        return get_item(ggx, self.slices)


def get_item(x: Variable, slices: tuple) -> Variable:
    """インデックスを指定して要素を取り出す関数

    Args:
        x: 入力
        slices: スライス

    Returns:
        y: 出力
    """
    f = GetItem(slices)
    return f(x)


def expand_dims(x: Variable, axis: int) -> Variable:
    """指定した軸に1を挿入して拡張する関数

    Args:
        x: 入力
        axis: 挿入する軸

    Returns:
        y: 出力
    """
    x = as_variable(x)
    shape = list(x.shape)
    shape.insert(axis, 1)
    return reshape(x, tuple(shape))


def flatten(x: Variable) -> Variable:
    """バッチの各要素の多次元配列を1次元配列に変換する関数

    Args:
        x: 入力

    Returns:
        y: 出力
    """
    return reshape(x, (x.shape[0], -1))


# =============================================================================
# sum / sum_to / broadcast_to / average / matmul / linear
# =============================================================================
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
        xp = cuda.get_array_module(x)
        y = xp.broadcast_to(x, self.shape)
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


class Linear(Function):
    """線形変換を計算する関数を表すクラス"""

    def forward(self, x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
        """順伝播

        Args:
            x: 入力
            w: 重み
            b: バイアス

        Returns:
            y: 出力
        """
        y = x.dot(w)
        if b is not None:
            y += b
        return y

    def backward(self, gy: Variable) -> tuple[Variable, Variable, Variable]:
        """逆伝播

        Args:
            gy: 出力側から伝わる微分

        Returns:
            gx: 入力側に伝わる微分,
            gw: 重み側に伝わる微分,
            gb: バイアス側に伝わる微分
        """
        x, w, b = self.inputs
        gb = None if b.data is None else sum_to(gy, b.shape)
        gx = matmul(gy, w.T)
        gw = matmul(x.T, gy)
        return gx, gw, gb


def linear(x: Variable, w: Variable, b: Variable = None) -> Variable:
    """線形変換を計算する関数

    Args:
        x: 入力
        w: 重み
        b: バイアス

    Returns:
        y: 出力
    """
    return Linear()(x, w, b)


def linear_simple(x: Variable, w: Variable, b: Variable = None) -> Variable:
    """線形変換を計算する関数の簡易版

    Args:
        x: 入力
        w: 入力
        b: 入力

    Returns:
        y: 出力
    """
    t = matmul(x, w)
    if b is None:
        return t

    y = t + b
    t.data = None  # tをメモリから削除
    return y


# =============================================================================
# activation function: sigmoid / relu / softmax / log_softmax / leaky_relu
# =============================================================================
class Sigmoid(Function):
    """シグモイド関数を計算する関数を表すクラス"""

    def forward(self, x: np.ndarray) -> np.ndarray:
        """順伝播

        Args:
            x: 入力

        Returns:
            y: 出力
        """
        xp = cuda.get_array_module(x)
        # y = 1 / (1 + xp.exp(-x))
        y = xp.tanh(x * 0.5) * 0.5 + 0.5  # 実装の改良
        return y

    def backward(self, gy: Variable) -> Variable:
        """逆伝播

        Args:
            gy: 出力側から伝わる微分

        Returns:
            gx: 入力側に伝わる微分
        """
        y = self.outputs[0]()
        gx = gy * y * (1 - y)
        return gx


def sigmoid(x: Variable) -> Variable:
    """シグモイド関数を計算する関数

    Args:
        x: 入力

    Returns:
        y: 出力
    """
    return Sigmoid()(x)


def sigmoid_simple(x: Variable) -> Variable:
    """シグモイド関数を計算する関数の簡易版

    Args:
        x: 入力

    Returns:
        y: 出力
    """
    x = as_variable(x)
    y = 1 / (1 + exp(-x))
    return y


class ReLU(Function):
    """ReLUを表すクラス"""

    def forward(self, x: np.ndarray) -> np.ndarray:
        """順伝播

        Args:
            x: 入力

        Returns:
            y: 出力
        """
        xp = cuda.get_array_module(x)
        y = xp.maximum(x, 0.0)
        return y

    def backward(self, gy: Variable) -> Variable:
        """逆伝播

        Args:
            gy: 出力側から伝わる微分

        Returns:
            gx: 入力側に伝わる微分
        """
        x = self.inputs[0]
        mask = x.data > 0
        gx = gy * mask
        return gx


def relu(x: Variable) -> Variable:
    """ReLU

    Args:
        x: 入力

    Returns:
        y: 出力
    """
    return ReLU()(x)


def softmax_simple(x: Variable, axis: int = 1) -> Variable:
    """ソフトマックス関数の簡易版

    Args:
        x: 入力
        axis: ソフトマックス関数を適用する軸

    Returns:
        y: 出力
    """
    x = as_variable(x)
    y = exp(x)
    sum_y = sum(y, axis=axis, keepdims=True)
    return y / sum_y


class Softmax(Function):
    """ソフトマックス関数を表すクラス

    Attributes:
        axis(int | tuple[int]): ソフトマックス関数を適用する軸
    """

    def __init__(self, axis: int = 1) -> None:
        """コンストラクタ

        Args:
            axis: ソフトマックス関数を適用する軸
        """
        self.axis = axis

    def forward(self, x: np.ndarray) -> np.ndarray:
        """順伝播

        Args:
            x: 入力

        Returns:
            y: 出力
        """
        xp = cuda.get_array_module(x)
        y = x - x.max(axis=self.axis, keepdims=True)
        y = xp.exp(y)
        y /= y.sum(axis=self.axis, keepdims=True)
        return y

    def backward(self, gy: Variable) -> Variable:
        """逆伝播

        Args:
            gy: 出力側から伝わる微分

        Returns:
            gx: 入力側に伝わる微分
        """
        y = self.outputs[0]()
        gx = gy * y
        sumdx = gx.sum(axis=self.axis, keepdims=True)
        gx -= y * sumdx
        return gx


def softmax(x: Variable, axis: int = 1) -> Variable:
    """ソフトマックス関数

    Args:
        x: 入力
        axis: ソフトマックス関数を適用する軸

    Returns:
        y: 出力
    """
    return Softmax(axis)(x)


class LogSoftmax(Function):
    """対数ソフトマックス関数を表すクラス

    Attributes:
        axis(int | tuple[int]): 対数ソフトマックス関数を適用する軸
    """

    def __init__(self, axis: int | tuple[int] = 1) -> None:
        """コンストラクタ

        Args:
            axis: 対数ソフトマックス関数を適用する軸
        """
        self.axis = axis

    def forward(self, x: np.ndarray) -> np.ndarray:
        """順伝播

        Args:
            x: 入力

        Returns:
            y: 出力
        """
        log_z = utils.logsumexp(x, self.axis)
        y = x - log_z
        return y

    def backward(self, gy: Variable) -> Variable:
        """逆伝播

        Args:
            gy: 出力側から伝わる微分

        Returns:
            gx: 入力側に伝わる微分
        """
        y = self.outputs[0]()
        gx = gy - exp(y) * gy.sum(axis=self.axis, keepdims=True)
        return gx


def log_softmax(x: Variable, axis: int | tuple[int] = 1) -> Variable:
    """対数ソフトマックス関数

    Args:
        x: 入力
        axis: 対数ソフトマックス関数を適用する軸

    Returns:
        y: 出力
    """
    return LogSoftmax(axis)(x)


class LeakyReLU(Function):
    """LeakyReLUを表すクラス

    Attributes:
        slope(float): 傾き
    """

    def __init__(self, slope: float) -> None:
        """コンストラクタ

        Args:
            slope: 傾き
        """
        self.slope = slope

    def forward(self, x: np.ndarray) -> np.ndarray:
        """順伝播

        Args:
            x: 入力

        Returns:
            y: 出力
        """
        y = x.copy()
        y[x <= 0] *= self.slope
        return y

    def backward(self, gy: Variable) -> Variable:
        """逆伝播

        Args:
            gy: 出力側から伝わる微分

        Returns:
            gx: 入力側に伝わる微分
        """
        (x,) = self.inputs
        mask = (x.data > 0).astype(gy.dtype)
        mask[mask <= 0] = self.slope
        gx = gy * mask
        return gx


def leaky_relu(x: Variable, slope: float = 0.2) -> Variable:
    """LeakyReLU

    Args:
        x: 入力

    Returns:
        y: 出力
    """
    return LeakyReLU(slope)(x)


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


def softmax_cross_entropy_simple(x: Variable, t: Variable) -> Variable:
    """ソフトマックス関数と交差エントロピー誤差を合わせて計算する関数の簡易版

    Args:
        x:
        t: 教師データ

    Returns:
        y: 出力
    """
    x, t = as_variable(x), as_variable(t)
    N = x.shape[0]

    p = softmax(x)
    p = clip(p, 1e-15, 1.0)  # log(0)を防ぐためにpの値を1e-15以上とする
    log_p = log(p)
    tlog_p = log_p[np.arange(N), t.data]
    y = -1 * sum(tlog_p) / N
    return y


class SoftmaxCrossEntropy(Function):
    """ソフトマックス関数と交差エントロピー誤差を合わせて計算する関数を表すクラス"""

    def forward(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        """順伝播

        Args:
            x: 入力
            t: 教師データ

        Returns:
            y: 出力
        """
        N = x.shape[0]
        log_z = utils.logsumexp(x, axis=1)
        log_p = x - log_z
        log_p = log_p[np.arange(N), t.ravel()]
        y = -log_p.sum() / np.float32(N)
        return y

    def backward(self, gy: Variable) -> Variable:
        """逆伝播

        Args:
            gy: 出力側から伝わる微分

        Returns:
            gx: 入力側に伝わる微分
        """
        x, t = self.inputs
        N, CLS_NUM = x.shape

        gy *= 1 / N
        y = softmax(x)
        xp = cuda.get_array_module(x)
        t_onehot = xp.eye(CLS_NUM, dtype=t.dtype)[t.data]
        y = (y - t_onehot) * gy
        return y


def softmax_cross_entropy(x: Variable, t: Variable) -> Variable:
    """ソフトマックス関数と交差エントロピー誤差を合わせて計算する関数

    Args:
        x: 入力
        t: 教師データ

    Returns:
        y: 出力
    """
    return SoftmaxCrossEntropy()(x, t)


def sigmoid_cross_entropy(x: Variable, t: Variable) -> Variable:
    """シグモイド関数と交差エントロピー誤差を合わせて計算する関数

    Args:
        x: 入力
        t: 教師データ

    Returns:
        y: 出力
    """
    if x.ndim != t.ndim:
        t = t.reshape(*x.shape)
    x, t = as_variable(x), as_variable(t)
    N = len(x)
    p = sigmoid(x)
    p = clip(p, 1e-15, 1.0)
    tlog_p = t * log(p) + (1 - t) * log(1 - p)
    y = -1 * sum(tlog_p) / N
    return y


def binary_cross_entropy(p: np.ndarray, t: np.ndarray) -> np.ndarray:
    """二値交差エントロピー誤差を計算する関数

    Args:
        p: 予測結果
        t: 正解データ

    Returns:
        y: 出力
    """
    if p.ndim != t.ndim:
        t = t.reshape(*p.shape)
    N = len(t)
    p = clip(p, 1e-15, 0.999)
    tlog_p = t * log(p) + (1 - t) * log(1 - p)
    y = -1 * sum(tlog_p) / N
    return y


# =============================================================================
# accuracy / dropout / batch_norm / embed_id
# =============================================================================
def accuracy(y: Variable, t: Variable) -> Variable:
    """正解率を計算する関数

    Args:
        y: 予測結果
        t: 正解データ

    Returns:
        acc: 正解率

    Note:
        この関数は微分不可能
    """
    y, t = as_variable(y), as_variable(t)

    pred = y.data.argmax(axis=1).reshape(t.shape)
    result = pred == t.data
    acc = result.mean()
    return Variable(as_array(acc))


def dropout(x: Variable, dropout_ratio: float = 0.5):
    """Dropoutを行う関数

    Args:
        x: 入力
        dropout_ratio: Dropoutの割合

    Returns:
        y: 出力
    """
    x = as_variable(x)

    if dezero.Config.train:
        xp = cuda.get_array_module(x)
        mask = xp.random.rand(*x.shape) > dropout_ratio
        scale = xp.array(1.0 - dropout_ratio).astype(x.dtype)
        y = x * mask / scale
        return y
    else:
        return x


class BatchNorm(Function):
    """Batch Normalizationを行うクラス

    Attributes:
        avg_mean(np.ndarray): 平均の移動平均
        avg_var(np.ndarray): 分散の移動平均
        decay(float): 移動平均の減衰率
        eps(float): 分散の微小値
        inv_std(np.ndarray): 逆数の標準偏差
    """

    def __init__(
        self, mean: np.ndarray, var: np.ndarray, decay: float, eps: float
    ) -> None:
        """コンストラクタ

        Args:
            mean: 平均
            var: 分散
            decay: 移動平均の減衰率
            eps: 分散の微小値
        """
        self.avg_mean = mean
        self.avg_var = var
        self.decay = decay
        self.eps = eps
        self.inv_std = None

    def forward(self, x: np.ndarray, gamma: np.ndarray, beta: np.ndarray) -> np.ndarray:
        """順伝播

        Args:
            x: 入力
            gamma: スケール係数
            beta: シフト係数

        Returns:
            y: 出力
        """
        assert x.ndim == 2 or x.ndim == 4

        x_ndim = x.ndim
        if x_ndim == 4:
            N, C, H, W = x.shape
            # (N, C, H, W) -> (N*H*W, C)
            x = x.transpose(0, 2, 3, 1).reshape(-1, C)

        xp = cuda.get_array_module(x)

        if dezero.Config.train:
            mean = x.mean(axis=0)
            var = x.var(axis=0)
            inv_std = 1 / xp.sqrt(var + self.eps)
            xc = (x - mean) * inv_std

            m = x.size // gamma.size
            s = m - 1.0 if m - 1.0 > 1.0 else 1.0
            adjust = m / s
            self.avg_mean *= self.decay
            self.avg_mean += (1 - self.decay) * mean
            self.avg_var *= self.decay
            self.avg_var += (1 - self.decay) * adjust * var
            self.inv_std = inv_std
        else:
            inv_std = 1 / xp.sqrt(self.avg_var + self.eps)
            xc = (x - self.avg_mean) * inv_std
        y = gamma * xc + beta

        if x_ndim == 4:
            # (N*H*W, C) -> (N, C, H, W)
            y = y.reshape(N, H, W, C).transpose(0, 3, 1, 2)
        return y

    def backward(self, gy: Variable) -> tuple[Variable, Variable, Variable]:
        """逆伝播

        Args:
            gy: 出力側から伝わる微分

        Returns:
            gx: 入力側に伝わる微分
            ggamma: スケール係数に伝わる微分
            gbeta: シフト係数に伝わる微分
        """
        gy_ndim = gy.ndim
        if gy_ndim == 4:
            N, C, H, W = gy.shape
            gy = gy.transpose(0, 2, 3, 1).reshape(-1, C)

        x, gamma, beta = self.inputs
        batch_size = len(gy)

        if x.ndim == 4:
            N, C, H, W = x.shape
            x = x.transpose(0, 2, 3, 1).reshape(-1, C)
        mean = x.sum(axis=0) / batch_size
        xc = (x - mean) * self.inv_std

        gbeta = sum(gy, axis=0)
        ggamma = sum(xc * gy, axis=0)
        gx = gy - gbeta / batch_size - xc * ggamma / batch_size
        gx *= gamma * self.inv_std

        if gy_ndim == 4:
            gx = gx.reshape(N, H, W, C).transpose(0, 3, 1, 2)
        return gx, ggamma, gbeta


def batch_norm(
    x: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    mean: np.ndarray,
    var: np.ndarray,
    decay: float = 0.9,
    eps: float = 2e-5,
) -> np.ndarray:
    """Batch Normalizationを行う関数

    Args:
        x: 入力
        gamma: スケール係数
        beta: シフト係数
        mean: 平均
        var: 分散
        decay: 移動平均の減衰率
        eps: 分散の微小値

    Returns:
        y: 出力
    """
    return BatchNorm(mean, var, decay, eps)(x, gamma, beta)


def embed_id(x: np.ndarray, W: np.ndarray) -> np.ndarray:
    """単語IDを単語ベクトルに変換する関数

    Args:
        x: 単語ID
        W: 単語ベクトルの重みパラメータ

    Returns:
        y: 単語ベクトル
    """
    return W[x]


# =============================================================================
# max / min / clip
# =============================================================================
class Max(Function):
    """最大値を求める

    Args:
        axis: 最大値を求める軸
        keepdims: 次元を保持するかどうか
    """

    def __init__(self, axis: int | tuple[int] = None, keepdims: bool = False) -> None:
        """コンストラクタ

        Args:
            axis: 最大値を求める軸
            keepdims: 次元を保持するかどうか
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
        y = x.max(axis=self.axis, keepdims=self.keepdims)
        return y

    def backward(self, gy: Variable) -> Variable:
        """逆伝播

        Args:
            gy: 出力側から伝わる微分

        Returns:
            gx: 入力側に伝わる微分
        """
        x = self.inputs[0]
        y = self.outputs[0]()  # weakref

        shape = utils.max_backward_shape(x, self.axis)
        gy = reshape(gy, shape)
        y = reshape(y, shape)
        cond = x.data == y.data
        gy = broadcast_to(gy, cond.shape)
        return gy * cond


class Min(Max):
    """最小値を求める"""

    def forward(self, x: np.ndarray) -> np.ndarray:
        """順伝播

        Args:
            x: 入力

        Returns:
            y: 出力
        """
        y = x.min(axis=self.axis, keepdims=self.keepdims)
        return y


def max(x: Variable, axis: int | tuple[int] = None, keepdims: bool = False) -> Variable:
    """最大値を求める関数

    Args:
        x: 入力

    Returns:
        y: 出力
    """
    return Max(axis, keepdims)(x)


def min(x: Variable, axis: int | tuple[int] = None, keepdims: bool = False) -> Variable:
    """最小値を求める関数

    Args:
        x: 入力

    Returns:
        y: 出力
    """
    return Min(axis, keepdims)(x)


class Clip(Function):
    """上限と下限の範囲内に収める関数を表すクラス

    Attributes:
        x_min (float): xの下限
        x_max (float): xの上限
    """

    def __init__(self, x_min: float, x_max: float) -> None:
        """コンストラクタ

        Args:
            x_min: xの下限
            x_max: xの上限
        """
        self.x_min = x_min
        self.x_max = x_max

    def forward(self, x: np.ndarray) -> np.ndarray:
        """順伝播

        Args:
            x: 入力

        Returns:
            y: 出力
        """
        xp = cuda.get_array_module(x)
        y = xp.clip(x, self.x_min, self.x_max)
        return y

    def backward(self, gy: Variable) -> Variable:
        """逆伝播

        Args:
            gy: 出力側から伝わる微分

        Returns:
            gx: 入力側に伝わる微分
        """
        x = self.inputs[0]
        mask = (x.data >= self.x_min) * (x.data <= self.x_max)
        gx = gy * mask
        return gx


def clip(x: Variable, x_min: float, x_max: float) -> Variable:
    """上限と下限の範囲内に収める関数

    Args:
        x: 入力
        x_min: xの下限
        x_max: xの上限

    Returns:
        y: 出力
    """
    return Clip(x_min, x_max)(x)


# =============================================================================
# conv2d / col2im / im2col / basic_math
# =============================================================================
from dezero.core import add, div, mul, neg, pow, rsub, sub
from dezero.functions_conv import (
    average_pooling,
    col2im,
    conv2d,
    conv2d_simple,
    deconv2d,
    im2col,
    pooling,
    pooling_simple,
)
