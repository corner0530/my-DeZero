import numpy as np

from dezero import utils
from dezero.core import Function, Variable, as_variable, as_array


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


class Exp(Function):
    """exp関数を表すクラス"""

    def forward(self, x: np.ndarray) -> np.ndarray:
        """順伝播

        Args:
            x: 入力

        Returns:
            y: 出力
        """
        y = np.exp(x)
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
        y = np.log(x)
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
        gx = np.zeros(self.in_shape)
        np.add.at(gx, self.slices, gy)
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


class Sigmoid(Function):
    """シグモイド関数を計算する関数を表すクラス"""

    def forward(self, x: np.ndarray) -> np.ndarray:
        """順伝播

        Args:
            x: 入力

        Returns:
            y: 出力
        """
        # y = 1 / (1 + np.exp(-x))
        y = np.tanh(x * 0.5) * 0.5 + 0.5  # 実装の改良
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
        y = np.maximum(x, 0.0)
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
        axis: ソフトマックス関数を適用する軸
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
        y = x - x.max(axis=self.axis, keepdims=True)
        y = np.exp(y)
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
        t_onehot = np.eye(CLS_NUM, dtype=t.dtype)[t.data]
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
    result = (pred == t.data)
    acc = result.mean()
    return Variable(as_array(acc))


class Clip(Function):
    """上限と下限の範囲内に収める関数を表すクラス

    Attributes:
        x_min: xの下限
        x_max: xの上限
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
        y = np.clip(x, self.x_min, self.x_max)
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
