import numpy as np

from dezero import cuda
from dezero.core import Function, Variable, as_variable
from dezero.functions import broadcast_to, linear
from dezero.utils import get_conv_outsize, get_deconv_outsize, pair


# =============================================================================
# [simple version] conv2d_simple / pooling_simple
# =============================================================================
def conv2d_simple(
    x: Variable,
    w: Variable,
    b: Variable = None,
    stride: int | tuple[int] = 1,
    pad: int | tuple[int] = 0,
) -> Variable:
    """畳み込み層の順伝播の簡易版

    Args:
        x: 入力
        w: 重み
        b: バイアス
        stride: ストライド
        pad: パディング

    Returns:
        y: 出力
    """
    x, w = as_variable(x), as_variable(w)

    N, C, H, W = x.shape
    OC, C, KH, KW = w.shape
    SH, SW = pair(stride)
    PH, PW = pair(pad)
    OH = get_conv_outsize(H, KH, SH, PH)
    OW = get_conv_outsize(W, KW, SW, PW)

    col = im2col(x, (KH, KW), stride, pad, to_matrix=True)  # 入力データを展開
    w = w.reshape(OC, -1).transpose()  # カーネルを展開
    t = linear(col, w, b)  # 入力データとカーネルで線形変換
    y = t.reshape(N, OH, OW, OC).transpose(0, 3, 1, 2)
    return y


def pooling_simple(
    x: Variable,
    kernel_size: int | tuple[int],
    stride: int | tuple[int] = 1,
    pad: int | tuple[int] = 0,
) -> Variable:
    """プーリング層の順伝播の簡易版

    Args:
        x: 入力
        kernel_size: カーネルサイズ
        stride: ストライド
        pad: パディング

    Returns:
        y: 出力
    """
    x = as_variable(x)

    N, C, H, W = x.shape
    KH, KW = pair(kernel_size)
    PH, PW = pair(pad)
    SH, SW = pair(stride)
    OH = get_conv_outsize(H, KH, SH, PH)
    OW = get_conv_outsize(W, KW, SW, PW)

    col = im2col(x, kernel_size, stride, pad, to_matrix=True)  # 入力データを展開
    col = col.reshape(-1, KH * KW)
    y = col.max(axis=1)  # 最大値を取得
    y = y.reshape(N, OH, OW, C).transpose(0, 3, 1, 2)  # 出力データを整形
    return y


# =============================================================================
#  conv2d / deconv2d
# =============================================================================
class Conv2d(Function):
    """畳み込み層の順伝播のクラス

    Attributes:
        stride (int | tuple[int]): ストライド
        pad (int | tuple[int]): パディング
    """

    def __init__(
        self,
        stride: int | tuple[int] = 1,
        pad: int | tuple[int] = 0,
    ) -> None:
        """コンストラクタ

        Args:
            stride: ストライド
            pad: パディング
        """
        super().__init__()
        self.stride = pair(stride)
        self.pad = pair(pad)

    def forward(self, x: Variable, w: Variable, b: Variable = None) -> Variable:
        """順伝播

        Args:
            x: 入力
            w: 重み
            b: バイアス

        Returns:
            y: 出力
        """
        xp = cuda.get_array_module(x)

        KH, KW = w.shape[2:]
        col = im2col_array(x, (KH, KW), self.stride, self.pad, to_matrix=False)

        y = xp.tensordot(col, w, ((1, 2, 3), (1, 2, 3)))
        if b is not None:
            y += b
        y = xp.rollaxis(y, 3, 1)
        return y

    def backward(self, gy: Variable) -> tuple[Variable, Variable, Variable]:
        """逆伝播

        Args:
            gy: 出力側から伝わる微分

        Returns:
            gx: 入力側に伝わる微分
            gw: 重み側に伝わる微分
            gb: バイアス側に伝わる微分
        """
        x, w, b = self.inputs
        # gx
        gx = deconv2d(
            gy,
            w,
            b=None,
            stride=self.stride,
            pad=self.pad,
            outsize=(x.shape[2], x.shape[3]),
        )
        # gw
        gw = Conv2dGradW(self)(x, gy)
        # gb
        gb = None
        if b.data is not None:
            gb = gy.sum(axis=(0, 2, 3))
        return gx, gw, gb


def conv2d(
    x: Variable,
    w: Variable,
    b: Variable = None,
    stride: int | tuple[int] = 1,
    pad: int | tuple[int] = 0,
) -> Variable:
    """畳み込み層の順伝播

    Args:
        x: 入力
        w: 重み
        b: バイアス
        stride: ストライド
        pad: パディング

    Returns:
        y: 出力
    """
    return Conv2d(stride, pad)(x, w, b)


class Deconv2d(Function):
    """畳み込み層の逆伝播のクラス

    Attributes:
        stride (int | tuple[int]): ストライド
        pad (int | tuple[int]): パディング
        outsize (tuple[int]): 出力サイズ
        no_bias (bool): バイアスを使用しないかどうか
    """

    def __init__(
        self,
        stride: int | tuple[int] = 1,
        pad: int | tuple[int] = 0,
        outsize: tuple[int] = None,
    ) -> None:
        """コンストラクタ

        Args:
            stride: ストライド
            pad: パディング
            outsize: 出力サイズ
        """
        super().__init__()
        self.stride = pair(stride)
        self.pad = pair(pad)
        self.outsize = outsize

    def forward(self, x: Variable, w: Variable, b: Variable = None) -> Variable:
        """順伝播

        Args:
            x: 入力
            w: 重み
            b: バイアス

        Returns:
            y: 出力
        """
        xp = cuda.get_array_module(x)

        SH, SW = self.stride
        PH, PW = self.pad
        C, OC, KH, KW = w.shape
        N, C, H, W = x.shape
        if self.outsize is None:
            out_h = get_deconv_outsize(H, KH, SH, PH)
            out_w = get_deconv_outsize(W, KW, SW, PW)
        else:
            out_h, out_w = pair(self.outsize)
        img_shape = (N, OC, out_h, out_w)

        gcol = xp.tensordot(w, x, (0, 1))
        gcol = xp.rollaxis(gcol, 3)
        y = col2im_array(
            gcol, img_shape, (KH, KW), self.stride, self.pad, to_matrix=False
        )

        if b is not None:
            self.no_bias = True
            y += b.reshape((1, b.size, 1, 1))
        return y

    def backward(self, gy: Variable) -> tuple[Variable, Variable, Variable]:
        """逆伝播

        Args:
            gy: 出力側から伝わる微分

        Returns:
            gx: 入力側に伝わる微分
            gw: 重み側に伝わる微分
            gb: バイアス側に伝わる微分
        """
        x, w, b = self.inputs

        # gx
        gx = conv2d(gy, w, b=None, stride=self.stride, pad=self.pad)
        # gw
        gw = Conv2dGradW(self)
        # gb
        gb = None
        if b.data is not None:
            gb = gy.sum(axis=(0, 2, 3))
        return gx, gw, gb


def deconv2d(
    x: Variable,
    w: Variable,
    b: Variable = None,
    stride: int | tuple[int] = 1,
    pad: int | tuple[int] = 0,
    outsize: tuple[int] = None,
) -> Variable:
    """畳み込み層の逆伝播

    Args:
        x: 入力
        w: 重み
        b: バイアス
        stride: ストライド
        pad: パディング
        outsize: 出力サイズ

    Returns:
        y: 出力
    """
    return Deconv2d(stride, pad, outsize)(x, w, b)


class Conv2dGradW(Function):
    """畳み込み層の重み側の微分のクラス

    Attributes:
        kernel_size (int | tuple[int]): カーネルサイズ
        stride (int | tuple[int]): ストライド
        pad (int | tuple[int]): パディング
    """

    def __init__(self, conv2d: Conv2d | Deconv2d) -> None:
        """コンストラクタ

        Args:
            conv2d: Conv2dまたはDeconv2dのインスタンス
        """
        w = conv2d.inputs[1]
        kh, kw = w.shape[2:]
        self.kernel_size = (kh, kw)
        self.stride = conv2d.stride
        self.pad = conv2d.pad

    def forward(self, x: Variable, gy: Variable) -> Variable:
        """順伝播

        Args:
            x: 入力
            gy: 出力側から伝わる微分

        Returns:
            gw: 重み側に伝わる微分
        """
        xp = cuda.get_array_module(x)
        col = im2col_array(x, self.kernel_size, self.stride, self.pad, to_matrix=False)
        gw = xp.tensordot(gy, col, axes=((0, 2, 3), (0, 4, 5)))
        return gw

    def backward(self, gys: list[Variable]) -> tuple[Variable]:
        """逆伝播

        Args:
            gys: 出力側から伝わる微分

        Returns:
            gx: 入力側に伝わる微分
            ggy: 出力側に伝わる微分
        """
        x, gy = self.inputs
        gw = self.outputs[0]

        xh, xw = x.shape[2:]
        gx = deconv2d(gy, gw, stride=self.stride, pad=self.pad, outsize=(xh, xw))
        ggy = conv2d(x, gw, stride=self.stride, pad=self.pad)
        return gx, ggy


# =============================================================================
#  pooling(max-pooling) / average_pooling
# =============================================================================
class Pooling(Function):
    """プーリング層のクラス

    Attributes:
        kernel_size (int | tuple[int]): カーネルサイズ
        stride (int | tuple[int]): ストライド
        pad (int | tuple[int]): パディング
    """

    def __init__(
        self,
        kernel_size: int | tuple[int],
        stride: int | tuple[int] = 1,
        pad: int | tuple[int] = 0,
    ) -> None:
        """コンストラクタ

        Args:
            kernel_size: カーネルサイズ
            stride: ストライド
            pad: パディング
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad

    def forward(self, x: Variable) -> Variable:
        """順伝播

        Args:
            x: 入力

        Returns:
            y: 出力
        """
        col = im2col_array(x, self.kernel_size, self.stride, self.pad, to_matrix=False)

        N, C, KH, KW, OH, OW = col.shape
        col = col.reshape(N, C, KH * KW, OH, OW)
        self.indexes = col.argmax(axis=2)
        y = col.max(axis=2)
        return y

    def backward(self, gy: Variable) -> Variable:
        """逆伝播

        Args:
            gy: 出力側から伝わる微分

        Returns:
            gx: 入力側に伝わる微分
        """
        return Pooling2DGrad(self)(gy)


class Pooling2DGrad(Function):
    """プーリング層の微分のクラス

    Attributes:
        mpool2d (Pooling): Poolingのインスタンス
        kernel_size (int | tuple[int]): カーネルサイズ
        stride (int | tuple[int]): ストライド
        pad (int | tuple[int]): パディング
        input_shape (tuple[int]): 入力の形状
        dtype (numpy.dtype): 入力のデータ型
        indexes (numpy.ndarray): 最大値のインデックス
    """

    def __init__(self, mpool2d: Pooling) -> None:
        """コンストラクタ

        Args:
            mpool2d: Poolingのインスタンス
        """
        self.mpool2d = mpool2d
        self.kernel_size = mpool2d.kernel_size
        self.stride = mpool2d.stride
        self.pad = mpool2d.pad
        self.input_shape = mpool2d.inputs[0].shape
        self.dtype = mpool2d.inputs[0].dtype
        self.indexes = mpool2d.indexes

    def forward(self, gy: Variable) -> Variable:
        """順伝播

        Args:
            gy: 出力側から伝わる微分

        Returns:
            gx: 入力側に伝わる微分
        """
        xp = cuda.get_array_module(gy)

        N, C, OH, OW = gy.shape
        N, C, H, W = self.input_shape
        KH, KW = pair(self.kernel_size)

        gcol = xp.zeros((N * C * OH * OW * KH * KW), dtype=self.dtype)

        indexes = self.indexes.ravel() + xp.arange(
            0, self.indexes.size * KH * KW, KH * KW
        )

        gcol[indexes] = gy.ravel()
        gcol = gcol.reshape(N, C, OH, OW, KH, KW)
        gcol = xp.swapaxes(gcol, 2, 4)
        gcol = xp.swapaxes(gcol, 3, 5)

        gx = col2im_array(
            gcol, (N, C, H, W), self.kernel_size, self.stride, self.pad, to_matrix=False
        )
        return gx

    def backward(self, ggx: Variable) -> tuple[Variable]:
        """逆伝播

        Args:
            ggx: 出力側から伝わる微分

        Returns:
            f(ggx): 入力側に伝わる微分
        """
        f = Pooling2DWithIndexes(self.mpool2d)
        return f(ggx)


class Pooling2DWithIndexes(Function):
    """プーリング層のインデックスを保持するクラス

    Attributes:
        kernel_size (int | tuple[int]): カーネルサイズ
        stride (int | tuple[int]): ストライド
        pad (int | tuple[int]): パディング
        input_shpae (tuple[int]): 入力の形状
        dtype (numpy.dtype): 入力のデータ型
        indexes (numpy.ndarray): 最大値のインデックス
    """

    def __init__(self, mpool2d: Pooling) -> None:
        """コンストラクタ

        Args:
            mpool2d: Poolingのインスタンス
        """
        self.kernel_size = mpool2d.kernel_size
        self.stride = mpool2d.stride
        self.pad = mpool2d.pad
        self.input_shpae = mpool2d.inputs[0].shape
        self.dtype = mpool2d.inputs[0].dtype
        self.indexes = mpool2d.indexes

    def forward(self, x: Variable) -> Variable:
        """順伝播

        Args:
            x: 入力

        Returns:
            y: 出力
        """
        col = im2col_array(x, self.kernel_size, self.stride, self.pad, to_matrix=False)
        N, C, KH, KW, OH, OW = col.shape
        col = col.reshape(N, C, KH * KW, OH, OW)
        col = col.transpose(0, 1, 3, 4, 2).reshape(-1, KH * KW)
        indexes = self.indexes.ravel()
        col = col[np.arange(len(indexes)), indexes]
        return col.reshape(N, C, OH, OW)


def pooling(
    x: Variable,
    kernel_size: int | tuple[int],
    stride: int | tuple[int] = 1,
    pad: int | tuple[int] = 0,
) -> Variable:
    """プーリング層

    Args:
        x: 入力
        kernel_size: カーネルサイズ
        stride: ストライド
        pad: パディング

    Returns:
        y: 出力
    """
    return Pooling(kernel_size, stride, pad)(x)


class AveragePooling(Function):
    """平均プーリング層のクラス

    Attributes:
        kernel_size (int | tuple[int]): カーネルサイズ
        stride (int | tuple[int]): ストライド
        pad (int | tuple[int]): パディング
        input_shape (tuple[int]): 入力の形状
    """

    def __init__(
        self,
        kernel_size: int | tuple[int],
        stride: int | tuple[int] = 1,
        pad: int | tuple[int] = 0,
    ) -> None:
        """コンストラクタ

        Args:
            kernel_size: カーネルサイズ
            stride: ストライド
            pad: パディング
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.input_shape = None

    def forward(self, x: Variable) -> Variable:
        """順伝播

        Args:
            x: 入力

        Returns:
            y: 出力
        """
        self.input_shape = x.shape
        col = im2col_array(x, self.kernel_size, self.stride, self.pad, to_matrix=False)
        y = col.mean(axis=(2, 3))
        return y

    def backward(self, gy: Variable) -> Variable:
        """逆伝播

        Args:
            gy: 出力側から伝わる微分

        Returns:
            gx: 入力側に伝わる微分
        """
        N, C, OH, OW = gy.shape
        KW, KH = pair(self.kernel_size)
        gy /= KW * KH
        gcol = broadcast_to(gy.reshape(-1), (KH, KW, N * C * OH * OW))
        gcol = gcol.reshape(KH, KW, N, C, OH, OW).transpose(2, 3, 0, 1, 4, 5)
        gx = col2im(
            gcol,
            self.input_shape,
            self.kernel_size,
            self.stride,
            self.pad,
            to_matrix=False,
        )
        return gx


def average_pooling(
    x: Variable,
    kernel_size: int | tuple[int],
    stride: int | tuple[int] = 1,
    pad: int | tuple[int] = 0,
) -> Variable:
    """平均プーリング層

    Args:
        x: 入力
        kernel_size: カーネルサイズ
        stride: ストライド
        pad: パディング

    Returns:
        y: 出力
    """
    return AveragePooling(kernel_size, stride, pad)(x)


# =============================================================================
#  im2col / col2im
# =============================================================================
class Im2col(Function):
    """im2col関数のクラス

    Attributes:
        input_shape (tuple[int]): 入力の形状
        kernel_size (int | tuple[int]): カーネルサイズ
        stride (int | tuple[int]): ストライド
        pad (int | tuple[int]): パディング
        to_matrix (bool): 行列に変換するかどうか
    """

    def __init__(
        self,
        kernel_size: int | tuple[int],
        stride: int | tuple[int],
        pad: int | tuple[int],
        to_matrix: bool,
    ) -> None:
        """コンストラクタ

        Args:
            kernel_size: カーネルサイズ
            stride: ストライド
            pad: パディング
            to_matrix: 行列に変換するかどうか
        """
        super().__init__()
        self.input_shape = None
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.to_matrix = to_matrix

    def forward(self, x: Variable) -> Variable:
        """順伝播

        Args:
            x: 入力

        Returns:
            y: 出力
        """
        self.input_shape = x.shape
        y = im2col_array(x, self.kernel_size, self.stride, self.pad, self.to_matrix)
        return y

    def backward(self, gy: Variable) -> Variable:
        """逆伝播

        Args:
            gy: 出力側から伝わる微分

        Returns:
            gx: 入力側に伝わる微分
        """
        gx = col2im(
            gy,
            self.input_shape,
            self.kernel_size,
            self.stride,
            self.pad,
            self.to_matrix,
        )
        return gx


def im2col(
    x: Variable,
    kernel_size: int | tuple[int],
    stride: int | tuple[int] = 1,
    pad: int | tuple[int] = 0,
    to_matrix: bool = True,
) -> Variable:
    """カーネルを適用する領域を取り出して1列に整形し行列に変換する

    Args:
        x: 入力(N, C, H, W)
        kernel_size: カーネルサイズ
        stride: ストライド
        pad: パディング
        to_matrix: 行列に変換するかどうか

    Returns:
        to_matrix=Trueの場合は(N * OH * OW, C * KH * KW)の行列
        to_matrix=Falseの場合は(N, C, KH, KW, OH, OW)の配列
    """
    y = Im2col(kernel_size, stride, pad, to_matrix)(x)
    return y


class Col2im(Function):
    """col2im関数のクラス

    Attributes:
        input_shape (tuple[int]): 入力の形状
        kernel_size (int | tuple[int]): カーネルサイズ
        stride (int | tuple[int]): ストライド
        pad (int | tuple[int]): パディング
        to_matrix (bool): 行列に変換するかどうか
    """

    def __init__(
        self,
        input_shape: int | tuple[int],
        kernel_size: int | tuple[int],
        stride: int | tuple[int],
        pad: int | tuple[int],
        to_matrix: bool,
    ) -> None:
        """コンストラクタ

        Args:
            input_shape: 入力の形状
            kernel_size: カーネルサイズ
            stride: ストライド
            pad: パディング
            to_matrix: 行列に変換するかどうか
        """
        super().__init__()
        self.input_shape = input_shape
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.to_matrix = to_matrix

    def forward(self, x: Variable) -> Variable:
        """順伝播

        Args:
            x: 入力

        Returns:
            y: 出力
        """
        y = col2im_array(
            x, self.input_shape, self.kernel_size, self.stride, self.pad, self.to_matrix
        )
        return y

    def backward(self, gy: Variable) -> Variable:
        """逆伝播

        Args:
            gy: 出力側から伝わる微分

        Returns:
            gx: 入力側に伝わる微分
        """
        gx = im2col(gy, self.kernel_size, self.stride, self.pad, self.to_matrix)
        return gx


def col2im(
    x: Variable,
    input_shape: int | tuple[int],
    kernel_size: int | tuple[int],
    stride: int | tuple[int] = 1,
    pad: int | tuple[int] = 0,
    to_matrix: bool = True,
) -> Variable:
    """im2colの逆変換

    Args:
        x: 入力
        input_shape: 入力の形状
        kernel_size: カーネルサイズ
        stride: ストライド
        pad: パディング
        to_matrix: 行列に変換するかどうか

    Returns:
        to_matrix=Trueの場合は(N, C, H, W)の配列
        to_matrix=Falseの場合は(N, C, KH, KW, OH, OW)の配列
    """
    return Col2im(input_shape, kernel_size, stride, pad, to_matrix)(x)


# =============================================================================
#  numpy im2col
# =============================================================================
def im2col_array(
    img: np.ndarray,
    kernel_size: int | tuple[int],
    stride: int | tuple[int],
    pad: int | tuple[int],
    to_matrix: bool = True,
) -> np.ndarray:
    """im2colのnumpy版

    Args:
        img: 入力(N, C, H, W)
        kernel_size: カーネルサイズ
        stride: ストライド
        pad: パディング
        to_matrix: 行列に変換するかどうか

    Returns:
        to_matrix=Trueの場合は(N * OH * OW, C * KH * KW)の行列
        to_matrix=Falseの場合は(N, C, KH, KW, OH, OW)の配列
    """
    N, C, H, W = img.shape
    KH, KW = pair(kernel_size)
    SH, SW = pair(stride)
    PH, PW = pair(pad)
    OH = get_conv_outsize(H, KH, SH, PH)
    OW = get_conv_outsize(W, KW, SW, PW)

    xp = cuda.get_array_module(img)
    if xp != np:
        col = _im2col_gpu(img, kernel_size, stride, pad)
    else:
        img = np.pad(
            img,
            ((0, 0), (0, 0), (PH, PH + SH - 1), (PW, PW + SW - 1)),
            mode="constant",
            constant_values=(0,),
        )
        col = np.ndarray((N, C, KH, KW, OH, OW), dtype=img.dtype)

        for j in range(KH):
            j_lim = j + SH * OH
            for i in range(KW):
                i_lim = i + SW * OW
                col[:, :, j, i, :, :] = img[:, :, j:j_lim:SH, i:i_lim:SW]

    if to_matrix:
        col = col.transpose((0, 4, 5, 1, 2, 3)).reshape((N * OH * OW, -1))

    return col


def col2im_array(
    col: np.ndarray,
    img_shape: int | tuple[int],
    kernel_size: int | tuple[int],
    stride: int | tuple[int],
    pad: int | tuple[int],
    to_matrix: bool = True,
) -> np.ndarray:
    """im2colのnumpy版

    Args:
        col: 入力(N * OH * OW, C * KH * KW)の行列
        img_shape: 入力の形状
        kernel_size: カーネルサイズ
        stride: ストライド
        pad: パディング
        to_matrix: 行列に変換するかどうか

    Returns:
        to_matrix=Trueの場合は(N, C, H, W)の配列
        to_matrix=Falseの場合は(N, C, KH, KW, OH, OW)の配列
    """
    N, C, H, W = img_shape
    KH, KW = pair(kernel_size)
    SH, SW = pair(stride)
    PH, PW = pair(pad)
    OH = get_conv_outsize(H, KH, SH, PH)
    OW = get_conv_outsize(W, KW, SW, PW)

    if to_matrix:
        col = col.reshape(N, OH, OW, C, KH, KW).transpose(0, 3, 4, 5, 1, 2)

    xp = cuda.get_array_module(col)
    if xp != np:
        img = _col2im_gpu(col, SH, SW, PH, PW, H, W)
        return img
    else:
        img = np.zeros(
            (N, C, H + 2 * PH + SH - 1, W + 2 * PW + SW - 1), dtype=col.dtype
        )
        for j in range(KH):
            j_lim = j + SH * OH
            for i in range(KW):
                i_lim = i + SW * OW
                img[:, :, j:j_lim:SH, i:i_lim:SW] += col[:, :, j, i, :, :]
        return img[:, :, PH : H + PH, PW : W + PW]


def _im2col_gpu(
    img: np.ndarray,
    kernel_size: int | tuple[int],
    stride: int | tuple[int],
    pad: int | tuple[int],
) -> np.ndarray:
    """im2colのGPU版

    Args:
        img: 入力
        kernel_size: カーネルサイズ
        stride: ストライド
        pad: パディング

    Returns:
        col: 出力
    """
    n, c, h, w = img.shape
    kh, kw = pair(kernel_size)
    sy, sx = pair(stride)
    ph, pw = pair(pad)
    out_h = get_conv_outsize(h, kh, sy, ph)
    out_w = get_conv_outsize(w, kw, sx, pw)
    dy, dx = 1, 1
    col = cuda.cupy.empty((n, c, kh, kw, out_h, out_w), dtype=img.dtype)

    cuda.cupy.ElementwiseKernel(
        "raw T img, int32 h, int32 w, int32 out_h, int32 out_w,"
        "int32 kh, int32 kw, int32 sy, int32 sx, int32 ph, int32 pw,"
        "int32 dy, int32 dx",
        "T col",
        """
           int c0 = i / (kh * kw * out_h * out_w);
           int ky = i / (kw * out_h * out_w) % kh;
           int kx = i / (out_h * out_w) % kw;
           int out_y = i / out_w % out_h;
           int out_x = i % out_w;
           int in_y = ky * dy + out_y * sy - ph;
           int in_x = kx * dx + out_x * sx - pw;
           if (in_y >= 0 && in_y < h && in_x >= 0 && in_x < w) {
             col = img[in_x + w * (in_y + h * c0)];
           } else {
             col = 0;
           }
        """,
        "im2col",
    )(img.reduced_view(), h, w, out_h, out_w, kh, kw, sy, sx, ph, pw, dy, dx, col)

    return col


def _col2im_gpu(
    col: np.ndarray, sy: int, sx: int, ph: int, pw: int, h: int, w: int
) -> np.ndarray:
    """col2imのGPU版

    Args:
        col: 入力
        sy: ストライド
        sx: ストライド
        ph: パディング
        pw: パディング
        h: 入力の高さ
        w: 入力の幅

    Returns:
        img: 出力
    """
    n, c, kh, kw, out_h, out_w = col.shape
    dx, dy = 1, 1
    img = cuda.cupy.empty((n, c, h, w), dtype=col.dtype)

    cuda.cupy.ElementwiseKernel(
        "raw T col, int32 h, int32 w, int32 out_h, int32 out_w,"
        "int32 kh, int32 kw, int32 sy, int32 sx, int32 ph, int32 pw,"
        "int32 dx, int32 dy",
        "T img",
        """
           int c0 = i / (h * w);
           int y  = i / w % h;
           int x  = i % w;
           T val = 0;
           for (int ky = 0; ky < kh; ++ky) {
             int out_y = (y + ph - ky * dy);
             if (0 > out_y || out_y >= out_h * sy) continue;
             if (out_y % sy != 0) continue;
             out_y /= sy;
             for (int kx = 0; kx < kw; ++kx) {
               int out_x = (x + pw - kx * dx);
               if (0 > out_x || out_x >= out_w * sx) continue;
               if (out_x % sx != 0) continue;
               out_x /= sx;
               int k = out_y + out_h * (kx + kw * (ky + kh * c0));
               val = val + col[out_x + out_w * k];
             }
           }
           img = val;
        """,
        "col2im",
    )(col.reduced_view(), h, w, out_h, out_w, kh, kw, sy, sx, ph, pw, dx, dy, img)
    return img
