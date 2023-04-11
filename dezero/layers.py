import os
import weakref

import numpy as np

import dezero.functions as F
from dezero import cuda
from dezero.core import Parameter
from dezero.utils import pair


class Layer:
    """パラメータを保持し，パラメータを使った変換を行うクラス

    Attributes:
        _params (set): パラメータを保持する集合
    """

    def __init__(self) -> None:
        """コンストラクタ"""
        self._params = set()

    def __setattr__(self, name: str, value: object) -> None:
        """インスタンス変数を追加するときに呼ばれる

        Args:
            name: インスタンス変数の名前
            value: インスタンス変数の値
        """
        if isinstance(value, (Parameter, Layer)):
            # Parameterクラスのインスタンスの場合は，_paramsに追加する
            self._params.add(name)
        super().__setattr__(name, value)

    def __call__(self, *inputs: object) -> tuple[object] | object:
        """インスタンスを関数のように呼び出したときに呼ばれる

        Args:
            inputs: 入力

        Returns:
            outputs: 出力
        """
        outputs = self.forward(*inputs)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        self.inputs = [weakref.ref(x) for x in inputs]
        self.outputs = [weakref.ref(y) for y in outputs]
        return outputs if 1 < len(outputs) else outputs[0]

    def forward(self, *inputs: object) -> tuple[object] | object:
        """順伝播

        Args:
            inputs: 入力

        Returns:
            outputs: 出力

        Raises:
            NotImplementedError: 未実装の場合
        """
        raise NotImplementedError()

    def params(self) -> Parameter:
        """自身が持つParameterインスタンスを返す

        Yields:
            自身が持つParameterインスタンス
        """
        for name in self._params:
            obj = self.__dict__[name]

            if isinstance(obj, Layer):
                # Layerクラスのインスタンスの場合は，そのparamsを返す
                yield from obj.params()
            else:
                yield obj

    def cleargrads(self) -> None:
        """全てのパラメータの勾配をリセット"""
        for param in self.params():
            param.cleargrad()

    def to_cpu(self) -> None:
        """全てのパラメータをCPUに移動"""
        for param in self.params():
            param.to_cpu()

    def to_gpu(self) -> None:
        """全てのパラメータをGPUに移動"""
        for param in self.params():
            param.to_gpu()

    def _flatten_params(self, params_dict: dict, parent_key: str = "") -> None:
        """パラメータを1つの辞書にまとめる

        Args:
            params_dict: パラメータを保持する辞書
            parent_key: 親のパラメータの名前
        """
        for name in self._params:
            obj = self.__dict__[name]
            key = parent_key + "/" + name if parent_key else name

            if isinstance(obj, Layer):
                obj._flatten_params(params_dict, key)
            else:
                params_dict[key] = obj

    def save_weights(self, path: str) -> None:
        """パラメータをファイルに保存する

        Args:
            path: 保存先のパス
        """
        self.to_cpu()

        params_dict = {}
        self._flatten_params(params_dict)
        array_dict = {key: param.data for key, param in params_dict.items()}
        try:
            np.savez_compressed(path, **array_dict)
        except (Exception, KeyboardInterrupt):
            if os.path.exists(path):
                os.remove(path)
            raise

    def load_weights(self, path: str) -> None:
        """ファイルからパラメータを読み込む

        Args:
            path: 読み込むファイルのパス
        """
        npz = np.load(path)
        params_dict = {}
        self._flatten_params(params_dict)
        for key, param in params_dict.items():
            param.data = npz[key]


# =============================================================================
# Linear / Conv2d / Deconv2d
# =============================================================================
class Linear(Layer):
    """線形変換を行う層

    Attributes:
        w (Parameter): 重み
        b (Parameter): バイアス
    """

    def __init__(
        self,
        out_size: int,
        nobias: bool = False,
        dtype: type = np.float32,
        in_size: int = None,
    ) -> None:
        """コンストラクタ

        Args:
            out_size: 出力サイズ
            nobias: バイアスを使用するかどうか
            dtype: パラメータのデータ型
            in_size: 入力サイズ
        """
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.dtype = dtype

        self.W = Parameter(None, name="W")
        if self.in_size is not None:  # in_sizeが指定されていない場合は後回し
            self._init_w()

        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_size, dtype=dtype), name="b")

    def _init_w(self, xp: object = np) -> None:
        """重みの初期化

        Args:
            xp: npかcupy
        """
        w_data = xp.random.randn(self.in_size, self.out_size).astype(
            self.dtype
        ) * np.sqrt(1 / self.in_size)
        self.W.data = w_data

    def forward(self, x: object) -> object:
        """順伝播

        Args:
            x: 入力

        Returns:
            y: 出力
        """
        # データを流すタイミングで重みを初期化
        if self.W.data is None:
            self.in_size = x.shape[1]
            xp = cuda.get_array_module(x)
            self._init_w(xp)

        y = F.linear(x, self.W, self.b)
        return y


class Conv2d(Layer):
    """2次元畳み込みを行う層

    Attributes:
        w (Parameter): 重み
        b (Parameter): バイアス
        in_channels (int): 入力チャンネル数
        out_channels (int): 出力チャンネル数
        kernel_size (int | tuple[int]): カーネルサイズ
        stride (int | tuple[int]): ストライド
        pad (int | tuple[int]): パディング
        dtype (np.dtype): パラメータのデータ型
    """

    def __init__(
        self,
        out_channels: int,
        kernel_size: int | tuple[int],
        stride: int | tuple[int] = 1,
        pad: int | tuple[int] = 0,
        nobias: bool = False,
        dtype: np.dtype = np.float32,
        in_channels: int = None,
    ) -> None:
        """コンストラクタ

        Args:
            out_channels: 出力チャンネル数
            kernel_size: カーネルサイズ
            stride: ストライド
            pad: パディング
            nobias: バイアスを使用するかどうか
            dtype: パラメータのデータ型
            in_channels: 入力チャンネル数
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.dtype = dtype

        self.W = Parameter(None, name="W")
        if in_channels is not None:
            self._init_w()

        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_channels, dtype=dtype), name="b")

    def _init_w(self, xp: object = np) -> None:
        """重みの初期化

        Args:
            xp: npかcupy
        """
        C = self.in_channels
        OC = self.out_channels
        KH, KW = pair(self.kernel_size)
        scale = np.sqrt(1 / (C * KH * KW))
        w_data = xp.random.randn(OC, C, KH, KW).astype(self.dtype) * scale
        self.W.data = w_data

    def forward(self, x: object) -> object:
        """順伝播

        Args:
            x: 入力

        Returns:
            y: 出力
        """
        if self.W.data is None:
            self.in_channels = x.shape[1]
            xp = cuda.get_array_module(x)
            self._init_w(xp)

        y = F.conv2d(x, self.W, self.b, self.stride, self.pad)
        return y


class Deconv2d(Layer):
    """2次元転置畳み込みを行う層

    Attributes:
        w (Parameter): 重み
        b (Parameter): バイアス
        in_channels (int): 入力チャンネル数
        out_channels (int): 出力チャンネル数
        kernel_size (int | tuple[int]): カーネルサイズ
        stride (int | tuple[int]): ストライド
        pad (int | tuple[int]): パディング
        dtype (np.dtype): パラメータのデータ型
    """

    def __init__(
        self,
        out_channels: int,
        kernel_size: int | tuple[int],
        stride: int | tuple[int] = 1,
        pad: int | tuple[int] = 0,
        nobias: bool = False,
        dtype: np.dtype = np.float32,
        in_channels: int = None,
    ) -> None:
        """コンストラクタ

        Args:
            out_channels: 出力チャンネル数
            kernel_size: カーネルサイズ
            stride: ストライド
            pad: パディング
            nobias: バイアスを使用するかどうか
            dtype: パラメータのデータ型
            in_channels: 入力チャンネル数
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.dtype = dtype

        self.W = Parameter(None, name="W")
        if in_channels is not None:
            self._init_w()

        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_channels, dtype=dtype), name="b")

    def _init_w(self, xp: object = np) -> None:
        """重みの初期化

        Args:
            xp: npかcupy
        """
        C = self.in_channels
        OC = self.out_channels
        KH, KW = pair(self.kernel_size)
        scale = np.sqrt(1 / (C * KH * KW))
        w_data = xp.random.randn(C, OC, KH, KW).astype(self.dtype) * scale
        self.w.data = w_data

    def forward(self, x: object) -> object:
        """順伝播

        Args:
            x: 入力

        Returns:
            y: 出力
        """
        if self.W.data is None:
            self.in_channels = x.shape[1]
            xp = cuda.get_array_module(x)
            self._init_w(xp)

        y = F.deconv2d(x, self.W, self.b, self.stride, self.pad)
        return y


class RNN(Layer):
    """RNNのクラス

    Attributes:
        x2h (Linear): 入力から隠れ状態への変換
        h2h (Linear): 隠れ状態から隠れ状態への変換
        h (ndarray): 隠れ状態
    """

    def __init__(self, hidden_size: int, in_size: int = None) -> None:
        """コンストラクタ

        Args:
            hidden_size: 隠れ状態のサイズ
            in_size: 入力サイズ
        """
        super().__init__()
        self.x2h = Linear(hidden_size, in_size=in_size)
        self.h2h = Linear(hidden_size, in_size=in_size, nobias=True)
        self.h = None

    def reset_state(self) -> None:
        """隠れ状態をリセットする"""
        self.h = None

    def forward(self, x: object) -> object:
        """順伝播

        Args:
            x: 入力

        Returns:
            h_new: 隠れ状態
        """
        if self.h is None:
            h_new = F.tanh(self.x2h(x))
        else:
            h_new = F.tanh(self.x2h(x) + self.h2h(self.h))
        self.h = h_new
        return h_new


class BatchNorm(Layer):
    """Batch Normalization

    Attributes:
        avg_mean (Parameter): 平均の移動平均
        avg_var (Parameter): 分散の移動平均
        gamma (Parameter): スケール係数
        beta (Parameter): シフト係数
    """

    def __init__(self) -> None:
        """コンストラクタ"""
        super().__init__()
        self.avg_mean = Parameter(None, name="avg_mean")
        self.avg_var = Parameter(None, name="avg_var")
        self.gamma = Parameter(None, name="gamma")
        self.beta = Parameter(None, name="beta")

    def _init_params(self, x):
        xp = cuda.get_array_module(x)
        D = x.shape[1]
        if self.avg_mean.data is None:
            self.avg_mean.data = xp.zeros(D, dtype=x.dtype)
        if self.avg_var.data is None:
            self.avg_var.data = xp.ones(D, dtype=x.dtype)
        if self.gamma.data is None:
            self.gamma.data = xp.ones(D, dtype=x.dtype)
        if self.beta.data is None:
            self.beta.data = xp.zeros(D, dtype=x.dtype)

    def __call__(self, x: object) -> object:
        if self.avg_mean.data is None:
            self._init_params(x)
        return F.batch_norm(
            x, self.gamma, self.beta, self.avg_mean.data, self.avg_var.data
        )
