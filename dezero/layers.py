import os
import weakref

import numpy as np

import dezero.functions as F
from dezero import cuda
from dezero.core import Parameter


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
            key = parent_key + '/' + name if parent_key else name

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

        self.w = Parameter(None, name="w")
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
        self.w.data = w_data

    def forward(self, x: object) -> object:
        """順伝播

        Args:
            x: 入力

        Returns:
            y: 出力
        """
        # データを流すタイミングで重みを初期化
        if self.w.data is None:
            self.in_size = x.shape[1]
            xp = cuda.get_array_module(x)
            self._init_w(xp)

        y = F.linear(x, self.w, self.b)
        return y
