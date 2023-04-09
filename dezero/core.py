import contextlib
import weakref

import numpy as np

import dezero

try:
    import cupy

    array_types = (np.ndarray, cupy.ndarray)
except ImportError:
    array_types = np.ndarray


# =============================================================================
# Config
# =============================================================================
class Config:
    """設定を管理するクラス

    Attributes:
        enable_backprop (bool): 逆伝播を有効にするかどうか
    """

    enable_backprop = True


@contextlib.contextmanager  # コンテキストマネージャを作成するデコレータ
def using_config(name: str, value: bool) -> contextlib._GeneratorContextManager:
    """設定を変更するコンテキストマネージャ

    Args:
        name: 設定名
        value: 設定値
    """
    old_value = getattr(Config, name)
    setattr(Config, name, value)  # 前処理として設定を変更
    try:
        yield  # 例外が発生するとここにも送られる
    finally:
        setattr(Config, name, old_value)  # 後処理として設定を元に戻す


def no_grad() -> contextlib._GeneratorContextManager:
    """withブロックの中で順伝播のコードのみ実行するコンテキストマネージャ"""
    return using_config("enable_backprop", False)


# =============================================================================
# Variable / Function
# =============================================================================
class Variable:
    """変数を表すクラス

    Attributes:
        data (np.ndarray): 変数の中身
        name (str): 変数の名前
        grad (Variable): 微分した値
        creator (Function): この変数を作った関数
        generation (int): 世代
        shape (tuple): 変数の形状
        ndim (int): 変数の次元数
        size (int): 変数の要素数
        dtype (np.dtype): 変数のデータ型
        T (Variable): 転置した変数
    """

    __array_priority__ = 200  # 演算子の優先度を設定

    def __init__(self, data: np.ndarray, name: str = None) -> None:
        """初期化

        Args:
            data: 変数の中身
            name: 変数の名前
        """
        if data is not None:
            if not isinstance(data, array_types):
                # Noneでなくかつndarrayインスタンスでないときは例外を発生させる
                raise TypeError(f"{type(data)} is not supported")

        self.data = data
        self.name = name
        self.grad = None
        self.creator = None
        self.generation = 0

    @property  # これによってインスタンス変数としてアクセスできる
    def shape(self) -> tuple[int]:
        """変数の形状"""
        return self.data.shape

    @property
    def ndim(self) -> int:
        """変数の次元数"""
        return self.data.ndim

    @property
    def size(self) -> int:
        """変数の要素数"""
        return self.data.size

    @property
    def dtype(self) -> np.dtype:
        """変数のデータ型"""
        return self.data.dtype

    def __len__(self) -> int:
        """変数の長さ"""
        return len(self.data)

    def __repr__(self) -> str:
        """変数の文字列表現"""
        if self.data is None:
            return "variable(None)"
        p = str(self.data).replace("\n", "\n" + " " * 9)  # 複数行の場合は文字の開始位置を調整
        return "variable(" + p + ")"

    def set_creator(self, func: "Function") -> None:
        """この変数を作った関数を設定する

        Args:
            func: この変数を作った関数
        """
        self.creator = func
        self.generation = func.generation + 1  # 親の関数より1大きい世代に設定

    def cleargrad(self) -> None:
        """微分を初期化する"""
        self.grad = None

    def backward(self, retain_grad: bool = False, create_graph: bool = False) -> None:
        """この変数の微分を計算する

        Args:
            retain_grad: 途中の変数の微分を保持するかどうか
            create_graph: 逆伝播の計算グラフを作成するかどうか
        """
        if self.grad is None:
            xp = dezero.cuda.get_array_module(self.data)
            self.grad = Variable(xp.ones_like(self.data))  # 逆伝播の初期値が無ければ1

        funcs = []  # 処理すべき関数のリスト
        seen_set = set()  # 関数の重複を避けるための集合

        def add_func(f: "Function") -> None:
            """関数をリストに追加して世代順に並び替える

            Args:
                f: 追加する関数
            """
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)  # この変数の生みの親をリストに追加

        while funcs:
            f = funcs.pop()  # 関数を取得
            gys = [output().grad for output in f.outputs]  # 関数の出力の微分をリストにまとめる

            with using_config("enable_backprop", create_graph):  # 逆伝播の計算グラフを作成する場合
                gxs = f.backward(*gys)  # 関数のbackwardメソッドを呼ぶ
                if not isinstance(gxs, tuple):
                    gxs = (gxs,)  # タプルでない場合はタプルに変換

                for x, gx in zip(f.inputs, gxs):  # 入力に微分を設定
                    if x.grad is None:  # 逆伝播の初期値が無ければ微分を設定
                        x.grad = gx
                    else:  # あれば加算
                        x.grad = x.grad + gx

                    if x.creator is not None:
                        add_func(x.creator)  # 1つ前の関数をリストに追加

            if not retain_grad:
                for y in f.outputs:
                    y().grad = None  # yはweakref

    def reshape(self, *shape: tuple[int]) -> "Variable":
        """形状を変更する

        Args:
            *shape: 変更後の形状

        Returns:
            変更後の変数
        """
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return dezero.functions.reshape(self, shape)

    def transpose(self, *axes: tuple[int]) -> "Variable":
        """転置して軸を入れ替える

        Args:
            *axes: 軸の順番

        Returns:
            転置した変数
        """
        if len(axes) == 0:
            axes = None
        elif len(axes) == 1:
            if isinstance(axes[0], (tuple, list)) or axes[0] is None:
                axes = axes[0]
        return dezero.functions.transpose(self, axes)

    @property
    def T(self) -> "Variable":
        """転置する"""
        return dezero.functions.transpose(self)

    def sum(self, axis: int = None, keepdims: bool = False) -> "Variable":
        """和を計算する

        Args:
            axis: 和をとる軸
            keepdims: 出力の形状を入力の形状に合わせるかどうか

        Returns:
            和
        """
        return dezero.functions.sum(self, axis, keepdims)

    def to_cpu(self) -> None:
        """CPUにデータを移動する"""
        if self.data is not None:
            self.data = dezero.cuda.as_numpy(self.data)

    def to_gpu(self) -> None:
        """GPUにデータを移動する"""
        if self.data is not None:
            self.data = dezero.cuda.as_cupy(self.data)


class Parameter(Variable):
    pass


def as_variable(obj: any) -> Variable:
    """Variableインスタンスに変換する

    Args:
        obj: 変換する値

    Returns:
        変換した変数
    """
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)


def as_array(x: any, array_module: object = np) -> np.ndarray:
    """スカラー値ならばndarrayに変換する

    Args:
        x: 変換する値
        array_module: ndarrayに変換するモジュール

    Returns:
        ndarrayに変換した値
    """
    if np.isscalar(x):
        return array_module.array(x)
    return x


class Function:
    """関数の基底クラス

    全ての関数に共通する機能を実装する

    Attributes:
        inputs (list): 入力された変数のリスト
        outputs (list): 出力された変数のリスト
        generation (int): 世代
    """

    def __call__(self, *inputs: Variable) -> list[Variable] | Variable:
        """呼び出されたときの処理

        Args:
            inputs: 入力のリスト

        Returns:
            outputs: 出力のリスト
        """
        inputs = [as_variable(x) for x in inputs]  # 入力の各要素をVariableに変換

        xs = [x.data for x in inputs]  # リストの各要素からデータを取り出す
        ys = self.forward(*xs)  # 入力のリストの要素を展開
        if not isinstance(ys, tuple):
            ys = (ys,)  # タプルでない場合はタプルに変換
        outputs = [Variable(as_array(y)) for y in ys]  # ndarrayにした計算結果をVariableに変換

        if Config.enable_backprop:
            self.generation = max(
                [x.generation for x in inputs]
            )  # 入力された変数の中で最も大きい世代を設定
            for output in outputs:
                output.set_creator(self)  # 出力変数に生みの親を覚えさせる
            self.inputs = inputs  # 入力された変数を覚える
            self.outputs = [weakref.ref(output) for output in outputs]  # 出力の弱参照を持つ

        return outputs if len(outputs) > 1 else outputs[0]  # 出力が1つのときはリストではなくそのまま返す

    def forward(self, xs: list[np.ndarray]) -> tuple[np.ndarray]:
        """順伝播

        Args:
            xs: 入力のリスト

        raises:
            NotImplementedError: 未実装の場合
        """
        raise NotImplementedError()

    def backward(self, gys: list[Variable]) -> tuple[Variable]:
        """逆伝播

        Args:
            gys: 出力側から伝わる微分のリスト

        raises:
            NotImplementedError: 未実装の場合
        """
        raise NotImplementedError()


# =============================================================================
# 四則演算 / 演算子のオーバーロード
# =============================================================================
class Add(Function):
    """加算を表すクラス

    Attributes:
        x0_shape (tuple): 入力x0の形状
        x1_shape (tuple): 入力x1の形状
    """

    def forward(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        """順伝播

        Args:
            x0: 入力
            x1: 入力

        Returns:
            y: 出力
        """
        self.x0_shape = x0.shape
        self.x1_shape = x1.shape
        y = x0 + x1
        return y

    def backward(self, gy: Variable) -> tuple[Variable]:
        """逆伝播

        Args:
            gy: 出力側から伝わる微分

        Returns:
            gx0: 入力側に伝わる微分,
            gx1: 入力側に伝わる微分
        """
        gx0 = gy
        gx1 = gy
        if self.x0_shape != self.x1_shape:  # ブロードキャストする場合
            gx0 = dezero.functions.sum_to(gx0, self.x0_shape)
            gx1 = dezero.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1


def add(x0: Variable, x1: any) -> Variable:
    """加算を計算する関数

    Args:
        x0: 入力
        x1: 入力

    Returns:
        y: 出力
    """
    x1 = as_array(
        x1, dezero.cuda.get_array_module(x0.data)
    )  # x1がintやfloatの場合はndarrayに変換
    return Add()(x0, x1)


class Mul(Function):
    """乗算を表すクラス"""

    def forward(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        """順伝播

        Args:
            x0: 入力
            x1: 入力

        Returns:
            y: 出力
        """
        y = x0 * x1
        return y

    def backward(self, gy: Variable) -> tuple[Variable]:
        """逆伝播

        Args:
            gy: 出力側から伝わる微分

        Returns:
            gx0: 入力側に伝わる微分,
            gx1: 入力側に伝わる微分
        """
        x0, x1 = self.inputs
        gx0 = gy * x1
        gx1 = gy * x0
        if x0.shape != x1.shape:  # ブロードキャストする場合
            gx0 = dezero.functions.sum_to(gx0, x0.shape)
            gx1 = dezero.functions.sum_to(gx1, x1.shape)
        return gx0, gx1


def mul(x0: Variable, x1: any) -> Variable:
    """乗算を計算する関数

    Args:
        x0: 入力
        x1: 入力

    Returns:
        y: 出力
    """
    x1 = as_array(
        x1, dezero.cuda.get_array_module(x0.data)
    )  # x1がintやfloatの場合はndarrayに変換
    return Mul()(x0, x1)


class Neg(Function):
    """負数を表すクラス"""

    def forward(self, x: np.ndarray) -> np.ndarray:
        """順伝播

        Args:
            x: 入力

        Returns:
            y: 出力
        """
        y = -x
        return y

    def backward(self, gy: Variable) -> Variable:
        """逆伝播

        Args:
            gy: 出力側から伝わる微分

        Returns:
            gx: 入力側に伝わる微分
        """
        return -gy


def neg(x: Variable) -> Variable:
    """負数を計算する関数

    Args:
        x: 入力

    Returns:
        y: 出力
    """
    return Neg()(x)


class Sub(Function):
    """引き算を表すクラス

    Attributes:
        x0_shape (tuple): 入力x0の形状
        x1_shape (tuple): 入力x1の形状
    """

    def forward(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        """順伝播

        Args:
            x0: 入力
            x1: 入力

        Returns:
            y: 出力
        """
        self.x0_shape = x0.shape
        self.x1_shape = x1.shape
        y = x0 - x1
        return y

    def backward(self, gy: Variable) -> tuple[Variable]:
        """逆伝播

        Args:
            gy: 出力側から伝わる微分

        Returns:
            gx0: 入力側に伝わる微分,
            gx1: 入力側に伝わる微分
        """
        gx0 = gy
        gx1 = -gy
        if self.x0_shape != self.x1_shape:  # for broadcast
            gx0 = dezero.functions.sum_to(gx0, self.x0_shape)
            gx1 = dezero.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1


def sub(x0: Variable, x1: any) -> Variable:
    """引き算を計算する関数

    Args:
        x0: 入力
        x1: 入力

    Returns:
        y: 出力
    """
    x1 = as_array(
        x1, dezero.cuda.get_array_module(x0.data)
    )  # x1がintやfloatの場合はndarrayに変換
    return Sub()(x0, x1)


def rsub(x0: Variable, x1: any) -> Variable:
    """subのx0がVariableでない場合の引き算を計算する関数

    Args:
        x0: 入力
        x1: 入力

    Returns:
        y: 出力
    """
    x1 = as_array(
        x1, dezero.cuda.get_array_module(x0.data)
    )  # x1がintやfloatの場合はndarrayに変換
    return sub(x1, x0)


class Div(Function):
    """除算を表すクラス"""

    def forward(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        """順伝播

        Args:
            x0: 入力
            x1: 入力

        Returns:
            y: 出力
        """
        y = x0 / x1
        return y

    def backward(self, gy: Variable) -> tuple[Variable]:
        """逆伝播

        Args:
            gy: 出力側から伝わる微分

        Returns:
            gx0: 入力側に伝わる微分,
            gx1: 入力側に伝わる微分
        """
        x0, x1 = self.inputs
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1**2)
        if x0.shape != x1.shape:  # for broadcast
            gx0 = dezero.functions.sum_to(gx0, x0.shape)
            gx1 = dezero.functions.sum_to(gx1, x1.shape)
        return gx0, gx1


def div(x0: Variable, x1: any) -> Variable:
    """除算を計算する関数

    Args:
        x0: 入力
        x1: 入力

    Returns:
        y: 出力
    """
    x1 = as_array(
        x1, dezero.cuda.get_array_module(x0.data)
    )  # x1がintやfloatの場合はndarrayに変換
    return Div()(x0, x1)


def rdiv(x0: Variable, x1: any) -> Variable:
    """divのx0がVariableでない場合の除算を計算する関数

    Args:
        x0: 入力
        x1: 入力

    Returns:
        y: 出力
    """
    x1 = as_array(
        x1, dezero.cuda.get_array_module(x0.data)
    )  # x1がintやfloatの場合はndarrayに変換
    return Div()(x1, x0)


class Pow(Function):
    """べき乗を表すクラス"""

    def __init__(self, c: int | float) -> None:
        """コンストラクタ

        Args:
            c: べき指数
        """
        self.c = c

    def forward(self, x: np.ndarray) -> np.ndarray:
        """順伝播

        Args:
            x: 入力

        Returns:
            y: 出力
        """
        y = x**self.c
        return y

    def backward(self, gy: Variable) -> Variable:
        """逆伝播

        Args:
            gy: 出力側から伝わる微分

        Returns:
            gx: 入力側に伝わる微分
        """
        x = self.inputs[0]
        c = self.c
        gx = c * x ** (c - 1) * gy
        return gx


def pow(x: Variable, c: int | float) -> Variable:
    """べき乗を計算する関数

    Args:
        x: 入力
        c: べき指数

    Returns:
        y: 出力
    """
    return Pow(c)(x)  # cを初期化時に与える


def setup_variable() -> None:
    """Variableの演算子のオーバーロードを行う関数"""
    Variable.__add__ = add  # +演算子をオーバーロード
    Variable.__radd__ = add  # 右側がVariableの場合のオーバーロード
    Variable.__mul__ = mul  # *演算子をオーバーロード
    Variable.__rmul__ = mul  # 右側がVariableの場合のオーバーロード
    Variable.__neg__ = neg  # 符号を表す-演算子をオーバーロード
    Variable.__sub__ = sub  # -演算子をオーバーロード
    Variable.__rsub__ = rsub  # 右側がVariableの場合のオーバーロード
    Variable.__truediv__ = div  # /演算子をオーバーロード
    Variable.__rtruediv__ = rdiv  # 右側がVariableの場合のオーバーロード
    Variable.__pow__ = pow  # **演算子をオーバーロード
    Variable.__getitem__ = dezero.functions.get_item

    Variable.matmul = dezero.functions.matmul
    Variable.dot = dezero.functions.matmul
