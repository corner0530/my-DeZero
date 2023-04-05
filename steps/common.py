import contextlib
import weakref

import numpy as np


class Config:
    """設定を管理するクラス

    Attributes:
        enable_backprop (bool): 逆伝播を有効にするかどうか
    """

    enable_backprop = True


class Variable:
    """変数を表すクラス

    Attributes:
        data (np.ndarray): 変数の中身
        name (str): 変数の名前
        grad (np.ndarray): 微分した値
        creator (Function): この変数を作った関数
        generation (int): 世代
        shape (tuple): 変数の形状
        ndim (int): 変数の次元数
        size (int): 変数の要素数
        dtype (np.dtype): 変数のデータ型
    """

    def __init__(self, data: np.ndarray, name: str = None) -> None:
        """初期化

        Args:
            data: 変数の中身
            name: 変数の名前
        """
        if data is not None:
            if not isinstance(data, np.ndarray):
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

    def backward(self, retain_grad: bool = False) -> None:
        """この変数の微分を計算する

        Args:
            retain_grad: 途中の変数の微分を保持するかどうか
        """
        if self.grad is None:
            self.grad = np.ones_like(self.data)  # 逆伝播の初期値が無ければ1

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


def as_array(x: any) -> np.ndarray:
    """スカラー値ならばndarrayに変換する

    Args:
        x: 変換する値

    Returns:
        ndarrayに変換した値
    """
    if np.isscalar(x):
        return np.array(x)
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

    def backward(self, gys: list[np.ndarray]) -> tuple[np.ndarray]:
        """逆伝播

        Args:
            gys: 出力側から伝わる微分のリスト

        raises:
            NotImplementedError: 未実装の場合
        """
        raise NotImplementedError()


class Square(Function):
    """二乗を表すクラス"""

    def forward(self, x: np.ndarray) -> np.ndarray:
        """順伝播

        Args:
            x: 入力

        Returns:
            y: 出力
        """
        y = x**2
        return y  # 具体的な計算を実装

    def backward(self, gy: np.ndarray) -> np.ndarray:
        """逆伝播

        Args:
            gy: 出力側から伝わる微分

        Returns:
            gx: 入力側に伝わる微分
        """
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx


def square(x: Variable) -> Variable:
    """二乗を計算する関数

    Args:
        x: 入力

    Returns:
        y: 出力
    """
    return Square()(x)


class Exp(Function):
    """指数関数を表すクラス"""

    def forward(self, x: np.ndarray) -> np.ndarray:
        """順伝播

        Args:
            x: 入力

        Returns:
            y: 出力
        """
        y = np.exp(x)
        return y

    def backward(self, gy: np.ndarray) -> np.ndarray:
        """逆伝播

        Args:
            gy: 出力側から伝わる微分

        Returns:
            gx: 入力側に伝わる微分
        """
        x = self.inputs[0].data
        gx = np.exp(x) * gy
        return gx


def exp(x: Variable) -> Variable:
    """指数関数を計算する関数

    Args:
        x: 入力

    Returns:
        y: 出力
    """
    return Exp()(x)


class Add(Function):
    """加算を表すクラス"""

    def forward(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        """順伝播

        Args:
            x0: 入力
            x1: 入力

        Returns:
            y: 出力
        """
        y = x0 + x1
        return y

    def backward(self, gy: np.ndarray) -> tuple[np.ndarray]:
        """逆伝播

        Args:
            gy: 出力側から伝わる微分

        Returns:
            gx0: 入力側に伝わる微分,
            gx1: 入力側に伝わる微分
        """
        return gy, gy


def add(x0: Variable, x1: Variable) -> Variable:
    """加算を計算する関数

    Args:
        x0: 入力
        x1: 入力

    Returns:
        y: 出力
    """
    return Add()(x0, x1)


Variable.__add__ = add  # +演算子をオーバーロード


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

    def backward(self, gy: np.ndarray) -> tuple[np.ndarray]:
        """逆伝播

        Args:
            gy: 出力側から伝わる微分

        Returns:
            gx0: 入力側に伝わる微分,
            gx1: 入力側に伝わる微分
        """
        x0 = self.inputs[0].data
        x1 = self.inputs[1].data
        return gy * x1, gy * x0


def mul(x0: Variable, x1: Variable) -> Variable:
    """乗算を計算する関数

    Args:
        x0: 入力
        x1: 入力

    Returns:
        y: 出力
    """
    return Mul()(x0, x1)


Variable.__mul__ = mul  # *演算子をオーバーロード


def numerical_diff(f: Function, x: Variable, eps: float = 1e-4) -> float:
    """数値微分を行う関数

    Args:
        f: 対象の関数
        x: 対象の変数
        eps: 微小な値

    Returns:
        中心差分近似における微分値
    """
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)


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
