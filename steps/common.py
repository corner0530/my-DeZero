import numpy as np


class Variable:
    """変数を表すクラス

    Attributes:
        data (np.ndarray): 変数の中身
        grad (np.ndarray): 微分した値
        creator (Function): この変数を作った関数
    """

    def __init__(self, data: np.ndarray) -> None:
        """初期化

        Args:
            data: 変数の中身
        """
        if data is not None:
            if not isinstance(data, np.ndarray):
                # Noneでなくかつndarrayインスタンスでないときは例外を発生させる
                raise TypeError(f"{type(data)} is not supported")

        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func: "Function") -> None:
        """この変数を作った関数を設定する

        Args:
            func: この変数を作った関数
        """
        self.creator = func

    def backward(self) -> None:
        """この変数の微分を計算する"""
        if self.grad is None:
            self.grad = np.ones_like(self.data)  # 逆伝播の初期値が無ければ1

        funcs = [self.creator]  # 処理すべき関数のリスト
        while funcs:
            f = funcs.pop()  # 関数を取得
            x = f.input  # 関数の入力を取得
            y = f.output  # 関数の出力を取得
            x.grad = f.backward(y.grad)  # 関数のbackwardメソッドを呼ぶ

            if x.creator is not None:
                funcs.append(x.creator)  # 1つ前の関数をリストに追加


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
        input (Variable): 入力された変数
        output (Variable): 出力された変数
    """

    def __call__(self, input: Variable) -> Variable:
        """呼び出されたときの処理

        Args:
            input: 入力

        Returns:
            output: 出力
        """
        x = input.data  # データを取り出す
        y = self.forward(x)  # 入力を受け取って計算
        output = Variable(as_array(y))  # ndarrayにした計算結果をVariableに変換
        output.set_creator(self)  # 出力変数に生みの親を覚えさせる
        self.input = input  # 入力された変数を覚える
        self.output = output  # 出力も覚える
        return output

    def forward(self, x: np.ndarray) -> np.ndarray:
        """順伝播

        Args:
            x: 入力

        raises:
            NotImplementedError: 未実装の場合
        """
        raise NotImplementedError()

    def backward(self, gy: np.ndarray) -> np.ndarray:
        """逆伝播

        Args:
            gy: 出力側から伝わる微分

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
        x = self.input.data
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
        x = self.input.data
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
