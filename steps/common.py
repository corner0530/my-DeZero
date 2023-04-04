import numpy as np


class Variable:
    """変数を表すクラス

    Attributes:
        data (np.ndarray): 変数の中身
    """

    def __init__(self, data: np.ndarray) -> None:
        """初期化

        Args:
            data: 変数の中身
        """
        self.data = data


class Function:
    """関数の基底クラス

    全ての関数に共通する機能を実装する
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
        output = Variable(y)  # 計算結果をVariableに変換
        return output

    def forward(self, x: np.ndarray) -> np.ndarray:
        """順伝播

        Args:
            x: 入力

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
