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
