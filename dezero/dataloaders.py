import math

import numpy as np

from dezero import Dataset


class DataLoader:
    def __init__(self, dataset: Dataset, batch_size: int, shuffle: bool = True) -> None:
        """コンストラクタ

        Args:
            dataset: データセット
            batch_size: バッチサイズ
            shuffle: エポックごとにデータをシャッフルするかどうか
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data_size = len(dataset)
        self.max_iter = math.ceil(self.data_size / self.batch_size)
        self.reset()

    def reset(self) -> None:
        """イテレータを初期化して必要に応じてデータをシャッフルする"""
        self.iteration = 0
        if self.shuffle:
            self.index = np.random.permutation(len(self.dataset))
        else:
            self.index = np.arange(len(self.dataset))

    def __iter__(self) -> "DataLoader":
        """イテレータを返す"""
        return self

    def __next__(self) -> tuple[np.ndarray, np.ndarray]:
        """次のミニバッチを返す

        Returns:
            x: 入力データ
            t: 教師データ

        Raises:
            StopIteration: イテレーション回数が最大値を超えた場合
        """
        if self.iteration >= self.max_iter:
            self.reset()
            raise StopIteration

        i, batch_size = self.iteration, self.batch_size
        batch_index = self.index[i * batch_size : (i + 1) * batch_size]
        batch = [self.dataset[i] for i in batch_index]
        x = np.array([example[0] for example in batch])
        t = np.array([example[1] for example in batch])

        self.iteration += 1
        return x, t

    def next(self) -> tuple[np.ndarray, np.ndarray]:
        """次のミニバッチを返す

        Returns:
            x: 入力データ
            t: 教師データ
        """
        return self.__next__()
