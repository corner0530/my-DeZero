import math

import numpy as np

from dezero import cuda
from dezero.datasets import Dataset


class DataLoader:
    """DataLoader

    Attributes:
        dataset (Dataset): データセット
        batch_size (int): バッチサイズ
        shuffle (bool): エポックごとにデータをシャッフルするかどうか
        data_size (int): データセットのサイズ
        max_iter (int): イテレーションの最大値
        gpu (bool): GPUを使用するかどうか
        iteration (int): 現在のイテレーション回数
        index (np.ndarray): データセットのインデックス
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool = True,
        gpu: bool = False,
    ) -> None:
        """コンストラクタ

        Args:
            dataset: データセット
            batch_size: バッチサイズ
            shuffle: エポックごとにデータをシャッフルするかどうか
            gpu: GPUを使用するかどうか
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data_size = len(dataset)
        self.max_iter = math.ceil(self.data_size / self.batch_size)
        self.gpu = gpu
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

        xp = cuda.cupy if self.gpu else np
        x = xp.array([example[0] for example in batch])
        t = xp.array([example[1] for example in batch])

        self.iteration += 1
        return x, t

    def next(self) -> tuple[np.ndarray, np.ndarray]:
        """次のミニバッチを返す

        Returns:
            x: 入力データ
            t: 教師データ
        """
        return self.__next__()

    def to_cpu(self) -> None:
        """データをCPUに移す"""
        self.gpu = False

    def to_gpu(self) -> None:
        """データをGPUに移す"""
        self.gpu = True


class SeqDataLoader(DataLoader):
    """系列データ用のDataLoader"""

    def __init__(self, dataset: Dataset, batch_size: int, gpu: bool = False) -> None:
        """コンストラクタ

        Args:
            dataset: データセット
            batch_size: バッチサイズ
            gpu: GPUを使用するかどうか
        """
        super().__init__(dataset=dataset, batch_size=batch_size, shuffle=False, gpu=gpu)

    def __next__(self) -> tuple[np.ndarray, np.ndarray]:
        """次のミニバッチを返す

        Returns:
            x: 入力データ
            t: 教師データ
        """
        if self.iteration >= self.max_iter:
            self.reset()
            raise StopIteration

        jump = self.data_size // self.batch_size
        batch_index = [
            (i * jump + self.iteration) % self.data_size for i in range(self.batch_size)
        ]
        batch = [self.dataset[i] for i in batch_index]

        xp = cuda.cupy if self.gpu else np
        x = xp.array([example[0] for example in batch])
        t = xp.array([example[1] for example in batch])

        self.iteration += 1
        return x, t
