import gzip

import matplotlib.pyplot as plt
import numpy as np

from dezero.transforms import Compose, Flatten, Normalize, ToFloat
from dezero.utils import get_file


class Dataset:
    """データセットの基底クラス

    Attributes:
        train (bool): 学習データかどうか
        transform (object): 1つの入力データに対する変換処理
        target_transform (object): 1つのラベルに対する変換処理
        data (np.ndarray): 入力データ
        label (np.ndarray): ラベル
    """

    def __init__(
        self,
        train: bool = True,
        transform: object = None,
        target_transform: object = None,
    ) -> None:
        """コンストラクタ

        Args:
            train: 学習データかどうか
            transform: 1つの入力データに対する変換処理
            target_transform: 1つのラベルに対する変換処理
        """
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        if self.transform is None:
            self.transform = lambda x: x
        if self.target_transform is None:
            self.target_transform = lambda x: x

        self.data = None
        self.label = None
        self.prepare()

    def __getitem__(self, index: int) -> tuple[object, object]:
        """指定されたインデックスのデータを取り出す

        Args:
            index: インデックス

        Returns:
            data: 入力データ
            label: ラベル(ラベルがない場合はNone)
        """
        assert np.isscalar(index)
        if self.label is None:
            return self.transform(self.data[index]), None
        else:
            return self.transform(self.data[index]), self.target_transform(
                self.label[index]
            )

    def __len__(self) -> int:
        """データセットの長さを返す

        Returns:
            データセットの長さ
        """
        return len(self.data)

    def prepare(self) -> None:
        """データの準備

        データの準備は継承先のクラスで実装する
        """
        pass


# =============================================================================
# Toy datasets
# =============================================================================
def get_spiral(train: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Spiralデータセットの取得

    Args:
        train: 学習データかどうか

    Returns:
        x: データ
        t: ラベル
    """
    seed = 1984 if train else 2020
    np.random.seed(seed=seed)

    num_data, num_class, input_dim = 100, 3, 2
    data_size = num_data * num_class
    x = np.zeros((data_size, input_dim), dtype=np.float32)
    t = np.zeros(data_size, dtype=np.int32)

    for j in range(num_class):
        for i in range(num_data):
            rate = i / num_data
            radius = 1.0 * rate
            theta = j * 4.0 + 4.0 * rate + np.random.randn() * 0.2
            ix = num_data * j + i
            x[ix] = np.array([radius * np.sin(theta), radius * np.cos(theta)]).flatten()
            t[ix] = j

    # 入れ替え
    indices = np.random.permutation(num_data * num_class)
    x = x[indices]
    t = t[indices]
    return x, t


class Spiral(Dataset):
    """Spiralデータセット"""

    def prepare(self) -> None:
        """データの準備"""
        self.data, self.label = get_spiral(self.train)


class MNIST(Dataset):
    """MNISTデータセット

    Attributes:
        data (np.ndarray): 入力データ
        label (np.ndarray): ラベル
    """

    def __init__(
        self,
        train: bool = True,
        transform: object = Compose([Flatten(), ToFloat(), Normalize(0.0, 255.0)]),
        target_transform: object = None,
    ) -> None:
        """コンストラクタ

        Args:
            train: 学習データかどうか
            transform: 1つの入力データに対する変換処理
            target_transform: 1つのラベルに対する変換処理
        """
        super().__init__(train, transform, target_transform)

    def prepare(self) -> None:
        """データの準備"""
        url = "http://yann.lecun.com/exdb/mnist/"
        train_files = {
            "target": "train-images-idx3-ubyte.gz",
            "label": "train-labels-idx1-ubyte.gz",
        }
        test_files = {
            "target": "t10k-images-idx3-ubyte.gz",
            "label": "t10k-labels-idx1-ubyte.gz",
        }

        files = train_files if self.train else test_files
        data_path = get_file(url + files["target"])
        label_path = get_file(url + files["label"])

        self.data = self._load_data(data_path)
        self.label = self._load_label(label_path)

    def _load_label(self, filepath: str) -> np.ndarray:
        """ラベルの読み込み

        Args:
            filepath: ファイルパス

        Returns:
            ラベル
        """
        with gzip.open(filepath, "rb") as f:
            labels = np.frombuffer(f.read(), np.uint8, offset=8)
        return labels

    def _load_data(self, filepath: str) -> np.ndarray:
        """データの読み込み

        Args:
            filepath: ファイルパス

        Returns:
            データ
        """
        with gzip.open(filepath, "rb") as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1, 1, 28, 28)
        return data

    def show(self, row: int = 10, col: int = 10) -> None:
        """データの表示

        Args:
            row: 行数
            col: 列数
        """
        H, W = 28, 28
        img = np.zeros((H * row, W * col))
        for r in range(row):
            for c in range(col):
                img[r * H : (r + 1) * H, c * W : (c + 1) * W] = self.data[
                    np.random.randint(0, len(self.data) - 1)
                ].reshape(H, W)
        plt.imshow(img, cmap="gray", interpolation="nearest")
        plt.axis("off")
        plt.show()

    @staticmethod
    def labels() -> dict[int, str]:
        """ラベルの取得

        Returns:
            ラベル
        """
        return {
            0: "0",
            1: "1",
            2: "2",
            3: "3",
            4: "4",
            5: "5",
            6: "6",
            7: "7",
            8: "8",
            9: "9",
        }
