import gzip
import os
import pickle
import tarfile

import matplotlib.pyplot as plt
import numpy as np

from dezero.transforms import Compose, Flatten, Normalize, ToFloat
from dezero.utils import cache_dir, get_file


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
        assert self.data is not None
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
        assert self.data is not None
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


# =============================================================================
# MNIST-like dataset: MNIST / CIFAR /
# =============================================================================
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
        assert self.data is not None
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


class CIFAR10(Dataset):
    """CIFAR10データセット

    Attributes:
        data (np.ndarray): 入力データ
        label (np.ndarray): ラベル
    """

    def __init__(
        self,
        train: bool = True,
        transform: object = Compose([ToFloat(), Normalize(mean=0.5, std=0.5)]),
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
        url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        self.data, self.label = load_cache_npz(url, self.train)
        if self.data is not None:
            return
        filepath = get_file(url)
        if self.train:
            self.data = np.empty((50000, 3 * 32 * 32))
            self.label = np.empty((50000), dtype=np.int32)
            for i in range(5):
                self.data[i * 10000 : (i + 1) * 10000] = self._load_data(
                    filepath, i + 1, "train"
                )
                self.label[i * 10000 : (i + 1) * 10000] = self._load_label(
                    filepath, i + 1, "train"
                )
        else:
            self.data = self._load_data(filepath, 5, "test")
            self.label = self._load_label(filepath, 5, "test")
        self.data = self.data.reshape(-1, 3, 32, 32)
        save_cache_npz(self.data, self.label, url, self.train)

    def _load_data(
        self, filename: str, idx: int, data_type: str = "train"
    ) -> np.ndarray:
        """データの読み込み

        Args:
            filename: ファイル名
            idx: ファイル番号
            data_type: データタイプ

        Returns:
            データ
        """
        assert data_type in ["train", "test"]
        with tarfile.open(filename, "r:gz") as file:
            for item in file.getmembers():
                if (
                    "data_batch_{}".format(idx) in item.name and data_type == "train"
                ) or ("test_batch" in item.name and data_type == "test"):
                    data_dict = pickle.load(file.extractfile(item), encoding="bytes")
                    data = data_dict[b"data"]
                    return data

    def _load_label(
        self, filename: str, idx: int, data_type: str = "train"
    ) -> np.ndarray:
        """ラベルの読み込み

        Args:
            filename: ファイル名
            idx: ファイル番号
            data_type: データタイプ

        Returns:
            ラベル
        """
        assert data_type in ["train", "test"]
        with tarfile.open(filename, "r:gz") as file:
            for item in file.getmembers():
                if (
                    "data_batch_{}".format(idx) in item.name and data_type == "train"
                ) or ("test_batch" in item.name and data_type == "test"):
                    data_dict = pickle.load(file.extractfile(item), encoding="bytes")
                    return np.array(data_dict[b"labels"])

    def show(self, row: int = 10, col: int = 10) -> None:
        """データの表示

        Args:
            row: 行数
            col: 列数
        """
        H, W = 32, 32
        img = np.zeros((H * row, W * col, 3))
        for r in range(row):
            for c in range(col):
                img[r * H : (r + 1) * H, c * W : (c + 1) * W] = (
                    self.data[np.random.randint(0, len(self.data) - 1)]
                    .reshape(3, H, W)
                    .transpose(1, 2, 0)
                    / 255
                )
        plt.imshow(img, interpolation="nearest")
        plt.axis("off")
        plt.show()

    @staticmethod
    def labels() -> dict[int, str]:
        """ラベルの辞書を返す

        Returns:
            ラベルの辞書
        """
        return {
            0: "ariplane",
            1: "automobile",
            2: "bird",
            3: "cat",
            4: "deer",
            5: "dog",
            6: "frog",
            7: "horse",
            8: "ship",
            9: "truck",
        }


class CIFAR100(CIFAR10):
    """CIFAR100データセット

    Attributes:
        data: 入力データ
        label: ラベル
    """

    def __init__(
        self,
        train: bool = True,
        transform: object = Compose([ToFloat(), Normalize(mean=0.5, std=0.5)]),
        target_transform: object = None,
        label_type: str = "fine",
    ) -> None:
        """コンストラクタ

        Args:
            train: 学習データかどうか
            transform: 1つの入力データに対する変換処理
            target_transform: 1つのラベルに対する変換処理
            label_type: ラベルの種類
        """
        assert label_type in ["fine", "coarse"]
        self.label_type = label_type
        super().__init__(train, transform, target_transform)

    def prepare(self) -> None:
        """データの準備"""
        url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
        self.data, self.label = load_cache_npz(url, self.train)
        if self.data is not None:
            return

        filepath = get_file(url)
        if self.train:
            self.data = self._load_data(filepath, "train")
            self.label = self._load_label(filepath, "train")
        else:
            self.data = self._load_data(filepath, "test")
            self.label = self._load_label(filepath, "test")
        self.data = self.data.reshape(-1, 3, 32, 32)
        save_cache_npz(self.data, self.label, url, self.train)

    def _load_data(self, filename: str, data_type: str = "train") -> np.ndarray:
        """データの読み込み

        Args:
            filename: ファイル名
            data_type: データタイプ

        Returns:
            データ
        """
        with tarfile.open(filename, "r:gz") as file:
            for item in file.getmembers():
                if data_type in item.name:
                    data_dict = pickle.load(file.extractfile(item), encoding="bytes")
                    data = data_dict[b"data"]
                    return data

    def _load_label(self, filename: str, data_type: str = "train") -> np.ndarray:
        """ラベルの読み込み

        Args:
            filename: ファイル名
            data_type: データタイプ

        Returns:
            ラベル
        """
        assert data_type in ["train", "test"]
        with tarfile.open(filename, "r:gz") as file:
            for item in file.getmembers():
                if data_type in item.name:
                    data_dict = pickle.load(file.extractfile(item), encoding="bytes")
                    if self.label_type == "fine":
                        return np.array(data_dict[b"fine_labels"])
                    elif self.label_type == "coarse":
                        return np.array(data_dict[b"coarse_labels"])

    @staticmethod
    def labels(label_type: str = "fine") -> dict[int, str]:
        """ラベルの辞書を返す

        Args:
            label_type: ラベルの種類

        Returns:
            ラベルの辞書
        """
        coarse_labels = dict(
            enumerate(
                [
                    "aquatic mammals",
                    "fish",
                    "flowers",
                    "food containers",
                    "fruit and vegetables",
                    "household electrical device",
                    "household furniture",
                    "insects",
                    "large carnivores",
                    "large man-made outdoor things",
                    "large natural outdoor scenes",
                    "large omnivores and herbivores",
                    "medium-sized mammals",
                    "non-insect invertebrates",
                    "people",
                    "reptiles",
                    "small mammals",
                    "trees",
                    "vehicles 1",
                    "vehicles 2",
                ]
            )
        )
        fine_labels = []
        return fine_labels if label_type == "fine" else coarse_labels


# =============================================================================
# Big datasets
# =============================================================================
class ImageNet(Dataset):
    """ImageNetデータセット"""

    def __init__(self) -> None:
        """コンストラクタ"""
        raise NotImplementedError

    @staticmethod
    def labels() -> dict[int, str]:
        """ラベルの辞書を返す

        Returns:
            ラベルの辞書
        """
        url = (
            "https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/"
            "238f720ff059c1f82f368259d1ca4ffa5dd8f9f5/imagenet1000_clsidx_to_labels.txt"
        )
        path = get_file(url)
        with open(path, "r") as f:
            labels = eval(f.read())
        return labels


class SinCurve(Dataset):
    """sinカーブのデータセット

    Attributes:
        data: 入力データ
        label: ラベル
    """

    def prepare(self) -> None:
        """データの準備"""
        num_data = 1000
        dtype = np.float64

        x = np.linspace(0, 2 * np.pi, num_data)
        noise_range = (-0.05, 0.05)
        noise = np.random.uniform(noise_range[0], noise_range[1], size=x.shape)
        if self.train:
            y = np.sin(x) + noise
        else:
            y = np.cos(x)
        y = y.astype(dtype)
        self.data = y[:-1][:, np.newaxis]
        self.label = y[1:][:, np.newaxis]


# =============================================================================
# Utils
# =============================================================================
def load_cache_npz(filename: str, train: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """キャッシュからデータを読み込む

    Args:
        filename: ファイル名
        train: 学習データかどうか

    Returns:
        データ,ラベル
    """
    filename = filename[filename.rfind("/") + 1 :]
    prefix = ".train.npz" if train else ".test.npz"
    filepath = os.path.join(cache_dir, filename + prefix)
    if not os.path.exists(filepath):
        return None, None

    loaded = np.load(filepath)
    return loaded["data"], loaded["label"]


def save_cache_npz(
    data: np.ndarray, label: np.ndarray, filename: str, train: bool = False
) -> str:
    """キャッシュにデータを保存する

    Args:
        data: データ
        label: ラベル
        filename: ファイル名
        train: 学習データかどうか

    Returns:
        ファイルパス
    """
    filename = filename[filename.rfind("/") + 1 :]
    prefix = ".train.npz" if train else ".test.npz"
    filepath = os.path.join(cache_dir, filename + prefix)

    if os.path.exists(filepath):
        return

    print("Saving: " + filename + prefix)
    try:
        np.savez_compressed(filepath, data=data, label=label)
    except (Exception, KeyboardInterrupt):
        if os.path.exists(filepath):
            os.remove(filepath)
        raise
    print(" Done")
    return filepath
