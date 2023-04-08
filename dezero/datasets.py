import numpy as np


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
