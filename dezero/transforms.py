import numpy as np

try:
    import Image
except ImportError:
    from PIL import Image

from dezero.utils import pair


class Compose:
    """与えられたリストを先頭から順に処理する

    Attributes:
        transforms (list): 処理のリスト
    """

    def __init__(self, transforms: list = []) -> None:
        """コンストラクタ

        Args:
            transforms (list): 処理のリスト
        """
        self.transforms = transforms

    def __call__(self, img: Image.Image) -> Image.Image:
        """処理の実行

        Args:
            img: 画像

        Returns:
            img: 処理後の画像
        """
        if not self.transforms:
            return img
        for t in self.transforms:
            img = t(img)
        return img


# =============================================================================
# PIL Imageの変換
# =============================================================================
class Convert:
    """画像を指定したモードに変換

    Attributes:
        mode (str): 変換するモード
    """

    def __init__(self, mode: str = "RGB") -> None:
        """コンストラクタ

        Args:
            mode: 変換するモード
        """
        self.mode = mode

    def __call__(self, img: Image.Image) -> Image.Image:
        """変換の実行

        Args:
            img: 画像

        Returns:
            img: 変換後の画像
        """
        if self.mode == "BGR":
            img = img.convert("RGB")
            r, g, b = img.split()
            img = Image.merge("RGB", (b, g, r))
            return img
        else:
            return img.convert(self.mode)


class Resize:
    """入力の画像を指定した形状に変換

    Attributes:
        size (int | tuple[int, int]): 変換後の形状
        mode (int): 補完方法
    """

    def __init__(self, size: int | tuple[int, int], mode: int = Image.BILINEAR) -> None:
        """コンストラクタ

        Args:
            size: 変換後の形状
            mode: 補完方法
        """
        self.size = pair(size)
        self.mode = mode

    def __call__(self, img: Image.Image) -> Image.Image:
        """変換の実行

        Args:
            img: 画像

        Returns:
            img: 変換後の画像
        """
        return img.resize(self.size, self.mode)


class CenterCrop:
    """入力の画像を中央を中心に指定した形状に変換

    Attributes:
        size (int | tuple[int, int]): 変換後の形状
    """

    def __init__(self, size: int | tuple[int, int]) -> None:
        """コンストラクタ

        Args:
            size: 変換後の形状
        """
        self.size = pair(size)

    def __call__(self, img: Image.Image) -> Image.Image:
        """変換の実行

        Args:
            img: 画像

        Returns:
            img: 変換後の画像
        """
        W, H = img.size
        OW, OH = self.size
        left = (W - OW) // 2
        right = W - ((W - OW) // 2 + (W - OW) % 2)
        up = (H - OH) // 2
        bottom = H - ((H - OH) // 2 + (H - OH) % 2)
        return img.crop((left, up, right, bottom))


class ToArray:
    """PIL Image を NumPy array に変換

    Attributes:
        dtype (np.dtype): 変換後のデータ型
    """

    def __init__(self, dtype: type = np.float32) -> None:
        """コンストラクタ

        Args:
            dtype: 変換後のデータ型
        """
        self.dtype = dtype

    def __call__(self, img: np.ndarray | Image.Image) -> np.ndarray:
        """変換の実行

        Args:
            img: 画像

        Returns:
            img: 変換後の画像

        Raises:
            TypeError: 画像の型が不正な場合
        """
        if isinstance(img, np.ndarray):
            return img
        if isinstance(img, Image.Image):
            img = np.asarray(img)
            img = img.transpose(2, 0, 1)
            img = img.astype(self.dtype)
            return img
        else:
            raise TypeError


class ToPIL:
    """NumPy array を PIL Image に変換"""

    def __call__(self, array: np.ndarray) -> Image.Image:
        """変換の実行

        Args:
            array: 画像

        Returns:
            img: 変換後の画像
        """
        data = array.transpose(1, 2, 0)
        return Image.fromarray(data)


class RandomHorizontalFlip:
    pass


# =============================================================================
# NumPy arrayの変換
# =============================================================================
class Normalize:
    """NumPy array を正規化

    Attributes:
        mean (float | list[float]): 平均値
        std (float | list[float]): 標準偏差
    """

    def __init__(self, mean: float = 0, std: float = 1) -> None:
        self.mean = mean
        self.std = std

    def __call__(self, array: np.ndarray) -> np.ndarray:
        """変換の実行

        Args:
            array: 配列

        Returns:
            array: 変換後の配列
        """
        mean, std = self.mean, self.std

        if not np.isscalar(mean):
            mshape = [1] * array.ndim
            mshape[0] = len(array) if len(self.mean) == 1 else len(self.mean)
            mean = np.array(self.mean, dtype=array.dtype).reshape(*mshape)
        if not np.isscalar(std):
            rshape = [1] * array.ndim
            rshape[0] = len(array) if len(self.std) == 1 else len(self.std)
            std = np.array(self.std, dtype=array.dtype).reshape(*rshape)
        return (array - mean) / std


class Flatten:
    """NumPy array を平坦化"""

    def __call__(self, array: np.ndarray) -> np.ndarray:
        """変換の実行

        Args:
            array: 配列

        Returns:
            array: 変換後の配列
        """
        return array.flatten()


class AsType:
    """NumPy array のデータ型を変換

    Attributes:
        dtype (np.dtype): 変換後のデータ型
    """

    def __init__(self, dtype: type = np.float32) -> None:
        """コンストラクタ

        Args:
            dtype: 変換後のデータ型
        """
        self.dtype = dtype

    def __call__(self, array: np.ndarray) -> np.ndarray:
        """変換の実行

        Args:
            array: 配列

        Returns:
            array: 変換後の配列
        """
        return array.astype(self.dtype)


ToFloat = AsType


class ToInt(AsType):
    """NumPy array のデータ型を int に変換

    Attributes:
        dtype (np.dtype): 変換後のデータ型
    """

    def __init__(self, dtype: type = np.int32) -> None:
        """コンストラクタ

        Args:
            dtype: 変換後のデータ型
        """
        self.dtype = dtype
