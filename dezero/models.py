import numpy as np
from PIL import Image

import dezero.functions as F
import dezero.layers as L
from dezero import Layer, Variable, utils


class Model(Layer):
    """モデルの基底クラス"""

    def plot(self, *inputs: object, to_file: str = "model.png") -> None:
        """モデルの構造を可視化する

        Args:
            inputs: 入力
            to_file: 出力ファイル名
        """
        y = self.forward(*inputs)
        return utils.plot_dot_graph(y, verbose=True, to_file=to_file)


class MLP(Model):
    """多層パーセプトロン

    Attributes:
        layers (list): 全結合層のリスト
        activation (object): 活性化関数
    """

    def __init__(
        self, fc_output_sizes: tuple[int] | list[int], activation: object = F.sigmoid
    ) -> None:
        """コンストラクタ

        Args:
            fc_output_sizes: 各全結合層の出力サイズ
            activation: 活性化関数
        """
        super().__init__()
        self.activation = activation
        self.layers = []

        for i, out_size in enumerate(fc_output_sizes):
            layer = L.Linear(out_size)
            setattr(self, "l" + str(i), layer)
            self.layers.append(layer)

    def forward(self, x: object) -> object:
        """順伝播

        Args:
            x: 入力

        Returns:
            y: 出力
        """
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.layers[-1](x)


# =============================================================================
# VGG
# =============================================================================
class VGG16(Model):
    """VGG16モデル

    Attributes:
        conv1_1 (Conv2d): 畳み込み層
        conv1_2 (Conv2d): 畳み込み層
        conv2_1 (Conv2d): 畳み込み層
        conv2_2 (Conv2d): 畳み込み層
        conv3_1 (Conv2d): 畳み込み層
        conv3_2 (Conv2d): 畳み込み層
        conv3_3 (Conv2d): 畳み込み層
        conv4_1 (Conv2d): 畳み込み層
        conv4_2 (Conv2d): 畳み込み層
        conv4_3 (Conv2d): 畳み込み層
        conv5_1 (Conv2d): 畳み込み層
        conv5_2 (Conv2d): 畳み込み層
        conv5_3 (Conv2d): 畳み込み層
        fc6 (Linear): 全結合層
        fc7 (Linear): 全結合層
        fc8 (Linear): 全結合層
    """

    WEIGHTS_PATH = (
        "https://github.com/koki0702/dezero-models/releases/download/v0.1/vgg16.npz"
    )

    def __init__(self, pretrained: bool = False) -> None:
        """コンストラクタ

        Args:
            pretrained: 学習済みパラメータを使用するかどうか
        """
        super().__init__()
        # 出力のチャンネル数だけ指定
        self.conv1_1 = L.Conv2d(64, kernel_size=3, stride=1, pad=1)
        self.conv1_2 = L.Conv2d(64, kernel_size=3, stride=1, pad=1)
        self.conv2_1 = L.Conv2d(128, kernel_size=3, stride=1, pad=1)
        self.conv2_2 = L.Conv2d(128, kernel_size=3, stride=1, pad=1)
        self.conv3_1 = L.Conv2d(256, kernel_size=3, stride=1, pad=1)
        self.conv3_2 = L.Conv2d(256, kernel_size=3, stride=1, pad=1)
        self.conv3_3 = L.Conv2d(256, kernel_size=3, stride=1, pad=1)
        self.conv4_1 = L.Conv2d(512, kernel_size=3, stride=1, pad=1)
        self.conv4_2 = L.Conv2d(512, kernel_size=3, stride=1, pad=1)
        self.conv4_3 = L.Conv2d(512, kernel_size=3, stride=1, pad=1)
        self.conv5_1 = L.Conv2d(512, kernel_size=3, stride=1, pad=1)
        self.conv5_2 = L.Conv2d(512, kernel_size=3, stride=1, pad=1)
        self.conv5_3 = L.Conv2d(512, kernel_size=3, stride=1, pad=1)
        self.fc6 = L.Linear(4096)
        self.fc7 = L.Linear(4096)
        self.fc8 = L.Linear(1000)

        if pretrained:
            weights_path = utils.get_file(VGG16.WEIGHTS_PATH)
            self.load_weights(weights_path)

    def forward(self, x: Variable) -> Variable:
        """順伝播

        Args:
            x: 入力

        Returns:
            y: 出力
        """
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = F.pooling(x, 2, 2)
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = F.pooling(x, 2, 2)
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = F.pooling(x, 2, 2)
        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        x = F.pooling(x, 2, 2)
        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.conv5_3(x))
        x = F.pooling(x, 2, 2)
        x = F.reshape(x, (x.shape[0], -1))  # 整形
        x = F.dropout(F.relu(self.fc6(x)))
        x = F.dropout(F.relu(self.fc7(x)))
        x = self.fc8(x)
        return x

    @staticmethod
    def preprocess(
        image: Image.Image, size: tuple[int] = (224, 224), dtype: type = np.float32
    ):
        """画像の前処理

        Args:
            image: 画像
            size: 画像のサイズ
            dtype: データ型
        """
        image = image.convert("RGB")
        if size:
            image = image.resize(size)
        image = np.asarray(image, dtype=dtype)
        image = image[:, :, ::-1]
        image -= np.array([103.939, 116.779, 123.68], dtype=dtype)
        image = image.transpose((2, 0, 1))
        return image


# =============================================================================
# ResNet
# =============================================================================
class ResNet(Model):
    """ResNet

    Attributes:
        conv1 (Conv2d): 畳み込み層
        bn1 (BatchNorm): バッチ正規化層
        res2 (ResBlock): ResBlock
        res3 (ResBlock): ResBlock
        res4 (ResBlock): ResBlock
        res5 (ResBlock): ResBlock
        fc (Linear): 全結合層
    """

    WEIGHTS_PATH = (
        "https://github.com/koki0702/dezero-models/releases/download/v0.1/resnet{}.npz"
    )

    def __init__(self, n_layers: int = 152, pretrained: bool = False):
        """コンストラクタ

        Args:
            n_layers: レイヤー数
            pretrained: 学習済みの重みを使用するかどうか
        """
        super().__init__()

        if n_layers == 50:
            block = [3, 4, 6, 3]
        elif n_layers == 101:
            block = [3, 4, 23, 3]
        elif n_layers == 152:
            block = [3, 8, 36, 3]
        else:
            raise ValueError(
                "The n_layers argument should be either 50, 101,"
                " or 152, but {} was given.".format(n_layers)
            )

        self.conv1 = L.Conv2d(64, 7, 2, 3)
        self.bn1 = L.BatchNorm()
        self.res2 = BuildingBlock(block[0], 64, 64, 256, 1)
        self.res3 = BuildingBlock(block[1], 256, 128, 512, 2)
        self.res4 = BuildingBlock(block[2], 512, 256, 1024, 2)
        self.res5 = BuildingBlock(block[3], 1024, 512, 2048, 2)
        self.fc6 = L.Linear(1000)

        if pretrained:
            weights_path = utils.get_file(ResNet.WEIGHTS_PATH.format(n_layers))
            self.load_weights(weights_path)

    def forward(self, x: Variable) -> Variable:
        """順伝播

        Args:
            x: 入力

        Returns:
            y: 出力
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.pooling(x, kernel_size=3, stride=2)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = _global_average_pooling_2d(x)
        x = self.fc6(x)
        return x


class ResNet152(ResNet):
    """ResNet152"""

    def __init__(self, pretrained: bool = False):
        """コンストラクタ

        Args:
            pretrained: 学習済みの重みを使用するかどうか
        """
        super().__init__(152, pretrained)


class ResNet101(ResNet):
    """ResNet101"""

    def __init__(self, pretrained: bool = False):
        """コンストラクタ

        Args:
            pretrained: 学習済みの重みを使用するかどうか
        """
        super().__init__(101, pretrained)


class ResNet50(ResNet):
    """ResNet50"""

    def __init__(self, pretrained: bool = False):
        """コンストラクタ

        Args:
            pretrained: 学習済みの重みを使用するかどうか
        """
        super().__init__(50, pretrained)


def _global_average_pooling_2d(x: Variable) -> Variable:
    """Global Average Pooling

    Args:
        x: 入力

    Returns:
        y: 出力
    """
    N, C, H, W = x.shape
    h = F.average_pooling(x, (H, W), stride=1)
    h = F.reshape(h, (N, C))
    return h


class BuildingBlock(Layer):
    """BuildingBlock

    Attributes:
        a (BottleneckA): BottleneckA
        b (BottleneckB): BottleneckB
        c (BottleneckB): BottleneckB
    """

    def __init__(
        self,
        n_layers: int = None,
        in_channels: int = None,
        mid_channels: int = None,
        out_channels: int = None,
        stride: int | tuple[int] = None,
        downsample_fb: bool = None,
    ) -> None:
        """コンストラクタ

        Args:
            n_layers: レイヤー数
            in_channels: 入力チャンネル数
            mid_channels: 中間チャンネル数
            out_channels: 出力チャンネル数
            stride: ストライド
            downsample_fb: downsampleを行うかどうか
        """
        super().__init__()

        self.a = BottleneckA(
            in_channels, mid_channels, out_channels, stride, downsample_fb
        )
        self._forward = ["a"]
        for i in range(n_layers - 1):
            name = "b{}".format(i + 1)
            bottleneck = BottleneckB(out_channels, mid_channels)
            setattr(self, name, bottleneck)
            self._forward.append(name)

    def forward(self, x: Variable) -> Variable:
        """順伝播

        Args:
            x: 入力

        Returns:
            y: 出力
        """
        for name in self._forward:
            layer = getattr(self, name)
            x = layer(x)
        return x


class BottleneckA(Layer):
    """BottleneckA

    Attributes:
        conv1 (Conv2d): 1x1 convolutional layers
        bn1 (BatchNorm): BatchNorm
        conv2 (Conv2d): 3x3 convolutional layers
        bn2 (BatchNorm): BatchNorm
        conv3 (Conv2d): 1x1 convolutional layers
        bn3 (BatchNorm): BatchNorm
        shortcut (Conv2d): 1x1 convolutional layers
        bn4 (BatchNorm): BatchNorm
    """

    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        stride: int | tuple[int] = 2,
        downsample_fb: bool = False,
    ):
        """コンストラクタ

        Args:
            in_channels: 入力チャンネル数
            mid_channels: 中間チャンネル数
            out_channels: 出力チャンネル数
            stride: ストライド
            downsample_fb: 3x3 convolutional layersでストライドを2にするかどうか
        """
        super().__init__()
        stride_1x1, stride_3x3 = (1, stride) if downsample_fb else (stride, 1)

        self.conv1 = L.Conv2d(mid_channels, 1, stride_1x1, 0, nobias=True)
        self.bn1 = L.BatchNorm()
        self.conv2 = L.Conv2d(mid_channels, 3, stride_3x3, 1, nobias=True)
        self.bn2 = L.BatchNorm()
        self.conv3 = L.Conv2d(out_channels, 1, 1, 0, nobias=True)
        self.bn3 = L.BatchNorm()
        self.conv4 = L.Conv2d(out_channels, 1, stride, 0, nobias=True)
        self.bn4 = L.BatchNorm()

    def forward(self, x: Variable) -> Variable:
        """順伝播

        Args:
            x: 入力

        Returns:
            y: 出力
        """
        h1 = F.relu(self.bn1(self.conv1(x)))
        h1 = F.relu(self.bn2(self.conv2(h1)))
        h1 = self.bn3(self.conv3(h1))
        h2 = self.bn4(self.conv4(x))
        return F.relu(h1 + h2)


class BottleneckB(Layer):
    """BottleneckB

    Attributes:
        conv1 (Conv2d): 1x1 convolutional layers
        bn1 (BatchNorm): BatchNorm
        conv2 (Conv2d): 3x3 convolutional layers
        bn2 (BatchNorm): BatchNorm
        conv3 (Conv2d): 1x1 convolutional layers
        bn3 (BatchNorm): BatchNorm
    """

    def __init__(self, in_channels: int, mid_channels: int) -> None:
        """コンストラクタ

        Args:
            in_channels: 入力チャンネル数
            mid_channels: 中間チャンネル数
        """
        super().__init__()

        self.conv1 = L.Conv2d(mid_channels, 1, 1, 0, nobias=True)
        self.bn1 = L.BatchNorm()
        self.conv2 = L.Conv2d(mid_channels, 3, 1, 1, nobias=True)
        self.bn2 = L.BatchNorm()
        self.conv3 = L.Conv2d(in_channels, 1, 1, 0, nobias=True)
        self.bn3 = L.BatchNorm()

    def forward(self, x: Variable) -> Variable:
        """順伝播

        Args:
            x: 入力

        Returns:
            y: 出力
        """
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = self.bn3(self.conv3(h))
        return F.relu(h + x)
