import dezero.functions as F
import dezero.layers as L
from dezero import Layer, utils


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
