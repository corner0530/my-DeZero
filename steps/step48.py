# 48. 多値分類
import math

import matplotlib.pyplot as plt
import numpy as np

import dezero
import dezero.functions as F
from dezero import optimizers
from dezero.models import MLP

if __name__ == "__main__":
    # ハイパーパラメータの設定
    max_epoch = 300
    batch_size = 30
    hidden_size = 10
    lr = 1.0

    # データの読み込み / モデル・オプティマイザの生成
    x, t = dezero.datasets.get_spiral(train=True)
    model = MLP((hidden_size, 3))
    optimizer = optimizers.SGD(lr).setup(model)

    data_size = len(x)
    max_iter = math.ceil(data_size / batch_size)  # 小数点の切り上げ

    for epoch in range(max_epoch):
        # データのシャッフル
        index = np.random.permutation(data_size)
        sum_loss = 0

        for i in range(max_iter):
            # ミニバッチの生成
            batch_index = index[i * batch_size : (i + 1) * batch_size]
            batch_x = x[batch_index]
            batch_t = t[batch_index]

            # 勾配の算出・パラメータを更新
            y = model(batch_x)
            loss = F.softmax_cross_entropy_simple(y, batch_t)
            model.cleargrads()
            loss.backward()
            optimizer.update()

            sum_loss += float(loss.data) * len(batch_t)

        # エポックごとの損失の表示
        avg_loss = sum_loss / data_size
        print("epoch %d, loss %.2f" % (epoch + 1, avg_loss))

    # データの描画
    h = 0.001
    x_min, x_max = x[:, 0].min() - 0.1, x[:, 0].max() + 0.1
    y_min, y_max = x[:, 1].min() - 0.1, x[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    X = np.c_[xx.ravel(), yy.ravel()]

    with dezero.no_grad():
        score = model(X)
    predict_cls = np.argmax(score.data, axis=1)
    Z = predict_cls.reshape(xx.shape)
    plt.contourf(xx, yy, Z)

    # データ点の描画
    N, CLS_NUM = 100, 3
    markers = ["o", "x", "^"]
    colors = ["orange", "blue", "green"]
    for i in range(len(x)):
        c = t[i]
        plt.scatter(x[i][0], x[i][1], s=40, marker=markers[c], c=colors[c])
    plt.show()
