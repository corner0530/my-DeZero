# 50. ミニバッチを取り出すDataLoader
import dezero
import dezero.functions as F
from dezero import DataLoader, optimizers
from dezero.models import MLP

if __name__ == "__main__":
    max_epoch = 300
    batch_size = 30
    hidden_size = 10
    lr = 1.0

    train_set = dezero.datasets.Spiral(train=True)
    test_set = dezero.datasets.Spiral(train=False)
    train_loader = DataLoader(train_set, batch_size)
    test_loader = DataLoader(test_set, batch_size, shuffle=False)

    model = MLP((hidden_size, 3))
    optimizer = optimizers.SGD(lr).setup(model)

    for epoch in range(max_epoch):
        sum_loss, sum_acc = 0, 0

        for x, t in train_loader:  # 訓練用のミニバッチデータ
            y = model(x)
            loss = F.softmax_cross_entropy(y, t)
            acc = F.accuracy(y, t)  # 訓練データの認識精度
            model.cleargrads()
            loss.backward()
            optimizer.update()

            sum_loss += float(loss.data) * len(t)
            sum_acc += float(acc.data) * len(t)

        print("epoch: {}".format(epoch + 1))
        print(
            "train loss: {:.4f}, accuracy: {:.4f}".format(
                sum_loss / len(train_set), sum_acc / len(train_set)
            )
        )

        sum_loss, sum_acc = 0, 0
        with dezero.no_grad():  # 勾配不要モード
            for x, t in test_loader:  # テスト用のミニバッチデータ
                y = model(x)
                loss = F.softmax_cross_entropy(y, t)
                acc = F.accuracy(y, t)  # テストデータの認識精度
                sum_loss += float(loss.data) * len(t)
                sum_acc += float(acc.data) * len(t)

        print(
            "test loss: {:.4f}, accuracy: {:.4f}".format(
                sum_loss / len(test_set), sum_acc / len(test_set)
            )
        )
