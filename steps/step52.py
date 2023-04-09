# 52. GPU対応
import time

import dezero
import dezero.functions as F
from dezero import DataLoader, optimizers
from dezero.models import MLP

if __name__ == "__main__":
    max_epoch = 5
    batch_size = 100

    train_set = dezero.datasets.MNIST(train=True)
    train_loader = DataLoader(train_set, batch_size)
    model = MLP((1000, 10))
    optimizer = optimizers.SGD().setup(model)

    # GPU対応
    if dezero.cuda.gpu_enable:
        train_loader.to_gpu()
        model.to_gpu()

    for epoch in range(max_epoch):
        start = time.time()
        sum_loss = 0

        for x, t in train_loader:
            y = model(x)
            loss = F.softmax_cross_entropy(y, t)
            model.cleargrads()
            loss.backward()
            optimizer.update()
            sum_loss += float(loss.data) * len(t)

        elapsed_time = time.time() - start
        print('epoch: {}, loss: {:.4f}, time: {:.4f}[sec]'.format(
            epoch + 1, sum_loss / len(train_set), elapsed_time))
