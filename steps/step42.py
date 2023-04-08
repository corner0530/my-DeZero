# 42. 線形回帰
import matplotlib.pyplot as plt
import numpy as np

import dezero.functions as F
from dezero import Variable

if __name__ == "__main__":
    # トイ・データセット
    np.random.seed(0)
    x = np.random.rand(100, 1)
    y = 5 + 2 * x + np.random.rand(100, 1)
    x = Variable(x)
    y = Variable(y)

    w = Variable(np.zeros((1, 1)))
    b = Variable(np.zeros(1))

    def predict(x):
        y = F.matmul(x, w) + b
        return y

    def mean_squared_error(x0, x1):
        diff = x0 - x1
        return F.sum(diff**2) / len(diff)

    lr = 0.1
    iters = 100

    for i in range(iters):
        y_pred = predict(x)
        loss = mean_squared_error(y, y_pred)

        w.cleargrad()
        b.cleargrad()
        loss.backward()

        # 更新
        w.data -= lr * w.grad.data
        b.data -= lr * b.grad.data
        print(w, b, loss)

    # プロット
    plt.scatter(x.data, y.data, s=10)
    plt.xlabel("x")
    plt.ylabel("y")
    y_pred = predict(x)
    plt.plot(x.data, y_pred.data, color="r")
    plt.show()
