# 34. sin関数の高階微分
import matplotlib.pyplot as plt
import numpy as np

import dezero.functions as F
from dezero import Variable

if __name__ == "__main__":
    x = Variable(np.linspace(-7, 7, 200))
    y = F.sin(x)
    y.backward(create_graph=True)

    logs = [y.data]

    for i in range(3):
        logs.append(x.grad.data)
        gx = x.grad
        x.cleargrad()
        gx.backward(create_graph=True)

    labels = ["y=sin(x)", "y'", "y''", "y'''"]
    for i, v in enumerate(logs):
        plt.plot(x.data, logs[i], label=labels[i])
    plt.legend(loc="lower right")
    plt.show()
