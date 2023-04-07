# 38. 形状を変える関数
import numpy as np

import dezero.functions as F
from dezero import Variable

if __name__ == "__main__":
    x = Variable(np.array([[0, 1, 2], [3, 4, 5]]))
    y = F.reshape(x, (6,))  # y = x.reshape(6)
    y.backward(retain_grad=True)
    print(x.grad)

    x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
    y = F.transpose(x)  # y = x.T
    y.backward()
    print(x.grad)
