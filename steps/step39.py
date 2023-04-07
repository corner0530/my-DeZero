# 39. 和を求める関数
import numpy as np

import dezero.functions as F
from dezero import Variable

if __name__ == '__main__':
    x = Variable(np.array([1, 2, 3, 4, 5, 6]))
    y = F.sum(x)
    y.backward()
    print(y)
    print(x.grad)

    x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
    y = F.sum(x)
    y.backward()
    print(y)
    print(x.grad)

    x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
    y = F.sum(x, axis=0)
    y.backward()
    print(y)
    print(x.grad)

    x = Variable(np.random.randn(2, 3, 4, 5))
    y = x.sum(keepdims=True)
    print(y.shape)
