# 41. 行列の積
import numpy as np

import dezero.functions as F
from dezero import Variable

if __name__ == "__main__":
    x = Variable(np.random.randn(2, 3))
    w = Variable(np.random.randn(3, 4))
    y = F.matmul(x, w)
    y.backward()

    print(x.grad.shape)
    print(w.grad.shape)
