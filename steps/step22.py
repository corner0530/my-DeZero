# 22. 演算子のオーバーロード(3)
import numpy as np
from common import Variable

if __name__ == "__main__":
    x = Variable(np.array(2.0))
    y = -x
    print(y)  # variable(-2.0)

    y1 = 2.0 - x
    y2 = x - 1.0
    print(y1)  # variable(0.0)
    print(y2)  # variable(1.0)

    y = 3.0 / x
    print(y)  # variable(1.5)

    y = x ** 3
    y.backward()
    print(y)  # variable(8.0)
