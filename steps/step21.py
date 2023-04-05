# 21. 演算子のオーバーロード(2)
import numpy as np
from common import Variable

if __name__ == "__main__":
    x = Variable(np.array(2.0))
    y = x + np.array(3.0)
    print(y)

    y = x + 3.0
    print(y)

    y = 3.0 * x + 1.0
    print(y)
