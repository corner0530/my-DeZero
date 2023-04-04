# 16. 複雑な計算グラフ(実装編)
import numpy as np
from common import Variable, add, square

if __name__ == "__main__":
    x = Variable(np.array(2.0))
    a = square(x)
    y = add(square(a), square(a))
    y.backward()

    print(y.data)
    print(x.grad)
