# 08. 再帰からループへ
import numpy as np
from common import Exp, Square, Variable

if __name__ == "__main__":
    A = Square()
    B = Exp()
    C = Square()

    # 順伝播
    x = Variable(np.array(0.5))
    a = A(x)
    b = B(a)
    y = C(b)

    # 逆伝播
    y.grad = np.array(1.0)
    y.backward()
    print(x.grad)
