# 03. 関数の連結
import numpy as np
from common import Exp, Square, Variable

if __name__ == "__main__":
    # Functionクラスの入出力はともにVariableクラスなので連結できる
    A = Square()
    B = Exp()
    C = Square()

    x = Variable(np.array(0.5))
    a = A(x)
    b = B(a)
    y = C(b)
    print(y.data)
