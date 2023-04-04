# 09. 関数をより便利に
import numpy as np
from common import Variable, exp, square

if __name__ == "__main__":
    # 1. 関数をPythonの関数として利用する
    x = Variable(np.array(0.5))
    y = square(exp(square(x)))  # 連続して適用

    # 2. 逆伝播の初期値を設定する必要をなくす
    y.backward()
    print(x.grad)

    # 3. Variableの初期化ではndarrayだけを扱う
    x = Variable(np.array(1.0))  # OK
    x = Variable(None)  # OK
    x = Variable(1.0)  # NG
