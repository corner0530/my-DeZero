# 07. バックプロパゲーションの自動化
import numpy as np
from common import Exp, Square, Variable

if __name__ == "__main__":
    # 関数と変数のつながりが計算が行われるときに作られる
    # このつながりはリンク付きノードと呼ばれる

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
