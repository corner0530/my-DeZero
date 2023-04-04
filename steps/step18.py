# 18. メモリ使用量を減らすモード
import numpy as np
from common import Variable, add, no_grad, square, using_config

if __name__ == "__main__":
    # 不要な微分は保持しない
    x0 = Variable(np.array(1.0))
    x1 = Variable(np.array(1.0))
    t = add(x0, x1)
    y = add(x0, t)
    y.backward()

    print(y.grad, t.grad)
    print(x0.grad, x1.grad)

    # with文による切り替え
    with using_config("enable_backprop", False):
        # この中では逆伝播が無効
        x = Variable(np.array(2.0))
        y = square(x)

    with no_grad():
        # この中でも逆伝播が無効
        x = Variable(np.array(2.0))
        y = square(x)
