# 19. 変数を使いやすく
import numpy as np
from common import Variable

if __name__ == "__main__":
    x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
    x.name = "x"

    print(x.name)
    print(x.shape)
    print(x)
