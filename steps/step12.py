# 12. 可変長の引数(改善編)
import numpy as np
from common import Variable, add

if __name__ == "__main__":
    x0 = Variable(np.array(2))
    x1 = Variable(np.array(3))
    y = add(x0, x1)
    print(y.data)
