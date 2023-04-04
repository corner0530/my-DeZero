# 01. 箱としての変数
import numpy as np
from common import Variable

if __name__ == "__main__":
    data = np.array(1.0)  # データ
    x = Variable(data)  # データを入れた箱
    print(x.data)

    x.data = np.array(2.0)  # データを代入
    print(x.data)  # データを参照
