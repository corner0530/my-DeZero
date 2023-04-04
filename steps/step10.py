# 10. テストを行う
import unittest

import numpy as np
from common import Variable, numerical_diff, square


class SquareTest(unittest.TestCase):
    """Squareクラスのテスト"""
    def test_forward(self):  # テストメソッドはtestから始める
        """順伝播のテスト"""
        x = Variable(np.array(2.0))
        y = square(x)
        expected = np.array(4.0)
        self.assertEqual(y.data, expected)  # 出力と期待した値が一致するか検証

    def test_backward(self):
        """逆伝播のテスト"""
        x = Variable(np.array(3.0))
        y = square(x)
        y.backward()
        expected = np.array(6.0)
        self.assertEqual(x.grad, expected)

    def test_gradient_check(self):
        """勾配確認のテスト"""
        x = Variable(np.random.rand(1))
        y = square(x)
        y.backward()
        num_grad = numerical_diff(square, x)
        flg = np.allclose(x.grad, num_grad)  # 両方のndarrayが近いかどうかを判定
        self.assertTrue(flg)


if __name__ == "__main__":
    unittest.main()
