import math

from dezero import Layer, Model, Parameter, cuda


# =============================================================================
# Optimizer (base class)
# =============================================================================
class Optimizer:
    """最適化を行う基底クラス

    Attributes:
        target (Model | Layer): 最適化の対象
        hooks (list): 前処理
    """

    def __init__(self) -> None:
        """コンストラクタ"""
        self.target = None
        self.hooks = []

    def setup(self, target: Model | Layer) -> "Optimizer":
        """最適化の対象の設定

        Args:
            target: 最適化の対象
        """
        self.target = target
        return self

    def update(self) -> None:
        """パラメータの更新"""
        # Noneのパラメータを除外
        params = [p for p in self.target.params() if p.grad is not None]

        # 前処理
        for f in self.hooks:
            f(params)

        # パラメータの更新
        for param in params:
            self.update_one(param)

    def update_one(self, param: Parameter) -> None:
        """1つのパラメータについての更新

        Args:
            param: パラメータ

        Raises:
            NotImplementedError: 未実装の場合
        """
        raise NotImplementedError()

    def add_hook(self, f: object) -> None:
        """前処理を追加する

        Args:
            f: 前処理
        """
        self.hooks.append(f)


# =============================================================================
# Hook functions
# =============================================================================
class WeightDecay:
    """重み減衰

    Attributes:
        rate (float): 減衰率
    """

    def __init__(self, rate: float) -> None:
        """コンストラクタ

        Args:
            rate: 減衰率
        """
        self.rate = rate

    def __call__(self, params: list[Parameter]) -> None:
        """重みを減衰させる

        Args:
            params: パラメータ
        """
        for param in params:
            param.grad.data += self.rate * param.data


class ClipGrad:
    """勾配のクリッピング

    Attributes:
        max_norm (float): 最大ノルム
    """

    def __init__(self, max_norm: float) -> None:
        """コンストラクタ

        Args:
            max_norm: 最大ノルム
        """
        self.max_norm = max_norm

    def __call__(self, params: list[Parameter]) -> None:
        """勾配のクリッピングを行う

        Args:
            params: パラメータ
        """
        total_norm = 0
        for param in params:
            total_norm += (param.grad.data**2).sum()
        total_norm = math.sqrt(float(total_norm))

        rate = self.max_norm / (total_norm + 1e-6)
        if rate < 1:
            for param in params:
                param.grad.data *= rate


class FreezeParam:
    """パラメータの固定

    Attributes:
        freeze_params (list): 固定するパラメータ
    """

    def __init__(self, *layers: Layer) -> None:
        """コンストラクタ

        Args:
            layers: 固定するパラメータを持つレイヤ
        """
        self.freeze_params = []
        for l in layers:
            if isinstance(l, Parameter):
                self.freeze_params.append(l)
            else:
                for p in l.params():
                    self.freeze_params.append(p)

    def __call__(self, params: list[Parameter]) -> None:
        """勾配を初期化する

        Args:
            params: パラメータ
        """
        for p in self.freeze_params:
            p.grad = None


# =============================================================================
# SGD / MomentumSGD / AdaGrad / AdaDelta / Adam
# =============================================================================
class SGD(Optimizer):
    """確率的勾配降下法

    Attributes:
        lr (float): 学習率
    """

    def __init__(self, lr: float = 0.01) -> None:
        """コンストラクタ

        Args:
            lr: 学習率
        """
        super().__init__()
        self.lr = lr

    def update_one(self, param: Parameter) -> None:
        """1つのパラメータについての更新

        Args:
            param: パラメータ
        """
        param.data -= self.lr * param.grad.data


class MomentumSGD(Optimizer):
    """Momentum SGD

    Attributes:
        lr (float): 学習率
        momentum (float): モーメンタム
        vs (dict): モーメンタムの値
    """

    def __init__(self, lr: float = 0.01, momentum: float = 0.9) -> None:
        """コンストラクタ

        Args:
            lr: 学習率
            momentum: モーメンタム
        """
        super().__init__()
        self.lr = lr
        self.momentum = momentum
        self.vs = {}

    def update_one(self, param: Parameter) -> None:
        """1つのパラメータについての更新

        Args:
            param: パラメータ
        """
        v_key = id(param)
        if v_key not in self.vs:
            xp = cuda.get_array_module(param.data)
            self.vs[v_key] = xp.zeros_like(param.data)

        v = self.vs[v_key]
        v *= self.momentum
        v -= self.lr * param.grad.data
        param.data += v


class AdaGrad(Optimizer):
    """AdaGrad

    Attributes:
        lr (float): 学習率
        eps (float): 0除算を防ぐための微小値
        hs (dict): 勾配の二乗和
    """

    def __init__(self, lr: float = 0.001, eps: float = 1e-8) -> None:
        """コンストラクタ

        Args:
            lr: 学習率
            eps: 0除算を防ぐための微小値
        """
        super().__init__()
        self.lr = lr
        self.eps = eps
        self.hs = {}

    def update_one(self, param: Parameter) -> None:
        """1つのパラメータについての更新

        Args:
            param: パラメータ
        """
        xp = cuda.get_array_module(param.data)

        h_key = id(param)
        if h_key not in self.hs:
            self.hs[h_key] = xp.zeros_like(param.data)

        h = self.hs[h_key]
        grad = param.grad.data
        h += grad * grad
        param.data -= self.lr * grad / (xp.sqrt(h) + self.eps)


class AdaDelta(Optimizer):
    """AdaDelta

    Attributes:
        rho (float): 指数関数の減衰率
        eps (float): 0除算を防ぐための微小値
        msg (dict): 勾配の二乗和
        msdx (dict): 更新量の二乗和
    """

    def __init__(self, rho: float = 0.95, eps: float = 1e-6) -> None:
        """コンストラクタ

        Args:
            rho: 指数関数の減衰率
            eps: 0除算を防ぐための微小値
        """
        super().__init__()
        self.rho = rho
        self.eps = eps
        self.msg = {}
        self.msdx = {}

    def update_one(self, param: Parameter) -> None:
        """1つのパラメータについての更新

        Args:
            param: パラメータ
        """
        xp = cuda.get_array_module(param.data)

        key = id(param)
        if key not in self.msg:
            self.msg[key] = xp.zeros_like(param.data)
            self.msdx[key] = xp.zeros_like(param.data)

        msg = self.msg[key]
        msdx = self.msdx[key]
        grad = param.grad.data

        msg *= self.rho
        msg += (1 - self.rho) * grad * grad
        dx = grad * xp.sqrt((msdx + self.eps) / (msg + self.eps))
        msdx *= self.rho
        msdx += (1 - self.rho) * dx * dx
        param.data -= dx


class Adam(Optimizer):
    """Adam

    Attributes:
        alpha (float): 学習率
        beta1 (float): 1次モーメントの減衰率
        beta2 (float): 2次モーメントの減衰率
        eps (float): 0除算を防ぐための微小値
        t (int): 更新回数
        ms (dict): 1次モーメント
        vs (dict): 2次モーメント
    """

    def __init__(
        self,
        alpha: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        """コンストラクタ

        Args:
            alpha: 学習率
            beta1: 1次モーメントの減衰率
            beta2: 2次モーメントの減衰率
            eps: 0除算を防ぐための微小値
        """
        super().__init__()
        self.t = 0
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.ms = {}
        self.vs = {}

    def update(self, *args: Parameter, **kwargs: Parameter) -> None:
        """パラメータの更新

        Args:
            *args: パラメータ
            **kwargs: パラメータ
        """
        self.t += 1
        super().update(*args, **kwargs)

    @property
    def lr(self) -> float:
        """学習率"""
        fix1 = 1.0 - math.pow(self.beta1, self.t)
        fix2 = 1.0 - math.pow(self.beta2, self.t)
        return self.alpha * math.sqrt(fix2) / fix1

    def update_one(self, param: Parameter) -> None:
        """1つのパラメータについての更新

        Args:
            param: パラメータ
        """
        xp = cuda.get_array_module(param.data)

        key = id(param)
        if key not in self.ms:
            self.ms[key] = xp.zeros_like(param.data)
            self.vs[key] = xp.zeros_like(param.data)

        m, v = self.ms[key], self.vs[key]
        grad = param.grad.data

        m += (1 - self.beta1) * (grad - m)
        v += (1 - self.beta2) * (grad * grad - v)
        param.data -= self.lr * m / (xp.sqrt(v) + self.eps)
