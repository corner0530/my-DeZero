import numpy as np

from dezero import Variable

gpu_enable = True
try:
    import cupy as cp

    cupy = cp
except ImportError:
    gpu_enable = False


def get_array_module(x: Variable | np.ndarray) -> object:
    """`x`に対応するモジュールを返す

    Args:
        x: NumPyかCuPyかを判定するための値

    Returns:
        `numpy` or `cupy`
    """
    if isinstance(x, Variable):
        x = x.data

    if not gpu_enable:
        return np
    xp = cp.get_array_module(x)
    return xp


def as_numpy(x: "cp.ndarray") -> np.ndarray:
    """`np.ndarray`に変換する

    Args:
        x: `np.ndarray`に変換する値

    Returns:
        `np.ndarray`に変換した値
    """
    if isinstance(x, Variable):
        x = x.data

    if np.isscalar(x):
        return np.array(x)
    elif isinstance(x, np.ndarray):
        return x
    return cp.asnumpy(x)


def as_cupy(x: np.ndarray) -> "cp.ndarray":
    """`cp.ndarray`に変換する

    Args:
        x: `cp.ndarray`に変換する値

    Returns:
        `cp.ndarray`に変換した値

    Raises:
        Exception: CuPyがインストールされていない場合
    """
    if isinstance(x, Variable):
        x = x.data

    if not gpu_enable:
        raise Exception("CuPy cannot be loaded. Install CuPy!")
    return cp.asarray(x)
