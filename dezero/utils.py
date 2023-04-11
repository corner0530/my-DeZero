import os
import subprocess
import urllib.request

import numpy as np

from dezero import Function, Variable, as_variable, cuda


# =============================================================================
# Visualize for computational graph
# =============================================================================
def _dot_var(v: Variable, verbose: bool = False) -> str:
    """変数をDOT言語に変換

    Args:
        v: 変換する変数
        verbose: 形状と型を合わせてラベルに表示するかどうか

    Returns:
        変数を表すDOT言語の文字列
    """
    dot_var = '{} [label="{}", color=orange, style=filled]\n'

    name = "" if v.name is None else v.name
    if verbose and v.data is not None:
        if v.name is not None:
            name += ": "
        name += str(v.shape) + " " + str(v.dtype)  # ラベル名は`変数名: 形状 型`とする

    return dot_var.format(id(v), name)  # オブジェクトのIDをノードIDとする


def _dot_func(f: Function) -> str:
    """関数をDOT言語に変換

    Args:
        f: 変換する関数

    Returns:
        関数を表すDOT言語の文字列
    """
    # 関数
    dot_func = '{} [label="{}", color=lightblue, style=filled, shape=box]\n'
    ret = dot_func.format(id(f), f.__class__.__name__)

    # 矢印
    dot_edge = "{} -> {}\n"
    for x in f.inputs:
        # 入力→関数
        ret += dot_edge.format(id(x), id(f))
    for y in f.outputs:
        # 関数→出力
        ret += dot_edge.format(id(f), id(y()))  # yはweakref
    return ret


def get_dot_graph(output: Variable, verbose: bool = True) -> str:
    """計算グラフをDOT言語で表現

    Args:
        output: 出力変数
        verbose: 形状と型を合わせてラベルに表示するかどうか

    Returns:
        計算グラフを表すDOT言語の文字列
    """
    txt = ""
    funcs = []
    seen_set = set()

    def add_func(f: Function) -> None:
        """関数をリストに追加

        Args:
            f: 追加する関数
        """
        if f not in seen_set:
            funcs.append(f)
            seen_set.add(f)

    add_func(output.creator)
    txt += _dot_var(output, verbose)
    while funcs:
        func = funcs.pop()
        txt += _dot_func(func)
        for x in func.inputs:
            txt += _dot_var(x, verbose)

            if x.creator is not None:
                add_func(x.creator)

    return "digraph g {\n" + txt + "}"


def plot_dot_graph(
    output: Variable, verbose: bool = True, to_file: str = "graph.png"
) -> None:
    """計算グラフを画像に保存

    Args:
        output: 出力変数
        verbose: 形状と型を合わせてラベルに表示するかどうか
        to_file: 保存するファイル名
    """
    dot_graph = get_dot_graph(output, verbose)

    # dotデータを`~/.dezero/tmp_graph.dot`として保存
    tmp_dir = os.path.join(os.path.expanduser("~"), ".dezero")
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    graph_path = os.path.join(tmp_dir, "tmp_graph.dot")

    with open(graph_path, "w") as f:
        f.write(dot_graph)

    # dotコマンドを実行して画像を生成
    extension = os.path.splitext(to_file)[1][1:]  # 拡張子
    cmd = "dot {} -T {} -o {}".format(graph_path, extension, to_file)
    subprocess.run(cmd, shell=True)

    # 画像を表示
    try:
        from IPython import display

        return display.Image(filename=to_file)
    except (ImportError, ValueError):
        pass


def sum_to(x: Variable, shape: tuple[int]) -> Variable:
    """要素の和を求めて指定した形状に整形する

    Args:
        x: 入力
        shape: 出力の形状

    Returns:
        指定した形状に整形した結果
    """
    ndim = len(shape)
    lead = x.ndim - ndim
    lead_axis = tuple(range(lead))

    axis = tuple([i + lead for i, sx in enumerate(shape) if sx == 1])
    y = x.sum(lead_axis + axis, keepdims=True)
    if lead > 0:
        y = y.squeeze(lead_axis)
    return y


def reshape_sum_backward(
    gy: Variable, x_shape: tuple, axis: int | tuple, keepdims: bool
) -> Variable:
    """sum関数の逆伝播で整形する

    Args:
        gy : 出力側から伝わる微分
        x_shape: 入力の形状
        axis: 順伝播の際のaxis
        keepdims: 順伝播の際のkeepdims
    Returns:
        整形した微分
    """
    ndim = len(x_shape)
    tupled_axis = axis
    if axis is None:
        tupled_axis = None
    elif not isinstance(axis, tuple):
        tupled_axis = (axis,)

    if not (ndim == 0 or tupled_axis is None or keepdims):
        actual_axis = [a if a >= 0 else a + ndim for a in tupled_axis]
        shape = list(gy.shape)
        for a in sorted(actual_axis):
            shape.insert(a, 1)
    else:
        shape = gy.shape

    gy = gy.reshape(shape)  # reshape
    return gy


def logsumexp(x: Variable, axis: int = 1):
    """logsumexp関数

    Args:
        x: 入力
        axis: 和をとる軸

    Returns:
        出力
    """
    m = x.max(axis=axis, keepdims=True)
    y = np.exp(x - m)
    s = np.log(y.sum(axis=axis, keepdims=True))
    m += s
    return m


def max_backward_shape(x, axis):
    if axis is None:
        axis = range(x.ndim)
    elif isinstance(axis, int):
        axis = (axis,)
    else:
        axis = axis

    shape = [s if ax not in axis else 1 for ax, s in enumerate(x.shape)]
    return shape


# =============================================================================
# Gradient check
# =============================================================================
def gradient_check(
    f: object,
    x: Variable,
    *args: object,
    rtol: float = 1e-4,
    atol: float = 1e-5,
    **kwargs: object
) -> bool:
    """勾配チェック

    Args:
        f: 関数
        x: 入力
        *args: 関数の引数
        rtol: 相対誤差
        atol: 絶対誤差
        **kwargs: 関数のキーワード引数

    Returns:
        結果が範囲内かどうか
    """
    x = as_variable(x)
    x.data = x.data.astype(np.float64)

    num_grad = numerical_grad(f, x, *args, **kwargs)
    y = f(x, *args, **kwargs)
    y.backward()
    bp_grad = x.grad.data

    assert bp_grad.shape == num_grad.shape
    res = array_allclose(num_grad, bp_grad, atol=atol, rtol=rtol)

    if not res:
        print("")
        print("========== FAILED (Gradient Check) ==========")
        print("Numerical Grad")
        print(" shape: {}".format(num_grad.shape))
        val = str(num_grad.flatten()[:10])
        print(" values: {} ...".format(val[1:-1]))
        print("Backprop Grad")
        print(" shape: {}".format(bp_grad.shape))
        val = str(bp_grad.flatten()[:10])
        print(" values: {} ...".format(val[1:-1]))
    return res


def numerical_grad(
    f: object, x: Variable, *args: object, **kwargs: object
) -> np.ndarray:
    """数値微分

    Args:
        f: 関数
        x: 入力
        *args: 関数の引数
        **kwargs: 関数のキーワード引数

    Returns:
        数値微分の結果
    """
    eps = 1e-4

    x = x.data if isinstance(x, Variable) else x
    xp = cuda.get_array_module(x)
    if xp is not np:
        np_x = cuda.as_numpy(x)
    else:
        np_x = x
    grad = xp.zeros_like(x)

    it = np.nditer(np_x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx].copy()

        x[idx] = tmp_val + eps
        y1 = f(x, *args, **kwargs)  # f(x+h)
        if isinstance(y1, Variable):
            y1 = y1.data
        y1 = y1.copy()

        x[idx] = tmp_val - eps
        y2 = f(x, *args, **kwargs)  # f(x-h)
        if isinstance(y2, Variable):
            y2 = y2.data
        y2 = y2.copy()

        diff = (y1 - y2).sum()
        grad[idx] = diff / (2 * eps)

        x[idx] = tmp_val
        it.iternext()
    return grad


def array_equal(a: np.ndarray, b: np.ndarray) -> bool:
    """2つの配列が等しいかどうかを判定する

    Args:
        a: 配列
        b: 配列
    """
    a = a.data if isinstance(a, Variable) else a
    b = b.data if isinstance(b, Variable) else b
    a, b = cuda.as_numpy(a), cuda.as_numpy(b)
    return np.array_equal(a, b)


def array_allclose(
    a: np.ndarray, b: np.ndarray, rtol: float = 1e-4, atol: float = 1e-5
) -> bool:
    """2つの配列が近いかどうかを判定する

    Args:
        a: 配列
        b: 配列
        rtol: 相対誤差
        atol: 絶対誤差

    Returns:
        結果が範囲内かどうか
    """
    a = a.data if isinstance(a, Variable) else a
    b = b.data if isinstance(b, Variable) else b
    a, b = cuda.as_numpy(a), cuda.as_numpy(b)
    return np.allclose(a, b, atol=atol, rtol=rtol)


# =============================================================================
# download function
# =============================================================================
def show_progress(block_num: int, block_size: int, total_size: int) -> None:
    """ダウンロードの進捗を表示

    Args:
        block_num: ダウンロードしたブロック数
        block_size: ブロックサイズ
        total_size: 全体のサイズ
    """
    bar_template = "\r[{}] {:.2f}%"

    downloaded = block_num * block_size
    p = downloaded / total_size * 100
    i = int(downloaded / total_size * 30)
    if p >= 100.0:
        p = 100.0
    if i >= 30:
        i = 30
    bar = "#" * i + "." * (30 - i)
    print(bar_template.format(bar, p), end="")


cache_dir = os.path.join(os.path.expanduser("~"), ".dezero")


def get_file(url: str, file_name: str = None) -> str:
    """キャッシュが無ければファイルをダウンロード

    Args:
        url: ダウンロードするファイルのURL
        file_name: ダウンロードするファイルのファイル名

    Returns:
        ダウンロードしたファイルの絶対パス
    """
    if file_name is None:
        file_name = url[url.rfind("/") + 1 :]
    file_path = os.path.join(cache_dir, file_name)

    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)

    if os.path.exists(file_path):
        return file_path

    print("Downloading: " + file_name)
    try:
        urllib.request.urlretrieve(url, file_path, show_progress)
    except (Exception, KeyboardInterrupt):
        if os.path.exists(file_path):
            os.remove(file_path)
        raise
    print(" Done")

    return file_path


# =============================================================================
# others
# =============================================================================
def get_deconv_outsize(size: int, k: int, s: int, p: int) -> int:
    """転置畳み込み層の出力サイズを計算

    Args:
        size: 入力サイズ
        k: カーネルサイズ
        s: ストライド
        p: パディング

    Returns:
        出力サイズ
    """
    return s * (size - 1) + k - 2 * p


def get_conv_outsize(input_size: int, kernel_size: int, stride: int, pad: int) -> int:
    """畳み込み層の出力サイズを計算

    Args:
        input_size: 入力サイズ
        kernel_size: カーネルサイズ
        stride: ストライド
        pad: パディング

    Returns:
        出力サイズ
    """
    return (input_size + pad * 2 - kernel_size) // stride + 1


def pair(x: int | tuple) -> tuple:
    """整数を2つの要素を持つタプルに変換

    Args:
        x: 整数またはタプル

    Returns:
        2つの要素を持つタプル
    """
    if isinstance(x, int):
        return (x, x)
    elif isinstance(x, tuple):
        assert len(x) == 2
        return x
    else:
        raise ValueError
