import os
import subprocess
import urllib.request

import numpy as np

from dezero import Function, Variable


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
    except ImportError:
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
