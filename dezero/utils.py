import os
import subprocess

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
