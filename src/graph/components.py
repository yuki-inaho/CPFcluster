"""連結成分抽出モジュール"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph
from src.types import OUTLIER, Int32Array


def extract_components(adj: sp.csr_matrix) -> Int32Array:
    """連結成分を抽出

    Args:
        adj: 隣接行列

    Returns:
        Int32Array: 成分ラベル (n,)
    """
    # Definition 9: connected components of the mutual k-NN graph.
    _, labels = csgraph.connected_components(adj, directed=False, return_labels=True)
    return labels.astype(np.int32)


def filter_by_component_size(
    components: Int32Array,
    min_size: int,
) -> Int32Array:
    """小さな成分を外れ値としてマーク

    Args:
        components: 成分ラベル
        min_size: 最小成分サイズ（これ以下は外れ値）

    Returns:
        Int32Array: 外れ値をOUTLIER(-1)でマークした成分ラベル
    """
    result = components.copy()
    labels, counts = np.unique(components, return_counts=True)
    small_components = labels[counts <= min_size]
    mask = np.isin(components, small_components)
    result[mask] = OUTLIER
    return result


def filter_by_edge_count(
    adj: sp.csr_matrix,
    components: Int32Array,
    min_edges: int,
) -> Int32Array:
    """辺数が少ない点を外れ値としてマーク（論文準拠）

    Args:
        adj: 隣接行列
        components: 成分ラベル
        min_edges: 最小辺数（これ以下は外れ値）

    Returns:
        Int32Array: 外れ値をOUTLIER(-1)でマークした成分ラベル
    """
    result = components.copy()
    degrees = np.array(adj.getnnz(axis=1)).flatten()
    mask = degrees <= min_edges
    result[mask] = OUTLIER
    return result
