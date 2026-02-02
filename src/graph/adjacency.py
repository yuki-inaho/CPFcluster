"""隣接行列生成モジュール"""

import numpy as np
import scipy.sparse as sp
from .knn import KnnResult


def build_knn_adjacency(knn: KnnResult) -> sp.csr_matrix:
    """kNN結果から隣接行列を構築

    Args:
        knn: kNN探索結果

    Returns:
        csr_matrix: 隣接行列 (n, n)
    """
    n, k = knn.indices.shape
    row_idx = np.repeat(np.arange(n), k)
    col_idx = knn.indices.flatten()
    data = knn.distances.flatten()
    return sp.csr_matrix((data, (row_idx, col_idx)), shape=(n, n), dtype=np.float32)


def make_mutual(adj: sp.csr_matrix) -> sp.csr_matrix:
    """相互k-NNグラフに変換（両方向でk近傍の場合のみエッジ保持）

    Args:
        adj: 隣接行列

    Returns:
        csr_matrix: 相互k-NN隣接行列
    """
    # Definition 7: mutual k-NN graph.
    return adj.minimum(adj.T)


def apply_mask(adj: sp.csr_matrix, mask: sp.csr_matrix) -> sp.csr_matrix:
    """マスクを適用（Spatial-CPF用）

    Args:
        adj: 隣接行列
        mask: マスク行列（0/1）

    Returns:
        csr_matrix: マスク適用後の隣接行列
    """
    return adj.multiply(mask)
