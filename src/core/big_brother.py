"""Big Brother計算モジュール"""

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from sklearn.neighbors import NearestNeighbors
from src.types import NO_PARENT, OUTLIER, Int32Array, Float32Array


@dataclass(frozen=True)
class BigBrotherResult:
    """Big Brother計算結果"""

    parent: Int32Array
    parent_dist: Float32Array


def compute_big_brother_for_component(
    X_cc: NDArray[np.float32],
    radius_cc: Float32Array,
    k: int,
) -> BigBrotherResult:
    """単一成分内でbig brotherを計算

    Args:
        X_cc: 成分内の点座標
        radius_cc: 成分内の各点のk近傍距離
        k: 近傍数

    Returns:
        BigBrotherResult: 成分内でのbig brother情報
    """
    nc = len(X_cc)
    parent = np.full(nc, NO_PARENT, dtype=np.int32)
    parent_dist = np.full(nc, np.inf, dtype=np.float32)

    if nc <= 1:
        return BigBrotherResult(parent=parent, parent_dist=parent_dist)

    k_local = max(1, min(k, nc - 1))
    kdt = NearestNeighbors(n_neighbors=k_local, metric="euclidean", algorithm="kd_tree")
    kdt.fit(X_cc)
    distances, neighbors = kdt.kneighbors(X_cc)

    # Definition 2: choose nearest higher-density neighbor as Big Brother.
    radius_diff = radius_cc[:, np.newaxis] - radius_cc[neighbors]
    rows, cols = np.where(radius_diff > 0)

    if len(rows) > 0:
        unique_rows, first_idx = np.unique(rows, return_index=True)
        cols = cols[first_idx]
        parent[unique_rows] = neighbors[unique_rows, cols]
        parent_dist[unique_rows] = distances[unique_rows, cols]

    no_parent_mask = parent == NO_PARENT
    if np.any(no_parent_mask):
        max_density_idx = np.where(no_parent_mask)[0]
        if len(max_density_idx) > 0:
            parent[max_density_idx[0]] = max_density_idx[0]
            parent_dist[max_density_idx[0]] = np.inf
            for idx in max_density_idx[1:]:
                parent[idx] = max_density_idx[0]
                parent_dist[idx] = np.linalg.norm(X_cc[idx] - X_cc[max_density_idx[0]])

    return BigBrotherResult(parent=parent, parent_dist=parent_dist)


def compute_big_brother(
    X: NDArray[np.float32],
    radius: Float32Array,
    components: Int32Array,
    k: int,
) -> BigBrotherResult:
    """全データに対してbig brotherを計算

    Args:
        X: 入力データ (n, d)
        radius: k近傍距離 (n,)
        components: 成分ラベル (n,)
        k: 近傍数

    Returns:
        BigBrotherResult: big brother情報
    """
    n = len(X)
    parent = np.full(n, NO_PARENT, dtype=np.int32)
    parent_dist = np.full(n, np.inf, dtype=np.float32)

    valid_components = np.unique(components[components != OUTLIER])

    for cc in valid_components:
        cc_idx = np.where(components == cc)[0]
        result_cc = compute_big_brother_for_component(X[cc_idx], radius[cc_idx], k)
        global_parent = np.where(
            result_cc.parent != NO_PARENT,
            cc_idx[result_cc.parent],
            NO_PARENT,
        )
        parent[cc_idx] = global_parent
        parent_dist[cc_idx] = result_cc.parent_dist

    return BigBrotherResult(parent=parent, parent_dist=parent_dist)
