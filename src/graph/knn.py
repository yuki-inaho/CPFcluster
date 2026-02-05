"""kNN探索モジュール"""

from dataclasses import dataclass
from typing import Any, Literal
import numpy as np
from numpy.typing import NDArray
import faiss
from sklearn.neighbors import NearestNeighbors


@dataclass(frozen=True)
class KnnResult:
    """kNN探索結果"""

    indices: NDArray[np.int32]  # shape: (n, k)
    distances: NDArray[np.float32]  # shape: (n, k)
    radius: NDArray[np.float32]  # shape: (n,) k番目近傍距離


KnnBackend = Literal["kd", "brute"]


def _ensure_self_neighbor(
    indices: NDArray[np.int32],
    distances: NDArray[np.float32],
) -> None:
    """Ensure self index is at position 0 with distance 0.

    sklearn's kneighbors should include self when querying training data,
    but duplicates can reorder neighbors. This function enforces the
    fastCPF invariant: self at index 0.
    """
    n, k = indices.shape
    if k == 0:
        return
    for i in range(n):
        if indices[i, 0] == i:
            distances[i, 0] = 0.0
            continue
        pos = np.where(indices[i] == i)[0]
        if pos.size > 0:
            j = int(pos[0])
            indices[i, 0], indices[i, j] = indices[i, j], indices[i, 0]
            distances[i, 0], distances[i, j] = 0.0, distances[i, 0]
        else:
            # Fallback: force self into the first slot.
            indices[i, 0] = i
            distances[i, 0] = 0.0


def knn_search_sklearn(
    X: NDArray[np.float32],
    k: int,
    backend: KnnBackend,
) -> KnnResult:
    """sklearnによるkNN探索（KD-tree / Brute）

    Args:
        X: 入力データ (n, d)
        k: 近傍数（selfを含む）
        backend: "kd" または "brute"

    Returns:
        KnnResult: 探索結果（Euclidean距離）
    """
    n = X.shape[0]
    if n == 0:
        empty = np.empty((0, 0), dtype=np.int32)
        return KnnResult(
            indices=empty,
            distances=np.empty((0, 0), dtype=np.float32),
            radius=np.empty((0,), dtype=np.float32),
        )
    if k <= 0:
        raise ValueError("k must be positive")
    k_eff = min(k, n)
    algorithm = "kd_tree" if backend == "kd" else "brute"
    nbrs = NearestNeighbors(
        n_neighbors=k_eff,
        metric="euclidean",
        algorithm=algorithm,
    ).fit(X)
    distances, indices = nbrs.kneighbors(X)
    indices = indices.astype(np.int32)
    distances = distances.astype(np.float32)
    _ensure_self_neighbor(indices, distances)
    radius = distances[:, k_eff - 1].astype(np.float32)
    return KnnResult(indices=indices, distances=distances, radius=radius)


def knn_search(
    X: NDArray[np.float32],
    k: int,
    backend: KnnBackend = "kd",
) -> KnnResult:
    """kNN探索（KD-tree / Brute の切替）"""
    if backend not in ("kd", "brute"):
        raise ValueError(f"Unsupported knn_backend: '{backend}'. Use 'kd' or 'brute'.")
    return knn_search_sklearn(X, k, backend)


def knn_search_faiss(X: NDArray[np.float32], k: int) -> KnnResult:
    """FAISSによるkNN探索

    Args:
        X: 入力データ (n, d)
        k: 近傍数

    Returns:
        KnnResult: 探索結果
    """
    n, d = X.shape
    index: Any = faiss.IndexFlatL2(d)  # type: ignore[possibly-missing-attribute]
    index.add(X)  # type: ignore[missing-argument]
    distances, indices = index.search(X, k)  # type: ignore[missing-argument]
    return KnnResult(
        indices=indices.astype(np.int32),
        distances=distances.astype(np.float32),
        radius=distances[:, k - 1].astype(np.float32),
    )
