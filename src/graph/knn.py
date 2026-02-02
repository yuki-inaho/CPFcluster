"""kNN探索モジュール"""

from dataclasses import dataclass
from typing import Any
import numpy as np
from numpy.typing import NDArray
import faiss


@dataclass(frozen=True)
class KnnResult:
    """kNN探索結果"""

    indices: NDArray[np.int32]  # shape: (n, k)
    distances: NDArray[np.float32]  # shape: (n, k)
    radius: NDArray[np.float32]  # shape: (n,) k番目近傍距離


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
