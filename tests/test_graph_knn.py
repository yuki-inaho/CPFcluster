"""graph層 kNN探索の単体テスト"""

import numpy as np
from src.graph.knn import knn_search_faiss


def test_knn_search_faiss_shape():
    X = np.random.randn(100, 2).astype(np.float32)
    result = knn_search_faiss(X, k=5)
    assert result.indices.shape == (100, 5)
    assert result.distances.shape == (100, 5)
    assert result.radius.shape == (100,)
