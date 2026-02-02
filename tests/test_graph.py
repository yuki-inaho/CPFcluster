"""graph層の単体テスト"""

import numpy as np
from src.graph import (
    knn_search_faiss,
    build_knn_adjacency,
    make_mutual,
    extract_components,
    filter_by_edge_count,
)
from src.types import OUTLIER


class TestKnnSearch:
    def test_knn_search_faiss_shape(self):
        X = np.random.randn(100, 2).astype(np.float32)
        result = knn_search_faiss(X, k=5)
        assert result.indices.shape == (100, 5)
        assert result.distances.shape == (100, 5)
        assert result.radius.shape == (100,)

    def test_knn_search_faiss_self_included(self):
        X = np.random.randn(50, 3).astype(np.float32)
        result = knn_search_faiss(X, k=3)
        assert np.allclose(result.distances[:, 0], 0, atol=1e-6)


class TestAdjacency:
    def test_build_knn_adjacency_shape(self):
        X = np.random.randn(30, 2).astype(np.float32)
        knn = knn_search_faiss(X, k=5)
        adj = build_knn_adjacency(knn)
        assert adj.shape == (30, 30)

    def test_make_mutual_symmetric(self):
        X = np.random.randn(20, 2).astype(np.float32)
        knn = knn_search_faiss(X, k=3)
        adj = build_knn_adjacency(knn)
        mutual = make_mutual(adj)
        diff = mutual - mutual.T
        assert diff.nnz == 0


class TestComponents:
    def test_extract_components(self):
        X = np.random.randn(50, 2).astype(np.float32)
        knn = knn_search_faiss(X, k=5)
        adj = make_mutual(build_knn_adjacency(knn))
        components = extract_components(adj)
        assert components.shape == (50,)
        assert components.dtype == np.int32

    def test_filter_by_edge_count(self):
        X = np.random.randn(50, 2).astype(np.float32)
        knn = knn_search_faiss(X, k=5)
        adj = make_mutual(build_knn_adjacency(knn))
        components = extract_components(adj)
        filtered = filter_by_edge_count(adj, components, min_edges=1)
        assert filtered.dtype == np.int32
        assert np.all((filtered >= 0) | (filtered == OUTLIER))
