"""core層の単体テスト"""

import numpy as np
from src.core import compute_peak_score, compute_big_brother
from src.graph import (
    knn_search_faiss,
    build_knn_adjacency,
    make_mutual,
    extract_components,
)
from src.types import NO_PARENT


class TestPeakScore:
    def test_compute_peak_score_basic(self):
        parent_dist = np.array([1.0, 2.0, 0.5], dtype=np.float32)
        radius = np.array([0.5, 1.0, 0.25], dtype=np.float32)
        peaked = compute_peak_score(parent_dist, radius)
        expected = np.array([2.0, 2.0, 2.0], dtype=np.float32)
        np.testing.assert_allclose(peaked, expected)

    def test_compute_peak_score_zero_radius(self):
        parent_dist = np.array([1.0, 0.0], dtype=np.float32)
        radius = np.array([0.0, 0.0], dtype=np.float32)
        peaked = compute_peak_score(parent_dist, radius)
        assert peaked[0] == np.inf
        assert peaked[1] == np.inf


class TestBigBrother:
    def test_compute_big_brother_shape(self):
        np.random.seed(42)
        X = np.random.randn(100, 2).astype(np.float32)
        knn = knn_search_faiss(X, k=10)
        adj = make_mutual(build_knn_adjacency(knn))
        components = extract_components(adj)
        result = compute_big_brother(X, knn.radius, components, k=10)
        assert result.parent.shape == (100,)
        assert result.parent_dist.shape == (100,)
        assert result.parent.dtype == np.int32

    def test_compute_big_brother_valid_indices(self):
        np.random.seed(42)
        X = np.random.randn(50, 2).astype(np.float32)
        knn = knn_search_faiss(X, k=5)
        adj = make_mutual(build_knn_adjacency(knn))
        components = extract_components(adj)
        result = compute_big_brother(X, knn.radius, components, k=5)
        valid = (result.parent == NO_PARENT) | (
            (result.parent >= 0) & (result.parent < 50)
        )
        assert np.all(valid)
