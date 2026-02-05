"""fastCPF parity tests for CPFcluster.fit_single API."""

import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

from src.cpf import CPFcluster


class TestCPFParity:
    """Test suite for CPFcluster fastCPF-compatible API."""

    @pytest.fixture
    def sample_data(self):
        X, y_true = make_blobs(
            n_samples=300,
            n_features=2,
            centers=3,
            cluster_std=0.5,
            random_state=42,
        )
        X = StandardScaler().fit_transform(X).astype(np.float32)
        return X, y_true

    def test_import(self):
        assert CPFcluster is not None

    def test_basic_clustering(self, sample_data):
        X, _ = sample_data
        model = CPFcluster(min_samples=10, rho=[0.4], alpha=[1.0])
        model.fit_single(X)
        labels = model.labels_
        assert len(labels) == len(X)
        assert model.n_clusters_ >= 1

    def test_labels_shape(self, sample_data):
        X, _ = sample_data
        model = CPFcluster(min_samples=10)
        model.fit_single(X)
        assert model.labels_.shape == (len(X),)
        assert model.labels_.dtype == np.int32

    def test_intermediate_results(self, sample_data):
        X, _ = sample_data
        k = 10
        model = CPFcluster(min_samples=k)
        model.fit_single(X)
        assert model.knn_indices_.shape == (len(X), k)
        assert model.knn_distances_.shape == (len(X), k)
        assert model.knn_radius_.shape == (len(X),)
        assert model.components_.shape == (len(X),)
        assert model.big_brother_.shape == (len(X),)
        assert model.big_brother_dist_.shape == (len(X),)
        assert model.peak_score_.shape == (len(X),)

    def test_unfitted_model_raises(self):
        model = CPFcluster()
        with pytest.raises(RuntimeError, match="not been fitted"):
            _ = model.labels_
        with pytest.raises(RuntimeError, match="not been fitted"):
            _ = model.n_clusters_

    def test_knn_backends(self, sample_data):
        X, _ = sample_data
        model_kd = CPFcluster(min_samples=10, knn_backend="kd")
        model_kd.fit_single(X)

        model_brute = CPFcluster(min_samples=10, knn_backend="brute")
        model_brute.fit_single(X)

        ari = adjusted_rand_score(model_kd.labels_, model_brute.labels_)
        assert ari > 0.95, f"ARI between backends: {ari}"

    def test_invalid_backend_raises(self):
        with pytest.raises(ValueError, match="Unsupported knn_backend"):
            CPFcluster(knn_backend="invalid")

    def test_density_method_median(self, sample_data):
        X, _ = sample_data
        model = CPFcluster(min_samples=10, density_method="median")
        model.fit_single(X)
        assert model.labels_.shape == (len(X),)

    def test_density_method_mean(self, sample_data):
        X, _ = sample_data
        model = CPFcluster(min_samples=10, density_method="mean")
        model.fit_single(X)
        assert model.labels_.shape == (len(X),)

    def test_invalid_density_method_raises(self):
        with pytest.raises(ValueError, match="Unsupported density_method"):
            CPFcluster(density_method="invalid")

    def test_clustering_quality(self, sample_data):
        X, y_true = sample_data
        model = CPFcluster(min_samples=10, rho=[0.4])
        model.fit_single(X)
        ari = adjusted_rand_score(y_true, model.labels_)
        assert ari > 0.8, f"ARI too low: {ari}"

    def test_outlier_count(self, sample_data):
        X, _ = sample_data
        model = CPFcluster(min_samples=10, cutoff=1)
        model.fit_single(X)
        expected_outliers = np.sum(model.labels_ == -1)
        assert model.n_outliers_ == expected_outliers

    def test_n_clusters_count(self, sample_data):
        X, _ = sample_data
        model = CPFcluster(min_samples=10)
        model.fit_single(X)
        unique_labels = set(model.labels_) - {-1}
        assert model.n_clusters_ == len(unique_labels)

    def test_empty_input_handling(self):
        X = np.random.randn(5, 2).astype(np.float32)
        model = CPFcluster(min_samples=3)
        model.fit_single(X)
        assert len(model.labels_) == 5
