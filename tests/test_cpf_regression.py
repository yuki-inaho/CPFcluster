"""
Regression tests for CPFcluster algorithm.

These tests verify that the core functionality works correctly
and produces consistent results across code changes.
"""

import os
import tempfile
import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score

# Import from project root
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import CPFcluster, build_CCgraph, get_density_dists_bb, get_y
from demo_cpf_visualize import generate_data, compute_visualization_data


class TestDataGeneration:
    """Tests for synthetic data generation."""

    def test_generate_data_shape(self):
        """Test that generated data has correct shape."""
        X, y_true = generate_data(n_per_cluster=100, random_state=42)
        assert X.shape == (300, 2), f"Expected (300, 2), got {X.shape}"
        assert y_true.shape == (300,), f"Expected (300,), got {y_true.shape}"

    def test_generate_data_labels(self):
        """Test that labels are correctly assigned."""
        X, y_true = generate_data(n_per_cluster=50, random_state=42)
        assert len(set(y_true)) == 3, "Should have exactly 3 clusters"
        assert sum(y_true == 0) == 50
        assert sum(y_true == 1) == 50
        assert sum(y_true == 2) == 50

    def test_generate_data_reproducibility(self):
        """Test that same seed produces same data."""
        X1, y1 = generate_data(n_per_cluster=50, random_state=123)
        X2, y2 = generate_data(n_per_cluster=50, random_state=123)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    def test_cluster_separation(self):
        """Test that clusters are reasonably separated."""
        X, y_true = generate_data(n_per_cluster=100, random_state=42)

        # Compute cluster centers
        centers = []
        for i in range(3):
            centers.append(X[y_true == i].mean(axis=0))

        # Check that at least one cluster is far from others
        distances = []
        for i in range(3):
            for j in range(i + 1, 3):
                dist = np.linalg.norm(centers[i] - centers[j])
                distances.append(dist)

        # Max distance should be significantly larger (cluster 3 is far)
        assert max(distances) > 2.0, "One cluster should be far from others"


class TestVisualizationData:
    """Tests for compute_visualization_data function."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample data for tests."""
        X, _ = generate_data(n_per_cluster=50, random_state=42)
        X = StandardScaler().fit_transform(X)
        return X

    def test_adjacency_list_not_empty(self, sample_data):
        """Test that adjacency list is not empty."""
        adj, rho, delta, big_brother = compute_visualization_data(sample_data, k=5)
        assert len(adj) > 0, "Adjacency list should not be empty"

    def test_density_positive(self, sample_data):
        """Test that all density values are positive."""
        adj, rho, delta, big_brother = compute_visualization_data(sample_data, k=5)
        assert np.all(rho > 0), "All density values should be positive"

    def test_delta_non_negative(self, sample_data):
        """Test that all delta values are non-negative."""
        adj, rho, delta, big_brother = compute_visualization_data(sample_data, k=5)
        assert np.all(delta >= 0), "All delta values should be non-negative"

    def test_big_brother_valid_indices(self, sample_data):
        """Test that big_brother contains valid indices."""
        adj, rho, delta, big_brother = compute_visualization_data(sample_data, k=5)
        N = len(sample_data)

        for i, bb in enumerate(big_brother):
            assert bb == -1 or (0 <= bb < N), f"Invalid big_brother index at {i}: {bb}"

    def test_highest_density_has_no_parent(self, sample_data):
        """Test that point with highest density has no big brother."""
        adj, rho, delta, big_brother = compute_visualization_data(sample_data, k=5)
        highest_density_idx = np.argmax(rho)
        assert big_brother[highest_density_idx] == -1, (
            "Highest density point should have no big brother"
        )


class TestCPFcluster:
    """Tests for CPFcluster class."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample data for tests."""
        X, y_true = generate_data(n_per_cluster=100, random_state=42)
        X = StandardScaler().fit_transform(X)
        return X, y_true

    def test_cpf_fit_returns_clusterings(self, sample_data):
        """Test that fit() returns clusterings dictionary."""
        X, _ = sample_data
        cpf = CPFcluster(min_samples=10, rho=[0.4], alpha=[1.0], n_jobs=1)
        result = cpf.fit(X)

        assert hasattr(cpf, "clusterings")
        assert len(cpf.clusterings) > 0

    def test_cpf_labels_shape(self, sample_data):
        """Test that labels have correct shape."""
        X, _ = sample_data
        cpf = CPFcluster(min_samples=10, rho=[0.4], alpha=[1.0], n_jobs=1)
        cpf.fit(X)

        labels = list(cpf.clusterings.values())[0]
        assert labels.shape == (len(X),), f"Labels shape mismatch"

    def test_cpf_finds_clusters(self, sample_data):
        """Test that CPF finds at least 2 clusters."""
        X, _ = sample_data
        cpf = CPFcluster(min_samples=10, rho=[0.4], alpha=[1.0], n_jobs=1)
        cpf.fit(X)

        labels = list(cpf.clusterings.values())[0]
        n_clusters = len(set(labels) - {-1})  # Exclude outliers
        assert n_clusters >= 2, f"Expected at least 2 clusters, got {n_clusters}"

    def test_cpf_reasonable_ari(self, sample_data):
        """Test that CPF achieves reasonable ARI on synthetic data."""
        X, y_true = sample_data
        cpf = CPFcluster(min_samples=10, rho=[0.4], alpha=[1.0], n_jobs=1)
        cpf.fit(X)

        labels = list(cpf.clusterings.values())[0]
        ari = adjusted_rand_score(y_true, labels)

        # Should achieve at least 0.5 ARI on well-separated clusters
        assert ari > 0.5, f"ARI too low: {ari}"

    def test_cpf_reproducibility(self, sample_data):
        """Test that CPF produces same results with same input."""
        X, _ = sample_data

        cpf1 = CPFcluster(min_samples=10, rho=[0.4], alpha=[1.0], n_jobs=1)
        cpf1.fit(X)
        labels1 = list(cpf1.clusterings.values())[0]

        cpf2 = CPFcluster(min_samples=10, rho=[0.4], alpha=[1.0], n_jobs=1)
        cpf2.fit(X)
        labels2 = list(cpf2.clusterings.values())[0]

        np.testing.assert_array_equal(labels1, labels2)


class TestBuildCCGraph:
    """Tests for build_CCgraph helper function."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample data for tests."""
        X, _ = generate_data(n_per_cluster=50, random_state=42)
        X = StandardScaler().fit_transform(X)
        return X

    def test_components_shape(self, sample_data):
        """Test that components array has correct shape."""
        X = sample_data
        components, CCmat, knn_radius = build_CCgraph(
            X, min_samples=5, cutoff=1, n_jobs=1
        )
        assert components.shape == (len(X),)

    def test_ccmat_shape(self, sample_data):
        """Test that CCmat has correct shape."""
        X = sample_data
        components, CCmat, knn_radius = build_CCgraph(
            X, min_samples=5, cutoff=1, n_jobs=1
        )
        assert CCmat.shape == (len(X), len(X))

    def test_knn_radius_positive(self, sample_data):
        """Test that knn_radius values are positive for non-outliers."""
        X = sample_data
        components, CCmat, knn_radius = build_CCgraph(
            X, min_samples=5, cutoff=1, n_jobs=1
        )

        # Non-NaN values should be positive
        valid_radii = knn_radius[~np.isnan(knn_radius)]
        assert np.all(valid_radii > 0)


class TestPlotGeneration:
    """Tests for plot generation (file output)."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample data for tests."""
        X, _ = generate_data(n_per_cluster=50, random_state=42)
        X = StandardScaler().fit_transform(X)
        return X

    def test_plot_file_created(self, sample_data):
        """Test that plot file is created."""
        import matplotlib

        matplotlib.use("Agg")  # Non-interactive backend for testing
        import matplotlib.pyplot as plt

        from demo_cpf_visualize import plot_four_panels

        X = sample_data

        # Run CPFcluster
        cpf = CPFcluster(min_samples=5, rho=[0.4], alpha=[1.0], n_jobs=1)
        cpf.fit(X)
        labels = list(cpf.clusterings.values())[0]

        # Compute visualization data
        adj, rho, delta, big_brother = compute_visualization_data(X, k=5)

        # Create temp file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            output_path = f.name

        try:
            # Monkey-patch plt.show to avoid display
            original_show = plt.show
            plt.show = lambda: None

            plot_four_panels(X, labels, adj, rho, delta, big_brother, output_path)

            plt.show = original_show

            assert os.path.exists(output_path), "Plot file should be created"
            assert os.path.getsize(output_path) > 0, "Plot file should not be empty"
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)


# Regression test with fixed expected values
class TestRegressionValues:
    """
    Regression tests with fixed expected values.
    These ensure consistency across code changes.
    """

    def test_fixed_data_clustering(self):
        """Test clustering on fixed data produces expected number of clusters."""
        X, y_true = generate_data(n_per_cluster=100, random_state=42)
        X = StandardScaler().fit_transform(X)

        cpf = CPFcluster(min_samples=10, rho=[0.4], alpha=[1.0], n_jobs=1)
        cpf.fit(X)
        labels = list(cpf.clusterings.values())[0]

        n_clusters = len(set(labels) - {-1})
        n_outliers = sum(labels == -1)

        # These values are based on the current implementation
        # Update if algorithm intentionally changes
        assert 2 <= n_clusters <= 6, f"Unexpected number of clusters: {n_clusters}"
        assert n_outliers < 20, f"Too many outliers: {n_outliers}"

    def test_visualization_data_consistency(self):
        """Test that visualization data is consistent."""
        X, _ = generate_data(n_per_cluster=50, random_state=42)
        X = StandardScaler().fit_transform(X)

        adj1, rho1, delta1, bb1 = compute_visualization_data(X, k=5)
        adj2, rho2, delta2, bb2 = compute_visualization_data(X, k=5)

        np.testing.assert_array_almost_equal(rho1, rho2)
        np.testing.assert_array_almost_equal(delta1, delta2)
        np.testing.assert_array_equal(bb1, bb2)
