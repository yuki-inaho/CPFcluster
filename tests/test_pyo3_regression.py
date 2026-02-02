"""PyO3 wrapper regression tests."""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score

import cpfcluster_pyo3
from demo_cpf_visualize import generate_data


def test_pyo3_fit_labels_shape():
    X, _ = generate_data(n_per_cluster=50, random_state=42)
    X = StandardScaler().fit_transform(X)
    labels = cpfcluster_pyo3.cpf_fit(X.astype(np.float32), 10, [0.4], [1.0], 1)
    assert labels.shape == (len(X),)


def test_pyo3_ari_reasonable():
    X, y_true = generate_data(n_per_cluster=100, random_state=42)
    X = StandardScaler().fit_transform(X)
    labels = cpfcluster_pyo3.cpf_fit(X.astype(np.float32), 10, [0.4], [1.0], 1)
    ari = adjusted_rand_score(y_true, labels)
    assert ari > 0.5
