"""Official demo script using PyO3 Rust backend.

This script demonstrates the full workflow of CPFcluster using the Rust backend:
1. Load and preprocess data
2. Run clustering with hyperparameter grid (via PyO3)
3. Cross-validate to find optimal parameters
4. Evaluate and visualize results

Note: This demo uses the paper-aligned CPF algorithm without merge post-processing.
(merge is a Python-only extension not described in the original paper)
"""

import itertools
import time
from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler

import cpfcluster_pyo3


def cross_validate(
    X: np.ndarray,
    clusterings: Dict[Tuple, np.ndarray],
    validation_index=calinski_harabasz_score,
) -> Tuple[Tuple, float]:
    """Find best parameters using validation index."""
    best_params = None
    best_score = -np.inf

    for params, labels in clusterings.items():
        unique_labels = set(labels) - {-1}
        if len(unique_labels) < 2:
            continue
        try:
            score = validation_index(X, labels)
            if score > best_score:
                best_score = score
                best_params = params
        except Exception:
            continue

    return best_params, best_score


def main() -> None:
    """Run the CPFcluster demo with the ecoli dataset using PyO3 backend."""
    start_time = time.perf_counter()

    # Load the dataset
    print("Loading ecoli dataset...")
    data = np.load("Data/ecoli.npy")
    X = data[:, :-1]
    y = data[:, -1].astype(int)  # true labels (used for evaluation, not clustering)

    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"True clusters: {len(set(y))}")

    # Normalize dataset for easier hyperparameter tuning
    X = StandardScaler().fit_transform(X)
    X = X.astype(np.float32)  # PyO3 requires float32

    # Hyperparameter grid
    k_values = [5, 10, 15]
    rho_values = [0.3, 0.5, 0.7, 0.9]
    alpha_values = [0.6, 0.8, 1.0, 1.2]
    cutoff = 1

    # Fit the model for all parameter combinations
    print("\nFitting CPFcluster (PyO3 Rust backend)...")
    clusterings: Dict[Tuple, np.ndarray] = {}

    total_combinations = len(k_values) * len(rho_values) * len(alpha_values)
    print(f"Running {total_combinations} parameter combinations...")

    fit_start = time.perf_counter()
    for k in k_values:
        for rho, alpha in itertools.product(rho_values, alpha_values):
            labels = cpfcluster_pyo3.cpf_fit(X, k, [rho], [alpha], cutoff)
            params = (k, rho, alpha)
            clusterings[params] = labels
    fit_elapsed = time.perf_counter() - fit_start
    print(f"Fitting completed in {fit_elapsed:.2f} seconds")

    # Cross-validate to find optimal parameters
    print("\nPerforming cross-validation...")
    best_params, best_score = cross_validate(
        X, clusterings, validation_index=calinski_harabasz_score
    )

    if best_params is None:
        print("Error: No valid clustering found")
        return

    print(
        f"Best Parameters: k={best_params[0]}, rho={best_params[1]:.2f}, "
        f"alpha={best_params[2]:.2f}"
    )
    print(f"Best Validation Score (Calinski-Harabasz Index): {best_score:.2f}")

    # Access cluster labels for the best parameter configuration
    best_labels = clusterings[best_params]
    n_clusters = len(set(best_labels) - {-1})
    n_outliers = np.sum(best_labels == -1)
    print(f"\nClustering result:")
    print(f"  Clusters found: {n_clusters}")
    print(f"  Outliers: {n_outliers}")

    # Evaluate clustering performance using Adjusted Rand Index (ARI)
    ari = adjusted_rand_score(y, best_labels)
    print(f"  Adjusted Rand Index (ARI): {ari:.2f}")

    elapsed = time.perf_counter() - start_time
    print(f"\nTotal execution time: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
