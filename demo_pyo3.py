#!/usr/bin/env python
"""Demo script using PyO3 Rust wrapper for CPFcluster.

Compares performance with pure Python implementation.
"""

import argparse
import os
import time

import numpy as np
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

import cpfcluster_pyo3
from demo_cpf_visualize import generate_data, compute_visualization_data, plot_four_panels
from src import CPFcluster


def run_pyo3_clustering(X: np.ndarray, min_samples: int, rho: list, alpha: list, cutoff: int):
    """Run clustering using PyO3 Rust backend."""
    start = time.perf_counter()
    labels = cpfcluster_pyo3.cpf_fit(
        X.astype(np.float32), min_samples, rho, alpha, cutoff
    )
    elapsed = time.perf_counter() - start
    return labels, elapsed


def run_python_clustering(X: np.ndarray, min_samples: int, rho: list, alpha: list, cutoff: int):
    """Run clustering using pure Python backend."""
    start = time.perf_counter()
    cpf = CPFcluster(min_samples=min_samples, rho=rho, alpha=alpha, cutoff=cutoff, n_jobs=1)
    cpf.fit(X)
    labels = list(cpf.clusterings.values())[0]
    elapsed = time.perf_counter() - start
    return labels, elapsed


def main():
    parser = argparse.ArgumentParser(description="PyO3 CPFcluster Demo")
    parser.add_argument("--n-points", type=int, default=150, help="Points per cluster")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--k", type=int, default=10, help="Number of neighbors")
    parser.add_argument("--output", "-o", type=str, default="pyo3_demo_result.png",
                        help="Output filename (saved in outputs/)")
    parser.add_argument("--benchmark-only", action="store_true",
                        help="Run benchmark without visualization")
    args = parser.parse_args()

    os.makedirs("outputs", exist_ok=True)

    # Generate data
    print(f"Generating data: {args.n_points} points per cluster (seed={args.seed})")
    X, y_true = generate_data(n_per_cluster=args.n_points, random_state=args.seed)
    X = StandardScaler().fit_transform(X)

    # Parameters
    min_samples = args.k
    rho = [0.4]
    alpha = [1.0]
    cutoff = 1

    # Run PyO3 (Rust) version
    print("\n--- PyO3 (Rust) Backend ---")
    labels_pyo3, time_pyo3 = run_pyo3_clustering(X, min_samples, rho, alpha, cutoff)
    n_clusters_pyo3 = len(set(labels_pyo3) - {-1})
    n_outliers_pyo3 = np.sum(labels_pyo3 == -1)
    ari_pyo3 = adjusted_rand_score(y_true, labels_pyo3)
    print(f"  Time: {time_pyo3:.4f} seconds")
    print(f"  Clusters: {n_clusters_pyo3}")
    print(f"  Outliers: {n_outliers_pyo3}")
    print(f"  ARI: {ari_pyo3:.4f}")

    # Run Python version
    print("\n--- Pure Python Backend ---")
    labels_py, time_py = run_python_clustering(X, min_samples, rho, alpha, cutoff)
    n_clusters_py = len(set(labels_py) - {-1})
    n_outliers_py = np.sum(labels_py == -1)
    ari_py = adjusted_rand_score(y_true, labels_py)
    print(f"  Time: {time_py:.4f} seconds")
    print(f"  Clusters: {n_clusters_py}")
    print(f"  Outliers: {n_outliers_py}")
    print(f"  ARI: {ari_py:.4f}")

    # Comparison
    print("\n--- Comparison ---")
    speedup = time_py / time_pyo3 if time_pyo3 > 0 else float('inf')
    print(f"  Speedup: {speedup:.2f}x")
    label_match = np.array_equal(labels_pyo3, labels_py)
    print(f"  Labels match: {label_match}")

    if not args.benchmark_only:
        # Visualization using PyO3 labels
        print("\nGenerating visualization...")
        adj, rho_vis, delta, big_brother = compute_visualization_data(X, k=args.k)
        output_path = os.path.join("outputs", args.output)
        plot_four_panels(X, labels_pyo3, adj, rho_vis, delta, big_brother, output_path)

    print("\nDone!")


if __name__ == "__main__":
    main()
