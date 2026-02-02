#!/usr/bin/env python
"""Simple demo of CPFcluster with synthetic Gaussian data."""

import argparse
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score
from src import CPFcluster, OutlierMethod


def generate_data(n_per_cluster=100, random_state=42):
    """Generate 3 Gaussian clusters (2 close, 1 far)."""
    np.random.seed(random_state)

    # Cluster 1: center at (0, 0)
    c1 = np.random.randn(n_per_cluster, 2) * 0.5 + [0, 0]

    # Cluster 2: close to cluster 1, center at (1.5, 0)
    c2 = np.random.randn(n_per_cluster, 2) * 0.5 + [1.5, 0]

    # Cluster 3: far away, center at (5, 5)
    c3 = np.random.randn(n_per_cluster, 2) * 0.5 + [5, 5]

    X = np.vstack([c1, c2, c3])
    y_true = np.array([0] * n_per_cluster + [1] * n_per_cluster + [2] * n_per_cluster)

    return X, y_true


def main():
    parser = argparse.ArgumentParser(description="CPFcluster simple demo")
    parser.add_argument("--plot", action="store_true", help="Show scatter plot")
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="simple_demo_result.png",
        help="Output filename for plot (saved in outputs/)",
    )
    parser.add_argument("--n-points", type=int, default=100, help="Points per cluster")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--outlier-method",
        type=str,
        choices=["component_size", "edge_count"],
        default="edge_count",
        help="Outlier detection method: 'edge_count' (paper-aligned, default) or 'component_size' (original)",
    )
    parser.add_argument(
        "--cutoff",
        type=int,
        default=1,
        help="Cutoff threshold for outlier detection",
    )
    args = parser.parse_args()

    # Map string to enum
    outlier_method = (
        OutlierMethod.EDGE_COUNT
        if args.outlier_method == "edge_count"
        else OutlierMethod.COMPONENT_SIZE
    )

    # Generate data
    X, y_true = generate_data(n_per_cluster=args.n_points, random_state=args.seed)
    X = StandardScaler().fit_transform(X)

    # Run CPFcluster
    cpf = CPFcluster(
        min_samples=10,
        rho=[0.4],
        alpha=[1.0],
        n_jobs=1,
        cutoff=args.cutoff,
        outlier_method=outlier_method,
    )
    cpf.fit(X)

    # Get labels
    labels = list(cpf.clusterings.values())[0]

    # CLI output
    print(f"Data: {len(X)} points, 2 dimensions")
    print(f"Outlier method: {args.outlier_method} (cutoff={args.cutoff})")
    print("True clusters: 3")
    print(f"Predicted clusters: {len(set(labels) - {-1})}")
    print(f"Outliers: {sum(labels == -1)}")
    print(f"ARI: {adjusted_rand_score(y_true, labels):.3f}")

    # Plot if requested
    if args.plot:
        import matplotlib.pyplot as plt

        # Ensure outputs directory exists
        os.makedirs("outputs", exist_ok=True)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].scatter(X[:, 0], X[:, 1], c=y_true, cmap="viridis", s=20)
        axes[0].set_title("True Labels")

        axes[1].scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", s=20)
        axes[1].set_title(f"CPFcluster ({args.outlier_method}, cutoff={args.cutoff})")

        plt.tight_layout()

        output_path = os.path.join("outputs", args.output)
        plt.savefig(output_path, dpi=150)
        print(f"Plot saved to: {output_path}")
        plt.show()


if __name__ == "__main__":
    main()
