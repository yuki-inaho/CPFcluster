#!/usr/bin/env python
"""
Demo of CPFcluster reproducing the 4-step visualization process.

This script visualizes the internal workings of the CPFcluster algorithm:
1. Construct the Mutual k-NN Graph
2. Compute the Peak-Finding Criterion
3. Assess Potential Centers
4. Assign Remaining Instances
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from core import CPFcluster


def generate_data(n_per_cluster=100, random_state=42):
    """Generate 3 Gaussian clusters (2 close, 1 far)."""
    np.random.seed(random_state)
    # Cluster 1 (Top right - isolated)
    c1 = np.random.randn(n_per_cluster, 2) * 0.3 + [1.5, 1.5]
    # Cluster 2 (Bottom left)
    c2 = np.random.randn(n_per_cluster, 2) * 0.3 + [-1.0, -0.8]
    # Cluster 3 (Bottom center - close to cluster 2, bridge structure)
    c3 = np.random.randn(n_per_cluster, 2) * 0.3 + [-0.2, -0.7]

    X = np.vstack([c1, c2, c3])
    y_true = np.array([0] * n_per_cluster + [1] * n_per_cluster + [2] * n_per_cluster)
    return X, y_true


def compute_visualization_data(X, k=10):
    """
    Replicates the internal steps of CPF to gather data for plotting:
    1. Mutual k-NN Graph
    2. Density (rho) and Delta (distance to higher density)
    3. Big Brother (parent pointers)
    """
    N = X.shape[0]
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X)
    distances, indices = nbrs.kneighbors(X)

    # --- Step 1: Mutual k-NN Graph ---
    adj = []
    for i in range(N):
        for idx in indices[i, 1:]:  # skip self
            # Check mutuality: i connected to idx IF idx in i's k-NN AND i in idx's k-NN
            if i in indices[idx, 1:]:
                adj.append([i, idx])

    # --- Step 2: Density & Delta ---
    # Density rho: inverse of distance to k-th neighbor
    k_dist = distances[:, -1]
    rho = 1.0 / (k_dist + 1e-10)

    # Sort points by density descending
    sorted_idxs = np.argsort(rho)[::-1]

    # Compute Delta and Big Brother
    delta = np.zeros(N)
    big_brother = np.full(N, -1, dtype=int)

    # Max distance in dataset for the global peak
    max_dist = np.max(np.linalg.norm(X[:, None] - X, axis=2))

    for i in range(N):
        idx = sorted_idxs[i]
        higher_density_candidates = sorted_idxs[:i]

        if len(higher_density_candidates) == 0:
            delta[idx] = max_dist
            continue

        # Find nearest neighbor among those with higher density
        dists_to_higher = np.linalg.norm(X[higher_density_candidates] - X[idx], axis=1)
        min_pos = np.argmin(dists_to_higher)
        delta[idx] = dists_to_higher[min_pos]
        big_brother[idx] = higher_density_candidates[min_pos]

    return adj, rho, delta, big_brother


def plot_four_panels(X, labels, adj, rho, delta, big_brother, output_file):
    """Draws the 4-panel figure matching the CPFcluster paper style."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    point_size = 20
    edge_color = "#D3D3D3"

    # --- Panel 1: Construct the Mutual k-NN Graph ---
    ax = axes[0, 0]
    ax.set_title("1. Construct the Mutual k-NN Graph.", fontweight="bold", loc="left")

    # Draw edges
    lines = [[X[u], X[v]] for u, v in adj]
    lc = LineCollection(lines, colors=edge_color, linewidths=0.5, alpha=0.6)
    ax.add_collection(lc)

    # Draw nodes
    ax.scatter(X[:, 0], X[:, 1], c="gray", s=point_size, alpha=0.8, edgecolors="none")
    ax.autoscale()
    ax.set_xticks([])
    ax.set_yticks([])

    # --- Panel 2: Compute Peak-Finding Criterion ---
    ax = axes[0, 1]
    ax.set_title(
        "2. Compute the Peak-Finding Criterion.", fontweight="bold", loc="left"
    )

    # Background edges
    ax.add_collection(
        LineCollection(lines, colors=edge_color, linewidths=0.3, alpha=0.3)
    )

    # Nodes: Size = Delta, Color = Density
    s_norm = (delta / delta.max()) * 200 + 10
    ax.scatter(
        X[:, 0],
        X[:, 1],
        c=rho,
        cmap="magma_r",
        s=s_norm,
        alpha=0.9,
        edgecolors="black",
        linewidth=0.5,
    )
    ax.autoscale()
    ax.set_xticks([])
    ax.set_yticks([])

    # --- Panel 3: Assess Potential Centers ---
    ax = axes[1, 0]
    ax.set_title("3. Assess Potential Centers.", fontweight="bold", loc="left")

    # Identify Centers (Heuristic: High Rho * Delta)
    gamma = rho * delta
    center_indices = np.argsort(gamma)[-3:]

    # Plot faint background points
    ax.scatter(X[:, 0], X[:, 1], c="gray", s=10, alpha=0.3)

    # Plot Centers
    ax.scatter(
        X[center_indices, 0],
        X[center_indices, 1],
        marker="^",
        c="yellow",
        s=150,
        edgecolors="black",
        linewidth=1.5,
        zorder=10,
    )

    # Highlight core connections
    lc_core = LineCollection(lines, colors="purple", linewidths=0.8, alpha=0.2)
    ax.add_collection(lc_core)
    ax.autoscale()
    ax.set_xticks([])
    ax.set_yticks([])

    # --- Panel 4: Assign Remaining Instances ---
    ax = axes[1, 1]
    ax.set_title("4. Assign Remaining Instances.", fontweight="bold", loc="left")

    # Plot clusters with assigned colors
    ax.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", s=30, alpha=0.8)

    # Draw Centers again
    ax.scatter(
        X[center_indices, 0],
        X[center_indices, 1],
        marker="^",
        c="yellow",
        s=150,
        edgecolors="black",
        linewidth=1.5,
        zorder=10,
    )

    # Draw Assignment Path (trace from lowest density point to center)
    start_node = np.argsort(rho)[0]
    path = [start_node]
    curr = start_node
    for _ in range(20):  # Trace up to 20 steps
        parent = big_brother[curr]
        if parent == -1 or parent == curr:
            break
        path.append(parent)
        curr = parent

    # Draw arrow path
    if len(path) > 1:
        path_coords = X[path]
        ax.plot(path_coords[:, 0], path_coords[:, 1], c="gold", linewidth=3, zorder=20)
        # Add arrow head at the end
        if len(path_coords) > 1:
            dx = path_coords[-1, 0] - path_coords[-2, 0]
            dy = path_coords[-1, 1] - path_coords[-2, 1]
            ax.annotate(
                "",
                xy=(path_coords[-1, 0], path_coords[-1, 1]),
                xytext=(path_coords[-2, 0], path_coords[-2, 1]),
                arrowprops=dict(arrowstyle="->", color="gold", lw=3),
                zorder=20,
            )

    ax.autoscale()
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"Visualization saved to: {output_file}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="CPFcluster Process Visualization")
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="cpf_process_viz.png",
        help="Output filename (saved in outputs/)",
    )
    parser.add_argument("--n-points", type=int, default=150, help="Points per cluster")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--k", type=int, default=10, help="Number of neighbors for k-NN graph"
    )
    args = parser.parse_args()

    # Ensure outputs directory exists
    os.makedirs("outputs", exist_ok=True)
    output_path = os.path.join("outputs", args.output)

    # 1. Generate Data
    print("Generating synthetic data...")
    X, y_true = generate_data(n_per_cluster=args.n_points, random_state=args.seed)
    X = StandardScaler().fit_transform(X)

    # 2. Run CPFcluster to get final labels
    print("Running CPFcluster...")
    cpf = CPFcluster(min_samples=args.k, rho=[0.4], alpha=[1.0], n_jobs=1)
    cpf.fit(X)

    if hasattr(cpf, "clusterings") and len(cpf.clusterings) > 0:
        labels = list(cpf.clusterings.values())[0]
    else:
        print("Warning: Could not extract labels from CPFcluster.")
        labels = np.zeros(len(X))

    # 3. Calculate Visualization Data
    print("Computing visualization data...")
    adj, rho, delta, big_brother = compute_visualization_data(X, k=args.k)

    # 4. Plot
    print("Generating 4-panel visualization...")
    plot_four_panels(X, labels, adj, rho, delta, big_brother, output_path)

    # CLI summary
    print(f"\nSummary:")
    print(f"  Data: {len(X)} points, 2 dimensions")
    print(f"  k (neighbors): {args.k}")
    print(f"  Predicted clusters: {len(set(labels) - {-1})}")
    print(f"  Outliers: {sum(labels == -1)}")


if __name__ == "__main__":
    main()
