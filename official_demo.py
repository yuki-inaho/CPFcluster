"""Official demo script for CPFcluster.

This script demonstrates the full workflow of CPFcluster:
1. Load and preprocess data
2. Initialize CPFcluster with hyperparameter grid
3. Fit the model
4. Cross-validate to find optimal parameters
5. Evaluate and visualize results
"""

import time

import numpy as np
from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler

from src import CPFcluster, OutlierMethod


def main() -> None:
    """Run the CPFcluster demo with the ecoli dataset."""
    start_time = time.perf_counter()

    # Load the dataset
    data = np.load("Data/ecoli.npy")
    X = data[:, :-1]
    y = data[:, -1]  # true labels (used for evaluation, not clustering)

    # Normalize dataset for easier hyperparameter tuning
    X = StandardScaler().fit_transform(X)

    # Initialize CPFcluster with hyperparameter grid
    cpf = CPFcluster(
        min_samples=10,
        rho=[0.3, 0.5, 0.7, 0.9],
        alpha=[0.6, 0.8, 1.0, 1.2],
        merge=True,
        merge_threshold=[0.6, 0.5, 0.4, 0.3],
        density_ratio_threshold=[0.1, 0.2, 0.3, 0.4],
        n_jobs=-1,
        cutoff=1,
        outlier_method=OutlierMethod.EDGE_COUNT,  # Paper-aligned outlier detection
        plot_tsne=True,
        plot_pca=True,
        plot_umap=True,
    )

    # Fit the model for a range of k values
    print("Fitting CPFcluster...")
    cpf.fit(X, k_values=[5, 10, 15])

    # Cross-validate to find optimal parameters
    print("Performing cross-validation...")
    best_params, best_score = cpf.cross_validate(
        X, validation_index=calinski_harabasz_score
    )
    print(
        f"Best Parameters: min_samples={best_params[0]}, rho={best_params[1]:.2f}, "
        f"alpha={best_params[2]:.2f}, merge_threshold={best_params[3]:.2f}, "
        f"density_ratio_threshold={best_params[4]:.2f}"
    )
    print(f"Best Validation Score (Calinski-Harabasz Index): {best_score:.2f}")

    # Access cluster labels for the best parameter configuration
    best_labels = cpf.clusterings[best_params]
    print(f"Cluster labels for best parameters: {best_labels}")

    # Evaluate clustering performance using Adjusted Rand Index (ARI)
    ari = adjusted_rand_score(y, best_labels)
    print(f"Adjusted Rand Index (ARI) for best parameters: {ari:.2f}")

    # Plot results for the best parameter configuration
    print("Plotting results...")
    cpf.plot_results(
        X,
        k=best_params[0],
        rho=best_params[1],
        alpha=best_params[2],
        merge_threshold=best_params[3],
        density_ratio_threshold=best_params[4],
    )

    elapsed = time.perf_counter() - start_time
    print(f"Total execution time: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
