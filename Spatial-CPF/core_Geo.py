import itertools
import warnings
from typing import Tuple

import numpy as np
from numpy.typing import NDArray
import scipy.sparse as sp

from src import CPFcluster as BaseCPFcluster
from src import OutlierMethod, get_density_dists_bb, get_y
from src.graph.adjacency import apply_mask, build_knn_adjacency, make_mutual
from src.graph.components import (
    extract_components,
    filter_by_component_size,
    filter_by_edge_count,
)
from src.graph.knn import knn_search_faiss
from src.types import OUTLIER


def build_CCgraph(
    X: NDArray[np.float32],
    geo_neighbor_adjacency_matrix: sp.spmatrix | NDArray[np.float32],
    min_samples: int,
    cutoff: int,
    n_jobs: int,
    distance_metric: str = "euclidean",
    outlier_method: OutlierMethod = OutlierMethod.EDGE_COUNT,
) -> Tuple[NDArray[np.int32], sp.csr_matrix, NDArray[np.float32]]:
    """Construct a geo-constrained mutual k-NN graph and components.

    This is the Spatial-CPF variant: it applies a geographic adjacency mask
    to the mutual k-NN graph before extracting connected components.
    """
    n = X.shape[0]

    if n < min_samples:
        warnings.warn(
            f"Dataset has fewer samples ({n}) than `min_samples` ({min_samples}). "
            "Returning empty graph."
        )
        components = np.full(n, OUTLIER, dtype=np.int32)
        CCmat = sp.csr_matrix((n, n), dtype=np.float32)
        knn_radius = np.full(n, np.nan, dtype=np.float32)
        return components, CCmat, knn_radius

    knn = knn_search_faiss(X.astype(np.float32), min_samples)
    # Definition 7: mutual k-NN graph.
    CCmat = make_mutual(build_knn_adjacency(knn))

    mask = geo_neighbor_adjacency_matrix
    if not sp.issparse(mask):
        mask = sp.csr_matrix(mask)
    CCmat = apply_mask(CCmat, mask)

    # Definition 9: connected components after geo mask.
    components = extract_components(CCmat)
    if outlier_method == OutlierMethod.EDGE_COUNT:
        components = filter_by_edge_count(CCmat, components, cutoff)
    else:
        components = filter_by_component_size(components, cutoff)

    return components, CCmat, knn.radius


class CPFcluster(BaseCPFcluster):
    """Spatial-CPF clusterer with geo-adjacency constraints."""

    def fit(
        self,
        X: NDArray[np.float32],
        geo_neighbor_adjacency_matrix: sp.spmatrix | NDArray[np.float32],
        k_values: list[int] | None = None,
    ):
        """Fit the Spatial-CPF model.

        The only difference from the base implementation is that the mutual k-NN
        graph is masked by geographic adjacency before clustering.
        """
        if not isinstance(X, np.ndarray):
            raise ValueError("X must be an n x d numpy array.")
        if self.remove_duplicates:
            X = np.unique(X, axis=0)
        n, d = X.shape

        if k_values is None:
            k_values = [self.min_samples]

        for k in k_values:
            components, CCmat, knn_radius = build_CCgraph(
                X,
                geo_neighbor_adjacency_matrix,
                k,
                self.cutoff,
                self.n_jobs,
                self.distance_metric,
            )
            self.CCmat = CCmat
            best_distance, big_brother = get_density_dists_bb(
                X, k, components, knn_radius, self.n_jobs
            )

            for (
                rho_val,
                alpha_val,
                merge_threshold_val,
                density_ratio_threshold_val,
            ) in itertools.product(
                self.rho, self.alpha, self.merge_threshold, self.density_ratio_threshold
            ):
                labels = get_y(
                    CCmat,
                    components,
                    knn_radius,
                    best_distance,
                    big_brother,
                    rho_val,
                    alpha_val,
                    d,
                )

                if self.merge:
                    centroids, densities = self.calculate_centroids_and_densities(
                        X, labels
                    )
                    labels = self.merge_clusters(
                        X,
                        centroids,
                        densities,
                        labels,
                        merge_threshold_val,
                        density_ratio_threshold_val,
                    )

                self.clusterings[
                    (
                        k,
                        rho_val,
                        alpha_val,
                        merge_threshold_val,
                        density_ratio_threshold_val,
                    )
                ] = labels

        return self.clusterings
