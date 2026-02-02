from typing import Iterable, Iterator, List, Tuple

import numpy as np
from numpy.typing import NDArray
import scipy.sparse
import multiprocessing as mp
from sklearn.neighbors import NearestNeighbors
import faiss
import gc
import itertools
from enum import Enum
import warnings

from .types import OUTLIER, NO_PARENT
from .core.peak_score import compute_peak_score
from . import utils
from .plotting import plot_clusters_tsne, plot_clusters_pca, plot_clusters_umap

from sklearn.metrics import calinski_harabasz_score


class OutlierMethod(Enum):
    """Outlier detection method for CPF clustering.

    COMPONENT_SIZE: Original implementation - marks components with size <= cutoff as outliers.
    EDGE_COUNT: Paper-aligned - marks points with edge count (degree) <= cutoff as outliers.
    """

    COMPONENT_SIZE = "component_size"  # Original: outlier if component size <= cutoff
    EDGE_COUNT = "edge_count"  # Paper: outlier if point's edge count <= cutoff


def build_CCgraph(
    X: NDArray[np.float32],
    min_samples: int,
    cutoff: int,
    n_jobs: int,
    distance_metric: str = "euclidean",
    outlier_method: OutlierMethod = OutlierMethod.EDGE_COUNT,
) -> Tuple[NDArray[np.int32], scipy.sparse.csr_matrix, NDArray[np.float32]]:
    """
    Constructs a connected component graph (CCgraph) for input data using k-nearest neighbors.
    Identifies connected components and removes outliers based on a specified cutoff.

    Parameters:
        X (np.ndarray): Input data of shape (n_samples, n_features).
        min_samples (int): Minimum number of neighbors to consider for connectivity.
        cutoff (int): Threshold for filtering out outliers.
            - If outlier_method=COMPONENT_SIZE: components with size <= cutoff are outliers.
            - If outlier_method=EDGE_COUNT: points with edge count <= cutoff are outliers.
        n_jobs (int): Number of parallel jobs for computation.
        distance_metric (str): Metric to use for distance computation.
        outlier_method (OutlierMethod): Method for outlier detection.
            - COMPONENT_SIZE (default): Original implementation, marks small components.
            - EDGE_COUNT: Paper-aligned, marks points with few edges.

    Returns:
        components (np.ndarray): Array indicating the component each sample belongs to.
        CCmat (scipy.sparse.csr_matrix): Sparse adjacency matrix representing connections.
        knn_radius (np.ndarray): Array of distances to the min_samples-th neighbor for each sample.
    """
    n = X.shape[0]

    # Handle the case where there are fewer samples than min_samples
    if n < min_samples:
        warnings.warn(
            f"Dataset has fewer samples ({n}) than `min_samples` ({min_samples}). Returning empty graph."
        )
        components = np.full(n, OUTLIER, dtype=np.int32)  # Assign all to outliers
        CCmat = scipy.sparse.csr_matrix((n, n), dtype=np.float32)
        knn_radius = np.full(n, np.nan, dtype=np.float32)
        return components, CCmat, knn_radius

    # Step 1 (Algorithm 2): Build mutual k-NN graph for level-set components.
    # kdt = NearestNeighbors(n_neighbors=min_samples, metric=distance_metric, n_jobs=n_jobs, algorithm='auto').fit(X)
    # CCmat = kdt.kneighbors_graph(X, mode='distance').astype(np.float32)
    # distances, _ = kdt.kneighbors(X)
    # knn_radius = distances[:, min_samples - 1]
    # CCmat = CCmat.minimum(CCmat.T)

    # FAISS Index for Nearest Neighbor Search
    index = faiss.IndexFlatL2(X.shape[1])  # L2 (Euclidean) distance index
    index.add(X.astype(np.float32))
    distances, indices = index.search(X.astype(np.float32), min_samples)

    # Construct adjacency matrix
    row_idx = np.repeat(np.arange(n), min_samples)
    col_idx = indices.flatten()
    data = distances.flatten()
    CCmat = scipy.sparse.csr_matrix(
        (data, (row_idx, col_idx)), shape=(n, n), dtype=np.float32
    )
    CCmat = CCmat.minimum(CCmat.T)  # Ensure symmetry
    knn_radius = distances[:, min_samples - 1]

    # Identify connected components
    _, components = scipy.sparse.csgraph.connected_components(
        CCmat, directed=False, return_labels=True
    )

    # Determine outliers based on the selected method
    if outlier_method == OutlierMethod.EDGE_COUNT:
        # Paper-aligned: mark points with edge count (degree) <= cutoff as outliers
        degrees = np.array(CCmat.getnnz(axis=1)).flatten()
        nanidx = degrees <= cutoff
    else:
        # Original: mark components with size <= cutoff as outliers
        comp_labs, comp_count = np.unique(components, return_counts=True)
        outlier_components = comp_labs[comp_count <= cutoff]
        nanidx = np.isin(components, outlier_components)

    components = components.astype(np.int32)
    components[nanidx] = OUTLIER

    return components, CCmat, knn_radius


def _normalize_n_jobs(n_jobs: int) -> int:
    if n_jobs == -1:
        return mp.cpu_count()
    if n_jobs <= 0:
        return 1
    return n_jobs


def _component_knn(
    X: NDArray[np.float32],
    cc_idx: NDArray[np.int32],
    k: int,
    n_jobs: int,
) -> Tuple[NDArray[np.float32], NDArray[np.int64]]:
    nc = len(cc_idx)
    kcc = max(1, min(k, nc - 1))
    kdt = NearestNeighbors(
        n_neighbors=kcc, metric="euclidean", n_jobs=n_jobs, algorithm="kd_tree"
    ).fit(X[cc_idx, :])
    distances, neighbors = kdt.kneighbors(X[cc_idx, :])
    return distances, neighbors


def _assign_direct_big_brother(
    cc_knn_radius: NDArray[np.float32],
    neighbors: NDArray[np.int64],
    distances: NDArray[np.float32],
) -> Tuple[NDArray[np.int32], NDArray[np.float32], NDArray[np.int32]]:
    nc = len(cc_knn_radius)
    cc_best_distance = np.full(nc, np.inf, dtype=np.float32)
    cc_big_brother = np.full(nc, NO_PARENT, dtype=np.int32)
    # Definition 2: prefer a higher-density neighbor within k-NN for each point.
    cc_radius_diff = cc_knn_radius[:, np.newaxis] - cc_knn_radius[neighbors]
    rows, cols = np.where(cc_radius_diff > 0)
    if rows.size:
        rows, unidx = np.unique(rows, return_index=True)
        cols = cols[unidx]
        cc_big_brother[rows] = neighbors[rows, cols]
        cc_best_distance[rows] = distances[rows, cols]
    search_idx = np.setdiff1d(np.arange(nc, dtype=np.int32), rows)
    return cc_big_brother, cc_best_distance, search_idx


def _handle_no_higher_density(
    GT_radius: NDArray[np.bool_],
    indx_chunk: NDArray[np.int32],
    cc_big_brother: NDArray[np.int32],
    cc_best_distance: NDArray[np.float32],
) -> Tuple[NDArray[np.bool_], NDArray[np.int32]]:
    row_sums = np.sum(GT_radius, axis=1)
    if not np.any(row_sums == 0):
        return GT_radius, indx_chunk
    max_i = [i for i in range(GT_radius.shape[0]) if row_sums[i] == 0]
    if len(max_i) > 1:
        for max_j in max_i[1:]:
            GT_radius[max_j, indx_chunk[max_i[0]]] = True
    max_i = max_i[0]
    cc_big_brother[indx_chunk[max_i]] = indx_chunk[max_i]
    cc_best_distance[indx_chunk[max_i]] = np.inf
    indx_chunk = np.delete(indx_chunk, max_i)
    GT_radius = np.delete(GT_radius, max_i, 0)
    return GT_radius, indx_chunk


def _iter_gt_distances(
    X_cc: NDArray[np.float32],
    indx_chunk: NDArray[np.int32],
    GT_radius: NDArray[np.bool_],
) -> Iterator[List[NDArray[np.float32]]]:
    return (
        [X_cc[indx_chunk[i], np.newaxis], X_cc[GT_radius[i, :], :]]
        for i in range(len(indx_chunk))
    )


def _compute_distances_serial(
    GT_distances: Iterable[List[NDArray[np.float32]]],
) -> List[NDArray[np.float32]]:
    return list(map(utils.density_broad_search_star, list(GT_distances)))


def _compute_distances_pool(
    GT_distances: Iterable[List[NDArray[np.float32]]], n_jobs: int
) -> List[NDArray[np.float32]] | None:
    pool = None
    try:
        pool = mp.Pool(processes=n_jobs)
        distances = []
        while True:
            distance_comp = pool.map(
                utils.density_broad_search_star, itertools.islice(GT_distances, 25)
            )
            if distance_comp:
                distances.append(distance_comp)
            else:
                break
        pool.terminate()
        return [dis_pair for dis_list in distances for dis_pair in dis_list]
    except Exception as e:
        print("POOL ERROR:", e)
        if pool is not None:
            pool.close()
            pool.terminate()
        return None


def _compute_distances(
    GT_radius: NDArray[np.bool_],
    GT_distances: Iterable[List[NDArray[np.float32]]],
    n_jobs: int,
) -> Tuple[List[NDArray[np.float32]], List[int]]:
    if GT_radius.shape[0] > 50:
        distances = _compute_distances_pool(GT_distances, n_jobs)
        if distances is None:
            distances = _compute_distances_serial(GT_distances)
    else:
        distances = _compute_distances_serial(GT_distances)
    argmin_distance = [np.argmin(dist_list) for dist_list in distances]
    return distances, argmin_distance


def _assign_chunk_big_brother(
    X_cc: NDArray[np.float32],
    cc_knn_radius: NDArray[np.float32],
    indx_chunk: NDArray[np.int32],
    cc_big_brother: NDArray[np.int32],
    cc_best_distance: NDArray[np.float32],
    n_jobs: int,
) -> None:
    if len(indx_chunk) == 0:
        return
    search_radius = cc_knn_radius[indx_chunk]
    GT_radius = cc_knn_radius < search_radius[:, np.newaxis]
    GT_radius, indx_chunk = _handle_no_higher_density(
        GT_radius, indx_chunk, cc_big_brother, cc_best_distance
    )
    if len(indx_chunk) == 0:
        return
    GT_distances = _iter_gt_distances(X_cc, indx_chunk, GT_radius)
    distances, argmin_distance = _compute_distances(GT_radius, GT_distances, n_jobs)
    for i in range(GT_radius.shape[0]):
        cc_big_brother[indx_chunk[i]] = np.where(GT_radius[i, :] == 1)[0][
            argmin_distance[i]
        ]
        cc_best_distance[indx_chunk[i]] = np.ravel(distances[i])[argmin_distance[i]]


def _process_search_chunks(
    X_cc: NDArray[np.float32],
    cc_knn_radius: NDArray[np.float32],
    search_idx: NDArray[np.int32],
    cc_big_brother: NDArray[np.int32],
    cc_best_distance: NDArray[np.float32],
    n_jobs: int,
) -> None:
    # Fallback: search higher-density points outside direct k-NN.
    for indx_chunk in utils.chunks(search_idx, 100):
        _assign_chunk_big_brother(
            X_cc, cc_knn_radius, indx_chunk, cc_big_brother, cc_best_distance, n_jobs
        )


def _compute_peaked(
    cc_best_distance: NDArray[np.float32],
    cc_knn_radius: NDArray[np.float32],
) -> NDArray[np.float32] | None:
    # Definition 3: peak-finding score for center selection.
    peaked = compute_peak_score(cc_best_distance, cc_knn_radius)
    if peaked.size == 0:
        return None
    return peaked


def _should_stop_by_radius(
    cc_knn_radius: NDArray[np.float32],
    prop_cent: int,
    not_tested: NDArray[np.bool_],
    CCmat_level: scipy.sparse.csr_matrix,
) -> bool:
    if cc_knn_radius[prop_cent] > max(cc_knn_radius[~not_tested]):
        cc_level_set = np.where(cc_knn_radius <= cc_knn_radius[prop_cent])[0]
        CCmat_check = CCmat_level[cc_level_set, :][:, cc_level_set]
        n_cc, _ = scipy.sparse.csgraph.connected_components(
            CCmat_check, directed=False, return_labels=True
        )
        return n_cc == 1
    return False


def _compute_component_labels(
    CCmat_level: scipy.sparse.csr_matrix,
    cc_knn_radius: NDArray[np.float32],
    prop_cent: int,
    rho: float,
    alpha: float,
    d: int,
) -> Tuple[NDArray[np.int64], NDArray[np.int64]]:
    # Definition 10: density level-set V_{x*} for modal-set selection.
    v_cutoff = cc_knn_radius[prop_cent] / (rho ** (1 / d))
    e_cutoff = cc_knn_radius[prop_cent] / alpha
    e_mask = np.abs(CCmat_level.data) > e_cutoff
    CCmat_level.data[e_mask] = 0
    CCmat_level.eliminate_zeros()
    cc_cut_idx = (
        np.where(cc_knn_radius < v_cutoff)[0]
        if cc_knn_radius[prop_cent] > 0
        else np.where(cc_knn_radius <= v_cutoff)[0]
    )
    reduced_CCmat = CCmat_level[cc_cut_idx, :][:, cc_cut_idx]
    _, cc_labels = scipy.sparse.csgraph.connected_components(
        reduced_CCmat, directed=False, return_labels=True
    )
    return cc_cut_idx, cc_labels


def _decide_candidate(
    cc_centers: List[int],
    not_tested: NDArray[np.bool_],
    prop_cent: int,
    peaked: NDArray[np.float32],
    cc_cut_idx: NDArray[np.int64],
    cc_labels: NDArray[np.int64],
) -> bool:
    center_comp = np.unique(cc_labels[np.isin(cc_cut_idx, cc_centers)])
    prop_cent_comp = cc_labels[np.where(cc_cut_idx == prop_cent)[0][0]]
    if prop_cent_comp in center_comp:
        if peaked[prop_cent] == min(peaked[cc_centers]):
            cc_centers.append(prop_cent)
            not_tested[prop_cent] = False
            return False
        return True
    cc_centers.append(prop_cent)
    not_tested[prop_cent] = False
    return False


def _select_centers_for_component(
    CCmat_level: scipy.sparse.csr_matrix,
    cc_knn_radius: NDArray[np.float32],
    peaked: NDArray[np.float32],
    rho: float,
    alpha: float,
    d: int,
) -> List[int]:
    nc = cc_knn_radius.size
    cc_centers = [np.argmax(peaked)]
    not_tested = np.ones(nc, dtype=bool)
    not_tested[cc_centers[0]] = False

    # Algorithm 2 (Steps 3–16): iterative modal-set selection.
    while np.sum(not_tested) > 0:
        prop_cent = np.argmax(peaked[not_tested])
        prop_cent = np.arange(peaked.shape[0])[not_tested][prop_cent]

        if _should_stop_by_radius(cc_knn_radius, prop_cent, not_tested, CCmat_level):
            break

        cc_cut_idx, cc_labels = _compute_component_labels(
            CCmat_level, cc_knn_radius, prop_cent, rho, alpha, d
        )
        gc.collect()

        if _decide_candidate(
            cc_centers, not_tested, prop_cent, peaked, cc_cut_idx, cc_labels
        ):
            break

    return cc_centers


def _assign_component_labels(
    cc_idx: NDArray[np.int64],
    big_brother: NDArray[np.int32],
    cc_centers: List[int],
    n_cent: int,
) -> Tuple[int, NDArray[np.int64]]:
    # Algorithm 2 (Steps 17–23): assign via Big Brother graph.
    nc = cc_idx.size
    local_index_map = {
        global_idx: local_idx for local_idx, global_idx in enumerate(cc_idx)
    }
    cc_big_brother = np.array(
        [local_index_map.get(b, NO_PARENT) for b in big_brother[cc_idx]]
    )
    valid_mask = cc_big_brother >= 0
    BBTree = np.column_stack((np.arange(nc)[valid_mask], cc_big_brother[valid_mask]))
    BBTree[cc_centers, 1] = cc_centers
    Clustmat = scipy.sparse.csr_matrix(
        (np.ones(BBTree.shape[0]), (BBTree[:, 0], BBTree[:, 1])), shape=(nc, nc)
    )
    n_clusts, cc_y_pred = scipy.sparse.csgraph.connected_components(
        Clustmat, directed=True, return_labels=True
    )
    cc_y_pred += n_cent
    return n_clusts, cc_y_pred


def get_density_dists_bb(
    X: NDArray[np.float32],
    k: int,
    components: NDArray[np.int32],
    knn_radius: NDArray[np.float32],
    n_jobs: int,
) -> Tuple[NDArray[np.float32], NDArray[np.int32]]:
    """
    Computes the best distance for each data point and identifies its 'big brother',
    which is the nearest point with a greater density.

    Parameters:
        X (np.ndarray): Input data of shape (n_samples, n_features).
        k (int): Number of neighbors to consider for density estimation.
        components (np.ndarray): Array indicating the component each sample belongs to.
        knn_radius (np.ndarray): Array of distances to the min_samples-th neighbor for each sample.
        n_jobs (int): Number of parallel jobs for computation.

    Returns:
        best_distance (np.ndarray): Array of best distances for each data point.
        big_brother (np.ndarray): Array indicating the index of the 'big brother' for each point.
    """
    n_jobs = _normalize_n_jobs(n_jobs)
    best_distance = np.full(X.shape[0], np.inf, dtype=np.float32)
    big_brother = np.full(X.shape[0], NO_PARENT, dtype=np.int32)
    comps = np.unique(components[components != OUTLIER]).astype(np.int32)

    # Step 2: compute ω(x) and big brother per component.
    for cc in comps:
        cc_idx = np.where(components == cc)[0].astype(np.int32)
        if len(cc_idx) == 0:
            continue
        X_cc = X[cc_idx, :]
        distances, neighbors = _component_knn(X, cc_idx, k, n_jobs)
        cc_knn_radius = knn_radius[cc_idx]
        cc_big_brother, cc_best_distance, search_idx = _assign_direct_big_brother(
            cc_knn_radius, neighbors, distances
        )
        _process_search_chunks(
            X_cc, cc_knn_radius, search_idx, cc_big_brother, cc_best_distance, n_jobs
        )

        big_brother[cc_idx] = cc_idx[cc_big_brother.astype(np.int32)]
        best_distance[cc_idx] = cc_best_distance

    return best_distance, big_brother


def get_y(
    CCmat: scipy.sparse.csr_matrix,
    components: NDArray[np.int32],
    knn_radius: NDArray[np.float32],
    best_distance: NDArray[np.float32],
    big_brother: NDArray[np.int32],
    rho: float,
    alpha: float,
    d: int,
) -> NDArray[np.int64]:
    """
    Assigns cluster labels to data points based on density and connectivity properties.
    Identifies peaks within each connected component to create clusters.

    Parameters:
        CCmat (scipy.sparse.csr_matrix): Sparse adjacency matrix representing connections.
        components (np.ndarray): Array indicating the component each sample belongs to.
        knn_radius (np.ndarray): Array of distances to the min_samples-th neighbor for each sample.
        best_distance (np.ndarray): Array of best distances for each data point.
        big_brother (np.ndarray): Array indicating the index of the 'big brother' for each point.
        rho (float): Density parameter controlling the radius cutoff.
        alpha (float): Parameter for edge-cutoff in cluster detection.
        d (int): Number of features (dimensions) in the data.

    Returns:
        y_pred (np.ndarray): Array of predicted cluster labels.
    """
    n = components.shape[0]
    y_pred = np.full(n, OUTLIER)
    valid_indices = components != OUTLIER
    peaks = []
    n_cent = 0
    comps = np.unique(components[valid_indices]).astype(int)

    # Algorithm 2: select centers, then assign via Big Brother.
    for cc in comps:
        cc_idx = np.where(components == cc)[0]
        if cc_idx.size == 0:
            continue
        cc_knn_radius = knn_radius[cc_idx]
        cc_best_distance = best_distance[cc_idx]

        peaked = _compute_peaked(cc_best_distance, cc_knn_radius)
        if peaked is None:
            warnings.warn(
                f"No valid peaked values found for component {cc}. Skipping clustering for this component."
            )
            continue

        CCmat_level = CCmat[cc_idx, :][:, cc_idx]
        cc_centers = _select_centers_for_component(
            CCmat_level, cc_knn_radius, peaked, rho, alpha, d
        )
        peaks.extend(cc_idx[cc_centers])

        n_clusts, cc_y_pred = _assign_component_labels(
            cc_idx, big_brother, cc_centers, n_cent
        )
        n_cent += n_clusts
        y_pred[cc_idx] = cc_y_pred

    return y_pred


class CPFcluster:
    """
    A class to perform CPF (Connected Components and Density-based) clustering.
    """

    def __init__(
        self,
        min_samples=5,
        rho=None,
        alpha=None,
        n_jobs=1,
        remove_duplicates=False,
        cutoff=1,
        distance_metric="euclidean",
        merge=False,
        merge_threshold=None,
        density_ratio_threshold=None,
        outlier_method=OutlierMethod.EDGE_COUNT,
        plot_umap=False,
        plot_pca=False,
        plot_tsne=False,
    ):
        self.min_samples = min_samples
        self.rho = rho if rho is not None else [0.4]
        self.alpha = alpha if alpha is not None else [1]
        self.n_jobs = n_jobs
        self.remove_duplicates = remove_duplicates
        self.cutoff = cutoff
        self.distance_metric = distance_metric
        self.merge = merge
        self.merge_threshold = merge_threshold if merge_threshold is not None else [0.5]
        self.density_ratio_threshold = (
            density_ratio_threshold if density_ratio_threshold is not None else [0.1]
        )
        self.outlier_method = outlier_method
        self.plot_umap = plot_umap
        self.plot_pca = plot_pca
        self.plot_tsne = plot_tsne
        self.clusterings = {}

    def fit(self, X, k_values=None):
        """
        Fits the CPF clustering model to the input data and optimizes for multiple parameter combinations.

        Parameters:
            X (np.ndarray): Input data of shape (n_samples, n_features).
            k_values (list): List of `k` values for constructing the neighborhood graph.

        Returns:
            dict: A dictionary of clusterings with keys as (k, rho, alpha, merge_threshold, density_ratio_threshold) tuples and values as cluster labels.
        """
        if not isinstance(X, np.ndarray):
            raise ValueError("X must be an n x d numpy array.")
        if self.remove_duplicates:
            X = np.unique(X, axis=0)
        n, d = X.shape

        if k_values is None:
            k_values = [
                self.min_samples
            ]  # Default to min_samples if no k_values are provided.

        for k in k_values:
            # Build the k-neighborhood graph
            components, CCmat, knn_radius = build_CCgraph(
                X,
                k,
                self.cutoff,
                self.n_jobs,
                self.distance_metric,
                self.outlier_method,
            )
            self.CCmat = CCmat
            # Compute best distance and big brother for the current k
            best_distance, big_brother = get_density_dists_bb(
                X, k, components, knn_radius, self.n_jobs
            )

            # Cluster for each parameter combination using the precomputed k-neighborhood graph
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

                # Merge clusters if required
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

    def calculate_centroids_and_densities(self, X, labels):
        """
        Calculates the centroids and average densities of clusters.

        Parameters:
            X (np.ndarray): Input data of shape (n_samples, n_features).
            labels (np.ndarray): Cluster labels for each sample.

        Returns:
            centroids (np.ndarray): Centroids of each cluster.
            densities (np.ndarray): Average density of each cluster.
        """
        valid_indices = labels != -1
        unique_labels = np.unique(labels[valid_indices])
        centroids = np.array([X[labels == k].mean(axis=0) for k in unique_labels])
        densities = np.array(
            [np.mean(self.CCmat[labels == k, :][:, labels == k]) for k in unique_labels]
        )
        return centroids, densities

    def merge_clusters(
        self, X, centroids, densities, labels, merge_threshold, density_ratio_threshold
    ):
        """
        Merges similar clusters based on distance and density ratio thresholds.

        Parameters:
            X (np.ndarray): Input data of shape (n_samples, n_features).
            centroids (np.ndarray): Centroids of each cluster.
            densities (np.ndarray): Average density of each cluster.
            labels (np.ndarray): Cluster labels for each sample.
            merge_threshold (float): Distance threshold for merging clusters.
            density_ratio_threshold (float): Density ratio threshold for merging clusters.

        Returns:
            labels (np.ndarray): Updated cluster labels after merging.
        """
        n_clusters = len(centroids)
        merge_map = {}
        for i in range(n_clusters):
            for j in range(i + 1, n_clusters):
                if (
                    np.linalg.norm(centroids[i] - centroids[j]) < merge_threshold
                    and abs(densities[i] - densities[j])
                    / max(densities[i], densities[j])
                    < density_ratio_threshold
                ):
                    smaller = i if densities[i] < densities[j] else j
                    larger = j if smaller == i else i
                    merge_map[smaller] = larger
                    centroids[larger] = (
                        centroids[larger] * np.sum(labels == larger)
                        + centroids[smaller] * np.sum(labels == smaller)
                    ) / (np.sum(labels == larger) + np.sum(labels == smaller))
                    densities[larger] = (densities[larger] + densities[smaller]) / 2
        for old, new in merge_map.items():
            labels[labels == old] = new
        return labels

    def cross_validate(self, X, validation_index=calinski_harabasz_score):
        """
        Cross-validates the CPF clustering model using a specified validation index.

        Parameters:
            X (np.ndarray): Input data of shape (n_samples, n_features).
            validation_index (callable): A function to compute the validation index (default: Calinski-Harabasz index).

        Returns:
            tuple: Optimal (k, rho, alpha, merge_threshold, density_ratio_threshold) parameters and their corresponding validation score.
        """
        if not self.clusterings:
            raise RuntimeError(
                "You need to call `fit` before running cross-validation."
            )

        best_score = -np.inf
        best_params = None
        for (
            k,
            rho,
            alpha,
            merge_threshold,
            density_ratio_threshold,
        ), labels in self.clusterings.items():
            if len(np.unique(labels)) > 1:
                score = validation_index(X, labels)
                if score > best_score:
                    best_score = score
                    best_params = (
                        k,
                        rho,
                        alpha,
                        merge_threshold,
                        density_ratio_threshold,
                    )
        return best_params, best_score

    def plot_results(
        self,
        X,
        k=None,
        rho=None,
        alpha=None,
        merge_threshold=None,
        density_ratio_threshold=None,
    ):
        """
        Plots the clustering results for a specific combination of parameters.

        Parameters:
            X (np.ndarray): Input data of shape (n_samples, n_features).
            k (int): Value of `k` for the neighborhood graph.
            rho (float): Value of `rho` for clustering.
            alpha (float): Value of `alpha` for clustering.
            merge_threshold (float): Distance threshold for merging clusters.
            density_ratio_threshold (float): Density ratio threshold for merging clusters.

        Returns:
            None
        """
        if (
            k,
            rho,
            alpha,
            merge_threshold,
            density_ratio_threshold,
        ) not in self.clusterings:
            raise ValueError(
                f"No clustering result found for (k={k}, rho={rho}, alpha={alpha}, merge_threshold={merge_threshold}, density_ratio_threshold={density_ratio_threshold})."
            )

        labels = self.clusterings[
            (k, rho, alpha, merge_threshold, density_ratio_threshold)
        ]
        if self.plot_umap:
            plot_clusters_umap(X, labels)
        if self.plot_pca:
            plot_clusters_pca(X, labels)
        if self.plot_tsne:
            plot_clusters_tsne(X, labels)
