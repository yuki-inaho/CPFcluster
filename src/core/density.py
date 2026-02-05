"""Density proxy computation (fastCPF parity)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from src.graph.knn import KnnResult
from src.types import Float32Array


def compute_density_radius(knn: KnnResult, method: str) -> Float32Array:
    """Compute density radius used in peak finding.

    Args:
        knn: kNN search result.
        method: "rk", "median", or "mean".

    Returns:
        Float32Array: Density radius per point.
    """
    key = method.lower()
    if key in {"rk", "r_k", "knn_radius"}:
        return knn.radius.astype(np.float32)
    if key == "median":
        return _median_radius(knn.distances, knn.radius)
    if key in {"mean", "avg", "average"}:
        return _mean_radius(knn.distances, knn.radius)
    raise ValueError(
        f"Unsupported density_method: '{method}'. Use 'rk', 'median', or 'mean'."
    )


def _median_radius(
    distances: NDArray[np.float32],
    fallback: Float32Array,
) -> Float32Array:
    if distances.shape[1] <= 1:
        return fallback.astype(np.float32)
    vals = distances[:, 1:]
    if vals.size == 0:
        return fallback.astype(np.float32)
    return np.median(vals, axis=1).astype(np.float32)


def _mean_radius(
    distances: NDArray[np.float32],
    fallback: Float32Array,
) -> Float32Array:
    if distances.shape[1] <= 1:
        return fallback.astype(np.float32)
    vals = distances[:, 1:]
    if vals.size == 0:
        return fallback.astype(np.float32)
    return np.mean(vals, axis=1).astype(np.float32)
