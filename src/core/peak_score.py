"""ピーク指標計算モジュール"""

import numpy as np
from src.types import Float32Array


def compute_peak_score(
    parent_dist: Float32Array,
    radius: Float32Array,
) -> Float32Array:
    """ピーク指標 γ(x) = ω(x) / r_k(x) を計算

    論文: γ(x) = f̂_k(x) · ω(x) ∝ ω(x) / r_k(x)

    Args:
        parent_dist: big brotherへの距離 ω(x)
        radius: k近傍距離 r_k(x)

    Returns:
        Float32Array: ピーク指標 γ(x)
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        peaked = np.divide(
            parent_dist,
            radius,
            out=np.zeros_like(parent_dist),
            where=radius != 0,
        )
    peaked[(parent_dist == 0) & (radius == 0)] = np.inf
    peaked[(radius == 0) & (parent_dist > 0)] = np.inf
    return peaked.astype(np.float32)
