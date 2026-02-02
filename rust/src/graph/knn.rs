//! kNN search (KD-tree + brute force).
//!
//! Matches the reference behavior:
//! - includes self at index 0
//! - returns k neighbors total (including self)
//! - radius = distance to k-th neighbor in the returned list

use crate::data::Dataset;
use crate::math::euclidean;
use kdtree::KdTree;
use rayon::prelude::*;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct KnnResult {
    pub indices: Vec<Vec<usize>>,   // shape: (n, k)
    pub distances: Vec<Vec<f32>>,   // shape: (n, k)
    pub radius: Vec<f32>,           // shape: (n,)
    pub k: usize,
}

#[derive(Debug, Clone, Copy)]
pub enum KnnBackend {
    /// KD-tree backend (default). Exact kNN with Euclidean distance.
    KdTree,
    /// Brute-force backend. Useful for small n or debugging.
    BruteForce,
}

impl Default for KnnBackend {
    fn default() -> Self {
        KnnBackend::KdTree
    }
}

pub fn knn_search(dataset: &Dataset, k: usize, backend: KnnBackend) -> KnnResult {
    match backend {
        KnnBackend::KdTree => knn_search_kdtree(dataset, k),
        KnnBackend::BruteForce => knn_search_bruteforce(dataset, k),
    }
}

/// Brute-force kNN (Definition 1: r_k is Euclidean distance).
pub fn knn_search_bruteforce(dataset: &Dataset, k: usize) -> KnnResult {
    let n = dataset.n();
    // Parallelize by query point; each point is independent.
    let results: Vec<(Vec<usize>, Vec<f32>, f32)> = (0..n)
        .into_par_iter()
        .map(|i| compute_knn_for_point_bruteforce(dataset, k, i))
        .collect();

    let mut indices = Vec::with_capacity(n);
    let mut distances = Vec::with_capacity(n);
    let mut radius = Vec::with_capacity(n);
    for (neigh, dist, rad) in results {
        indices.push(neigh);
        distances.push(dist);
        radius.push(rad);
    }

    KnnResult {
        indices,
        distances,
        radius,
        k,
    }
}

/// KD-tree kNN (Definition 1: r_k is Euclidean distance).
pub fn knn_search_kdtree(dataset: &Dataset, k: usize) -> KnnResult {
    let n = dataset.n();
    let d = dataset.d();
    let mut tree: KdTree<f32, usize, Vec<f32>> = KdTree::new(d);
    for i in 0..n {
        let _ = tree.add(dataset.row(i).to_vec(), i);
    }
    let tree = Arc::new(tree);
    let k_query = k.min(n.max(1));

    // Parallelize queries; KD-tree is read-only after build.
    let results: Vec<(Vec<usize>, Vec<f32>, f32)> = (0..n)
        .into_par_iter()
        .map(|i| compute_knn_for_point_kdtree(&tree, k, k_query, i, dataset.row(i)))
        .collect();

    let mut indices = Vec::with_capacity(n);
    let mut distances = Vec::with_capacity(n);
    let mut radius = Vec::with_capacity(n);
    for (neigh, dist, rad) in results {
        indices.push(neigh);
        distances.push(dist);
        radius.push(rad);
    }

    KnnResult {
        indices,
        distances,
        radius,
        k,
    }
}

/// Sort distance pairs, extract top-k, ensure self at index 0, and compute radius.
/// Common post-processing for both brute-force and KD-tree backends.
#[inline]
fn sort_and_extract_knn(
    mut dist_pairs: Vec<(f32, usize)>,
    query_idx: usize,
    k: usize,
) -> (Vec<usize>, Vec<f32>, f32) {
    // 1. Sort by distance (tie-break by index for determinism).
    dist_pairs.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.1.cmp(&b.1))
    });

    // 2. Extract top-k.
    let take = k.min(dist_pairs.len());
    let mut neigh = Vec::with_capacity(take);
    let mut dist = Vec::with_capacity(take);
    for t in 0..take {
        dist.push(dist_pairs[t].0);
        neigh.push(dist_pairs[t].1);
    }

    // 3. Ensure query point is at index 0.
    if take > 0 && neigh[0] != query_idx {
        if let Some(pos) = neigh.iter().position(|&x| x == query_idx) {
            neigh.swap(0, pos);
            dist.swap(0, pos);
            dist[0] = 0.0;
        }
    }

    // 4. r_k(x): distance to the k-th neighbor (including self at index 0).
    let rad = if take == 0 { 0.0 } else { dist[take - 1] };
    (neigh, dist, rad)
}

/// Brute-force kNN for a single query point.
fn compute_knn_for_point_bruteforce(
    dataset: &Dataset,
    k: usize,
    i: usize,
) -> (Vec<usize>, Vec<f32>, f32) {
    let n = dataset.n();
    let xi = dataset.row(i);
    let dv: Vec<(f32, usize)> = (0..n)
        .map(|j| (euclidean(xi, dataset.row(j)), j))
        .collect();
    sort_and_extract_knn(dv, i, k)
}

/// KD-tree kNN for a single query point.
fn compute_knn_for_point_kdtree(
    tree: &Arc<KdTree<f32, usize, Vec<f32>>>,
    k: usize,
    k_query: usize,
    i: usize,
    point: &[f32],
) -> (Vec<usize>, Vec<f32>, f32) {
    let result: Vec<(f32, usize)> = tree
        .nearest(point, k_query, &euclidean)
        .unwrap_or_default()
        .into_iter()
        .map(|(d, &idx)| (d, idx))
        .collect();
    sort_and_extract_knn(result, i, k)
}
