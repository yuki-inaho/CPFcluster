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
        .map(|i| compute_knn_for_point_kdtree(dataset, &tree, k, k_query, i))
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

/// Brute-force kNN for a single query point.
fn compute_knn_for_point_bruteforce(
    dataset: &Dataset,
    k: usize,
    i: usize,
) -> (Vec<usize>, Vec<f32>, f32) {
    let n = dataset.n();
    let xi = dataset.row(i);
    let mut dv: Vec<(f32, usize)> = Vec::with_capacity(n);
    for j in 0..n {
        let d = euclidean(xi, dataset.row(j));
        dv.push((d, j));
    }
    dv.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.1.cmp(&b.1))
    });

    let take = k.min(dv.len());
    let mut neigh = Vec::with_capacity(take);
    let mut dist = Vec::with_capacity(take);
    for t in 0..take {
        dist.push(dv[t].0);
        neigh.push(dv[t].1);
    }

    if take > 0 && neigh[0] != i {
        if let Some(pos) = neigh.iter().position(|&x| x == i) {
            neigh.swap(0, pos);
            dist.swap(0, pos);
            dist[0] = 0.0;
        }
    }

    // r_k(x): distance to the k-th neighbor (including self at index 0).
    let rad = if take == 0 { 0.0 } else { dist[take - 1] };
    (neigh, dist, rad)
}

/// KD-tree kNN for a single query point.
fn compute_knn_for_point_kdtree(
    dataset: &Dataset,
    tree: &Arc<KdTree<f32, usize, Vec<f32>>>,
    k: usize,
    k_query: usize,
    i: usize,
) -> (Vec<usize>, Vec<f32>, f32) {
    let point = dataset.row(i);
    let mut result = tree
        .nearest(point, k_query, &euclidean)
        .unwrap_or_default()
        .into_iter()
        .map(|(d, &idx)| (d, idx))
        .collect::<Vec<(f32, usize)>>();

    result.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.1.cmp(&b.1))
    });

    let take = k.min(result.len());
    let mut neigh = Vec::with_capacity(take);
    let mut dist = Vec::with_capacity(take);
    for t in 0..take {
        dist.push(result[t].0);
        neigh.push(result[t].1);
    }

    if take > 0 && neigh[0] != i {
        if let Some(pos) = neigh.iter().position(|&x| x == i) {
            neigh.swap(0, pos);
            dist.swap(0, pos);
            dist[0] = 0.0;
        }
    }

    let rad = if take == 0 { 0.0 } else { dist[take - 1] };
    (neigh, dist, rad)
}
