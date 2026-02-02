//! kNN search (brute force).
//!
//! Matches the reference behavior:
//! - includes self at index 0
//! - returns k neighbors total (including self)
//! - radius = distance to k-th neighbor in the returned list

use crate::data::Dataset;
use crate::math::euclidean;

#[derive(Debug, Clone)]
pub struct KnnResult {
    pub indices: Vec<Vec<usize>>,   // shape: (n, k)
    pub distances: Vec<Vec<f32>>,   // shape: (n, k)
    pub radius: Vec<f32>,           // shape: (n,)
    pub k: usize,
}

pub fn knn_search_bruteforce(dataset: &Dataset, k: usize) -> KnnResult {
    let n = dataset.n();
    let mut indices = Vec::with_capacity(n);
    let mut distances = Vec::with_capacity(n);
    let mut radius = vec![0.0f32; n];

    for i in 0..n {
        let xi = dataset.row(i);
        let mut dv: Vec<(f32, usize)> = Vec::with_capacity(n);
        for j in 0..n {
            let d = euclidean(xi, dataset.row(j));
            dv.push((d, j));
        }
        dv.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let take = k.min(dv.len());
        let mut neigh = Vec::with_capacity(take);
        let mut dist = Vec::with_capacity(take);
        for t in 0..take {
            dist.push(dv[t].0);
            neigh.push(dv[t].1);
        }

        // Ensure self at index 0.
        if take > 0 && neigh[0] != i {
            if let Some(pos) = neigh.iter().position(|&x| x == i) {
                neigh.swap(0, pos);
                dist.swap(0, pos);
                dist[0] = 0.0;
            }
        }

        radius[i] = if take == 0 { 0.0 } else { dist[take - 1] };
        indices.push(neigh);
        distances.push(dist);
    }

    KnnResult {
        indices,
        distances,
        radius,
        k,
    }
}
