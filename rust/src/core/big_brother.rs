//! Big Brother computation (Definition 2).

use crate::data::Dataset;
use crate::math::euclidean;
use crate::types::{NO_PARENT, OUTLIER};
use rayon::prelude::*;

/// Big Brother output: parent index and distance to parent (omega).
#[derive(Debug, Clone)]
pub struct BigBrotherResult {
    pub parent: Vec<i32>,
    pub parent_dist: Vec<f32>,
}

/// Compute Big Brother for all components (Definition 2).
pub fn compute_big_brother(
    dataset: &Dataset,
    knn_radius: &[f32],
    components: &[i32],
    k: usize,
) -> BigBrotherResult {
    let n = dataset.n();
    let mut parent = vec![NO_PARENT; n];
    let mut parent_dist = vec![f32::INFINITY; n];

    let mut comp_map: std::collections::BTreeMap<i32, Vec<usize>> =
        std::collections::BTreeMap::new();
    for i in 0..n {
        let c = components[i];
        if c != OUTLIER {
            comp_map.entry(c).or_default().push(i);
        }
    }

    let comp_list: Vec<Vec<usize>> = comp_map.into_values().collect();
    // Components are independent, so compute in parallel.
    let results: Vec<(Vec<usize>, BigBrotherResult)> = comp_list
        .par_iter()
        .map(|cc_idx| {
            let mut radius_cc = Vec::with_capacity(cc_idx.len());
            for &gi in cc_idx {
                radius_cc.push(knn_radius[gi]);
            }
            let bb = compute_big_brother_for_component(dataset, cc_idx, &radius_cc, k);
            (cc_idx.clone(), bb)
        })
        .collect();

    for (cc_idx, cc) in results {
        for (li, &gi) in cc_idx.iter().enumerate() {
            let p_local = cc.parent[li];
            if p_local == NO_PARENT {
                parent[gi] = NO_PARENT;
            } else {
                parent[gi] = cc_idx[p_local as usize] as i32;
            }
            parent_dist[gi] = cc.parent_dist[li];
        }
    }

    BigBrotherResult { parent, parent_dist }
}

pub fn compute_big_brother_for_component(
    dataset: &Dataset,
    cc_idx: &[usize],
    radius_cc: &[f32],
    k: usize,
) -> BigBrotherResult {
    let nc = cc_idx.len();
    let mut parent = vec![NO_PARENT; nc];
    let mut parent_dist = vec![f32::INFINITY; nc];

    if nc <= 1 {
        return BigBrotherResult { parent, parent_dist };
    }

    let k_local = std::cmp::max(1, std::cmp::min(k, nc - 1));

    // First pass: choose nearest higher-density neighbor within local k-NN.
    for i in 0..nc {
        let xi = dataset.row(cc_idx[i]);
        let mut dv: Vec<(f32, usize)> = Vec::with_capacity(nc);
        for j in 0..nc {
            let d = euclidean(xi, dataset.row(cc_idx[j]));
            dv.push((d, j));
        }
        dv.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let take = k_local.min(dv.len());
        for t in 0..take {
            let j = dv[t].1;
            if radius_cc[i] - radius_cc[j] > 0.0 {
                parent[i] = j as i32;
                parent_dist[i] = dv[t].0;
                break;
            }
        }
    }

    // Second pass: for remaining points, search all higher-density points.
    let mut maxima: Vec<usize> = Vec::new();
    for i in 0..nc {
        if parent[i] != NO_PARENT {
            continue;
        }
        let mut best = (f32::INFINITY, NO_PARENT);
        for j in 0..nc {
            if radius_cc[j] < radius_cc[i] {
                let d = euclidean(dataset.row(cc_idx[i]), dataset.row(cc_idx[j]));
                if d < best.0 {
                    best = (d, j as i32);
                }
            }
        }
        if best.1 != NO_PARENT {
            parent[i] = best.1;
            parent_dist[i] = best.0;
        } else {
            maxima.push(i);
        }
    }

    // If multiple maxima exist, pick one as root and attach the rest to it.
    if !maxima.is_empty() {
        let root = maxima[0];
        parent[root] = root as i32;
        parent_dist[root] = f32::INFINITY;
        for &idx in maxima.iter().skip(1) {
            parent[idx] = root as i32;
            parent_dist[idx] =
                euclidean(dataset.row(cc_idx[idx]), dataset.row(cc_idx[root]));
        }
    }

    BigBrotherResult { parent, parent_dist }
}
