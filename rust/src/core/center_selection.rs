//! Center selection (Algorithm 2, Steps 3-16).

use crate::graph::WeightedGraph;

pub fn select_centers_for_component(
    graph: &WeightedGraph,
    cc_idx: &[usize],
    cc_knn_radius: &[f32],
    peaked: &[f32],
    rho: f32,
    alpha: f32,
    dim: usize,
) -> Vec<usize> {
    let nc = cc_idx.len();
    if nc == 0 {
        return Vec::new();
    }

    let mut centers = vec![argmax(peaked, None).unwrap_or(0)];
    let mut not_tested = vec![true; nc];
    not_tested[centers[0]] = false;

    let mut edge_threshold = f32::INFINITY;
    let global_to_local = build_global_to_local(cc_idx, graph.n());

    while not_tested.iter().any(|&v| v) {
        let prop_cent = argmax(peaked, Some(&not_tested)).unwrap();

        if should_stop_by_radius(
            graph,
            cc_idx,
            &global_to_local,
            cc_knn_radius,
            prop_cent,
            &not_tested,
            edge_threshold,
        ) {
            break;
        }

        let v_cutoff = cc_knn_radius[prop_cent] / rho.powf(1.0 / dim as f32);
        let e_cutoff = cc_knn_radius[prop_cent] / alpha;
        edge_threshold = edge_threshold.min(e_cutoff);

        let mut cc_cut_idx: Vec<usize> = Vec::new();
        if cc_knn_radius[prop_cent] > 0.0 {
            for i in 0..nc {
                if cc_knn_radius[i] < v_cutoff {
                    cc_cut_idx.push(i);
                }
            }
        } else {
            for i in 0..nc {
                if cc_knn_radius[i] <= v_cutoff {
                    cc_cut_idx.push(i);
                }
            }
        }

        if cc_cut_idx.is_empty() {
            break;
        }

        let cc_labels = connected_components_for_subset(
            graph,
            cc_idx,
            &global_to_local,
            &cc_cut_idx,
            edge_threshold,
        );

        let prop_pos = cc_cut_idx.iter().position(|&x| x == prop_cent);
        if prop_pos.is_none() {
            not_tested[prop_cent] = false;
            continue;
        }
        let prop_comp = cc_labels[prop_pos.unwrap()];

        let mut center_comps: Vec<i32> = Vec::new();
        for &c in &centers {
            if let Some(pos) = cc_cut_idx.iter().position(|&x| x == c) {
                center_comps.push(cc_labels[pos]);
            }
        }

        let intersects = center_comps.iter().any(|&x| x == prop_comp);
        if intersects {
            let min_center = min_peaked(centers.iter().map(|&i| peaked[i]));
            if (peaked[prop_cent] - min_center).abs() <= 1e-12 {
                centers.push(prop_cent);
                not_tested[prop_cent] = false;
                continue;
            } else {
                break;
            }
        } else {
            centers.push(prop_cent);
            not_tested[prop_cent] = false;
        }
    }

    centers
}

fn build_global_to_local(cc_idx: &[usize], n: usize) -> Vec<i32> {
    let mut map = vec![-1i32; n];
    for (li, &gi) in cc_idx.iter().enumerate() {
        map[gi] = li as i32;
    }
    map
}

fn argmax(values: &[f32], mask: Option<&[bool]>) -> Option<usize> {
    if values.is_empty() {
        return None;
    }
    let mut best_i: Option<usize> = None;
    let mut best_v = f32::NEG_INFINITY;
    match mask {
        None => {
            for (i, &v) in values.iter().enumerate() {
                if v > best_v {
                    best_v = v;
                    best_i = Some(i);
                }
            }
        }
        Some(m) => {
            for (i, (&v, &ok)) in values.iter().zip(m.iter()).enumerate() {
                if ok && v > best_v {
                    best_v = v;
                    best_i = Some(i);
                }
            }
        }
    }
    best_i
}

fn min_peaked<I: Iterator<Item = f32>>(iter: I) -> f32 {
    let mut best = f32::INFINITY;
    for v in iter {
        if v < best {
            best = v;
        }
    }
    best
}

fn should_stop_by_radius(
    graph: &WeightedGraph,
    cc_idx: &[usize],
    global_to_local: &[i32],
    cc_knn_radius: &[f32],
    prop_cent: usize,
    not_tested: &[bool],
    edge_threshold: f32,
) -> bool {
    let mut max_tested = f32::NEG_INFINITY;
    for (i, &r) in cc_knn_radius.iter().enumerate() {
        if !not_tested[i] {
            if r > max_tested {
                max_tested = r;
            }
        }
    }
    if cc_knn_radius[prop_cent] > max_tested {
        let mut cc_level_set: Vec<usize> = Vec::new();
        for i in 0..cc_knn_radius.len() {
            if cc_knn_radius[i] <= cc_knn_radius[prop_cent] {
                cc_level_set.push(i);
            }
        }
        let n_cc = count_components_for_subset(
            graph,
            cc_idx,
            global_to_local,
            &cc_level_set,
            edge_threshold,
        );
        return n_cc == 1;
    }
    false
}

fn count_components_for_subset(
    graph: &WeightedGraph,
    cc_idx: &[usize],
    global_to_local: &[i32],
    subset_local: &[usize],
    edge_threshold: f32,
) -> usize {
    let nc = cc_idx.len();
    let mut in_subset = vec![false; nc];
    for &li in subset_local {
        in_subset[li] = true;
    }

    let mut visited = vec![false; nc];
    let mut n_cc = 0usize;
    let mut stack: Vec<usize> = Vec::new();

    for &start in subset_local {
        if visited[start] {
            continue;
        }
        n_cc += 1;
        visited[start] = true;
        stack.push(start);

        while let Some(v_local) = stack.pop() {
            let v_global = cc_idx[v_local];
            for &(u_global, w) in &graph.adj[v_global] {
                if w > edge_threshold {
                    continue;
                }
                let u_local = global_to_local[u_global];
                if u_local >= 0 {
                    let u_local = u_local as usize;
                    if in_subset[u_local] && !visited[u_local] {
                        visited[u_local] = true;
                        stack.push(u_local);
                    }
                }
            }
        }
    }

    n_cc
}

fn connected_components_for_subset(
    graph: &WeightedGraph,
    cc_idx: &[usize],
    global_to_local: &[i32],
    subset_local: &[usize],
    edge_threshold: f32,
) -> Vec<i32> {
    let nc = cc_idx.len();
    let mut in_subset = vec![false; nc];
    for &li in subset_local {
        in_subset[li] = true;
    }

    let mut labels = vec![-1i32; nc];
    let mut comp_id = 0i32;
    let mut stack: Vec<usize> = Vec::new();

    for &start in subset_local {
        if labels[start] != -1 {
            continue;
        }
        labels[start] = comp_id;
        stack.push(start);

        while let Some(v_local) = stack.pop() {
            let v_global = cc_idx[v_local];
            for &(u_global, w) in &graph.adj[v_global] {
                if w > edge_threshold {
                    continue;
                }
                let u_local = global_to_local[u_global];
                if u_local >= 0 {
                    let u_local = u_local as usize;
                    if in_subset[u_local] && labels[u_local] == -1 {
                        labels[u_local] = comp_id;
                        stack.push(u_local);
                    }
                }
            }
        }

        comp_id += 1;
    }

    let mut out = Vec::with_capacity(subset_local.len());
    for &li in subset_local {
        out.push(labels[li]);
    }
    out
}
