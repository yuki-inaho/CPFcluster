//! Connected components and outlier filtering.

use crate::graph::adjacency::WeightedGraph;
use crate::types::OUTLIER;

#[derive(Debug, Clone, Copy)]
pub enum OutlierFilter {
    None,
    EdgeCount,
    ComponentSize,
}

pub fn extract_components(graph: &WeightedGraph) -> Vec<i32> {
    let n = graph.n();
    let mut labels = vec![-1i32; n];
    let mut comp_id = 0i32;

    let mut stack: Vec<usize> = Vec::new();
    for start in 0..n {
        if labels[start] != -1 {
            continue;
        }
        labels[start] = comp_id;
        stack.push(start);

        while let Some(v) = stack.pop() {
            for &(u, _) in &graph.adj[v] {
                if labels[u] == -1 {
                    labels[u] = comp_id;
                    stack.push(u);
                }
            }
        }

        comp_id += 1;
    }

    labels
}

pub fn filter_by_edge_count(
    graph: &WeightedGraph,
    components: &[i32],
    min_edges: usize,
) -> Vec<i32> {
    let n = graph.n();
    let mut out = components.to_vec();
    for i in 0..n {
        if graph.degree(i) <= min_edges {
            out[i] = OUTLIER;
        }
    }
    out
}

pub fn filter_by_component_size(components: &[i32], min_size: usize) -> Vec<i32> {
    let mut counts: std::collections::HashMap<i32, usize> = std::collections::HashMap::new();
    for &c in components {
        *counts.entry(c).or_insert(0) += 1;
    }

    let mut out = components.to_vec();
    for (i, &c) in components.iter().enumerate() {
        if let Some(&sz) = counts.get(&c) {
            if sz <= min_size {
                out[i] = OUTLIER;
            }
        }
    }

    out
}

pub fn apply_outlier_filter(
    graph: &WeightedGraph,
    components: &[i32],
    filter: OutlierFilter,
    cutoff: usize,
) -> Vec<i32> {
    match filter {
        OutlierFilter::None => components.to_vec(),
        OutlierFilter::EdgeCount => filter_by_edge_count(graph, components, cutoff),
        OutlierFilter::ComponentSize => filter_by_component_size(components, cutoff),
    }
}
