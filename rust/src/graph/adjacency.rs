//! Adjacency construction for mutual kNN graphs.

use crate::graph::knn::KnnResult;

#[derive(Debug, Clone)]
pub struct WeightedGraph {
    pub adj: Vec<Vec<(usize, f32)>>, // undirected adjacency list
}

impl WeightedGraph {
    pub fn new(n: usize) -> Self {
        Self {
            adj: vec![Vec::new(); n],
        }
    }

    pub fn n(&self) -> usize {
        self.adj.len()
    }

    pub fn degree(&self, i: usize) -> usize {
        self.adj[i].len()
    }
}

/// Build the mutual kNN graph (Definition 7).
/// An undirected edge (i,j) exists iff i and j are in each other's kNN lists.
pub fn build_mutual_knn_graph(knn: &KnnResult) -> WeightedGraph {
    let n = knn.indices.len();
    let k = knn.k;
    let mut graph = WeightedGraph::new(n);

    // Prepare neighbor sets for mutual check.
    let mut neighbor_sets: Vec<Vec<usize>> = Vec::with_capacity(n);
    for i in 0..n {
        let mut row = knn.indices[i].clone();
        row.sort_unstable();
        neighbor_sets.push(row);
    }

    let mut has_self = vec![false; n];

    for i in 0..n {
        for pos in 0..k.min(knn.indices[i].len()) {
            let j = knn.indices[i][pos];
            let dij = knn.distances[i][pos];

            // mutual kNN check
            if neighbor_sets[j].binary_search(&i).is_err() {
                continue;
            }

            if i == j {
                if !has_self[i] {
                    graph.adj[i].push((i, dij));
                    has_self[i] = true;
                }
                continue;
            }

            if i < j {
                graph.adj[i].push((j, dij));
                graph.adj[j].push((i, dij));
            }
        }
    }

    graph
}
