use cpfcluster_rs::data::Dataset;
use cpfcluster_rs::graph::{
    apply_outlier_filter, build_mutual_knn_graph, extract_components, knn_search_bruteforce,
    OutlierFilter,
};

fn make_dataset(n: usize, d: usize, seed: u64) -> Dataset {
    use rand::prelude::*;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut data = vec![0.0f32; n * d];
    for i in 0..n * d {
        data[i] = rng.gen::<f32>();
    }
    Dataset::new(n, d, data).unwrap()
}

#[test]
fn test_knn_search_shape_and_self() {
    let ds = make_dataset(50, 3, 42);
    let k = 5;
    let knn = knn_search_bruteforce(&ds, k);

    assert_eq!(knn.indices.len(), 50);
    assert_eq!(knn.distances.len(), 50);
    assert_eq!(knn.radius.len(), 50);
    assert_eq!(knn.indices[0].len(), k);
    assert_eq!(knn.distances[0].len(), k);

    for i in 0..50 {
        assert_eq!(knn.indices[i][0], i);
        assert_eq!(knn.distances[i][0], 0.0);
    }
}

#[test]
fn test_mutual_graph_symmetry() {
    let ds = make_dataset(40, 2, 123);
    let knn = knn_search_bruteforce(&ds, 4);
    let graph = build_mutual_knn_graph(&knn);

    for i in 0..graph.n() {
        for &(j, _) in &graph.adj[i] {
            let has_back = graph.adj[j].iter().any(|&(u, _)| u == i);
            assert!(has_back, "edge {}->{} must be symmetric", i, j);
        }
    }
}

#[test]
fn test_extract_components_and_outlier_filter() {
    let ds = make_dataset(30, 2, 7);
    let knn = knn_search_bruteforce(&ds, 3);
    let graph = build_mutual_knn_graph(&knn);
    let comps = extract_components(&graph);
    assert_eq!(comps.len(), 30);

    let filtered = apply_outlier_filter(&graph, &comps, OutlierFilter::EdgeCount, 1);
    assert_eq!(filtered.len(), 30);
    assert!(filtered.iter().all(|&c| c >= -1));
}
