use cpfcluster_rs::core::{compute_big_brother, compute_peak_score};
use cpfcluster_rs::data::Dataset;
use cpfcluster_rs::graph::{build_mutual_knn_graph, extract_components, knn_search_bruteforce};
use cpfcluster_rs::types::NO_PARENT;

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
fn test_compute_peak_score_basic() {
    let parent = vec![1.0f32, 2.0, 0.5];
    let radius = vec![0.5f32, 1.0, 0.25];
    let peaked = compute_peak_score(&parent, &radius);
    assert!((peaked[0] - 2.0).abs() < 1e-6);
    assert!((peaked[1] - 2.0).abs() < 1e-6);
    assert!((peaked[2] - 2.0).abs() < 1e-6);
}

#[test]
fn test_compute_peak_score_zero_radius() {
    let parent = vec![1.0f32, 0.0];
    let radius = vec![0.0f32, 0.0];
    let peaked = compute_peak_score(&parent, &radius);
    assert!(peaked[0].is_infinite());
    assert!(peaked[1].is_infinite());
}

#[test]
fn test_compute_big_brother_shapes() {
    let ds = make_dataset(60, 2, 99);
    let k = 5;
    let knn = knn_search_bruteforce(&ds, k);
    let graph = build_mutual_knn_graph(&knn);
    let components = extract_components(&graph);

    let result = compute_big_brother(&ds, &knn.radius, &components, k);
    assert_eq!(result.parent.len(), 60);
    assert_eq!(result.parent_dist.len(), 60);

    let valid = result
        .parent
        .iter()
        .all(|&p| p == NO_PARENT || (p >= 0 && (p as usize) < 60));
    assert!(valid);
}
