use cpfcluster_rs::app::{CpfCluster, CpfConfig};
use cpfcluster_rs::data::Dataset;
use cpfcluster_rs::graph::{build_mutual_knn_graph, extract_components, knn_search_bruteforce};
use cpfcluster_rs::types::OUTLIER;
use cpfcluster_rs::vis::compute_visualization_data;

fn generate_data(n_per_cluster: usize, seed: u64) -> (Vec<f32>, Vec<i32>) {
    use rand::prelude::*;
    use rand_distr::Normal;

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0, 0.3).unwrap();

    let mut data: Vec<f32> = Vec::with_capacity(n_per_cluster * 3 * 2);
    let mut labels: Vec<i32> = Vec::with_capacity(n_per_cluster * 3);

    // Cluster 1: [1.5, 1.5]
    for _ in 0..n_per_cluster {
        let x = normal.sample(&mut rng) as f32 + 1.5;
        let y = normal.sample(&mut rng) as f32 + 1.5;
        data.push(x);
        data.push(y);
        labels.push(0);
    }
    // Cluster 2: [-1.0, -0.8]
    for _ in 0..n_per_cluster {
        let x = normal.sample(&mut rng) as f32 - 1.0;
        let y = normal.sample(&mut rng) as f32 - 0.8;
        data.push(x);
        data.push(y);
        labels.push(1);
    }
    // Cluster 3: [-0.2, -0.7]
    for _ in 0..n_per_cluster {
        let x = normal.sample(&mut rng) as f32 - 0.2;
        let y = normal.sample(&mut rng) as f32 - 0.7;
        data.push(x);
        data.push(y);
        labels.push(2);
    }

    (data, labels)
}

fn standardize_in_place(data: &mut [f32], n: usize, d: usize) {
    let mut mean = vec![0.0f32; d];
    let mut var = vec![0.0f32; d];

    for i in 0..n {
        for j in 0..d {
            mean[j] += data[i * d + j];
        }
    }
    for j in 0..d {
        mean[j] /= n as f32;
    }

    for i in 0..n {
        for j in 0..d {
            let diff = data[i * d + j] - mean[j];
            var[j] += diff * diff;
        }
    }
    for j in 0..d {
        var[j] /= n as f32;
    }

    for i in 0..n {
        for j in 0..d {
            let std = var[j].sqrt();
            let val = data[i * d + j];
            data[i * d + j] = if std > 0.0 { (val - mean[j]) / std } else { 0.0 };
        }
    }
}

fn adjusted_rand_index(labels_true: &[i32], labels_pred: &[i32]) -> f64 {
    use std::collections::HashMap;
    let n = labels_true.len();
    assert_eq!(n, labels_pred.len());

    let mut map_true: HashMap<i32, usize> = HashMap::new();
    let mut map_pred: HashMap<i32, usize> = HashMap::new();

    for &c in labels_true {
        if !map_true.contains_key(&c) {
            let next = map_true.len();
            map_true.insert(c, next);
        }
    }
    for &c in labels_pred {
        if !map_pred.contains_key(&c) {
            let next = map_pred.len();
            map_pred.insert(c, next);
        }
    }

    let n_true = map_true.len();
    let n_pred = map_pred.len();

    let mut table = vec![vec![0usize; n_pred]; n_true];
    for i in 0..n {
        let ti = map_true[&labels_true[i]];
        let pi = map_pred[&labels_pred[i]];
        table[ti][pi] += 1;
    }

    let comb2 = |x: usize| -> f64 { (x as f64) * ((x as f64) - 1.0) / 2.0 };

    let mut sum_comb = 0.0;
    let mut sum_true = 0.0;
    let mut sum_pred = 0.0;

    for i in 0..n_true {
        let mut row_sum = 0usize;
        for j in 0..n_pred {
            sum_comb += comb2(table[i][j]);
            row_sum += table[i][j];
        }
        sum_true += comb2(row_sum);
    }

    for j in 0..n_pred {
        let mut col_sum = 0usize;
        for i in 0..n_true {
            col_sum += table[i][j];
        }
        sum_pred += comb2(col_sum);
    }

    let total = comb2(n);
    if total == 0.0 {
        return 1.0;
    }

    let expected = (sum_true * sum_pred) / total;
    let max_index = 0.5 * (sum_true + sum_pred);
    if (max_index - expected).abs() < 1e-12 {
        return 1.0;
    }
    (sum_comb - expected) / (max_index - expected)
}

#[test]
fn test_generate_data_reproducibility() {
    let (d1, l1) = generate_data(50, 123);
    let (d2, l2) = generate_data(50, 123);
    assert_eq!(d1, d2);
    assert_eq!(l1, l2);
}

#[test]
fn test_generate_data_shape_and_labels() {
    let (data, labels) = generate_data(100, 42);
    assert_eq!(data.len(), 300 * 2);
    assert_eq!(labels.len(), 300);
    assert_eq!(labels.iter().filter(|&&c| c == 0).count(), 100);
    assert_eq!(labels.iter().filter(|&&c| c == 1).count(), 100);
    assert_eq!(labels.iter().filter(|&&c| c == 2).count(), 100);
}

#[test]
fn test_cluster_separation() {
    let (data, labels) = generate_data(100, 42);
    let n = labels.len();
    let d = 2;

    let mut centers = vec![[0.0f32; 2]; 3];
    let mut counts = vec![0usize; 3];

    for i in 0..n {
        let c = labels[i] as usize;
        centers[c][0] += data[i * d];
        centers[c][1] += data[i * d + 1];
        counts[c] += 1;
    }
    for c in 0..3 {
        centers[c][0] /= counts[c] as f32;
        centers[c][1] /= counts[c] as f32;
    }

    let mut distances = Vec::new();
    for i in 0..3 {
        for j in (i + 1)..3 {
            let dx = centers[i][0] - centers[j][0];
            let dy = centers[i][1] - centers[j][1];
            distances.push((dx * dx + dy * dy).sqrt());
        }
    }
    assert!(distances.iter().cloned().fold(0.0f32, f32::max) > 2.0);
}

#[test]
fn test_build_cc_graph_shapes() {
    let (mut data, _) = generate_data(50, 42);
    standardize_in_place(&mut data, 150, 2);
    let ds = Dataset::new(150, 2, data).unwrap();

    let knn = knn_search_bruteforce(&ds, 5);
    let graph = build_mutual_knn_graph(&knn);
    let comps = extract_components(&graph);

    assert_eq!(comps.len(), 150);
    let valid_radii = knn.radius.iter().filter(|&&r| r.is_finite());
    assert!(valid_radii.clone().all(|&r| r >= 0.0));
}

#[test]
fn test_cpf_fit_and_quality() {
    let (mut data, labels_true) = generate_data(100, 42);
    standardize_in_place(&mut data, 300, 2);
    let ds = Dataset::new(300, 2, data).unwrap();

    let cfg = CpfConfig {
        min_samples: 10,
        rho: vec![0.4],
        alpha: vec![1.0],
        ..CpfConfig::default()
    };
    let cpf = CpfCluster::new(cfg);
    let results = cpf.fit(&ds, None);
    assert!(!results.is_empty());

    let labels = &results[0].labels;
    assert_eq!(labels.len(), 300);

    let n_clusters = labels.iter().filter(|&&c| c != OUTLIER).collect::<std::collections::HashSet<_>>().len();
    assert!(n_clusters >= 2);

    let ari = adjusted_rand_index(&labels_true, labels);
    assert!(ari > 0.5, "ARI too low: {}", ari);

    let labels2 = cpf.fit(&ds, None)[0].labels.clone();
    assert_eq!(*labels, labels2);
}

#[test]
fn test_fixed_data_clustering_bounds() {
    let (mut data, _labels_true) = generate_data(100, 42);
    standardize_in_place(&mut data, 300, 2);
    let ds = Dataset::new(300, 2, data).unwrap();

    let cfg = CpfConfig {
        min_samples: 10,
        rho: vec![0.4],
        alpha: vec![1.0],
        ..CpfConfig::default()
    };
    let cpf = CpfCluster::new(cfg);
    let labels = &cpf.fit(&ds, None)[0].labels;

    let n_clusters = labels.iter().filter(|&&c| c != OUTLIER).collect::<std::collections::HashSet<_>>().len();
    let n_outliers = labels.iter().filter(|&&c| c == OUTLIER).count();

    assert!(n_clusters >= 2 && n_clusters <= 6, "unexpected clusters: {}", n_clusters);
    assert!(n_outliers < 20, "too many outliers: {}", n_outliers);
}

#[test]
fn test_visualization_data_consistency() {
    let (mut data, _labels_true) = generate_data(50, 42);
    standardize_in_place(&mut data, 150, 2);
    let ds = Dataset::new(150, 2, data).unwrap();

    let v1 = compute_visualization_data(&ds, 5);
    let v2 = compute_visualization_data(&ds, 5);

    assert_eq!(v1.edges.len(), v2.edges.len());
    assert!(v1.rho.iter().zip(v2.rho.iter()).all(|(a, b)| (*a - *b).abs() < 1e-6));
    assert!(v1.delta.iter().zip(v2.delta.iter()).all(|(a, b)| (*a - *b).abs() < 1e-6));
    assert_eq!(v1.big_brother, v2.big_brother);
}

#[test]
fn test_visualization_data_properties() {
    let (mut data, _labels_true) = generate_data(50, 42);
    standardize_in_place(&mut data, 150, 2);
    let ds = Dataset::new(150, 2, data).unwrap();

    let v = compute_visualization_data(&ds, 5);
    assert!(!v.edges.is_empty());
    assert!(v.rho.iter().all(|&r| r > 0.0));
    assert!(v.delta.iter().all(|&d| d >= 0.0));

    let n = v.rho.len();
    let max_idx = v.rho.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
    assert_eq!(v.big_brother[max_idx], -1);

    for (i, &bb) in v.big_brother.iter().enumerate() {
        assert!(bb == -1 || (bb >= 0 && (bb as usize) < n), "invalid big_brother at {}", i);
    }
}
