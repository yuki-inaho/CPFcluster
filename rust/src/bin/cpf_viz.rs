use cpfcluster_rs::app::{CpfCluster, CpfConfig};
use cpfcluster_rs::data::Dataset;
use cpfcluster_rs::vis::{compute_visualization_data, plot_four_panels};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut output = "outputs/cpf_process_viz.png".to_string();
    let mut n_points = 150usize;
    let mut seed = 42u64;
    let mut k = 10usize;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--output" | "-o" => {
                if let Some(val) = args.next() {
                    output = val;
                }
            }
            "--n-points" => {
                if let Some(val) = args.next() {
                    n_points = val.parse().unwrap_or(n_points);
                }
            }
            "--seed" => {
                if let Some(val) = args.next() {
                    seed = val.parse().unwrap_or(seed);
                }
            }
            "--k" => {
                if let Some(val) = args.next() {
                    k = val.parse().unwrap_or(k);
                }
            }
            _ => {}
        }
    }

    std::fs::create_dir_all("outputs")?;

    let (mut data, _labels) = generate_data(n_points, seed);
    standardize_in_place(&mut data, n_points * 3, 2);
    let dataset = Dataset::new(n_points * 3, 2, data)?;

    let cfg = CpfConfig {
        min_samples: k,
        rho: vec![0.4],
        alpha: vec![1.0],
        ..CpfConfig::default()
    };
    let cpf = CpfCluster::new(cfg);
    let results = cpf.fit(&dataset, None);

    let labels = if results.is_empty() {
        vec![0i32; dataset.n()]
    } else {
        results[0].labels.clone()
    };

    let vis = compute_visualization_data(&dataset, k);
    plot_four_panels(&dataset, &labels, &vis, &output)?;
    println!("Visualization saved to: {}", output);
    Ok(())
}

fn generate_data(n_per_cluster: usize, seed: u64) -> (Vec<f32>, Vec<i32>) {
    use rand::prelude::*;
    use rand_distr::Normal;

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0, 0.3).unwrap();

    let mut data: Vec<f32> = Vec::with_capacity(n_per_cluster * 3 * 2);
    let mut labels: Vec<i32> = Vec::with_capacity(n_per_cluster * 3);

    for _ in 0..n_per_cluster {
        let x = normal.sample(&mut rng) as f32 + 1.5;
        let y = normal.sample(&mut rng) as f32 + 1.5;
        data.push(x);
        data.push(y);
        labels.push(0);
    }
    for _ in 0..n_per_cluster {
        let x = normal.sample(&mut rng) as f32 - 1.0;
        let y = normal.sample(&mut rng) as f32 - 0.8;
        data.push(x);
        data.push(y);
        labels.push(1);
    }
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
