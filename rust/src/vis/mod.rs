//! Visualization utilities (data prep + plotting).

use crate::data::Dataset;
use crate::math::euclidean;
use plotters::prelude::*;

#[derive(Debug, Clone)]
pub struct VisualizationData {
    pub edges: Vec<(usize, usize)>,
    pub rho: Vec<f32>,
    pub delta: Vec<f32>,
    pub big_brother: Vec<i32>,
}

/// Reproduce the visualization data from demo_cpf_visualize.py (adj/rho/delta/big_brother).
pub fn compute_visualization_data(dataset: &Dataset, k: usize) -> VisualizationData {
    let n = dataset.n();

    let mut indices: Vec<Vec<usize>> = Vec::with_capacity(n);
    let mut distances: Vec<Vec<f32>> = Vec::with_capacity(n);

    for i in 0..n {
        let xi = dataset.row(i);
        let mut dv: Vec<(f32, usize)> = Vec::with_capacity(n);
        for j in 0..n {
            let d = euclidean(xi, dataset.row(j));
            dv.push((d, j));
        }
        dv.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let take = (k + 1).min(dv.len());
        let mut neigh = Vec::with_capacity(take);
        let mut dist = Vec::with_capacity(take);
        for t in 0..take {
            dist.push(dv[t].0);
            neigh.push(dv[t].1);
        }
        indices.push(neigh);
        distances.push(dist);
    }

    // Step 1: mutual kNN edges.
    let mut neighbor_sets: Vec<Vec<usize>> = Vec::with_capacity(n);
    for i in 0..n {
        let mut v = indices[i].clone();
        v.sort_unstable();
        neighbor_sets.push(v);
    }

    let mut edges: Vec<(usize, usize)> = Vec::new();
    for i in 0..n {
        for &j in indices[i].iter().skip(1).take(k) {
            if neighbor_sets[j].binary_search(&i).is_ok() {
                edges.push((i, j));
            }
        }
    }

    // Step 2: density (rho) from k-th neighbor distance.
    let mut rho = vec![0.0f32; n];
    for i in 0..n {
        let k_dist = *distances[i].last().unwrap_or(&0.0);
        rho[i] = 1.0 / (k_dist + 1e-10);
    }

    // Step 3: delta + big brother
    let mut sorted: Vec<usize> = (0..n).collect();
    sorted.sort_by(|&a, &b| rho[b].partial_cmp(&rho[a]).unwrap_or(std::cmp::Ordering::Equal));

    let mut delta = vec![0.0f32; n];
    let mut big_brother = vec![-1i32; n];

    let mut max_dist = 0.0f32;
    for i in 0..n {
        for j in (i + 1)..n {
            let d = euclidean(dataset.row(i), dataset.row(j));
            if d > max_dist {
                max_dist = d;
            }
        }
    }

    for (rank, &idx) in sorted.iter().enumerate() {
        let higher = &sorted[..rank];
        if higher.is_empty() {
            delta[idx] = max_dist;
            continue;
        }
        let mut best = (f32::INFINITY, -1i32);
        for &h in higher {
            let d = euclidean(dataset.row(idx), dataset.row(h));
            if d < best.0 {
                best = (d, h as i32);
            }
        }
        delta[idx] = best.0;
        big_brother[idx] = best.1;
    }

    VisualizationData {
        edges,
        rho,
        delta,
        big_brother,
    }
}

/// Render the 4-panel CPF visualization (approximate match to Python demo).
pub fn plot_four_panels(
    dataset: &Dataset,
    labels: &[i32],
    vis: &VisualizationData,
    output_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(output_path, (1400, 1000)).into_drawing_area();
    root.fill(&WHITE)?;
    let areas = root.split_evenly((2, 2));

    let (min_x, max_x, min_y, max_y) = data_bounds(dataset);
    let x_range = min_x..max_x;
    let y_range = min_y..max_y;

    // Panel 1: mutual kNN graph
    {
        let mut chart = ChartBuilder::on(&areas[0])
            .margin(10)
            .caption("1. Construct the Mutual k-NN Graph.", ("sans-serif", 20))
            .build_cartesian_2d(x_range.clone(), y_range.clone())?;
        chart
            .configure_mesh()
            .disable_mesh()
            .x_labels(0)
            .y_labels(0)
            .draw()?;

        for &(u, v) in &vis.edges {
            let (xu, yu) = point_at(dataset, u);
            let (xv, yv) = point_at(dataset, v);
            chart.draw_series(std::iter::once(PathElement::new(
                vec![(xu, yu), (xv, yv)],
                RGBAColor(211, 211, 211, 0.6).stroke_width(1),
            )))?;
        }

        chart.draw_series((0..dataset.n()).map(|i| {
            let (x, y) = point_at(dataset, i);
            Circle::new((x, y), 3, ShapeStyle::from(&RGBColor(120, 120, 120)).filled())
        }))?;
    }

    // Panel 2: peak-finding criterion
    {
        let mut chart = ChartBuilder::on(&areas[1])
            .margin(10)
            .caption("2. Compute the Peak-Finding Criterion.", ("sans-serif", 20))
            .build_cartesian_2d(x_range.clone(), y_range.clone())?;
        chart
            .configure_mesh()
            .disable_mesh()
            .x_labels(0)
            .y_labels(0)
            .draw()?;

        for &(u, v) in &vis.edges {
            let (xu, yu) = point_at(dataset, u);
            let (xv, yv) = point_at(dataset, v);
            chart.draw_series(std::iter::once(PathElement::new(
                vec![(xu, yu), (xv, yv)],
                RGBAColor(211, 211, 211, 0.3).stroke_width(1),
            )))?;
        }

        let max_delta = vis.delta.iter().cloned().fold(0.0f32, f32::max).max(1e-6);
        let max_rho = vis.rho.iter().cloned().fold(0.0f32, f32::max).max(1e-6);
        chart.draw_series((0..dataset.n()).map(|i| {
            let (x, y) = point_at(dataset, i);
            let size = 3 + ((vis.delta[i] / max_delta) * 8.0) as i32;
            let color = magma_color(vis.rho[i] / max_rho);
            Circle::new((x, y), size, ShapeStyle::from(&color).filled())
        }))?;
    }

    // Panel 3: assess potential centers
    {
        let mut chart = ChartBuilder::on(&areas[2])
            .margin(10)
            .caption("3. Assess Potential Centers.", ("sans-serif", 20))
            .build_cartesian_2d(x_range.clone(), y_range.clone())?;
        chart
            .configure_mesh()
            .disable_mesh()
            .x_labels(0)
            .y_labels(0)
            .draw()?;

        // background points
        chart.draw_series((0..dataset.n()).map(|i| {
            let (x, y) = point_at(dataset, i);
            Circle::new((x, y), 2, RGBAColor(128, 128, 128, 0.3).filled())
        }))?;

        // edges
        for &(u, v) in &vis.edges {
            let (xu, yu) = point_at(dataset, u);
            let (xv, yv) = point_at(dataset, v);
            chart.draw_series(std::iter::once(PathElement::new(
                vec![(xu, yu), (xv, yv)],
                RGBAColor(128, 0, 128, 0.2).stroke_width(1),
            )))?;
        }

        let centers = top_centers(vis, 3);
        chart.draw_series(centers.iter().map(|&i| {
            let (x, y) = point_at(dataset, i);
            TriangleMarker::new((x, y), 10, ShapeStyle::from(&YELLOW).filled().stroke_width(1))
        }))?;
    }

    // Panel 4: assign remaining instances
    {
        let mut chart = ChartBuilder::on(&areas[3])
            .margin(10)
            .caption("4. Assign Remaining Instances.", ("sans-serif", 20))
            .build_cartesian_2d(x_range.clone(), y_range.clone())?;
        chart
            .configure_mesh()
            .disable_mesh()
            .x_labels(0)
            .y_labels(0)
            .draw()?;

        chart.draw_series((0..dataset.n()).map(|i| {
            let (x, y) = point_at(dataset, i);
            let color = label_color(labels[i]);
            Circle::new((x, y), 3, ShapeStyle::from(&color).filled())
        }))?;

        let centers = top_centers(vis, 3);
        chart.draw_series(centers.iter().map(|&i| {
            let (x, y) = point_at(dataset, i);
            TriangleMarker::new((x, y), 10, ShapeStyle::from(&YELLOW).filled().stroke_width(1))
        }))?;

        let start = argmin(&vis.rho);
        let mut path = vec![start];
        let mut curr = start;
        for _ in 0..20 {
            let parent = vis.big_brother[curr];
            if parent < 0 || parent as usize == curr {
                break;
            }
            curr = parent as usize;
            path.push(curr);
        }
        if path.len() > 1 {
            let coords: Vec<(f32, f32)> = path.iter().map(|&i| point_at(dataset, i)).collect();
            chart.draw_series(std::iter::once(PathElement::new(
                coords,
                ShapeStyle::from(&RGBColor(255, 215, 0)).stroke_width(3),
            )))?;
        }
    }

    root.present()?;
    Ok(())
}

fn data_bounds(dataset: &Dataset) -> (f32, f32, f32, f32) {
    let mut min_x = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_y = f32::NEG_INFINITY;
    for i in 0..dataset.n() {
        let (x, y) = point_at(dataset, i);
        min_x = min_x.min(x);
        max_x = max_x.max(x);
        min_y = min_y.min(y);
        max_y = max_y.max(y);
    }
    let pad_x = (max_x - min_x) * 0.05;
    let pad_y = (max_y - min_y) * 0.05;
    (min_x - pad_x, max_x + pad_x, min_y - pad_y, max_y + pad_y)
}

fn point_at(dataset: &Dataset, idx: usize) -> (f32, f32) {
    let row = dataset.row(idx);
    (row[0], row[1])
}

fn argmin(values: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_v = f32::INFINITY;
    for (i, &v) in values.iter().enumerate() {
        if v < best_v {
            best_v = v;
            best = i;
        }
    }
    best
}

fn top_centers(vis: &VisualizationData, k: usize) -> Vec<usize> {
    let mut gamma: Vec<(f32, usize)> = vis
        .rho
        .iter()
        .zip(vis.delta.iter())
        .enumerate()
        .map(|(i, (&r, &d))| (r * d, i))
        .collect();
    gamma.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    gamma.iter().rev().take(k).map(|&(_, i)| i).collect()
}

fn magma_color(t: f32) -> RGBColor {
    let t = t.clamp(0.0, 1.0);
    let r = (255.0 * (0.5 + 0.5 * t)) as u8;
    let g = (255.0 * (0.1 + 0.4 * t)) as u8;
    let b = (255.0 * (0.2 + 0.2 * (1.0 - t))) as u8;
    RGBColor(r, g, b)
}

fn label_color(label: i32) -> RGBColor {
    if label < 0 {
        return RGBColor(160, 160, 160);
    }
    let palette = [
        RGBColor(68, 1, 84),
        RGBColor(58, 82, 139),
        RGBColor(32, 144, 140),
        RGBColor(94, 201, 98),
        RGBColor(253, 231, 37),
        RGBColor(122, 4, 2),
        RGBColor(0, 109, 44),
        RGBColor(68, 117, 180),
    ];
    let idx = (label as usize) % palette.len();
    palette[idx]
}
