//! Application layer: high-level CPF clustering API.

use crate::core::{assign_labels_for_component, compute_big_brother, compute_peak_score};
use crate::core::select_centers_for_component;
use crate::data::Dataset;
use crate::graph::{
    apply_outlier_filter, build_mutual_knn_graph, extract_components, knn_search, KnnBackend,
    OutlierFilter, WeightedGraph,
};
use crate::types::OUTLIER;

/// Outlier detection strategy (paper vs. original implementation).
#[derive(Debug, Clone, Copy)]
pub enum OutlierMethod {
    EdgeCount,
    ComponentSize,
}

impl From<OutlierMethod> for OutlierFilter {
    fn from(v: OutlierMethod) -> Self {
        match v {
            OutlierMethod::EdgeCount => OutlierFilter::EdgeCount,
            OutlierMethod::ComponentSize => OutlierFilter::ComponentSize,
        }
    }
}

/// High-level configuration for CPF.
#[derive(Debug, Clone)]
pub struct CpfConfig {
    pub min_samples: usize,
    pub rho: Vec<f32>,
    pub alpha: Vec<f32>,
    pub cutoff: usize,
    pub outlier_method: OutlierMethod,
    pub knn_backend: KnnBackend,
    pub merge: bool,
    pub merge_threshold: Vec<f32>,
    pub density_ratio_threshold: Vec<f32>,
}

impl Default for CpfConfig {
    fn default() -> Self {
        Self {
            min_samples: 5,
            rho: vec![0.4],
            alpha: vec![1.0],
            cutoff: 1,
            outlier_method: OutlierMethod::EdgeCount,
            knn_backend: KnnBackend::KdTree,
            merge: false,
            merge_threshold: vec![0.5],
            density_ratio_threshold: vec![0.1],
        }
    }
}

/// Mutual kNN graph + components + r_k(x).
#[derive(Debug, Clone)]
pub struct CcGraph {
    pub graph: WeightedGraph,
    pub components: Vec<i32>,
    pub radius: Vec<f32>,
}

/// Parameter set used for grid search outputs.
#[derive(Debug, Clone, Copy)]
pub struct Params {
    pub k: usize,
    pub rho: f32,
    pub alpha: f32,
    pub merge_threshold: f32,
    pub density_ratio_threshold: f32,
}

/// One clustering result for a parameter combination.
#[derive(Debug, Clone)]
pub struct FitResult {
    pub params: Params,
    pub labels: Vec<i32>,
}

/// High-level CPF clusterer (mirrors Python CPFcluster).
#[derive(Debug, Clone)]
pub struct CpfCluster {
    cfg: CpfConfig,
}

impl CpfCluster {
    pub fn new(cfg: CpfConfig) -> Self {
        Self { cfg }
    }

    /// Build mutual kNN graph, components, and kNN radius (Algorithm 2, Steps 1-2).
    pub fn build_cc_graph(&self, dataset: &Dataset, k: usize) -> CcGraph {
        let knn = knn_search(dataset, k, self.cfg.knn_backend);
        let graph = build_mutual_knn_graph(&knn);
        let comps = extract_components(&graph);
        let comps = apply_outlier_filter(&graph, &comps, self.cfg.outlier_method.into(), self.cfg.cutoff);
        CcGraph {
            graph,
            components: comps,
            radius: knn.radius,
        }
    }

    /// Fit the model for all parameter combinations (grid search).
    pub fn fit(&self, dataset: &Dataset, k_values: Option<&[usize]>) -> Vec<FitResult> {
        let ks: Vec<usize> = k_values.map(|v| v.to_vec()).unwrap_or_else(|| vec![self.cfg.min_samples]);
        let mut out = Vec::new();

        for k in ks {
            let cc = self.build_cc_graph(dataset, k);
            let bb = compute_big_brother(dataset, &cc.radius, &cc.components, k);
            let best_distance = bb.parent_dist;
            let big_brother = bb.parent;

            for &rho in &self.cfg.rho {
                for &alpha in &self.cfg.alpha {
                    for &mt in &self.cfg.merge_threshold {
                        for &drt in &self.cfg.density_ratio_threshold {
                            let labels = self.get_labels_for_params(
                                dataset,
                                &cc,
                                &best_distance,
                                &big_brother,
                                rho,
                                alpha,
                            );
                            out.push(FitResult {
                                params: Params { k, rho, alpha, merge_threshold: mt, density_ratio_threshold: drt },
                                labels,
                            });
                        }
                    }
                }
            }
        }

        out
    }

    /// Cluster assignment for a single (rho, alpha) pair.
    fn get_labels_for_params(
        &self,
        dataset: &Dataset,
        cc: &CcGraph,
        best_distance: &[f32],
        big_brother: &[i32],
        rho: f32,
        alpha: f32,
    ) -> Vec<i32> {
        let n = dataset.n();
        let d = dataset.d();
        let mut labels = vec![OUTLIER; n];

        let mut comp_map: std::collections::BTreeMap<i32, Vec<usize>> =
            std::collections::BTreeMap::new();
        for i in 0..n {
            let c = cc.components[i];
            if c != OUTLIER {
                comp_map.entry(c).or_default().push(i);
            }
        }

        let mut label_offset: i32 = 0;
        for (_cid, cc_idx) in comp_map {
            if cc_idx.is_empty() {
                continue;
            }

            let mut cc_radius = Vec::with_capacity(cc_idx.len());
            let mut cc_best = Vec::with_capacity(cc_idx.len());
            for &gi in &cc_idx {
                cc_radius.push(cc.radius[gi]);
                cc_best.push(best_distance[gi]);
            }
            let peaked = compute_peak_score(&cc_best, &cc_radius);
            if peaked.is_empty() {
                continue;
            }

            let centers = select_centers_for_component(
                &cc.graph,
                &cc_idx,
                &cc_radius,
                &peaked,
                rho,
                alpha,
                d,
            );

            let (n_comp, local_labels) =
                assign_labels_for_component(&cc_idx, big_brother, &centers, label_offset);

            for (li, &gi) in cc_idx.iter().enumerate() {
                labels[gi] = local_labels[li];
            }
            label_offset += n_comp;
        }

        labels
    }
}
