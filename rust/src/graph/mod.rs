//! Graph layer: kNN search, mutual adjacency, components.

pub mod knn;
pub mod adjacency;
pub mod components;

pub use knn::{knn_search_bruteforce, KnnResult};
pub use adjacency::{build_mutual_knn_graph, WeightedGraph};
pub use components::{
    apply_outlier_filter, extract_components, filter_by_component_size, filter_by_edge_count,
    OutlierFilter,
};
