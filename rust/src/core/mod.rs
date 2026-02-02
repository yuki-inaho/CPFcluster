//! Core layer: peak score, big brother, center selection, assignment.

pub mod peak_score;
pub mod big_brother;
pub mod center_selection;
pub mod assignment;

pub use peak_score::compute_peak_score;
pub use big_brother::{compute_big_brother, BigBrotherResult};
pub use center_selection::select_centers_for_component;
pub use assignment::assign_labels_for_component;
