//! Math helpers for distance computations.

#[inline]
pub fn squared_euclidean(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = 0.0f32;
    for i in 0..a.len() {
        let diff = a[i] - b[i];
        acc += diff * diff;
    }
    acc
}

#[inline]
pub fn euclidean(a: &[f32], b: &[f32]) -> f32 {
    squared_euclidean(a, b).sqrt()
}
