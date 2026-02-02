//! Dataset container (row-major f32).

#[derive(Debug, Clone)]
pub struct Dataset {
    n: usize,
    d: usize,
    data: Vec<f32>,
}

impl Dataset {
    pub fn new(n: usize, d: usize, data: Vec<f32>) -> Result<Self, String> {
        if data.len() != n * d {
            return Err(format!(
                "Dataset::new: data length mismatch (expected {}, got {})",
                n * d,
                data.len()
            ));
        }
        Ok(Self { n, d, data })
    }

    pub fn n(&self) -> usize {
        self.n
    }

    pub fn d(&self) -> usize {
        self.d
    }

    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    pub fn row(&self, i: usize) -> &[f32] {
        let start = i * self.d;
        &self.data[start..start + self.d]
    }
}
