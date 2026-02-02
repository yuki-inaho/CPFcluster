use numpy::{IntoPyArray, PyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use cpfcluster_rs::app::{CpfCluster, CpfConfig};
use cpfcluster_rs::data::Dataset;

fn dataset_from_numpy(array: PyReadonlyArray2<f32>) -> PyResult<Dataset> {
    let arr = array.as_array();
    let (n, d) = arr.dim();
    let mut data = Vec::with_capacity(n * d);
    for i in 0..n {
        for j in 0..d {
            data.push(arr[(i, j)]);
        }
    }
    Dataset::new(n, d, data).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))
}

#[pyfunction]
fn cpf_fit(
    py: Python<'_>,
    x: PyReadonlyArray2<f32>,
    min_samples: usize,
    rho: Vec<f32>,
    alpha: Vec<f32>,
    cutoff: usize,
) -> PyResult<Py<PyArray1<i32>>> {
    let dataset = dataset_from_numpy(x)?;
    let cfg = CpfConfig {
        min_samples,
        rho,
        alpha,
        cutoff,
        ..CpfConfig::default()
    };
    let cpf = CpfCluster::new(cfg);
    let results = cpf.fit(&dataset, None);
    let labels = if results.is_empty() {
        vec![0i32; dataset.n()]
    } else {
        results[0].labels.clone()
    };
    Ok(labels.into_pyarray(py).to_owned())
}

#[pymodule]
fn cpfcluster_pyo3(py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(cpf_fit, m)?)?;
    // minimal API: return labels; visualization stays in Python
    let _ = py;
    Ok(())
}
