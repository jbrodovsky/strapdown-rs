use strapdown::add as add_strapdown;

use pyo3::prelude::*;

#[pyfunction]
fn add(a: f64, b: f64) -> PyResult<f64> {
    Ok(add_strapdown(a, b))
}

#[pymodule]
fn strapdown_py(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(add, m)?)?;
    Ok(())
}