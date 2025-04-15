use pyo3::prelude::*;

#[pyfunction]
fn add(a: f64, b: f64) -> f64 {
    strapdown_core::add(a, b)
}

#[pymodule]
fn strapdown_py(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(add, m)?)?;
    Ok(())
}