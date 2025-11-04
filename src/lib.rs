use pyo3::prelude::*;
use pyo3_polars::PolarsAllocator;

mod expressions;


#[global_allocator]
static ALLOC: PolarsAllocator = PolarsAllocator::new();

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

/// A Python module implemented in Rust.
#[pymodule]
fn lppm_expr(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    Ok(())
}
