use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use pyo3_polars::export::polars_core::utils::CustomIterTools;

#[polars_expr(output_type=Float64)]
fn stateful_acc(inputs: &[Series]) -> PolarsResult<Series> {
    const NEW_WEIGHT: f64 = 0.03;
    let old_weight: f64 = (1.0 - NEW_WEIGHT * NEW_WEIGHT).sqrt();
    let input_series = &inputs[0];
    let mut acc: f64 = input_series.f64()?.into_no_null_iter().next().unwrap();
    let input_iter = input_series.f64()?.into_no_null_iter();
    // acc has value of first element of input series
    let output_values: ChunkedArray<Float64Type> = input_iter
        .map(|val| {
            acc = acc * old_weight + val * NEW_WEIGHT;
            acc
        })
        .collect_ca(input_series.name().clone());
    Ok(output_values.into_series())
}

#[polars_expr(output_type=Float64)]
fn vertical_scan(inputs: &[Series]) -> PolarsResult<Series> {
    const NEW_WEIGHT: f64 = 0.03;
    let old_weight: f64 = (1.0 - NEW_WEIGHT * NEW_WEIGHT).sqrt();
    let s = &inputs[0];
    // init state is first value of series, which we unwrap safely because it can't be null
    let init_state: f64 = s.f64()?.get(0).unwrap();
    let ca: &Float64Chunked = s.f64()?;
    let out: Float64Chunked = ca
        .iter()
        .scan(init_state, |state: &mut f64, x: Option<f64>| match x {
            Some(x) => {
                *state = *state * old_weight + x * NEW_WEIGHT;
                Some(Some(*state))
            }
            None => Some(Some(*state)),
        })
        .collect_trusted();
    Ok(out.into_series())
}
