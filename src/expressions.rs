use itertools::Itertools;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use pyo3_polars::export::polars_core::utils::rayon::iter::{
    IntoParallelIterator, ParallelIterator,
};
use pyo3_polars::export::polars_core::utils::CustomIterTools;
use pyo3_polars::export::polars_plan::dsl::lit;
use rand_distr::{Distribution, Normal};
use serde::Deserialize;

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
    let init_state: f64 = s
        .f64()?
        .get(0)
        .expect("First value of vertical scan can't be null");
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

#[polars_expr(output_type=Float64)]
fn lazy_fill_random(inputs: &[Series]) -> PolarsResult<Series> {
    let s = &inputs[0];
    let length = s.len();
    let mut rng = rand::rng();
    let normal = Normal::new(0.0, 1.0).unwrap();
    let out: Float64Chunked = (0..length).map(|_| Some(normal.sample(&mut rng))).collect();
    Ok(out.into_series())
}

#[derive(Deserialize)]
struct AddThresholdKwargs {
    threshold: f32,
}

#[polars_expr(output_type_func=is_social_link_output)]
fn is_social_link_par(inputs: &[Series], kwargs: AddThresholdKwargs) -> PolarsResult<Series> {
    let threshold = kwargs.threshold;
    let row_idx = inputs[0].u32()?;
    let user_id = inputs[1].i32()?;
    let lon_rad = inputs[2].f32()?;
    let lat_rad = inputs[3].f32()?;
    let event_start = inputs[4].i32()?;
    let event_end = inputs[5].i32()?;
    let offset = inputs[6].u32()?;

    let len = row_idx.len();

    let ((user1_vec, user2_vec), time_vec): ((Vec<_>, Vec<_>), Vec<_>) = (0..len)
        .into_par_iter()
        .flat_map(|i| {
            // SAFETY: bounds checked by iterator
            let user1 = unsafe { user_id.get_unchecked(i).unwrap() };
            let lon1 = unsafe { lon_rad.get_unchecked(i).unwrap() };
            let lat1 = unsafe { lat_rad.get_unchecked(i).unwrap() };
            let time_start1 = unsafe { event_start.get_unchecked(i).unwrap() };
            let time_end1 = unsafe { event_end.get_unchecked(i).unwrap() };
            let offset_i = unsafe { offset.get_unchecked(i).unwrap() as usize };

            (i..offset_i).into_par_iter().filter_map(move |j| {
                let user2 = unsafe { user_id.get_unchecked(j).unwrap() };
                if user1 == user2 {
                    return None;
                }
                let lon2 = unsafe { lon_rad.get_unchecked(j).unwrap() };
                let lat2 = unsafe { lat_rad.get_unchecked(j).unwrap() };
                if haversine(lon1, lat1, lon2, lat2) > threshold {
                    return None;
                }

                let time_start2 = unsafe { event_start.get_unchecked(j).unwrap() };
                let time_end2 = unsafe { event_end.get_unchecked(j).unwrap() };
                let time_together = time_end1.min(time_end2) - time_start2.max(time_start1);
                Some(((user1, user2), time_together as u32))
            })
        })
        .unzip();
    // get len now
    let res_len = user1_vec.len();
    // Create individual Series
    let user1_series = &Series::new("user1".into(), user1_vec);
    let user2_series = &Series::new("user2".into(), user2_vec);
    let time_series = &Series::new("time_together".into(), time_vec);
    let all_series = [user1_series, user2_series, time_series];

    // Combine into a Struct Series
    let struct_series =
        StructChunked::from_series("link_info".into(), res_len, all_series.into_iter())?;

    Ok(struct_series.into_series())
}

#[polars_expr(output_type_func=is_social_link_output)]
fn is_social_link(inputs: &[Series], kwargs: AddThresholdKwargs) -> PolarsResult<Series> {
    let threshold = kwargs.threshold;
    // let row_idx = inputs[0].u32()?;
    // let lowest_idx = row_idx.min().unwrap();
    let user_id = inputs[1].i32()?;
    let lon_rad = inputs[2].f32()?;
    let lat_rad = inputs[3].f32()?;
    let event_start = inputs[4].i32()?;
    let event_end = inputs[5].i32()?;
    let offset = inputs[6].u32()?;

    // let event_end_vec = event_end.into_no_null_iter();
    // let event_start_vec = event_start.into_no_null_iter().collect_vec();
    // let offset: ChunkedArray<UInt32Type> = inputs[6].u32()?.into_no_null_iter().map(|v| v - lowest_idx).collect_ca("".into());

    // calculate offsets with partition_point between time_start1 and time_end1
    // let offset = event_end_vec
    //     .map(|end| event_start_vec.partition_point(|&start| start <= end))
    //     .collect_vec();

    let ((user1_vec, user2_vec), time_vec): ((Vec<_>, Vec<_>), Vec<_>) = (0..lon_rad.len())
        // .into_par_iter()
        .flat_map(|i| {
            // SAFETY: bounds checked by iterator
            let user1 = unsafe { user_id.get_unchecked(i).unwrap() };
            let lon1 = unsafe { lon_rad.get_unchecked(i).unwrap() };
            let lat1 = unsafe { lat_rad.get_unchecked(i).unwrap() };
            let time_start1 = unsafe { event_start.get_unchecked(i).unwrap() };
            let time_end1 = unsafe { event_end.get_unchecked(i).unwrap() };
            // let offset_i = offset[i] as usize;
            let offset_i = unsafe { offset.get_unchecked(i).unwrap() as usize };

            assert!(
                offset_i >= i,
                "Offset must be greater than or equal to current row index"
            );
            (i..offset_i)
                // .into_par_iter()
                .filter_map(move |j| {
                    let user2 = unsafe { user_id.get_unchecked(j).unwrap() };
                    if user1 == user2 {
                        return None;
                    }
                    let lon2 = unsafe { lon_rad.get_unchecked(j).unwrap() };
                    let lat2 = unsafe { lat_rad.get_unchecked(j).unwrap() };
                    if haversine(lon1, lat1, lon2, lat2) > threshold {
                        return None;
                    }

                    let time_start2 = unsafe { event_start.get_unchecked(j).unwrap() };
                    let time_end2 = unsafe { event_end.get_unchecked(j).unwrap() };
                    let time_together = time_end1.min(time_end2) - time_start2.max(time_start1);
                    Some(((user1, user2), time_together as u32))
                })
        })
        .unzip();
    // get len now
    let res_len = user1_vec.len();
    // Create individual Series
    let user1_series = &Series::new("user1".into(), user1_vec);
    let user2_series = &Series::new("user2".into(), user2_vec);
    let time_series = &Series::new("time_together".into(), time_vec);
    let all_series = [user1_series, user2_series, time_series];

    // Combine into a Struct Series
    let struct_series =
        StructChunked::from_series("link_info".into(), res_len, all_series.into_iter())?;

    Ok(struct_series.into_series())
}

// 1. Define the output type function
fn is_social_link_output(_input_fields: &[Field]) -> PolarsResult<Field> {
    // Define the nested fields of the output Struct
    let struct_fields: Vec<Field> = vec![
        Field::new("user1".into(), DataType::Int32),
        Field::new("user2".into(), DataType::Int32),
        Field::new("time_together".into(), DataType::UInt32),
    ];

    // Create the Struct DataType
    let struct_data_type = DataType::Struct(struct_fields);

    // Return the final output Field with the desired name and Struct DataType
    Ok(Field::new("link_info".into(), struct_data_type))
}

fn haversine(lon1: f32, lat1: f32, lon2: f32, lat2: f32) -> f32 {
    const R_EARTH: f32 = 6371e3; // in meters
    let dlat = lat2 - lat1;
    let dlon = lon2 - lon1;
    let a = (dlat / 2.0).sin().powi(2) + lat1.cos() * lat2.cos() * (dlon / 2.0).sin().powi(2);
    let c = 2.0 * a.sqrt().asin(); // Changed: atan2 -> asin
    R_EARTH * c
}
