use std::collections::HashMap;

use itertools::{izip, Itertools};
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use pyo3_polars::export::polars_core::utils::rayon::iter::{
    IntoParallelIterator, ParallelBridge, ParallelIterator,
};
use pyo3_polars::export::polars_core::utils::CustomIterTools;
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
    let row_idx: &ChunkedArray<UInt32Type> = inputs[0].u32()?;
    let user_id = inputs[1].i32()?;
    let lon_rad = inputs[2].f32()?;
    let lat_rad = inputs[3].f32()?;
    let event_start = inputs[4].i32()?;
    let event_end = inputs[5].i32()?;
    let offset = inputs[6].u32()?;

    let len = user_id.len();

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
                if haversine(&lon1, &lat1, &lon2, &lat2) > threshold {
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
fn is_social_link_simple(inputs: &[Series], kwargs: AddThresholdKwargs) -> PolarsResult<Series> {
    let threshold = kwargs.threshold;
    let user_id = inputs[0].i32()?.into_no_null_iter().collect_vec();
    let lon_rad = inputs[1].f32()?.into_no_null_iter().collect_vec();
    let lat_rad = inputs[2].f32()?.into_no_null_iter().collect_vec();
    let event_start = inputs[3].i32()?.into_no_null_iter().collect_vec();
    let event_end = inputs[4].i32()?.into_no_null_iter().collect_vec();
    let offset = inputs[5].u32()?.into_no_null_iter().collect_vec();

    // Process each row in parallel
    let results: Vec<_> = izip!(&user_id, &lon_rad, &lat_rad, &event_start, &event_end, &offset)
        .enumerate()
        .par_bridge()
        .flat_map(
            |(i, (user1, lon1, lat1, time_start1, time_end1, offset))| {
                let slice_len = *offset as usize;

                // Slice and filter candidates
                // let user2_ca = user_id.slice(i as i64, slice_len);
                let user2_slice = &user_id[i..i + slice_len];
                let lon2_slice = &lon_rad[i..i + slice_len];
                let lat2_slice = &lat_rad[i..i + slice_len];
                let time_start2_slice = &event_start[i..i + slice_len];
                let time_end2_slice = &event_end[i..i + slice_len];

                izip!(user2_slice, lon2_slice, lat2_slice, time_start2_slice, time_end2_slice)
                    .par_bridge()
                    .filter_map(|(user2, lon2, lat2, start2, end2)| {
                        if user1 == user2 {
                            return None;
                        }
                        if haversine(lon1, lat1, lon2, lat2) >= threshold {
                            return None;
                        }
                        Some((*user1, time_start1, time_end1, *user2, *start2, *end2))
                    })
                    .collect::<Vec<_>>()
            },
        )
        .collect();

    // Unpack results
    let (
        user1_vec,
        event_start_1_vec,
        event_end_1_vec,
        user2_vec,
        event_start_2_vec,
        event_end_2_vec,
    ): (Vec<i32>, Vec<i32>, Vec<i32>, Vec<i32>, Vec<i32>, Vec<i32>) =
        results.into_iter().multiunzip();

    let res_len = user1_vec.len();

    let user1_series = Series::new("user1".into(), user1_vec);
    let event_start_1_series = Series::new("event_start_1".into(), event_start_1_vec);
    let event_end_1_series = Series::new("event_end_1".into(), event_end_1_vec);
    let user2_series = Series::new("user2".into(), user2_vec);
    let event_start_2_series = Series::new("event_start_2".into(), event_start_2_vec);
    let event_end_2_series = Series::new("event_end_2".into(), event_end_2_vec);

    let struct_series = StructChunked::from_series(
        "link_info".into(),
        res_len,
        [
            user1_series,
            event_start_1_series,
            event_end_1_series,
            user2_series,
            event_start_2_series,
            event_end_2_series,
        ]
        .iter(),
    )?;

    Ok(struct_series.into_series())
}

#[polars_expr(output_type_func=is_social_link_output)]
fn is_social_link(inputs: &[Series], kwargs: AddThresholdKwargs) -> PolarsResult<Series> {
    let threshold = kwargs.threshold;
    let user_id = inputs[0].i32()?;
    let lon_rad = inputs[1].f32()?;
    let lat_rad = inputs[2].f32()?;
    let event_start = inputs[3].i32()?;
    let event_end = inputs[4].i32()?;
    let offset = inputs[5].u32()?;
    let day = inputs[6].i32()?;
    // find day boundaries using shift and enumerate

    let mut day_bounds = vec![0];
    // shift and compare
    for i in 1..day.len() {
        let current_day = day.get(i).unwrap();
        let previous_day = day.get(i - 1).unwrap();
        if current_day != previous_day {
            day_bounds.push(i);
        }
    }
    day_bounds.push(day.len());

    let mut daygroups: HashMap<
        i32,
        (
            Int32Chunked,
            Float32Chunked,
            Float32Chunked,
            Int32Chunked,
            Int32Chunked,
            UInt32Chunked,
        ),
    > = HashMap::new();
    // for each group of day bounds, slice the inputs and store in daygroups
    for w in day_bounds.windows(2) {
        let start = w[0];
        let end = w[1];
        let day_value = day.get(start).unwrap();
        let user_id_slice = user_id.slice(start as i64, end - start);
        let lon_rad_slice = lon_rad.slice(start as i64, end - start);
        let lat_rad_slice = lat_rad.slice(start as i64, end - start);
        let event_start_slice = event_start.slice(start as i64, end - start);
        let event_end_slice = event_end.slice(start as i64, end - start);
        let offset_slice = offset.slice(start as i64, end - start);
        daygroups.insert(
            day_value,
            (
                user_id_slice,
                lon_rad_slice,
                lat_rad_slice,
                event_start_slice,
                event_end_slice,
                offset_slice,
            ),
        );
    }
    // // Process each day group independently in parallel
    let results: Vec<_> = daygroups
        .into_par_iter()
        .flat_map(
            |(_day, (user_gr, lons_gr, lats_gr, starts_gr, ends_gr, offsets_gr))| {
                // Process this day group
                process_day_group(
                    &user_gr,
                    &lons_gr,
                    &lats_gr,
                    &starts_gr,
                    &ends_gr,
                    &offsets_gr,
                    0,
                    user_gr.len(), // Use the group length, not total length
                    threshold,
                )
            },
        )
        .collect();

    // Unpack results
    let (
        user1_vec,
        event_start_1_vec,
        event_end_1_vec,
        user2_vec,
        event_start_2_vec,
        event_end_2_vec,
    ): (Vec<i32>, Vec<i32>, Vec<i32>, Vec<i32>, Vec<i32>, Vec<i32>) =
        results.into_iter().multiunzip();

    let res_len = user1_vec.len();

    let user1_series = Series::new("user1".into(), user1_vec);
    let event_start_1_series = Series::new("event_start_1".into(), event_start_1_vec);
    let event_end_1_series = Series::new("event_end_1".into(), event_end_1_vec);
    let user2_series = Series::new("user2".into(), user2_vec);
    let event_start_2_series = Series::new("event_start_2".into(), event_start_2_vec);
    let event_end_2_series = Series::new("event_end_2".into(), event_end_2_vec);

    let struct_series = StructChunked::from_series(
        "link_info".into(),
        res_len,
        [
            user1_series,
            event_start_1_series,
            event_end_1_series,
            user2_series,
            event_start_2_series,
            event_end_2_series,
        ]
        .iter(),
    )?;

    Ok(struct_series.into_series())
}

// Separate function that processes one day group
fn process_day_group(
    user_id: &Int32Chunked,
    lon_rad: &Float32Chunked,
    lat_rad: &Float32Chunked,
    event_start: &Int32Chunked,
    event_end: &Int32Chunked,
    offset: &UInt32Chunked,
    start: usize,
    end: usize,
    threshold: f32,
) -> Vec<(i32, i32, i32, i32, i32, i32)> {
    // Slice views (no ownership transfer)
    let user_id_group = user_id.slice(start as i64, end - start);
    let lon_rad_group = lon_rad.slice(start as i64, end - start);
    let lat_rad_group = lat_rad.slice(start as i64, end - start);
    let event_start_group = event_start.slice(start as i64, end - start);
    let event_end_group = event_end.slice(start as i64, end - start);

    // Process pairs within this day
    (0..user_id_group.len())
        .flat_map(|i| {
            let user1 = unsafe { user_id_group.get_unchecked(i).unwrap() };
            let lon1 = unsafe { lon_rad_group.get_unchecked(i).unwrap() };
            let lat1 = unsafe { lat_rad_group.get_unchecked(i).unwrap() };
            let time_start1 = unsafe { event_start_group.get_unchecked(i).unwrap() };
            let time_end1 = unsafe { event_end_group.get_unchecked(i).unwrap() };
            let slice_len = { unsafe { offset.get_unchecked(i).unwrap() as usize } };

            // Slice and filter candidates
            let user2_ca = user_id_group.slice(i as i64, slice_len);
            let lon2_ca = lon_rad_group.slice(i as i64, slice_len);
            let lat2_ca = lat_rad_group.slice(i as i64, slice_len);

            let mask = haversine_ca_predicate(lon1, lat1, &lon2_ca, &lat2_ca, threshold);

            let time_start2_ca = event_start_group.slice(i as i64, slice_len);
            let time_end2_ca = event_end_group.slice(i as i64, slice_len);

            let user2_filtered = user2_ca.filter(&mask).unwrap();
            let time_start2_filtered = time_start2_ca.filter(&mask).unwrap();
            let time_end2_filtered = time_end2_ca.filter(&mask).unwrap();

            user2_filtered
                .into_iter()
                .zip(time_start2_filtered.into_iter())
                .zip(time_end2_filtered.into_iter())
                .filter_map(|((user2_opt, start2_opt), end2_opt)| {
                    let user2 = user2_opt?;
                    if user1 == user2 {
                        return None;
                    }
                    let start2 = start2_opt?;
                    let end2 = end2_opt?;

                    Some((user1, time_start1, time_end1, user2, start2, end2))
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

// 1. Define the output type function
fn is_social_link_output(_input_fields: &[Field]) -> PolarsResult<Field> {
    // Define the nested fields of the output Struct
    let struct_fields: Vec<Field> = vec![
        Field::new("user1".into(), DataType::Int32),
        Field::new("event_start_1".into(), DataType::Int32),
        Field::new("event_end_1".into(), DataType::Int32),
        Field::new("user2".into(), DataType::Int32),
        Field::new("event_start_2".into(), DataType::Int32),
        Field::new("event_end_2".into(), DataType::Int32),
    ];

    // Create the Struct DataType
    let struct_data_type = DataType::Struct(struct_fields);

    // Return the final output Field with the desired name and Struct DataType
    Ok(Field::new("link_info".into(), struct_data_type))
}

#[inline]
fn haversine(lon1: &f32, lat1: &f32, lon2: &f32, lat2: &f32) -> f32 {
    const R_EARTH: f32 = 6371e3; // in meters
    let dlat = lat2 - lat1;
    let dlon = lon2 - lon1;
    let a = (dlat / 2.0).sin().powi(2) + lat1.cos() * lat2.cos() * (dlon / 2.0).sin().powi(2);
    let c = 2.0 * a.sqrt().asin(); // Changed: atan2 -> asin
    R_EARTH * c
}

fn haversine_ca_predicate(
    lon1: f32,
    lat1: f32,
    lon2: &ChunkedArray<Float32Type>,
    lat2: &ChunkedArray<Float32Type>,
    threshold: f32,
) -> ChunkedArray<BooleanType> {
    const R_EARTH: f32 = 6371e3; // in meters
    let dlat = lat2.apply_values(|l2| l2 - lat1);
    let dlon = lon2.apply_values(|l2| l2 - lon1);
    let lat1_cos = lat1.cos();
    let a = dlat.apply_values(|dl| (dl / 2.0).sin().powi(2))
        + lat2.apply_values(|l2| lat1_cos * l2.cos())
            * dlon.apply_values(|dl| (dl / 2.0).sin().powi(2));
    let c = a.apply_values(|a_val| 2.0 * a_val.sqrt().asin()); // Changed: atan2 -> asin
    let distances = c.apply_values(|c_val| R_EARTH * c_val);
    distances.lt(threshold)
}
