//! Performance plotting module for strapdown INS simulation results.
//!
//! This module provides functionality to generate performance plots comparing
//! navigation output with GPS ground truth measurements.

use chrono::{DateTime, Utc};
use plotters::prelude::*;
use std::error::Error;
use std::path::Path;
use strapdown::sim::{NavigationResult, TestDataRecord};

/// Calculate the haversine distance between two points on Earth's surface.
///
/// # Arguments
/// * `lat1` - Latitude of first point in degrees
/// * `lon1` - Longitude of first point in degrees
/// * `lat2` - Latitude of second point in degrees
/// * `lon2` - Longitude of second point in degrees
///
/// # Returns
/// Distance in meters
pub(crate) fn haversine_distance(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    const EARTH_RADIUS_M: f64 = 6_371_000.0;

    let lat1_rad = lat1.to_radians();
    let lat2_rad = lat2.to_radians();
    let delta_lat = (lat2 - lat1).to_radians();
    let delta_lon = (lon2 - lon1).to_radians();

    let a = (delta_lat / 2.0).sin().powi(2)
        + lat1_rad.cos() * lat2_rad.cos() * (delta_lon / 2.0).sin().powi(2);
    let c = 2.0 * a.sqrt().atan2((1.0 - a).sqrt());

    EARTH_RADIUS_M * c
}

/// A plotted series of `(elapsed seconds, value)` points.
type Series = Vec<(f64, f64)>;

/// Seconds from `origin` to `time`, at millisecond resolution.
fn elapsed_seconds(origin: DateTime<Utc>, time: DateTime<Utc>) -> f64 {
    (time - origin).num_milliseconds() as f64 / 1000.0
}

/// Horizontal and vertical error, as `(elapsed seconds, metres)`, at every epoch where the
/// reference has a GNSS fix.
///
/// Each solution is paired with the record carrying the *same* timestamp. Every run writes one
/// result per record, the initial state included, so the positional pairing this replaced --
/// solution `i` against record `i + 1` -- compared each solution with the *next* record's fix,
/// adding a speed x sample-interval error to every point (1 s of travel at 1 Hz).
///
/// Epochs without a fix are dropped rather than carried as NaN. Nine records in ten have none
/// at the 10 Hz preprocessing rate, and plotters maps a NaN coordinate to the bottom of the
/// axis, so a NaN-carrying series drew a comb from zero up to each fix instead of a trace.
fn error_series(
    nav_results: &[NavigationResult],
    gps_records: &[TestDataRecord],
) -> (Series, Series) {
    let mut horizontal = Vec::new();
    let mut vertical = Vec::new();
    let Some(origin) = nav_results.first().map(|nav| nav.timestamp) else {
        return (horizontal, vertical);
    };
    // Both sequences are in time order, so a single forward pass pairs them.
    let mut records = gps_records.iter().peekable();
    for nav in nav_results {
        while records.next_if(|gps| gps.time < nav.timestamp).is_some() {}
        let Some(gps) = records.peek().filter(|gps| gps.time == nav.timestamp) else {
            continue;
        };
        let elapsed = elapsed_seconds(origin, nav.timestamp);
        let h_error = haversine_distance(nav.latitude, nav.longitude, gps.latitude, gps.longitude);
        if h_error.is_finite() {
            horizontal.push((elapsed, h_error));
        }
        let v_error = (nav.altitude - gps.altitude).abs();
        if v_error.is_finite() {
            vertical.push((elapsed, v_error));
        }
    }
    (horizontal, vertical)
}

/// Generate a performance plot comparing navigation results with GPS measurements.
///
/// The plot includes:
/// - 2D haversine horizontal error (meters)
/// - Vertical altitude error (meters)
/// - GPS horizontal accuracy field
/// - GPS vertical accuracy field
///
/// # Arguments
/// * `nav_results` - Vector of navigation results from simulation
/// * `gps_records` - Vector of GPS measurements from input data
/// * `output_path` - Path where the plot image will be saved
///
/// # Returns
/// Result indicating success or error
pub(crate) fn plot_performance(
    nav_results: &[NavigationResult],
    gps_records: &[TestDataRecord],
    output_path: &Path,
) -> Result<(), Box<dyn Error>> {
    // Create the output image
    let root = BitMapBackend::new(output_path, (1200, 400)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut gps_h_accuracy = Vec::new();
    let mut gps_v_accuracy = Vec::new();

    // Collect GPS accuracy data
    for gps in gps_records {
        let elapsed = elapsed_seconds(gps_records[0].time, gps.time);

        if !gps.horizontal_accuracy.is_nan() && gps.horizontal_accuracy > 0.0 {
            gps_h_accuracy.push((elapsed, gps.horizontal_accuracy));
        }
        if !gps.vertical_accuracy.is_nan() && gps.vertical_accuracy > 0.0 {
            gps_v_accuracy.push((elapsed, gps.vertical_accuracy));
        }
    }

    let (horizontal_errors, vertical_errors) = error_series(nav_results, gps_records);
    if horizontal_errors.is_empty() && vertical_errors.is_empty() {
        return Err("No GNSS fix coincides with a navigation solution".into());
    }

    // Find the maximum time and error for axis scaling
    let max_time = match (nav_results.first(), nav_results.last()) {
        (Some(first), Some(last)) => elapsed_seconds(first.timestamp, last.timestamp),
        _ => 0.0,
    };
    let max_error = horizontal_errors
        .iter()
        .chain(vertical_errors.iter())
        .map(|(_, v)| v)
        .chain(gps_h_accuracy.iter().map(|(_, v)| v))
        .chain(gps_v_accuracy.iter().map(|(_, v)| v))
        .copied()
        .fold(0.0f64, f64::max);

    // Cap the y-axis at 50m if max_error is reasonable, otherwise use max_error + 10%
    let y_max = if max_error < 50.0 {
        50.0
    } else {
        max_error * 1.1
    };

    // Create chart
    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Strapdown INS Simulation Performance with GPS Comparison",
            ("sans-serif", 16).into_font(),
        )
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(0.0..max_time, 0.0..y_max)?;

    chart
        .configure_mesh()
        .x_desc("Time (s)")
        .y_desc("2D Haversine Error (m)")
        .draw()?;

    // Plot 2D haversine error
    chart
        .draw_series(LineSeries::new(horizontal_errors, &RED))?
        .label("2D Haversine Error")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED));

    // Plot altitude error
    chart
        .draw_series(LineSeries::new(vertical_errors, &BLUE))?
        .label("Altitude Error")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE));

    // Plot GPS horizontal accuracy (dashed line)
    if !gps_h_accuracy.is_empty() {
        chart
            .draw_series(
                gps_h_accuracy
                    .iter()
                    .copied()
                    .map(|(x, y)| Circle::new((x, y), 2, GREEN.filled())),
            )?
            .label("GPS Horizontal Accuracy")
            .legend(|(x, y)| Circle::new((x + 10, y), 3, GREEN.filled()));
    }

    // Plot GPS vertical accuracy (dashed line)
    if !gps_v_accuracy.is_empty() {
        chart
            .draw_series(
                gps_v_accuracy
                    .iter()
                    .copied()
                    .map(|(x, y)| Circle::new((x, y), 2, MAGENTA.filled())),
            )?
            .label("GPS Vertical Accuracy")
            .legend(|(x, y)| Circle::new((x + 10, y), 3, MAGENTA.filled()));
    }

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;

    root.present()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    use chrono::TimeDelta;
    use strapdown::StrapdownState;

    /// A 10 Hz track due north at ~20 m/s, with a GNSS fix on every `fix_every`-th record and
    /// NaN position on the rest, as the preprocessed Sensor Logger inputs have; and a
    /// navigation solution that sits exactly on the true track at every record.
    fn track(n: i64, fix_every: i64) -> (Vec<NavigationResult>, Vec<TestDataRecord>) {
        let start = DateTime::<Utc>::from_timestamp(1_700_000_000, 0).unwrap();
        let mut nav_results = Vec::new();
        let mut records = Vec::new();
        for i in 0..n {
            let time = start + TimeDelta::milliseconds(100 * i);
            let latitude = 40.0 + 2e-5 * i as f64;
            let state = StrapdownState {
                latitude: latitude.to_radians(),
                longitude: (-75.0_f64).to_radians(),
                altitude: 100.0,
                ..StrapdownState::default()
            };
            nav_results.push(NavigationResult::from((&time, &state)));
            let has_fix = i % fix_every == 0;
            let fix = |value: f64| if has_fix { value } else { f64::NAN };
            records.push(TestDataRecord {
                time,
                latitude: fix(latitude),
                longitude: fix(-75.0),
                altitude: fix(100.0),
                ..TestDataRecord::default()
            });
        }
        (nav_results, records)
    }

    #[test]
    fn errors_pair_each_solution_with_the_record_at_its_own_timestamp() {
        let (nav_results, records) = track(50, 1);
        let (horizontal, vertical) = error_series(&nav_results, &records);
        assert_eq!(horizontal.len(), 50);
        // A solution on the true track scores zero. Pairing it with the next record instead
        // scores the 2.2 m the vehicle covers in one 0.1 s sample.
        for (_, error) in horizontal.iter().chain(&vertical) {
            assert!(*error < 1e-6, "solution on the track scored {error} m");
        }
    }

    #[test]
    fn epochs_without_a_fix_are_dropped_not_carried_as_nan() {
        let (nav_results, records) = track(100, 10);
        let (horizontal, vertical) = error_series(&nav_results, &records);
        assert_eq!(horizontal.len(), 10);
        assert_eq!(vertical.len(), 10);
        assert!(
            horizontal
                .iter()
                .chain(&vertical)
                .all(|(t, e)| t.is_finite() && e.is_finite())
        );
        let times: Vec<f64> = horizontal.iter().map(|(t, _)| *t).collect();
        assert!(
            (times[1] - 1.0).abs() < 1e-9,
            "second fix at {} s, not 1 s",
            times[1]
        );
    }

    #[test]
    fn records_without_a_matching_solution_are_skipped() {
        let (nav_results, records) = track(20, 1);
        // Every other solution: the unmatched records in between are passed over, and the
        // matched ones still pair by timestamp.
        let sparse: Vec<NavigationResult> = nav_results.into_iter().step_by(2).collect();
        let (horizontal, _) = error_series(&sparse, &records);
        assert_eq!(horizontal.len(), 10);
        assert!(horizontal.iter().all(|(_, e)| *e < 1e-6));
    }

    #[test]
    fn test_haversine_distance_zero() {
        let dist = haversine_distance(0.0, 0.0, 0.0, 0.0);
        assert!(dist < 0.01, "Distance should be near zero");
    }

    #[test]
    fn test_haversine_distance_equator() {
        // 1 degree at equator is approximately 111 km
        let dist = haversine_distance(0.0, 0.0, 0.0, 1.0);
        assert!(
            (dist - 111_000.0).abs() < 1000.0,
            "Distance should be approximately 111 km"
        );
    }

    #[test]
    fn test_haversine_distance_pole() {
        // Distance from pole to pole
        let dist = haversine_distance(90.0, 0.0, -90.0, 0.0);
        assert!(
            (dist - 20_015_000.0).abs() < 10_000.0,
            "Distance should be approximately half Earth's circumference"
        );
    }
}
