//! Performance plotting module for strapdown INS simulation results.
//!
//! This module provides functionality to generate performance plots comparing
//! navigation output with GPS ground truth measurements.

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
pub fn haversine_distance(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
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
pub fn plot_performance(
    nav_results: &[NavigationResult],
    gps_records: &[TestDataRecord],
    output_path: &Path,
) -> Result<(), Box<dyn Error>> {
    // Create the output image
    let root = BitMapBackend::new(output_path, (1200, 400)).into_drawing_area();
    root.fill(&WHITE)?;

    // Calculate errors
    let mut horizontal_errors = Vec::new();
    let mut vertical_errors = Vec::new();
    let mut timestamps = Vec::new();
    let mut gps_h_accuracy = Vec::new();
    let mut gps_v_accuracy = Vec::new();

    // Collect GPS accuracy data
    for gps in gps_records {
        let elapsed = (gps.time - gps_records[0].time).num_milliseconds() as f64 / 1000.0;

        if !gps.horizontal_accuracy.is_nan() && gps.horizontal_accuracy > 0.0 {
            gps_h_accuracy.push((elapsed, gps.horizontal_accuracy));
        }
        if !gps.vertical_accuracy.is_nan() && gps.vertical_accuracy > 0.0 {
            gps_v_accuracy.push((elapsed, gps.vertical_accuracy));
        }
    }

    // Calculate errors for each navigation result
    // Skip first GPS record as it's used for initialization
    for (i, nav) in nav_results.iter().enumerate() {
        if i + 1 >= gps_records.len() {
            break;
        }

        let gps = &gps_records[i + 1];
        let elapsed = (nav.timestamp - nav_results[0].timestamp).num_milliseconds() as f64 / 1000.0;

        // Calculate 2D haversine distance error
        let h_error = haversine_distance(nav.latitude, nav.longitude, gps.latitude, gps.longitude);

        // Calculate vertical error
        let v_error = (nav.altitude - gps.altitude).abs();

        timestamps.push(elapsed);
        horizontal_errors.push(h_error);
        vertical_errors.push(v_error);
    }

    if timestamps.is_empty() {
        return Err("No data points to plot".into());
    }

    // Find the maximum time and error for axis scaling
    let max_time = timestamps.iter().copied().fold(0.0f64, f64::max);
    let max_error = horizontal_errors
        .iter()
        .chain(vertical_errors.iter())
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
        .draw_series(LineSeries::new(
            timestamps
                .iter()
                .copied()
                .zip(horizontal_errors.iter().copied()),
            &RED,
        ))?
        .label("2D Haversine Error")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED));

    // Plot altitude error
    chart
        .draw_series(LineSeries::new(
            timestamps
                .iter()
                .copied()
                .zip(vertical_errors.iter().copied()),
            &BLUE,
        ))?
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
