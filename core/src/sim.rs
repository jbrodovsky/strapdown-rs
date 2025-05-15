//! Simulation utilities and CSV data loading for strapdown inertial navigation.
//!
//! This module provides:
//! - A struct (`TestDataRecord`) for deserializing rows from a test data CSV file.
//! - A function (`read_test_data_csv`) to load and parse CSV data into Rust structs.
//! - Unit tests for CSV reading and error handling.

use serde::Deserialize;

/// Struct representing a single row of test data from the CSV file.
///
/// Fields correspond to columns in the CSV, with appropriate renaming for Rust style.
#[derive(Debug, Deserialize)]
pub struct TestDataRecord {
    pub time: String,
    #[serde(rename = "bearingAccuracy")]
    pub bearing_accuracy: f64,
    #[serde(rename = "speedAccuracy")]
    pub speed_accuracy: f64,
    #[serde(rename = "verticalAccuracy")]
    pub vertical_accuracy: f64,
    #[serde(rename = "horizontalAccuracy")]
    pub horizontal_accuracy: f64,
    pub speed: f64,
    pub bearing: f64,
    pub altitude: f64,
    pub longitude: f64,
    pub latitude: f64,
    pub qz: f64,
    pub qy: f64,
    pub qx: f64,
    pub qw: f64,
    pub roll: f64,
    pub pitch: f64,
    pub yaw: f64,
    pub acc_z: f64,
    pub acc_y: f64,
    pub acc_x: f64,
    pub gyro_z: f64,
    pub gyro_y: f64,
    pub gyro_x: f64,
    pub mag_z: f64,
    pub mag_y: f64,
    pub mag_x: f64,
    #[serde(rename = "relativeAltitude")]
    pub relative_altitude: f64,
    pub pressure: f64,
    pub grav_z: f64,
    pub grav_y: f64,
    pub grav_x: f64,
}

/// Reads a CSV file and returns a vector of `TestDataRecord` structs.
///
/// # Arguments
/// * `path` - Path to the CSV file to read.
///
/// # Returns
/// * `Ok(Vec<TestDataRecord>)` if successful.
/// * `Err` if the file cannot be read or parsed.
///
pub fn read_test_data_csv<P: AsRef<std::path::Path>>(path: P) -> Result<Vec<TestDataRecord>, Box<dyn std::error::Error>> {
    let mut rdr = csv::Reader::from_path(path)?;
    let mut records = Vec::new();
    for result in rdr.deserialize() {
        let record: TestDataRecord = result?;
        records.push(record);
    }
    Ok(records)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    /// Test that reading a valid CSV file returns records and parses fields correctly.
    #[test]
    fn test_read_test_data_csv_success() {
        let path = Path::new("./data/test_data.csv");
        let records = read_test_data_csv(path).expect("Failed to read test_data.csv");
        assert!(!records.is_empty(), "CSV should not be empty");
        // Check a few fields of the first record
        let first = &records[0];
        assert!(first.time.len() > 0);
        assert!(first.latitude.abs() > 0.0);
        assert!(first.longitude.abs() > 0.0);
    }

    /// Test that reading a missing file returns an error.
    #[test]
    fn test_read_test_data_csv_invalid_path() {
        let path = Path::new("nonexistent.csv");
        let result = read_test_data_csv(path);
        assert!(result.is_err(), "Should error on missing file");
    }
}