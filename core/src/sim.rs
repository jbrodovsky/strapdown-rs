//! Simulation utilities and CSV data loading for strapdown inertial navigation.
//!
//! This module provides:
//! - A struct (`TestDataRecord`) for reading and writing test data to/from CSV files
//! - Functions for running simulations like dead reckoning
//! - `NavigationResult` structure for storing and analyzing navigation solutions
//! - CSV import/export functionality for both test data and navigation results
//! - Unit tests for validating functionality

use serde::{Deserialize, Serialize};
use nalgebra::Vector3;
use crate::{IMUData, StrapdownState};
use chrono::DateTime;
use std::fs::File;
use std::io::{self, Write};
use std::path::Path;

/// Struct representing a single row of test data from the CSV file.
///
/// Fields correspond to columns in the CSV, with appropriate renaming for Rust style.
/// This struct is setup to capture the data recorded from the [Sensor Logger](https://www.tszheichoi.com/sensorlogger) app.
/// Primarily, this represents IMU data as (relative to the device) and GPS data.
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct TestDataRecord {
    /// Date-time string: YYYY-MM-DD hh:mm:ss+UTCTZ
    pub time: String,
    /// accuracy of the bearing (magnetic heading) in degrees
    #[serde(rename = "bearingAccuracy")]
    pub bearing_accuracy: f64,
    /// accuracy of the speed in m/s
    #[serde(rename = "speedAccuracy")]
    pub speed_accuracy: f64,
    /// accuracy of the altitude in meters
    #[serde(rename = "verticalAccuracy")]
    pub vertical_accuracy: f64,
    /// accuracy of the horizontal position in meters
    #[serde(rename = "horizontalAccuracy")]
    pub horizontal_accuracy: f64,
    /// Speed in m/s
    pub speed: f64,
    /// Bearing in degrees
    pub bearing: f64,
    /// Altitude in meters
    pub altitude: f64,
    /// Longitude in degrees
    pub longitude: f64,
    /// Latitude in degrees
    pub latitude: f64,
    /// Quaternion component representing the rotation around the z-axis
    pub qz: f64,
    /// Quaternion component representing the rotation around the y-axis
    pub qy: f64,
    /// Quaternion component representing the rotation around the x-axis
    pub qx: f64,
    /// Quaternion component representing the rotation around the w-axis
    pub qw: f64,
    /// Roll angle in radians
    pub roll: f64,
    /// Pitch angle in radians
    pub pitch: f64,
    /// Yaw angle in radians
    pub yaw: f64,
    /// Z-acceleration in m/s^2
    pub acc_z: f64,
    /// Y-acceleration in m/s^2
    pub acc_y: f64,
    /// X-acceleration in m/s^2
    pub acc_x: f64,
    /// Rotation rate around the z-axis in radians/s
    pub gyro_z: f64,
    /// Rotation rate around the y-axis in radians/s
    pub gyro_y: f64,
    /// Rotation rate around the x-axis in radians/s
    pub gyro_x: f64,
    /// Magnetic field strength in the z-direction in microteslas
    pub mag_z: f64,
    /// Magnetic field strength in the y-direction in microteslas
    pub mag_y: f64,
    /// Magnetic field strength in the x-direction in microteslas
    pub mag_x: f64,
    /// Change in altitude in meters
    #[serde(rename = "relativeAltitude")]
    pub relative_altitude: f64,
    /// pressure in millibars
    pub pressure: f64,
    /// Acceleration due to gravity in the z-direction in m/s^2
    pub grav_z: f64,
    /// Acceleration due to gravity in the y-direction in m/s^2
    pub grav_y: f64,
    /// Acceleration due to gravity in the x-direction in m/s^2
    pub grav_x: f64,
}

impl TestDataRecord {
    /// Reads a CSV file and returns a vector of `TestDataRecord` structs.
    ///
    /// # Arguments
    /// * `path` - Path to the CSV file to read.
    ///
    /// # Returns
    /// * `Ok(Vec<TestDataRecord>)` if successful.
    /// * `Err` if the file cannot be read or parsed.
    ///
    /// # Example
    ///
    /// ```
    /// use strapdown::sim::TestDataRecord;
    /// use std::path::Path;
    ///
    /// let records = TestDataRecord::from_csv("./data/test_data.csv")
    ///     .expect("Failed to read test data");
    /// println!("Loaded {} records", records.len());
    /// ```
    pub fn from_csv<P: AsRef<std::path::Path>>(path: P) -> Result<Vec<Self>, Box<dyn std::error::Error>> {
        let mut rdr = csv::Reader::from_path(path)?;
        let mut records = Vec::new();
        for result in rdr.deserialize() {
            let record: Self = result?;
            records.push(record);
        }
        Ok(records)
    }

    /// Writes a vector of TestDataRecord structs to a CSV file.
    ///
    /// # Arguments
    /// * `records` - Vector of TestDataRecord structs to write
    /// * `path` - Path where the CSV file will be saved
    ///
    /// # Returns
    /// * `io::Result<()>` - Ok if successful, Err otherwise
    ///
    /// # Example
    ///
    /// ```
    /// use strapdown::sim::TestDataRecord;
    /// use std::path::Path;
    ///
    /// let records = TestDataRecord::from_csv("./data/test_data.csv").unwrap();
    /// // Create a subset or modify records
    /// let subset = records.into_iter().take(10).collect::<Vec<_>>();
    /// // Save the subset to a new file
    /// TestDataRecord::to_csv(&subset, "subset_data.csv").expect("Failed to write CSV");
    /// ```
    pub fn to_csv<P: AsRef<Path>>(records: &[Self], path: P) -> io::Result<()> {
        let mut writer = csv::Writer::from_path(path)?;
        
        for record in records {
            writer.serialize(record)?;
        }
        
        writer.flush()?;
        Ok(())
    }
}

/// Generic result struct for navigation simulations.
/// 
/// This structure contains a sequence of navigation states (StrapdownState instances) 
/// representing the navigation solution over time, along with timestamps for each state.
/// It can be used across different types of navigation simulations such as dead reckoning,
/// Kalman filtering, or any other navigation algorithm.
#[derive(Debug)]
pub struct NavigationResult {
    /// Vector of StrapdownState instances at each time step
    pub states: Vec<StrapdownState>,
    /// Vector of timestamps corresponding to each state
    pub timestamps: Vec<String>,
    /// Time difference between consecutive records in seconds
    pub dt: Vec<f64>,
    /// Optional name or identifier for this navigation solution
    pub name: String,
}

impl NavigationResult {
    /// Creates a new NavigationResult with empty vectors.
    pub fn new(name: &str) -> Self {
        NavigationResult {
            states: Vec::new(),
            timestamps: Vec::new(),
            dt: Vec::new(),
            name: name.to_string(),
        }
    }

    /// Writes the navigation result to a CSV file.
    ///
    /// This function exports the position, velocity, attitude (as Euler angles), and timestamps
    /// to a CSV file for further analysis or visualization.
    ///
    /// # Arguments
    /// * `path` - Path where the CSV file will be saved
    ///
    /// # Returns
    /// * `io::Result<()>` - Ok if successful, Err otherwise
    ///
    /// # Example
    ///
    /// ```
    /// use strapdown::sim::{NavigationResult, TestDataRecord};
    /// use strapdown::StrapdownState;
    /// 
    /// // Load test data and run simulation
    /// let records = TestDataRecord::from_csv("./data/test_data.csv").unwrap();
    /// let result = strapdown::sim::dead_reckoning(&records);
    /// 
    /// // Save results to CSV
    /// result.to_csv("navigation_result.csv").expect("Failed to write results");
    /// ```
    pub fn to_csv<P: AsRef<Path>>(&self, path: P) -> io::Result<()> {
        let mut file = File::create(path)?;
        
        // Write CSV header
        writeln!(file, "time,lat_deg,lon_deg,alt_m,vel_n_mps,vel_e_mps,vel_d_mps,roll_deg,pitch_deg,yaw_deg,dt_s")?;
        
        // Write each state
        for i in 0..self.states.len() {
            let state = &self.states[i];
            let timestamp = &self.timestamps[i];
            
            // Extract Euler angles (roll, pitch, yaw) from the attitude rotation matrix
            let (roll, pitch, yaw) = state.attitude.euler_angles();
            
            // Write the data row
            let dt_value = if i > 0 { self.dt[i-1] } else { 0.0 };
            
            writeln!(
                file,
                "{},{:.8},{:.8},{:.4},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}",
                timestamp,
                state.position[0],  // latitude (deg)
                state.position[1],  // longitude (deg)
                state.position[2],  // altitude (m)
                state.velocity[0],  // north velocity (m/s)
                state.velocity[1],  // east velocity (m/s)
                state.velocity[2],  // down velocity (m/s)
                roll.to_degrees(),  // roll (deg)
                pitch.to_degrees(), // pitch (deg)
                yaw.to_degrees(),   // yaw (deg)
                dt_value            // time step (s)
            )?;
        }
        
        Ok(())
    }
    
    /// Creates a NavigationResult from a CSV file.
    ///
    /// This function imports position, velocity, attitude, and timestamp data from a CSV file
    /// that was previously created using the `to_csv` method.
    ///
    /// # Arguments
    /// * `path` - Path to the CSV file to read
    /// * `name` - Name for the navigation result
    ///
    /// # Returns
    /// * `Result<NavigationResult, Box<dyn std::error::Error>>` - The loaded navigation result or an error
    ///
    /// # Example
    ///
    /// ```
    /// use strapdown::sim::NavigationResult;
    /// 
    /// // Load results from CSV
    /// let result = NavigationResult::from_csv("navigation_result.csv", "Imported Solution")
    ///     .expect("Failed to read results");
    /// println!("Loaded {} states from {}", result.states.len(), result.name);
    /// ```
    pub fn from_csv<P: AsRef<Path>>(path: P, name: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let mut rdr = csv::ReaderBuilder::new()
            .has_headers(true)
            .from_reader(file);

        let mut result = NavigationResult::new(name);
        
        for record in rdr.records() {
            let record = record?;
            
            if record.len() < 11 {
                return Err("CSV record doesn't have enough fields".into());
            }
            
            // Parse timestamp
            let timestamp = record[0].to_string();
            
            // Parse position (lat, lon, alt)
            let lat = record[1].parse::<f64>()?;
            let lon = record[2].parse::<f64>()?;
            let alt = record[3].parse::<f64>()?;
            
            // Parse velocity (n, e, d)
            let vel_n = record[4].parse::<f64>()?;
            let vel_e = record[5].parse::<f64>()?;
            let vel_d = record[6].parse::<f64>()?;
            
            // Parse attitude (roll, pitch, yaw in degrees)
            let roll_deg = record[7].parse::<f64>()?;
            let pitch_deg = record[8].parse::<f64>()?;
            let yaw_deg = record[9].parse::<f64>()?;
            
            // Parse dt
            let dt = record[10].parse::<f64>()?;
            
            // Create StrapdownState
            let mut state = StrapdownState::new();
            state.position = Vector3::new(lat, lon, alt);
            state.velocity = Vector3::new(vel_n, vel_e, vel_d);
            state.attitude = nalgebra::Rotation3::from_euler_angles(
                roll_deg.to_radians(),
                pitch_deg.to_radians(),
                yaw_deg.to_radians()
            );
            
            // Add to result
            result.states.push(state);
            result.timestamps.push(timestamp);
            if !result.dt.is_empty() || dt > 0.0 {
                result.dt.push(dt);
            }
        }
        
        Ok(result)
    }
}

/// Run dead reckoning simulation using test data.
///
/// This function processes a sequence of sensor records through a StrapdownState, using
/// the "forward" method to propagate the state based on IMU measurements. It initializes
/// the StrapdownState with position, velocity, and attitude from the first record, and
/// then applies the IMU measurements from subsequent records.
///
/// # Arguments
/// * `records` - Vector of test data records containing IMU measurements and other sensor data
///
/// # Returns
/// * `NavigationResult` containing the sequence of StrapdownState instances over time,
///   along with timestamps and time differences.
pub fn dead_reckoning(records: &Vec<TestDataRecord>) -> NavigationResult {
    if records.is_empty() {
        return NavigationResult::new("Dead Reckoning");
    }

    // Initialize the result vectors
    let mut result = NavigationResult::new("Dead Reckoning");
    result.states = Vec::with_capacity(records.len());
    result.timestamps = Vec::with_capacity(records.len());
    result.dt = Vec::with_capacity(records.len() - 1);
    
    // Initialize the StrapdownState with the first record
    let first_record = &records[0];
    let mut state = StrapdownState::new();
    
    // Set initial position (latitude, longitude, altitude)
    state.position = Vector3::new(
        first_record.latitude,
        first_record.longitude,
        first_record.altitude
    );
    
    // Set initial velocity (assuming zero initial velocity)
    state.velocity = Vector3::new(0.0, 0.0, 0.0);
    
    // Set initial attitude (roll, pitch, yaw from first record)
    state.attitude = nalgebra::Rotation3::from_euler_angles(
        first_record.roll,
        first_record.pitch,
        first_record.yaw
    );
    
    // Store the initial state
    result.states.push(state.clone());
    result.timestamps.push(first_record.time.clone());
    
    // For the time difference calculation, use a fixed value if timestamps can't be parsed
    let mut prev_time_str = &first_record.time;
    
    // Process each subsequent record
    for record in records.iter().skip(1) {
        // Try to calculate time difference from timestamps, default to 1 second if parsing fails
        let dt = match (
            DateTime::parse_from_str(prev_time_str, "%Y-%m-%d %H:%M:%S%z"),
            DateTime::parse_from_str(&record.time, "%Y-%m-%d %H:%M:%S%z")
        ) {
            (Ok(prev), Ok(current)) => {
                (current - prev).num_milliseconds() as f64 / 1000.0
            },
            _ => 1.0 // Default to 1 second if parsing fails
        };
        
        result.dt.push(dt);
        
        // Create IMU data from the record
        // Note: We use relative acceleration by subtracting gravity
        let imu_data = IMUData::new_from_vec(
            vec![
                record.acc_x - record.grav_x,
                record.acc_y - record.grav_y,
                record.acc_z - record.grav_z
            ],
            vec![record.gyro_x, record.gyro_y, record.gyro_z]
        );
        
        // Propagate the state forward
        state.forward(&imu_data, dt);
        
        // Store the updated state
        result.states.push(state.clone());
        result.timestamps.push(record.time.clone());
        
        // Update previous time string for next iteration
        prev_time_str = &record.time;
    }
    
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    /// Test that reading a valid CSV file returns records and parses fields correctly.
    #[test]
    fn test_test_data_record_from_csv() {
        let path = Path::new("./data/test_data.csv");
        let records = TestDataRecord::from_csv(path).expect("Failed to read test_data.csv");
        assert!(!records.is_empty(), "CSV should not be empty");
        // Check a few fields of the first record
        let first = &records[0];
        assert!(first.time.len() > 0);
        assert!(first.latitude.abs() > 0.0);
        assert!(first.longitude.abs() > 0.0);
    }

    /// Test that reading a missing file returns an error.
    #[test]
    fn test_test_data_record_from_csv_invalid_path() {
        let path = Path::new("nonexistent.csv");
        let result = TestDataRecord::from_csv(path);
        assert!(result.is_err(), "Should error on missing file");
    }
    
    /// Test writing TestDataRecord to CSV and reading it back
    #[test]
    fn test_test_data_record_to_csv() {
        // Read original records
        let path = Path::new("./data/test_data.csv");
        let original_records = TestDataRecord::from_csv(path).expect("Failed to read test_data.csv");
        
        // Take just a few records to keep the test fast
        let subset = if original_records.len() > 3 {
            original_records[0..3].to_vec()
        } else {
            original_records.clone()
        };
        
        // Write to temporary file
        let temp_file = std::env::temp_dir().join("test_data_subset.csv");
        let temp_path = temp_file.to_string_lossy().to_string();
        
        TestDataRecord::to_csv(&subset, &temp_path).expect("Failed to write CSV");
        
        // Read back from temporary file
        let read_records = TestDataRecord::from_csv(&temp_path).expect("Failed to read temporary CSV");
        
        // Verify count
        assert_eq!(subset.len(), read_records.len(), "Should read same number of records as written");
        
        // Clean up
        let _ = std::fs::remove_file(&temp_path);
    }

    /// Test the dead_reckoning function with a small set of sample data
    #[test]
    fn test_dead_reckoning() {
        let path = Path::new("./data/test_data.csv");
        let records = TestDataRecord::from_csv(path).expect("Failed to read test_data.csv");
        
        // Only use a few records for the test to keep it fast
        let sample_records = if records.len() > 5 {
            records[0..5].to_vec()
        } else {
            records
        };
        
        let result = dead_reckoning(&sample_records);
        
        // Check that we have the expected number of states
        assert_eq!(result.states.len(), sample_records.len());
        assert_eq!(result.timestamps.len(), sample_records.len());
        assert_eq!(result.dt.len(), sample_records.len() - 1);
        
        // Check that the first state has the expected position
        let first_state = &result.states[0];
        let first_record = &sample_records[0];
        assert!((first_state.position[0] - first_record.latitude).abs() < 1e-10);
        assert!((first_state.position[1] - first_record.longitude).abs() < 1e-10);
        assert!((first_state.position[2] - first_record.altitude).abs() < 1e-10);
    }
    
    /// Test the to_csv function of NavigationResult
    #[test]
    fn test_navigation_result_to_csv() {
        let mut result = NavigationResult::new("Test Navigation");
        
        // Create a simple state 
        let mut state = StrapdownState::new();
        state.position = Vector3::new(40.0, -75.0, 100.0);
        state.velocity = Vector3::new(1.0, 2.0, 0.5);
        
        // Add it to the result
        result.states.push(state);
        result.timestamps.push("2023-08-04 21:47:58+00:00".to_string());
        result.dt.push(1.0);
        
        // Create a temporary file for the test
        let temp_file = std::env::temp_dir().join("nav_test.csv");
        let temp_path = temp_file.to_string_lossy().to_string();
        
        // Write to CSV
        result.to_csv(&temp_path).expect("Failed to write CSV");
        
        // Check that the file exists and is not empty
        let metadata = std::fs::metadata(&temp_path).expect("Failed to read metadata");
        assert!(metadata.len() > 0, "CSV file should not be empty");
        
        // Clean up
        let _ = std::fs::remove_file(&temp_path);
    }
    
    /// Test the from_csv and to_csv functions of NavigationResult together
    #[test]
    fn test_navigation_result_from_csv() {
        // Create a navigation result with a few states
        let mut original_result = NavigationResult::new("Original Navigation");
        
        // Add a few states with different values
        for i in 0..3 {
            let mut state = StrapdownState::new();
            state.position = Vector3::new(40.0 + i as f64 * 0.1, -75.0 - i as f64 * 0.1, 100.0 + i as f64);
            state.velocity = Vector3::new(1.0 + i as f64, 2.0 - i as f64, 0.5);
            
            // Use a different attitude for each state
            state.attitude = nalgebra::Rotation3::from_euler_angles(
                (i as f64 * 5.0).to_radians(), 
                (i as f64 * 2.0).to_radians(), 
                (i as f64 * 10.0).to_radians()
            );
            
            original_result.states.push(state);
            original_result.timestamps.push(format!("2023-08-04 21:{}:00+00:00", 47 + i));
            original_result.dt.push(1.0);
        }
        
        // Write to a temporary file
        let temp_file = std::env::temp_dir().join("nav_result_roundtrip.csv");
        let temp_path = temp_file.to_string_lossy().to_string();
        
        original_result.to_csv(&temp_path).expect("Failed to write CSV");
        
        // Read back
        let read_result = NavigationResult::from_csv(&temp_path, "Read Navigation")
            .expect("Failed to read CSV");
        
        // Verify contents
        assert_eq!(read_result.states.len(), original_result.states.len());
        assert_eq!(read_result.timestamps.len(), original_result.timestamps.len());
        assert_eq!(read_result.dt.len(), original_result.dt.len()-1);
        assert_eq!(read_result.name, "Read Navigation");
        
        // Compare values from first state
        let original_state = &original_result.states[0];
        let read_state = &read_result.states[0];
        
        assert!((original_state.position[0] - read_state.position[0]).abs() < 1e-7);
        assert!((original_state.position[1] - read_state.position[1]).abs() < 1e-7);
        assert!((original_state.position[2] - read_state.position[2]).abs() < 1e-3);
        
        assert!((original_state.velocity[0] - read_state.velocity[0]).abs() < 1e-5);
        assert!((original_state.velocity[1] - read_state.velocity[1]).abs() < 1e-5);
        assert!((original_state.velocity[2] - read_state.velocity[2]).abs() < 1e-5);
        
        // Clean up
        let _ = std::fs::remove_file(&temp_path);
    }
}