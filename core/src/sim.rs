//! Simulation utilities and CSV data loading for strapdown inertial navigation.
//!
//! This module provides:
//! - A struct (`TestDataRecord`) for reading and writing test data to/from CSV files
//! - Functions for running simulations like dead reckoning
//! - `NavigationResult` structure for storing and analyzing navigation solutions
//! - CSV import/export functionality for both test data and navigation results
//! - Unit tests for validating functionality
use std::fmt::Display;
use std::io::{self};
use std::path::Path;

use chrono::DateTime;
use nalgebra::{DMatrix, DVector, Vector3};
use serde::{Deserialize, Serialize, Serializer, Deserializer};

use crate::{IMUData, StrapdownState};
use crate::filter::{UKF, position_measurement_model};
use crate::earth::{METERS_TO_DEGREES};
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
    /// let records = TestDataRecord::from_csv("data/test_data.csv")
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
    /// // doctest cleanup
    /// std::fs::remove_file("subset_data.csv").unwrap();
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
impl Display for TestDataRecord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "TestDataRecord(time: {}, latitude: {}, longitude: {}, altitude: {}, speed: {}, bearing: {})",
            self.time, self.latitude, self.longitude, self.altitude, self.speed, self.bearing
        )
    }
}
/// Generic result struct for navigation simulations.
/// 
/// This structure contains a single row of position, velocity, and attitude vectors
/// representing the navigation solution at a specific timestamp, along with error estimates
/// and confidence values derived from filter covariance, when available.
///
/// It can be used across different types of navigation simulations such as dead reckoning,
/// Kalman filtering, or any other navigation algorithm.
#[derive(Debug, Serialize, Deserialize)]
pub struct NavigationResult {
    /// Timestamp corresponding to the state
    pub timestamp: String,
    /// Position values (latitude, longitude, altitude)
    pub latitude: f64,
    pub longitude: f64,
    pub altitude: f64,
    /// Velocity values (north, east, down)
    pub velocity_n: f64,
    pub velocity_e: f64,
    pub velocity_d: f64,
    /// Attitude values (roll, pitch, yaw)
    pub roll: f64,
    pub pitch: f64,
    pub yaw: f64,
    /// Position error estimates (latitude, longitude, altitude)
    pub position_error_lat: f64,
    pub position_error_lon: f64,
    pub position_error_alt: f64,
    /// Velocity error estimates (north, east, down)
    pub velocity_error_n: f64,
    pub velocity_error_e: f64,
    pub velocity_error_d: f64,
    /// Attitude error estimates (roll, pitch, yaw)
    pub attitude_error_roll: f64,
    pub attitude_error_pitch: f64,
    pub attitude_error_yaw: f64,
    /// Full state covariance matrix if available - serialized as a string in CSV
    #[serde(serialize_with = "serialize_covariance", deserialize_with = "deserialize_covariance")]
    pub covariance: Option<Vec<f64>>,
}

impl NavigationResult {
    /// Creates a new NavigationResult with default values.
    pub fn new(timestamp: &str,
            latitude: &f64,
            longitude: &f64,
            altitude: &f64,
            velocity_n: &f64,
            velocity_e: &f64,
            velocity_d: &f64,
            roll: &f64,
            pitch: &f64,
            yaw: &f64,
            position_error_lat: &f64,
            position_error_lon: &f64,
            position_error_alt: &f64,
            velocity_error_n: &f64,
            velocity_error_e: &f64,
            velocity_error_d: &f64,
            attitude_error_roll: &f64,
            attitude_error_pitch: &f64,
            attitude_error_yaw: &f64,
            covariance: Option<Vec<f64>>,
            ) -> Self {
        NavigationResult {
            timestamp: timestamp.to_string(),
            latitude: *latitude,
            longitude: *longitude,
            altitude: *altitude,
            velocity_n: *velocity_n,
            velocity_e: *velocity_e,
            velocity_d: *velocity_d,
            roll: *roll,
            pitch: *pitch,
            yaw: *yaw,
            position_error_lat: *position_error_lat,
            position_error_lon: *position_error_lon,
            position_error_alt: *position_error_alt,
            velocity_error_n: *velocity_error_n,
            velocity_error_e: *velocity_error_e,
            velocity_error_d: *velocity_error_d,
            attitude_error_roll: *attitude_error_roll,
            attitude_error_pitch: *attitude_error_pitch,
            attitude_error_yaw: *attitude_error_yaw,
            covariance
        }
    }
    /// Creates a new NavigationResult from a StrapdownState, and covariance.
    /// 
    /// This function takes a StrapdownState and a INS filter covariance matrix, and
    /// constructs a NavigationResult with the state values and covariance.
    /// 
    /// # Arguments
    /// * `state` - StrapdownState containing the current state of the navigation system
    /// * `timestamp` - Timestamp of the navigation solution
    /// * `covariance` - Covariance matrix from the filter, if available
    pub fn new_from_nav_state(state: &StrapdownState, timestamp: String, covariance: Option<DMatrix<f64>>) -> Self {
        let cov_vec: Option<Vec<f64>>;
        if let Some(cov) = covariance {
            cov_vec = Some(cov.as_slice().to_vec());
        } else {
            cov_vec = None;
        }
        NavigationResult {
            timestamp: timestamp.clone(),
            latitude: state.position[0],
            longitude: state.position[1],
            altitude: state.position[2],
            velocity_n: state.velocity[0],
            velocity_e: state.velocity[1],
            velocity_d: state.velocity[2],
            roll: state.attitude.euler_angles().0,
            pitch: state.attitude.euler_angles().1,
            yaw: state.attitude.euler_angles().2,
            position_error_lat: 0.0,
            position_error_lon: 0.0,
            position_error_alt: 0.0,
            velocity_error_n: 0.0,
            velocity_error_e: 0.0,
            velocity_error_d: 0.0,
            attitude_error_roll: 0.0,
            attitude_error_pitch: 0.0,
            attitude_error_yaw: 0.0,
            covariance: cov_vec,
        }
    }

    /// Creates a new NavigationResult from an nalgebra vector and covariance.
    ///
    /// This function creates a NavigationResult directly from a DVector representing the 9-state NED navigation solution,
    /// and an optional covariance matrix.
    /// 
    /// # Arguments
    /// * `state` - DVector containing the navigation state (latitude, longitude, altitude, velocity_n, velocity_e, velocity_d, roll, pitch, yaw)
    /// * `timestamp` - Timestamp of the navigation solution
    /// * `covariance` - Optional DMatrix representing the covariance of the state
    /// 
    /// # Returns
    /// * `NavigationResult` - A new NavigationResult instance with the state values and covariance.
    pub fn new_from_vector(
        state: &DVector<f64>,
        timestamp: String,
        covariance: Option<&DMatrix<f64>>,
    ) -> Self {
        let cov_vec: Option<Vec<f64>>;
        if let Some(cov) = covariance {
            cov_vec = Some(cov.as_slice().to_vec());
        } else {
            cov_vec = None;
        }
        NavigationResult {
            timestamp: timestamp.clone(),
            latitude: state[0],
            longitude: state[1],
            altitude: state[2],
            velocity_n: state[3],
            velocity_e: state[4],
            velocity_d: state[5],
            roll: state[6],
            pitch: state[7],
            yaw: state[8],
            position_error_lat: 0.0,
            position_error_lon: 0.0,
            position_error_alt: 0.0,
            velocity_error_n: 0.0,
            velocity_error_e: 0.0,
            velocity_error_d: 0.0,
            attitude_error_roll: 0.0,
            attitude_error_pitch: 0.0,
            attitude_error_yaw: 0.0,
            covariance: cov_vec,
        }
    }
    /// Writes the NavigationResult to a CSV file.
    /// 
    /// # Arguments
    /// * `records` - Vector of NavigationResult structs to write
    /// * `path` - Path where the CSV file will be saved
    /// 
    /// # Returns
    /// * `io::Result<()>` - Ok if successful, Err otherwise
    /// 
    /// # Example
    /// ```
    /// use strapdown::sim::NavigationResult;
    /// use std::path::Path;
    /// let records = vec![NavigationResult::new(
    ///     "2023-01-01 00:00:00+00:00", &0.0, &0.0, &0.0, &0.0, &0.0, &0.0, &0.0, &0.0, &0.0, 
    ///     &0.0, &0.0, &0.0, &0.0, &0.0, &0.0, &0.0, &0.0, &0.0, None )];
    /// NavigationResult::to_csv(&records, "navigation_results.csv").expect("Failed to write CSV");
    /// // doctest cleanup
    /// std::fs::remove_file("navigation_results.csv").unwrap();
    /// ```
    pub fn to_csv<P: AsRef<Path>>(records: &[Self], path: P) -> io::Result<()> {
        let mut writer = csv::Writer::from_path(path)?;
        
        for record in records {
            writer.serialize(record)?;
        }
        
        writer.flush()?;
        Ok(())
    }
    /// Reads a CSV file and returns a vector of NavigationResult structs.
    /// 
    /// # Arguments
    /// * `path` - Path to the CSV file to read.
    /// 
    /// # Returns
    /// * `Ok(Vec<NavigationResult>)` if successful.
    /// * `Err` if the file cannot be read or parsed.
    pub fn from_csv<P: AsRef<std::path::Path>>(path: P) -> Result<Vec<Self>, Box<dyn std::error::Error>> {
        let mut rdr = csv::Reader::from_path(path)?;
        let mut records = Vec::new();
        for result in rdr.deserialize() {
            let record: Self = result?;
            records.push(record);
        }
        Ok(records)
    }

}

/// Custom serializer for the covariance field to serialize it as a single string in CSV
fn serialize_covariance<S>(cov: &Option<Vec<f64>>, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    match cov {
        Some(vec) => {
            // Join the vector elements with commas and serialize as a single string
            let cov_str = vec
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<String>>()
                .join(",");
            serializer.serialize_str(&cov_str)
        }
        None => serializer.serialize_none(),
    }
}

/// Custom deserializer for the covariance field to deserialize from a string in CSV
fn deserialize_covariance<'de, D>(deserializer: D) -> Result<Option<Vec<f64>>, D::Error>
where
    D: Deserializer<'de>,
{
    let s: Option<String> = Option::deserialize(deserializer)?;
    match s {
        Some(s) if !s.is_empty() => {
            // Parse the comma-separated string back to a Vec<f64>
            let values: Result<Vec<f64>, _> = s
                .split(',')
                .map(|v| v.parse::<f64>())
                .collect();
            
            match values {
                Ok(vec) => Ok(Some(vec)),
                Err(_) => Ok(None), // Handle parsing errors gracefully
            }
        }
        _ => Ok(None),
    }
}

/// Run dead reckoning or "open-loop" simulation using test data.
///
/// This function processes a sequence of sensor records through a StrapdownState, using
/// the "forward" method to propagate the state based on IMU measurements. It initializes
/// the StrapdownState with position, velocity, and attitude from the first record, and
/// then applies the IMU measurements from subsequent records. It does not record the 
/// errors or confidence values, as this is a simple dead reckoning simulation and in testing
/// these values would be used as a baseline for comparison.
///
/// # Arguments
/// * `records` - Vector of test data records containing IMU measurements and other sensor data
///
/// # Returns
/// * `Vec<NavigationResult>` containing the sequence of StrapdownState instances over time,
///   along with timestamps and time differences.
pub fn dead_reckoning(records: &Vec<TestDataRecord>) -> Vec<NavigationResult> {
    if records.is_empty() {
        return Vec::new();
    }
    // Initialize the result vector
    let mut results = Vec::with_capacity(records.len());
    // Initialize the StrapdownState with the first record
    let first_record = &records[0];
    let mut state = StrapdownState::new();
    // Set initial position (latitude, longitude, altitude)
    state.position = Vector3::new(
        first_record.latitude,
        first_record.longitude,
        first_record.altitude,
    );
    // Set initial velocity (assuming zero initial velocity)
    state.velocity = Vector3::new(0.0, 0.0, 0.0);
    // Set initial attitude (roll, pitch, yaw from first record)
    state.attitude = nalgebra::Rotation3::from_euler_angles(
        first_record.roll,
        first_record.pitch,
        first_record.yaw,
    );
    // Store the initial state and metadata
    results.push(NavigationResult {
        timestamp: first_record.time.clone(),
        latitude: state.position[0],
        longitude: state.position[1],
        altitude: state.position[2],
        velocity_n: state.velocity[0],
        velocity_e: state.velocity[1],
        velocity_d: state.velocity[2],
        roll: state.attitude.euler_angles().0,
        pitch: state.attitude.euler_angles().1,
        yaw: state.attitude.euler_angles().2,
        position_error_lat: 0.0,
        position_error_lon: 0.0,
        position_error_alt: 0.0,
        velocity_error_n: 0.0,
        velocity_error_e: 0.0,
        velocity_error_d: 0.0,
        attitude_error_roll: 0.0,
        attitude_error_pitch: 0.0,
        attitude_error_yaw: 0.0,
        covariance: None,
    });
    // For the time difference calculation, use a fixed value if timestamps can't be parsed
    let mut prev_time_str = &first_record.time;
    // Process each subsequent record
    for record in records.iter().skip(1) {
        // Try to calculate time difference from timestamps, default to 1 second if parsing fails
        let dt = match (
            DateTime::parse_from_str(prev_time_str, "%Y-%m-%d %H:%M:%S%z"),
            DateTime::parse_from_str(&record.time, "%Y-%m-%d %H:%M:%S%z"),
        ) {
            (Ok(prev), Ok(current)) => (current - prev).num_milliseconds() as f64 / 1000.0,
            _ => 1.0, // Default to 1 second if parsing fails
        };
        // Create IMU data from the record
        let imu_data = IMUData::new_from_vec(
            vec![record.acc_x, record.acc_y, record.acc_z],
            vec![record.gyro_x, record.gyro_y, record.gyro_z],
        );

        // Propagate the state forward
        state.forward(&imu_data, dt);

        // Store the state and metadata
        results.push(NavigationResult {
            timestamp: record.time.clone(),
            latitude: state.position[0],
            longitude: state.position[1],
            altitude: state.position[2],
            velocity_n: state.velocity[0],
            velocity_e: state.velocity[1],
            velocity_d: state.velocity[2],
            roll: state.attitude.euler_angles().0,
            pitch: state.attitude.euler_angles().1,
            yaw: state.attitude.euler_angles().2,
            position_error_lat: 0.0,
            position_error_lon: 0.0,
            position_error_alt: 0.0,
            velocity_error_n: 0.0,
            velocity_error_e: 0.0,
            velocity_error_d: 0.0,
            attitude_error_roll: 0.0,
            attitude_error_pitch: 0.0,
            attitude_error_yaw: 0.0,
            covariance: None,
        });
        // Update previous time string for next iteration
        prev_time_str = &record.time;
    }
    results
}

/// Closed-loop GPS-aided inertial navigation simulation.
/// 
/// This function simulates a closed-loop full-state navigation system where GPS measurements are used 
/// to correct the inertial navigation solution. It implements an Unscented Kalman Filter (UKF) to propagate 
/// the state and update it with GPS measurements when available.
/// 
/// # Arguments
/// * `records` - Vector of test data records containing IMU measurements and GPS data.
/// # Returns
/// * `Vec<NavigationResult>` - A vector of navigation results containing the state estimates and covariances at each timestamp. 
pub fn closed_loop(records: &Vec<TestDataRecord>) -> Vec<NavigationResult> {
    let mut results: Vec<NavigationResult> = Vec::with_capacity(records.len());
    // Initialize the UKF with the first record
    let mut ukf = initialize_ukf(records[0].clone(), None, None);
    // Set the initial result to the UKF initial state
    results.push(NavigationResult::new_from_vector(
        &ukf.get_mean(),
        records[0].time.clone(),
        Some(&ukf.get_covariance()),
    ));
    // Clip to the first 1000 records for performance
    let records = if records.len() > 1000 {
        &records[0..3]
    } else {
        records
    };
    // Iterate through the records, updating the UKF with each IMU measurement
    for record in records.iter().skip(1) {
        // Calculate time difference from the previous record
        let dt = match (
            DateTime::parse_from_str(&results.last().unwrap().timestamp, "%Y-%m-%d %H:%M:%S%z"),
            DateTime::parse_from_str(&record.time, "%Y-%m-%d %H:%M:%S%z"),
        ) {
            (Ok(prev), Ok(current)) => (current - prev).num_milliseconds() as f64 / 1000.0,
            _ => 1.0, // Default to 1 second if parsing fails
        };
        // Create IMU data from the record
        let imu_data = IMUData::new_from_vec(
            vec![record.acc_x, record.acc_y, record.acc_z],
            vec![record.gyro_x, record.gyro_y, record.gyro_z],
        );
        // Update the UKF with the IMU data
        ukf.propagate(&imu_data, dt);
        // If GPS data is available, update the UKF with the GPS measurement
        if !record.latitude.is_nan() && !record.longitude.is_nan() && !record.altitude.is_nan() {
            let measurement = DVector::from_vec(
                vec![
                    record.longitude,
                    record.latitude,
                    record.altitude
                ]
            );
            // Create the measurement sigma points using the position measurement model
            let measurement_sigma_points = position_measurement_model(&ukf.get_sigma_points(), true);
            // Update the UKF with the GPS measurement
            ukf.update(&measurement, &measurement_sigma_points);
        } 
        // Store the current state and covariance in results
        results.push(NavigationResult::new_from_vector(
            &ukf.get_mean(),
            record.time.clone(),
            Some(&ukf.get_covariance()),
        ));
    }
    results
}

/// Helper function to initialize a UKF for closed-loop mode.
/// 
/// This function sets up the Unscented Kalman Filter (UKF) with initial pose, attitude covariance, and IMU biases based on 
/// the provided `TestDataRecord`. It initializes the UKF with position, velocity, attitude, and covariance matrices.
/// Optional parameters for attitude covariance and IMU biases can be provided to customize the filter's initial state.
/// 
/// # Arguments
/// 
/// * `initial_pose` - A `TestDataRecord` containing the initial pose information.
/// * `attitude_covariance` - Optional vector of f64 representing the initial attitude covariance (default is a small value).
/// * `imu_biases` - Optional vector of f64 representing the initial IMU biases (default is a small value).
/// 
/// # Returns
/// 
/// * `UKF` - An instance of the Unscented Kalman Filter initialized with the provided parameters. 
pub fn initialize_ukf(initial_pose: TestDataRecord, attitude_covariance: Option<Vec<f64>>, imu_biases: Option<Vec<f64>>) -> UKF {
    // Initialize the UKF with the initial pose
    let position = vec![initial_pose.longitude, initial_pose.latitude, initial_pose.altitude];
    let velocity = vec![
        initial_pose.speed * initial_pose.bearing.cos(),
        initial_pose.speed * initial_pose.bearing.sin(),
        0.0, // Assuming no vertical velocity for simplicity
    ];
    let attitude = vec![initial_pose.roll, initial_pose.pitch, initial_pose.yaw];
    let position_accuracy = initial_pose.horizontal_accuracy.sqrt();
    let mut covariance_diagonal = vec![
        position_accuracy * METERS_TO_DEGREES,
        position_accuracy * METERS_TO_DEGREES, 
        initial_pose.vertical_accuracy,
        initial_pose.speed_accuracy,
        initial_pose.speed_accuracy,
        initial_pose.speed_accuracy
    ];
    // extend the covariance diagonal if attitude covariance is provided
    match attitude_covariance {
        Some(att_cov) => covariance_diagonal.extend(att_cov),
        None => covariance_diagonal.extend(vec![1e-3; 3]), // Default values if not provided
    }
    // extend the covariance diagonal if imu biases are provided
    match imu_biases {
        Some(imu_biases) => covariance_diagonal.extend(imu_biases),
        None => covariance_diagonal.extend(vec![1e-3; 6]), // Default values if not provided
    }
    println!("Covariance diagonal: {:?}", covariance_diagonal);
    let mut process_noise_diagonal = vec![0.0; 9];
    process_noise_diagonal.extend(vec![1e-3; 6]); // Process noise for imu biases
    let process_noise_diagonal = DVector::from_vec(process_noise_diagonal);
    //DVector::from_vec(vec![0.0; 15]);
    let measurement_noise_diagonal = DVector::from_vec(vec![1e-12; 3]);
    UKF::new(
        position,
        velocity,
        attitude,
        vec![1e-3; 3],
        vec![0.0; 3],
        covariance_diagonal,
        DMatrix::from_diagonal(&process_noise_diagonal),
        DMatrix::from_diagonal(&measurement_noise_diagonal),
        1e-3,
        2.0,
        0.0,
    )
}


#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;
    use std::fs::File;

    /// Generate a test record for northward motion at constant velocity (1 knot = 1852 m/h).
    /// This helper returns a Vec<TestDataRecord> for 1 hour, sampled once per second.
    fn generate_northward_motion_records() -> Vec<TestDataRecord> {
        let mut records: Vec<TestDataRecord> = Vec::with_capacity(3601);
        let start_lat: f64 = 0.0;
        let start_lon: f64 = 0.0;
        let start_alt: f64 = 0.0;
        let velocity_mps: f64 = 1852.0 / 3600.0; // 1 knot in m/s
        let earth_radius: f64 = 6371000.0_f64; // meters

        for t in 0..3600 {
            // Each second, latitude increases by dlat = (v / R) * (180/pi)
            let dlat: f64 = (velocity_mps * t as f64) / earth_radius * (180.0 / std::f64::consts::PI);
            let time_str: String = format!("2023-01-01 00:{:02}:{:02}+00:00", t / 60, t % 60);

            records.push(TestDataRecord {
                time: time_str,
                bearing_accuracy: 0.0,
                speed_accuracy: 0.0,
                vertical_accuracy: 0.0,
                horizontal_accuracy: 0.0,
                speed: velocity_mps,
                bearing: 0.0,
                altitude: start_alt,
                longitude: start_lon,
                latitude: start_lat + dlat,
                qz: 0.0,
                qy: 0.0,
                qx: 0.0,
                qw: 1.0,
                roll: 0.0,
                pitch: 0.0,
                yaw: 0.0,
                acc_z: 0.0,
                acc_y: 0.0,
                acc_x: 0.0,
                gyro_z: 0.0,
                gyro_y: 0.0,
                gyro_x: 0.0,
                mag_z: 0.0,
                mag_y: 0.0,
                mag_x: 0.0,
                relative_altitude: 0.0,
                pressure: 1000.0,
                grav_z: 9.81,
                grav_y: 0.0,
                grav_x: 0.0,
            });
        }
        records
    }
    #[test]
    fn test_generate_northward_motion_records_end_latitude() {
        let records = generate_northward_motion_records();
        // The last record should have latitude close to 0.016667 (1 knot north in 1 hour)
        let last = records.last().unwrap();
        let expected_lat = 0.016667;
        let tolerance = 1e-3;
        assert!(
            (last.latitude - expected_lat).abs() < tolerance,
            "Ending latitude {} not within {} of expected {}",
            last.latitude,
            tolerance,
            expected_lat
        );
        // write to CSV
        let northward = File::create("northward_motion.csv").unwrap();
        let mut writer = csv::Writer::from_writer(northward);
        for record in &records {
            writer.serialize(record).unwrap();
        }
        writer.flush().unwrap();
        // Clean up the test file
        let _ = std::fs::remove_file("northward_motion.csv");
    }
    /// Test that reading a valid CSV file returns records and parses fields correctly.
    #[test]
    fn test_test_data_record_from_csv() {
        let path = Path::new("./data/test_data.csv");
        let records = TestDataRecord::from_csv(path).expect("Failed to read test_data.csv");
        assert!(!records.is_empty(), "CSV should not be empty");
        // Check a few fields of the first record
        let first = &records[0];
        assert!(!first.time.is_empty());
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
    
}