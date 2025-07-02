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

use chrono::{DateTime, Utc};
use nalgebra::{DMatrix, DVector, Rotation3, Vector3};
use serde::{Deserialize, Serialize};

use crate::earth;
use crate::earth::METERS_TO_DEGREES;
use crate::filter::{GPS, StrapdownParams, UKF};
use crate::{IMUData, StrapdownState, forward};
/// Struct representing a single row of test data from the CSV file.
///
/// Fields correspond to columns in the CSV, with appropriate renaming for Rust style.
/// This struct is setup to capture the data recorded from the [Sensor Logger](https://www.tszheichoi.com/sensorlogger) app.
/// Primarily, this represents IMU data as (relative to the device) and GPS data.
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct TestDataRecord {
    /// Date-time string: YYYY-MM-DD hh:mm:ss+UTCTZ
    //#[serde(with = "ts_seconds")]
    pub time: DateTime<Utc>,
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
    pub fn from_csv<P: AsRef<std::path::Path>>(
        path: P,
    ) -> Result<Vec<Self>, Box<dyn std::error::Error>> {
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
    /// let record = TestDataRecord {
    ///     time: chrono::Utc::now(),
    ///     bearing_accuracy: 0.1,
    ///     speed_accuracy: 0.1,
    ///     vertical_accuracy: 0.1,
    ///     horizontal_accuracy: 0.1,
    ///     speed: 1.0,
    ///     bearing: 90.0,
    ///     altitude: 100.0,
    ///     longitude: -122.0,
    ///     latitude: 37.0,
    ///     qz: 0.0,
    ///     qy: 0.0,
    ///     qx: 0.0,
    ///     qw: 1.0,
    ///     roll: 0.0,
    ///     pitch: 0.0,
    ///     yaw: 0.0,
    ///     acc_z: 9.81,
    ///     acc_y: 0.0,
    ///     acc_x: 0.0,
    ///     gyro_z: 0.01,
    ///     gyro_y: 0.01,
    ///     gyro_x: 0.01,
    ///     mag_z: 50.0,
    ///     mag_y: -30.0,
    ///     mag_x: -20.0,
    ///     relative_altitude: 0.0,
    ///     pressure: 1013.25,
    ///     grav_z: 9.81,
    ///     grav_y: 0.0,
    ///     grav_x: 0.0,
    /// };
    /// let records = vec![record];
    /// TestDataRecord::to_csv(&records, "data.csv")
    ///    .expect("Failed to write test data to CSV");
    /// // doctest cleanup
    /// std::fs::remove_file("data.csv").unwrap();
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
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct NavigationResult {
    /// Timestamp corresponding to the state
    pub timestamp: DateTime<Utc>,
    /// Latitude in radians
    pub latitude: f64,
    /// Longitude in radians
    pub longitude: f64,
    /// Altitude in meters
    pub altitude: f64,
    /// Northward velocity in m/s
    pub velocity_north: f64,
    /// Eastward velocity in m/s
    pub velocity_east: f64,
    /// Downward velocity in m/s
    pub velocity_down: f64,
    /// Roll angle in radians
    pub roll: f64,
    /// Pitch angle in radians
    pub pitch: f64,
    /// Yaw angle in radians
    pub yaw: f64,
    /// IMU accelerometer x-axis bias in m/s^2
    pub acc_bias_x: f64,
    /// IMU accelerometer y-axis bias in m/s^2
    pub acc_bias_y: f64,
    /// IMU accelerometer z-axis bias in m/s^2
    pub acc_bias_z: f64,
    /// IMU gyroscope x-axis bias in radians/s
    pub gyro_bias_x: f64,
    /// IMU gyroscope y-axis bias in radians/s
    pub gyro_bias_y: f64,
    /// IMU gyroscope z-axis bias in radians/s
    pub gyro_bias_z: f64,
    /// Latitude covariance
    pub latitude_cov: f64,
    /// Longitude covariance
    pub longitude_cov: f64,
    /// Altitude covariance
    pub altitude_cov: f64,
    /// Northward velocity covariance
    pub velocity_n_cov: f64,
    /// Eastward velocity covariance
    pub velocity_e_cov: f64,
    /// Downward velocity covariance
    pub velocity_d_cov: f64,
    /// Roll covariance
    pub roll_cov: f64,
    /// Pitch covariance
    pub pitch_cov: f64,
    /// Yaw covariance
    pub yaw_cov: f64,
    /// Accelerometer x-axis bias covariance
    pub acc_bias_x_cov: f64,
    /// Accelerometer y-axis bias covariance
    pub acc_bias_y_cov: f64,
    /// Accelerometer z-axis bias covariance
    pub acc_bias_z_cov: f64,
    /// Gyroscope x-axis bias covariance
    pub gyro_bias_x_cov: f64,
    /// Gyroscope y-axis bias covariance
    pub gyro_bias_y_cov: f64,
    /// Gyroscope z-axis bias covariance
    pub gyro_bias_z_cov: f64,
    // ---- Input and measurement values for the navigation solution ----
    /// X-acceleration in m/s^2
    pub acc_x: f64,
    /// Y-acceleration in m/s^2
    pub acc_y: f64,
    /// Z-acceleration in m/s^2
    pub acc_z: f64,
    /// Rotation rate around the x-axis in radians/s
    pub gyro_x: f64,
    /// Rotation rate around the y-axis in radians/s
    pub gyro_y: f64,
    /// Rotation rate around the z-axis in radians/s
    pub gyro_z: f64,
    /// Magnetic field strength in the x-direction in microteslas
    pub mag_x: f64,
    /// Magnetic field strength in the y-direction in microteslas
    pub mag_y: f64,
    /// Magnetic field strength in the z-direction in microteslas
    pub mag_z: f64,
    /// Pressure in millibars
    pub pressure: f64,
    // ---- Calculated geophysical values that may be useful ----
    /// Free-air gravity anomaly in mGal
    pub freeair: f64,
    /// Magnetic field strength anomaly in nT
    pub mag_anomaly: f64,
}
impl Default for NavigationResult {
    fn default() -> Self {
        Self::new()
    }
}

impl NavigationResult {
    /// Creates a new NavigationResult with default values.
    pub fn new() -> Self {
        NavigationResult {
            timestamp: Utc::now(),
            latitude: 0.0,
            longitude: 0.0,
            altitude: 0.0,
            velocity_north: 0.0,
            velocity_east: 0.0,
            velocity_down: 0.0,
            roll: 0.0,
            pitch: 0.0,
            yaw: 0.0,
            acc_bias_x: 0.0,
            acc_bias_y: 0.0,
            acc_bias_z: 0.0,
            gyro_bias_x: 0.0,
            gyro_bias_y: 0.0,
            gyro_bias_z: 0.0,
            latitude_cov: 1e-6, // default covariance values
            longitude_cov: 1e-6,
            altitude_cov: 1e-6,
            velocity_n_cov: 1e-6,
            velocity_e_cov: 1e-6,
            velocity_d_cov: 1e-6,
            roll_cov: 1e-6,
            pitch_cov: 1e-6,
            yaw_cov: 1e-6,
            acc_bias_x_cov: 1e-6,
            acc_bias_y_cov: 1e-6,
            acc_bias_z_cov: 1e-6,
            gyro_bias_x_cov: 1e-6,
            gyro_bias_y_cov: 1e-6,
            gyro_bias_z_cov: 1e-6,
            acc_x: 0.0,
            acc_y: 0.0,
            acc_z: 9.81, // assuming standard gravity
            gyro_x: 0.0,
            gyro_y: 0.0,
            gyro_z: 0.0,
            mag_x: earth::MAGNETIC_FIELD_STRENGTH, //50.0, // default magnetic field strength
            mag_y: -30.0,                          // default values
            mag_z: -20.0,                          // default values
            pressure: 1013.25,                     // standard atmospheric pressure in millibars
            freeair: 9.81 * METERS_TO_DEGREES,     // free-air gravity anomaly in mGal
            mag_anomaly: -30.0,                    // default magnetic anomaly in nT
        }
    }
    /// Creates a new NavigationResult from a StrapdownState, and covariance.
    ///
    /// This function takes a StrapdownState and a INS filter covariance matrix, and
    /// constructs a NavigationResult with the state values and covariance.
    ///
    /// # Arguments
    /// * `state` - StrapdownState containing the current state of the navigation system
    /// * `timestamp` - DateTime<Utc>, the timestamp of the navigation solution
    pub fn new_from_nav_state(
        state: &StrapdownState,
        timestamp: DateTime<Utc>,
        acc_bias_x: f64,
        acc_bias_y: f64,
        acc_bias_z: f64,
        gyro_bias_x: f64,
        gyro_bias_y: f64,
        gyro_bias_z: f64,
        latitude_cov: f64,
        longitude_cov: f64,
        altitude_cov: f64,
        velocity_n_cov: f64,
        velocity_e_cov: f64,
        velocity_d_cov: f64,
        roll_cov: f64,
        pitch_cov: f64,
        yaw_cov: f64,
        acc_bias_x_cov: f64,
        acc_bias_y_cov: f64,
        acc_bias_z_cov: f64,
        gyro_bias_x_cov: f64,
        gyro_bias_y_cov: f64,
        gyro_bias_z_cov: f64,
        acc_x: f64,
        acc_y: f64,
        acc_z: f64,
        gyro_x: f64,
        gyro_y: f64,
        gyro_z: f64,
        mag_x: f64,
        mag_y: f64,
        mag_z: f64,
        pressure: f64,
        freeair: f64,
        mag_anomaly: f64,
    ) -> Self {
        NavigationResult {
            timestamp,
            latitude: state.latitude,
            longitude: state.longitude,
            altitude: state.altitude,
            velocity_north: state.velocity_north,
            velocity_east: state.velocity_east,
            velocity_down: state.velocity_down,
            roll: state.attitude.euler_angles().0,
            pitch: state.attitude.euler_angles().1,
            yaw: state.attitude.euler_angles().2,
            acc_bias_x,
            acc_bias_y,
            acc_bias_z,
            gyro_bias_x,
            gyro_bias_y,
            gyro_bias_z,
            latitude_cov,
            longitude_cov,
            altitude_cov,
            velocity_n_cov,
            velocity_e_cov,
            velocity_d_cov,
            roll_cov,
            pitch_cov,
            yaw_cov,
            acc_bias_x_cov,
            acc_bias_y_cov,
            acc_bias_z_cov,
            gyro_bias_x_cov,
            gyro_bias_y_cov,
            gyro_bias_z_cov,
            acc_x,
            acc_y,
            acc_z,
            gyro_x,
            gyro_y,
            gyro_z,
            mag_x,
            mag_y,
            mag_z,
            pressure,
            freeair,
            mag_anomaly,
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
        timestamp: DateTime<Utc>,
        state: &DVector<f64>,
        covariance: Option<&DMatrix<f64>>,
        imu_data: &IMUData,
        mag_x: f64,
        mag_y: f64,
        mag_z: f64,
        pressure: f64,
        freeair: f64,
        mag_anomaly: f64,
    ) -> Self {
        let covariance_vec = match covariance {
            Some(cov) => cov.diagonal().iter().cloned().collect(),
            None => vec![0.0; state.len()],
        };
        NavigationResult {
            timestamp,
            latitude: state[0],
            longitude: state[1],
            altitude: state[2],
            velocity_north: state[3],
            velocity_east: state[4],
            velocity_down: state[5],
            roll: state[6],
            pitch: state[7],
            yaw: state[8],
            acc_bias_x: state[9],
            acc_bias_y: state[10],
            acc_bias_z: state[11],
            gyro_bias_x: state[12],
            gyro_bias_y: state[13],
            gyro_bias_z: state[14],
            latitude_cov: covariance_vec.first().cloned().unwrap_or(0.0),
            longitude_cov: covariance_vec.get(1).cloned().unwrap_or(0.0),
            altitude_cov: covariance_vec.get(2).cloned().unwrap_or(0.0),
            velocity_n_cov: covariance_vec.get(3).cloned().unwrap_or(0.0),
            velocity_e_cov: covariance_vec.get(4).cloned().unwrap_or(0.0),
            velocity_d_cov: covariance_vec.get(5).cloned().unwrap_or(0.0),
            roll_cov: covariance_vec.get(6).cloned().unwrap_or(0.0),
            pitch_cov: covariance_vec.get(7).cloned().unwrap_or(0.0),
            yaw_cov: covariance_vec.get(8).cloned().unwrap_or(0.0),
            acc_bias_x_cov: covariance_vec.get(9).cloned().unwrap_or(0.0),
            acc_bias_y_cov: covariance_vec.get(10).cloned().unwrap_or(0.0),
            acc_bias_z_cov: covariance_vec.get(11).cloned().unwrap_or(0.0),
            gyro_bias_x_cov: covariance_vec.get(12).cloned().unwrap_or(0.0),
            gyro_bias_y_cov: covariance_vec.get(13).cloned().unwrap_or(0.0),
            gyro_bias_z_cov: covariance_vec.get(14).cloned().unwrap_or(0.0),
            acc_x: imu_data.accel[0],
            acc_y: imu_data.accel[1],
            acc_z: imu_data.accel[2],
            gyro_x: imu_data.gyro[0],
            gyro_y: imu_data.gyro[1],
            gyro_z: imu_data.gyro[2],
            mag_x,
            mag_y,
            mag_z,
            pressure,
            freeair,
            mag_anomaly,
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
    pub fn from_csv<P: AsRef<std::path::Path>>(
        path: P,
    ) -> Result<Vec<Self>, Box<dyn std::error::Error>> {
        let mut rdr = csv::Reader::from_path(path)?;
        let mut records = Vec::new();
        for result in rdr.deserialize() {
            let record: Self = result?;
            records.push(record);
        }
        Ok(records)
    }
}
/// Convert NED UKF to NavigationResult.
impl From<(&DateTime<Utc>, &UKF, &IMUData, &f64, &f64, &f64, &f64)> for NavigationResult {
    fn from(
        (timestamp, ukf, imu_data, mag_x, mag_y, mag_z, pressure): (
            &DateTime<Utc>,
            &UKF,
            &IMUData,
            &f64,
            &f64,
            &f64,
            &f64,
        ),
    ) -> Self {
        let state = ukf.get_mean();
        let covariance = ukf.get_covariance();
        NavigationResult {
            timestamp: *timestamp,
            latitude: state[0].to_degrees(),
            longitude: state[1].to_degrees(),
            altitude: state[2],
            velocity_north: state[3],
            velocity_east: state[4],
            velocity_down: state[5],
            roll: state[6],
            pitch: state[7],
            yaw: state[8],
            acc_bias_x: state[9],
            acc_bias_y: state[10],
            acc_bias_z: state[11],
            gyro_bias_x: state[12],
            gyro_bias_y: state[13],
            gyro_bias_z: state[14],
            latitude_cov: covariance[(0, 0)],
            longitude_cov: covariance[(1, 1)],
            altitude_cov: covariance[(2, 2)],
            velocity_n_cov: covariance[(3, 3)],
            velocity_e_cov: covariance[(4, 4)],
            velocity_d_cov: covariance[(5, 5)],
            roll_cov: covariance[(6, 6)],
            pitch_cov: covariance[(7, 7)],
            yaw_cov: covariance[(8, 8)],
            acc_bias_x_cov: covariance[(9, 9)],
            acc_bias_y_cov: covariance[(10, 10)],
            acc_bias_z_cov: covariance[(11, 11)],
            gyro_bias_x_cov: covariance[(12, 12)],
            gyro_bias_y_cov: covariance[(13, 13)],
            gyro_bias_z_cov: covariance[(14, 14)],
            acc_x: imu_data.accel[0],
            acc_y: imu_data.accel[1],
            acc_z: imu_data.accel[2],
            gyro_x: imu_data.gyro[0],
            gyro_y: imu_data.gyro[1],
            gyro_z: imu_data.gyro[2],
            mag_x: *mag_x,
            mag_y: *mag_y,
            mag_z: *mag_z,
            pressure: *pressure,
            freeair: earth::gravity_anomaly(
                &state[0].to_radians(),
                &state[2],
                &state[3],
                &state[4],
                &(imu_data.accel[0].powi(2)
                    + imu_data.accel[1].powi(2)
                    + imu_data.accel[2].powi(2))
                .sqrt(),
            ),
            mag_anomaly: earth::magnetic_anomaly(
                state[0].to_radians(),
                state[1].to_radians(),
                state[2],
                *mag_x,
                *mag_y,
                *mag_z,
            ),
        }
    }
}
impl
    From<(
        &DateTime<Utc>,
        &StrapdownState,
        &IMUData,
        &f64,
        &f64,
        &f64,
        &f64,
    )> for NavigationResult
{
    fn from(
        (timestamp, state, imu_data, mag_x, mag_y, mag_z, pressure): (
            &DateTime<Utc>,
            &StrapdownState,
            &IMUData,
            &f64,
            &f64,
            &f64,
            &f64,
        ),
    ) -> Self {
        //let state = ukf.get_mean();
        //let covariance = ukf.get_covariance();
        NavigationResult {
            timestamp: *timestamp,
            latitude: state.latitude.to_degrees(),
            longitude: state.longitude.to_degrees(),
            altitude: state.altitude,
            velocity_north: state.velocity_north,
            velocity_east: state.velocity_east,
            velocity_down: state.velocity_down,
            roll: state.attitude.euler_angles().0,
            pitch: state.attitude.euler_angles().1,
            yaw: state.attitude.euler_angles().2,
            acc_bias_x: 0.0,
            acc_bias_y: 0.0,
            acc_bias_z: 0.0,
            gyro_bias_x: 0.0,
            gyro_bias_y: 0.0,
            gyro_bias_z: 0.0,
            latitude_cov: 0.0,
            longitude_cov: 0.0,
            altitude_cov: 0.0,
            velocity_n_cov: 0.0,
            velocity_e_cov: 0.0,
            velocity_d_cov: 0.0,
            roll_cov: 0.0,
            pitch_cov: 0.0,
            yaw_cov: 0.0,
            acc_bias_x_cov: 0.0,
            acc_bias_y_cov: 0.0,
            acc_bias_z_cov: 0.0,
            gyro_bias_x_cov: 0.0,
            gyro_bias_y_cov: 0.0,
            gyro_bias_z_cov: 0.0,
            acc_x: imu_data.accel[0],
            acc_y: imu_data.accel[1],
            acc_z: imu_data.accel[2],
            gyro_x: imu_data.gyro[0],
            gyro_y: imu_data.gyro[1],
            gyro_z: imu_data.gyro[2],
            mag_x: *mag_x,
            mag_y: *mag_y,
            mag_z: *mag_z,
            pressure: *pressure,
            freeair: earth::gravity_anomaly(
                &state.latitude,
                &state.altitude,
                &state.velocity_north,
                &state.velocity_east,
                &(imu_data.accel[0].powi(2)
                    + imu_data.accel[1].powi(2)
                    + imu_data.accel[2].powi(2))
                .sqrt(),
            ),
            mag_anomaly: earth::magnetic_anomaly(
                state.latitude,
                state.longitude,
                state.altitude,
                *mag_x,
                *mag_y,
                *mag_z,
            ),
        }
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
pub fn dead_reckoning(records: &[TestDataRecord]) -> Vec<NavigationResult> {
    if records.is_empty() {
        return Vec::new();
    }
    // Initialize the result vector
    let mut results = Vec::with_capacity(records.len());
    // Initialize the StrapdownState with the first record
    let first_record = &records[0];
    let attitude = nalgebra::Rotation3::from_euler_angles(
        first_record.roll,
        first_record.pitch,
        first_record.yaw,
    );
    let mut state = StrapdownState {
        latitude: first_record.latitude.to_radians(),
        longitude: first_record.longitude.to_radians(),
        altitude: first_record.altitude,
        velocity_north: first_record.speed * first_record.bearing.cos(),
        velocity_east: first_record.speed * first_record.bearing.sin(),
        velocity_down: 0.0, // initial velocities
        attitude,
        coordinate_convention: true,
    };
    // Store the initial state and metadata
    results.push(NavigationResult::from((
        &records[0].time,
        &state,
        &IMUData::new_from_vec(
            vec![first_record.acc_x, first_record.acc_y, first_record.acc_z],
            vec![
                first_record.gyro_x,
                first_record.gyro_y,
                first_record.gyro_z,
            ],
        ),
        &first_record.mag_x,
        &first_record.mag_y,
        &first_record.mag_z,
        &first_record.pressure,
    )));
    let mut previous_time = records[0].time;
    // Process each subsequent record
    for record in records.iter().skip(1) {
        // Try to calculate time difference from timestamps, default to 1 second if parsing fails
        let current_time = record.time;
        let dt = (current_time - previous_time).as_seconds_f64();
        // Create IMU data from the record
        let imu_data = IMUData::new_from_vec(
            vec![record.acc_x, record.acc_y, record.acc_z],
            vec![record.gyro_x, record.gyro_y, record.gyro_z],
        );
        // Propagate the state forward (replace with stub for now)
        state = forward(state, imu_data, dt);
        results.push(NavigationResult::from((
            &current_time,
            &state,
            &imu_data,
            &record.mag_x,
            &record.mag_y,
            &record.mag_z,
            &record.pressure,
        )));
        previous_time = record.time;
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
pub fn closed_loop(
    records: &[TestDataRecord],
    gps_interval: Option<usize>,
) -> Vec<NavigationResult> {
    let gps_interval = gps_interval.unwrap_or(1); // Default to every record if not specified
    let mut results: Vec<NavigationResult> = Vec::with_capacity(records.len());
    // Initialize the UKF with the first record
    let mut ukf = initialize_ukf(
        records[0].clone(),
        None, //Some(vec![1e-9; 3]), // attitude covariance
        None, //Some(vec![1e-6; 6])
    );
    // Set the initial result to the UKF initial state
    results.push(NavigationResult::from((
        &records[0].time,
        &ukf,
        &IMUData::new_from_vec(
            vec![
                records[0].acc_x, // initial accelerometer data
                records[0].acc_y,
                records[0].acc_z,
            ],
            vec![
                records[0].gyro_x, // initial gyroscope data
                records[0].gyro_y,
                records[0].gyro_z,
            ],
        ),
        &records[0].mag_x,
        &records[0].mag_y,
        &records[0].mag_z,
        &records[0].pressure,
    )));
    let mut previous_timestamp = records[0].time;
    // Iterate through the records, updating the UKF with each IMU measurement
    let total: usize = records.len();
    let mut i: usize = 1;
    for record in records.iter().skip(1) {
        // Print progress every 100 iterations
        if i % 10 == 0 || i == total - 1 {
            print!(
                "\rProcessing data {:.2}%...",
                (i as f64 / total as f64) * 100.0
            );
            //print_ukf(&ukf, record);
            use std::io::Write;
            std::io::stdout().flush().ok();
        }
        // Calculate time difference from the previous record
        let current_timestamp = record.time;
        let dt = (current_timestamp - previous_timestamp).as_seconds_f64();
        // Create IMU data from the record subtracting out biases
        let mean = ukf.get_mean();
        let imu_data = IMUData::new_from_vec(
            vec![
                record.acc_x - mean[9], // subtract accel bias
                record.acc_y - mean[10],
                record.acc_z - mean[11],
            ],
            vec![
                record.gyro_x - mean[12], // subtract gyro bias
                record.gyro_y - mean[13],
                record.gyro_z - mean[14],
            ],
        );
        // Update the UKF with the IMU data
        ukf.predict(&imu_data, dt);
        // If GPS data is available, update the UKF with the GPS measurement
        if !record.latitude.is_nan()
            && !record.longitude.is_nan()
            && !record.altitude.is_nan()
            && i % gps_interval == 0
        {
            let mean = ukf.get_mean();
            //let angles = mean.columns(6, 3);
            //println!("UKF attitude angles: {:?}", angles);
            let attitude = Rotation3::from_euler_angles(mean[5], mean[6], mean[7]);
            let velocity = attitude * Vector3::new(record.speed, 0.0, 0.0);
            let measurement = DVector::from_vec(vec![
                record.latitude.to_radians(),
                record.longitude.to_radians(),
                record.altitude,
                velocity[0],
                velocity[1],
                velocity[2],
            ]);
            // Create the measurement sigma points using the position measurement model
            let measurement_sigma_points = ukf.position_and_velocity_measurement_model(true);
            // print the sigma points for debugging
            // println!(
            //     "Measurement Sigma Points:"
            // );
            // for point in measurement_sigma_points.iter() {
            //     println!("{:?}", point);
            // }
            let measurement_noise = ukf.position_and_velocity_measurement_noise(true);
            //println!("Measurement noise: {:?}", &measurement);
            // Update the UKF with the GPS measurement
            ukf.update(&measurement, &measurement_sigma_points, &measurement_noise);
        }
        // Store the current state and covariance in results
        results.push(NavigationResult::from((
            &current_timestamp,
            &ukf,
            &imu_data,
            &record.mag_x,
            &record.mag_y,
            &record.mag_z,
            &record.pressure,
        )));
        i += 1;
        previous_timestamp = current_timestamp;
    }
    // Print newline at the end to avoid overwriting the last line
    println!("Done!");
    results
}
/// Print the UKF state and covariance for debugging purposes.
pub fn print_ukf(ukf: &UKF, record: &TestDataRecord) {
    println!(
        "\rUKF position: ({:.4}, {:.4}, {:.4})  |  Covariance: {:.4e}, {:.4e}, {:.4}  |  Error: {:.4e}, {:.4e}, {:.4}",
        ukf.get_mean()[0].to_degrees(),
        ukf.get_mean()[1].to_degrees(),
        ukf.get_mean()[2],
        ukf.get_covariance()[(0, 0)],
        ukf.get_covariance()[(1, 1)],
        ukf.get_covariance()[(2, 2)],
        ukf.get_mean()[0].to_degrees() - record.latitude,
        ukf.get_mean()[1].to_degrees() - record.longitude,
        ukf.get_mean()[2] - record.altitude
    );
    println!(
        "\rUKF velocity: ({:.4}, {:.4}, {:.4})  | Covariance: {:.4}, {:.4}, {:.4}  | Error: {:.4}, {:.4}, {:.4}",
        ukf.get_mean()[3],
        ukf.get_mean()[4],
        ukf.get_mean()[5],
        ukf.get_covariance()[(3, 3)],
        ukf.get_covariance()[(4, 4)],
        ukf.get_covariance()[(5, 5)],
        ukf.get_mean()[3] - record.speed * record.bearing.cos(),
        ukf.get_mean()[4] - record.speed * record.bearing.sin(),
        ukf.get_mean()[5] - 0.0 // Assuming no vertical velocity
    );
    println!(
        "\rUKF attitude: ({:.4}, {:.4}, {:.4})  | Covariance: {:.4}, {:.4}, {:.4}  | Error: {:.4}, {:.4}, {:.4}",
        ukf.get_mean()[6],
        ukf.get_mean()[7],
        ukf.get_mean()[8],
        ukf.get_covariance()[(6, 6)],
        ukf.get_covariance()[(7, 7)],
        ukf.get_covariance()[(8, 8)],
        ukf.get_mean()[6] - record.roll,
        ukf.get_mean()[7] - record.pitch,
        ukf.get_mean()[8] - record.yaw
    );
    println!(
        "\rUKF accel biases: ({:.4}, {:.4}, {:.4})  | Covariance: {:.4e}, {:.4e}, {:.4e}",
        ukf.get_mean()[9],
        ukf.get_mean()[10],
        ukf.get_mean()[11],
        ukf.get_covariance()[(9, 9)],
        ukf.get_covariance()[(10, 10)],
        ukf.get_covariance()[(11, 11)]
    );
    println!(
        "\rUKF gyro biases: ({:.4}, {:.4}, {:.4})  | Covariance: {:.4e}, {:.4e}, {:.4e}",
        ukf.get_mean()[12],
        ukf.get_mean()[13],
        ukf.get_mean()[14],
        ukf.get_covariance()[(12, 12)],
        ukf.get_covariance()[(13, 13)],
        ukf.get_covariance()[(14, 14)]
    );
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
pub fn initialize_ukf(
    initial_pose: TestDataRecord,
    attitude_covariance: Option<Vec<f64>>,
    imu_biases: Option<Vec<f64>>,
) -> UKF {
    let ukf_params = StrapdownParams {
        latitude: initial_pose.latitude,
        longitude: initial_pose.longitude,
        altitude: initial_pose.altitude,
        northward_velocity: initial_pose.speed * initial_pose.bearing.cos(),
        eastward_velocity: initial_pose.speed * initial_pose.bearing.sin(),
        downward_velocity: 0.0, // Assuming no vertical velocity for simplicity
        roll: initial_pose.roll,
        pitch: initial_pose.pitch,
        yaw: initial_pose.yaw,
        in_degrees: true,
    };
    // Covariance parameters
    let position_accuracy = initial_pose.horizontal_accuracy; //.sqrt();
    let mut covariance_diagonal = vec![
        (position_accuracy * METERS_TO_DEGREES).powf(2.0),
        (position_accuracy * METERS_TO_DEGREES).powf(2.0),
        initial_pose.vertical_accuracy.powf(2.0),
        initial_pose.speed_accuracy.powf(2.0),
        initial_pose.speed_accuracy.powf(2.0),
        initial_pose.speed_accuracy.powf(2.0),
    ];
    // extend the covariance diagonal if attitude covariance is provided
    match attitude_covariance {
        Some(att_cov) => covariance_diagonal.extend(att_cov),
        None => covariance_diagonal.extend(vec![1e-9; 3]), // Default values if not provided
    }
    // extend the covariance diagonal if imu biases are provided
    let imu_bias = match imu_biases {
        Some(imu_biases) => {
            covariance_diagonal.extend(&imu_biases);
            imu_biases
        }
        None => {
            covariance_diagonal.extend(vec![1e-3; 6]);
            vec![1e-3; 6] // Default values if not provided
        }
    };
    let mut process_noise_diagonal = vec![1e-9; 9]; // adds a minor amount of noise to base states
    process_noise_diagonal.extend(vec![1e-9; 6]); // Process noise for imu biases
    let process_noise_diagonal = DVector::from_vec(process_noise_diagonal);
    //DVector::from_vec(vec![0.0; 15]);
    UKF::new(
        ukf_params,
        imu_bias,
        None,
        covariance_diagonal,
        DMatrix::from_diagonal(&process_noise_diagonal),
        1e-3, // Use a scalar for measurement noise as expected by UKF::new
        2.0,
        0.0,
    )
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::path::Path;
    use std::vec;

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
            let dlat: f64 =
                (velocity_mps * t as f64) / earth_radius * (180.0 / std::f64::consts::PI);
            let time_str: String = format!("2023-01-01 00:{:02}:{:02}+00:00", t / 60, t % 60);

            records.push(TestDataRecord {
                time: DateTime::parse_from_str(&time_str, "%Y-%m-%d %H:%M:%S%z")
                    .map(|dt| dt.with_timezone(&Utc))
                    .unwrap(),
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
    /// Test that reading a missing file returns an error.
    #[test]
    fn test_test_data_record_from_csv_invalid_path() {
        let path = Path::new("nonexistent.csv");
        let result = TestDataRecord::from_csv(path);
        assert!(result.is_err(), "Should error on missing file");
    }
    /// Test writing TestDataRecord to CSV and reading it back
    #[test]
    fn test_data_record_to_and_from_csv() {
        // Read original records
        let path = Path::new("test_file.csv");
        let mut records: Vec<TestDataRecord> = vec![];
        // Create some test data records
        records.push(TestDataRecord {
            time: DateTime::parse_from_str("2023-01-01 00:00:00+00:00", "%Y-%m-%d %H:%M:%S%z")
                .unwrap()
                .with_timezone(&Utc),
            bearing_accuracy: 0.1,
            speed_accuracy: 0.1,
            vertical_accuracy: 0.1,
            horizontal_accuracy: 0.1,
            speed: 1.0,
            bearing: 90.0,
            altitude: 100.0,
            longitude: -122.0,
            latitude: 37.0,
            qz: 0.0,
            qy: 0.0,
            qx: 0.0,
            qw: 1.0,
            roll: 0.0,
            pitch: 0.0,
            yaw: 0.0,
            acc_z: 9.81,
            acc_y: 0.0,
            acc_x: 0.0,
            gyro_z: 0.01,
            gyro_y: 0.01,
            gyro_x: 0.01,
            mag_z: 50.0,
            mag_y: -30.0,
            mag_x: -20.0,
            relative_altitude: 5.0,
            pressure: 1013.25,
            grav_z: 9.81,
            grav_y: 0.0,
            grav_x: 0.0,
        });
        records.push(TestDataRecord {
            time: DateTime::parse_from_str("2023-01-01 00:01:00+00:00", "%Y-%m-%d %H:%M:%S%z")
                .unwrap()
                .with_timezone(&Utc),
            bearing_accuracy: 0.1,
            speed_accuracy: 0.1,
            vertical_accuracy: 0.1,
            horizontal_accuracy: 0.1,
            speed: 2.0,
            bearing: 180.0,
            altitude: 200.0,
            longitude: -121.0,
            latitude: 38.0,
            qz: 0.0,
            qy: 0.0,
            qx: 0.0,
            qw: 1.0,
            roll: 0.1,
            pitch: 0.1,
            yaw: 0.1,
            acc_z: 9.81,
            acc_y: 0.01,
            acc_x: -0.01,
            gyro_z: 0.02,
            gyro_y: -0.02,
            gyro_x: 0.02,
            mag_z: 55.0,
            mag_y: -25.0,
            mag_x: -15.0,
            relative_altitude: 10.0,
            pressure: 1012.25,
            grav_z: 9.81,
            grav_y: 0.01,
            grav_x: -0.01,
        });
        // Write to CSV
        TestDataRecord::to_csv(&records, path).expect("Failed to write test data to CSV");
        // Check to make sure the file exists
        assert!(path.exists(), "Test data CSV file should exist");
        // Read back from CSV
        let read_records =
            TestDataRecord::from_csv(path).expect("Failed to read test data from CSV");
        // Check that the read records match the original
        assert_eq!(
            read_records.len(),
            records.len(),
            "Record count should match"
        );
        for (i, record) in read_records.iter().enumerate() {
            assert_eq!(record.time, records[i].time, "Timestamps should match");
            assert!(
                (record.latitude - records[i].latitude).abs() < 1e-6,
                "Latitudes should match"
            );
            assert!(
                (record.longitude - records[i].longitude).abs() < 1e-6,
                "Longitudes should match"
            );
            assert!(
                (record.altitude - records[i].altitude).abs() < 1e-6,
                "Altitudes should match"
            );
            // Add more assertions as needed for other fields
        }
        // Clean up
        let _ = std::fs::remove_file(path);
    }
    #[test]
    fn test_navigation_result_new() {
        let nav = NavigationResult::new();
        //let expected_timestamp = chrono::Utc::now();
        //assert_eq!(nav.timestamp, expected_timestamp);
        assert_eq!(nav.latitude, 0.0);
        assert_eq!(nav.longitude, 0.0);
        assert_eq!(nav.altitude, 0.0);
        assert_eq!(nav.velocity_north, 0.0);
        assert_eq!(nav.velocity_east, 0.0);
        assert_eq!(nav.velocity_down, 0.0);
    }
    #[test]
    fn test_navigation_result_new_from_nav_state() {
        let mut state = StrapdownState::new();
        state.latitude = 1.0;
        state.longitude = 2.0;
        state.altitude = 3.0;
        state.velocity_north = 4.0;
        state.velocity_east = 5.0;
        state.velocity_down = 6.0;
        state.attitude = nalgebra::Rotation3::from_euler_angles(7.0, 8.0, 9.0);
        let timestamp = chrono::Utc::now();
        let nav = NavigationResult::from((
            &timestamp,
            &state,
            &IMUData::new_from_vec(vec![0.0; 3], vec![0.0; 3]),
            &0.0,
            &0.0,
            &0.0,
            &1000.0, // dummy pressure
        ));
        assert_eq!(nav.latitude, (1.0_f64).to_degrees());
        assert_eq!(nav.longitude, (2.0_f64).to_degrees());
        assert_eq!(nav.altitude, 3.0);
        assert_eq!(nav.velocity_north, 4.0);
        assert_eq!(nav.velocity_east, 5.0);
        assert_eq!(nav.velocity_down, 6.0);
    }
    #[test]
    fn test_navigation_result_to_csv_and_from_csv() {
        let mut nav = NavigationResult::new();
        nav.latitude = 1.0;
        nav.longitude = 2.0;
        nav.altitude = 3.0;
        nav.velocity_north = 4.0;
        nav.velocity_east = 5.0;
        nav.velocity_down = 6.0;
        //nav.attitude = nalgebra::Rotation3::from_euler_angles(0.1, 0.2, 0.3);
        let temp_file = std::env::temp_dir().join("test_nav_result.csv");
        NavigationResult::to_csv(&[nav.clone()], &temp_file).unwrap();
        let read = NavigationResult::from_csv(&temp_file).unwrap();
        assert_eq!(read.len(), 1);
        assert_eq!(read[0].latitude, 1.0);
        assert_eq!(read[0].longitude, 2.0);
        assert_eq!(read[0].altitude, 3.0);
        assert_eq!(read[0].velocity_north, 4.0);
        assert_eq!(read[0].velocity_east, 5.0);
        assert_eq!(read[0].velocity_down, 6.0);
        //assert!(read[0].attitude.matrix().abs_diff_eq(
        //    nalgebra::Rotation3::from_euler_angles(0.1, 0.2, 0.3).matrix(),
        //    1e-12
        //));
        let _ = std::fs::remove_file(&temp_file);
    }
    #[test]
    fn test_dead_reckoning_empty_and_single() {
        let empty: Vec<TestDataRecord> = vec![];
        let res = dead_reckoning(&empty);
        assert!(res.is_empty());
        let rec = TestDataRecord::from_csv("./data/test_data.csv")
            .ok()
            .and_then(|v| v.into_iter().next())
            .unwrap_or_else(|| TestDataRecord {
                time: chrono::Utc::now(),
                bearing_accuracy: 0.0,
                speed_accuracy: 0.0,
                vertical_accuracy: 0.0,
                horizontal_accuracy: 0.0,
                speed: 0.0,
                bearing: 0.0,
                altitude: 0.0,
                longitude: 0.0,
                latitude: 0.0,
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
                pressure: 0.0,
                grav_z: 0.0,
                grav_y: 0.0,
                grav_x: 0.0,
            });
        let res = dead_reckoning(&[rec.clone()]);
        assert_eq!(res.len(), 1);
        let mut rec2 = rec.clone();
        rec2.time = chrono::Utc::now();
        let res = dead_reckoning(&[rec.clone(), rec2]);
        assert_eq!(res.len(), 2);
    }
    #[test]
    fn test_closed_loop_minimal() {
        let rec = TestDataRecord::from_csv("./data/test_data.csv")
            .ok()
            .and_then(|v| v.into_iter().next())
            .unwrap_or_else(|| TestDataRecord {
                time: chrono::Utc::now(),
                bearing_accuracy: 0.0,
                speed_accuracy: 0.0,
                vertical_accuracy: 0.0,
                horizontal_accuracy: 0.0,
                speed: 0.0,
                bearing: 0.0,
                altitude: 0.0,
                longitude: 0.0,
                latitude: 0.0,
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
                pressure: 0.0,
                grav_z: 0.0,
                grav_y: 0.0,
                grav_x: 0.0,
            });
        let res = closed_loop(&vec![rec.clone()], Some(1));
        assert!(!res.is_empty());
    }
    #[test]
    fn test_initialize_ukf_default_and_custom() {
        let rec = TestDataRecord {
            time: chrono::Utc::now(),
            bearing_accuracy: 0.0,
            speed_accuracy: 0.0,
            vertical_accuracy: 1.0,
            horizontal_accuracy: 4.0,
            speed: 1.0,
            bearing: 0.0,
            altitude: 10.0,
            longitude: 20.0,
            latitude: 30.0,
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
            pressure: 0.0,
            grav_z: 0.0,
            grav_y: 0.0,
            grav_x: 0.0,
        };
        let ukf = initialize_ukf(rec.clone(), None, None);
        assert!(!ukf.get_mean().is_empty());
        let ukf2 = initialize_ukf(
            rec,
            Some(vec![0.1, 0.2, 0.3]),
            Some(vec![0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
        );
        assert!(!ukf2.get_mean().is_empty());
    }
}
