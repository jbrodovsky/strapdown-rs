//! Simulation utilities and CSV data loading for strapdown inertial navigation.
//!
//! This module provides tools for simulating and evaluting strapdown inertial navigation systems.
//! It is primarily designed to work with data produced from the [Sensor Logger](https://www.tszheichoi.com/sensorlogger) 
//! app, as such it makes assumptions about the data format and structure that that corresponds to 
//! how that app records data. That data is typically stored in CSV format and is represented by the
//! `TestDataRecord` struct. This struct is fairly comprehensive and should be easily reusable for
//! other applications. Modeling off of that struct is the `NavigationResult` struct which is used 
//! to store navigation solutions from simulations, such as dead reckoning or Kalman filtering.
//! 
//! This module also provides basic functionality for analyzing cannonical strapdown inertial navigation
//! systems via the `dead_reckoning` and `closed_loop` functions. The `closed_loop` function in particular
//! can also be used to simulate various types of GNSS-denied scenarios, such as intermittent, degraded,
//! or intermitent and degraded GNSS via the measurement models provided in this module. You can install 
//! the programs that execute this generic simulation by installing the binary via `cargo install strapdown-rs`.
use core::f64;
use std::fmt::Display;
use std::io::{self};
use std::path::Path;

use chrono::{DateTime, Utc};
use nalgebra::{DMatrix, DVector, Vector3};
use serde::{Deserialize, Serialize};

use crate::earth;
use crate::earth::METERS_TO_DEGREES;
use crate::filter::{GPSPositionAndVelocityMeasurement, RelativeAltitudeMeasurement, StrapdownParams, UKF};
use crate::{IMUData, StrapdownState, forward};
/// Struct representing a single row of test data from the CSV file.
///
/// Fields correspond to columns in the CSV, with appropriate renaming for Rust style.
/// This struct is setup to capture the data recorded from the [Sensor Logger](https://www.tszheichoi.com/sensorlogger) app.
/// Primarily, this represents IMU data as (relative to the device) and GPS data.
#[derive(Debug, Default, Deserialize, Serialize, Clone)]
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

// ==== Helper structus for navigation simulations ====
/// Struct representing the covariance diagonal of a navigation solution in NED coordinates.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct NEDCovariance {
    pub latitude_cov: f64,
    pub longitude_cov: f64,
    pub altitude_cov: f64,
    pub velocity_n_cov: f64,
    pub velocity_e_cov: f64,
    pub velocity_d_cov: f64,
    pub roll_cov: f64,
    pub pitch_cov: f64,
    pub yaw_cov: f64,
    pub acc_bias_x_cov: f64,
    pub acc_bias_y_cov: f64,
    pub acc_bias_z_cov: f64,
    pub gyro_bias_x_cov: f64,
    pub gyro_bias_y_cov: f64,
    pub gyro_bias_z_cov: f64,
}

/// Generic result struct for navigation simulations.
///
/// This structure contains a single row of position, velocity, and attitude vectors
/// representing the navigation solution at a specific timestamp, along with the covariance diagonal,
/// input IMU measurements, and derived geophysical values.
/// 
/// It can be used across different types of navigation simulations such as dead reckoning,
/// Kalman filtering, or any other navigation algorithm.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NavigationResult {
    /// Timestamp corresponding to the state
    pub timestamp: DateTime<Utc>,
    // ---- Navigation solution states ----
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
    // ---- Covariance values for the navigation solution ----
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
    /// Free-air gravity anomaly in Gal (note that most maps will be in mGal)
    pub freeair: f64,
    /// Magnetic field strength anomaly in nT
    pub mag_anomaly: f64,
}
impl Default for NavigationResult {
    fn default() -> Self {
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
            mag_x: earth::MAGNETIC_FIELD_STRENGTH,  // default magnetic field strength
            mag_y: 0.0,                             // default values
            mag_z: 0.0,                             // default values
            pressure: 1013.25,                      // standard atmospheric pressure in millibars
            freeair: 0.0,                           // free-air gravity anomaly in mGal
            mag_anomaly: 0.0,                       // default magnetic anomaly in nT
        }
    }
}
impl NavigationResult {
    /// Creates a new NavigationResult with default values.
    pub fn new() -> Self {
        // TODO: #70 Re-implement NavigationResult construcutor with validation
        NavigationResult::default() // add in validation
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
/// Convert DVectors containing the navigation state mean and covariance into a NavigationResult
/// struct.
/// 
/// This implementation is useful for converting the output of a Kalman filter or UKF into a
/// NavigationResult, which can then be used for further processing or analysis.
/// 
/// # Arguments
/// - `timestamp`: The timestamp of the navigation solution.
/// - `state`: A DVector containing the navigation state mean.
/// - `covariance`: A DMatrix containing the covariance of the state.
/// - `imu_data`: An IMUData struct containing the IMU measurements.
/// - `mag_x`, `mag_y`, `mag_z`: Magnetic field strength in microteslas.
/// - `pressure`: Pressure in millibars.
/// - `freeair`: Free-air gravity anomaly in mGal.
/// 
/// # Returns
/// A NavigationResult struct containing the navigation solution.
impl From<(&DateTime<Utc>, &DVector<f64>, &DMatrix<f64>, &IMUData, &Vector3<f64>, &f64)> for NavigationResult {
    fn from (
        (timestamp, state, covariance, imu_data, magnetic_vector, pressure): (
            &DateTime<Utc>,
            &DVector<f64>,
            &DMatrix<f64>,
            &IMUData,
            &Vector3<f64>,
            &f64,
        ),
    ) -> Self {
        assert!(state.len() == 15, "State vector must have 15 elements; got {}", state.len());
        assert!(covariance.nrows() == 15 && covariance.ncols() == 15, "Covariance matrix must be 15x15");
        let covariance = DVector::from_vec(covariance.diagonal().iter().map(|&x| x).collect());
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
            latitude_cov: covariance[0],
            longitude_cov: covariance[1],
            altitude_cov: covariance[2],
            velocity_n_cov: covariance[3],
            velocity_e_cov: covariance[4],
            velocity_d_cov: covariance[5],
            roll_cov: covariance[6],
            pitch_cov: covariance[7],
            yaw_cov: covariance[8],
            acc_bias_x_cov: covariance[9],
            acc_bias_y_cov: covariance[10],
            acc_bias_z_cov: covariance[11],
            gyro_bias_x_cov: covariance[12],
            gyro_bias_y_cov: covariance[13],
            gyro_bias_z_cov: covariance[14],
            acc_x: imu_data.accel[0],
            acc_y: imu_data.accel[1],
            acc_z: imu_data.accel[2],
            gyro_x: imu_data.gyro[0],
            gyro_y: imu_data.gyro[1],
            gyro_z: imu_data.gyro[2],
            mag_x: magnetic_vector.x,
            mag_y: magnetic_vector.y,
            mag_z: magnetic_vector.z,
            pressure: *pressure,
            freeair: earth::gravity_anomaly(
                &state[0],
                &state[2], // altitude
                &state[3], // velocity_north
                &state[4], // velocity_east
                &(imu_data.accel[0].powi(2) + imu_data.accel[1].powi(2) + imu_data.accel[2].powi(2)).sqrt(),
            ),
            mag_anomaly: earth::magnetic_anomaly(
                state[0].to_radians(),
                state[1].to_radians(),
                state[2], // altitude
                magnetic_vector.x,
                magnetic_vector.y,
                magnetic_vector.z,
            ),
        }
    }
}
/// Convert NED UKF to NavigationResult.
/// 
/// This implementation is useful for converting the output of a UKF into a
/// NavigationResult, which can then be used for further processing or analysis.
/// 
/// # Arguments
/// - `timestamp`: The timestamp of the navigation solution.
/// - `ukf`: A reference to the UKF instance containing the navigation state mean and covariance.
/// - `imu_data`: An IMUData struct containing the IMU measurements.
/// - `magnetic_vector`: Magnetic field strength measurement in microteslas (body frame x, y, z).
/// - `pressure`: Pressure in millibars.
/// 
/// # Returns
/// A NavigationResult struct containing the navigation solution.
impl From<(&DateTime<Utc>, &UKF, &IMUData, &Vector3<f64>, &f64)> for NavigationResult {
    fn from(
        (timestamp, ukf, imu_data, magnetic_vector, pressure): (
            &DateTime<Utc>,
            &UKF,
            &IMUData,
            &Vector3<f64>,
            &f64,
        ),
    ) -> Self {
        let state = &ukf.get_mean();
        let covariance = ukf.get_covariance();
        NavigationResult {
            timestamp: timestamp.clone(),
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
            mag_x: magnetic_vector.x,
            mag_y: magnetic_vector.y,
            mag_z: magnetic_vector.z,
            pressure: *pressure,
            freeair: earth::gravity_anomaly(
                &state[0],
                &state[2], // altitude
                &state[3], // velocity_north
                &state[4], // velocity_east
                &(imu_data.accel[0].powi(2) + imu_data.accel[1].powi(2) + imu_data.accel[2].powi(2)).sqrt(),
            ),
            mag_anomaly: earth::magnetic_anomaly(
                state[0].to_radians(),
                state[1].to_radians(),
                state[2], // altitude   
                magnetic_vector.x,
                magnetic_vector.y,
                magnetic_vector.z,
            ),  
        }
    }
}
/// Convert StrapdownState to NavigationResult.
/// 
/// This implementation is useful for converting the output of a StrapdownState into a
/// NavigationResult, which can then be used for further processing or analysis.
/// 
/// # Arguments
/// - `timestamp`: The timestamp of the navigation solution.
/// - `state`: A reference to the StrapdownState instance containing the navigation state.
/// - `imu_data`: An IMUData struct containing the IMU measurements.
/// - `magnetic_vector`: Magnetic field strength measurement in microteslas (body frame x, y, z).
/// - `pressure`: Pressure in millibars.
/// # Returns
/// A NavigationResult struct containing the navigation solution.
impl From<(&DateTime<Utc>, &StrapdownState, &IMUData, &Vector3<f64>, &f64)> for NavigationResult {
    fn from(
        (timestamp, state, imu_data, magnetic_vector, pressure): (
            &DateTime<Utc>,
            &StrapdownState,
            &IMUData,
            &Vector3<f64>,
            &f64,
        ),
    ) -> Self {
        NavigationResult {
            timestamp: timestamp.clone(),
            latitude: state.latitude.to_degrees(),
            longitude: state.longitude.to_degrees(),
            altitude: state.altitude,
            velocity_north: state.velocity_north,
            velocity_east: state.velocity_east,
            velocity_down: state.velocity_down,
            roll: state.attitude.euler_angles().0,
            pitch: state.attitude.euler_angles().1,
            yaw: state.attitude.euler_angles().2,
            acc_bias_x: 0.0, // StrapdownState does not store biases
            acc_bias_y: 0.0,
            acc_bias_z: 0.0,
            gyro_bias_x: 0.0,
            gyro_bias_y: 0.0,
            gyro_bias_z: 0.0,
            latitude_cov: f64::NAN, // default covariance values
            longitude_cov: f64::NAN,
            altitude_cov: f64::NAN,
            velocity_n_cov: f64::NAN,
            velocity_e_cov: f64::NAN,
            velocity_d_cov: f64::NAN,
            roll_cov: f64::NAN,
            pitch_cov: f64::NAN,
            yaw_cov: f64::NAN,
            acc_bias_x_cov: f64::NAN, 
            acc_bias_y_cov: f64::NAN,
            acc_bias_z_cov: f64::NAN,
            gyro_bias_x_cov: f64::NAN,
            gyro_bias_y_cov: f64::NAN,
            gyro_bias_z_cov: f64::NAN,
            acc_x: imu_data.accel[0],
            acc_y: imu_data.accel[1],
            acc_z: imu_data.accel[2],
            gyro_x: imu_data.gyro[0],
            gyro_y: imu_data.gyro[1],
            gyro_z: imu_data.gyro[2],
            mag_x: magnetic_vector.x,
            mag_y: magnetic_vector.y,
            mag_z: magnetic_vector.z,
            pressure: *pressure,
            freeair: earth::gravity_anomaly(
                &state.latitude,
                &state.altitude,
                &state.velocity_north,
                &state.velocity_east,
                &(imu_data.accel[0].powi(2) + imu_data.accel[1].powi(2) + imu_data.accel[2].powi(2)).sqrt(),
            ),
            mag_anomaly: earth::magnetic_anomaly(
                state.latitude.to_radians(),
                state.longitude.to_radians(),
                state.altitude,
                magnetic_vector.x,
                magnetic_vector.y,
                magnetic_vector.z,
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
/// these values would be used as a baseline for comparison. Keep in mind that this toolbox
/// is designed for the local level frame of reference and the forward mechanization is typically
/// only valid at lower latitude (e.g. < 60 degrees) and at low altitudes (e.g. < 1000m). With 
/// that, remember that dead reckoning is subject to drift and errors accumulate over time relative
/// to the quality of the IMU data. Poor quality IMU data (e.g. MEMS grade IMUs) will lead to 
/// significant drift very quickly which may cause this function to produce unrealistic results,
/// hang, or crash.
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
        &first_record.time,
        &state.into(),
        &DMatrix::from_diagonal(&DVector::from_element(15, 0.0)),
        &IMUData {
            accel: Vector3::new(first_record.acc_x, first_record.acc_y, first_record.acc_z),
            gyro: Vector3::new(first_record.gyro_x, first_record.gyro_y, first_record.gyro_z),
        },
        &Vector3::new( first_record.mag_x, first_record.mag_y, first_record.mag_z),
        &first_record.pressure,
    )
    ));
    let mut previous_time = records[0].time;
    // Process each subsequent record
    for record in records.iter().skip(1) {
        // Try to calculate time difference from timestamps, default to 1 second if parsing fails
        let current_time = record.time;
        let dt = (current_time - previous_time).as_seconds_f64();
        // Create IMU data from the record
        let imu_data = IMUData {
            accel: Vector3::new(record.acc_x, record.acc_y, record.acc_z),
            gyro: Vector3::new(record.gyro_x, record.gyro_y, record.gyro_z),
        };
        forward(&mut state, imu_data, dt);
        results.push(NavigationResult::from((
            &current_time,
            &state.into(),
            &DMatrix::from_diagonal(&DVector::from_element(15, 0.0)),
            &imu_data,
            &Vector3::new(
                record.mag_x,
                record.mag_y,
                record.mag_z,
            ),
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
    gps_interval: Option<usize>
) -> Vec<NavigationResult> {
    let gps_interval = gps_interval.unwrap_or(1); // Default to every record if not specified
    let reference_altitude = records[0].altitude; // Use the first record's pressure as reference
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
        &IMUData {
            accel: Vector3::new(records[0].acc_x, records[0].acc_y, records[0].acc_z),
            gyro: Vector3::new(records[0].gyro_x, records[0].gyro_y, records[0].gyro_z),
        },
        &Vector3::new(
            records[0].mag_x,
            records[0].mag_y,
            records[0].mag_z,
        ),
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
        let imu_data = IMUData {
            accel: Vector3::new(
                record.acc_x - mean[9], // subtract accel bias
                record.acc_y - mean[10],
                record.acc_z - mean[11],
            ),
            gyro: Vector3::new(
                record.gyro_x - mean[12], // subtract gyro bias
                record.gyro_y - mean[13],
                record.gyro_z - mean[14],
            ),
        };
        // Update the UKF with the IMU data
        ukf.predict(imu_data, dt);
        // ---- Perform various measurement updates based on the available data ----
        // If GPS data is available, update the UKF with the GPS position and speed measurement
        if !record.latitude.is_nan()
            && !record.longitude.is_nan()
            && !record.altitude.is_nan()
            && !record.bearing.is_nan()
            && !record.speed.is_nan()
            && i % gps_interval == 0
        {
            let measurement = GPSPositionAndVelocityMeasurement {
                latitude: record.latitude,
                longitude: record.longitude,
                altitude: record.altitude,
                northward_velocity: record.speed * record.bearing.cos(),
                eastward_velocity: record.speed * record.bearing.sin(),
                horizontal_noise_std: record.horizontal_accuracy.sqrt(),
                vertical_noise_std: record.vertical_accuracy,
                velocity_noise_std: record.speed_accuracy,
            };
            ukf.update(measurement);
        }
        // If barometric altimeter data is available, update the UKF with the altitude measurement
        if !record.pressure.is_nan() {
            let altitude = RelativeAltitudeMeasurement {
                relative_altitude: record.relative_altitude,
                reference_altitude: reference_altitude,
            };
            ukf.update(altitude);
        }
        // Store the current state and covariance in results
        results.push(NavigationResult::from((
            &current_timestamp,
            &ukf,
            &imu_data,
            &Vector3::new(
                record.mag_x,
                record.mag_y,
                record.mag_z,
            ),
            &record.pressure,
        )));
        i += 1;
        previous_timestamp = current_timestamp;
    }
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
    //let mut process_noise_diagonal = vec![1e-9; 9]; // adds a minor amount of noise to base states
    //process_noise_diagonal.extend(vec![1e-9; 6]); // Process noise for imu biases
    let process_noise_diagonal = DVector::from_vec(
        vec![
            1e-6, // position noise
            1e-6, // position noise
            1e-6, // altitude noise
            1e-3, // velocity north noise
            1e-3, // velocity east noise
            1e-3, // velocity down noise
            1e-5, // roll noise
            1e-5, // pitch noise
            1e-5, // yaw noise
            1e-6, // acc bias x noise
            1e-6, // acc bias y noise
            1e-6, // acc bias z noise
            1e-8, // gyro bias x noise
            1e-8, // gyro bias y noise
            1e-8, // gyro bias z noise
        ],
    );
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
        let nav = NavigationResult::default();
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
    fn test_navigation_result_from_strapdown_state() {
        let mut state = StrapdownState::default();
        state.latitude = 1.0;
        state.longitude = 2.0;
        state.altitude = 3.0;
        state.velocity_north = 4.0;
        state.velocity_east = 5.0;
        state.velocity_down = 6.0;
        state.attitude = nalgebra::Rotation3::from_euler_angles(7.0, 8.0, 9.0);

        let state_vector: DVector<f64> = DVector::from_vec(vec![
            state.latitude,
            state.longitude,
            state.altitude,
            state.velocity_north,
            state.velocity_east,
            state.velocity_down,
            state.attitude.euler_angles().0, // roll
            state.attitude.euler_angles().1, // pitch
            state.attitude.euler_angles().2, // yaw
            0.0, // acc_bias_x
            0.0, // acc_bias_y
            0.0, // acc_bias_z
            0.0, // gyro_bias_x
            0.0, // gyro_bias_y
            0.0, // gyro_bias_z
        ]);
        let timestamp = chrono::Utc::now();
        let nav = NavigationResult::from((
            &timestamp,
            &state_vector.into(),
            &DMatrix::from_diagonal(&DVector::from_element(15, 0.0)), // dummy covariance
            &IMUData {
                accel: Vector3::new(0.0, 0.0, 0.0), // dummy IMU data
                gyro: Vector3::new(0.0, 0.0, 0.0),
            },
            &Vector3::new(0.0, 0.0, 0.0), // dummy magnetic vector
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
    // #[test]
    // fn test_dead_reckoning_empty_and_single() {
    //     let empty: Vec<TestDataRecord> = vec![];
    //     let res = dead_reckoning(&empty);
    //     assert!(res.is_empty());
    //     let rec = TestDataRecord::from_csv("./data/test_data.csv")
    //         .ok()
    //         .and_then(|v| v.into_iter().next())
    //         .unwrap_or_else(|| TestDataRecord {
    //             time: chrono::Utc::now(),
    //             bearing_accuracy: 0.0,
    //             speed_accuracy: 0.0,
    //             vertical_accuracy: 0.0,
    //             horizontal_accuracy: 0.0,
    //             speed: 0.0,
    //             bearing: 0.0,
    //             altitude: 0.0,
    //             longitude: 0.0,
    //             latitude: 0.0,
    //             qz: 0.0,
    //             qy: 0.0,
    //             qx: 0.0,
    //             qw: 1.0,
    //             roll: 0.0,
    //             pitch: 0.0,
    //             yaw: 0.0,
    //             acc_z: 0.0,
    //             acc_y: 0.0,
    //             acc_x: 0.0,
    //             gyro_z: 0.0,
    //             gyro_y: 0.0,
    //             gyro_x: 0.0,
    //             mag_z: 0.0,
    //             mag_y: 0.0,
    //             mag_x: 0.0,
    //             relative_altitude: 0.0,
    //             pressure: 0.0,
    //             grav_z: 0.0,
    //             grav_y: 0.0,
    //             grav_x: 0.0,
    //         });
    //     let res = dead_reckoning(&[rec.clone()]);
    //     assert_eq!(res.len(), 1);
    //     let mut rec2 = rec.clone();
    //     rec2.time = chrono::Utc::now();
    //     let res = dead_reckoning(&[rec.clone(), rec2]);
    //     assert_eq!(res.len(), 2);
    // }
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
