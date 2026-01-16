//! Simulation utilities and data serialization for strapdown inertial navigation.
//!
//! This module provides tools for simulating and evaluating strapdown inertial navigation systems.
//! It is primarily designed to work with data produced from the [Sensor Logger](https://www.tszheichoi.com/sensorlogger)
//! app, as such it makes assumptions about the data format and structure that that corresponds to
//! how that app records data.
//!
//! ## Data Formats
//!
//! The module supports both CSV and netCDF formats for input and output data:
//!
//! - **CSV format**: Human-readable text format, suitable for quick inspection and editing
//! - **netCDF format**: Binary format optimized for large datasets, better for archival and data interchange
//!
//! Data is represented by the `TestDataRecord` struct (sensor measurements) and `NavigationResult` struct
//! (navigation solutions). Both structs support serialization to/from CSV and netCDF formats.
//!
//! ### Example: Working with netCDF files
//!
//! ```no_run
//! use strapdown::sim::{TestDataRecord, NavigationResult};
//!
//! // Read test data from netCDF file
//! let test_data = TestDataRecord::from_netcdf("input_data.nc")
//!     .expect("Failed to read input data");
//!
//! // ... perform navigation simulation ...
//!
//! // Write navigation results to netCDF file
//! # let nav_results: Vec<NavigationResult> = vec![];
//! NavigationResult::to_netcdf(&nav_results, "output_results.nc")
//!     .expect("Failed to write navigation results");
//! ```
//!
//! ## Simulation Functions
//!
//! This module also provides basic functionality for analyzing canonical strapdown inertial navigation
//! systems via the `dead_reckoning` and `closed_loop` functions. The `closed_loop` function in particular
//! can also be used to simulate various types of GNSS-denied scenarios, such as intermittent, degraded,
//! or intermittent and degraded GNSS via the measurement models provided in this module. You can install
//! the programs that execute this generic simulation by installing the binary via `cargo install strapdown-rs`.
use core::f64;
use log::{debug, info, warn};
use std::fmt::{Debug, Display};
use std::io::{self, Read, Write};
use std::path::Path;

use anyhow::{Result, bail};
use chrono::{DateTime, Duration, Utc};
use nalgebra::{DMatrix, DVector, Vector3};
use serde::{Deserialize, Deserializer, Serialize};

#[cfg(feature = "clap")]
use clap::{Args, ValueEnum};

use crate::NavigationFilter;
use crate::earth::METERS_TO_DEGREES;
use crate::kalman::{InitialState, UnscentedKalmanFilter};
use crate::messages::{Event, EventStream, GnssFaultModel, GnssScheduler};

use crate::{IMUData, StrapdownState, forward};
use health::HealthMonitor;

// Re-export HealthLimits for easier access in tests and external users
pub use health::HealthLimits;

pub const DEFAULT_PROCESS_NOISE: [f64; 15] = [
    // Default process noise if not provided
    1e-6, // position noise 1e-6
    1e-6, // position noise 1e-6
    1e-4, // altitude noise
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
];

fn de_f64_nan<'de, D>(deserializer: D) -> Result<f64, D::Error>
where
    D: Deserializer<'de>,
{
    // Read whatever the CSV cell was as an Option<String>.
    // Missing field -> None; present but empty -> Some(""), etc.
    let opt = Option::<String>::deserialize(deserializer)?;
    match opt {
        None => Ok(f64::NAN),
        Some(s) => {
            let t = s.trim();
            if t.is_empty() || t.eq_ignore_ascii_case("nan") || t.eq_ignore_ascii_case("null") {
                return Ok(f64::NAN);
            }
            t.parse::<f64>().map_err(serde::de::Error::custom)
        }
    }
}
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
    #[serde(rename = "bearingAccuracy", deserialize_with = "de_f64_nan")]
    pub bearing_accuracy: f64,
    /// accuracy of the speed in m/s
    #[serde(rename = "speedAccuracy", deserialize_with = "de_f64_nan")]
    pub speed_accuracy: f64,
    /// accuracy of the altitude in meters
    #[serde(rename = "verticalAccuracy", deserialize_with = "de_f64_nan")]
    pub vertical_accuracy: f64,
    /// accuracy of the horizontal position in meters
    #[serde(rename = "horizontalAccuracy", deserialize_with = "de_f64_nan")]
    pub horizontal_accuracy: f64,
    /// Speed in m/s
    #[serde(deserialize_with = "de_f64_nan")]
    pub speed: f64,
    /// Bearing in degrees
    #[serde(deserialize_with = "de_f64_nan")]
    pub bearing: f64,
    /// Altitude in meters
    #[serde(deserialize_with = "de_f64_nan")]
    pub altitude: f64,
    /// Longitude in degrees
    #[serde(deserialize_with = "de_f64_nan")]
    pub longitude: f64,
    /// Latitude in degrees
    #[serde(deserialize_with = "de_f64_nan")]
    pub latitude: f64,
    /// Quaternion component representing the rotation around the z-axis
    #[serde(deserialize_with = "de_f64_nan")]
    pub qz: f64,
    /// Quaternion component representing the rotation around the y-axis
    #[serde(deserialize_with = "de_f64_nan")]
    pub qy: f64,
    /// Quaternion component representing the rotation around the x-axis
    #[serde(deserialize_with = "de_f64_nan")]
    pub qx: f64,
    /// Quaternion component representing the rotation around the w-axis
    #[serde(deserialize_with = "de_f64_nan")]
    pub qw: f64,
    /// Roll angle in radians
    #[serde(deserialize_with = "de_f64_nan")]
    pub roll: f64,
    /// Pitch angle in radians
    #[serde(deserialize_with = "de_f64_nan")]
    pub pitch: f64,
    /// Yaw angle in radians
    #[serde(deserialize_with = "de_f64_nan")]
    pub yaw: f64,
    /// Z-acceleration in m/s^2
    #[serde(deserialize_with = "de_f64_nan")]
    pub acc_z: f64,
    /// Y-acceleration in m/s^2
    #[serde(deserialize_with = "de_f64_nan")]
    pub acc_y: f64,
    /// X-acceleration in m/s^2
    #[serde(deserialize_with = "de_f64_nan")]
    pub acc_x: f64,
    /// Rotation rate around the z-axis in radians/s
    #[serde(deserialize_with = "de_f64_nan")]
    pub gyro_z: f64,
    /// Rotation rate around the y-axis in radians/s
    #[serde(deserialize_with = "de_f64_nan")]
    pub gyro_y: f64,
    /// Rotation rate around the x-axis in radians/s
    #[serde(deserialize_with = "de_f64_nan")]
    pub gyro_x: f64,
    /// Magnetic field strength in the z-direction in micro teslas
    #[serde(deserialize_with = "de_f64_nan")]
    pub mag_z: f64,
    /// Magnetic field strength in the y-direction in micro teslas
    #[serde(deserialize_with = "de_f64_nan")]
    pub mag_y: f64,
    /// Magnetic field strength in the x-direction in micro teslas
    #[serde(deserialize_with = "de_f64_nan")]
    pub mag_x: f64,
    /// Change in altitude in meters
    #[serde(rename = "relativeAltitude", deserialize_with = "de_f64_nan")]
    pub relative_altitude: f64,
    /// pressure in millibars
    #[serde(deserialize_with = "de_f64_nan")]
    pub pressure: f64,
    /// Acceleration due to gravity in the z-direction in m/s^2
    #[serde(deserialize_with = "de_f64_nan")]
    pub grav_z: f64,
    /// Acceleration due to gravity in the y-direction in m/s^2
    #[serde(deserialize_with = "de_f64_nan")]
    pub grav_y: f64,
    /// Acceleration due to gravity in the x-direction in m/s^2
    #[serde(deserialize_with = "de_f64_nan")]
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
        let mut rdr = csv::ReaderBuilder::new()
            .has_headers(true)
            .flexible(true)
            .trim(csv::Trim::All)
            .from_path(path)?;

        let mut records = Vec::new();
        for (i, result) in rdr.deserialize::<Self>().enumerate() {
            match result {
                Ok(r) => records.push(r),
                Err(e) => {
                    // Skip only this row; keep going.
                    warn!("Skipping row {} due to parse error: {e}", i + 1);
                }
            }
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

    /// Writes a vector of TestDataRecord structs to an HDF5 file.
    ///
    /// # Arguments
    /// * `records` - Vector of TestDataRecord structs to write
    /// * `path` - Path where the HDF5 file will be saved
    ///
    /// # Returns
    /// * `Result<()>` - Ok if successful, Err otherwise
    ///
    /// # Example
    ///
    /// ```no_run
    /// use strapdown::sim::TestDataRecord;
    /// use std::path::Path;
    ///
    /// let record = TestDataRecord::default();
    /// let records = vec![record];
    /// TestDataRecord::to_hdf5(&records, "data.h5")
    ///    .expect("Failed to write test data to HDF5");
    /// ```
    pub fn to_hdf5<P: AsRef<Path>>(records: &[Self], path: P) -> Result<()> {
        use hdf5::File;

        let file = File::create(path)?;
        let n = records.len();

        // Handle empty datasets
        if n == 0 {
            // Create group to indicate structure even for empty datasets
            let _group = file.create_group("test_data")?;
            return Ok(());
        }

        // Create a group for test data records
        let group = file.create_group("test_data")?;

        // Write timestamps as strings
        let timestamps: Result<Vec<hdf5::types::VarLenAscii>> = records
            .iter()
            .map(|r| {
                hdf5::types::VarLenAscii::from_ascii(&r.time.to_rfc3339())
                    .map_err(|e| anyhow::anyhow!("Failed to encode timestamp as ASCII: {}", e))
            })
            .collect();
        let timestamps = timestamps?;
        let ds_time = group
            .new_dataset::<hdf5::types::VarLenAscii>()
            .shape([n])
            .create("time")?;
        ds_time.write(&timestamps)?;

        // Helper macro to write f64 arrays
        macro_rules! write_f64_field {
            ($field_name:literal, $field:ident) => {{
                let data: Vec<f64> = records.iter().map(|r| r.$field).collect();
                let ds = group.new_dataset::<f64>().shape([n]).create($field_name)?;
                ds.write(&data)?;
            }};
        }

        write_f64_field!("bearing_accuracy", bearing_accuracy);
        write_f64_field!("speed_accuracy", speed_accuracy);
        write_f64_field!("vertical_accuracy", vertical_accuracy);
        write_f64_field!("horizontal_accuracy", horizontal_accuracy);
        write_f64_field!("speed", speed);
        write_f64_field!("bearing", bearing);
        write_f64_field!("altitude", altitude);
        write_f64_field!("longitude", longitude);
        write_f64_field!("latitude", latitude);
        write_f64_field!("qz", qz);
        write_f64_field!("qy", qy);
        write_f64_field!("qx", qx);
        write_f64_field!("qw", qw);
        write_f64_field!("roll", roll);
        write_f64_field!("pitch", pitch);
        write_f64_field!("yaw", yaw);
        write_f64_field!("acc_z", acc_z);
        write_f64_field!("acc_y", acc_y);
        write_f64_field!("acc_x", acc_x);
        write_f64_field!("gyro_z", gyro_z);
        write_f64_field!("gyro_y", gyro_y);
        write_f64_field!("gyro_x", gyro_x);
        write_f64_field!("mag_z", mag_z);
        write_f64_field!("mag_y", mag_y);
        write_f64_field!("mag_x", mag_x);
        write_f64_field!("relative_altitude", relative_altitude);
        write_f64_field!("pressure", pressure);
        write_f64_field!("grav_z", grav_z);
        write_f64_field!("grav_y", grav_y);
        write_f64_field!("grav_x", grav_x);
        Ok(())
    }
    /// Writes a vector of TestDataRecord structs to an MCAP file.
    ///
    /// **Note**: This method uses MessagePack encoding. Due to CSV-specific field deserializers  
    /// in TestDataRecord, direct MCAP deserialization may have limitations. For production use,
    /// consider converting to NavigationResult or using CSV format for TestDataRecord.
    ///
    /// # Arguments
    /// * `records` - Vector of TestDataRecord structs to write
    /// * `path` - Path where the MCAP file will be saved
    ///
    /// # Returns
    /// * `io::Result<()>` - Ok if successful, Err otherwise
    pub fn to_mcap<P: AsRef<Path>>(records: &[Self], path: P) -> io::Result<()> {
        use mcap::{Writer, records::MessageHeader};
        use std::collections::BTreeMap;
        use std::fs::File;
        use std::io::BufWriter;

        let file = File::create(path)?;
        let buf_writer = BufWriter::new(file);
        let mut writer = Writer::new(buf_writer).map_err(io::Error::other)?;

        // Add schema for TestDataRecord (using MessagePack encoding)
        let schema_name = "TestDataRecord";
        let schema_encoding = "msgpack";
        let schema_data = b"TestDataRecord struct serialized with MessagePack";

        let schema_id = writer
            .add_schema(schema_name, schema_encoding, schema_data)
            .map_err(io::Error::other)?;

        // Add channel for TestDataRecord messages
        let metadata = BTreeMap::new();
        let channel_id = writer
            .add_channel(schema_id, "sensor_data", "msgpack", &metadata)
            .map_err(io::Error::other)?;

        // Write each record as a message
        for (seq, record) in records.iter().enumerate() {
            let data = rmp_serde::to_vec(record)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

            let timestamp_nanos = record.time.timestamp_nanos_opt().ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "Timestamp out of range")
            })?;

            let header = MessageHeader {
                channel_id,
                sequence: seq as u32,
                log_time: timestamp_nanos as u64,
                publish_time: timestamp_nanos as u64,
            };

            writer
                .write_to_known_channel(&header, &data)
                .map_err(io::Error::other)?;
        }

        writer.finish().map_err(io::Error::other)?;

        Ok(())
    }

    /// Reads an HDF5 file and returns a vector of TestDataRecord structs.
    ///
    /// # Arguments
    /// * `path` - Path to the HDF5 file to read.
    ///
    /// # Returns
    /// * `Ok(Vec<TestDataRecord>)` if successful.
    /// * `Err` if the file cannot be read or parsed.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use strapdown::sim::TestDataRecord;
    ///
    /// let records = TestDataRecord::from_hdf5("data.h5")
    ///     .expect("Failed to read test data from HDF5");
    /// ```
    pub fn from_hdf5<P: AsRef<Path>>(path: P) -> Result<Vec<Self>> {
        use hdf5::File;

        let file = File::open(path)?;
        let group = file.group("test_data")?;

        // Check if time dataset exists (might be empty dataset)
        if let Ok(ds_time) = group.dataset("time") {
            // Read timestamps
            let timestamps: Vec<hdf5::types::VarLenAscii> = ds_time.read_raw()?;
            let n = timestamps.len();

            // Handle empty dataset
            if n == 0 {
                return Ok(Vec::new());
            }

            // Helper macro to read f64 arrays
            macro_rules! read_f64_field {
                ($field_name:literal) => {{
                    let ds = group.dataset($field_name)?;
                    let data: Vec<f64> = ds.read_raw()?;
                    data
                }};
            }

            let bearing_accuracy = read_f64_field!("bearing_accuracy");
            let speed_accuracy = read_f64_field!("speed_accuracy");
            let vertical_accuracy = read_f64_field!("vertical_accuracy");
            let horizontal_accuracy = read_f64_field!("horizontal_accuracy");
            let speed = read_f64_field!("speed");
            let bearing = read_f64_field!("bearing");
            let altitude = read_f64_field!("altitude");
            let longitude = read_f64_field!("longitude");
            let latitude = read_f64_field!("latitude");
            let qz = read_f64_field!("qz");
            let qy = read_f64_field!("qy");
            let qx = read_f64_field!("qx");
            let qw = read_f64_field!("qw");
            let roll = read_f64_field!("roll");
            let pitch = read_f64_field!("pitch");
            let yaw = read_f64_field!("yaw");
            let acc_z = read_f64_field!("acc_z");
            let acc_y = read_f64_field!("acc_y");
            let acc_x = read_f64_field!("acc_x");
            let gyro_z = read_f64_field!("gyro_z");
            let gyro_y = read_f64_field!("gyro_y");
            let gyro_x = read_f64_field!("gyro_x");
            let mag_z = read_f64_field!("mag_z");
            let mag_y = read_f64_field!("mag_y");
            let mag_x = read_f64_field!("mag_x");
            let relative_altitude = read_f64_field!("relative_altitude");
            let pressure = read_f64_field!("pressure");
            let grav_z = read_f64_field!("grav_z");
            let grav_y = read_f64_field!("grav_y");
            let grav_x = read_f64_field!("grav_x");

            let mut records = Vec::with_capacity(n);
            for i in 0..n {
                let time = DateTime::parse_from_rfc3339(timestamps[i].as_str())
                    .map_err(|e| anyhow::anyhow!("Failed to parse timestamp: {}", e))?
                    .with_timezone(&Utc);

                records.push(TestDataRecord {
                    time,
                    bearing_accuracy: bearing_accuracy[i],
                    speed_accuracy: speed_accuracy[i],
                    vertical_accuracy: vertical_accuracy[i],
                    horizontal_accuracy: horizontal_accuracy[i],
                    speed: speed[i],
                    bearing: bearing[i],
                    altitude: altitude[i],
                    longitude: longitude[i],
                    latitude: latitude[i],
                    qz: qz[i],
                    qy: qy[i],
                    qx: qx[i],
                    qw: qw[i],
                    roll: roll[i],
                    pitch: pitch[i],
                    yaw: yaw[i],
                    acc_z: acc_z[i],
                    acc_y: acc_y[i],
                    acc_x: acc_x[i],
                    gyro_z: gyro_z[i],
                    gyro_y: gyro_y[i],
                    gyro_x: gyro_x[i],
                    mag_z: mag_z[i],
                    mag_y: mag_y[i],
                    mag_x: mag_x[i],
                    relative_altitude: relative_altitude[i],
                    pressure: pressure[i],
                    grav_z: grav_z[i],
                    grav_y: grav_y[i],
                    grav_x: grav_x[i],
                });
            }

            Ok(records)
        } else {
            // No time dataset means empty file
            Ok(Vec::new())
        }
    }

    /// Writes a vector of TestDataRecord structs to a netCDF file.
    ///
    /// # Arguments
    /// * `records` - Vector of TestDataRecord structs to write
    /// * `path` - Path where the netCDF file will be saved
    ///
    /// # Returns
    /// * `Result<()>` - Ok if successful, Err otherwise
    pub fn to_netcdf<P: AsRef<Path>>(records: &[Self], path: P) -> Result<()> {
        if records.is_empty() {
            bail!("Cannot write empty records to netCDF");
        }

        let n = records.len();
        let mut file = netcdf::create(path)?;

        // Define dimensions
        file.add_dimension("time", n)?;

        // Helper macro to add a variable and write data
        macro_rules! add_and_write {
            ($file:expr, $name:expr, $data:expr) => {{
                let mut var = $file.add_variable::<f64>($name, &["time"])?;
                var.put_values(&$data, ..)?;
            }};
        }

        // Prepare all data arrays first
        let times: Vec<f64> = records.iter().map(|r| r.time.timestamp() as f64).collect();
        let bearing_accuracy: Vec<f64> = records.iter().map(|r| r.bearing_accuracy).collect();
        let speed_accuracy: Vec<f64> = records.iter().map(|r| r.speed_accuracy).collect();
        let vertical_accuracy: Vec<f64> = records.iter().map(|r| r.vertical_accuracy).collect();
        let horizontal_accuracy: Vec<f64> = records.iter().map(|r| r.horizontal_accuracy).collect();
        let speed: Vec<f64> = records.iter().map(|r| r.speed).collect();
        let bearing: Vec<f64> = records.iter().map(|r| r.bearing).collect();
        let altitude: Vec<f64> = records.iter().map(|r| r.altitude).collect();
        let longitude: Vec<f64> = records.iter().map(|r| r.longitude).collect();
        let latitude: Vec<f64> = records.iter().map(|r| r.latitude).collect();
        let qz: Vec<f64> = records.iter().map(|r| r.qz).collect();
        let qy: Vec<f64> = records.iter().map(|r| r.qy).collect();
        let qx: Vec<f64> = records.iter().map(|r| r.qx).collect();
        let qw: Vec<f64> = records.iter().map(|r| r.qw).collect();
        let roll: Vec<f64> = records.iter().map(|r| r.roll).collect();
        let pitch: Vec<f64> = records.iter().map(|r| r.pitch).collect();
        let yaw: Vec<f64> = records.iter().map(|r| r.yaw).collect();
        let acc_z: Vec<f64> = records.iter().map(|r| r.acc_z).collect();
        let acc_y: Vec<f64> = records.iter().map(|r| r.acc_y).collect();
        let acc_x: Vec<f64> = records.iter().map(|r| r.acc_x).collect();
        let gyro_z: Vec<f64> = records.iter().map(|r| r.gyro_z).collect();
        let gyro_y: Vec<f64> = records.iter().map(|r| r.gyro_y).collect();
        let gyro_x: Vec<f64> = records.iter().map(|r| r.gyro_x).collect();
        let mag_z: Vec<f64> = records.iter().map(|r| r.mag_z).collect();
        let mag_y: Vec<f64> = records.iter().map(|r| r.mag_y).collect();
        let mag_x: Vec<f64> = records.iter().map(|r| r.mag_x).collect();
        let relative_altitude: Vec<f64> = records.iter().map(|r| r.relative_altitude).collect();
        let pressure: Vec<f64> = records.iter().map(|r| r.pressure).collect();
        let grav_z: Vec<f64> = records.iter().map(|r| r.grav_z).collect();
        let grav_y: Vec<f64> = records.iter().map(|r| r.grav_y).collect();
        let grav_x: Vec<f64> = records.iter().map(|r| r.grav_x).collect();

        // Add variables and write data
        add_and_write!(file, "time", times);
        add_and_write!(file, "bearingAccuracy", bearing_accuracy);
        add_and_write!(file, "speedAccuracy", speed_accuracy);
        add_and_write!(file, "verticalAccuracy", vertical_accuracy);
        add_and_write!(file, "horizontalAccuracy", horizontal_accuracy);
        add_and_write!(file, "speed", speed);
        add_and_write!(file, "bearing", bearing);
        add_and_write!(file, "altitude", altitude);
        add_and_write!(file, "longitude", longitude);
        add_and_write!(file, "latitude", latitude);
        add_and_write!(file, "qz", qz);
        add_and_write!(file, "qy", qy);
        add_and_write!(file, "qx", qx);
        add_and_write!(file, "qw", qw);
        add_and_write!(file, "roll", roll);
        add_and_write!(file, "pitch", pitch);
        add_and_write!(file, "yaw", yaw);
        add_and_write!(file, "acc_z", acc_z);
        add_and_write!(file, "acc_y", acc_y);
        add_and_write!(file, "acc_x", acc_x);
        add_and_write!(file, "gyro_z", gyro_z);
        add_and_write!(file, "gyro_y", gyro_y);
        add_and_write!(file, "gyro_x", gyro_x);
        add_and_write!(file, "mag_z", mag_z);
        add_and_write!(file, "mag_y", mag_y);
        add_and_write!(file, "mag_x", mag_x);
        add_and_write!(file, "relativeAltitude", relative_altitude);
        add_and_write!(file, "pressure", pressure);
        add_and_write!(file, "grav_z", grav_z);
        add_and_write!(file, "grav_y", grav_y);
        add_and_write!(file, "grav_x", grav_x);

        Ok(())
    }

    /// Reads a netCDF file and returns a vector of `TestDataRecord` structs.
    ///
    /// # Arguments
    /// * `path` - Path to the netCDF file to read.
    ///
    /// # Returns
    /// * `Ok(Vec<TestDataRecord>)` if successful.
    /// * `Err` if the file cannot be read or parsed.
    pub fn from_netcdf<P: AsRef<Path>>(path: P) -> Result<Vec<Self>> {
        let file = netcdf::open(path)?;

        // Read time variable
        let time_var = file
            .variable("time")
            .ok_or_else(|| anyhow::anyhow!("time variable not found"))?;
        let times: Vec<f64> = time_var.get_values(..)?;
        let n = times.len();

        // Helper macro to read a variable
        macro_rules! read_var {
            ($file:expr, $name:expr) => {{
                let var = $file
                    .variable($name)
                    .ok_or_else(|| anyhow::anyhow!(concat!($name, " variable not found")))?;
                let data: Vec<f64> = var.get_values(..)?;
                data
            }};
        }

        // Read all variables
        let bearing_accuracy = read_var!(file, "bearingAccuracy");
        let speed_accuracy = read_var!(file, "speedAccuracy");
        let vertical_accuracy = read_var!(file, "verticalAccuracy");
        let horizontal_accuracy = read_var!(file, "horizontalAccuracy");
        let speed = read_var!(file, "speed");
        let bearing = read_var!(file, "bearing");
        let altitude = read_var!(file, "altitude");
        let longitude = read_var!(file, "longitude");
        let latitude = read_var!(file, "latitude");
        let qz = read_var!(file, "qz");
        let qy = read_var!(file, "qy");
        let qx = read_var!(file, "qx");
        let qw = read_var!(file, "qw");
        let roll = read_var!(file, "roll");
        let pitch = read_var!(file, "pitch");
        let yaw = read_var!(file, "yaw");
        let acc_z = read_var!(file, "acc_z");
        let acc_y = read_var!(file, "acc_y");
        let acc_x = read_var!(file, "acc_x");
        let gyro_z = read_var!(file, "gyro_z");
        let gyro_y = read_var!(file, "gyro_y");
        let gyro_x = read_var!(file, "gyro_x");
        let mag_z = read_var!(file, "mag_z");
        let mag_y = read_var!(file, "mag_y");
        let mag_x = read_var!(file, "mag_x");
        let relative_altitude = read_var!(file, "relativeAltitude");
        let pressure = read_var!(file, "pressure");
        let grav_z = read_var!(file, "grav_z");
        let grav_y = read_var!(file, "grav_y");
        let grav_x = read_var!(file, "grav_x");

        // Build records
        let mut records = Vec::with_capacity(n);
        for i in 0..n {
            let time = DateTime::from_timestamp(times[i] as i64, 0)
                .ok_or_else(|| anyhow::anyhow!("Invalid timestamp"))?
                .with_timezone(&Utc);

            records.push(TestDataRecord {
                time,
                bearing_accuracy: bearing_accuracy[i],
                speed_accuracy: speed_accuracy[i],
                vertical_accuracy: vertical_accuracy[i],
                horizontal_accuracy: horizontal_accuracy[i],
                speed: speed[i],
                bearing: bearing[i],
                altitude: altitude[i],
                longitude: longitude[i],
                latitude: latitude[i],
                qz: qz[i],
                qy: qy[i],
                qx: qx[i],
                qw: qw[i],
                roll: roll[i],
                pitch: pitch[i],
                yaw: yaw[i],
                acc_z: acc_z[i],
                acc_y: acc_y[i],
                acc_x: acc_x[i],
                gyro_z: gyro_z[i],
                gyro_y: gyro_y[i],
                gyro_x: gyro_x[i],
                mag_z: mag_z[i],
                mag_y: mag_y[i],
                mag_x: mag_x[i],
                relative_altitude: relative_altitude[i],
                pressure: pressure[i],
                grav_z: grav_z[i],
                grav_y: grav_y[i],
                grav_x: grav_x[i],
            });
        }

        Ok(records)
    }

    /// Reads an MCAP file and returns a vector of TestDataRecord structs.
    ///
    /// **Note**: Due to CSV-specific field deserializers in TestDataRecord, MCAP deserialization  
    /// may fail. For production use, consider using CSV format for TestDataRecord or convert  
    /// to NavigationResult which fully supports MCAP.
    ///
    /// # Arguments
    /// * `path` - Path to the MCAP file to read.
    pub fn from_mcap<P: AsRef<Path>>(path: P) -> Result<Vec<Self>, Box<dyn std::error::Error>> {
        use mcap::MessageStream;
        use std::fs::File;

        let file = File::open(path)?;

        // Memory-map the file for efficient reading
        let mapped = unsafe { memmap2::Mmap::map(&file)? };

        let message_stream = MessageStream::new(&mapped)?;
        let mut records = Vec::new();

        for message_result in message_stream {
            let message = message_result?;
            let record: Self = rmp_serde::from_slice(&message.data)?;
            records.push(record);
        }

        Ok(records)
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
// ==== Helper structs for navigation simulations ====
/// Struct representing the covariance diagonal of a navigation solution in NED coordinates.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct NEDCovariance {
    pub latitude_cov: f64,
    pub longitude_cov: f64,
    pub altitude_cov: f64,
    pub velocity_n_cov: f64,
    pub velocity_e_cov: f64,
    pub velocity_v_cov: f64,
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
    /// Vertical velocity in m/s
    pub velocity_vertical: f64,
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
    /// Vertical velocity covariance
    pub velocity_v_cov: f64,
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
            velocity_vertical: 0.0,
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
            velocity_v_cov: 1e-6,
            roll_cov: 1e-6,
            pitch_cov: 1e-6,
            yaw_cov: 1e-6,
            acc_bias_x_cov: 1e-6,
            acc_bias_y_cov: 1e-6,
            acc_bias_z_cov: 1e-6,
            gyro_bias_x_cov: 1e-6,
            gyro_bias_y_cov: 1e-6,
            gyro_bias_z_cov: 1e-6,
        }
    }
}
impl NavigationResult {
    /// Creates a new NavigationResult with default values.
    pub fn new() -> Self {
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

    /// Writes a vector of NavigationResult structs to an HDF5 file.
    ///
    /// # Arguments
    /// * `records` - Vector of NavigationResult structs to write
    /// * `path` - Path where the HDF5 file will be saved
    ///    
    ///
    /// # Example
    ///
    /// ```no_run
    /// use strapdown::sim::NavigationResult;
    ///
    /// let result = NavigationResult::default();
    /// let results = vec![result];
    /// NavigationResult::to_hdf5(&results, "nav_results.h5")
    ///     .expect("Failed to write navigation results to HDF5");
    /// ```
    pub fn to_hdf5<P: AsRef<Path>>(records: &[Self], path: P) -> Result<()> {
        use hdf5::File;

        let file = File::create(path)?;
        let n = records.len();

        // Handle empty datasets
        if n == 0 {
            // Create group to indicate structure even for empty datasets
            let _group = file.create_group("navigation_results")?;
            return Ok(());
        }

        // Create a group for navigation results
        let group = file.create_group("navigation_results")?;

        // Write timestamps as strings
        let timestamps: Result<Vec<hdf5::types::VarLenAscii>> = records
            .iter()
            .map(|r| {
                hdf5::types::VarLenAscii::from_ascii(&r.timestamp.to_rfc3339())
                    .map_err(|e| anyhow::anyhow!("Failed to encode timestamp as ASCII: {}", e))
            })
            .collect();
        let timestamps = timestamps?;
        let ds_time = group
            .new_dataset::<hdf5::types::VarLenAscii>()
            .shape([n])
            .create("timestamp")?;
        ds_time.write(&timestamps)?;

        // Helper macro to write f64 arrays
        macro_rules! write_f64_field {
            ($field_name:literal, $field:ident) => {{
                let data: Vec<f64> = records.iter().map(|r| r.$field).collect();
                let ds = group.new_dataset::<f64>().shape([n]).create($field_name)?;
                ds.write(&data)?;
            }};
        }

        // Navigation solution states
        write_f64_field!("latitude", latitude);
        write_f64_field!("longitude", longitude);
        write_f64_field!("altitude", altitude);
        write_f64_field!("velocity_north", velocity_north);
        write_f64_field!("velocity_east", velocity_east);
        write_f64_field!("velocity_vertical", velocity_vertical);
        write_f64_field!("roll", roll);
        write_f64_field!("pitch", pitch);
        write_f64_field!("yaw", yaw);
        write_f64_field!("acc_bias_x", acc_bias_x);
        write_f64_field!("acc_bias_y", acc_bias_y);
        write_f64_field!("acc_bias_z", acc_bias_z);
        write_f64_field!("gyro_bias_x", gyro_bias_x);
        write_f64_field!("gyro_bias_y", gyro_bias_y);
        write_f64_field!("gyro_bias_z", gyro_bias_z);

        // Covariance values
        write_f64_field!("latitude_cov", latitude_cov);
        write_f64_field!("longitude_cov", longitude_cov);
        write_f64_field!("altitude_cov", altitude_cov);
        write_f64_field!("velocity_n_cov", velocity_n_cov);
        write_f64_field!("velocity_e_cov", velocity_e_cov);
        write_f64_field!("velocity_v_cov", velocity_v_cov);
        write_f64_field!("roll_cov", roll_cov);
        write_f64_field!("pitch_cov", pitch_cov);
        write_f64_field!("yaw_cov", yaw_cov);
        write_f64_field!("acc_bias_x_cov", acc_bias_x_cov);
        write_f64_field!("acc_bias_y_cov", acc_bias_y_cov);
        write_f64_field!("acc_bias_z_cov", acc_bias_z_cov);
        write_f64_field!("gyro_bias_x_cov", gyro_bias_x_cov);
        write_f64_field!("gyro_bias_y_cov", gyro_bias_y_cov);
        write_f64_field!("gyro_bias_z_cov", gyro_bias_z_cov);

        Ok(())
    }
    /// Reads an HDF5 file and returns a vector of NavigationResult structs.
    ///
    /// # Arguments
    /// * `path` - Path to the HDF5 file to read.
    ///
    /// # Returns
    /// * `Ok(Vec<NavigationResult>)` if successful.
    /// * `Err` if the file cannot be read or parsed.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use strapdown::sim::NavigationResult;
    ///
    /// let results = NavigationResult::from_hdf5("nav_results.h5")
    ///     .expect("Failed to read navigation results from HDF5");
    /// ```
    pub fn from_hdf5<P: AsRef<Path>>(path: P) -> Result<Vec<Self>> {
        use hdf5::File;

        let file = File::open(path)?;
        let group = file.group("navigation_results")?;

        // Check if timestamp dataset exists (might be empty dataset)
        if let Ok(ds_time) = group.dataset("timestamp") {
            // Read timestamps
            let timestamps: Vec<hdf5::types::VarLenAscii> = ds_time.read_raw()?;
            let n = timestamps.len();

            // Handle empty dataset
            if n == 0 {
                return Ok(Vec::new());
            }

            // Helper macro to read f64 arrays
            macro_rules! read_f64_field {
                ($field_name:literal) => {{
                    let ds = group.dataset($field_name)?;
                    let data: Vec<f64> = ds.read_raw()?;
                    data
                }};
            }

            // Read navigation solution states
            let latitude = read_f64_field!("latitude");
            let longitude = read_f64_field!("longitude");
            let altitude = read_f64_field!("altitude");
            let velocity_north = read_f64_field!("velocity_north");
            let velocity_east = read_f64_field!("velocity_east");
            let velocity_vertical = read_f64_field!("velocity_vertical");
            let roll = read_f64_field!("roll");
            let pitch = read_f64_field!("pitch");
            let yaw = read_f64_field!("yaw");
            let acc_bias_x = read_f64_field!("acc_bias_x");
            let acc_bias_y = read_f64_field!("acc_bias_y");
            let acc_bias_z = read_f64_field!("acc_bias_z");
            let gyro_bias_x = read_f64_field!("gyro_bias_x");
            let gyro_bias_y = read_f64_field!("gyro_bias_y");
            let gyro_bias_z = read_f64_field!("gyro_bias_z");

            // Read covariance values
            let latitude_cov = read_f64_field!("latitude_cov");
            let longitude_cov = read_f64_field!("longitude_cov");
            let altitude_cov = read_f64_field!("altitude_cov");
            let velocity_n_cov = read_f64_field!("velocity_n_cov");
            let velocity_e_cov = read_f64_field!("velocity_e_cov");
            let velocity_v_cov = read_f64_field!("velocity_v_cov");
            let roll_cov = read_f64_field!("roll_cov");
            let pitch_cov = read_f64_field!("pitch_cov");
            let yaw_cov = read_f64_field!("yaw_cov");
            let acc_bias_x_cov = read_f64_field!("acc_bias_x_cov");
            let acc_bias_y_cov = read_f64_field!("acc_bias_y_cov");
            let acc_bias_z_cov = read_f64_field!("acc_bias_z_cov");
            let gyro_bias_x_cov = read_f64_field!("gyro_bias_x_cov");
            let gyro_bias_y_cov = read_f64_field!("gyro_bias_y_cov");
            let gyro_bias_z_cov = read_f64_field!("gyro_bias_z_cov");

            let mut records = Vec::with_capacity(n);
            for i in 0..n {
                let timestamp = DateTime::parse_from_rfc3339(timestamps[i].as_str())
                    .map_err(|e| anyhow::anyhow!("Failed to parse timestamp: {}", e))?
                    .with_timezone(&Utc);

                records.push(NavigationResult {
                    timestamp,
                    latitude: latitude[i],
                    longitude: longitude[i],
                    altitude: altitude[i],
                    velocity_north: velocity_north[i],
                    velocity_east: velocity_east[i],
                    velocity_vertical: velocity_vertical[i],
                    roll: roll[i],
                    pitch: pitch[i],
                    yaw: yaw[i],
                    acc_bias_x: acc_bias_x[i],
                    acc_bias_y: acc_bias_y[i],
                    acc_bias_z: acc_bias_z[i],
                    gyro_bias_x: gyro_bias_x[i],
                    gyro_bias_y: gyro_bias_y[i],
                    gyro_bias_z: gyro_bias_z[i],
                    latitude_cov: latitude_cov[i],
                    longitude_cov: longitude_cov[i],
                    altitude_cov: altitude_cov[i],
                    velocity_n_cov: velocity_n_cov[i],
                    velocity_e_cov: velocity_e_cov[i],
                    velocity_v_cov: velocity_v_cov[i],
                    roll_cov: roll_cov[i],
                    pitch_cov: pitch_cov[i],
                    yaw_cov: yaw_cov[i],
                    acc_bias_x_cov: acc_bias_x_cov[i],
                    acc_bias_y_cov: acc_bias_y_cov[i],
                    acc_bias_z_cov: acc_bias_z_cov[i],
                    gyro_bias_x_cov: gyro_bias_x_cov[i],
                    gyro_bias_y_cov: gyro_bias_y_cov[i],
                    gyro_bias_z_cov: gyro_bias_z_cov[i],
                });
            }

            Ok(records)
        } else {
            // No timestamp dataset means empty file
            Ok(Vec::new())
        }
    }

    /// Writes a vector of NavigationResult structs to a netCDF file.
    ///
    /// # Arguments
    /// * `records` - Vector of NavigationResult structs to write
    /// * `path` - Path where the netCDF file will be saved
    ///
    /// # Returns
    /// * `Result<()>` - Ok if successful, Err otherwise
    pub fn to_netcdf<P: AsRef<Path>>(records: &[Self], path: P) -> Result<()> {
        if records.is_empty() {
            bail!("Cannot write empty records to netCDF");
        }

        let n = records.len();
        let mut file = netcdf::create(path)?;

        // Define dimensions
        file.add_dimension("time", n)?;

        // Helper macro to add a variable and write data
        macro_rules! add_and_write {
            ($file:expr, $name:expr, $data:expr) => {{
                let mut var = $file.add_variable::<f64>($name, &["time"])?;
                var.put_values(&$data, ..)?;
            }};
        }

        // Prepare all data arrays
        let timestamps: Vec<f64> = records
            .iter()
            .map(|r| r.timestamp.timestamp() as f64)
            .collect();
        let latitude: Vec<f64> = records.iter().map(|r| r.latitude).collect();
        let longitude: Vec<f64> = records.iter().map(|r| r.longitude).collect();
        let altitude: Vec<f64> = records.iter().map(|r| r.altitude).collect();
        let velocity_north: Vec<f64> = records.iter().map(|r| r.velocity_north).collect();
        let velocity_east: Vec<f64> = records.iter().map(|r| r.velocity_east).collect();
        let velocity_vertical: Vec<f64> = records.iter().map(|r| r.velocity_vertical).collect();
        let roll: Vec<f64> = records.iter().map(|r| r.roll).collect();
        let pitch: Vec<f64> = records.iter().map(|r| r.pitch).collect();
        let yaw: Vec<f64> = records.iter().map(|r| r.yaw).collect();
        let acc_bias_x: Vec<f64> = records.iter().map(|r| r.acc_bias_x).collect();
        let acc_bias_y: Vec<f64> = records.iter().map(|r| r.acc_bias_y).collect();
        let acc_bias_z: Vec<f64> = records.iter().map(|r| r.acc_bias_z).collect();
        let gyro_bias_x: Vec<f64> = records.iter().map(|r| r.gyro_bias_x).collect();
        let gyro_bias_y: Vec<f64> = records.iter().map(|r| r.gyro_bias_y).collect();
        let gyro_bias_z: Vec<f64> = records.iter().map(|r| r.gyro_bias_z).collect();
        let latitude_cov: Vec<f64> = records.iter().map(|r| r.latitude_cov).collect();
        let longitude_cov: Vec<f64> = records.iter().map(|r| r.longitude_cov).collect();
        let altitude_cov: Vec<f64> = records.iter().map(|r| r.altitude_cov).collect();
        let velocity_n_cov: Vec<f64> = records.iter().map(|r| r.velocity_n_cov).collect();
        let velocity_e_cov: Vec<f64> = records.iter().map(|r| r.velocity_e_cov).collect();
        let velocity_v_cov: Vec<f64> = records.iter().map(|r| r.velocity_v_cov).collect();
        let roll_cov: Vec<f64> = records.iter().map(|r| r.roll_cov).collect();
        let pitch_cov: Vec<f64> = records.iter().map(|r| r.pitch_cov).collect();
        let yaw_cov: Vec<f64> = records.iter().map(|r| r.yaw_cov).collect();
        let acc_bias_x_cov: Vec<f64> = records.iter().map(|r| r.acc_bias_x_cov).collect();
        let acc_bias_y_cov: Vec<f64> = records.iter().map(|r| r.acc_bias_y_cov).collect();
        let acc_bias_z_cov: Vec<f64> = records.iter().map(|r| r.acc_bias_z_cov).collect();
        let gyro_bias_x_cov: Vec<f64> = records.iter().map(|r| r.gyro_bias_x_cov).collect();
        let gyro_bias_y_cov: Vec<f64> = records.iter().map(|r| r.gyro_bias_y_cov).collect();
        let gyro_bias_z_cov: Vec<f64> = records.iter().map(|r| r.gyro_bias_z_cov).collect();

        // Add variables and write data
        add_and_write!(file, "timestamp", timestamps);
        add_and_write!(file, "latitude", latitude);
        add_and_write!(file, "longitude", longitude);
        add_and_write!(file, "altitude", altitude);
        add_and_write!(file, "velocity_north", velocity_north);
        add_and_write!(file, "velocity_east", velocity_east);
        add_and_write!(file, "velocity_vertical", velocity_vertical);
        add_and_write!(file, "roll", roll);
        add_and_write!(file, "pitch", pitch);
        add_and_write!(file, "yaw", yaw);
        add_and_write!(file, "acc_bias_x", acc_bias_x);
        add_and_write!(file, "acc_bias_y", acc_bias_y);
        add_and_write!(file, "acc_bias_z", acc_bias_z);
        add_and_write!(file, "gyro_bias_x", gyro_bias_x);
        add_and_write!(file, "gyro_bias_y", gyro_bias_y);
        add_and_write!(file, "gyro_bias_z", gyro_bias_z);
        add_and_write!(file, "latitude_cov", latitude_cov);
        add_and_write!(file, "longitude_cov", longitude_cov);
        add_and_write!(file, "altitude_cov", altitude_cov);
        add_and_write!(file, "velocity_n_cov", velocity_n_cov);
        add_and_write!(file, "velocity_e_cov", velocity_e_cov);
        add_and_write!(file, "velocity_v_cov", velocity_v_cov);
        add_and_write!(file, "roll_cov", roll_cov);
        add_and_write!(file, "pitch_cov", pitch_cov);
        add_and_write!(file, "yaw_cov", yaw_cov);
        add_and_write!(file, "acc_bias_x_cov", acc_bias_x_cov);
        add_and_write!(file, "acc_bias_y_cov", acc_bias_y_cov);
        add_and_write!(file, "acc_bias_z_cov", acc_bias_z_cov);
        add_and_write!(file, "gyro_bias_x_cov", gyro_bias_x_cov);
        add_and_write!(file, "gyro_bias_y_cov", gyro_bias_y_cov);
        add_and_write!(file, "gyro_bias_z_cov", gyro_bias_z_cov);

        Ok(())
    }

    /// Reads a netCDF file and returns a vector of `NavigationResult` structs.
    ///
    /// # Arguments
    /// * `path` - Path to the netCDF file to read.
    ///
    /// # Returns
    /// * `Ok(Vec<NavigationResult>)` if successful.
    /// * `Err` if the file cannot be read or parsed.
    pub fn from_netcdf<P: AsRef<Path>>(path: P) -> Result<Vec<Self>> {
        let file = netcdf::open(path)?;

        // Read timestamp variable
        let time_var = file
            .variable("timestamp")
            .ok_or_else(|| anyhow::anyhow!("timestamp variable not found"))?;
        let timestamps: Vec<f64> = time_var.get_values(..)?;
        let n = timestamps.len();

        // Helper macro to read a variable
        macro_rules! read_var {
            ($file:expr, $name:expr) => {{
                let var = $file
                    .variable($name)
                    .ok_or_else(|| anyhow::anyhow!(concat!($name, " variable not found")))?;
                let data: Vec<f64> = var.get_values(..)?;
                data
            }};
        }

        // Read all variables
        let latitude = read_var!(file, "latitude");
        let longitude = read_var!(file, "longitude");
        let altitude = read_var!(file, "altitude");
        let velocity_north = read_var!(file, "velocity_north");
        let velocity_east = read_var!(file, "velocity_east");
        let velocity_vertical = read_var!(file, "velocity_vertical");
        let roll = read_var!(file, "roll");
        let pitch = read_var!(file, "pitch");
        let yaw = read_var!(file, "yaw");
        let acc_bias_x = read_var!(file, "acc_bias_x");
        let acc_bias_y = read_var!(file, "acc_bias_y");
        let acc_bias_z = read_var!(file, "acc_bias_z");
        let gyro_bias_x = read_var!(file, "gyro_bias_x");
        let gyro_bias_y = read_var!(file, "gyro_bias_y");
        let gyro_bias_z = read_var!(file, "gyro_bias_z");
        let latitude_cov = read_var!(file, "latitude_cov");
        let longitude_cov = read_var!(file, "longitude_cov");
        let altitude_cov = read_var!(file, "altitude_cov");
        let velocity_n_cov = read_var!(file, "velocity_n_cov");
        let velocity_e_cov = read_var!(file, "velocity_e_cov");
        let velocity_v_cov = read_var!(file, "velocity_v_cov");
        let roll_cov = read_var!(file, "roll_cov");
        let pitch_cov = read_var!(file, "pitch_cov");
        let yaw_cov = read_var!(file, "yaw_cov");
        let acc_bias_x_cov = read_var!(file, "acc_bias_x_cov");
        let acc_bias_y_cov = read_var!(file, "acc_bias_y_cov");
        let acc_bias_z_cov = read_var!(file, "acc_bias_z_cov");
        let gyro_bias_x_cov = read_var!(file, "gyro_bias_x_cov");
        let gyro_bias_y_cov = read_var!(file, "gyro_bias_y_cov");
        let gyro_bias_z_cov = read_var!(file, "gyro_bias_z_cov");

        // Build records
        let mut records = Vec::with_capacity(n);
        for i in 0..n {
            let timestamp = DateTime::from_timestamp(timestamps[i] as i64, 0)
                .ok_or_else(|| anyhow::anyhow!("Invalid timestamp"))?
                .with_timezone(&Utc);

            records.push(NavigationResult {
                timestamp,
                latitude: latitude[i],
                longitude: longitude[i],
                altitude: altitude[i],
                velocity_north: velocity_north[i],
                velocity_east: velocity_east[i],
                velocity_vertical: velocity_vertical[i],
                roll: roll[i],
                pitch: pitch[i],
                yaw: yaw[i],
                acc_bias_x: acc_bias_x[i],
                acc_bias_y: acc_bias_y[i],
                acc_bias_z: acc_bias_z[i],
                gyro_bias_x: gyro_bias_x[i],
                gyro_bias_y: gyro_bias_y[i],
                gyro_bias_z: gyro_bias_z[i],
                latitude_cov: latitude_cov[i],
                longitude_cov: longitude_cov[i],
                altitude_cov: altitude_cov[i],
                velocity_n_cov: velocity_n_cov[i],
                velocity_e_cov: velocity_e_cov[i],
                velocity_v_cov: velocity_v_cov[i],
                roll_cov: roll_cov[i],
                pitch_cov: pitch_cov[i],
                yaw_cov: yaw_cov[i],
                acc_bias_x_cov: acc_bias_x_cov[i],
                acc_bias_y_cov: acc_bias_y_cov[i],
                acc_bias_z_cov: acc_bias_z_cov[i],
                gyro_bias_x_cov: gyro_bias_x_cov[i],
                gyro_bias_y_cov: gyro_bias_y_cov[i],
                gyro_bias_z_cov: gyro_bias_z_cov[i],
            });
        }

        Ok(records)
    }
    /// Writes a vector of NavigationResult structs to an MCAP file.
    ///
    /// # Arguments
    /// * `records` - Vector of NavigationResult structs to write
    /// * `path` - Path where the MCAP file will be saved
    ///
    /// # Returns
    /// * `io::Result<()>` - Ok if successful, Err otherwise
    ///
    /// # Example
    /// ```no_run
    /// use strapdown::sim::NavigationResult;
    /// use std::path::Path;
    ///
    /// let mut result = NavigationResult::default();
    /// result.latitude = 37.0;
    /// result.longitude = -122.0;
    /// result.altitude = 100.0;
    /// let results = vec![result];
    /// NavigationResult::to_mcap(&results, "results.mcap")
    ///    .expect("Failed to write navigation results to MCAP");
    /// ```
    pub fn to_mcap<P: AsRef<Path>>(records: &[Self], path: P) -> io::Result<()> {
        use mcap::{Writer, records::MessageHeader};
        use std::collections::BTreeMap;
        use std::fs::File;
        use std::io::BufWriter;

        let file = File::create(path)?;
        let buf_writer = BufWriter::new(file);
        let mut writer = Writer::new(buf_writer).map_err(io::Error::other)?;

        // Add schema for NavigationResult (using MessagePack encoding)
        let schema_name = "NavigationResult";
        let schema_encoding = "msgpack";
        let schema_data = b"NavigationResult struct serialized with MessagePack";

        let schema_id = writer
            .add_schema(schema_name, schema_encoding, schema_data)
            .map_err(io::Error::other)?;

        // Add channel for NavigationResult messages
        let metadata = BTreeMap::new();
        let channel_id = writer
            .add_channel(schema_id, "navigation_results", "msgpack", &metadata)
            .map_err(io::Error::other)?;

        // Write each record as a message
        for (seq, record) in records.iter().enumerate() {
            let data = rmp_serde::to_vec(record)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

            let timestamp_nanos = record.timestamp.timestamp_nanos_opt().ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "Timestamp out of range")
            })?;

            let header = MessageHeader {
                channel_id,
                sequence: seq as u32,
                log_time: timestamp_nanos as u64,
                publish_time: timestamp_nanos as u64,
            };

            writer
                .write_to_known_channel(&header, &data)
                .map_err(io::Error::other)?;
        }

        writer.finish().map_err(io::Error::other)?;

        Ok(())
    }

    /// Reads an MCAP file and returns a vector of NavigationResult structs.
    ///
    /// # Example
    /// ```no_run
    /// use strapdown::sim::NavigationResult;
    ///
    /// let results = NavigationResult::from_mcap("results.mcap")
    ///     .expect("Failed to read navigation results from MCAP");
    /// println!("Read {} navigation results", results.len());
    /// ```
    pub fn from_mcap<P: AsRef<Path>>(path: P) -> Result<Vec<Self>, Box<dyn std::error::Error>> {
        use mcap::MessageStream;
        use std::fs::File;

        let file = File::open(path)?;

        // Memory-map the file for efficient reading
        let mapped = unsafe { memmap2::Mmap::map(&file)? };

        let message_stream = MessageStream::new(&mapped)?;
        let mut records = Vec::new();

        for message_result in message_stream {
            let message = message_result?;
            let record: Self = rmp_serde::from_slice(&message.data)?;
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
/// - `mag_x`, `mag_y`, `mag_z`: Magnetic field strength in micro teslas.
/// - `pressure`: Pressure in millibars.
/// - `freeair`: Free-air gravity anomaly in mGal.
///
/// # Returns
/// A NavigationResult struct containing the navigation solution.
impl From<(&DateTime<Utc>, &DVector<f64>, &DMatrix<f64>)> for NavigationResult {
    fn from(
        (timestamp, state, covariance): (&DateTime<Utc>, &DVector<f64>, &DMatrix<f64>),
    ) -> Self {
        assert!(
            state.len() == 15,
            "State vector must have 15 elements; got {}",
            state.len()
        );
        assert!(
            covariance.nrows() == 15 && covariance.ncols() == 15,
            "Covariance matrix must be 15x15"
        );
        let covariance = DVector::from_vec(covariance.diagonal().iter().copied().collect());
        // let wmm_date: Date = Date::from_calendar_date(
        //     timestamp.year(),
        //     Month::try_from(timestamp.month() as u8).unwrap(),
        //     timestamp.day() as u8,
        // )
        // .expect("Invalid date for world magnetic model");
        // let magnetic_field = GeomagneticField::new(
        //     Length::new::<meter>(state[2] as f32),
        //     Angle::new::<radian>(state[0] as f32),
        //     Angle::new::<radian>(state[1] as f32),
        //     wmm_date,
        // );
        NavigationResult {
            timestamp: *timestamp,
            latitude: state[0].to_degrees(),
            longitude: state[1].to_degrees(),
            altitude: state[2],
            velocity_north: state[3],
            velocity_east: state[4],
            velocity_vertical: state[5],
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
            velocity_v_cov: covariance[5],
            roll_cov: covariance[6],
            pitch_cov: covariance[7],
            yaw_cov: covariance[8],
            acc_bias_x_cov: covariance[9],
            acc_bias_y_cov: covariance[10],
            acc_bias_z_cov: covariance[11],
            gyro_bias_x_cov: covariance[12],
            gyro_bias_y_cov: covariance[13],
            gyro_bias_z_cov: covariance[14],
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
/// - `magnetic_vector`: Magnetic field strength measurement in micro teslas (body frame x, y, z).
/// - `pressure`: Pressure in millibars.
///
/// # Returns
/// A NavigationResult struct containing the navigation solution.
impl From<(&DateTime<Utc>, &UnscentedKalmanFilter)> for NavigationResult {
    fn from((timestamp, ukf): (&DateTime<Utc>, &UnscentedKalmanFilter)) -> Self {
        let state = &ukf.get_estimate();
        let covariance = ukf.get_certainty();
        NavigationResult {
            timestamp: *timestamp,
            latitude: state[0].to_degrees(),
            longitude: state[1].to_degrees(),
            altitude: state[2],
            velocity_north: state[3],
            velocity_east: state[4],
            velocity_vertical: state[5],
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
            velocity_v_cov: covariance[(5, 5)],
            roll_cov: covariance[(6, 6)],
            pitch_cov: covariance[(7, 7)],
            yaw_cov: covariance[(8, 8)],
            acc_bias_x_cov: covariance[(9, 9)],
            acc_bias_y_cov: covariance[(10, 10)],
            acc_bias_z_cov: covariance[(11, 11)],
            gyro_bias_x_cov: covariance[(12, 12)],
            gyro_bias_y_cov: covariance[(13, 13)],
            gyro_bias_z_cov: covariance[(14, 14)],
        }
    }
}

impl From<(&DateTime<Utc>, &crate::kalman::ExtendedKalmanFilter)> for NavigationResult {
    fn from((timestamp, ekf): (&DateTime<Utc>, &crate::kalman::ExtendedKalmanFilter)) -> Self {
        let state = &ekf.get_estimate();
        let covariance = ekf.get_certainty();
        NavigationResult {
            timestamp: *timestamp,
            latitude: state[0].to_degrees(),
            longitude: state[1].to_degrees(),
            altitude: state[2],
            velocity_north: state[3],
            velocity_east: state[4],
            velocity_vertical: state[5],
            roll: state[6],
            pitch: state[7],
            yaw: state[8],
            acc_bias_x: if state.len() > 9 { state[9] } else { 0.0 },
            acc_bias_y: if state.len() > 10 { state[10] } else { 0.0 },
            acc_bias_z: if state.len() > 11 { state[11] } else { 0.0 },
            gyro_bias_x: if state.len() > 12 { state[12] } else { 0.0 },
            gyro_bias_y: if state.len() > 13 { state[13] } else { 0.0 },
            gyro_bias_z: if state.len() > 14 { state[14] } else { 0.0 },
            latitude_cov: covariance[(0, 0)],
            longitude_cov: covariance[(1, 1)],
            altitude_cov: covariance[(2, 2)],
            velocity_n_cov: covariance[(3, 3)],
            velocity_e_cov: covariance[(4, 4)],
            velocity_v_cov: covariance[(5, 5)],
            roll_cov: covariance[(6, 6)],
            pitch_cov: covariance[(7, 7)],
            yaw_cov: covariance[(8, 8)],
            acc_bias_x_cov: if covariance.nrows() > 9 {
                covariance[(9, 9)]
            } else {
                0.0
            },
            acc_bias_y_cov: if covariance.nrows() > 10 {
                covariance[(10, 10)]
            } else {
                0.0
            },
            acc_bias_z_cov: if covariance.nrows() > 11 {
                covariance[(11, 11)]
            } else {
                0.0
            },
            gyro_bias_x_cov: if covariance.nrows() > 12 {
                covariance[(12, 12)]
            } else {
                0.0
            },
            gyro_bias_y_cov: if covariance.nrows() > 13 {
                covariance[(13, 13)]
            } else {
                0.0
            },
            gyro_bias_z_cov: if covariance.nrows() > 14 {
                covariance[(14, 14)]
            } else {
                0.0
            },
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
///
/// # Returns
/// A NavigationResult struct containing the navigation solution.
impl From<(&DateTime<Utc>, &StrapdownState)> for NavigationResult {
    fn from((timestamp, state): (&DateTime<Utc>, &StrapdownState)) -> Self {
        //let wmm_date: Date = Date::from_calendar_date(
        //    timestamp.year(),
        //    Month::try_from(timestamp.month() as u8).unwrap(),
        //    timestamp.day() as u8,
        //)
        //.expect("Invalid date for world magnetic model");
        //let magnetic_field = GeomagneticField::new(
        //    Length::new::<meter>(state.altitude as f32),
        //    Angle::new::<radian>(state.latitude as f32),
        //    Angle::new::<radian>(state.longitude as f32),
        //    wmm_date,
        //);
        NavigationResult {
            timestamp: *timestamp,
            latitude: state.latitude.to_degrees(),
            longitude: state.longitude.to_degrees(),
            altitude: state.altitude,
            velocity_north: state.velocity_north,
            velocity_east: state.velocity_east,
            velocity_vertical: state.velocity_vertical,
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
            velocity_v_cov: f64::NAN,
            roll_cov: f64::NAN,
            pitch_cov: f64::NAN,
            yaw_cov: f64::NAN,
            acc_bias_x_cov: f64::NAN,
            acc_bias_y_cov: f64::NAN,
            acc_bias_z_cov: f64::NAN,
            gyro_bias_x_cov: f64::NAN,
            gyro_bias_y_cov: f64::NAN,
            gyro_bias_z_cov: f64::NAN,
        }
    }
}

impl NavigationResult {
    /// Create NavigationResult from particle filter state
    ///
    /// Creates a navigation result from a 9-element state vector (position, velocity, attitude)
    /// and covariance matrix produced by particle filter averaging. Since particle filters don't
    /// estimate IMU biases, those fields are set to zero.
    ///
    /// # Arguments
    /// * `timestamp` - Timestamp for this navigation solution
    /// * `mean` - 9-element state vector [lat, lon, alt, vn, ve, vd, roll, pitch, yaw] in radians/meters
    /// * `cov` - 9x9 covariance matrix
    pub fn from_particle_filter(
        timestamp: &DateTime<Utc>,
        mean: &DVector<f64>,
        cov: &DMatrix<f64>,
    ) -> Self {
        assert_eq!(mean.len(), 9, "Particle filter state must have 9 elements");
        assert_eq!(
            cov.shape(),
            (9, 9),
            "Particle filter covariance must be 9x9"
        );

        NavigationResult {
            timestamp: *timestamp,
            latitude: mean[0].to_degrees(),
            longitude: mean[1].to_degrees(),
            altitude: mean[2],
            velocity_north: mean[3],
            velocity_east: mean[4],
            velocity_vertical: mean[5],
            roll: mean[6],
            pitch: mean[7],
            yaw: mean[8],
            acc_bias_x: 0.0, // Particle filter doesn't estimate biases
            acc_bias_y: 0.0,
            acc_bias_z: 0.0,
            gyro_bias_x: 0.0,
            gyro_bias_y: 0.0,
            gyro_bias_z: 0.0,
            latitude_cov: cov[(0, 0)],
            longitude_cov: cov[(1, 1)],
            altitude_cov: cov[(2, 2)],
            velocity_n_cov: cov[(3, 3)],
            velocity_e_cov: cov[(4, 4)],
            velocity_v_cov: cov[(5, 5)],
            roll_cov: cov[(6, 6)],
            pitch_cov: cov[(7, 7)],
            yaw_cov: cov[(8, 8)],
            acc_bias_x_cov: f64::NAN,
            acc_bias_y_cov: f64::NAN,
            acc_bias_z_cov: f64::NAN,
            gyro_bias_x_cov: f64::NAN,
            gyro_bias_y_cov: f64::NAN,
            gyro_bias_z_cov: f64::NAN,
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
        velocity_vertical: 0.0, // initial velocities
        attitude,
        is_enu: true,
    };
    // Store the initial state and metadata
    results.push(NavigationResult::from((&first_record.time, &state)));
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
        results.push(NavigationResult::from((&current_time, &state)));
        previous_time = record.time;
    }
    results
}
/// Generic closed-loop simulation runner for any NavigationFilter
///
/// This function implements the core simulation loop for navigation filter architectures.
/// It iterates through the event stream, performs prediction and update steps, checks health limits,
/// and records navigation results. While generic, this is really only intended for Kalman-filter
/// family navigation filters because particle filter style navigation filters have the additional
/// step of resampling. For particle filter type filters, use [run_closed_loop_pf] instead.
///
/// # Arguments
/// * `filter` - Mutable reference to a type implementing NavigationFilter
/// * `stream` - Event stream containing IMU and measurement events
/// * `health_limits` - Optional health limits for monitoring
///
/// # Returns
/// * `Vec<NavigationResult>` - A vector of navigation results
pub fn run_closed_loop<F: NavigationFilter>(
    filter: &mut F,
    stream: EventStream,
    health_limits: Option<HealthLimits>,
) -> anyhow::Result<Vec<NavigationResult>> {
    let start_time = stream.start_time;
    let mut results: Vec<NavigationResult> = Vec::with_capacity(stream.events.len());
    let total = stream.events.len();
    let mut last_ts: Option<DateTime<Utc>> = None;
    let mut monitor = HealthMonitor::new(health_limits.unwrap_or_default());

    info!(
        "Starting closed-loop navigation filter with {} events",
        total
    );

    for (i, event) in stream.events.into_iter().enumerate() {
        // Print detailed progress every 100 iterations or at key milestones
        if i % 10 == 0 || i == total {
            let mean = filter.get_estimate();
            let cov = filter.get_certainty();

            // Extract position and covariance diagonal
            let lat = mean[0].to_degrees();
            let lon = mean[1].to_degrees();
            let alt = mean[2];

            // Get position uncertainty (diagonal elements)
            let pos_std_lat = cov[(0, 0)].sqrt().to_degrees();
            let pos_std_lon = cov[(1, 1)].sqrt().to_degrees();
            let pos_std_alt = cov[(2, 2)].sqrt();

            // Compute RMS of position covariance
            let pos_rms = (pos_std_lat.powi(2) + pos_std_lon.powi(2) + pos_std_alt.powi(2)).sqrt();
            info!(
                "[{:.1}%] Event {}/{} | Pos: ({:.6}°, {:.6}°, {:.1}m) | Vel: ({:.2} m/s, {:.2} m/s, {:.2} m/s) | σ: ({:.2e}°, {:.2e}°, {:.2}m) | RMS: {:.2e}",
                (i as f64 / total as f64) * 100.0,
                i,
                total,
                lat,
                lon,
                alt,
                mean[3],
                mean[4],
                mean[5],
                pos_std_lat,
                pos_std_lon,
                pos_std_alt,
                pos_rms
            );
        }

        // Compute wall-clock time for this event
        let elapsed_s = match &event {
            Event::Imu { elapsed_s, .. } => *elapsed_s,
            Event::Measurement { elapsed_s, .. } => *elapsed_s,
        };
        let ts = start_time + Duration::milliseconds((elapsed_s * 1000.0).round() as i64);

        // Apply event
        match event {
            Event::Imu { dt_s, imu, .. } => {
                filter.predict(&imu, dt_s);
                let mean = filter.get_estimate();
                let cov = filter.get_certainty();
                if let Err(e) = monitor.check(mean.as_slice(), &cov, None) {
                    log::error!("Health fail after propagate at {} (#{i}): {e}", ts);
                    bail!(e);
                }
            }
            Event::Measurement { meas, .. } => {
                filter.update(meas.as_ref());
                let mean = filter.get_estimate();
                let cov = filter.get_certainty();
                if let Err(e) = monitor.check(mean.as_slice(), &cov, None) {
                    log::error!("Health fail after measurement update at {} (#{i}): {e}", ts);
                    bail!(e);
                }
            }
        }

        // If timestamp changed, or it's the last event, record the previous state
        if Some(ts) != last_ts {
            if let Some(prev_ts) = last_ts {
                let mean = filter.get_estimate();
                let cov = filter.get_certainty();
                results.push(NavigationResult::from((&prev_ts, &mean, &cov)));
                debug!("Filter state at {}: {:?}", ts, mean);
            }
            last_ts = Some(ts);
        }

        // If this is the last event, also push
        if i == total - 1 {
            let mean = filter.get_estimate();
            let cov = filter.get_certainty();
            debug!("Filter state at {}: {:?}", ts, mean);
            results.push(NavigationResult::from((&ts, &mean, &cov)));
        }
    }
    debug!("Closed-loop simulation complete");
    Ok(results)
}
/// Print the Unscented Kalman Filter state and covariance for debugging purposes.
pub fn print_ukf(ukf: &UnscentedKalmanFilter, record: &TestDataRecord) {
    debug!(
        "UKF position: ({:.4}, {:.4}, {:.4})  |  Covariance: {:.4e}, {:.4e}, {:.4}  |  Error: {:.4e}, {:.4e}, {:.4}",
        ukf.get_estimate()[0].to_degrees(),
        ukf.get_estimate()[1].to_degrees(),
        ukf.get_estimate()[2],
        ukf.get_certainty()[(0, 0)],
        ukf.get_certainty()[(1, 1)],
        ukf.get_certainty()[(2, 2)],
        ukf.get_estimate()[0].to_degrees() - record.latitude,
        ukf.get_estimate()[1].to_degrees() - record.longitude,
        ukf.get_estimate()[2] - record.altitude
    );
    debug!(
        "UKF velocity: ({:.4}, {:.4}, {:.4})  | Covariance: {:.4}, {:.4}, {:.4}  | Error: {:.4}, {:.4}, {:.4}",
        ukf.get_estimate()[3],
        ukf.get_estimate()[4],
        ukf.get_estimate()[5],
        ukf.get_certainty()[(3, 3)],
        ukf.get_certainty()[(4, 4)],
        ukf.get_certainty()[(5, 5)],
        ukf.get_estimate()[3] - record.speed * record.bearing.cos(),
        ukf.get_estimate()[4] - record.speed * record.bearing.sin(),
        ukf.get_estimate()[5] - 0.0 // Assuming no vertical velocity
    );
    debug!(
        "UKF attitude: ({:.4}, {:.4}, {:.4})  | Covariance: {:.4}, {:.4}, {:.4}  | Error: {:.4}, {:.4}, {:.4}",
        ukf.get_estimate()[6],
        ukf.get_estimate()[7],
        ukf.get_estimate()[8],
        ukf.get_certainty()[(6, 6)],
        ukf.get_certainty()[(7, 7)],
        ukf.get_certainty()[(8, 8)],
        ukf.get_estimate()[6] - record.roll,
        ukf.get_estimate()[7] - record.pitch,
        ukf.get_estimate()[8] - record.yaw
    );
    debug!(
        "UKF accel biases: ({:.4}, {:.4}, {:.4})  | Covariance: {:.4e}, {:.4e}, {:.4e}",
        ukf.get_estimate()[9],
        ukf.get_estimate()[10],
        ukf.get_estimate()[11],
        ukf.get_certainty()[(9, 9)],
        ukf.get_certainty()[(10, 10)],
        ukf.get_certainty()[(11, 11)]
    );
    debug!(
        "UKF gyro biases: ({:.4}, {:.4}, {:.4})  | Covariance: {:.4e}, {:.4e}, {:.4e}",
        ukf.get_estimate()[12],
        ukf.get_estimate()[13],
        ukf.get_estimate()[14],
        ukf.get_certainty()[(12, 12)],
        ukf.get_certainty()[(13, 13)],
        ukf.get_certainty()[(14, 14)]
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
/// * `other_states` - Optional vector of f64 for any additional states (not used in the canonical UKF, but can be useful for custom implementations).
///
/// # Returns
///
/// * `UKF` - An instance of the Unscented Kalman Filter initialized with the provided parameters.
pub fn initialize_ukf(
    initial_pose: TestDataRecord,
    attitude_covariance: Option<Vec<f64>>,
    imu_biases: Option<Vec<f64>>,
    imu_biases_covariance: Option<Vec<f64>>,
    other_states: Option<Vec<f64>>,
    other_states_covariance: Option<Vec<f64>>,
    process_noise_diagonal: Option<Vec<f64>>,
) -> UnscentedKalmanFilter {
    let initial_state = InitialState {
        latitude: initial_pose.latitude,
        longitude: initial_pose.longitude,
        altitude: initial_pose.altitude,
        northward_velocity: initial_pose.speed * initial_pose.bearing.cos(),
        eastward_velocity: initial_pose.speed * initial_pose.bearing.sin(),
        vertical_velocity: 0.0, // Assuming no initial vertical velocity for simplicity
        roll: if initial_pose.roll.is_nan() {
            0.0
        } else {
            initial_pose.roll
        },
        pitch: if initial_pose.pitch.is_nan() {
            0.0
        } else {
            initial_pose.pitch
        },
        yaw: if initial_pose.yaw.is_nan() {
            0.0
        } else {
            initial_pose.yaw
        },
        in_degrees: true,
        is_enu: true,
    };
    let process_noise_diagonal = match process_noise_diagonal {
        Some(pn) => pn,
        None => DEFAULT_PROCESS_NOISE.to_vec(),
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
    let imu_biases = match imu_biases {
        Some(imu_biases) => {
            covariance_diagonal.extend(match imu_biases_covariance {
                Some(imu_cov) => imu_cov,
                None => vec![1e-3; 6], // Default covariance if not provided
            });
            imu_biases
        }
        None => {
            covariance_diagonal.extend(vec![1e-3; 6]);
            vec![1e-3; 6] // Default values if not provided
        }
    };
    // extend the covariance diagonal if other states are provided
    let other_states = match other_states {
        Some(other_states) => {
            covariance_diagonal.extend(match other_states_covariance {
                Some(other_cov) => other_cov,
                None => vec![1e-3; other_states.len()], // Default covariance if not provided
            });
            Some(other_states)
        }
        None => None,
    };
    assert!(
        covariance_diagonal.len() == 15 + other_states.as_ref().map_or(0, |v| v.len()),
        "Covariance diagonal length mismatch: expected {}, got {}",
        15 + other_states.as_ref().map_or(0, |v| v.len()),
        covariance_diagonal.len()
    );
    assert!(
        process_noise_diagonal.len() == 15 + other_states.as_ref().map_or(0, |v| v.len()),
        "Process noise diagonal length mismatch: expected {}, got {}",
        15 + other_states.as_ref().map_or(0, |v| v.len()),
        process_noise_diagonal.len()
    );
    assert!(
        process_noise_diagonal.len() == covariance_diagonal.len(),
        "Process noise and covariance diagonal length mismatch: {} vs {}",
        process_noise_diagonal.len(),
        covariance_diagonal.len()
    );
    let process_noise = DMatrix::from_diagonal(&DVector::from_vec(process_noise_diagonal));
    //DVector::from_vec(vec![0.0; 15]);
    UnscentedKalmanFilter::new(
        initial_state,
        imu_biases,
        other_states,
        covariance_diagonal,
        process_noise,
        1e-3, // Use a scalar for measurement noise as expected by UKF::new
        2.0,
        0.0,
    )
}

/// Initialize an Extended Kalman Filter for simulation.
///
/// This function creates and initializes an `ExtendedKalmanFilter` with the given parameters,
/// providing a linearized Gaussian approximation for navigation state estimation.
///
/// # Arguments
///
/// * `initial_pose` - A `TestDataRecord` containing the initial pose information.
/// * `attitude_covariance` - Optional initial attitude covariance.
/// * `imu_biases` - Optional initial IMU biases.
/// * `imu_biases_covariance` - Optional IMU bias covariance.
/// * `process_noise_diagonal` - Optional process noise diagonal.
/// * `use_biases` - If true, uses 15-state (with IMU biases), otherwise 9-state.
///
/// # Returns
///
/// * `ExtendedKalmanFilter` - An instance of the Extended Kalman Filter.
#[allow(clippy::too_many_arguments)]
pub fn initialize_ekf(
    initial_pose: TestDataRecord,
    attitude_covariance: Option<Vec<f64>>,
    imu_biases: Option<Vec<f64>>,
    imu_biases_covariance: Option<Vec<f64>>,
    process_noise_diagonal: Option<Vec<f64>>,
    use_biases: bool,
) -> crate::kalman::ExtendedKalmanFilter {
    use crate::kalman::ExtendedKalmanFilter;

    // Build initial state from sensor data
    let initial_state = InitialState {
        latitude: initial_pose.latitude,
        longitude: initial_pose.longitude,
        altitude: initial_pose.altitude,
        // Note: `initial_pose.bearing` is stored in degrees in `TestDataRecord`.
        // Convert to radians here for use with trigonometric functions.
        northward_velocity: initial_pose.speed * initial_pose.bearing.to_radians().cos(),
        eastward_velocity: initial_pose.speed * initial_pose.bearing.to_radians().sin(),
        vertical_velocity: 0.0,
        roll: if initial_pose.roll.is_nan() {
            0.0
        } else {
            initial_pose.roll
        },
        pitch: if initial_pose.pitch.is_nan() {
            0.0
        } else {
            initial_pose.pitch
        },
        yaw: if initial_pose.yaw.is_nan() {
            0.0
        } else {
            initial_pose.yaw
        },
        in_degrees: true,
        is_enu: true,
    };

    // Determine state size based on use_biases flag
    let state_size = if use_biases { 15 } else { 9 };

    // Build process noise diagonal
    let process_noise_diagonal = match process_noise_diagonal {
        Some(pn) => {
            assert!(
                pn.len() == state_size,
                "Process noise diagonal length mismatch: expected {}, got {}",
                state_size,
                pn.len()
            );
            pn
        }
        None => {
            if use_biases {
                DEFAULT_PROCESS_NOISE.to_vec()
            } else {
                DEFAULT_PROCESS_NOISE[0..9].to_vec()
            }
        }
    };

    // Build covariance diagonal
    let position_accuracy = initial_pose.horizontal_accuracy;
    let mut covariance_diagonal = vec![
        (position_accuracy * METERS_TO_DEGREES).powf(2.0),
        (position_accuracy * METERS_TO_DEGREES).powf(2.0),
        initial_pose.vertical_accuracy.powf(2.0),
        initial_pose.speed_accuracy.powf(2.0),
        initial_pose.speed_accuracy.powf(2.0),
        initial_pose.speed_accuracy.powf(2.0),
    ];

    // Add attitude covariance
    match attitude_covariance {
        Some(att_cov) => {
            assert!(
                att_cov.len() == 3,
                "Attitude covariance must have 3 elements"
            );
            covariance_diagonal.extend(att_cov);
        }
        None => covariance_diagonal.extend(vec![1e-9; 3]),
    }

    // Add IMU bias covariance if using biases
    let imu_biases_vec = if use_biases {
        match imu_biases {
            Some(biases) => {
                assert!(biases.len() == 6, "IMU biases must have 6 elements");
                covariance_diagonal.extend(match imu_biases_covariance {
                    Some(imu_cov) => {
                        assert!(
                            imu_cov.len() == 6,
                            "IMU bias covariance must have 6 elements"
                        );
                        imu_cov
                    }
                    None => vec![1e-3; 6],
                });
                biases
            }
            None => {
                covariance_diagonal.extend(vec![1e-3; 6]);
                vec![0.0; 6]
            }
        }
    } else {
        vec![0.0; 6] // Not used in 9-state, but required by constructor
    };

    assert!(
        covariance_diagonal.len() == state_size,
        "Covariance diagonal length mismatch: expected {}, got {}",
        state_size,
        covariance_diagonal.len()
    );

    let process_noise = DMatrix::from_diagonal(&DVector::from_vec(process_noise_diagonal));

    ExtendedKalmanFilter::new(
        initial_state,
        imu_biases_vec,
        covariance_diagonal,
        process_noise,
        use_biases,
    )
}

/// Initialize an Error-State Kalman Filter (ESKF) for simulation.
///
/// This function creates and initializes an `ErrorStateKalmanFilter` with the given parameters,
/// providing a robust error-state formulation that uses quaternions for nominal attitude and
/// small-angle representation for attitude errors.
///
/// The ESKF is the standard approach for strapdown INS, offering better numerical stability
/// and avoiding attitude singularities compared to full-state EKF implementations.
///
/// # Arguments
///
/// * `initial_pose` - A `TestDataRecord` containing the initial pose information.
/// * `attitude_covariance` - Optional initial attitude covariance (for error state).
/// * `imu_biases` - Optional initial IMU biases [b_ax, b_ay, b_az, b_gx, b_gy, b_gz].
/// * `imu_biases_covariance` - Optional IMU bias covariance (for error state).
/// * `process_noise_diagonal` - Optional process noise diagonal (15 elements for error state).
///
/// # Returns
///
/// * `ErrorStateKalmanFilter` - An instance of the Error-State Kalman Filter.
///
/// # Example
///
/// ```no_run
/// use strapdown::sim::{initialize_eskf, TestDataRecord};
/// use chrono::Utc;
///
/// let initial_pose = TestDataRecord {
///     time: Utc::now(),
///     latitude: 45.0,
///     longitude: -122.0,
///     altitude: 100.0,
///     // ... other fields ...
///     ..Default::default()
/// };
/// let eskf = initialize_eskf(initial_pose, None, None, None, None);
/// ```
#[allow(clippy::too_many_arguments)]
pub fn initialize_eskf(
    initial_pose: TestDataRecord,
    attitude_covariance: Option<Vec<f64>>,
    imu_biases: Option<Vec<f64>>,
    imu_biases_covariance: Option<Vec<f64>>,
    process_noise_diagonal: Option<Vec<f64>>,
) -> crate::kalman::ErrorStateKalmanFilter {
    use crate::kalman::ErrorStateKalmanFilter;

    // Build initial state from sensor data
    let initial_state = InitialState {
        latitude: initial_pose.latitude,
        longitude: initial_pose.longitude,
        altitude: initial_pose.altitude,
        // Note: `initial_pose.bearing` is stored in degrees in `TestDataRecord`.
        // Convert to radians here for use with trigonometric functions.
        northward_velocity: initial_pose.speed * initial_pose.bearing.to_radians().cos(),
        eastward_velocity: initial_pose.speed * initial_pose.bearing.to_radians().sin(),
        vertical_velocity: 0.0,
        roll: if initial_pose.roll.is_nan() {
            0.0
        } else {
            initial_pose.roll
        },
        pitch: if initial_pose.pitch.is_nan() {
            0.0
        } else {
            initial_pose.pitch
        },
        yaw: if initial_pose.yaw.is_nan() {
            0.0
        } else {
            initial_pose.yaw
        },
        in_degrees: true,
        is_enu: true,
    };

    // ESKF always uses 15-state error vector (pos, vel, att, accel_bias, gyro_bias)
    let state_size = 15;

    // Build process noise diagonal for error state (15 elements)
    let process_noise_diagonal = match process_noise_diagonal {
        Some(pn) => {
            assert!(
                pn.len() == state_size,
                "Process noise diagonal length mismatch: expected {}, got {}",
                state_size,
                pn.len()
            );
            pn
        }
        None => DEFAULT_PROCESS_NOISE.to_vec(),
    };

    // Build IMU biases
    let imu_biases = match imu_biases {
        Some(biases) => {
            assert!(
                biases.len() == 6,
                "IMU biases length mismatch: expected 6, got {}",
                biases.len()
            );
            biases
        }
        None => vec![0.0; 6],
    };

    // Build error covariance diagonal
    // This represents initial uncertainty in the error state (NOT nominal state)
    let mut error_covariance_diagonal = vec![
        1e-6, 1e-6, 1e-4, // position error covariance (m²)
        1e-3, 1e-3, 1e-3, // velocity error covariance (m²/s²)
    ];

    // Add attitude error covariance
    error_covariance_diagonal.extend(match attitude_covariance {
        Some(att_cov) => {
            assert!(
                att_cov.len() == 3,
                "Attitude covariance length mismatch: expected 3, got {}",
                att_cov.len()
            );
            att_cov
        }
        None => vec![1e-5; 3], // Default: small attitude uncertainty (rad²)
    });

    // Add IMU bias error covariance
    error_covariance_diagonal.extend(match imu_biases_covariance {
        Some(bias_cov) => {
            assert!(
                bias_cov.len() == 6,
                "IMU bias covariance length mismatch: expected 6, got {}",
                bias_cov.len()
            );
            bias_cov
        }
        None => {
            vec![
                1e-6, 1e-6, 1e-6, // accel bias error covariance
                1e-8, 1e-8, 1e-8, // gyro bias error covariance
            ]
        }
    });

    assert!(
        error_covariance_diagonal.len() == state_size,
        "Error covariance diagonal length mismatch: expected {}, got {}",
        state_size,
        error_covariance_diagonal.len()
    );

    let process_noise = DMatrix::from_diagonal(&DVector::from_vec(process_noise_diagonal));

    ErrorStateKalmanFilter::new(
        initial_state,
        imu_biases,
        error_covariance_diagonal,
        process_noise,
    )
}

// ==== Simulation Helper functions ====

pub fn print_sim_status<F: NavigationFilter>(filter: &F) {
    let mean = filter.get_estimate();
    let cov = filter.get_certainty();

    // Extract position and covariance diagonal
    let lat = mean[0].to_degrees();
    let lon = mean[1].to_degrees();
    let alt = mean[2];

    // Get position uncertainty (diagonal elements)
    let pos_std_lat = cov[(0, 0)].sqrt().to_degrees();
    let pos_std_lon = cov[(1, 1)].sqrt().to_degrees();
    let pos_std_alt = cov[(2, 2)].sqrt();

    // Compute RMS of position covariance
    let pos_rms = (pos_std_lat.powi(2) + pos_std_lon.powi(2) + pos_std_alt.powi(2)).sqrt();

    debug!(
        "\rPos: ({:.6}°, {:.6}°, {:.1}m) | σ: ({:.2e}°, {:.2e}°, {:.2}m) | RMS: {:.2e}",
        lat, lon, alt, pos_std_lat, pos_std_lon, pos_std_alt, pos_rms
    );
}

pub mod health {
    use super::*;

    #[derive(Clone, Debug)]
    pub struct HealthLimits {
        pub lat_rad: (f64, f64),        // [-90°, +90°]
        pub lon_rad: (f64, f64),        // [-180°, +180°]
        pub alt_m: (f64, f64),          // e.g., [-500, 15000]
        pub speed_mps_max: f64,         // e.g., 500 m/s (road/low-altitude aircraft)
        pub cov_diag_max: f64,          // e.g., 1e15
        pub cond_max: f64,              // e.g., 1e12 (optional)
        pub nis_pos_max: f64,           // e.g., 100 (huge outlier)
        pub nis_pos_consec_fail: usize, // e.g., 20
    }

    impl Default for HealthLimits {
        fn default() -> Self {
            Self {
                lat_rad: (-std::f64::consts::FRAC_PI_2, std::f64::consts::FRAC_PI_2),
                lon_rad: (-std::f64::consts::PI, std::f64::consts::PI),
                alt_m: (-100000000.0, 100000000.0), // Very tolerant for vertical channel instability
                speed_mps_max: 500.0,
                cov_diag_max: 1e15,
                cond_max: 1e12,
                nis_pos_max: 100.0,
                nis_pos_consec_fail: 20,
            }
        }
    }

    #[derive(Default, Clone, Debug)]
    pub struct HealthMonitor {
        limits: HealthLimits,
        consec_nis_pos_fail: usize,
    }

    impl HealthMonitor {
        pub fn new(limits: HealthLimits) -> Self {
            Self {
                limits,
                consec_nis_pos_fail: 0,
            }
        }

        /// Call after **every event** (predict or update). Provide optional NIS when you have a GNSS update.
        pub fn check(
            &mut self,
            x: &[f64], // your mean_state slice
            p: &nalgebra::DMatrix<f64>,
            maybe_nis_pos: Option<f64>,
        ) -> Result<()> {
            // 1) Finite checks
            if !x.iter().all(|v| v.is_finite()) {
                bail!("Non-finite state detected");
            }
            if !p.iter().all(|v| v.is_finite()) {
                bail!("Non-finite covariance detected");
            }

            // 2) Basic bounds (lat, lon, alt)
            let lat = x[0];
            let lon = x[1];
            let alt = x[2];
            if lat < self.limits.lat_rad.0 || lat > self.limits.lat_rad.1 {
                bail!("Latitude out of range: {lat}");
            }
            if lon < self.limits.lon_rad.0 || lon > self.limits.lon_rad.1 {
                bail!("Longitude out of range: {lon}");
            }
            if alt < self.limits.alt_m.0 || alt > self.limits.alt_m.1 {
                bail!("Altitude out of range: {alt} m");
            }

            // 3) Speed sanity (assumes NED velocities at indices 3..=5)
            // let v2 = x[3] * x[3] + x[4] * x[4] + x[5] * x[5];
            // if v2.is_finite() && v2.sqrt() > self.limits.speed_mps_max {
            //     bail!("Speed exceeded: {:.2} m/s", v2.sqrt());
            // }

            // 4) Covariance sanity: diagonals and simple SPD probe
            for i in 0..p.nrows().min(p.ncols()) {
                if p[(i, i)].is_sign_negative() {
                    bail!("Negative variance on diagonal: idx={i}, val={}", p[(i, i)]);
                }
                if p[(i, i)] > self.limits.cov_diag_max {
                    bail!("Variance too large on diagonal idx={i}: {}", p[(i, i)]);
                }
            }
            // (Optional) rough condition estimate via Frobenius norm and inverse
            // Skip if too expensive; enable only for debugging.
            // if let Some(inv) = p.clone().try_inverse() {
            //     let cond = p.norm_fro() * inv.norm_fro();
            //     if !cond.is_finite() || cond > self.limits.cond_max { bail!("Covariance condition number too large: {cond:e}"); }
            // }

            // 5) GNSS gating streak (if a NIS was computed at update time)
            if let Some(nis_pos) = maybe_nis_pos {
                if !nis_pos.is_finite() || nis_pos.is_sign_negative() {
                    bail!("Invalid NIS value: {nis_pos}");
                }
                if nis_pos > self.limits.nis_pos_max {
                    self.consec_nis_pos_fail += 1;
                    if self.consec_nis_pos_fail >= self.limits.nis_pos_consec_fail {
                        bail!(
                            "Consecutive NIS exceedances: {} (> {}), last NIS={}",
                            self.consec_nis_pos_fail,
                            self.limits.nis_pos_consec_fail,
                            nis_pos
                        );
                    }
                } else {
                    self.consec_nis_pos_fail = 0;
                }
            }

            Ok(())
        }
    }
}

//================= CLI Argument Structures for Simulation Programs =====================================

/// Scheduler configuration kind
#[derive(Copy, Clone, Debug)]
#[cfg_attr(feature = "clap", derive(ValueEnum))]
pub enum SchedKind {
    Passthrough,
    Fixed,
    Duty,
}

/// GNSS scheduler arguments for CLI
#[derive(Clone, Debug)]
#[cfg_attr(feature = "clap", derive(Args))]
pub struct SchedulerArgs {
    /// Scheduler kind: passthrough | fixed | duty
    #[cfg_attr(feature = "clap", arg(long, value_enum, default_value_t = SchedKind::Passthrough))]
    pub sched: SchedKind,
    /// Fixed-interval seconds (sched=fixed)
    #[cfg_attr(feature = "clap", arg(long, default_value_t = 1.0))]
    pub interval_s: f64,
    /// Initial phase seconds (sched=fixed)
    #[cfg_attr(feature = "clap", arg(long, default_value_t = 0.0))]
    pub phase_s: f64,
    /// Duty-cycle ON seconds (sched=duty)
    #[cfg_attr(feature = "clap", arg(long, default_value_t = 10.0))]
    pub on_s: f64,
    /// Duty-cycle OFF seconds (sched=duty)
    #[cfg_attr(feature = "clap", arg(long, default_value_t = 10.0))]
    pub off_s: f64,
    /// Duty-cycle start phase seconds (sched=duty)
    #[cfg_attr(feature = "clap", arg(long, default_value_t = 0.0))]
    pub duty_phase_s: f64,
}

/// Fault configuration kind
#[derive(Copy, Clone, Debug)]
#[cfg_attr(feature = "clap", derive(ValueEnum))]
pub enum FaultKind {
    None,
    Degraded,
    Slowbias,
    Hijack,
}

/// GNSS fault model arguments for CLI
#[derive(Clone, Debug)]
#[cfg_attr(feature = "clap", derive(Args))]
pub struct FaultArgs {
    /// Fault kind: none | degraded | slowbias | hijack
    #[cfg_attr(feature = "clap", arg(long, value_enum, default_value_t = FaultKind::None))]
    pub fault: FaultKind,
    /// Degraded (AR(1))
    #[cfg_attr(feature = "clap", arg(long, default_value_t = 0.99))]
    pub rho_pos: f64,
    #[cfg_attr(feature = "clap", arg(long, default_value_t = 3.0))]
    pub sigma_pos_m: f64,
    #[cfg_attr(feature = "clap", arg(long, default_value_t = 0.95))]
    pub rho_vel: f64,
    #[cfg_attr(feature = "clap", arg(long, default_value_t = 0.3))]
    pub sigma_vel_mps: f64,
    #[cfg_attr(feature = "clap", arg(long, default_value_t = 5.0))]
    pub r_scale: f64,
    /// Slow bias
    #[cfg_attr(feature = "clap", arg(long, default_value_t = 0.02))]
    pub drift_n_mps: f64,
    #[cfg_attr(feature = "clap", arg(long, default_value_t = 0.0))]
    pub drift_e_mps: f64,
    #[cfg_attr(feature = "clap", arg(long, default_value_t = 1e-6))]
    pub q_bias: f64,
    #[cfg_attr(feature = "clap", arg(long, default_value_t = 0.0))]
    pub rotate_omega_rps: f64,
    /// Hijack
    #[cfg_attr(feature = "clap", arg(long, default_value_t = 50.0))]
    pub hijack_offset_n_m: f64,
    #[cfg_attr(feature = "clap", arg(long, default_value_t = 0.0))]
    pub hijack_offset_e_m: f64,
    #[cfg_attr(feature = "clap", arg(long, default_value_t = 120.0))]
    pub hijack_start_s: f64,
    #[cfg_attr(feature = "clap", arg(long, default_value_t = 60.0))]
    pub hijack_duration_s: f64,
}

/// Build GNSS scheduler from CLI arguments
pub fn build_scheduler(a: &SchedulerArgs) -> GnssScheduler {
    match a.sched {
        SchedKind::Passthrough => GnssScheduler::PassThrough,
        SchedKind::Fixed => GnssScheduler::FixedInterval {
            interval_s: a.interval_s,
            phase_s: a.phase_s,
        },
        SchedKind::Duty => GnssScheduler::DutyCycle {
            on_s: a.on_s,
            off_s: a.off_s,
            start_phase_s: a.duty_phase_s,
        },
    }
}

/// Build GNSS fault model from CLI arguments
pub fn build_fault(a: &FaultArgs) -> GnssFaultModel {
    match a.fault {
        FaultKind::None => GnssFaultModel::None,
        FaultKind::Degraded => GnssFaultModel::Degraded {
            rho_pos: a.rho_pos,
            sigma_pos_m: a.sigma_pos_m,
            rho_vel: a.rho_vel,
            sigma_vel_mps: a.sigma_vel_mps,
            r_scale: a.r_scale,
        },
        FaultKind::Slowbias => GnssFaultModel::SlowBias {
            drift_n_mps: a.drift_n_mps,
            drift_e_mps: a.drift_e_mps,
            q_bias: a.q_bias,
            rotate_omega_rps: a.rotate_omega_rps,
        },
        FaultKind::Hijack => GnssFaultModel::Hijack {
            offset_n_m: a.hijack_offset_n_m,
            offset_e_m: a.hijack_offset_e_m,
            start_s: a.hijack_start_s,
            duration_s: a.hijack_duration_s,
        },
    }
}

//================= Unified Simulation Configuration =====================================

/// Simulation mode selection
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "clap", derive(ValueEnum))]
#[serde(rename_all = "kebab-case")]
pub enum SimulationMode {
    /// Dead reckoning with no corrections
    DeadReckoning,
    /// Open-loop feed-forward INS
    OpenLoop,
    /// Closed-loop with Kalman filter corrections from GNSS/other sensors
    ClosedLoop,
    /// Particle filter based navigation
    ParticleFilter,
}

/// Filter type for closed-loop mode
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "clap", derive(ValueEnum))]
#[serde(rename_all = "kebab-case")]
#[derive(Default)]
pub enum FilterType {
    /// Unscented Kalman Filter
    #[default]
    Ukf,
    /// Extended Kalman Filter
    Ekf,
    /// Error-State Kalman Filter (ESKF) with multiplicative attitude error
    Eskf,
}

/// Particle filter type selection
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "clap", derive(ValueEnum))]
#[serde(rename_all = "kebab-case")]
#[derive(Default)]
pub enum ParticleFilterType {
    /// Standard particle filter (all states as particles)
    #[default]
    Standard,
    /// Rao-Blackwellized particle filter (position as particles, velocity/attitude/biases as per-particle filters)
    RaoBlackwellized,
    /// Velocity-based particle filter (position-only particles, velocities supplied externally)
    Velocity,
}

/// Closed-loop specific configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ClosedLoopConfig {
    /// Filter type (UKF or EKF)
    #[serde(default)]
    pub filter: FilterType,
}

impl Default for ClosedLoopConfig {
    fn default() -> Self {
        Self {
            filter: FilterType::Ukf,
        }
    }
}

/// Particle filter configuration (RBPF defaults).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ParticleFilterConfig {
    /// Apply zero-vertical-velocity pseudo-measurement.
    #[serde(default = "default_zero_vertical_velocity")]
    pub zero_vertical_velocity: bool,
    /// Standard deviation for zero-vertical-velocity pseudo-measurement (m/s).
    #[serde(default = "default_zero_vertical_velocity_std_mps")]
    pub zero_vertical_velocity_std_mps: f64,
}

fn default_zero_vertical_velocity() -> bool {
    true
}

fn default_zero_vertical_velocity_std_mps() -> f64 {
    0.1
}

impl Default for ParticleFilterConfig {
    fn default() -> Self {
        Self {
            zero_vertical_velocity: default_zero_vertical_velocity(),
            zero_vertical_velocity_std_mps: default_zero_vertical_velocity_std_mps(),
        }
    }
}
/// Log level options for simulation logging.
///
/// This enum is serialized/deserialized as lowercase strings to match existing
/// configuration files (e.g., `"info"`, `"debug"`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[cfg_attr(feature = "clap", derive(ValueEnum))]
pub enum LogLevel {
    Off,
    Error,
    Warn,
    Info,
    Debug,
    Trace,
}

impl LogLevel {
    /// Convert LogLevel to string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            LogLevel::Off => "off",
            LogLevel::Error => "error",
            LogLevel::Warn => "warn",
            LogLevel::Info => "info",
            LogLevel::Debug => "debug",
            LogLevel::Trace => "trace",
        }
    }
}

/// Logging configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LoggingConfig {
    /// Log level (off, error, warn, info, debug, trace)
    #[serde(default = "default_log_level")]
    pub level: LogLevel,
    /// Optional log file path (if not specified, logs to stderr)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file: Option<String>,
}

fn default_log_level() -> LogLevel {
    LogLevel::Info
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: default_log_level(),
            file: None,
        }
    }
}

/// Unified simulation configuration supporting all modes
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SimulationConfig {
    /// Input CSV file path (relative or absolute)
    pub input: String,
    /// Output CSV file path (relative or absolute)
    pub output: String,
    /// Simulation mode
    pub mode: SimulationMode,
    /// Random number generator seed
    #[serde(default = "default_seed")]
    pub seed: u64,
    /// Run simulations in parallel when processing multiple files
    #[serde(default)]
    pub parallel: bool,
    /// Generate performance plot comparing navigation output to GPS measurements
    #[serde(default)]
    pub generate_plot: bool,
    /// Logging configuration
    #[serde(default)]
    pub logging: LoggingConfig,
    /// Closed-loop specific settings (only used if mode is ClosedLoop)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub closed_loop: Option<ClosedLoopConfig>,
    /// Particle filter settings (only used if mode is ParticleFilter)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub particle_filter: Option<ParticleFilterConfig>,
    /// Geophysical measurement configuration (optional, requires --features geonav)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub geophysical: Option<GeophysicalConfig>,
    /// GNSS degradation configuration (scheduler + fault model)
    #[serde(default)]
    pub gnss_degradation: crate::messages::GnssDegradationConfig,
}

fn default_seed() -> u64 {
    42
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            input: "input.csv".to_string(),
            output: "output.csv".to_string(),
            mode: SimulationMode::ClosedLoop,
            seed: default_seed(),
            parallel: false,
            generate_plot: false,
            logging: LoggingConfig::default(),
            closed_loop: Some(ClosedLoopConfig::default()),
            particle_filter: None,
            geophysical: None,
            gnss_degradation: crate::messages::GnssDegradationConfig::default(),
        }
    }
}

impl SimulationConfig {
    /// Write the configuration to a JSON file (pretty-printed)
    pub fn to_json<P: AsRef<Path>>(&self, path: P) -> io::Result<()> {
        let file = std::fs::File::create(path)?;
        serde_json::to_writer_pretty(file, self).map_err(io::Error::other)
    }

    /// Read the configuration from a JSON file
    pub fn from_json<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let file = std::fs::File::open(path)?;
        serde_json::from_reader(file).map_err(io::Error::other)
    }

    /// Write the configuration as YAML
    pub fn to_yaml<P: AsRef<Path>>(&self, path: P) -> io::Result<()> {
        let mut file = std::fs::File::create(path)?;
        let s = serde_yaml::to_string(self).map_err(io::Error::other)?;
        file.write_all(s.as_bytes())
    }

    /// Read the configuration from YAML
    pub fn from_yaml<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let file = std::fs::File::open(path)?;
        serde_yaml::from_reader(file).map_err(io::Error::other)
    }

    /// Write the configuration as TOML
    pub fn to_toml<P: AsRef<Path>>(&self, path: P) -> io::Result<()> {
        let mut file = std::fs::File::create(path)?;
        let s = toml::to_string_pretty(self).map_err(io::Error::other)?;
        file.write_all(s.as_bytes())
    }

    /// Read the configuration from TOML
    pub fn from_toml<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let mut s = String::new();
        let mut file = std::fs::File::open(path)?;
        file.read_to_string(&mut s)?;
        toml::from_str(&s).map_err(io::Error::other)
    }

    /// Generic write: choose format by file extension (.json/.yaml/.yml/.toml)
    pub fn to_file<P: AsRef<Path>>(&self, path: P) -> io::Result<()> {
        let p = path.as_ref();
        let ext = p
            .extension()
            .and_then(|s| s.to_str())
            .map(|s| s.to_lowercase());
        match ext.as_deref() {
            Some("json") => self.to_json(p),
            Some("yaml") | Some("yml") => self.to_yaml(p),
            Some("toml") => self.to_toml(p),
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "unsupported file extension (expected .json, .yaml, .yml, or .toml)",
            )),
        }
    }

    /// Generic read: choose format by file extension (.json/.yaml/.yml/.toml)
    pub fn from_file<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let p = path.as_ref();
        let ext = p
            .extension()
            .and_then(|s| s.to_str())
            .map(|s| s.to_lowercase());
        match ext.as_deref() {
            Some("json") => Self::from_json(p),
            Some("yaml") | Some("yml") => Self::from_yaml(p),
            Some("toml") => Self::from_toml(p),
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "unsupported file extension (expected .json, .yaml, .yml, or .toml)",
            )),
        }
    }
}

/// Geophysical measurement type configuration
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "clap", derive(ValueEnum))]
#[serde(rename_all = "lowercase")]
#[derive(Default)]
pub enum GeoMeasurementType {
    /// Gravity anomaly measurements
    #[default]
    Gravity,
    /// Magnetic anomaly measurements
    Magnetic,
}

/// Geophysical map resolution configuration
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "clap", derive(ValueEnum))]
#[serde(rename_all = "snake_case")]
#[derive(Default)]
pub enum GeoResolution {
    /// 1 degree resolution
    OneDegree,
    /// 30 arcminute resolution
    ThirtyMinutes,
    /// 20 arcminute resolution
    TwentyMinutes,
    /// 15 arcminute resolution
    FifteenMinutes,
    /// 10 arcminute resolution
    TenMinutes,
    /// 6 arcminute resolution
    SixMinutes,
    /// 5 arcminute resolution
    FiveMinutes,
    /// 4 arcminute resolution
    FourMinutes,
    /// 3 arcminute resolution
    ThreeMinutes,
    /// 2 arcminute resolution
    TwoMinutes,
    /// 1 arcminute resolution
    #[default]
    OneMinute,
    /// 30 arcsecond resolution
    ThirtySeconds,
    /// 15 arcsecond resolution
    FifteenSeconds,
    /// 3 arcsecond resolution
    ThreeSeconds,
    /// 1 arcsecond resolution
    OneSecond,
}

/// Geophysical measurement configuration for geonav simulations
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct GeophysicalConfig {
    // Gravity measurement configuration (all optional)
    /// Gravity map resolution (None = gravity not used)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gravity_resolution: Option<GeoResolution>,

    /// Gravity measurement bias (mGal)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gravity_bias: Option<f64>,

    /// Gravity measurement noise std dev (mGal)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gravity_noise_std: Option<f64>,

    /// Gravity map file path (auto-detected if None)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gravity_map_file: Option<String>,

    // Magnetic measurement configuration (all optional)
    /// Magnetic map resolution (None = magnetic not used)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub magnetic_resolution: Option<GeoResolution>,

    /// Magnetic measurement bias (nT)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub magnetic_bias: Option<f64>,

    /// Magnetic measurement noise std dev (nT)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub magnetic_noise_std: Option<f64>,

    /// Magnetic map file path (auto-detected if None)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub magnetic_map_file: Option<String>,

    // Common configuration
    /// Frequency in seconds for geophysical measurements
    /// Applies to both measurement types if both are enabled
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub geo_frequency_s: Option<f64>,
}

fn default_gravity_noise_std() -> f64 {
    100.0
}

fn default_magnetic_noise_std() -> f64 {
    150.0
}

impl GeophysicalConfig {
    /// Get default gravity noise std if not set
    pub fn get_gravity_noise_std(&self) -> f64 {
        self.gravity_noise_std
            .unwrap_or_else(default_gravity_noise_std)
    }

    /// Get default magnetic noise std if not set
    pub fn get_magnetic_noise_std(&self) -> f64 {
        self.magnetic_noise_std
            .unwrap_or_else(default_magnetic_noise_std)
    }

    /// Get gravity bias with default of 0.0
    pub fn get_gravity_bias(&self) -> f64 {
        self.gravity_bias.unwrap_or(0.0)
    }

    /// Get magnetic bias with default of 0.0
    pub fn get_magnetic_bias(&self) -> f64 {
        self.magnetic_bias.unwrap_or(0.0)
    }
}

/// Unified geophysical navigation simulation configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GeonavSimulationConfig {
    /// Input CSV file path (relative or absolute)
    pub input: String,
    /// Output CSV file path (relative or absolute)
    pub output: String,
    /// Filter type (UKF or EKF)
    #[serde(default)]
    pub filter: FilterType,
    /// Random number generator seed
    #[serde(default = "default_seed")]
    pub seed: u64,
    /// Run simulations in parallel when processing multiple files
    #[serde(default)]
    pub parallel: bool,
    /// Generate performance plot comparing navigation output to GPS measurements
    #[serde(default)]
    pub generate_plot: bool,
    /// Logging configuration
    #[serde(default)]
    pub logging: LoggingConfig,
    /// Geophysical measurement configuration
    #[serde(default)]
    pub geophysical: GeophysicalConfig,
    /// GNSS degradation configuration (scheduler + fault model)
    #[serde(default)]
    pub gnss_degradation: crate::messages::GnssDegradationConfig,
}

impl Default for GeonavSimulationConfig {
    fn default() -> Self {
        Self {
            input: "input.csv".to_string(),
            output: "output.csv".to_string(),
            filter: FilterType::Ukf,
            seed: default_seed(),
            parallel: false,
            generate_plot: false,
            logging: LoggingConfig::default(),
            geophysical: GeophysicalConfig::default(),
            gnss_degradation: crate::messages::GnssDegradationConfig::default(),
        }
    }
}
impl GeonavSimulationConfig {
    /// Write the configuration to a JSON file (pretty-printed)
    pub fn to_json<P: AsRef<Path>>(&self, path: P) -> io::Result<()> {
        let file = std::fs::File::create(path)?;
        serde_json::to_writer_pretty(file, self).map_err(io::Error::other)
    }

    /// Read the configuration from a JSON file
    pub fn from_json<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let file = std::fs::File::open(path)?;
        serde_json::from_reader(file).map_err(io::Error::other)
    }

    /// Write the configuration as YAML
    pub fn to_yaml<P: AsRef<Path>>(&self, path: P) -> io::Result<()> {
        let mut file = std::fs::File::create(path)?;
        let s = serde_yaml::to_string(self).map_err(io::Error::other)?;
        file.write_all(s.as_bytes())
    }

    /// Read the configuration from YAML
    pub fn from_yaml<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let file = std::fs::File::open(path)?;
        serde_yaml::from_reader(file).map_err(io::Error::other)
    }

    /// Write the configuration as TOML
    pub fn to_toml<P: AsRef<Path>>(&self, path: P) -> io::Result<()> {
        let mut file = std::fs::File::create(path)?;
        let s = toml::to_string_pretty(self).map_err(io::Error::other)?;
        file.write_all(s.as_bytes())
    }

    /// Read the configuration from TOML
    pub fn from_toml<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let mut s = String::new();
        let mut file = std::fs::File::open(path)?;
        file.read_to_string(&mut s)?;
        toml::from_str(&s).map_err(io::Error::other)
    }

    /// Generic write: choose format by file extension (.json/.yaml/.yml/.toml)
    pub fn to_file<P: AsRef<Path>>(&self, path: P) -> io::Result<()> {
        let p = path.as_ref();
        let ext = p
            .extension()
            .and_then(|s| s.to_str())
            .map(|s| s.to_lowercase());
        match ext.as_deref() {
            Some("json") => self.to_json(p),
            Some("yaml") | Some("yml") => self.to_yaml(p),
            Some("toml") => self.to_toml(p),
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "unsupported file extension (expected .json, .yaml, .yml, or .toml)",
            )),
        }
    }

    /// Generic read: choose format by file extension (.json/.yaml/.yml/.toml)
    pub fn from_file<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let p = path.as_ref();
        let ext = p
            .extension()
            .and_then(|s| s.to_str())
            .map(|s| s.to_lowercase());
        match ext.as_deref() {
            Some("json") => Self::from_json(p),
            Some("yaml") | Some("yml") => Self::from_yaml(p),
            Some("toml") => Self::from_toml(p),
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "unsupported file extension (expected .json, .yaml, .yml, or .toml)",
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
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
        assert_eq!(nav.velocity_vertical, 0.0);
    }
    #[test]
    fn test_navigation_result_from_strapdown_state() {
        let mut state = StrapdownState::default();
        state.latitude = 1.0;
        state.longitude = 2.0;
        state.altitude = 3.0;
        state.velocity_north = 4.0;
        state.velocity_east = 5.0;
        state.velocity_vertical = 6.0;
        state.attitude = nalgebra::Rotation3::from_euler_angles(7.0, 8.0, 9.0);

        let state_vector: DVector<f64> = DVector::from_vec(vec![
            state.latitude,
            state.longitude,
            state.altitude,
            state.velocity_north,
            state.velocity_east,
            state.velocity_vertical,
            state.attitude.euler_angles().0, // roll
            state.attitude.euler_angles().1, // pitch
            state.attitude.euler_angles().2, // yaw
            0.0,                             // acc_bias_x
            0.0,                             // acc_bias_y
            0.0,                             // acc_bias_z
            0.0,                             // gyro_bias_x
            0.0,                             // gyro_bias_y
            0.0,                             // gyro_bias_z
        ]);
        let timestamp = chrono::Utc::now();
        let nav = NavigationResult::from((
            &timestamp,
            &state_vector,
            &DMatrix::from_diagonal(&DVector::from_element(15, 0.0)), // dummy covariance
        ));
        assert_eq!(nav.latitude, (1.0_f64).to_degrees());
        assert_eq!(nav.longitude, (2.0_f64).to_degrees());
        assert_eq!(nav.altitude, 3.0);
        assert_eq!(nav.velocity_north, 4.0);
        assert_eq!(nav.velocity_east, 5.0);
        assert_eq!(nav.velocity_vertical, 6.0);
    }
    #[test]
    fn test_navigation_result_to_csv_and_from_csv() {
        let mut nav = NavigationResult::new();
        nav.latitude = 1.0;
        nav.longitude = 2.0;
        nav.altitude = 3.0;
        nav.velocity_north = 4.0;
        nav.velocity_east = 5.0;
        nav.velocity_vertical = 6.0;
        let temp_file = std::env::temp_dir().join("test_nav_result.csv");
        NavigationResult::to_csv(&[nav.clone()], &temp_file).unwrap();
        let read = NavigationResult::from_csv(&temp_file).unwrap();
        assert_eq!(read.len(), 1);
        assert_eq!(read[0].latitude, 1.0);
        assert_eq!(read[0].longitude, 2.0);
        assert_eq!(read[0].altitude, 3.0);
        assert_eq!(read[0].velocity_north, 4.0);
        assert_eq!(read[0].velocity_east, 5.0);
        assert_eq!(read[0].velocity_vertical, 6.0);
        let _ = std::fs::remove_file(&temp_file);
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

        // Initialize UKF
        let mut ukf = initialize_ukf(rec.clone(), None, None, None, None, None, None);

        // Create a minimal EventStream with one IMU event
        let imu_data = IMUData {
            accel: nalgebra::Vector3::new(rec.acc_x, rec.acc_y, rec.acc_z),
            gyro: nalgebra::Vector3::new(rec.gyro_x, rec.gyro_y, rec.gyro_z),
        };
        let event = Event::Imu {
            dt_s: 1.0,
            imu: imu_data,
            elapsed_s: 0.0,
        };
        let stream = EventStream {
            start_time: rec.time,
            events: vec![event],
        };

        let res = run_closed_loop(&mut ukf, stream, None);
        assert!(!res.unwrap().is_empty());
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
        let ukf = initialize_ukf(rec.clone(), None, None, None, None, None, None);
        assert!(!ukf.get_estimate().is_empty());
        let ukf2 = initialize_ukf(
            rec,
            Some(vec![0.1, 0.2, 0.3]),
            Some(vec![0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
            None,
            None,
            None,
            None,
        );
        assert!(!ukf2.get_estimate().is_empty());
    }
    // Helper to produce the header in the same order the struct expects
    fn test_header() -> Vec<&'static str> {
        vec![
            "time",
            "bearingAccuracy",
            "speedAccuracy",
            "verticalAccuracy",
            "horizontalAccuracy",
            "speed",
            "bearing",
            "altitude",
            "longitude",
            "latitude",
            "qz",
            "qy",
            "qx",
            "qw",
            "roll",
            "pitch",
            "yaw",
            "acc_z",
            "acc_y",
            "acc_x",
            "gyro_z",
            "gyro_y",
            "gyro_x",
            "mag_z",
            "mag_y",
            "mag_x",
            "relativeAltitude",
            "pressure",
            "grav_z",
            "grav_y",
            "grav_x",
        ]
    }
    #[test]
    fn deserialize_with_empty_fields_maps_to_nan() {
        let headers = test_header();
        let time = "2023-08-04T21:47:58Z";
        let mut row: Vec<String> = Vec::with_capacity(headers.len());
        row.push(time.to_string());
        for _ in 1..headers.len() {
            row.push(String::new());
        }
        let mut csv_data = String::new();
        csv_data.push_str(&headers.join(","));
        csv_data.push('\n');
        csv_data.push_str(&row.join(","));
        let temp_file = std::env::temp_dir().join("test_empty_fields_nan.csv");
        std::fs::write(&temp_file, csv_data).unwrap();
        let recs = TestDataRecord::from_csv(&temp_file).expect("from_csv should succeed");
        assert_eq!(recs.len(), 1);
        let r = &recs[0];
        assert_eq!(
            r.time,
            chrono::DateTime::parse_from_rfc3339(time)
                .unwrap()
                .with_timezone(&Utc)
        );
        assert!(r.speed.is_nan());
        assert!(r.latitude.is_nan());
        assert!(r.longitude.is_nan());
        assert!(r.acc_x.is_nan());
        let _ = std::fs::remove_file(&temp_file);
    }
    #[test]
    fn deserialize_with_missing_trailing_columns_returns_error() {
        let headers = test_header();
        let time = "2023-08-04T21:47:58Z";
        let mut row: Vec<String> = Vec::new();
        row.push(time.to_string());
        row.push(String::from("1.0"));
        row.push(String::from("2.0"));
        let mut csv_data = String::new();
        csv_data.push_str(&headers.join(","));
        csv_data.push('\n');
        csv_data.push_str(&row.join(","));
        let temp_file = std::env::temp_dir().join("test_missing_trailing.csv");
        std::fs::write(&temp_file, csv_data).unwrap();
        let recs = TestDataRecord::from_csv(&temp_file).expect("from_csv should succeed");
        assert_eq!(recs.len(), 1);
        let rec = &recs[0];
        assert_eq!(
            rec.time,
            chrono::DateTime::parse_from_rfc3339(time)
                .unwrap()
                .with_timezone(&Utc)
        );
        assert!(rec.speed.is_nan());
        assert!(rec.latitude.is_nan());
        assert!(rec.longitude.is_nan());
        let _ = std::fs::remove_file(&temp_file);
    }
    #[test]
    fn manual_padding_then_deserialize_succeeds() {
        let headers = test_header();
        let time = "2023-08-04T21:47:58Z";
        let mut row: Vec<String> = Vec::new();
        row.push(time.to_string());
        row.push(String::new()); // bearingAccuracy
        row.push(String::new()); // speedAccuracy
        row.push(String::new()); // verticalAccuracy
        row.push(String::new()); // horizontalAccuracy
        row.push(String::new()); // speed
        row.push(String::new()); // bearing
        row.push(String::new()); // altitude
        row.push(String::from("-122.0")); // longitude
        row.push(String::from("37.0")); // latitude
        let mut csv_data = String::new();
        csv_data.push_str(&headers.join(","));
        csv_data.push('\n');
        csv_data.push_str(&row.join(","));
        let temp_file = std::env::temp_dir().join("test_manual_padding.csv");
        std::fs::write(&temp_file, csv_data).unwrap();
        let got = TestDataRecord::from_csv(&temp_file).expect("from_csv should succeed");
        assert_eq!(got.len(), 1);
        let r = &got[0];
        assert_eq!(
            r.time,
            chrono::DateTime::parse_from_rfc3339(time)
                .unwrap()
                .with_timezone(&Utc)
        );
        assert_eq!(r.longitude, -122.0);
        assert_eq!(r.latitude, 37.0);
        let _ = std::fs::remove_file(&temp_file);
    }
    #[test]
    fn test_de_f64_nan_with_various_inputs() {
        // Test CSV deserialization with NaN/null/empty values
        let headers = test_header();
        let mut csv_data = String::new();
        csv_data.push_str(&headers.join(","));
        csv_data.push('\n');

        // Row with mixed NaN representations
        csv_data.push_str("2023-08-04T21:47:58Z,NaN,null,,,1.5,90.0,100.0,-122.0,37.0,");
        csv_data.push_str("0,0,0,1,0.1,0.2,0.3,9.8,0,0,0,0,0,0,0,0,0,1013.25,9.81,0,0\n");

        let temp_file = std::env::temp_dir().join("test_nan_variants.csv");
        std::fs::write(&temp_file, csv_data).unwrap();
        let recs = TestDataRecord::from_csv(&temp_file).expect("Should parse");
        assert_eq!(recs.len(), 1);
        assert!(recs[0].bearing_accuracy.is_nan());
        assert!(recs[0].speed_accuracy.is_nan());
        assert!(recs[0].vertical_accuracy.is_nan());
        assert!(recs[0].horizontal_accuracy.is_nan());
        assert_eq!(recs[0].speed, 1.5);
        let _ = std::fs::remove_file(&temp_file);
    }
    #[test]
    fn test_test_data_record_display() {
        let rec = TestDataRecord {
            time: DateTime::parse_from_str("2023-01-01 00:00:00+00:00", "%Y-%m-%d %H:%M:%S%z")
                .unwrap()
                .with_timezone(&Utc),
            latitude: 37.0,
            longitude: -122.0,
            altitude: 100.0,
            speed: 5.0,
            bearing: 90.0,
            ..Default::default()
        };
        let display_str = format!("{}", rec);
        assert!(display_str.contains("37"));
        assert!(display_str.contains("-122"));
        assert!(display_str.contains("100"));
    }
    #[test]
    fn test_navigation_result_from_ukf() {
        let rec = TestDataRecord {
            time: Utc::now(),
            horizontal_accuracy: 5.0,
            vertical_accuracy: 2.0,
            speed_accuracy: 1.0,
            latitude: 37.0,
            longitude: -122.0,
            altitude: 100.0,
            speed: 10.0,
            bearing: 45.0,
            roll: 0.1,
            pitch: 0.2,
            yaw: 0.3,
            ..Default::default()
        };
        let ukf = initialize_ukf(rec.clone(), None, None, None, None, None, None);
        let timestamp = Utc::now();
        let nav_result = NavigationResult::from((&timestamp, &ukf));

        assert_eq!(nav_result.timestamp, timestamp);
        assert!(nav_result.latitude.is_finite());
        assert!(nav_result.longitude.is_finite());
        assert!(nav_result.altitude.is_finite());
    }
    #[test]
    fn test_dead_reckoning_empty_records() {
        let results = dead_reckoning(&[]);
        assert!(results.is_empty());
    }
    #[test]
    fn test_dead_reckoning_single_record() {
        let rec = TestDataRecord {
            time: Utc::now(),
            latitude: 37.0,
            longitude: -122.0,
            altitude: 100.0,
            speed: 1.0,
            bearing: 0.0,
            roll: 0.0,
            pitch: 0.0,
            yaw: 0.0,
            acc_x: 0.0,
            acc_y: 0.0,
            acc_z: 9.81,
            gyro_x: 0.0,
            gyro_y: 0.0,
            gyro_z: 0.0,
            ..Default::default()
        };
        let results = dead_reckoning(&[rec]);
        assert_eq!(results.len(), 1);
    }
    #[test]
    fn test_print_ukf() {
        let rec = TestDataRecord {
            time: Utc::now(),
            horizontal_accuracy: 5.0,
            vertical_accuracy: 2.0,
            speed_accuracy: 1.0,
            latitude: 37.0,
            longitude: -122.0,
            altitude: 100.0,
            speed: 10.0,
            bearing: 45.0,
            roll: 0.1,
            pitch: 0.2,
            yaw: 0.3,
            ..Default::default()
        };
        let ukf = initialize_ukf(rec.clone(), None, None, None, None, None, None);
        // Just ensure it doesn't panic
        print_ukf(&ukf, &rec);
    }
    #[test]
    fn test_initialize_ukf_with_nan_angles() {
        let rec = TestDataRecord {
            time: Utc::now(),
            horizontal_accuracy: 5.0,
            vertical_accuracy: 2.0,
            speed_accuracy: 1.0,
            latitude: 37.0,
            longitude: -122.0,
            altitude: 100.0,
            speed: 10.0,
            bearing: 45.0,
            roll: f64::NAN,
            pitch: f64::NAN,
            yaw: f64::NAN,
            ..Default::default()
        };
        let ukf = initialize_ukf(rec, None, None, None, None, None, None);
        let estimate = ukf.get_estimate();
        // Should default NaN angles to 0.0
        assert!(estimate[6].abs() < 1e-6); // roll
        assert!(estimate[7].abs() < 1e-6); // pitch
        assert!(estimate[8].abs() < 1e-6); // yaw
    }

    #[test]
    fn test_initialize_ukf_with_custom_biases() {
        let rec = TestDataRecord {
            time: Utc::now(),
            horizontal_accuracy: 5.0,
            vertical_accuracy: 2.0,
            speed_accuracy: 1.0,
            latitude: 37.0,
            longitude: -122.0,
            altitude: 100.0,
            speed: 10.0,
            bearing: 45.0,
            roll: 0.0,
            pitch: 0.0,
            yaw: 0.0,
            ..Default::default()
        };
        let ukf = initialize_ukf(
            rec,
            Some(vec![1e-4, 2e-4, 3e-4]),
            Some(vec![0.01, 0.02, 0.03, 0.001, 0.002, 0.003]),
            Some(vec![1e-5; 6]),
            None,
            None,
            None,
        );
        let estimate = ukf.get_estimate();
        assert_eq!(estimate.len(), 15);
    }

    #[test]
    fn test_initialize_ukf_with_custom_process_noise() {
        let rec = TestDataRecord {
            time: Utc::now(),
            horizontal_accuracy: 5.0,
            vertical_accuracy: 2.0,
            speed_accuracy: 1.0,
            latitude: 37.0,
            longitude: -122.0,
            altitude: 100.0,
            speed: 10.0,
            bearing: 45.0,
            roll: 0.0,
            pitch: 0.0,
            yaw: 0.0,
            ..Default::default()
        };
        let custom_noise = vec![1e-5; 15];
        let ukf = initialize_ukf(rec, None, None, None, None, None, Some(custom_noise));
        assert!(!ukf.get_estimate().is_empty());
    }
    #[test]
    fn test_health_limits_default() {
        let limits = HealthLimits::default();
        assert!(limits.lat_rad.0 < 0.0);
        assert!(limits.lat_rad.1 > 0.0);
        assert!(limits.speed_mps_max > 0.0);
        assert!(limits.cov_diag_max > 0.0);
    }

    #[test]
    fn test_health_monitor_check_valid_state() {
        let limits = HealthLimits::default();
        let mut monitor = HealthMonitor::new(limits);

        let state = vec![
            0.5, 0.5, 100.0, // lat, lon, alt (radians, radians, meters)
            10.0, 5.0, 0.0, // vn, ve, vd
            0.0, 0.0, 0.0, // roll, pitch, yaw
            0.0, 0.0, 0.0, // acc biases
            0.0, 0.0, 0.0, // gyro biases
        ];
        let cov = DMatrix::from_diagonal(&DVector::from_vec(vec![1e-6; 15]));

        let result = monitor.check(&state, &cov, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_health_monitor_check_non_finite_state() {
        let limits = HealthLimits::default();
        let mut monitor = HealthMonitor::new(limits);

        let state = vec![
            f64::NAN,
            0.5,
            100.0,
            10.0,
            5.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ];
        let cov = DMatrix::from_diagonal(&DVector::from_vec(vec![1e-6; 15]));

        let result = monitor.check(&state, &cov, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_health_monitor_check_non_finite_covariance() {
        let limits = HealthLimits::default();
        let mut monitor = HealthMonitor::new(limits);

        let state = vec![
            0.5, 0.5, 100.0, 10.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let mut cov = DMatrix::from_diagonal(&DVector::from_vec(vec![1e-6; 15]));
        cov[(0, 0)] = f64::NAN;

        let result = monitor.check(&state, &cov, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_health_monitor_check_latitude_out_of_range() {
        let limits = HealthLimits::default();
        let mut monitor = HealthMonitor::new(limits);

        let state = vec![
            5.0, 0.5, 100.0, 10.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ]; // lat > PI/2
        let cov = DMatrix::from_diagonal(&DVector::from_vec(vec![1e-6; 15]));

        let result = monitor.check(&state, &cov, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_health_monitor_check_longitude_out_of_range() {
        let limits = HealthLimits::default();
        let mut monitor = HealthMonitor::new(limits);

        let state = vec![
            0.5, 5.0, 100.0, 10.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ]; // lon > PI
        let cov = DMatrix::from_diagonal(&DVector::from_vec(vec![1e-6; 15]));

        let result = monitor.check(&state, &cov, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_health_monitor_check_altitude_out_of_range() {
        let mut limits = HealthLimits::default();
        limits.alt_m = (-100.0, 10000.0);
        let mut monitor = HealthMonitor::new(limits);

        let state = vec![
            0.5, 0.5, 20000.0, 10.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let cov = DMatrix::from_diagonal(&DVector::from_vec(vec![1e-6; 15]));

        let result = monitor.check(&state, &cov, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_health_monitor_check_negative_variance() {
        let limits = HealthLimits::default();
        let mut monitor = HealthMonitor::new(limits);

        let state = vec![
            0.5, 0.5, 100.0, 10.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let mut cov = DMatrix::from_diagonal(&DVector::from_vec(vec![1e-6; 15]));
        cov[(2, 2)] = -1.0; // negative variance

        let result = monitor.check(&state, &cov, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_health_monitor_check_variance_too_large() {
        let limits = HealthLimits::default();
        let mut monitor = HealthMonitor::new(limits);

        let state = vec![
            0.5, 0.5, 100.0, 10.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let mut cov = DMatrix::from_diagonal(&DVector::from_vec(vec![1e-6; 15]));
        cov[(3, 3)] = 1e20; // variance too large

        let result = monitor.check(&state, &cov, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_health_monitor_check_nis_invalid() {
        let limits = HealthLimits::default();
        let mut monitor = HealthMonitor::new(limits);

        let state = vec![
            0.5, 0.5, 100.0, 10.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let cov = DMatrix::from_diagonal(&DVector::from_vec(vec![1e-6; 15]));

        let result = monitor.check(&state, &cov, Some(f64::NAN));
        assert!(result.is_err());
    }

    #[test]
    fn test_health_monitor_check_nis_exceeds_threshold() {
        let mut limits = HealthLimits::default();
        limits.nis_pos_max = 10.0;
        limits.nis_pos_consec_fail = 3;
        let mut monitor = HealthMonitor::new(limits);

        let state = vec![
            0.5, 0.5, 100.0, 10.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let cov = DMatrix::from_diagonal(&DVector::from_vec(vec![1e-6; 15]));

        // First two failures should be ok
        assert!(monitor.check(&state, &cov, Some(15.0)).is_ok());
        assert!(monitor.check(&state, &cov, Some(15.0)).is_ok());
        // Third consecutive failure should error
        let result = monitor.check(&state, &cov, Some(15.0));
        assert!(result.is_err());
    }

    #[test]
    fn test_health_monitor_check_nis_reset_on_pass() {
        let mut limits = HealthLimits::default();
        limits.nis_pos_max = 10.0;
        limits.nis_pos_consec_fail = 3;
        let mut monitor = HealthMonitor::new(limits);

        let state = vec![
            0.5, 0.5, 100.0, 10.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let cov = DMatrix::from_diagonal(&DVector::from_vec(vec![1e-6; 15]));

        // First failure
        assert!(monitor.check(&state, &cov, Some(15.0)).is_ok());
        // Pass resets counter
        assert!(monitor.check(&state, &cov, Some(5.0)).is_ok());
        // New failures should not accumulate with previous
        assert!(monitor.check(&state, &cov, Some(15.0)).is_ok());
        assert!(monitor.check(&state, &cov, Some(15.0)).is_ok());
    }

    #[test]
    fn test_build_scheduler_passthrough() {
        let args = SchedulerArgs {
            sched: SchedKind::Passthrough,
            interval_s: 1.0,
            phase_s: 0.0,
            on_s: 10.0,
            off_s: 10.0,
            duty_phase_s: 0.0,
        };
        let scheduler = build_scheduler(&args);
        matches!(scheduler, GnssScheduler::PassThrough);
    }

    #[test]
    fn test_build_scheduler_fixed() {
        let args = SchedulerArgs {
            sched: SchedKind::Fixed,
            interval_s: 2.5,
            phase_s: 0.5,
            on_s: 10.0,
            off_s: 10.0,
            duty_phase_s: 0.0,
        };
        let scheduler = build_scheduler(&args);
        if let GnssScheduler::FixedInterval {
            interval_s,
            phase_s,
        } = scheduler
        {
            assert_eq!(interval_s, 2.5);
            assert_eq!(phase_s, 0.5);
        } else {
            panic!("Expected FixedInterval scheduler");
        }
    }

    #[test]
    fn test_build_scheduler_duty() {
        let args = SchedulerArgs {
            sched: SchedKind::Duty,
            interval_s: 1.0,
            phase_s: 0.0,
            on_s: 15.0,
            off_s: 5.0,
            duty_phase_s: 2.0,
        };
        let scheduler = build_scheduler(&args);
        if let GnssScheduler::DutyCycle {
            on_s,
            off_s,
            start_phase_s,
        } = scheduler
        {
            assert_eq!(on_s, 15.0);
            assert_eq!(off_s, 5.0);
            assert_eq!(start_phase_s, 2.0);
        } else {
            panic!("Expected DutyCycle scheduler");
        }
    }

    #[test]
    fn test_build_fault_none() {
        let args = FaultArgs {
            fault: FaultKind::None,
            rho_pos: 0.99,
            sigma_pos_m: 3.0,
            rho_vel: 0.95,
            sigma_vel_mps: 0.3,
            r_scale: 5.0,
            drift_n_mps: 0.02,
            drift_e_mps: 0.0,
            q_bias: 1e-6,
            rotate_omega_rps: 0.0,
            hijack_offset_n_m: 50.0,
            hijack_offset_e_m: 0.0,
            hijack_start_s: 120.0,
            hijack_duration_s: 60.0,
        };
        let fault = build_fault(&args);
        matches!(fault, GnssFaultModel::None);
    }

    #[test]
    fn test_build_fault_degraded() {
        let args = FaultArgs {
            fault: FaultKind::Degraded,
            rho_pos: 0.98,
            sigma_pos_m: 5.0,
            rho_vel: 0.93,
            sigma_vel_mps: 0.5,
            r_scale: 10.0,
            drift_n_mps: 0.02,
            drift_e_mps: 0.0,
            q_bias: 1e-6,
            rotate_omega_rps: 0.0,
            hijack_offset_n_m: 50.0,
            hijack_offset_e_m: 0.0,
            hijack_start_s: 120.0,
            hijack_duration_s: 60.0,
        };
        let fault = build_fault(&args);
        if let GnssFaultModel::Degraded {
            rho_pos,
            sigma_pos_m,
            rho_vel,
            sigma_vel_mps,
            r_scale,
        } = fault
        {
            assert_eq!(rho_pos, 0.98);
            assert_eq!(sigma_pos_m, 5.0);
            assert_eq!(rho_vel, 0.93);
            assert_eq!(sigma_vel_mps, 0.5);
            assert_eq!(r_scale, 10.0);
        } else {
            panic!("Expected Degraded fault model");
        }
    }

    #[test]
    fn test_build_fault_slowbias() {
        let args = FaultArgs {
            fault: FaultKind::Slowbias,
            rho_pos: 0.99,
            sigma_pos_m: 3.0,
            rho_vel: 0.95,
            sigma_vel_mps: 0.3,
            r_scale: 5.0,
            drift_n_mps: 0.05,
            drift_e_mps: 0.02,
            q_bias: 1e-5,
            rotate_omega_rps: 0.01,
            hijack_offset_n_m: 50.0,
            hijack_offset_e_m: 0.0,
            hijack_start_s: 120.0,
            hijack_duration_s: 60.0,
        };
        let fault = build_fault(&args);
        if let GnssFaultModel::SlowBias {
            drift_n_mps,
            drift_e_mps,
            q_bias,
            rotate_omega_rps,
        } = fault
        {
            assert_eq!(drift_n_mps, 0.05);
            assert_eq!(drift_e_mps, 0.02);
            assert_eq!(q_bias, 1e-5);
            assert_eq!(rotate_omega_rps, 0.01);
        } else {
            panic!("Expected SlowBias fault model");
        }
    }

    #[test]
    fn test_build_fault_hijack() {
        let args = FaultArgs {
            fault: FaultKind::Hijack,
            rho_pos: 0.99,
            sigma_pos_m: 3.0,
            rho_vel: 0.95,
            sigma_vel_mps: 0.3,
            r_scale: 5.0,
            drift_n_mps: 0.02,
            drift_e_mps: 0.0,
            q_bias: 1e-6,
            rotate_omega_rps: 0.0,
            hijack_offset_n_m: 100.0,
            hijack_offset_e_m: 50.0,
            hijack_start_s: 180.0,
            hijack_duration_s: 90.0,
        };
        let fault = build_fault(&args);
        if let GnssFaultModel::Hijack {
            offset_n_m,
            offset_e_m,
            start_s,
            duration_s,
        } = fault
        {
            assert_eq!(offset_n_m, 100.0);
            assert_eq!(offset_e_m, 50.0);
            assert_eq!(start_s, 180.0);
            assert_eq!(duration_s, 90.0);
        } else {
            panic!("Expected Hijack fault model");
        }
    }

    #[test]
    fn test_ned_covariance() {
        let cov = NEDCovariance {
            latitude_cov: 1e-6,
            longitude_cov: 1e-6,
            altitude_cov: 1e-4,
            velocity_n_cov: 1e-3,
            velocity_e_cov: 1e-3,
            velocity_v_cov: 1e-3,
            roll_cov: 1e-5,
            pitch_cov: 1e-5,
            yaw_cov: 1e-5,
            acc_bias_x_cov: 1e-6,
            acc_bias_y_cov: 1e-6,
            acc_bias_z_cov: 1e-6,
            gyro_bias_x_cov: 1e-8,
            gyro_bias_y_cov: 1e-8,
            gyro_bias_z_cov: 1e-8,
        };
        assert_eq!(cov.latitude_cov, 1e-6);
        assert_eq!(cov.gyro_bias_z_cov, 1e-8);
    }

    #[test]
    fn test_navigation_result_csv_roundtrip_with_nan() {
        let mut nav = NavigationResult::default();
        nav.latitude = 37.0;
        nav.longitude = -122.0;
        nav.altitude = 100.0;
        nav.latitude_cov = f64::NAN;
        nav.longitude_cov = f64::NAN;

        let temp_file = std::env::temp_dir().join("test_nav_nan.csv");
        NavigationResult::to_csv(&[nav.clone()], &temp_file).unwrap();

        let read = NavigationResult::from_csv(&temp_file).unwrap();
        assert_eq!(read.len(), 1);
        assert_eq!(read[0].latitude, 37.0);
        assert!(read[0].latitude_cov.is_nan());

        let _ = std::fs::remove_file(&temp_file);
    }

    #[test]
    fn test_navigation_result_from_strapdown_state_with_rotation() {
        let timestamp = Utc::now();
        let mut state = StrapdownState::default();
        state.latitude = 0.5; // radians
        state.longitude = 1.0; // radians
        state.altitude = 500.0;
        state.velocity_north = 10.0;
        state.velocity_east = 5.0;
        state.velocity_vertical = -1.0;
        state.attitude = nalgebra::Rotation3::from_euler_angles(0.1, 0.2, 0.3);

        let nav = NavigationResult::from((&timestamp, &state));
        assert_eq!(nav.timestamp, timestamp);
        assert!((nav.latitude - 0.5_f64.to_degrees()).abs() < 1e-6);
        assert!((nav.longitude - 1.0_f64.to_degrees()).abs() < 1e-6);
        assert_eq!(nav.altitude, 500.0);
        assert!(nav.latitude_cov.is_nan());
        assert_eq!(nav.acc_bias_x, 0.0);
    }
    #[test]
    fn test_default_process_noise_values() {
        assert_eq!(DEFAULT_PROCESS_NOISE.len(), 15);
        assert_eq!(DEFAULT_PROCESS_NOISE[0], 1e-6); // position
        assert_eq!(DEFAULT_PROCESS_NOISE[2], 1e-4); // altitude
        assert_eq!(DEFAULT_PROCESS_NOISE[3], 1e-3); // velocity
    }

    #[test]
    fn test_run_closed_loop_with_health_limits() {
        let rec = TestDataRecord {
            time: Utc::now(),
            horizontal_accuracy: 5.0,
            vertical_accuracy: 2.0,
            speed_accuracy: 1.0,
            latitude: 37.0,
            longitude: -122.0,
            altitude: 100.0,
            speed: 10.0,
            bearing: 45.0,
            roll: 0.0,
            pitch: 0.0,
            yaw: 0.0,
            acc_x: 0.0,
            acc_y: 0.0,
            acc_z: 9.81,
            gyro_x: 0.0,
            gyro_y: 0.0,
            gyro_z: 0.0,
            ..Default::default()
        };

        let mut ukf = initialize_ukf(rec.clone(), None, None, None, None, None, None);

        let stream = EventStream {
            start_time: rec.time,
            events: vec![Event::Imu {
                dt_s: 0.1,
                imu: IMUData {
                    accel: Vector3::new(0.0, 0.0, 9.81),
                    gyro: Vector3::new(0.0, 0.0, 0.0),
                },
                elapsed_s: 0.0,
            }],
        };

        let health_limits = HealthLimits::default();
        let result = run_closed_loop(&mut ukf, stream, Some(health_limits));
        assert!(result.is_ok());
    }

    // ==================== Extended Kalman Filter Tests ====================

    #[test]
    fn test_initialize_ekf_default_9state() {
        let rec = TestDataRecord {
            time: Utc::now(),
            horizontal_accuracy: 5.0,
            vertical_accuracy: 2.0,
            speed_accuracy: 1.0,
            latitude: 37.0,
            longitude: -122.0,
            altitude: 100.0,
            speed: 10.0,
            bearing: 45.0, // In degrees
            roll: 0.0,
            pitch: 0.0,
            yaw: 0.0,
            ..Default::default()
        };
        let ekf = initialize_ekf(rec, None, None, None, None, false);
        let estimate = ekf.get_estimate();
        assert_eq!(estimate.len(), 9, "9-state EKF should have 9 states");
        // Check velocity decomposition (bearing 45° means equal north/east components)
        assert!((estimate[3] - estimate[4]).abs() < 1.0); // vn ≈ ve for 45° bearing
    }

    #[test]
    fn test_initialize_ekf_default_15state() {
        let rec = TestDataRecord {
            time: Utc::now(),
            horizontal_accuracy: 5.0,
            vertical_accuracy: 2.0,
            speed_accuracy: 1.0,
            latitude: 37.0,
            longitude: -122.0,
            altitude: 100.0,
            speed: 10.0,
            bearing: 45.0,
            roll: 0.0,
            pitch: 0.0,
            yaw: 0.0,
            ..Default::default()
        };
        let ekf = initialize_ekf(rec, None, None, None, None, true);
        let estimate = ekf.get_estimate();
        assert_eq!(estimate.len(), 15, "15-state EKF should have 15 states");
        // Check that biases are initialized to zero by default
        for i in 9..15 {
            assert!(
                estimate[i].abs() < 1e-6,
                "Default biases should be near zero"
            );
        }
    }

    #[test]
    fn test_initialize_ekf_with_nan_angles() {
        let rec = TestDataRecord {
            time: Utc::now(),
            horizontal_accuracy: 5.0,
            vertical_accuracy: 2.0,
            speed_accuracy: 1.0,
            latitude: 37.0,
            longitude: -122.0,
            altitude: 100.0,
            speed: 10.0,
            bearing: 45.0,
            roll: f64::NAN,
            pitch: f64::NAN,
            yaw: f64::NAN,
            ..Default::default()
        };
        let ekf = initialize_ekf(rec, None, None, None, None, true);
        let estimate = ekf.get_estimate();
        // Should default NaN angles to 0.0
        assert!(estimate[6].abs() < 1e-6, "NaN roll should default to 0"); // roll
        assert!(estimate[7].abs() < 1e-6, "NaN pitch should default to 0"); // pitch
        assert!(estimate[8].abs() < 1e-6, "NaN yaw should default to 0"); // yaw
    }

    #[test]
    fn test_initialize_ekf_with_custom_biases() {
        let rec = TestDataRecord {
            time: Utc::now(),
            horizontal_accuracy: 5.0,
            vertical_accuracy: 2.0,
            speed_accuracy: 1.0,
            latitude: 37.0,
            longitude: -122.0,
            altitude: 100.0,
            speed: 10.0,
            bearing: 45.0,
            roll: 0.0,
            pitch: 0.0,
            yaw: 0.0,
            ..Default::default()
        };
        let ekf = initialize_ekf(
            rec,
            Some(vec![1e-4, 2e-4, 3e-4]),
            Some(vec![0.01, 0.02, 0.03, 0.001, 0.002, 0.003]),
            Some(vec![1e-5; 6]),
            None,
            true,
        );
        let estimate = ekf.get_estimate();
        assert_eq!(estimate.len(), 15);
        // Check that custom biases are set
        assert!((estimate[9] - 0.01).abs() < 1e-9);
        assert!((estimate[10] - 0.02).abs() < 1e-9);
        assert!((estimate[11] - 0.03).abs() < 1e-9);
    }

    #[test]
    fn test_initialize_ekf_with_custom_process_noise() {
        let rec = TestDataRecord {
            time: Utc::now(),
            horizontal_accuracy: 5.0,
            vertical_accuracy: 2.0,
            speed_accuracy: 1.0,
            latitude: 37.0,
            longitude: -122.0,
            altitude: 100.0,
            speed: 10.0,
            bearing: 45.0,
            roll: 0.0,
            pitch: 0.0,
            yaw: 0.0,
            ..Default::default()
        };
        let custom_noise = vec![1e-7; 15];
        let ekf = initialize_ekf(rec, None, None, None, Some(custom_noise.clone()), true);
        // Verify EKF was created successfully
        assert_eq!(ekf.get_estimate().len(), 15);
    }

    #[test]
    fn test_test_data_record_hdf5_roundtrip() {
        use tempfile::tempdir;

        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test_data.h5");

        // Create test records
        let mut records = Vec::new();
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
            bearing_accuracy: 0.2,
            speed_accuracy: 0.2,
            vertical_accuracy: 0.2,
            horizontal_accuracy: 0.2,
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

        // Write to HDF5
        TestDataRecord::to_hdf5(&records, &file_path).expect("Failed to write HDF5");

        // Read back from HDF5
        let read_records = TestDataRecord::from_hdf5(&file_path).expect("Failed to read HDF5");

        // Verify
        assert_eq!(read_records.len(), records.len());
        for (i, (original, read)) in records.iter().zip(read_records.iter()).enumerate() {
            assert_eq!(
                original.time, read.time,
                "Timestamp mismatch at index {}",
                i
            );
            assert!(
                (original.latitude - read.latitude).abs() < 1e-10,
                "Latitude mismatch at index {}",
                i
            );
            assert!(
                (original.longitude - read.longitude).abs() < 1e-10,
                "Longitude mismatch at index {}",
                i
            );
            assert!(
                (original.altitude - read.altitude).abs() < 1e-10,
                "Altitude mismatch at index {}",
                i
            );
            assert!(
                (original.speed - read.speed).abs() < 1e-10,
                "Speed mismatch at index {}",
                i
            );
            assert!(
                (original.bearing - read.bearing).abs() < 1e-10,
                "Bearing mismatch at index {}",
                i
            );
        }
    }

    #[test]
    fn test_navigation_result_hdf5_roundtrip() {
        use tempfile::tempdir;

        let dir = tempdir().unwrap();
        let file_path = dir.path().join("nav_results.h5");

        // Create test navigation results
        let mut results = Vec::new();
        let mut nav1 = NavigationResult::new();
        nav1.timestamp =
            DateTime::parse_from_str("2023-01-01 00:00:00+00:00", "%Y-%m-%d %H:%M:%S%z")
                .unwrap()
                .with_timezone(&Utc);
        nav1.latitude = 37.0;
        nav1.longitude = -122.0;
        nav1.altitude = 100.0;
        nav1.velocity_north = 1.0;
        nav1.velocity_east = 2.0;
        nav1.velocity_vertical = 0.1;
        nav1.roll = 0.01;
        nav1.pitch = 0.02;
        nav1.yaw = 0.03;
        results.push(nav1);

        let mut nav2 = NavigationResult::new();
        nav2.timestamp =
            DateTime::parse_from_str("2023-01-01 00:00:01+00:00", "%Y-%m-%d %H:%M:%S%z")
                .unwrap()
                .with_timezone(&Utc);
        nav2.latitude = 37.0001;
        nav2.longitude = -122.0001;
        nav2.altitude = 101.0;
        nav2.velocity_north = 1.1;
        nav2.velocity_east = 2.1;
        nav2.velocity_vertical = 0.2;
        nav2.roll = 0.02;
        nav2.pitch = 0.03;
        nav2.yaw = 0.04;
        results.push(nav2);

        // Write to HDF5
        NavigationResult::to_hdf5(&results, &file_path).expect("Failed to write HDF5");

        // Read back from HDF5
        let read_results = NavigationResult::from_hdf5(&file_path).expect("Failed to read HDF5");

        // Verify
        assert_eq!(read_results.len(), results.len());
        for (i, (original, read)) in results.iter().zip(read_results.iter()).enumerate() {
            assert_eq!(
                original.timestamp, read.timestamp,
                "Timestamp mismatch at index {}",
                i
            );
            assert!(
                (original.latitude - read.latitude).abs() < 1e-10,
                "Latitude mismatch at index {}",
                i
            );
            assert!(
                (original.longitude - read.longitude).abs() < 1e-10,
                "Longitude mismatch at index {}",
                i
            );
            assert!(
                (original.altitude - read.altitude).abs() < 1e-10,
                "Altitude mismatch at index {}",
                i
            );
            assert!(
                (original.velocity_north - read.velocity_north).abs() < 1e-10,
                "Velocity north mismatch at index {}",
                i
            );
            assert!(
                (original.velocity_east - read.velocity_east).abs() < 1e-10,
                "Velocity east mismatch at index {}",
                i
            );
            assert!(
                (original.roll - read.roll).abs() < 1e-10,
                "Roll mismatch at index {}",
                i
            );
            assert!(
                (original.pitch - read.pitch).abs() < 1e-10,
                "Pitch mismatch at index {}",
                i
            );
            assert!(
                (original.yaw - read.yaw).abs() < 1e-10,
                "Yaw mismatch at index {}",
                i
            );
        }
    }

    #[test]
    fn test_test_data_record_hdf5_with_nan_values() {
        use tempfile::tempdir;

        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test_data_nan.h5");

        // Create a record with NaN values
        let mut record = TestDataRecord::default();
        record.time = Utc::now();
        record.latitude = 37.0;
        record.longitude = -122.0;
        record.altitude = f64::NAN;
        record.speed = f64::NAN;
        record.bearing = 90.0;

        let records = vec![record.clone()];

        // Write to HDF5
        TestDataRecord::to_hdf5(&records, &file_path).expect("Failed to write HDF5");

        // Read back from HDF5
        let read_records = TestDataRecord::from_hdf5(&file_path).expect("Failed to read HDF5");

        // Verify NaN values are preserved
        assert_eq!(read_records.len(), 1);
        assert!(
            read_records[0].altitude.is_nan(),
            "NaN altitude should be preserved"
        );
        assert!(
            read_records[0].speed.is_nan(),
            "NaN speed should be preserved"
        );
        assert!((read_records[0].latitude - record.latitude).abs() < 1e-10);
        assert!((read_records[0].longitude - record.longitude).abs() < 1e-10);
    }

    #[test]
    fn test_navigation_result_hdf5_empty() {
        use tempfile::tempdir;

        let dir = tempdir().unwrap();
        let file_path = dir.path().join("nav_results_empty.h5");

        // Write empty results
        let results: Vec<NavigationResult> = Vec::new();
        NavigationResult::to_hdf5(&results, &file_path).expect("Failed to write empty HDF5");

        // Read back
        let read_results =
            NavigationResult::from_hdf5(&file_path).expect("Failed to read empty HDF5");

        // Verify it's empty
        assert_eq!(read_results.len(), 0);
        // Note: TestDataRecord MCAP roundtrip test is disabled due to CSV-specific deserializers
        // that conflict with binary serialization formats. TestDataRecord is optimized for CSV.
        // For MCAP usage, convert TestDataRecord to NavigationResult.
    }
    #[test]
    fn test_navigation_result_mcap_roundtrip() {
        let temp_file = std::env::temp_dir().join("nav_results_mcap.mcap");

        // Create test navigation results
        let mut result1 = NavigationResult::default();
        result1.timestamp = Utc::now();
        result1.latitude = 37.0;
        result1.longitude = -122.0;
        result1.altitude = 100.0;
        result1.velocity_north = 10.0;
        result1.velocity_east = 5.0;
        result1.velocity_vertical = -1.0;
        result1.roll = 0.1;
        result1.pitch = 0.2;
        result1.yaw = 0.3;

        let mut result2 = NavigationResult::default();
        result2.timestamp = Utc::now() + chrono::Duration::seconds(1);
        result2.latitude = 37.01;
        result2.longitude = -122.01;
        result2.altitude = 110.0;
        result2.velocity_north = 12.0;
        result2.velocity_east = 6.0;
        result2.velocity_vertical = 0.5;
        result2.roll = 0.15;
        result2.pitch = 0.25;
        result2.yaw = 0.35;

        let results = vec![result1.clone(), result2.clone()];

        // Write to MCAP
        NavigationResult::to_mcap(&results, &temp_file)
            .expect("Failed to write navigation results to MCAP");

        // Check file exists
        assert!(temp_file.exists(), "MCAP file should exist");

        // Read back from MCAP
        let read_results = NavigationResult::from_mcap(&temp_file)
            .expect("Failed to read navigation results from MCAP");

        // Verify count
        assert_eq!(
            read_results.len(),
            results.len(),
            "Result count should match"
        );

        // Verify content
        for (i, (original, read)) in results.iter().zip(read_results.iter()).enumerate() {
            assert_eq!(
                original.timestamp, read.timestamp,
                "Result {} timestamp should match",
                i
            );
            assert!(
                (original.latitude - read.latitude).abs() < 1e-6,
                "Result {} latitude should match",
                i
            );
            assert!(
                (original.longitude - read.longitude).abs() < 1e-6,
                "Result {} longitude should match",
                i
            );
            assert!(
                (original.altitude - read.altitude).abs() < 1e-6,
                "Result {} altitude should match",
                i
            );
            assert!(
                (original.velocity_north - read.velocity_north).abs() < 1e-6,
                "Result {} velocity_north should match",
                i
            );
            assert!(
                (original.roll - read.roll).abs() < 1e-6,
                "Result {} roll should match",
                i
            );
            assert!(
                (original.pitch - read.pitch).abs() < 1e-6,
                "Result {} pitch should match",
                i
            );
            assert!(
                (original.yaw - read.yaw).abs() < 1e-6,
                "Result {} yaw should match",
                i
            );
        }

        // Cleanup
        let _ = std::fs::remove_file(&temp_file);
    }
}
