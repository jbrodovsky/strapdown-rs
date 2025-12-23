//! Rust toolbox for geophysical navigation and map matching
//!
//! This module works alongside the `strapdown-core` crate to provide complimentary geophysical navigation aiding
//! and map matching functionality. It also provides a set of tools for working with geophysical maps, such as
//! relief, gravity, and magnetic maps. These maps are downloaded from the GMT database and can be used for
//! navigation and map matching purposes.
//!
//! Anomaly measurement models require some degree of knowledge about the vehicle state. Due to the way the measurement
//! event stream is constructed, this state is not known at the time of simulation initialization. As such, the measurement
//! models corresponding to geophysical anomalies are not implemented as standalone models, but rather as a specific
//! processing configuration that must be implemented in the closed loop configuration.
//!
//! For example: gravity anomaly calculation requires knowledge of the vehicle velocity, to make the Eotvos correction.
//! The measurement event stream can be constructed to include the gravity vector measurements from TestDataRecord (`grav_x``,
//! `grav_y`, `grav_z`), but these values are not the specific anomaly. The scalar gravity must be calculated and corrected
//! using the vehicle velocity (Eotvos correction) and the reference gravity at the current position (from a gravity map) to
//! calculate the free air anomaly.
use std::any::Any;
use std::fmt::{Debug, Display};
use std::path::PathBuf;
use std::rc::Rc;

use anyhow::{Result, bail};
use chrono::{DateTime, Datelike, Duration, Utc};
use log::debug;
use nalgebra::{DMatrix, DVector, Vector3};
use world_magnetic_model::GeomagneticField;
use world_magnetic_model::time::Date;
use world_magnetic_model::uom::si::angle::degree;
use world_magnetic_model::uom::si::f32::{Angle, Length};
use world_magnetic_model::uom::si::length::meter;

use strapdown::earth::gravity_anomaly;
use strapdown::kalman::UnscentedKalmanFilter;
use strapdown::measurements::{
    GPSPositionAndVelocityMeasurement, MeasurementModel, RelativeAltitudeMeasurement,
};
use strapdown::messages::{
    Event, EventStream, FaultState, GnssDegradationConfig, GnssScheduler, apply_fault,
};
use strapdown::sim::health::{HealthLimits, HealthMonitor};
use strapdown::sim::{NavigationResult, TestDataRecord};
use strapdown::{IMUData, NavigationFilter, StrapdownState};

//================= Map Information ========================================================================
/// Resolution values for bathymetric or terrain relief maps
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReliefResolution {
    OneDegree,
    ThirtyMinutes,
    TwentyMinutes,
    FifteenMinutes,
    TenMinutes,
    SixMinutes,
    FiveMinutes,
    FourMinutes,
    ThreeMinutes,
    TwoMinutes,
    OneMinute,
    ThirtySeconds,
    FifteenSeconds,
    ThreeSeconds,
    OneSecond,
}
impl Display for ReliefResolution {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let res = match self {
            ReliefResolution::OneDegree => "01d",
            ReliefResolution::ThirtyMinutes => "30m",
            ReliefResolution::TwentyMinutes => "20m",
            ReliefResolution::FifteenMinutes => "15m",
            ReliefResolution::TenMinutes => "10m",
            ReliefResolution::SixMinutes => "06m",
            ReliefResolution::FiveMinutes => "05m",
            ReliefResolution::FourMinutes => "04m",
            ReliefResolution::ThreeMinutes => "03m",
            ReliefResolution::TwoMinutes => "02m",
            ReliefResolution::OneMinute => "01m",
            ReliefResolution::ThirtySeconds => "30s",
            ReliefResolution::FifteenSeconds => "15s",
            ReliefResolution::ThreeSeconds => "03s",
            ReliefResolution::OneSecond => "01s",
        };
        write!(f, "{}", res)
    }
}
/// Resolution values for gravity maps
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GravityResolution {
    OneDegree,
    ThirtyMinutes,
    TwentyMinutes,
    FifteenMinutes,
    TenMinutes,
    SixMinutes,
    FiveMinutes,
    FourMinutes,
    ThreeMinutes,
    TwoMinutes,
    OneMinute,
}
impl Display for GravityResolution {
    /// Convert the resolution to a string. This can be used for calling the GMT library
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let res = match self {
            GravityResolution::OneDegree => "01d",
            GravityResolution::ThirtyMinutes => "30m",
            GravityResolution::TwentyMinutes => "20m",
            GravityResolution::FifteenMinutes => "15m",
            GravityResolution::TenMinutes => "10m",
            GravityResolution::SixMinutes => "06m",
            GravityResolution::FiveMinutes => "05m",
            GravityResolution::FourMinutes => "04m",
            GravityResolution::ThreeMinutes => "03m",
            GravityResolution::TwoMinutes => "02m",
            GravityResolution::OneMinute => "01m",
        };
        write!(f, "{}", res)
    }
}
/// Resolution values for magnetic maps
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MagneticResolution {
    OneDegree,
    ThirtyMinutes,
    TwentyMinutes,
    FifteenMinutes,
    TenMinutes,
    SixMinutes,
    FiveMinutes,
    FourMinutes,
    ThreeMinutes,
    TwoMinutes,
}
impl Display for MagneticResolution {
    /// Convert the resolution to a string. This can be used for calling the GMT library
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let res = match self {
            MagneticResolution::OneDegree => "01d",
            MagneticResolution::ThirtyMinutes => "30m",
            MagneticResolution::TwentyMinutes => "20m",
            MagneticResolution::FifteenMinutes => "15m",
            MagneticResolution::TenMinutes => "10m",
            MagneticResolution::SixMinutes => "06m",
            MagneticResolution::FiveMinutes => "05m",
            MagneticResolution::FourMinutes => "04m",
            MagneticResolution::ThreeMinutes => "03m",
            MagneticResolution::TwoMinutes => "02m",
        };
        write!(f, "{}", res)
    }
}
/// Enum for the different types of maps. A GeoMap is defined by its measurement type and resolution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GeophysicalMeasurementType {
    Relief(ReliefResolution),
    Gravity(GravityResolution),
    Magnetic(MagneticResolution),
}
impl Display for GeophysicalMeasurementType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GeophysicalMeasurementType::Relief(res) => write!(f, "Relief {}", res),
            GeophysicalMeasurementType::Gravity(res) => write!(f, "Gravity {}", res),
            GeophysicalMeasurementType::Magnetic(res) => write!(f, "Magnetic {}", res),
        }
    }
}
/// Struct for the GeoMap object
/// This struct contains the latitude and longitude vectors, the data matrix, and the type of map
/// The data matrix is a 2D matrix of data values, where the rows are the latitudes and the columns
/// are the longitudes. The data values are the values at the corresponding lat/lon points.
/// The map type is an enum that indicates the type of map (Relief, Gravity, Magnetic)
/// The lat/lon vectors are used to determine the bounds of the map and to interpolate the data values
#[derive(Clone, PartialEq)]
pub struct GeoMap {
    lats: DVector<f64>,
    lons: DVector<f64>,
    data: DMatrix<f64>,
    map_type: GeophysicalMeasurementType,
}
impl Debug for GeoMap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "GeoMap {{ {:?} x {:?}, map_type: {:?} }}",
            self.lats.len(),
            self.lons.len(),
            self.map_type
        )
    }
}
impl Display for GeoMap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "GeoMap: {} | Size: {} x {} | Lat range: [{}, {}] | Lon range: [{}, {}]",
            self.map_type,
            self.lats.len(),
            self.lons.len(),
            self.lats[0],
            self.lats[self.lats.len() - 1],
            self.lons[0],
            self.lons[self.lons.len() - 1]
        )
    }
}
impl GeoMap {
    /// Create a new GeoMap object from the supplied latitudes, longitudes, data matrix, and map type
    ///
    /// # Arguments
    /// - `lats` - A vector of latitudes
    /// - `lons` - A vector of longitudes
    /// - `data` - A matrix of data values
    /// - `map_type` - The type of map (Relief, Gravity, Magnetic)
    ///
    /// # Returns
    /// - A new GeoMap object
    ///
    /// # Example
    /// ```rust
    /// use nalgebra::{DVector, DMatrix};
    /// use geonav::{GeoMap, GeophysicalMeasurementType, ReliefResolution};
    /// let lats = DVector::from_vec(vec![1.0, 2.0, 3.0]);
    /// let lons = DVector::from_vec(vec![1.0, 2.0, 3.0]);
    /// let data = DMatrix::from_vec(3, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    /// let map_type = GeophysicalMeasurementType::Relief(ReliefResolution::OneDegree);
    /// let map = GeoMap::new(lats, lons, data, map_type);
    /// ```
    pub fn new(
        lats: DVector<f64>,
        lons: DVector<f64>,
        data: DMatrix<f64>,
        map_type: GeophysicalMeasurementType,
    ) -> Self {
        GeoMap {
            lats,
            lons,
            data,
            map_type,
        }
    }
    /// Load a GeoMap from a netcdf file. GMT processing does not encode the map type in the file, so this
    /// function requires the user to specify the type of map along with the filename.
    ///
    /// # Arguments
    /// - `filename` - The PathBuf of the netcdf file
    /// - `map_type` - The type of map (Relief, Gravity, Magnetic)
    ///
    /// # Returns
    /// - A Result containing a reference to the GeoMap object or an error message
    ///
    /// # Example
    /// ```ignore
    /// use geonav::{GeoMap, ReliefResolution, GeophysicalMeasurementType};
    /// use std::path::PathBuf;
    /// let map = GeoMap::load_geomap(PathBuf::from("path/to/file.nc"), GeophysicalMeasurementType::Relief(ReliefResolution::OneDegree));
    /// ```
    pub fn load_geomap(
        filename: PathBuf,
        map_type: GeophysicalMeasurementType,
    ) -> Result<GeoMap, String> {
        // Open the netcdf file
        let file = match netcdf::open(filename) {
            Ok(file) => file,
            Err(e) => panic!("Error opening file: {e:?}"),
        };
        // Get the lat/lon variables
        let lats: &netcdf::Variable<'_> =
            &file.variable("lat").expect("Could not find variable 'lat'");
        let lons: &netcdf::Variable<'_> =
            &file.variable("lon").expect("Could not find variable 'lon'");
        // Get the data variable
        let data: netcdf::Variable<'_> = file.variable("z").expect("Could not find variable 'z'");
        // Conversion to basic types
        let lats: Vec<f64> = lats.get_values(..).unwrap();
        let lons: Vec<f64> = lons.get_values(..).unwrap();
        let data: Vec<f64> = data.get_values(..).unwrap();
        // Convert the data to DVector
        let lats = DVector::from_vec(lats.to_vec());
        let lons = DVector::from_vec(lons.to_vec());
        // Convert the data to DMatrix
        let data = DMatrix::from_row_slice(lats.len(), lons.len(), &data);
        // Create the GeoMap object
        Ok(GeoMap::new(lats, lons, data, map_type))
    }
    /// Get the latitude vector
    /// # Returns
    /// - A reference to the latitude vector
    pub fn get_lats(&self) -> &DVector<f64> {
        &self.lats
    }
    /// Get the longitude vector
    /// # Returns
    /// - A reference to the longitude vector
    pub fn get_lons(&self) -> &DVector<f64> {
        &self.lons
    }
    /// Get the data matrix
    /// # Returns
    /// - A reference to the data matrix
    pub fn get_data(&self) -> &DMatrix<f64> {
        &self.data
    }
    /// Get the map type
    /// # Returns
    /// - A reference to the map type
    pub fn get_map_type(&self) -> &GeophysicalMeasurementType {
        &self.map_type
    }
    /// Get the map type as a string
    /// # Returns
    /// - A string representation of the map type
    pub fn get_map_type_str(&self) -> String {
        match self.map_type {
            GeophysicalMeasurementType::Relief(_) => "Relief".to_string(),
            GeophysicalMeasurementType::Gravity(_) => "Gravity".to_string(),
            GeophysicalMeasurementType::Magnetic(_) => "Magnetic".to_string(),
        }
    }
    /// Get the map data contained at a specific point.
    ///
    /// Queries the map for the data at a specific point given by the latitude and longitude coordinates
    /// and interpolates using a bilinear interpolation method.
    ///
    /// # Arguments
    /// - `lat` - The latitude of the point in degrees
    /// - `lon` - The longitude of the point in degrees
    ///
    /// # Returns
    /// - An Option containing the data value at the point, or None if the point is not in the map
    ///
    /// # Example
    /// ```ignore
    /// use geonav::{GeoMap, GeophysicalMeasurementType, ReliefResolution};
    /// use std::path::PathBuf;
    ///
    /// let map = GeoMap::load_geomap(PathBuf::from("path/to/file.nc"), GeophysicalMeasurementType::Relief(ReliefResolution::OneDegree));
    /// let value = map.get_point(&1.5, &1.5);
    /// ```
    /// # Panics
    /// - Panics if the lat/lon are out of bounds
    /// - Panics if the lat/lon are not in the map
    pub fn get_point(&self, lat: &f64, lon: &f64) -> Option<f64> {
        // Check if the lat/lon are within the bounds of the map
        if lat < &self.lats[0] || lat > &self.lats[self.lats.len() - 1] {
            panic!(
                "Latitude out of bounds: {} not in [{}, {}]",
                lat,
                self.lats[0],
                self.lats[self.lats.len() - 1]
            );
        }
        if lon < &self.lons[0] || lon > &self.lons[self.lons.len() - 1] {
            panic!("Longitude out of bounds");
        }
        // Check if the lat/lon are at the origin or the end of the map
        if lat == &self.lats[0] && lon == &self.lons[0] {
            // If the lat/lon are at the origin, return the first data point
            return Some(self.data[(0, 0)]);
        }
        if lat == &self.lats[self.lats.len() - 1] && lon == &self.lons[self.lons.len() - 1] {
            // If the lat/lon are at the end, return the last data point
            return Some(self.data[(self.lats.len() - 1, self.lons.len() - 1)]);
        }
        // Structure the interpolation in a few different ways. If the lat/lon are on the edge of the map,
        // only interpolate using the coordinate that is not on the edge.
        let lat_index = self.lats.iter().position(|&x| x >= *lat).unwrap();
        let lon_index = self.lons.iter().position(|&x| x >= *lon).unwrap();
        if lat == &self.lats[0] || lat == &self.lats[self.lats.len() - 1] {
            // If the latitude is on the edge, only interpolate using longitude
            let lon_index = self.lons.iter().position(|&x| x >= *lon).unwrap();
            // Special case for the edges of the map
            if lon_index == 0 {
                return Some(self.data[(lat_index, lon_index)]);
            }
            let lon1_index = lon_index - 1;
            let a = self.data[(lat_index, lon_index)];
            let b = self.data[(lat_index, lon1_index)];
            debug!("Bilinear interpolation edge case - a: {}, b: {}", a, b);
            let lon_diff = self.lons[lon_index] - self.lons[lon1_index];
            let result = ((a - b) / lon_diff) * (lon - self.lons[lon1_index]) + b;
            return Some(result);
        }
        if lon == &self.lons[0] || lon == &self.lons[self.lons.len() - 1] {
            // If the longitude is on the edge, only interpolate using latitude
            let lat_index = self.lats.iter().position(|&x| x >= *lat).unwrap();
            if lat_index == 0 {
                return Some(self.data[(lat_index, lon_index)]);
            }
            let lat1_index = lat_index - 1;
            let a = self.data[(lat_index, lon_index)];
            let b = self.data[(lat1_index, lon_index)];
            let lat_diff = self.lats[lat_index] - self.lats[lat1_index];
            return Some(((a - b) / lat_diff) * (lat - self.lats[lat1_index]) + b);
        }
        // If the lat/lon are not on the edge, use normal bilinear interpolation
        self.bilinear_interpolation(lat, lon)
    }
    /// Bilinear interpolation helper method for get_point
    fn bilinear_interpolation(&self, lat: &f64, lon: &f64) -> Option<f64> {
        // Find the indices that surround the point
        let lat2_index: usize = self.lats.iter().position(|&x| x >= *lat).unwrap();
        let lon2_index: usize = self.lons.iter().position(|&x| x >= *lon).unwrap();
        let lat1_index: usize = lat2_index - 1;
        let lon1_index: usize = lon2_index - 1;
        // Get the four surrounding points
        let q11: f64 = self.data[(lat1_index, lon1_index)];
        let q12: f64 = self.data[(lat2_index, lon1_index)];
        let q21: f64 = self.data[(lat1_index, lon2_index)];
        let q22: f64 = self.data[(lat2_index, lon2_index)];
        // Get the coordinates of the four surrounding points
        let lon1: f64 = self.lons[lon1_index];
        let lat1: f64 = self.lats[lat1_index];
        let lon2: f64 = self.lons[lon2_index];
        let lat2: f64 = self.lats[lat2_index];
        // Perform bilinear interpolation via weighted mean
        let w11: f64 = ((lon2 - lon) * (lat2 - lat)) / ((lon2 - lon1) * (lat2 - lat1));
        let w12: f64 = ((lon2 - lon) * (lat - lat1)) / ((lon2 - lon1) * (lat2 - lat1));
        let w21: f64 = ((lon - lon1) * (lat2 - lat)) / ((lon2 - lon1) * (lat2 - lat1));
        let w22: f64 = ((lon - lon1) * (lat - lat1)) / ((lon2 - lon1) * (lat2 - lat1));
        Some(w11 * q11 + w12 * q12 + w21 * q21 + w22 * q22)
    }
    // TODO: #95 Implement direct GMT interface using system shell calls
}
//================= Geophysical Measurement Models =========================================================
/// Trait for geophysical anomaly measurement models
///
/// Geophysical anomaly measurements require some degree of knowledge about the vehicle state. Due to the way
/// the measurement event stream is constructed, this state is not known at the time of simulation initialization.
/// As such, the measurement models corresponding to geophysical anomalies are not implemented as standalone
/// models, but rather as a specific processing configuration that must be implemented in the closed loop configuration.
///
/// For example: gravity anomaly calculation requires knowledge of the vehicle velocity, to make the Eotvos correction.
/// Magnetic anomaly calculation requires knowledge of the vehicle pose and the date to compute the reference magnetic
/// field using the World Magnetic Model (WMM).
pub trait GeophysicalAnomalyMeasurementModel: MeasurementModel {
    fn get_anomaly(&self) -> f64;
    fn set_state(&mut self, state: &StrapdownState);
}
/// Gravity measurement model
#[derive(Clone, Debug)]
pub struct GravityMeasurement {
    /// Source map
    pub map: Rc<GeoMap>,
    /// Measurement Noise
    pub noise_std: f64,
    /// Observed gravity magnitude (m/s^2)
    pub gravity_observed: f64,
    /// Current latitude
    latitude: f64,
    /// Current altitude (m)
    altitude: f64,
    /// Current north velocity (m/s)
    north_velocity: f64,
    /// Current east velocity (m/s)
    east_velocity: f64,
}
/// Geophysical anomaly measurement model implementation for gravity. This trait provides a method to compute
/// the gravity anomaly given the current state. Free air anomaly correction needs knowledge of the vehicle
/// velocity to compute the Eotvos correction.
impl GeophysicalAnomalyMeasurementModel for GravityMeasurement {
    fn get_anomaly(&self) -> f64 {
        gravity_anomaly(
            &self.latitude,
            &self.altitude,
            &self.north_velocity,
            &self.east_velocity,
            &self.gravity_observed,
        )
    }
    fn set_state(&mut self, state: &StrapdownState) {
        self.latitude = state.latitude;
        self.altitude = state.altitude;
        self.north_velocity = state.velocity_north;
        self.east_velocity = state.velocity_east;
    }
}
impl MeasurementModel for GravityMeasurement {
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn get_dimension(&self) -> usize {
        1 // Single measurement: map value at current position
    }
    fn get_vector(&self) -> DVector<f64> {
        // Return the observed gravity as the measurement vector
        // The expected measurement will be computed in get_sigma_points using the current state
        DVector::from_vec(vec![self.get_anomaly()])
    }
    fn get_noise(&self) -> DMatrix<f64> {
        DMatrix::from_diagonal(&DVector::from_element(
            self.get_dimension(),
            self.noise_std.powi(2),
        ))
    }
    fn get_expected_measurement(&self, state: &DVector<f64>) -> DVector<f64> {
        let lat = state[0];
        let lon = state[1];
        let map_value = self
            .map
            .get_point(&lat.to_degrees(), &lon.to_degrees())
            .unwrap_or(f64::NAN);
        DVector::from_vec(vec![map_value])
    }
    // fn get_sigma_points(&self, state_sigma_points: &DMatrix<f64>) -> DMatrix<f64> {
    //     let num_sigma_points = state_sigma_points.ncols();
    //     let mut measurement_sigma_points = DMatrix::zeros(1, num_sigma_points);
    //
    //     for i in 0..num_sigma_points {
    //         let state = state_sigma_points.column(i);
    //         let lat = state[0];
    //         let lon = state[1];
    //
    //         measurement_sigma_points[(0, i)] = self
    //             .map
    //             .get_point(&lat.to_degrees(), &lon.to_degrees())
    //             .unwrap_or(f64::NAN);
    //     }
    //     measurement_sigma_points
    // }
}
/// Magnetic anomaly measurement model
#[derive(Clone, Debug)]
pub struct MagneticAnomalyMeasurement {
    /// Source map
    pub map: Rc<GeoMap>,
    /// Measurement Noise
    pub noise_std: f64,
    /// Measured magnetic field x-component (micro teslas)
    pub mag_obs: f64,
    /// Year for WMM calculation
    pub year: i32,
    /// Day of year for WMM calculation
    pub day: u16,
    /// Latitude (degrees)
    pub latitude: f64,
    /// Longitude (degrees)
    pub longitude: f64,
    /// Altitude (meters)
    pub altitude: f64,
}
impl GeophysicalAnomalyMeasurementModel for MagneticAnomalyMeasurement {
    fn get_anomaly(&self) -> f64 {
        let magnetic_field = GeomagneticField::new(
            Length::new::<meter>(self.altitude as f32),
            Angle::new::<degree>(self.latitude as f32),
            Angle::new::<degree>(self.longitude as f32),
            Date::from_ordinal_date(self.year, self.day).unwrap(),
        )
        .expect("Failed to create GeomagneticField");
        self.mag_obs - magnetic_field.f().value as f64
    }
    fn set_state(&mut self, state: &StrapdownState) {
        self.latitude = state.latitude.to_degrees();
        self.longitude = state.longitude.to_degrees();
        self.altitude = state.altitude;
    }
}
impl MeasurementModel for MagneticAnomalyMeasurement {
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn get_dimension(&self) -> usize {
        1 // Single measurement: map value at current position
    }
    fn get_vector(&self) -> DVector<f64> {
        DVector::from_vec(vec![self.get_anomaly()])
    }
    fn get_noise(&self) -> DMatrix<f64> {
        DMatrix::from_diagonal(&DVector::from_element(
            self.get_dimension(),
            self.noise_std.powi(2),
        ))
    }
    fn get_expected_measurement(&self, state: &DVector<f64>) -> DVector<f64> {
        let lat = state[0];
        let lon = state[1];
        let map_value = self
            .map
            .get_point(&lat.to_degrees(), &lon.to_degrees())
            .unwrap_or(f64::NAN);
        DVector::from_vec(vec![map_value])
    }
    // fn get_sigma_points(&self, state_sigma_points: &DMatrix<f64>) -> DMatrix<f64> {
    //     let num_sigma_points = state_sigma_points.ncols();
    //     let mut measurement_sigma_points = DMatrix::zeros(1, num_sigma_points);
    //
    //     for i in 0..num_sigma_points {
    //         let state = state_sigma_points.column(i);
    //         let lat = state[0];
    //         let lon = state[1];
    //
    //         measurement_sigma_points[(0, i)] = self
    //             .map
    //             .get_point(&lat.to_degrees(), &lon.to_degrees())
    //             .unwrap_or(f64::NAN);
    //     }
    //     measurement_sigma_points
    // }
}
//================= Geophysical Navigation Simulation ======================================================
/// Builds and initializes an event stream that also contains geophysical measurements
///
/// This function builds a generic geophysical measurement model and adds it to the event stream.
/// The geophysical measurement model is initialized with the provided map and noise standard deviation.
///
/// # Arguments
/// * `records` - Vector of test data records
/// * `cfg` - GNSS degradation configuration
/// * `geomap` - Geophysical map for measurements
/// * `geo_noise_std` - Standard deviation for geophysical measurement noise
/// * `geo_frequency_s` - Frequency in seconds for geophysical measurements (None for every available measurement)
pub fn build_event_stream(
    records: &[TestDataRecord],
    cfg: &GnssDegradationConfig,
    geomap: Rc<GeoMap>,
    geo_noise_std: Option<f64>,
    geo_frequency_s: Option<f64>,
) -> EventStream {
    let start_time = records[0].time;
    let records_with_elapsed: Vec<(f64, &TestDataRecord)> = records
        .iter()
        .map(|r| ((r.time - start_time).num_milliseconds() as f64 / 1000.0, r))
        .collect();
    let mut events = Vec::with_capacity(records_with_elapsed.len() * 2);
    let mut st = FaultState::new(cfg.seed);

    // Scheduler state
    let mut next_emit_time = match cfg.scheduler {
        GnssScheduler::PassThrough => 0.0,
        GnssScheduler::FixedInterval { phase_s, .. } => phase_s,
        GnssScheduler::DutyCycle { start_phase_s, .. } => start_phase_s,
    };
    let mut duty_on = true;

    // Geophysical measurement scheduling state
    let mut next_geo_time = 0.0;
    // Through preprocessing we assert that the first record must have a NED position
    // but it may or may not have IMU or other such measurements.
    let reference_altitude = records[0].altitude;
    for w in records_with_elapsed.windows(2) {
        let (t0, _) = (&w[0].0, &w[0].1);
        let (t1, r1) = (&w[1].0, &w[1].1);
        let dt = t1 - t0;

        // Build IMU event at t1 only if accel and gyro components are present
        let imu_components = [
            r1.acc_x, r1.acc_y, r1.acc_z, r1.gyro_x, r1.gyro_y, r1.gyro_z,
        ];
        let imu_present = imu_components.iter().all(|v| !v.is_nan());
        if imu_present {
            let imu = IMUData {
                accel: Vector3::new(r1.acc_x, r1.acc_y, r1.acc_z),
                gyro: Vector3::new(r1.gyro_x, r1.gyro_y, r1.gyro_z),
                // add other fields as your UKF expects
            };
            events.push(Event::Imu {
                dt_s: dt,
                imu,
                elapsed_s: *t1,
            });
        }
        // Decide if GNSS should be emitted at t1
        let should_emit = match cfg.scheduler {
            GnssScheduler::PassThrough => true,
            GnssScheduler::FixedInterval { interval_s, .. } => {
                if *t1 + 1e-9 >= next_emit_time {
                    next_emit_time += interval_s;
                    true
                } else {
                    false
                }
            }
            GnssScheduler::DutyCycle { on_s, off_s, .. } => {
                let window = if duty_on { on_s } else { off_s };
                if *t1 + 1e-9 >= next_emit_time {
                    duty_on = !duty_on;
                    next_emit_time += window;
                    duty_on // only emit when toggling into ON
                } else {
                    false
                }
            }
        };

        if should_emit {
            // Only create GNSS event when the core GNSS values are present
            let gnss_required = [r1.latitude, r1.longitude, r1.altitude, r1.speed, r1.bearing];
            let gnss_present = gnss_required.iter().all(|v| !v.is_nan());
            if gnss_present {
                // Truth-like GNSS from r1
                let lat = r1.latitude;
                let lon = r1.longitude;
                let alt = r1.altitude;
                let bearing_rad = r1.bearing.to_radians();
                let vn = r1.speed * bearing_rad.cos();
                let ve = r1.speed * bearing_rad.sin();

                // Use your provided accuracies (adjust if these are variances vs std).
                // If an accuracy is missing (NaN), substitute a conservative default
                // to avoid propagating NaN into the measurement noise.
                let horiz_std = if r1.horizontal_accuracy.is_nan() {
                    1000.0
                } else {
                    r1.horizontal_accuracy.max(1e-3)
                };
                let vert_std = if r1.vertical_accuracy.is_nan() {
                    1000.0
                } else {
                    r1.vertical_accuracy.max(1e-3)
                };
                let vel_std = if r1.speed_accuracy.is_nan() {
                    100.0
                } else {
                    r1.speed_accuracy.max(0.1)
                };

                let (lat_c, lon_c, alt_c, vn_c, ve_c, horiz_c, vel_c) = apply_fault(
                    &cfg.fault, &mut st, *t1, dt, lat, lon, alt, vn, ve, horiz_std, vert_std,
                    vel_std,
                );

                let meas = GPSPositionAndVelocityMeasurement {
                    latitude: lat_c,
                    longitude: lon_c,
                    altitude: alt_c,
                    northward_velocity: vn_c,
                    eastward_velocity: ve_c,
                    horizontal_noise_std: horiz_c,
                    vertical_noise_std: vert_std, // pass-through here; you can also degrade it if desired
                    velocity_noise_std: vel_c,
                };
                events.push(Event::Measurement {
                    meas: Box::new(meas),
                    elapsed_s: *t1,
                });
            }
        }
        if !r1.relative_altitude.is_nan() {
            let baro: RelativeAltitudeMeasurement = RelativeAltitudeMeasurement {
                relative_altitude: r1.relative_altitude,
                reference_altitude,
            };
            events.push(Event::Measurement {
                meas: Box::new(baro),
                elapsed_s: *t1,
            });
        }
        let gravity = [r1.grav_x, r1.grav_y, r1.grav_z];
        let magnetic = [r1.mag_x, r1.mag_y, r1.mag_z];
        let gravity_present = gravity.iter().all(|v| !v.is_nan());
        let magnetic_present = magnetic.iter().all(|v| !v.is_nan());

        // Determine if we should emit a geophysical measurement at this time
        let should_emit_geo = match geo_frequency_s {
            Some(freq) => {
                if *t1 + 1e-9 >= next_geo_time {
                    next_geo_time += freq;
                    true
                } else {
                    false
                }
            }
            None => true, // Emit for every available measurement if no frequency specified
        };

        // Create geophysical measurements based on the map type
        if should_emit_geo {
            match &geomap.map_type {
                GeophysicalMeasurementType::Relief(_) => {
                    // Relief maps don't currently have specific measurements
                    // Could potentially be used for terrain-aided navigation
                }
                GeophysicalMeasurementType::Gravity(_) => {
                    if gravity_present {
                        // Calculate observed gravity magnitude
                        let observed_gravity =
                            (r1.grav_x.powi(2) + r1.grav_y.powi(2) + r1.grav_z.powi(2)).sqrt();
                        let meas = GravityMeasurement {
                            map: geomap.clone(),
                            noise_std: geo_noise_std.unwrap_or(100.0), // Use provided or default value
                            gravity_observed: observed_gravity,
                            latitude: f64::NAN, // to be set in closed-loop using state
                            altitude: f64::NAN, // to be set in closed-loop using state
                            north_velocity: f64::NAN, // to be set in closed-loop using state
                            east_velocity: f64::NAN, // to be set in closed-loop using state
                        };
                        events.push(Event::Measurement {
                            meas: Box::new(meas),
                            elapsed_s: *t1,
                        });
                    }
                }
                GeophysicalMeasurementType::Magnetic(_) => {
                    if magnetic_present {
                        let datetime = r1.time;
                        let observed_magnetic =
                            (r1.mag_x.powi(2) + r1.mag_y.powi(2) + r1.mag_z.powi(2)).sqrt();
                        let meas = MagneticAnomalyMeasurement {
                            map: geomap.clone(),
                            noise_std: geo_noise_std.unwrap_or(100.0), // Use provided or default value
                            mag_obs: observed_magnetic,
                            latitude: f64::NAN, // to be set in closed-loop using state
                            longitude: f64::NAN, // to be set in closed-loop using state
                            altitude: f64::NAN, // to be set in closed-loop using state
                            year: datetime.year(),
                            day: datetime.ordinal() as u16,
                        };
                        events.push(Event::Measurement {
                            meas: Box::new(meas),
                            elapsed_s: *t1,
                        });
                    }
                }
            }
        }
    }
    EventStream { start_time, events }
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
pub fn geo_closed_loop(
    ukf: &mut UnscentedKalmanFilter,
    stream: EventStream,
) -> anyhow::Result<Vec<NavigationResult>> {
    let start_time = stream.start_time;
    let mut results: Vec<NavigationResult> = Vec::with_capacity(stream.events.len());
    let total = stream.events.len();
    let mut last_ts: Option<DateTime<Utc>> = None;
    let mut monitor = HealthMonitor::new(HealthLimits::default());

    for (i, event) in stream.events.into_iter().enumerate() {
        // Print progress every 10 iterations
        if i % 10 == 0 || i == total {
            print!(
                "\rProcessing data {:.2}%...",
                (i as f64 / total as f64) * 100.0
            );
            //print_ukf(&ukf, record);
            use std::io::Write;
            std::io::stdout().flush().ok();
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
                ukf.predict(&imu, dt_s);
                if let Err(e) =
                    monitor.check(ukf.get_estimate().as_slice(), &ukf.get_certainty(), None)
                {
                    // log::error!("Health fail after predict at {} (#{i}): {e}", ts);
                    bail!(e);
                }
            }
            Event::Measurement { mut meas, .. } => {
                // Here we will need to sort out the measurement type
                // GPS measurements are handled by existing logic
                // GeophysicalMeasurements need to relate the measured vector/scalar
                // to the current state (lat, lon, alt) and the map
                if let Some(gravity) = meas.as_any_mut().downcast_mut::<GravityMeasurement>() {
                    // Handle GravityMeasurement-specific logic here if needed
                    // dbg!("Processing GravityMeasurement at time {}", ts);
                    let mean_vec = ukf.get_estimate();
                    // dbg!("Current State: {:?}", mean_vec.as_slice());
                    let mean = mean_vec.as_slice();
                    let strapdown: StrapdownState = (&mean[..9]).try_into().unwrap();
                    gravity.set_state(&strapdown);
                    ukf.update(gravity);
                } else if let Some(magnetic) = meas
                    .as_any_mut()
                    .downcast_mut::<MagneticAnomalyMeasurement>()
                {
                    // Handle MagneticAnomalyMeasurement-specific logic here if needed
                    //let mean: StrapdownState = ukf.get_mean().as_slice().try_into().unwrap();
                    let mean_vec = ukf.get_estimate();
                    let mean = mean_vec.as_slice();
                    let strapdown: StrapdownState = (&mean[..9]).try_into().unwrap();
                    magnetic.set_state(&strapdown);
                    ukf.update(magnetic);
                } else {
                    // Handle other built-in core measurement types (e.g., GPS, baro, etc.)
                    ukf.update(meas.as_ref());
                }
                if let Err(e) =
                    monitor.check(ukf.get_estimate().as_slice(), &ukf.get_certainty(), None)
                {
                    bail!(e);
                }
                // Health check after measurement update
                if let Err(e) =
                    monitor.check(ukf.get_estimate().as_slice(), &ukf.get_certainty(), None)
                {
                    // log::error!("Health fail after measurement update at {} (#{i}): {e}", ts);
                    bail!(e);
                }
            }
        }
        // If timestamp changed, or it's the last event, record the previous state
        if Some(ts) != last_ts {
            if let Some(prev_ts) = last_ts {
                results.push(NavigationResult::from((&prev_ts, &*ukf)));
            }
            last_ts = Some(ts);
        }
        // If this is the last event, also push
        if i == total - 1 {
            results.push(NavigationResult::from((&ts, &*ukf)));
        }
    }
    debug!("Geophysical navigation simulation complete");
    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert_approx_eq::assert_approx_eq;
    use chrono::{DateTime, Utc};
    use nalgebra::{DMatrix, DVector};
    use std::rc::Rc;
    use strapdown::earth::GP;
    use strapdown::messages::{
        GnssDegradationConfig, GnssFaultModel, GnssScheduler, MagnetometerConfig,
    };
    use strapdown::sim::TestDataRecord;

    /// Helper function to create a simple test gravity map
    fn create_test_gravity_map() -> GeoMap {
        let lats = DVector::from_vec(vec![40.0, 41.0, 42.0]);
        let lons = DVector::from_vec(vec![-74.0, -73.0, -72.0]);
        // Simple 3x3 gravity anomaly data in mGal
        let data = DMatrix::from_vec(3, 3, vec![-10.0, -5.0, 0.0, -5.0, 0.0, 5.0, 0.0, 5.0, 10.0]);
        let map_type = GeophysicalMeasurementType::Gravity(GravityResolution::OneMinute);

        GeoMap::new(lats, lons, data, map_type)
    }

    /// Helper function to create a simple test magnetic map
    fn create_test_magnetic_map() -> GeoMap {
        let lats = DVector::from_vec(vec![40.0, 41.0, 42.0]);
        let lons = DVector::from_vec(vec![-74.0, -73.0, -72.0]);
        // Simple 3x3 magnetic anomaly data in nT
        let data = DMatrix::from_vec(
            3,
            3,
            vec![-100.0, -50.0, 0.0, -50.0, 0.0, 50.0, 0.0, 50.0, 100.0],
        );
        let map_type = GeophysicalMeasurementType::Magnetic(MagneticResolution::TwoMinutes);

        GeoMap::new(lats, lons, data, map_type)
    }

    /// Helper function to create test data records
    fn create_test_records() -> Vec<TestDataRecord> {
        let mut records = Vec::new();
        let base_time = "2023-08-04T21:47:58+00:00"
            .parse::<DateTime<Utc>>()
            .unwrap();

        for i in 0..10 {
            let mut record = TestDataRecord::default();
            record.time = base_time + chrono::Duration::seconds(i);
            record.latitude = 40.5 + (i as f64) * 0.001; // Small movement
            record.longitude = -73.5 + (i as f64) * 0.001;
            record.altitude = 100.0 + (i as f64) * 0.1;
            record.speed = 5.0; // 5 m/s
            record.bearing = 45.0; // 45 degrees

            // IMU data
            record.acc_x = 0.1;
            record.acc_y = 0.1;
            record.acc_z = 9.8;
            record.gyro_x = 0.01;
            record.gyro_y = 0.01;
            record.gyro_z = 0.01;

            // Gravity data
            record.grav_x = 0.1;
            record.grav_y = 0.1;
            record.grav_z = 9.8;

            // Magnetic data
            record.mag_x = 20000.0; // micro teslas
            record.mag_y = 5000.0;
            record.mag_z = 45000.0;

            // Other measurements
            record.relative_altitude = i as f64 * 0.1;
            record.pressure = 1013.25 - (i as f64) * 0.1;
            record.horizontal_accuracy = 3.0;
            record.vertical_accuracy = 5.0;
            record.speed_accuracy = 0.1;

            records.push(record);
        }

        records
    }

    #[test]
    fn test_geomap_creation() {
        let map = create_test_gravity_map();

        assert_eq!(map.get_lats().len(), 3);
        assert_eq!(map.get_lons().len(), 3);
        assert_eq!(map.get_data().nrows(), 3);
        assert_eq!(map.get_data().ncols(), 3);

        match map.get_map_type() {
            GeophysicalMeasurementType::Gravity(_) => {}
            _ => panic!("Expected gravity map type"),
        }
    }

    #[test]
    fn test_geomap_interpolation() {
        let map = create_test_gravity_map();

        // Test exact point
        let value = map.get_point(&40.0, &-74.0);
        assert!(value.is_some());
        assert!((value.unwrap() - (-10.0)).abs() < 1e-10);

        // Test interpolated point within bounds
        let value = map.get_point(&40.5, &-73.5);
        assert!(value.is_some());
        // Should be interpolated value between surrounding points
        assert!(value.unwrap().abs() < 10.0);
    }

    #[test]
    fn test_gravity_anomaly_measurement() {
        let map = Rc::new(create_test_gravity_map());
        let measurement = GravityMeasurement {
            map: map.clone(),
            noise_std: 100.0,
            gravity_observed: GP,
            latitude: 0.0,
            altitude: 0.0,
            north_velocity: 3.5, // cos(45°) * 5 m/s
            east_velocity: 3.5,  // sin(45°) * 5 m/s
        };

        assert_eq!(measurement.get_dimension(), 1);

        let measurement_vector = measurement.get_vector();
        assert_eq!(measurement_vector.len(), 1);
        assert_approx_eq!(measurement_vector[0], 0.00, 0.1);

        let noise_matrix = measurement.get_noise();
        assert_eq!(noise_matrix.nrows(), 1);
        assert_eq!(noise_matrix.ncols(), 1);
        assert!((noise_matrix[(0, 0)] - 100.0_f64.powi(2)).abs() < 1e-3);
    }

    #[test]
    fn test_magnetic_anomaly_measurement() {
        let map = Rc::new(create_test_magnetic_map());
        let measurement = MagneticAnomalyMeasurement {
            map: map.clone(),
            noise_std: 100.0,
            mag_obs: (20000.0_f64.powi(2) + 5000.0_f64.powi(2) + 45000.0_f64.powi(2)).sqrt(),
            latitude: 40.5,
            longitude: -73.5,
            altitude: 100.0,
            year: 2023,
            day: 216, // August 4th
        };

        assert_eq!(measurement.get_dimension(), 1);

        let measurement_vector = measurement.get_vector();
        assert_eq!(measurement_vector.len(), 1);

        // Should compute magnitude: sqrt(20000^2 + 5000^2 + 45000^2)
        let expected_magnitude =
            (20000.0_f64.powi(2) + 5000.0_f64.powi(2) + 45000.0_f64.powi(2)).sqrt();
        assert_approx_eq!(measurement_vector[0], expected_magnitude, 1e-4);

        let noise_matrix = measurement.get_noise();
        assert_eq!(noise_matrix.nrows(), 1);
        assert_eq!(noise_matrix.ncols(), 1);
        assert!((noise_matrix[(0, 0)] - 100.0_f64.powi(2)).abs() < 1e-6);
    }

    #[test]
    fn test_measurement_sigma_points() {
        let map = Rc::new(create_test_gravity_map());
        let measurement = GravityMeasurement {
            map: map.clone(),
            noise_std: 100.0,
            gravity_observed: 9.8,
            latitude: f64::NAN,
            altitude: f64::NAN,
            north_velocity: 3.5,
            east_velocity: 3.5,
        };

        // Create mock sigma points (position states in radians and meters)
        let mut sigma_points = DMatrix::zeros(16, 5); // 16 states, 5 sigma points

        // Set position states (lat, lon, alt) for sigma points within map bounds
        sigma_points[(0, 0)] = 40.5_f64.to_radians(); // lat in radians
        sigma_points[(1, 0)] = (-73.5_f64).to_radians(); // lon in radians
        sigma_points[(2, 0)] = 100.0; // alt in meters

        sigma_points[(0, 1)] = 40.6_f64.to_radians();
        sigma_points[(1, 1)] = (-73.4_f64).to_radians();
        sigma_points[(2, 1)] = 101.0;

        sigma_points[(0, 2)] = 40.7_f64.to_radians();
        sigma_points[(1, 2)] = (-73.3_f64).to_radians();
        sigma_points[(2, 2)] = 102.0;

        sigma_points[(0, 3)] = 40.8_f64.to_radians();
        sigma_points[(1, 3)] = (-73.2_f64).to_radians();
        sigma_points[(2, 3)] = 103.0;

        sigma_points[(0, 4)] = 40.9_f64.to_radians();
        sigma_points[(1, 4)] = (-73.1_f64).to_radians();
        sigma_points[(2, 4)] = 104.0;

        // Set velocity states (vn, ve, vd)
        for i in 0..5 {
            sigma_points[(3, i)] = 3.5; // vn
            sigma_points[(4, i)] = 3.5; // ve
            sigma_points[(5, i)] = 0.0; // vd
        }

        // let measurement_sigma_points = measurement.get_sigma_points(&sigma_points);
        let num_sigma_points = sigma_points.ncols();
        let mut measurement_sigma_points = DMatrix::zeros(1, num_sigma_points);

        for i in 0..num_sigma_points {
            let state = sigma_points.column(i).into_owned();
            measurement_sigma_points[(0, i)] = measurement.get_expected_measurement(&state)[0];
        }

        assert_eq!(measurement_sigma_points.nrows(), 1);
        assert_eq!(measurement_sigma_points.ncols(), 5);

        // All measurements should be finite
        for i in 0..5 {
            assert!(measurement_sigma_points[(0, i)].is_finite());
        }
    }

    #[test]
    fn test_build_event_stream() {
        let records = create_test_records();
        let config = GnssDegradationConfig {
            scheduler: GnssScheduler::PassThrough,
            fault: GnssFaultModel::None,
            seed: 42,
            magnetometer: MagnetometerConfig::default(),
        };
        let geomap = Rc::new(create_test_gravity_map());

        let event_stream = build_event_stream(&records, &config, geomap, None, None);

        assert_eq!(event_stream.start_time, records[0].time);
        assert!(!event_stream.events.is_empty());

        // Should have IMU events
        let imu_events: Vec<_> = event_stream
            .events
            .iter()
            .filter(|e| matches!(e, Event::Imu { .. }))
            .collect();
        assert!(!imu_events.is_empty());

        // Should have measurement events (GNSS + geophysical)
        let measurement_events: Vec<_> = event_stream
            .events
            .iter()
            .filter(|e| matches!(e, Event::Measurement { .. }))
            .collect();
        assert!(!measurement_events.is_empty());
    }

    #[test]
    fn test_build_event_stream_magnetic() {
        let records = create_test_records();
        let config = GnssDegradationConfig {
            scheduler: GnssScheduler::PassThrough,
            fault: GnssFaultModel::None,
            seed: 42,
            magnetometer: MagnetometerConfig::default(),
        };
        let geomap = Rc::new(create_test_magnetic_map());

        let event_stream = build_event_stream(&records, &config, geomap, None, None);

        assert_eq!(event_stream.start_time, records[0].time);
        assert!(!event_stream.events.is_empty());

        // Check that we have magnetic measurement events
        let has_magnetic_measurements = event_stream.events.iter().any(|event| {
            if let Event::Measurement { meas, .. } = event {
                meas.as_any()
                    .downcast_ref::<MagneticAnomalyMeasurement>()
                    .is_some()
            } else {
                false
            }
        });

        assert!(
            has_magnetic_measurements,
            "Event stream should contain magnetic anomaly measurements"
        );
    }

    #[test]
    fn test_geophysical_measurement_type_display() {
        let gravity_type = GeophysicalMeasurementType::Gravity(GravityResolution::OneMinute);
        let magnetic_type = GeophysicalMeasurementType::Magnetic(MagneticResolution::TwoMinutes);

        assert_eq!(gravity_type.to_string(), "Gravity 01m");
        assert_eq!(magnetic_type.to_string(), "Magnetic 02m");
    }

    #[test]
    fn test_resolution_display() {
        assert_eq!(GravityResolution::OneMinute.to_string(), "01m");
        assert_eq!(GravityResolution::FiveMinutes.to_string(), "05m");
        assert_eq!(GravityResolution::OneDegree.to_string(), "01d");

        assert_eq!(MagneticResolution::TwoMinutes.to_string(), "02m");
        assert_eq!(MagneticResolution::TenMinutes.to_string(), "10m");
        assert_eq!(MagneticResolution::OneDegree.to_string(), "01d");
    }

    #[test]
    fn test_map_bounds_checking() {
        let map = create_test_gravity_map();

        // Test points within bounds
        assert!(map.get_point(&40.5, &-73.5).is_some());
        assert!(map.get_point(&41.0, &-73.0).is_some());

        // Test corner points (should be valid)
        assert!(map.get_point(&40.0, &-74.0).is_some()); // Bottom-left
        assert!(map.get_point(&42.0, &-72.0).is_some()); // Top-right
    }

    #[test]
    #[should_panic(expected = "Latitude out of bounds")]
    fn test_map_out_of_bounds_panic() {
        let map = create_test_gravity_map();
        // This should panic according to the current implementation
        map.get_point(&39.0, &-73.0);
    }

    #[test]
    fn test_configurable_noise_std() {
        let map = Rc::new(create_test_gravity_map());

        // Test different noise standard deviations
        let measurement1 = GravityMeasurement {
            map: map.clone(),
            noise_std: 50.0,
            gravity_observed: 9.8,
            latitude: f64::NAN,
            altitude: f64::NAN,
            north_velocity: 3.5,
            east_velocity: 3.5,
        };

        let measurement2 = GravityMeasurement {
            map: map.clone(),
            noise_std: 150.0,
            gravity_observed: 9.8,
            latitude: f64::NAN,
            altitude: f64::NAN,
            north_velocity: 3.5,
            east_velocity: 3.5,
        };

        let noise1 = measurement1.get_noise();
        let noise2 = measurement2.get_noise();

        assert!((noise1[(0, 0)] - 50.0_f64.powi(2)).abs() < 1e-10);
        assert!((noise2[(0, 0)] - 150.0_f64.powi(2)).abs() < 1e-10);
        assert!(noise2[(0, 0)] > noise1[(0, 0)]);
    }

    #[test]
    fn test_build_event_stream_custom_noise() {
        let records = create_test_records();
        let config = GnssDegradationConfig {
            scheduler: GnssScheduler::PassThrough,
            fault: GnssFaultModel::None,
            seed: 42,
            magnetometer: MagnetometerConfig::default(),
        };
        let geomap = Rc::new(create_test_gravity_map());

        // Test with custom noise standard deviation
        let event_stream = build_event_stream(&records, &config, geomap, Some(25.0), None);

        // Find a gravity measurement event and verify its noise
        let has_gravity_with_custom_noise = event_stream.events.iter().any(|event| {
            if let Event::Measurement { meas, .. } = event {
                if let Some(gravity_meas) = meas.as_any().downcast_ref::<GravityMeasurement>() {
                    (gravity_meas.noise_std - 25.0).abs() < 1e-10
                } else {
                    false
                }
            } else {
                false
            }
        });

        assert!(
            has_gravity_with_custom_noise,
            "Event stream should contain gravity measurements with custom noise std"
        );
    }

    #[test]
    fn test_build_event_stream_with_frequency() {
        let records = create_test_records();
        let config = GnssDegradationConfig {
            scheduler: GnssScheduler::PassThrough,
            fault: GnssFaultModel::None,
            seed: 42,
            magnetometer: MagnetometerConfig::default(),
        };
        let geomap = Rc::new(create_test_gravity_map());

        // Test with geophysical measurement frequency of 2 seconds
        let event_stream =
            build_event_stream(&records, &config, geomap.clone(), Some(25.0), Some(2.0));

        // Count gravity measurement events
        let gravity_events: Vec<_> = event_stream
            .events
            .iter()
            .filter(|event| {
                if let Event::Measurement { meas, .. } = event {
                    meas.as_any().downcast_ref::<GravityMeasurement>().is_some()
                } else {
                    false
                }
            })
            .collect();

        // Test with no frequency limit (should have more measurements)
        let event_stream_no_limit = build_event_stream(&records, &config, geomap, Some(25.0), None);

        let gravity_events_no_limit: Vec<_> = event_stream_no_limit
            .events
            .iter()
            .filter(|event| {
                if let Event::Measurement { meas, .. } = event {
                    meas.as_any().downcast_ref::<GravityMeasurement>().is_some()
                } else {
                    false
                }
            })
            .collect();

        // With frequency limit, we should have fewer or equal number of measurements
        assert!(gravity_events.len() <= gravity_events_no_limit.len());
    }
}
