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
//! For example: gravity anomaly calculation requires knowledge of the vehicle velocity, to make the Eötvös correction.
//! The measurement event stream can be constructed to include the gravity vector measurements from TestDataRecord (`grav_x``,
//! `grav_y`, `grav_z`), but these values are not the specific anomaly. The scalar gravity must be calculated and corrected
//! using the vehicle velocity (Eötvös correction) and the reference gravity at the current position (from a gravity map) to
//! calculate the free air anomaly.

// NOTE: The `velocity_particle` module is temporarily disabled while the particle-based
// velocity navigation algorithms are being refactored and validated. It will be
// re-enabled once the API is stabilized and comprehensive tests are in place.
// See core/src/particle.rs for the new implementation.
// pub mod velocity_particle;

use std::any::Any;
use std::fmt::{Debug, Display};
use std::path::Path;
use std::rc::Rc;

use anyhow::Result;
use chrono::Datelike;
use log::debug;
use nalgebra::{DMatrix, DVector, Vector3};
use strapdown::StrapdownError;
use world_magnetic_model::GeomagneticField;
use world_magnetic_model::time::Date;
use world_magnetic_model::uom::si::angle::degree;
use world_magnetic_model::uom::si::f32::{Angle, Length};
use world_magnetic_model::uom::si::length::meter;

use strapdown::earth::gravity_anomaly;
use strapdown::measurements::{
    GPSPositionAndVelocityMeasurement, MeasurementModel, RelativeAltitudeMeasurement,
};
use strapdown::messages::{
    Event, EventStream, FaultState, GnssDegradationConfig, GnssScheduler, apply_fault,
};
use strapdown::sim::TestDataRecord;
use strapdown::{IMUData, StrapdownState};

/// Conversion factor from radians to degrees (180/π)
const RAD_TO_DEG: f64 = 180.0 / std::f64::consts::PI;

/// World Magnetic Model valid altitude range (meters)
/// The WMM is typically valid from -1km below sea level to ~850km above
const WMM_MIN_ALTITUDE_M: f64 = -1000.0;
const WMM_MAX_ALTITUDE_M: f64 = 850000.0;

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
            Self::OneDegree => "01d",
            Self::ThirtyMinutes => "30m",
            Self::TwentyMinutes => "20m",
            Self::FifteenMinutes => "15m",
            Self::TenMinutes => "10m",
            Self::SixMinutes => "06m",
            Self::FiveMinutes => "05m",
            Self::FourMinutes => "04m",
            Self::ThreeMinutes => "03m",
            Self::TwoMinutes => "02m",
            Self::OneMinute => "01m",
            Self::ThirtySeconds => "30s",
            Self::FifteenSeconds => "15s",
            Self::ThreeSeconds => "03s",
            Self::OneSecond => "01s",
        };
        write!(f, "{res}")
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
            Self::OneDegree => "01d",
            Self::ThirtyMinutes => "30m",
            Self::TwentyMinutes => "20m",
            Self::FifteenMinutes => "15m",
            Self::TenMinutes => "10m",
            Self::SixMinutes => "06m",
            Self::FiveMinutes => "05m",
            Self::FourMinutes => "04m",
            Self::ThreeMinutes => "03m",
            Self::TwoMinutes => "02m",
            Self::OneMinute => "01m",
        };
        write!(f, "{res}")
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
            Self::OneDegree => "01d",
            Self::ThirtyMinutes => "30m",
            Self::TwentyMinutes => "20m",
            Self::FifteenMinutes => "15m",
            Self::TenMinutes => "10m",
            Self::SixMinutes => "06m",
            Self::FiveMinutes => "05m",
            Self::FourMinutes => "04m",
            Self::ThreeMinutes => "03m",
            Self::TwoMinutes => "02m",
        };
        write!(f, "{res}")
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
            Self::Relief(res) => write!(f, "Relief {res}"),
            Self::Gravity(res) => write!(f, "Gravity {res}"),
            Self::Magnetic(res) => write!(f, "Magnetic {res}"),
        }
    }
}
/// Struct for the GeoMap object.
///
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
    pub const fn new(
        lats: DVector<f64>,
        lons: DVector<f64>,
        data: DMatrix<f64>,
        map_type: GeophysicalMeasurementType,
    ) -> Self {
        Self {
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
    /// use std::path::{Path, PathBuf};
    /// let map = GeoMap::load_geomap(&PathBuf::from("path/to/file.nc"), GeophysicalMeasurementType::Relief(ReliefResolution::OneDegree));
    /// ```
    /// # Errors
    /// [`StrapdownError::MapLoad`] if the file cannot be opened, does not carry the `lat`,
    /// `lon` and `z` variables, or holds a `z` grid whose length is not `lat.len() *
    /// lon.len()`.
    ///
    /// This function always returned `Result`; until #254 every one of those paths panicked
    /// instead, so the `Err` variant was unreachable. The final shape check is new: a
    /// transposed or multi-band grid used to panic inside `DMatrix::from_row_slice`.
    pub fn load_geomap(
        filename: &Path,
        map_type: GeophysicalMeasurementType,
    ) -> Result<Self, StrapdownError> {
        let map_err = |detail: String| StrapdownError::MapLoad {
            path: filename.to_path_buf(),
            detail,
        };
        // Open the netcdf file
        let file = netcdf::open(filename)
            .map_err(|e| map_err(format!("could not open the NetCDF file: {e}")))?;
        // Get the lat/lon variables
        let lats = file
            .variable("lat")
            .ok_or_else(|| map_err("no variable named `lat`".to_owned()))?;
        let lons = file
            .variable("lon")
            .ok_or_else(|| map_err("no variable named `lon`".to_owned()))?;
        // Get the data variable
        let data = file
            .variable("z")
            .ok_or_else(|| map_err("no variable named `z`".to_owned()))?;
        // Conversion to basic types
        let lats: Vec<f64> = lats
            .get_values(..)
            .map_err(|e| map_err(format!("could not read `lat` as f64: {e}")))?;
        let lons: Vec<f64> = lons
            .get_values(..)
            .map_err(|e| map_err(format!("could not read `lon` as f64: {e}")))?;
        let data: Vec<f64> = data
            .get_values(..)
            .map_err(|e| map_err(format!("could not read `z` as f64: {e}")))?;
        // Convert the data to DVector
        let lats = DVector::from_vec(lats);
        let lons = DVector::from_vec(lons);
        // `from_row_slice` panics on a length mismatch, which happens for a transposed grid,
        // a pixel- versus gridline-registered grid, or one carrying a third dimension.
        if data.len() != lats.len() * lons.len() {
            return Err(map_err(format!(
                "`z` has {} values but `lat` x `lon` is {} x {} = {}",
                data.len(),
                lats.len(),
                lons.len(),
                lats.len() * lons.len()
            )));
        }
        // Convert the data to DMatrix
        let data = DMatrix::from_row_slice(lats.len(), lons.len(), &data);
        // Create the GeoMap object
        Ok(Self::new(lats, lons, data, map_type))
    }
    /// Get the latitude vector
    /// # Returns
    /// - A reference to the latitude vector
    pub const fn get_lats(&self) -> &DVector<f64> {
        &self.lats
    }
    /// Get the longitude vector
    /// # Returns
    /// - A reference to the longitude vector
    pub const fn get_lons(&self) -> &DVector<f64> {
        &self.lons
    }
    /// Get the data matrix
    /// # Returns
    /// - A reference to the data matrix
    pub const fn get_data(&self) -> &DMatrix<f64> {
        &self.data
    }
    /// Get the map type
    /// # Returns
    /// - A reference to the map type
    pub const fn get_map_type(&self) -> &GeophysicalMeasurementType {
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
    /// use std::path::{Path, PathBuf};
    ///
    /// let map = GeoMap::load_geomap(&PathBuf::from("path/to/file.nc"), GeophysicalMeasurementType::Relief(ReliefResolution::OneDegree));
    /// let value = map.get_point(&1.5, &1.5);
    /// ```
    ///
    /// # Errors
    /// * [`StrapdownError::NonFinite`] if either coordinate is `NaN` or infinite. Checked
    ///   *first*, and deliberately: every comparison against `NaN` is false, so `NaN` passed
    ///   straight through the bounds tests below and reached the index search, which then
    ///   found no element and panicked on `unwrap`. Geophysical measurements are constructed
    ///   with `NaN` position placeholders, so this was reachable in normal use.
    /// * [`StrapdownError::OutOfMapBounds`] if the point lies outside the loaded tile. This
    ///   is a routine condition — a filter estimate near a tile edge, or any particle in the
    ///   tail of the distribution — and [`StrapdownError::is_recoverable`] reports it as such
    ///   so callers skip the measurement rather than aborting the run.
    pub fn get_point(&self, lat: &f64, lon: &f64) -> Result<f64, StrapdownError> {
        if !lat.is_finite() {
            return Err(StrapdownError::NonFinite {
                what: "map query latitude",
            });
        }
        if !lon.is_finite() {
            return Err(StrapdownError::NonFinite {
                what: "map query longitude",
            });
        }
        // Check if the lat/lon are within the bounds of the map
        if lat < &self.lats[0] || lat > &self.lats[self.lats.len() - 1] {
            return Err(StrapdownError::OutOfMapBounds {
                axis: "latitude",
                value: *lat,
                min: self.lats[0],
                max: self.lats[self.lats.len() - 1],
            });
        }
        if lon < &self.lons[0] || lon > &self.lons[self.lons.len() - 1] {
            return Err(StrapdownError::OutOfMapBounds {
                axis: "longitude",
                value: *lon,
                min: self.lons[0],
                max: self.lons[self.lons.len() - 1],
            });
        }
        // Check if the lat/lon are at the origin or the end of the map
        if lat == &self.lats[0] && lon == &self.lons[0] {
            // If the lat/lon are at the origin, return the first data point
            return Ok(self.data[(0, 0)]);
        }
        if lat == &self.lats[self.lats.len() - 1] && lon == &self.lons[self.lons.len() - 1] {
            // If the lat/lon are at the end, return the last data point
            return Ok(self.data[(self.lats.len() - 1, self.lons.len() - 1)]);
        }
        // Structure the interpolation in a few different ways. If the lat/lon are on the edge
        // of the map, only interpolate using the coordinate that is not on the edge.
        //
        // Both indices are searched once here. The bounds and finiteness checks above
        // guarantee the searches succeed, so these are the only two `position` calls the
        // function needs -- the edge branches below used to repeat them verbatim.
        let lat_index =
            self.lats
                .iter()
                .position(|&x| x >= *lat)
                .ok_or(StrapdownError::OutOfMapBounds {
                    axis: "latitude",
                    value: *lat,
                    min: self.lats[0],
                    max: self.lats[self.lats.len() - 1],
                })?;
        let lon_index =
            self.lons
                .iter()
                .position(|&x| x >= *lon)
                .ok_or(StrapdownError::OutOfMapBounds {
                    axis: "longitude",
                    value: *lon,
                    min: self.lons[0],
                    max: self.lons[self.lons.len() - 1],
                })?;
        if lat == &self.lats[0] || lat == &self.lats[self.lats.len() - 1] {
            // If the latitude is on the edge, only interpolate using longitude
            // Special case for the edges of the map
            if lon_index == 0 {
                return Ok(self.data[(lat_index, lon_index)]);
            }
            let lon1_index = lon_index - 1;
            let a = self.data[(lat_index, lon_index)];
            let b = self.data[(lat_index, lon1_index)];
            debug!("Bilinear interpolation edge case - a: {a}, b: {b}");
            let lon_diff = self.lons[lon_index] - self.lons[lon1_index];
            let result = ((a - b) / lon_diff) * (lon - self.lons[lon1_index]) + b;
            return Ok(result);
        }
        if lon == &self.lons[0] || lon == &self.lons[self.lons.len() - 1] {
            // If the longitude is on the edge, only interpolate using latitude
            if lat_index == 0 {
                return Ok(self.data[(lat_index, lon_index)]);
            }
            let lat1_index = lat_index - 1;
            let a = self.data[(lat_index, lon_index)];
            let b = self.data[(lat1_index, lon_index)];
            let lat_diff = self.lats[lat_index] - self.lats[lat1_index];
            return Ok(((a - b) / lat_diff) * (lat - self.lats[lat1_index]) + b);
        }
        // If the lat/lon are not on the edge, use normal bilinear interpolation.
        // The surrounding indices are already known; passing them avoids two further
        // `position` searches and makes the `- 1` below provably safe.
        self.bilinear_interpolation(*lat, *lon, lat_index, lon_index)
    }
    /// Bilinear interpolation helper method for get_point
    ///
    /// `lat2_index` / `lon2_index` are the upper bracketing indices already located by
    /// [`Self::get_point`]. They are parameters rather than recomputed here because the
    /// `- 1` below underflows on a `usize` when the index is 0 — reachable for a coordinate
    /// one ULP above the first grid line, which the exact-equality edge tests miss.
    fn bilinear_interpolation(
        &self,
        lat: f64,
        lon: f64,
        lat2_index: usize,
        lon2_index: usize,
    ) -> Result<f64, StrapdownError> {
        if lat2_index == 0 {
            return Err(StrapdownError::OutOfMapBounds {
                axis: "latitude",
                value: lat,
                min: self.lats[0],
                max: self.lats[self.lats.len() - 1],
            });
        }
        if lon2_index == 0 {
            return Err(StrapdownError::OutOfMapBounds {
                axis: "longitude",
                value: lon,
                min: self.lons[0],
                max: self.lons[self.lons.len() - 1],
            });
        }
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
        Ok(w11 * q11 + w12 * q12 + w21 * q21 + w22 * q22)
    }

    /// Compute numerical gradient of the map at a given point (lat, lon)
    ///
    /// This function computes the partial derivatives ∂z/∂lat and ∂z/∂lon using
    /// finite differences. The gradient is used in EKF measurement Jacobians.
    ///
    /// # Arguments
    /// - `lat` - Latitude in degrees
    /// - `lon` - Longitude in degrees
    /// - `epsilon` - Step size for numerical differentiation (default: 1e-6 degrees)
    ///
    /// # Returns
    /// - A tuple (∂z/∂lat, ∂z/∂lon) representing the map gradient at the point
    ///
    /// # Example
    /// ```ignore
    /// let (dlat, dlon) = map.get_gradient(&40.5, &-73.5, 1e-6);
    /// ```
    /// # Errors
    /// Propagates [`StrapdownError::OutOfMapBounds`] / [`StrapdownError::NonFinite`] from
    /// [`Self::get_point`].
    ///
    /// Each sample used to be `?`, which looked defensive and was not: an
    /// off-map sample silently produced a **zero gradient**, which is a zero row in the EKF
    /// measurement Jacobian — an aiding measurement that quietly stops constraining anything
    /// rather than reporting that it cannot. Note also that the latitude branch below guards
    /// only the latitude, so an out-of-range *longitude* reached `get_point` regardless and
    /// panicked there.
    pub fn get_gradient(
        &self,
        lat: &f64,
        lon: &f64,
        epsilon: f64,
    ) -> Result<(f64, f64), StrapdownError> {
        // Central difference for latitude derivative
        let lat_plus = lat + epsilon;
        let lat_minus = lat - epsilon;

        // Check bounds before computing
        let dlat = if lat_minus >= self.lats[0] && lat_plus <= self.lats[self.lats.len() - 1] {
            let z_plus = self.get_point(&lat_plus, lon)?;
            let z_minus = self.get_point(&lat_minus, lon)?;
            (z_plus - z_minus) / (2.0 * epsilon)
        } else {
            // Fall back to forward/backward difference at boundaries
            let z_center = self.get_point(lat, lon)?;
            if lat_plus <= self.lats[self.lats.len() - 1] {
                let z_plus = self.get_point(&lat_plus, lon)?;
                (z_plus - z_center) / epsilon
            } else {
                let z_minus = self.get_point(&lat_minus, lon)?;
                (z_center - z_minus) / epsilon
            }
        };

        // Central difference for longitude derivative
        let lon_plus = lon + epsilon;
        let lon_minus = lon - epsilon;

        let dlon = if lon_minus >= self.lons[0] && lon_plus <= self.lons[self.lons.len() - 1] {
            let z_plus = self.get_point(lat, &lon_plus)?;
            let z_minus = self.get_point(lat, &lon_minus)?;
            (z_plus - z_minus) / (2.0 * epsilon)
        } else {
            // Fall back to forward/backward difference at boundaries
            let z_center = self.get_point(lat, lon)?;
            if lon_plus <= self.lons[self.lons.len() - 1] {
                let z_plus = self.get_point(lat, &lon_plus)?;
                (z_plus - z_center) / epsilon
            } else {
                let z_minus = self.get_point(lat, &lon_minus)?;
                (z_center - z_minus) / epsilon
            }
        };

        Ok((dlat, dlon))
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
    /// The anomaly value for the model's current state.
    ///
    /// # Errors
    /// [`StrapdownError::ExternalModel`] when an underlying geophysical model rejects the
    /// query — chiefly the World Magnetic Model, which has hard validity ranges in position
    /// and a coefficient epoch that expires. `core`'s magnetometer model already degrades
    /// gracefully on exactly this condition; geonav used to panic on it, on the per-particle
    /// path, so a single outlier particle ended the run.
    fn get_anomaly(&self) -> Result<f64, StrapdownError>;
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
    /// Optional bias state index from the end of the state vector.
    pub bias_from_end: Option<usize>,
}
/// Geophysical anomaly measurement model implementation for gravity. This trait provides a method to compute
/// the gravity anomaly given the current state. Free air anomaly correction needs knowledge of the vehicle
/// velocity to compute the Eotvos correction.
impl GeophysicalAnomalyMeasurementModel for GravityMeasurement {
    fn get_anomaly(&self) -> Result<f64, StrapdownError> {
        Ok(gravity_anomaly(
            &self.latitude,
            &self.altitude,
            &self.north_velocity,
            &self.east_velocity,
            &self.gravity_observed,
        ))
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
    fn get_measurement(&self, state: &DVector<f64>) -> Result<DVector<f64>, StrapdownError> {
        // Return the observed gravity anomaly as the measurement vector.
        // Use provided state if available (for per-particle updates), otherwise fallback to stored state.
        let anomaly = if let Some((lat, alt, v_n, v_e)) = Self::extract_state_inputs(state) {
            gravity_anomaly(&lat, &alt, &v_n, &v_e, &self.gravity_observed)
        } else {
            self.get_anomaly()?
        };
        Ok(DVector::from_vec(vec![anomaly]))
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
        let bias = self.bias_from_end.and_then(|offset| {
            if offset == 0 || state.len() < offset {
                None
            } else {
                Some(state[state.len() - offset])
            }
        });
        DVector::from_vec(vec![map_value + bias.unwrap_or(0.0)])
    }

    fn get_jacobian(&self, state: &DVector<f64>) -> Result<DMatrix<f64>, StrapdownError> {
        self.get_jacobian_internal(state)
    }
}

impl GravityMeasurement {
    fn extract_state_inputs(state: &DVector<f64>) -> Option<(f64, f64, f64, f64)> {
        if state.len() >= 5
            && state[0].is_finite()
            && state[2].is_finite()
            && state[3].is_finite()
            && state[4].is_finite()
        {
            Some((state[0], state[2], state[3], state[4]))
        } else {
            None
        }
    }
    /// Compute measurement Jacobian for EKF
    ///
    /// Returns 1×9 Jacobian matrix where only the first two columns (∂z/∂lat, ∂z/∂lon)
    /// are non-zero, representing how the map value changes with position.
    ///
    /// # Arguments
    ///
    /// * `state` - Current navigation state vector [lat, lon, alt, v_n, v_e, v_d, roll, pitch, yaw]
    ///
    /// # Returns
    ///
    /// 1×9 Jacobian matrix H for gravity anomaly measurement
    ///
    /// # Errors
    /// Propagates [`StrapdownError::OutOfMapBounds`] when the estimate has left the loaded
    /// tile. That is recoverable: the caller should skip this measurement, not abort.
    pub fn get_jacobian_internal(
        &self,
        state: &DVector<f64>,
    ) -> Result<DMatrix<f64>, StrapdownError> {
        let mut h = DMatrix::<f64>::zeros(1, 9);

        let lat = state[0];
        let lon = state[1];

        // Compute numerical gradient from the geophysical map
        let (dlat_deg, dlon_deg) =
            self.map
                .get_gradient(&lat.to_degrees(), &lon.to_degrees(), 1e-6)?;

        // Convert gradient from per-degree to per-radian
        h[(0, 0)] = dlat_deg * RAD_TO_DEG;
        h[(0, 1)] = dlon_deg * RAD_TO_DEG;

        Ok(h)
    }
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
    /// Optional bias state index from the end of the state vector.
    pub bias_from_end: Option<usize>,
}
impl GeophysicalAnomalyMeasurementModel for MagneticAnomalyMeasurement {
    fn get_anomaly(&self) -> Result<f64, StrapdownError> {
        // Clamp altitude to valid WMM range to prevent errors
        let alt_clamped = self.altitude.clamp(WMM_MIN_ALTITUDE_M, WMM_MAX_ALTITUDE_M);

        let date = Date::from_ordinal_date(self.year, self.day).map_err(|e| {
            StrapdownError::ExternalModel {
                model: "WMM",
                detail: format!("invalid date (year {}, day {}): {e}", self.year, self.day),
            }
        })?;
        let magnetic_field = GeomagneticField::new(
            Length::new::<meter>(alt_clamped as f32),
            Angle::new::<degree>(self.latitude as f32),
            Angle::new::<degree>(self.longitude as f32),
            date,
        )
        .map_err(|e| StrapdownError::ExternalModel {
            model: "WMM",
            detail: format!(
                "unavailable at lat={}, lon={}, alt={alt_clamped}: {e:?}",
                self.latitude, self.longitude
            ),
        })?;
        Ok(self.mag_obs - f64::from(magnetic_field.f().value))
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
    fn get_measurement(&self, state: &DVector<f64>) -> Result<DVector<f64>, StrapdownError> {
        // Return the observed magnetic anomaly as the measurement vector.
        // Use provided state if available (for per-particle updates), otherwise fallback to stored state.
        let anomaly = if let Some((lat_deg, lon_deg, alt)) = Self::extract_state_inputs(state) {
            // Clamp altitude to valid WMM range to prevent errors
            let alt_clamped = alt.clamp(WMM_MIN_ALTITUDE_M, WMM_MAX_ALTITUDE_M);

            if (alt - alt_clamped).abs() > 1.0 {
                log::warn!("Altitude {alt} m out of WMM bounds, clamped to {alt_clamped} m");
            }

            let date = Date::from_ordinal_date(self.year, self.day).map_err(|e| {
                StrapdownError::ExternalModel {
                    model: "WMM",
                    detail: format!("invalid date (year {}, day {}): {e}", self.year, self.day),
                }
            })?;
            let magnetic_field = GeomagneticField::new(
                Length::new::<meter>(alt_clamped as f32),
                Angle::new::<degree>(lat_deg as f32),
                Angle::new::<degree>(lon_deg as f32),
                date,
            )
            .map_err(|e| StrapdownError::ExternalModel {
                model: "WMM",
                detail: format!(
                    "unavailable at lat={lat_deg}, lon={lon_deg}, alt={alt} (clamped {alt_clamped}): {e:?}"
                ),
            })?;
            self.mag_obs - f64::from(magnetic_field.f().value)
        } else {
            self.get_anomaly()?
        };
        Ok(DVector::from_vec(vec![anomaly]))
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
        let bias = self.bias_from_end.and_then(|offset| {
            if offset == 0 || state.len() < offset {
                None
            } else {
                Some(state[state.len() - offset])
            }
        });
        DVector::from_vec(vec![map_value + bias.unwrap_or(0.0)])
    }

    fn get_jacobian(&self, state: &DVector<f64>) -> Result<DMatrix<f64>, StrapdownError> {
        self.get_jacobian_internal(state)
    }
}

impl MagneticAnomalyMeasurement {
    fn extract_state_inputs(state: &DVector<f64>) -> Option<(f64, f64, f64)> {
        if state.len() >= 3 && state[0].is_finite() && state[1].is_finite() && state[2].is_finite()
        {
            Some((state[0].to_degrees(), state[1].to_degrees(), state[2]))
        } else {
            None
        }
    }
    /// Compute measurement Jacobian for EKF
    ///
    /// Returns 1×9 Jacobian matrix where only the first two columns (∂z/∂lat, ∂z/∂lon)
    /// are non-zero, representing how the map value changes with position.
    ///
    /// # Arguments
    ///
    /// * `state` - Current navigation state vector [lat, lon, alt, v_n, v_e, v_d, roll, pitch, yaw]
    ///
    /// # Returns
    ///
    /// 1×9 Jacobian matrix H for magnetic anomaly measurement
    ///
    /// # Errors
    /// Propagates [`StrapdownError::OutOfMapBounds`] when the estimate has left the loaded
    /// tile. That is recoverable: the caller should skip this measurement, not abort.
    pub fn get_jacobian_internal(
        &self,
        state: &DVector<f64>,
    ) -> Result<DMatrix<f64>, StrapdownError> {
        let mut h = DMatrix::<f64>::zeros(1, 9);

        let lat = state[0];
        let lon = state[1];

        // Compute numerical gradient from the geophysical map
        let (dlat_deg, dlon_deg) =
            self.map
                .get_gradient(&lat.to_degrees(), &lon.to_degrees(), 1e-6)?;

        // Convert gradient from per-degree to per-radian
        h[(0, 0)] = dlat_deg * RAD_TO_DEG;
        h[(0, 1)] = dlon_deg * RAD_TO_DEG;

        Ok(h)
    }
}

/// Combined gravity and magnetic anomaly measurement model
///
/// When both gravity and magnetic anomaly maps are available, this model jointly processes
/// them as a single 2-dimensional measurement update. This enables cross-correlation between
/// the two modalities and provides a more statistically efficient update than processing them
/// sequentially as independent 1D measurements.
///
/// The measurement vector is `[gravity_anomaly, magnetic_anomaly]` and the noise covariance
/// is block-diagonal (assumes independence between gravity and magnetic noise).
#[derive(Clone, Debug)]
pub struct CombinedGeophysicalMeasurement {
    /// Gravity anomaly measurement model
    pub gravity: GravityMeasurement,
    /// Magnetic anomaly measurement model
    pub magnetic: MagneticAnomalyMeasurement,
}

impl GeophysicalAnomalyMeasurementModel for CombinedGeophysicalMeasurement {
    fn get_anomaly(&self) -> Result<f64, StrapdownError> {
        // Return gravity anomaly as the primary scalar value.
        // The combined model produces two anomalies, but this trait method
        // returns a single value for API compatibility.
        self.gravity.get_anomaly()
    }
    fn set_state(&mut self, state: &StrapdownState) {
        self.gravity.set_state(state);
        self.magnetic.set_state(state);
    }
}

impl MeasurementModel for CombinedGeophysicalMeasurement {
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn get_dimension(&self) -> usize {
        2
    }
    fn get_measurement(&self, state: &DVector<f64>) -> Result<DVector<f64>, StrapdownError> {
        let grav = self.gravity.get_measurement(state)?;
        let mag = self.magnetic.get_measurement(state)?;
        Ok(DVector::from_vec(vec![grav[0], mag[0]]))
    }
    fn get_noise(&self) -> DMatrix<f64> {
        let grav_noise = self.gravity.get_noise();
        let mag_noise = self.magnetic.get_noise();
        DMatrix::from_diagonal(&DVector::from_vec(vec![
            grav_noise[(0, 0)],
            mag_noise[(0, 0)],
        ]))
    }
    fn get_expected_measurement(&self, state: &DVector<f64>) -> DVector<f64> {
        let grav = self.gravity.get_expected_measurement(state);
        let mag = self.magnetic.get_expected_measurement(state);
        DVector::from_vec(vec![grav[0], mag[0]])
    }
    fn get_jacobian(&self, state: &DVector<f64>) -> Result<DMatrix<f64>, StrapdownError> {
        // Either sub-model going off-map fails the combined measurement: a half-populated
        // Jacobian would silently drop one of the two aiding channels.
        let grav_j = self.gravity.get_jacobian(state)?;
        let mag_j = self.magnetic.get_jacobian(state)?;
        let ncols = grav_j.ncols();
        let mut h = DMatrix::<f64>::zeros(2, ncols);
        h.row_mut(0).copy_from(&grav_j.row(0));
        h.row_mut(1).copy_from(&mag_j.row(0));
        Ok(h)
    }
}

//================= Geophysical Navigation Simulation ======================================================
/// Builds and initializes an event stream that also contains geophysical measurements
///
/// This function builds a generic geophysical measurement model and adds it to the event stream.
/// The geophysical measurement models are initialized with the provided maps and noise standard deviations.
/// Supports gravity-only, magnetic-only, combined (both), or no geophysical measurements.
///
/// # Arguments
/// * `records` - Vector of test data records
/// * `cfg` - GNSS degradation configuration
/// * `gravity_map` - Optional gravity map for measurements
/// * `gravity_noise_std` - Standard deviation for gravity measurement noise (if gravity_map is Some)
/// * `magnetic_map` - Optional magnetic map for measurements
/// * `magnetic_noise_std` - Standard deviation for magnetic measurement noise (if magnetic_map is Some)
/// * `geo_frequency_s` - Frequency in seconds for geophysical measurements (None for every available measurement)
#[allow(
    clippy::needless_pass_by_value,
    reason = "the `Rc<GeoMap>` handles are stored by the measurements this builds; cloning an \
              `Rc` is a refcount bump, so taking them by value is cheaper than borrowing and \
              cloning internally"
)]
pub fn build_event_stream(
    records: &[TestDataRecord],
    cfg: &GnssDegradationConfig,
    gravity_map: Option<Rc<GeoMap>>,
    gravity_noise_std: Option<f64>,
    magnetic_map: Option<Rc<GeoMap>>,
    magnetic_noise_std: Option<f64>,
    geo_frequency_s: Option<f64>,
) -> EventStream {
    let start_time = records[0].time;
    let bias_count = usize::from(gravity_map.is_some()) + usize::from(magnetic_map.is_some());
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
                    20.0
                } else {
                    r1.vertical_accuracy.max(1e-3)
                };
                let vel_std = if r1.speed_accuracy.is_nan() {
                    100.0
                } else {
                    r1.speed_accuracy.max(0.1)
                };

                let (lat_c, lon_c, alt_c, vn_c, ve_c, horiz_c, vel_c) = apply_fault(
                    &cfg.fault, &mut st, *t1, dt, lat, lon, alt, vn, ve, horiz_std, vel_std,
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

        // Create geophysical measurements based on loaded maps
        if should_emit_geo {
            // Bind the maps in the condition rather than testing `is_some()` and then
            // unwrapping: the availability test and the value then cannot drift apart.
            // `Option<&Rc<GeoMap>>` is `Copy`, so each branch may use these freely.
            let available_gravity = gravity_map.as_ref().filter(|_| gravity_present);
            let available_magnetic = magnetic_map.as_ref().filter(|_| magnetic_present);

            if let (Some(g_map), Some(m_map)) = (available_gravity, available_magnetic) {
                // Both maps available: emit a single combined 2D measurement
                let observed_gravity =
                    (r1.grav_x.powi(2) + r1.grav_y.powi(2) + r1.grav_z.powi(2)).sqrt();
                let datetime = r1.time;
                let observed_magnetic =
                    (r1.mag_x.powi(2) + r1.mag_y.powi(2) + r1.mag_z.powi(2)).sqrt();
                let meas = CombinedGeophysicalMeasurement {
                    gravity: GravityMeasurement {
                        map: g_map.clone(),
                        noise_std: gravity_noise_std.unwrap_or(100.0),
                        gravity_observed: observed_gravity,
                        latitude: f64::NAN,
                        altitude: f64::NAN,
                        north_velocity: f64::NAN,
                        east_velocity: f64::NAN,
                        bias_from_end: Some(bias_count), // bias_count == 2 when both maps present
                    },
                    magnetic: MagneticAnomalyMeasurement {
                        map: m_map.clone(),
                        noise_std: magnetic_noise_std.unwrap_or(150.0),
                        mag_obs: observed_magnetic,
                        latitude: f64::NAN,
                        longitude: f64::NAN,
                        altitude: f64::NAN,
                        year: datetime.year(),
                        day: datetime.ordinal() as u16,
                        bias_from_end: Some(1),
                    },
                };
                events.push(Event::Measurement {
                    meas: Box::new(meas),
                    elapsed_s: *t1,
                });
            } else if let Some(g_map) = available_gravity {
                // Gravity-only
                let observed_gravity =
                    (r1.grav_x.powi(2) + r1.grav_y.powi(2) + r1.grav_z.powi(2)).sqrt();
                let meas = GravityMeasurement {
                    map: g_map.clone(),
                    noise_std: gravity_noise_std.unwrap_or(100.0),
                    gravity_observed: observed_gravity,
                    latitude: f64::NAN,
                    altitude: f64::NAN,
                    north_velocity: f64::NAN,
                    east_velocity: f64::NAN,
                    bias_from_end: if bias_count > 0 {
                        Some(bias_count)
                    } else {
                        None
                    },
                };
                events.push(Event::Measurement {
                    meas: Box::new(meas),
                    elapsed_s: *t1,
                });
            } else if let Some(m_map) = available_magnetic {
                // Magnetic-only
                let datetime = r1.time;
                let observed_magnetic =
                    (r1.mag_x.powi(2) + r1.mag_y.powi(2) + r1.mag_z.powi(2)).sqrt();
                let meas = MagneticAnomalyMeasurement {
                    map: m_map.clone(),
                    noise_std: magnetic_noise_std.unwrap_or(150.0),
                    mag_obs: observed_magnetic,
                    latitude: f64::NAN,
                    longitude: f64::NAN,
                    altitude: f64::NAN,
                    year: datetime.year(),
                    day: datetime.ordinal() as u16,
                    bias_from_end: if bias_count > 0 { Some(1) } else { None },
                };
                events.push(Event::Measurement {
                    meas: Box::new(meas),
                    elapsed_s: *t1,
                });
            }
        }
    }
    EventStream { start_time, events }
}
// NOTE: `geo_closed_loop_ukf`, `geo_closed_loop_ekf` and `geo_closed_loop_rbpf` were
// removed in favour of `strapdown::sim::run_closed_loop`.
//
// They existed only to call `GeophysicalAnomalyMeasurementModel::set_state` before each
// filter update and to route EKF updates through a pass-through helper. Neither is needed:
// `get_measurement`, `get_expected_measurement` and `get_jacobian` all read the state vector
// the filter passes in, and the EKF already sources its Jacobian from the measurement model.
//
// Using the shared driver also brings geophysical runs configurable health and execution
// limits, structured logging, and the same result-recording behaviour as every other filter.

#[cfg(test)]
mod tests {
    use super::*;
    use assert_approx_eq::assert_approx_eq;
    use chrono::{DateTime, Utc};
    use nalgebra::{DMatrix, DVector};
    use std::rc::Rc;
    use strapdown::earth::GP;
    use strapdown::messages::{GnssDegradationConfig, GnssFaultModel, GnssScheduler};
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
            records.push(TestDataRecord {
                time: base_time + chrono::Duration::seconds(i),
                latitude: 40.5 + (i as f64) * 0.001, // Small movement
                longitude: -73.5 + (i as f64) * 0.001,
                altitude: 100.0 + (i as f64) * 0.1,
                speed: 5.0,    // 5 m/s
                bearing: 45.0, // 45 degrees
                acc_x: 0.1,
                acc_y: 0.1,
                acc_z: 9.8,
                gyro_x: 0.01,
                gyro_y: 0.01,
                gyro_z: 0.01,
                grav_x: 0.1,
                grav_y: 0.1,
                grav_z: 9.8,
                mag_x: 20000.0, // micro teslas
                mag_y: 5000.0,
                mag_z: 45000.0,
                relative_altitude: i as f64 * 0.1,
                pressure: 1013.25 - (i as f64) * 0.1,
                horizontal_accuracy: 3.0,
                vertical_accuracy: 5.0,
                speed_accuracy: 0.1,
                ..Default::default()
            });
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
        let value = map.get_point(&40.0, &-74.0).unwrap();
        assert!((value - (-10.0)).abs() < 1e-10);

        // Test interpolated point within bounds
        let value = map.get_point(&40.5, &-73.5).unwrap();
        // Should be interpolated value between surrounding points
        assert!(value.abs() < 10.0);
    }

    #[test]
    fn test_gravity_anomaly_measurement() {
        let map = Rc::new(create_test_gravity_map());
        let measurement = GravityMeasurement {
            map,
            noise_std: 100.0,
            gravity_observed: GP,
            latitude: 0.0,
            altitude: 0.0,
            north_velocity: 3.5, // cos(45°) * 5 m/s
            east_velocity: 3.5,  // sin(45°) * 5 m/s
            bias_from_end: None,
        };

        assert_eq!(measurement.get_dimension(), 1);

        // Dummy state for get_measurement (forces fallback to stored state)
        let dummy_state = DVector::from_vec(vec![f64::NAN; 9]);
        let measurement_vector = measurement.get_measurement(&dummy_state).unwrap();
        assert_eq!(measurement_vector.len(), 1);
        assert_approx_eq!(
            measurement_vector[0],
            measurement.get_anomaly().unwrap(),
            1e-6
        );

        let noise_matrix = measurement.get_noise();
        assert_eq!(noise_matrix.nrows(), 1);
        assert_eq!(noise_matrix.ncols(), 1);
        assert!((noise_matrix[(0, 0)] - 100.0_f64.powi(2)).abs() < 1e-3);
    }

    #[test]
    fn test_magnetic_anomaly_measurement() {
        let map = Rc::new(create_test_magnetic_map());
        let measurement = MagneticAnomalyMeasurement {
            map,
            noise_std: 100.0,
            mag_obs: (20000.0_f64.powi(2) + 5000.0_f64.powi(2) + 45000.0_f64.powi(2)).sqrt(),
            latitude: 40.5,
            longitude: -73.5,
            altitude: 100.0,
            year: 2023,
            day: 216, // August 4th
            bias_from_end: None,
        };

        assert_eq!(measurement.get_dimension(), 1);

        // Dummy state for get_measurement (forces fallback to stored state)
        let dummy_state = DVector::from_vec(vec![f64::NAN; 9]);
        let measurement_vector = measurement.get_measurement(&dummy_state).unwrap();
        assert_eq!(measurement_vector.len(), 1);

        assert_approx_eq!(
            measurement_vector[0],
            measurement.get_anomaly().unwrap(),
            1e-6
        );

        let noise_matrix = measurement.get_noise();
        assert_eq!(noise_matrix.nrows(), 1);
        assert_eq!(noise_matrix.ncols(), 1);
        assert!((noise_matrix[(0, 0)] - 100.0_f64.powi(2)).abs() < 1e-6);
    }

    #[test]
    fn test_measurement_sigma_points() {
        let map = Rc::new(create_test_gravity_map());
        let measurement = GravityMeasurement {
            map,
            noise_std: 100.0,
            gravity_observed: 9.8,
            latitude: f64::NAN,
            altitude: f64::NAN,
            north_velocity: 3.5,
            east_velocity: 3.5,
            bias_from_end: None,
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
        };
        let geomap = Rc::new(create_test_gravity_map());

        let event_stream =
            build_event_stream(&records, &config, Some(geomap), None, None, None, None);

        assert_eq!(event_stream.start_time, records[0].time);
        assert!(!event_stream.events.is_empty());

        // Should have IMU events
        assert!(
            event_stream
                .events
                .iter()
                .any(|e| matches!(e, Event::Imu { .. }))
        );

        // Should have measurement events (GNSS + geophysical)
        assert!(
            event_stream
                .events
                .iter()
                .any(|e| matches!(e, Event::Measurement { .. }))
        );
    }

    #[test]
    fn test_build_event_stream_magnetic() {
        let records = create_test_records();
        let config = GnssDegradationConfig {
            scheduler: GnssScheduler::PassThrough,
            fault: GnssFaultModel::None,
            seed: 42,
        };
        let geomap = Rc::new(create_test_magnetic_map());

        let event_stream =
            build_event_stream(&records, &config, None, None, Some(geomap), None, None);

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
        assert!(map.get_point(&40.5, &-73.5).is_ok());
        assert!(map.get_point(&41.0, &-73.0).is_ok());

        // Test corner points (should be valid)
        assert!(map.get_point(&40.0, &-74.0).is_ok()); // Bottom-left
        assert!(map.get_point(&42.0, &-72.0).is_ok()); // Top-right
    }

    #[test]
    fn test_map_out_of_bounds_errors() {
        let map = create_test_gravity_map();
        // Was `test_map_out_of_bounds_panic`. Leaving the loaded tile is a routine
        // condition for a filter estimate, not a defect, so it is now reported rather
        // than fatal -- and `is_recoverable` tells the caller to skip the measurement.
        let got = map.get_point(&39.0, &-73.0);
        assert!(
            matches!(
                got,
                Err(StrapdownError::OutOfMapBounds {
                    axis: "latitude",
                    ..
                })
            ),
            "expected OutOfMapBounds on latitude, got {got:?}"
        );
        assert!(got.unwrap_err().is_recoverable());
    }

    /// `NaN` used to reach the index search and panic on `unwrap`: every comparison
    /// against `NaN` is false, so it passed straight through the bounds guards. The
    /// geophysical measurement models are built with `NaN` position placeholders, so this
    /// was reachable in ordinary use rather than only under a diverged filter.
    #[test]
    fn test_map_rejects_non_finite_coordinates() {
        let map = create_test_gravity_map();
        assert!(matches!(
            map.get_point(&f64::NAN, &-73.0),
            Err(StrapdownError::NonFinite { .. })
        ));
        assert!(matches!(
            map.get_point(&40.5, &f64::NAN),
            Err(StrapdownError::NonFinite { .. })
        ));
        assert!(matches!(
            map.get_point(&f64::INFINITY, &-73.0),
            Err(StrapdownError::NonFinite { .. })
        ));
    }

    /// A coordinate one ULP above the first grid line skips the exact-equality edge
    /// branches and lands on `lat2_index == 0`, where the old `lat2_index - 1` underflowed
    /// a `usize`. It must produce an error or a value, never a panic.
    #[test]
    fn test_map_handles_coordinate_just_above_first_gridline() {
        let map = create_test_gravity_map();
        let first_lat = map.get_lats()[0];
        let just_above = f64::from_bits(first_lat.to_bits() + 1);
        // Either outcome is acceptable; panicking is not.
        let _ = map.get_point(&just_above, &-73.5);
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
            bias_from_end: None,
        };

        let measurement2 = GravityMeasurement {
            map,
            noise_std: 150.0,
            gravity_observed: 9.8,
            latitude: f64::NAN,
            altitude: f64::NAN,
            north_velocity: 3.5,
            east_velocity: 3.5,
            bias_from_end: None,
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
        };
        let geomap = Rc::new(create_test_gravity_map());

        // Test with custom noise standard deviation
        let event_stream = build_event_stream(
            &records,
            &config,
            Some(geomap),
            Some(25.0),
            None,
            None,
            None,
        );

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
        };
        let geomap = Rc::new(create_test_gravity_map());

        // Test with geophysical measurement frequency of 2 seconds
        let event_stream = build_event_stream(
            &records,
            &config,
            Some(geomap.clone()),
            Some(25.0),
            None,
            None,
            Some(2.0),
        );

        // Count gravity measurement events
        let gravity_events = event_stream
            .events
            .iter()
            .filter(|event| {
                if let Event::Measurement { meas, .. } = event {
                    meas.as_any().downcast_ref::<GravityMeasurement>().is_some()
                } else {
                    false
                }
            })
            .count();

        // Test with no frequency limit (should have more measurements)
        let event_stream_no_limit = build_event_stream(
            &records,
            &config,
            Some(geomap),
            Some(25.0),
            None,
            None,
            None,
        );

        let gravity_events_no_limit = event_stream_no_limit
            .events
            .iter()
            .filter(|event| {
                if let Event::Measurement { meas, .. } = event {
                    meas.as_any().downcast_ref::<GravityMeasurement>().is_some()
                } else {
                    false
                }
            })
            .count();

        // With frequency limit, we should have fewer or equal number of measurements
        assert!(gravity_events <= gravity_events_no_limit);
    }

    #[test]
    fn test_geomap_gradient() {
        let map = create_test_gravity_map();

        // Test gradient computation at center point
        let (dlat, dlon) = map.get_gradient(&41.0, &-73.0, 1e-6).unwrap();

        // Gradient should be non-zero for a non-constant map
        assert!(
            dlat.abs() > 1e-10 || dlon.abs() > 1e-10,
            "Expected non-zero gradient at center of map"
        );

        // Test gradient at corner (should handle boundaries gracefully)
        let (dlat_corner, dlon_corner) = map.get_gradient(&40.0, &-74.0, 1e-6).unwrap();
        assert!(
            dlat_corner.is_finite() && dlon_corner.is_finite(),
            "Gradient should be finite at map corners"
        );
    }

    #[test]
    fn test_gravity_measurement_jacobian() {
        let map = Rc::new(create_test_gravity_map());
        let measurement = GravityMeasurement {
            map,
            noise_std: 100.0,
            gravity_observed: 9.8,
            latitude: 41.0_f64.to_radians(),
            altitude: 100.0,
            north_velocity: 0.0,
            east_velocity: 0.0,
            bias_from_end: None,
        };

        // Create a state vector for Jacobian computation
        let state = DVector::from_vec(vec![
            41.0_f64.to_radians(),  // lat
            -73.0_f64.to_radians(), // lon
            100.0,                  // alt
            0.0,
            0.0,
            0.0, // velocities
            0.0,
            0.0,
            0.0, // attitude
        ]);

        let jacobian = measurement.get_jacobian_internal(&state).unwrap();

        // Jacobian should be 1x9
        assert_eq!(jacobian.nrows(), 1);
        assert_eq!(jacobian.ncols(), 9);

        // First two columns (lat, lon derivatives) may be non-zero
        // Other columns should be zero for direct position dependence
        for j in 2..9 {
            assert_approx_eq!(jacobian[(0, j)], 0.0, 1e-10);
        }
    }

    #[test]
    fn test_magnetic_measurement_jacobian() {
        let map = Rc::new(create_test_magnetic_map());
        let measurement = MagneticAnomalyMeasurement {
            map,
            noise_std: 100.0,
            mag_obs: 48000.0,
            latitude: 41.0,
            longitude: -73.0,
            altitude: 100.0,
            year: 2023,
            day: 216,
            bias_from_end: None,
        };

        // Create a state vector for Jacobian computation
        let state = DVector::from_vec(vec![
            41.0_f64.to_radians(),  // lat
            -73.0_f64.to_radians(), // lon
            100.0,                  // alt
            0.0,
            0.0,
            0.0, // velocities
            0.0,
            0.0,
            0.0, // attitude
        ]);

        let jacobian = measurement.get_jacobian_internal(&state).unwrap();

        // Jacobian should be 1x9
        assert_eq!(jacobian.nrows(), 1);
        assert_eq!(jacobian.ncols(), 9);

        // First two columns (lat, lon derivatives) may be non-zero
        // Other columns should be zero for direct position dependence
        for j in 2..9 {
            assert_approx_eq!(jacobian[(0, j)], 0.0, 1e-10);
        }
    }

    #[test]
    fn test_gravity_measurement_trait_jacobian() {
        // Test that the MeasurementModel trait method works correctly
        let map = Rc::new(create_test_gravity_map());
        let measurement: Box<dyn strapdown::measurements::MeasurementModel> =
            Box::new(GravityMeasurement {
                map,
                noise_std: 100.0,
                gravity_observed: 9.8,
                latitude: 41.0_f64.to_radians(),
                altitude: 100.0,
                north_velocity: 0.0,
                east_velocity: 0.0,
                bias_from_end: None,
            });

        let state = DVector::from_vec(vec![
            41.0_f64.to_radians(),
            -73.0_f64.to_radians(),
            100.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]);

        // Test that get_jacobian returns a valid Jacobian
        let jacobian = measurement.get_jacobian(&state).unwrap();
        assert_eq!(jacobian.nrows(), 1);
        assert_eq!(jacobian.ncols(), 9);
    }

    #[test]
    fn test_magnetic_measurement_trait_jacobian() {
        // Test that the MeasurementModel trait method works correctly
        let map = Rc::new(create_test_magnetic_map());
        let measurement: Box<dyn strapdown::measurements::MeasurementModel> =
            Box::new(MagneticAnomalyMeasurement {
                map,
                noise_std: 100.0,
                mag_obs: 48000.0,
                latitude: 41.0,
                longitude: -73.0,
                altitude: 100.0,
                year: 2023,
                day: 216,
                bias_from_end: None,
            });

        let state = DVector::from_vec(vec![
            41.0_f64.to_radians(),
            -73.0_f64.to_radians(),
            100.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]);

        // Test that get_jacobian returns a valid Jacobian
        let jacobian = measurement.get_jacobian(&state).unwrap();
        assert_eq!(jacobian.nrows(), 1);
        assert_eq!(jacobian.ncols(), 9);
    }
}
