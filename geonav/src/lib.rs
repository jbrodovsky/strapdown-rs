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
//! The measurement event stream can be constructed to include the gravity vector measurements from `TestDataRecord` (`grav_x`,
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
use nalgebra::{DMatrix, DVector};
use strapdown::StrapdownError;
use world_magnetic_model::GeomagneticField;
use world_magnetic_model::time::Date;
use world_magnetic_model::uom::si::angle::degree;
use world_magnetic_model::uom::si::f32::{Angle, Length};
use world_magnetic_model::uom::si::length::meter;
use world_magnetic_model::uom::si::magnetic_flux_density::nanotesla;

use strapdown::StrapdownState;
use strapdown::earth::gravity_anomaly;
use strapdown::measurements::MeasurementModel;
use strapdown::messages::{AidingConfig, Event, EventStream};
use strapdown::sim::TestDataRecord;

/// Conversion factor from radians to degrees (180/π)
const RAD_TO_DEG: f64 = 180.0 / std::f64::consts::PI;

/// Conversion from the microtesla the magnetometer reports to the nanotesla anomalies are in.
///
/// The crate keeps two magnetic units and they are not interchangeable. Body-frame magnetometer
/// readings are **microtesla** -- [`TestDataRecord`]'s `mag_x`/`mag_y`/`mag_z` and
/// [`strapdown::measurements::MagnetometerYawMeasurement`] both document that -- while anomaly
/// quantities are **nanotesla**: the maps [`GeoMap`] loads,
/// [`strapdown::sim::NavigationResult::magnetic_bias`],
/// and [`strapdown::sim::GeophysicalConfig`]'s `magnetic_bias` and `magnetic_noise_std`. An
/// anomaly is differenced against a map, so nanotesla is the unit this model works in and the
/// observation has to be converted on the way in.
const MICROTESLA_TO_NANOTESLA: f64 = 1000.0;

/// Navigation states every filter state vector starts with: position, velocity, attitude.
///
/// Anything a filter carries beyond these -- IMU biases, map biases -- is appended after
/// them, so this is the base a [`GeoBiasLayout`] is measured from for a filter that reports no
/// IMU bias states. Every filter in this workspace reports them, the RBPF included, so in
/// practice that base is [`NAVIGATION_AND_IMU_BIAS_STATE_DIM`].
pub const NAVIGATION_STATE_DIM: usize = 9;

/// The nine navigation states plus the six IMU bias states.
///
/// The base a [`GeoBiasLayout`] is measured from for the UKF, the EKF and the RBPF, whose
/// reported state is `[9 navigation, 3 accelerometer bias, 3 gyroscope bias, ..map biases]`.
pub const NAVIGATION_AND_IMU_BIAS_STATE_DIM: usize = 15;

/// Where a consuming filter carries one geophysical map-bias state.
///
/// A measurement model cannot discover this from the state vector it is handed. A filter's
/// state is a bare `DVector` with no labels, so index 14 of a 15-state EKF is the z gyro
/// bias while index 14 of a UKF carrying map biases is a map bias, and nothing in the vector
/// distinguishes them.
///
/// This type replaces a bare "count back N entries from the end", which assumed the state
/// ended with exactly the map biases the loaded maps implied -- a promise nothing checked.
/// Carrying `state_dim` next to the index turns it into one: [`resolve_bias_index`] rejects
/// a state vector of any other width, so a measurement built for one filter and handed to
/// another fails loudly instead of reading an IMU bias as a map bias.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BiasState {
    /// Width of the state vector the consuming filter hands the measurement models.
    pub state_dim: usize,
    /// Index of this bias within that vector; at or after [`NAVIGATION_STATE_DIM`].
    pub index: usize,
}

/// Where a consuming filter carries its geophysical map-bias states, if it carries any.
///
/// Built by whoever knows the filter -- `strapdown-sim` configures an RBPF whose
/// `map_bias_channels` are exactly these biases, and a UKF whose `other_states` are -- and
/// handed to [`build_event_stream`], which stamps it onto every geophysical measurement it
/// emits. Passing `None` there says the consuming filter carries no map-bias states, and
/// the measurements then declare none rather than inferring them from which maps happened
/// to be loaded.
///
/// Construct it with [`GeoBiasLayout::appended`] unless the filter puts its map biases
/// somewhere other than the end of its state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GeoBiasLayout {
    state_dim: usize,
    gravity_index: Option<usize>,
    magnetic_index: Option<usize>,
}

impl GeoBiasLayout {
    /// A layout with explicit indices into a state vector of `state_dim` entries.
    ///
    /// # Errors
    /// [`StrapdownError::InvalidConfiguration`] if `state_dim` cannot hold the nine
    /// navigation states, if an index falls inside them or at or past `state_dim`, or if the
    /// two biases name the same entry.
    pub fn new(
        state_dim: usize,
        gravity_index: Option<usize>,
        magnetic_index: Option<usize>,
    ) -> Result<Self, StrapdownError> {
        if state_dim < NAVIGATION_STATE_DIM {
            return Err(StrapdownError::InvalidConfiguration {
                field: "GeoBiasLayout::state_dim",
                reason: format!(
                    "a filter state is at least the {NAVIGATION_STATE_DIM} navigation states, \
                     got {state_dim}"
                ),
            });
        }
        for (index, field) in [
            (gravity_index, "GeoBiasLayout::gravity_index"),
            (magnetic_index, "GeoBiasLayout::magnetic_index"),
        ] {
            let Some(index) = index else { continue };
            if index < NAVIGATION_STATE_DIM || index >= state_dim {
                return Err(StrapdownError::InvalidConfiguration {
                    field,
                    reason: format!(
                        "a map bias lives after the {NAVIGATION_STATE_DIM} navigation states \
                         and inside the {state_dim}-state vector, so it cannot be at {index}"
                    ),
                });
            }
        }
        if gravity_index.is_some() && gravity_index == magnetic_index {
            return Err(StrapdownError::InvalidConfiguration {
                field: "GeoBiasLayout::magnetic_index",
                reason: "the gravity and magnetic biases are separate states and cannot share \
                         an index"
                    .to_owned(),
            });
        }
        Ok(Self {
            state_dim,
            gravity_index,
            magnetic_index,
        })
    }

    /// The layout of map biases appended after a filter's `base_state_dim` own states,
    /// gravity first -- the convention every filter in this workspace follows.
    ///
    /// `base_state_dim` is the width of the filter's state *before* the map biases:
    /// [`NAVIGATION_AND_IMU_BIAS_STATE_DIM`] for the UKF, EKF and RBPF, whose map biases
    /// follow their IMU biases, and [`NAVIGATION_STATE_DIM`] for a filter reporting none.
    ///
    /// Returns `Ok(None)` when neither map contributes a bias state, which is the "carries
    /// no map biases" case [`build_event_stream`] takes.
    ///
    /// # Errors
    /// [`StrapdownError::InvalidConfiguration`] if `base_state_dim` is smaller than the nine
    /// navigation states.
    pub fn appended(
        base_state_dim: usize,
        gravity: bool,
        magnetic: bool,
    ) -> Result<Option<Self>, StrapdownError> {
        if !gravity && !magnetic {
            return Ok(None);
        }
        let gravity_index = gravity.then_some(base_state_dim);
        let magnetic_index = magnetic.then(|| base_state_dim + usize::from(gravity));
        let state_dim = base_state_dim + usize::from(gravity) + usize::from(magnetic);
        Self::new(state_dim, gravity_index, magnetic_index).map(Some)
    }

    /// Width of the state vector this layout describes.
    #[must_use]
    pub const fn state_dim(&self) -> usize {
        self.state_dim
    }

    /// How many map-bias states the filter carries.
    #[must_use]
    pub const fn bias_count(&self) -> usize {
        self.gravity_index.is_some() as usize + self.magnetic_index.is_some() as usize
    }

    /// The gravity map bias, if the filter carries one.
    #[must_use]
    pub const fn gravity_bias(&self) -> Option<BiasState> {
        match self.gravity_index {
            Some(index) => Some(BiasState {
                state_dim: self.state_dim,
                index,
            }),
            None => None,
        }
    }

    /// The magnetic map bias, if the filter carries one.
    #[must_use]
    pub const fn magnetic_bias(&self) -> Option<BiasState> {
        match self.magnetic_index {
            Some(index) => Some(BiasState {
                state_dim: self.state_dim,
                index,
            }),
            None => None,
        }
    }
}

/// Resolve a declared [`BiasState`] against the state vector a filter actually handed over.
///
/// `None` means this model has no bias state and the map value is used as-is.
///
/// The width check is the point. A map bias used to be addressed by counting back from the
/// end of whatever vector arrived, which silently accepted any vector: a 9-vector with
/// "one from the end" resolved to `state[8]`, the yaw angle, and a 15-state EKF carrying no
/// map biases resolved to `state[14]`, the z gyro bias. Both were then added to the map
/// value as though they were a map bias. Requiring the width the layout was built for makes
/// a measurement built for one filter and handed to another an error rather than a
/// plausible-looking number.
///
/// # Errors
/// [`StrapdownError::DimensionMismatch`] when `state_len` is not the width the bias was
/// declared against, or when the declared index does not lie after the nine navigation
/// states and inside the vector -- unreachable through [`GeoBiasLayout`], which validates
/// that at construction, but [`BiasState`]'s fields are public.
pub const fn resolve_bias_index(
    state_len: usize,
    bias: Option<BiasState>,
) -> Result<Option<usize>, StrapdownError> {
    let Some(bias) = bias else {
        return Ok(None);
    };
    if state_len != bias.state_dim {
        return Err(StrapdownError::DimensionMismatch {
            what: "geophysical bias state: filter state width",
            expected: bias.state_dim,
            got: state_len,
        });
    }
    if bias.index < NAVIGATION_STATE_DIM || bias.index >= state_len {
        return Err(StrapdownError::DimensionMismatch {
            what: "geophysical bias state: index within the filter state",
            expected: state_len,
            got: bias.index,
        });
    }
    Ok(Some(bias.index))
}

/// World Magnetic Model valid altitude range (meters)
/// The WMM is typically valid from -1km below sea level to ~850km above
const WMM_MIN_ALTITUDE_M: f64 = -1000.0;
const WMM_MAX_ALTITUDE_M: f64 = 850000.0;

//================= Map Information ========================================================================
/// Resolution values for bathymetric or terrain relief maps
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReliefResolution {
    /// One-degree grid spacing; formats as `01d`.
    OneDegree,
    /// Thirty arc-minute grid spacing; formats as `30m`.
    ThirtyMinutes,
    /// Twenty arc-minute grid spacing; formats as `20m`.
    TwentyMinutes,
    /// Fifteen arc-minute grid spacing; formats as `15m`.
    FifteenMinutes,
    /// Ten arc-minute grid spacing; formats as `10m`.
    TenMinutes,
    /// Six arc-minute grid spacing; formats as `06m`.
    SixMinutes,
    /// Five arc-minute grid spacing; formats as `05m`.
    FiveMinutes,
    /// Four arc-minute grid spacing; formats as `04m`.
    FourMinutes,
    /// Three arc-minute grid spacing; formats as `03m`.
    ThreeMinutes,
    /// Two arc-minute grid spacing; formats as `02m`.
    TwoMinutes,
    /// One arc-minute grid spacing; formats as `01m`.
    OneMinute,
    /// Thirty arc-second grid spacing; formats as `30s`.
    ThirtySeconds,
    /// Fifteen arc-second grid spacing; formats as `15s`.
    FifteenSeconds,
    /// Three arc-second grid spacing; formats as `03s`.
    ThreeSeconds,
    /// One arc-second grid spacing; formats as `01s`. The finest relief grid offered.
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
    /// One-degree grid spacing; formats as `01d`.
    OneDegree,
    /// Thirty arc-minute grid spacing; formats as `30m`.
    ThirtyMinutes,
    /// Twenty arc-minute grid spacing; formats as `20m`.
    TwentyMinutes,
    /// Fifteen arc-minute grid spacing; formats as `15m`.
    FifteenMinutes,
    /// Ten arc-minute grid spacing; formats as `10m`.
    TenMinutes,
    /// Six arc-minute grid spacing; formats as `06m`.
    SixMinutes,
    /// Five arc-minute grid spacing; formats as `05m`.
    FiveMinutes,
    /// Four arc-minute grid spacing; formats as `04m`.
    FourMinutes,
    /// Three arc-minute grid spacing; formats as `03m`.
    ThreeMinutes,
    /// Two arc-minute grid spacing; formats as `02m`.
    TwoMinutes,
    /// One arc-minute grid spacing; formats as `01m`. The finest gravity grid offered.
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
    /// One-degree grid spacing; formats as `01d`.
    OneDegree,
    /// Thirty arc-minute grid spacing; formats as `30m`.
    ThirtyMinutes,
    /// Twenty arc-minute grid spacing; formats as `20m`.
    TwentyMinutes,
    /// Fifteen arc-minute grid spacing; formats as `15m`.
    FifteenMinutes,
    /// Ten arc-minute grid spacing; formats as `10m`.
    TenMinutes,
    /// Six arc-minute grid spacing; formats as `06m`.
    SixMinutes,
    /// Five arc-minute grid spacing; formats as `05m`.
    FiveMinutes,
    /// Four arc-minute grid spacing; formats as `04m`.
    FourMinutes,
    /// Three arc-minute grid spacing; formats as `03m`.
    ThreeMinutes,
    /// Two arc-minute grid spacing; formats as `02m`. The finest magnetic grid offered.
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
/// Enum for the different types of maps. A `GeoMap` is defined by its measurement type and resolution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GeophysicalMeasurementType {
    /// Bathymetric or terrain relief map at the given [`ReliefResolution`]; displays as `Relief <res>`.
    Relief(ReliefResolution),
    /// Gravity anomaly map at the given [`GravityResolution`]; displays as `Gravity <res>`.
    Gravity(GravityResolution),
    /// Magnetic anomaly map at the given [`MagneticResolution`]; displays as `Magnetic <res>`.
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
/// Struct for the `GeoMap` object.
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
    /// Create a new `GeoMap` object from the supplied latitudes, longitudes, data matrix, and map type
    ///
    /// # Arguments
    /// - `lats` - A vector of latitudes
    /// - `lons` - A vector of longitudes
    /// - `data` - A matrix of data values
    /// - `map_type` - The type of map (Relief, Gravity, Magnetic)
    ///
    /// # Returns
    /// - A new `GeoMap` object
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
    /// Load a `GeoMap` from a netcdf file. GMT processing does not encode the map type in the file, so this
    /// function requires the user to specify the type of map along with the filename.
    ///
    /// # Arguments
    /// - `filename` - The `PathBuf` of the netcdf file
    /// - `map_type` - The type of map (Relief, Gravity, Magnetic)
    ///
    /// # Returns
    /// - A Result containing a reference to the `GeoMap` object or an error message
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
    /// Bilinear interpolation helper method for `get_point`
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
    /// Cache the vehicle state that [`Self::get_anomaly`] needs.
    ///
    /// The event stream is built before any navigation estimate exists, so each implementor
    /// copies out only the fields its anomaly depends on: [`GravityMeasurement`] takes
    /// latitude, altitude and the north/east velocities (the Eötvös correction needs the
    /// velocity), [`MagneticAnomalyMeasurement`] takes the position the World Magnetic Model
    /// is evaluated at, and [`CombinedGeophysicalMeasurement`] delegates to both. `state`
    /// stores latitude and longitude in radians; an implementor whose underlying model takes
    /// degrees must convert. Both concrete implementors do: the World Magnetic Model and
    /// [`gravity_anomaly`] are each specified in degrees.
    fn set_state(&mut self, state: &StrapdownState);
}
/// Gravity measurement model
///
/// Computes the free-air anomaly at the current state by differencing the observed gravity
/// magnitude against Somigliana normal gravity, with the Eötvös correction for platform
/// motion ([`gravity_anomaly`]). The expected measurement is read from a [`GeoMap`].
///
/// # Units
///
/// This model works in **milligal**, because that is what the maps [`GeoMap`] loads are in.
/// [`gravity_observed`](Self::gravity_observed) is the one exception -- it is the raw
/// accelerometer magnitude in $m/s^2$, and [`gravity_anomaly`] converts as it differences.
///
/// That conversion was missing until the fix that added `earth::MGAL_PER_M_PER_S2`: the
/// observation arrived in $m/s^2$, the map value in milligal, and the innovation $z - h$ was
/// therefore just $-h$ to five significant figures. It never tripped a gate, because a
/// tens-of-milligal innovation against the default 100 mGal noise is a NIS of about 0.16.
/// This is the same defect the magnetic channel had and the same fix; see
/// [`MICROTESLA_TO_NANOTESLA`].
///
/// # Latitude units
///
/// [`gravity_anomaly`] takes **degrees**, while [`StrapdownState`] stores radians, so both
/// of this type's anomaly paths convert: [`GeophysicalAnomalyMeasurementModel::set_state`]
/// and the per-particle path through [`Self::extract_state_inputs`]. Neither did before
/// #330, which evaluated normal gravity near the equator whatever the true latitude -- a
/// -2136 mGal error at 40 deg N, against map anomalies of tens of mGal. (That figure was
/// itself written while the anomaly was still $m/s^2$; it is only literally true in milligal
/// now that the conversion above is applied.)
#[derive(Clone, Debug)]
pub struct GravityMeasurement {
    /// Source map
    pub map: Rc<GeoMap>,
    /// Measurement noise standard deviation, **milligal** -- the unit of the map this model
    /// differences against, not the $m/s^2$ of the accelerometer it reads.
    pub noise_std: f64,
    /// Observed gravity magnitude (m/s^2).
    ///
    /// The norm of the record's three `grav_*` axes, in SI. Converted to milligal by
    /// [`gravity_anomaly`], so this is the only field here that is not already milligal.
    pub gravity_observed: f64,
    /// Current latitude in **degrees**, converted from the radian-valued
    /// [`StrapdownState`] on the way in, because [`gravity_anomaly`] takes degrees (#330).
    latitude: f64,
    /// Current altitude (m)
    altitude: f64,
    /// Current north velocity (m/s)
    north_velocity: f64,
    /// Current east velocity (m/s)
    east_velocity: f64,
    /// Where the consuming filter carries this model's map bias, if it carries one.
    ///
    /// `None` means no bias state, and the map value is used as-is. A [`BiasState`] names
    /// both the index of the bias within the filter's state vector *and* the width of that
    /// vector, so handing this model to a filter of a different width is an error rather
    /// than a plausible-looking number -- see [`resolve_bias_index`]. Build one with
    /// [`GeoBiasLayout`] rather than by hand, which validates the placement up front.
    pub bias: Option<BiasState>,
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
        // Degrees. `gravity_anomaly` documents and uses degrees -- it forwards to `gravity`
        // and `eotvos`, both of which call `.to_radians()` internally -- while
        // `StrapdownState` stores radians. Passing the radian value through evaluated the
        // Somigliana model near the equator whatever the true latitude, a -2136 mGal error
        // at 40 deg N against map anomalies of tens of mGal (#330). Same conversion
        // `MagneticAnomalyMeasurement::set_state` has always done.
        self.latitude = state.latitude.to_degrees();
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
        // Validate the declared bias against the state actually handed over, and do it on
        // this method because it is the one method every filter calls. The EKF and the RBPF
        // also reach `get_jacobian`, which checks the same thing, but the UKF maps sigma
        // points through `get_expected_measurement` and asks for no Jacobian at all -- so
        // without this a UKF whose state does not carry the declared bias would silently
        // drop it and run an inconsistent model.
        resolve_bias_index(state.len(), self.bias)?;
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
        // Infallible by trait signature, so a state vector of the wrong width drops the
        // bias here rather than reporting it. The loud report is `get_measurement`, which
        // every filter calls and which validates the same thing: the EKF and RBPF also
        // reach it through `get_jacobian`, but the UKF maps sigma points and asks for no
        // Jacobian at all, so the Jacobian alone would leave that path silent.
        let bias = resolve_bias_index(state.len(), self.bias)
            .ok()
            .flatten()
            .map_or(0.0, |index| state[index]);
        DVector::from_vec(vec![map_value + bias])
    }

    fn get_jacobian(&self, state: &DVector<f64>) -> Result<DMatrix<f64>, StrapdownError> {
        self.get_jacobian_internal(state)
    }
}

impl GravityMeasurement {
    /// Pull the `gravity_anomaly` inputs out of a raw filter state vector.
    ///
    /// Returns `(latitude_deg, altitude_m, north_velocity, east_velocity)`. The latitude is
    /// converted, because `state[0]` is radians and `gravity_anomaly` takes degrees -- the
    /// per-particle half of #330.
    fn extract_state_inputs(state: &DVector<f64>) -> Option<(f64, f64, f64, f64)> {
        if state.len() >= 5
            && state[0].is_finite()
            && state[2].is_finite()
            && state[3].is_finite()
            && state[4].is_finite()
        {
            Some((state[0].to_degrees(), state[2], state[3], state[4]))
        } else {
            None
        }
    }
    /// Compute measurement Jacobian for EKF
    ///
    /// The first two columns (∂z/∂lat, ∂z/∂lon) carry the map gradient; when
    /// [`Self::bias`] is set, the column it addresses carries the 1.0 with which
    /// the bias enters the predicted measurement, and every other column is zero.
    ///
    /// # Width
    ///
    /// `1 × state.len()`, with the nine navigation states as a floor. A filter whose state
    /// is wider than the vector returned here zero-pads it -- see
    /// `strapdown::linearize::expand_measurement_jacobian` -- so returning the caller's own
    /// width is both correct and compatible. It has to be the caller's width rather than a
    /// fixed nine, because a declared bias column may lie outside the first nine.
    ///
    /// # Arguments
    ///
    /// * `state` - Current navigation state vector [lat, lon, alt, `v_n`, `v_e`, `v_d`, roll,
    ///   pitch, yaw], followed by whatever else the consuming filter carries. When
    ///   [`Self::bias`] is set, it names the index of the map bias within this vector and
    ///   the width this vector must have; both are checked rather than assumed.
    ///
    /// # Errors
    /// Propagates [`StrapdownError::OutOfMapBounds`] when the estimate has left the loaded
    /// tile. That is recoverable: the caller should skip this measurement, not abort.
    /// Returns [`StrapdownError::DimensionMismatch`] from `resolve_bias_index` when
    /// `state` is too narrow to carry the declared bias -- which is not recoverable, since
    /// it means the measurement and the filter disagree about the state layout.
    pub fn get_jacobian_internal(
        &self,
        state: &DVector<f64>,
    ) -> Result<DMatrix<f64>, StrapdownError> {
        // Resolve before touching the map: a layout disagreement is a wiring error and
        // should be reported whether or not the estimate also happens to be off-map.
        let bias_index = resolve_bias_index(state.len(), self.bias)?;
        let mut h = DMatrix::<f64>::zeros(1, state.len().max(NAVIGATION_STATE_DIM));

        let lat = state[0];
        let lon = state[1];

        // Compute numerical gradient from the geophysical map
        let (dlat_deg, dlon_deg) =
            self.map
                .get_gradient(&lat.to_degrees(), &lon.to_degrees(), 1e-6)?;

        // Convert gradient from per-degree to per-radian
        h[(0, 0)] = dlat_deg * RAD_TO_DEG;
        h[(0, 1)] = dlon_deg * RAD_TO_DEG;
        if let Some(index) = bias_index {
            h[(0, index)] = 1.0;
        }

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
    /// Observed total magnetic field magnitude, **in nanotesla**.
    ///
    /// Not a single component, despite what this said before: [`build_event_stream`] builds it
    /// as the norm of the record's three magnetometer axes. Those are microtesla, so the stream
    /// scales them by a thousand on the way in -- the anomaly this observation feeds is
    /// differenced against a nanotesla map, so it has to arrive in the map's unit.
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
    /// Where the consuming filter carries this model's map bias, if it carries one.
    ///
    /// `None` means no bias state, and the map value is used as-is. A [`BiasState`] names
    /// both the index of the bias within the filter's state vector *and* the width of that
    /// vector, so handing this model to a filter of a different width is an error rather
    /// than a plausible-looking number -- see [`resolve_bias_index`]. Build one with
    /// [`GeoBiasLayout`] rather than by hand, which validates the placement up front.
    pub bias: Option<BiasState>,
}
impl GeophysicalAnomalyMeasurementModel for MagneticAnomalyMeasurement {
    fn get_anomaly(&self) -> Result<f64, StrapdownError> {
        Ok(self.mag_obs - self.reference_field_nt(self.latitude, self.longitude, self.altitude)?)
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
        // Validate the declared bias against the state actually handed over, and do it on
        // this method because it is the one method every filter calls. The EKF and the RBPF
        // also reach `get_jacobian`, which checks the same thing, but the UKF maps sigma
        // points through `get_expected_measurement` and asks for no Jacobian at all -- so
        // without this a UKF whose state does not carry the declared bias would silently
        // drop it and run an inconsistent model.
        resolve_bias_index(state.len(), self.bias)?;
        // Return the observed magnetic anomaly as the measurement vector.
        // Use provided state if available (for per-particle updates), otherwise fallback to stored state.
        let anomaly = if let Some((lat_deg, lon_deg, alt)) = Self::extract_state_inputs(state) {
            self.mag_obs - self.reference_field_nt(lat_deg, lon_deg, alt)?
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
        // Infallible by trait signature, so a state vector of the wrong width drops the
        // bias here rather than reporting it. The loud report is `get_measurement`, which
        // every filter calls and which validates the same thing: the EKF and RBPF also
        // reach it through `get_jacobian`, but the UKF maps sigma points and asks for no
        // Jacobian at all, so the Jacobian alone would leave that path silent.
        let bias = resolve_bias_index(state.len(), self.bias)
            .ok()
            .flatten()
            .map_or(0.0, |index| state[index]);
        DVector::from_vec(vec![map_value + bias])
    }

    fn get_jacobian(&self, state: &DVector<f64>) -> Result<DMatrix<f64>, StrapdownError> {
        self.get_jacobian_internal(state)
    }
}

impl MagneticAnomalyMeasurement {
    /// The World Magnetic Model's total field at a position, **in nanotesla**.
    ///
    /// The one place this model evaluates the WMM. It had two, one per anomaly path, and both
    /// read the field as `magnetic_field.f().value` -- which is uom's *base* unit, tesla. A
    /// reference field of 5.1e-5 subtracted from an observation of tens of thousands is a no-op
    /// to eleven significant figures, so the reference removal that makes this an *anomaly*
    /// never happened and the raw field magnitude reached the filter against a map of anomalies.
    /// Asking uom for the unit by name rather than taking the raw `.value` is what makes that
    /// unrepresentable; having one call site is what keeps the two paths from disagreeing again.
    ///
    /// # Errors
    /// [`StrapdownError::ExternalModel`] when the date is invalid or the World Magnetic Model
    /// rejects the position. Altitude is clamped into the model's valid band first, so only a
    /// position the model genuinely cannot serve reaches the error.
    fn reference_field_nt(
        &self,
        latitude_deg: f64,
        longitude_deg: f64,
        altitude_m: f64,
    ) -> Result<f64, StrapdownError> {
        let alt_clamped = altitude_m.clamp(WMM_MIN_ALTITUDE_M, WMM_MAX_ALTITUDE_M);
        if (altitude_m - alt_clamped).abs() > 1.0 {
            log::warn!("Altitude {altitude_m} m out of WMM bounds, clamped to {alt_clamped} m");
        }

        let date = Date::from_ordinal_date(self.year, self.day).map_err(|e| {
            StrapdownError::ExternalModel {
                model: "WMM",
                detail: format!("invalid date (year {}, day {}): {e}", self.year, self.day),
            }
        })?;
        let magnetic_field = GeomagneticField::new(
            Length::new::<meter>(alt_clamped as f32),
            Angle::new::<degree>(latitude_deg as f32),
            Angle::new::<degree>(longitude_deg as f32),
            date,
        )
        .map_err(|e| StrapdownError::ExternalModel {
            model: "WMM",
            detail: format!(
                "unavailable at lat={latitude_deg}, lon={longitude_deg}, alt={altitude_m} \
                 (clamped {alt_clamped}): {e:?}"
            ),
        })?;
        Ok(f64::from(magnetic_field.f().get::<nanotesla>()))
    }

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
    /// The first two columns (∂z/∂lat, ∂z/∂lon) carry the map gradient; when
    /// [`Self::bias`] is set, the column it addresses carries the 1.0 with which
    /// the bias enters the predicted measurement, and every other column is zero.
    ///
    /// # Width
    ///
    /// `1 × state.len()`, with the nine navigation states as a floor. A filter whose state
    /// is wider than the vector returned here zero-pads it -- see
    /// `strapdown::linearize::expand_measurement_jacobian` -- so returning the caller's own
    /// width is both correct and compatible. It has to be the caller's width rather than a
    /// fixed nine, because a declared bias column may lie outside the first nine.
    ///
    /// # Arguments
    ///
    /// * `state` - Current navigation state vector [lat, lon, alt, `v_n`, `v_e`, `v_d`, roll,
    ///   pitch, yaw], followed by whatever else the consuming filter carries. When
    ///   [`Self::bias`] is set, it names the index of the map bias within this vector and
    ///   the width this vector must have; both are checked rather than assumed.
    ///
    /// # Errors
    /// Propagates [`StrapdownError::OutOfMapBounds`] when the estimate has left the loaded
    /// tile. That is recoverable: the caller should skip this measurement, not abort.
    /// Returns [`StrapdownError::DimensionMismatch`] from `resolve_bias_index` when
    /// `state` is too narrow to carry the declared bias -- which is not recoverable, since
    /// it means the measurement and the filter disagree about the state layout.
    pub fn get_jacobian_internal(
        &self,
        state: &DVector<f64>,
    ) -> Result<DMatrix<f64>, StrapdownError> {
        // Resolve before touching the map: a layout disagreement is a wiring error and
        // should be reported whether or not the estimate also happens to be off-map.
        let bias_index = resolve_bias_index(state.len(), self.bias)?;
        let mut h = DMatrix::<f64>::zeros(1, state.len().max(NAVIGATION_STATE_DIM));

        let lat = state[0];
        let lon = state[1];

        // Compute numerical gradient from the geophysical map
        let (dlat_deg, dlon_deg) =
            self.map
                .get_gradient(&lat.to_degrees(), &lon.to_degrees(), 1e-6)?;

        // Convert gradient from per-degree to per-radian
        h[(0, 0)] = dlat_deg * RAD_TO_DEG;
        h[(0, 1)] = dlon_deg * RAD_TO_DEG;
        if let Some(index) = bias_index {
            h[(0, index)] = 1.0;
        }

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
/// The record's total magnetic field magnitude, converted into the nanotesla anomalies use.
///
/// [`TestDataRecord`]'s three magnetometer axes are microtesla; the map this observation is
/// differenced against is nanotesla. The scaling used to be missing entirely, which -- together
/// with a reference field read in tesla -- left the "anomaly" as the raw field magnitude in the
/// wrong unit. See [`MICROTESLA_TO_NANOTESLA`].
fn observed_field_nt(record: &TestDataRecord) -> f64 {
    (record.mag_x.powi(2) + record.mag_y.powi(2) + record.mag_z.powi(2)).sqrt()
        * MICROTESLA_TO_NANOTESLA
}

/// The geophysical half of a `--geo` event stream: which maps aid it, how noisy they are,
/// how often they are read, and where the filter keeps their bias states.
///
/// A struct rather than six positional parameters, which is the shape `clippy.toml`'s
/// `too-many-arguments-threshold` asks for and the same move #296 made on `initialize_ekf`
/// and `initialize_eskf`. It also removes a real hazard: `gravity_noise_std` and
/// `magnetic_noise_std` are both `Option<f64>` and were adjacent, so transposing them was a
/// silent unit error -- milligal into a nanotesla slot.
///
/// Not `#[non_exhaustive]`, deliberately. That attribute buys the ability to add a field
/// without a major version bump, and `geonav` is held at 0.x precisely so its API can move
/// (see `geonav/Cargo.toml`), so it would be cost without the benefit.
#[derive(Clone, Debug, Default)]
pub struct GeophysicalAiding {
    /// Gravity anomaly map, or `None` for a run with no gravity aiding.
    pub gravity_map: Option<Rc<GeoMap>>,
    /// Gravity measurement noise, milligal. Defaults to 100.0 when `None`.
    pub gravity_noise_std: Option<f64>,
    /// Magnetic anomaly map, or `None` for a run with no magnetic aiding.
    pub magnetic_map: Option<Rc<GeoMap>>,
    /// Magnetic measurement noise, nanotesla. Defaults to 150.0 when `None`.
    pub magnetic_noise_std: Option<f64>,
    /// Seconds *between* geophysical measurements, so a larger value means fewer of them.
    /// `None` emits one for every record that carries the data.
    pub interval_s: Option<f64>,
    /// Where the filter consuming this stream keeps its map-bias states, or `None` when it
    /// keeps none. Not inferable from the maps: loading a gravity map says a gravity
    /// *measurement* is available, not that the filter has a state to absorb its bias.
    pub bias_layout: Option<GeoBiasLayout>,
}

/// Builds and initializes an event stream that also contains geophysical measurements
///
/// This function builds a generic geophysical measurement model and adds it to the event stream.
/// The geophysical measurement models are initialized with the provided maps and noise standard deviations.
/// Supports gravity-only, magnetic-only, combined (both), or no geophysical measurements.
///
/// # Arguments
/// * `records` - Vector of test data records
/// * `cfg` - Aiding configuration: the GNSS schedule and fault model, plus the barometer
///   and magnetometer schedules
/// * `gravity_map` - Optional gravity map for measurements
/// * `gravity_noise_std` - Standard deviation for gravity measurement noise (if `gravity_map` is Some)
/// * `magnetic_map` - Optional magnetic map for measurements
/// * `magnetic_noise_std` - Standard deviation for magnetic measurement noise (if `magnetic_map` is Some)
/// * `geo_interval_s` - Seconds *between* geophysical measurements, so a larger value means
///   fewer of them (None for every available measurement)
/// * `bias_layout` - Where the filter that will consume this stream carries its map-bias
///   states, or `None` when it carries none. This is not inferable from the maps: loading a
///   gravity map says a gravity *measurement* is available, not that the filter estimating
///   from it has a state to absorb that map's bias. Getting it from the caller, who knows
///   the filter, is what keeps a stream built for one filter from being read against
///   another's states -- see [`GeoBiasLayout`] and [`resolve_bias_index`].
///
/// # Errors
/// [`StrapdownError::InvalidConfiguration`] if `records` is empty. The first record supplies
/// both the stream's `start_time` and the reference altitude for relative-altitude
/// measurements, and neither has a defensible default. A slice of length one is *accepted*
/// and yields an empty event list, so the boundary is emptiness, not "fewer than two". This
/// mirrors [`strapdown::messages::build_event_stream`], which this function shadows with
/// geophysical measurements added.
pub fn build_event_stream(
    records: &[TestDataRecord],
    cfg: &AidingConfig,
    is_enu: bool,
    geophysical: &GeophysicalAiding,
) -> Result<EventStream, StrapdownError> {
    // Everything that is not geophysical comes from `strapdown::messages::build_event_stream`.
    //
    // This function used to reimplement it -- the elapsed clock, the GNSS schedule, the fault
    // model, the barometer, the IMU events -- and the two had diverged in three ways, all of
    // which changed what a `--geo` run actually simulated (#411):
    //
    // * `DutyCycle` emitted **one** fix per ON window instead of every fix during it, because
    //   the copy here toggled a `duty_on` flag on each emit and returned it, rather than
    //   deriving the window from the elapsed clock. `--sched duty --on-s 100 --off-s 50`
    //   delivered a fix every 150 s rather than for 100 s out of every 150.
    // * `start_phase_s` was destructured away with `..` and never applied.
    // * The barometer was emitted on **every record**, ignoring `cfg.baro_scheduler`
    //   entirely -- 50 updates a second on a 50 Hz log against the 1 Hz every other path
    //   uses -- and no magnetometer update was ever emitted at all.
    //
    // Delegating rather than re-fixing is the point: a second copy of a scheduler is a second
    // copy of every future scheduler bug. `the_two_builders_agree_on_every_non_geophysical_event`
    // holds them equal.
    let mut stream = strapdown::messages::build_event_stream(records, cfg, is_enu)?;

    // `build_event_stream` has already rejected an empty slice, so this cannot fail; it is
    // written as a `let ... else` rather than an index so that stays true if that changes.
    let Some(first) = records.first() else {
        return Ok(stream);
    };
    let start_time = first.time;

    // Whether a bias is *declared* comes from the layout the caller passed, not from which
    // maps were loaded. The two used to be the same expression, which is how a stream could
    // promise bias states a filter did not carry.
    let gravity_bias = geophysical
        .bias_layout
        .and_then(|layout| layout.gravity_bias());
    let magnetic_bias = geophysical
        .bias_layout
        .and_then(|layout| layout.magnetic_bias());
    let geo_interval_s = geophysical.interval_s;
    let gravity_map = geophysical.gravity_map.as_ref();
    let magnetic_map = geophysical.magnetic_map.as_ref();
    let gravity_noise_std = geophysical.gravity_noise_std;
    let magnetic_noise_std = geophysical.magnetic_noise_std;

    let mut geophysical: Vec<Event> = Vec::new();
    let mut next_geo_time = 0.0;

    for window in records.windows(2) {
        let r1 = &window[1];
        let t1 = (r1.time - start_time).num_milliseconds() as f64 / 1000.0;

        let gravity = [r1.grav_x, r1.grav_y, r1.grav_z];
        let magnetic = [r1.mag_x, r1.mag_y, r1.mag_z];
        let gravity_present = gravity.iter().all(|v| !v.is_nan());
        let magnetic_present = magnetic.iter().all(|v| !v.is_nan());

        // The geophysical channel keeps its own clock, on the same footing as the three in
        // the core builder: an interval in seconds *between* measurements, `None` meaning
        // every record that carries one.
        let should_emit_geo = match geo_interval_s {
            Some(interval) => {
                if t1 + 1e-9 >= next_geo_time {
                    next_geo_time += interval;
                    true
                } else {
                    false
                }
            }
            None => true,
        };
        if !should_emit_geo {
            continue;
        }

        // Bind the maps in the condition rather than testing `is_some()` and then
        // unwrapping: the availability test and the value then cannot drift apart.
        // `Option<&Rc<GeoMap>>` is `Copy`, so each branch may use these freely.
        let available_gravity = gravity_map.filter(|_| gravity_present);
        let available_magnetic = magnetic_map.filter(|_| magnetic_present);

        let observed_gravity = (r1.grav_x.powi(2) + r1.grav_y.powi(2) + r1.grav_z.powi(2)).sqrt();
        let datetime = r1.time;

        let measurement: Option<Box<dyn MeasurementModel>> =
            match (available_gravity, available_magnetic) {
                // Both maps available: emit a single combined 2D measurement.
                (Some(g_map), Some(m_map)) => Some(Box::new(CombinedGeophysicalMeasurement {
                    gravity: GravityMeasurement {
                        map: g_map.clone(),
                        noise_std: gravity_noise_std.unwrap_or(DEFAULT_GRAVITY_NOISE_MGAL),
                        gravity_observed: observed_gravity,
                        latitude: f64::NAN,
                        altitude: f64::NAN,
                        north_velocity: f64::NAN,
                        east_velocity: f64::NAN,
                        bias: gravity_bias,
                    },
                    magnetic: MagneticAnomalyMeasurement {
                        map: m_map.clone(),
                        noise_std: magnetic_noise_std.unwrap_or(DEFAULT_MAGNETIC_NOISE_NT),
                        mag_obs: observed_field_nt(r1),
                        latitude: f64::NAN,
                        longitude: f64::NAN,
                        altitude: f64::NAN,
                        year: datetime.year(),
                        day: datetime.ordinal() as u16,
                        bias: magnetic_bias,
                    },
                })),
                (Some(g_map), None) => Some(Box::new(GravityMeasurement {
                    map: g_map.clone(),
                    noise_std: gravity_noise_std.unwrap_or(DEFAULT_GRAVITY_NOISE_MGAL),
                    gravity_observed: observed_gravity,
                    latitude: f64::NAN,
                    altitude: f64::NAN,
                    north_velocity: f64::NAN,
                    east_velocity: f64::NAN,
                    bias: gravity_bias,
                })),
                (None, Some(m_map)) => Some(Box::new(MagneticAnomalyMeasurement {
                    map: m_map.clone(),
                    noise_std: magnetic_noise_std.unwrap_or(DEFAULT_MAGNETIC_NOISE_NT),
                    mag_obs: observed_field_nt(r1),
                    latitude: f64::NAN,
                    longitude: f64::NAN,
                    altitude: f64::NAN,
                    year: datetime.year(),
                    day: datetime.ordinal() as u16,
                    bias: magnetic_bias,
                })),
                (None, None) => None,
            };

        if let Some(meas) = measurement {
            geophysical.push(Event::Measurement {
                meas,
                elapsed_s: t1,
            });
        }
    }

    stream.events = merge_by_elapsed(std::mem::take(&mut stream.events), geophysical);
    Ok(stream)
}

/// Default gravity-measurement noise, milligal, when the caller names none.
///
/// Public so `strapdown-sim` can resolve a configuration file's omitted `gravity_noise_std`
/// to the same number this crate would. It used to be private, so the binary wrote `100.0`
/// itself in three places and this crate in a fourth.
///
/// Note that this value has never been measured against the maps it is differenced from --
/// `analyze geostats` in the `analysis` package derives one that has.
pub const DEFAULT_GRAVITY_NOISE_MGAL: f64 = 100.0;

/// Default magnetic-anomaly measurement noise, nanotesla, when the caller names none.
///
/// Public for the same reason as [`DEFAULT_GRAVITY_NOISE_MGAL`], and carrying the same
/// caveat: it is a default, not a measurement.
pub const DEFAULT_MAGNETIC_NOISE_NT: f64 = 150.0;

/// Elapsed time of an event, whichever variant it is.
const fn elapsed_of(event: &Event) -> f64 {
    match event {
        Event::Imu { elapsed_s, .. } | Event::Measurement { elapsed_s, .. } => *elapsed_s,
    }
}

/// Interleave the geophysical events into the core stream by elapsed time.
///
/// Both inputs are already sorted, so this is a merge rather than a sort, and it is **stable
/// with the geophysical event last at an equal timestamp**. That ordering is the one the
/// hand-rolled builder produced -- IMU, then GNSS, then barometer, then the geophysical
/// measurement -- and a filter that updates in event order would otherwise see the map
/// measurement applied before the GNSS fix at the same instant.
fn merge_by_elapsed(core: Vec<Event>, geophysical: Vec<Event>) -> Vec<Event> {
    let mut merged = Vec::with_capacity(core.len() + geophysical.len());
    let mut core = core.into_iter().peekable();
    let mut geophysical = geophysical.into_iter().peekable();

    loop {
        match (core.peek(), geophysical.peek()) {
            (Some(c), Some(g)) => {
                // `<=` keeps the geophysical event second at a tie.
                if elapsed_of(c) <= elapsed_of(g) {
                    merged.push(core.next().unwrap_or_else(|| unreachable!("peeked")));
                } else {
                    merged.push(geophysical.next().unwrap_or_else(|| unreachable!("peeked")));
                }
            }
            (Some(_), None) => merged.extend(core.by_ref()),
            (None, Some(_)) => merged.extend(geophysical.by_ref()),
            (None, None) => break,
        }
    }
    merged
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
    use strapdown::messages::{AidingConfig, GnssFaultModel, MeasurementScheduler};
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
                // Micro teslas, as `TestDataRecord` documents. These read 20000/5000/45000
                // under that same label, which is 400x Earth's field -- nanotesla numbers
                // wearing a microtesla unit, the same confusion the anomaly itself had.
                mag_x: 20.0,
                mag_y: 5.0,
                mag_z: 45.0,
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

    /// #330: both of `GravityMeasurement`'s anomaly paths must hand `gravity_anomaly` a
    /// latitude in **degrees**, not the radians `StrapdownState` stores.
    ///
    /// Asserted against an independently computed anomaly rather than against whatever the
    /// code returns, and at 40 deg N rather than the equator -- at 0 deg the bug is
    /// invisible, because 0 rad and 0 deg are the same number. That is exactly why the
    /// pre-existing `test_gravity_anomaly_measurement` above, which seeds `latitude: 0.0`,
    /// passed throughout.
    #[test]
    fn gravity_measurement_converts_latitude_to_degrees() {
        let latitude_deg = 40.0_f64;
        let altitude = 1000.0_f64;
        let north_velocity = 12.0_f64;
        let east_velocity = -4.0_f64;
        let observed = GP + 3.0e-4;

        let mut measurement = GravityMeasurement {
            map: Rc::new(create_test_gravity_map()),
            noise_std: 1.0,
            gravity_observed: observed,
            latitude: f64::NAN,
            altitude: f64::NAN,
            north_velocity: f64::NAN,
            east_velocity: f64::NAN,
            bias: None,
        };

        let state = StrapdownState::new(
            latitude_deg,
            -73.0,
            altitude,
            north_velocity,
            east_velocity,
            0.0,
            nalgebra::Rotation3::identity(),
            true, // in_degrees: the constructor converts to the radians the state stores
            Some(false),
        )
        .unwrap();

        // The oracle: `gravity_anomaly`'s own documented contract, evaluated in degrees.
        let expected = gravity_anomaly(
            &latitude_deg,
            &altitude,
            &north_velocity,
            &east_velocity,
            &observed,
        );

        // Path 1: the cached-state path.
        measurement.set_state(&state);
        assert_approx_eq!(measurement.get_anomaly().unwrap(), expected, 1e-12);

        // Path 2: the per-particle path, which reads the raw state vector.
        let state_vector: DVector<f64> = (&state).into();
        let per_particle = measurement.get_measurement(&state_vector).unwrap();
        assert_approx_eq!(per_particle[0], expected, 1e-12);

        // Non-degenerate: passing radians instead would be wrong by ~2100 mGal here, four
        // orders of magnitude above the tolerance above, so this test cannot pass by
        // accident the way an equatorial one would.
        let with_radians = gravity_anomaly(
            &latitude_deg.to_radians(),
            &altitude,
            &north_velocity,
            &east_velocity,
            &observed,
        );
        assert!(
            (with_radians - expected).abs() > 1e-2,
            "the radians/degrees confusion should be worth >1e-2 m/s^2 at 40 deg N, got {}",
            (with_radians - expected).abs()
        );
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
            bias: None,
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
            // Nanotesla, the unit `mag_obs` is in: a 49.5 uT reading scaled up.
            mag_obs: (20.0_f64.powi(2) + 5.0_f64.powi(2) + 45.0_f64.powi(2)).sqrt()
                * MICROTESLA_TO_NANOTESLA,
            latitude: 40.5,
            longitude: -73.5,
            altitude: 100.0,
            year: 2023,
            day: 216, // August 4th
            bias: None,
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
            bias: None,
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

    /// An empty slice must return the error, not index out of bounds (#311). The geonav copy
    /// of `build_event_stream` carried the identical defect, so it gets the identical test.
    #[test]
    fn empty_records_are_an_error_not_a_panic() {
        let config = {
            let mut built = AidingConfig::default();
            built.scheduler = MeasurementScheduler::PassThrough;
            built.fault = GnssFaultModel::None;
            built.seed = 42;
            built
        };
        let geomap = Rc::new(create_test_gravity_map());

        let err = build_event_stream(
            &[],
            &config,
            false,
            &GeophysicalAiding {
                gravity_map: Some(geomap),
                gravity_noise_std: None,
                magnetic_map: None,
                magnetic_noise_std: None,
                interval_s: None,
                bias_layout: None,
            },
        )
        .expect_err("an empty record slice cannot produce a stream");
        assert!(
            matches!(
                err,
                StrapdownError::InvalidConfiguration { field, .. } if field == "event stream records"
            ),
            "an empty record slice must report an invalid configuration, got: {err}"
        );
    }

    /// A single record is the boundary the guard must not move: it supplies `start_time` and
    /// the altitude reference, and the event list is empty because events are built from
    /// adjacent pairs. A guard written as `len() < 2` would wrongly reject this.
    #[test]
    fn single_record_yields_a_stream_with_no_events() {
        let records = create_test_records();
        let config = {
            let mut built = AidingConfig::default();
            built.scheduler = MeasurementScheduler::PassThrough;
            built.fault = GnssFaultModel::None;
            built.seed = 42;
            built
        };
        let geomap = Rc::new(create_test_gravity_map());

        let event_stream = build_event_stream(
            &records[..1],
            &config,
            false,
            &GeophysicalAiding {
                gravity_map: Some(geomap),
                gravity_noise_std: None,
                magnetic_map: None,
                magnetic_noise_std: None,
                interval_s: None,
                bias_layout: None,
            },
        )
        .unwrap();

        assert_eq!(event_stream.start_time, records[0].time);
        assert!(
            event_stream.events.is_empty(),
            "one record spans no interval, so it can produce no events"
        );
    }

    #[test]
    fn test_build_event_stream() {
        let records = create_test_records();
        let config = {
            let mut built = AidingConfig::default();
            built.scheduler = MeasurementScheduler::PassThrough;
            built.fault = GnssFaultModel::None;
            built.seed = 42;
            built
        };
        let geomap = Rc::new(create_test_gravity_map());

        let event_stream = build_event_stream(
            &records,
            &config,
            false,
            &GeophysicalAiding {
                gravity_map: Some(geomap),
                gravity_noise_std: None,
                magnetic_map: None,
                magnetic_noise_std: None,
                interval_s: None,
                bias_layout: None,
            },
        )
        .unwrap();

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
        let config = {
            let mut built = AidingConfig::default();
            built.scheduler = MeasurementScheduler::PassThrough;
            built.fault = GnssFaultModel::None;
            built.seed = 42;
            built
        };
        let geomap = Rc::new(create_test_magnetic_map());

        let event_stream = build_event_stream(
            &records,
            &config,
            false,
            &GeophysicalAiding {
                gravity_map: None,
                gravity_noise_std: None,
                magnetic_map: Some(geomap),
                magnetic_noise_std: None,
                interval_s: None,
                bias_layout: None,
            },
        )
        .unwrap();

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

    fn gravity_measurement_with_bias(bias: Option<BiasState>) -> GravityMeasurement {
        GravityMeasurement {
            map: Rc::new(create_test_gravity_map()),
            noise_std: 1.0,
            gravity_observed: 9.8,
            latitude: 40.5,
            altitude: 100.0,
            north_velocity: 0.0,
            east_velocity: 0.0,
            bias,
        }
    }

    /// A magnetic model at a real position, with a physically sensible observed field.
    ///
    /// `mag_obs` is nanotesla: a 49.5 uT reading, which is what a magnetometer on the ground in
    /// Pennsylvania actually sees.
    fn magnetic_measurement_with_bias(bias: Option<BiasState>) -> MagneticAnomalyMeasurement {
        MagneticAnomalyMeasurement {
            map: Rc::new(create_test_magnetic_map()),
            noise_std: 1.0,
            mag_obs: 49.5 * MICROTESLA_TO_NANOTESLA,
            latitude: 40.5,
            longitude: -73.5,
            altitude: 100.0,
            year: 2025,
            day: 60,
            bias,
        }
    }

    /// A state vector positioned inside the test map whose tail beyond the nine navigation
    /// states is `extras`.
    fn state_with_extras(extras: &[f64]) -> DVector<f64> {
        let mut v = vec![
            40.5_f64.to_radians(),
            (-73.5_f64).to_radians(),
            100.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.9, // yaw, distinct from any bias below
        ];
        v.extend_from_slice(extras);
        DVector::from_vec(v)
    }

    /// The layout places the map biases after the filter's own states, gravity first.
    ///
    /// The indices here are the ones the old "count back from the end" arithmetic produced
    /// for the RBPF, so this pins the translation: it must describe the same states, only
    /// now saying which filter width it describes them for.
    #[test]
    fn test_geo_bias_layout_appends_biases_after_the_filter_states() {
        // Neither map contributes a bias: the filter carries none, and there is no layout.
        assert_eq!(
            GeoBiasLayout::appended(NAVIGATION_STATE_DIM, false, false).unwrap(),
            None
        );

        // A filter reporting no IMU bias states: the map biases follow the nine navigation
        // states.
        let both = GeoBiasLayout::appended(NAVIGATION_STATE_DIM, true, true)
            .unwrap()
            .unwrap();
        assert_eq!(both.state_dim(), 11);
        assert_eq!(both.bias_count(), 2);
        assert_eq!(
            both.gravity_bias(),
            Some(BiasState {
                state_dim: 11,
                index: 9
            })
        );
        assert_eq!(
            both.magnetic_bias(),
            Some(BiasState {
                state_dim: 11,
                index: 10
            })
        );

        // Magnetic only: it takes the first appended slot, not the second.
        let magnetic_only = GeoBiasLayout::appended(NAVIGATION_STATE_DIM, false, true)
            .unwrap()
            .unwrap();
        assert_eq!(magnetic_only.gravity_bias(), None);
        assert_eq!(
            magnetic_only.magnetic_bias(),
            Some(BiasState {
                state_dim: 10,
                index: 9
            })
        );

        // UKF/EKF: the map biases follow the six IMU bias states.
        let ukf = GeoBiasLayout::appended(NAVIGATION_AND_IMU_BIAS_STATE_DIM, true, true)
            .unwrap()
            .unwrap();
        assert_eq!(ukf.state_dim(), 17);
        assert_eq!(
            ukf.gravity_bias(),
            Some(BiasState {
                state_dim: 17,
                index: 15
            })
        );
        assert_eq!(
            ukf.magnetic_bias(),
            Some(BiasState {
                state_dim: 17,
                index: 16
            })
        );
    }

    /// A layout that does not describe a real state vector is rejected at construction.
    #[test]
    fn test_geo_bias_layout_rejects_impossible_placements() {
        // Inside the navigation states.
        assert!(matches!(
            GeoBiasLayout::new(11, Some(8), None),
            Err(StrapdownError::InvalidConfiguration { .. })
        ));
        // Past the end of the vector it claims to describe.
        assert!(matches!(
            GeoBiasLayout::new(11, Some(11), None),
            Err(StrapdownError::InvalidConfiguration { .. })
        ));
        // Two biases cannot be the same state.
        assert!(matches!(
            GeoBiasLayout::new(11, Some(9), Some(9)),
            Err(StrapdownError::InvalidConfiguration { .. })
        ));
        // A state vector too short to be a filter state at all.
        assert!(matches!(
            GeoBiasLayout::appended(4, true, false),
            Err(StrapdownError::InvalidConfiguration { .. })
        ));
    }

    /// A bias declared against one filter width must not be read against another's states.
    ///
    /// This is the failure the layout exists to prevent. A bias was addressed by counting
    /// back from the end of whatever vector arrived, so any width was accepted: nine states
    /// resolved "one from the end" to `state[8]`, the yaw angle, and a 15-state EKF carrying
    /// no map biases resolved it to `state[14]`, the z gyro bias. Both were added to the map
    /// value as though they were a map bias, and nothing said so.
    #[test]
    fn test_resolve_bias_index_requires_the_width_it_was_declared_for() {
        let rbpf = GeoBiasLayout::appended(NAVIGATION_STATE_DIM, true, false)
            .unwrap()
            .unwrap();
        let gravity = rbpf.gravity_bias();
        assert_eq!(resolve_bias_index(10, gravity).unwrap(), Some(9));

        // The nine-state summary that used to resolve to yaw.
        assert!(matches!(
            resolve_bias_index(9, gravity),
            Err(StrapdownError::DimensionMismatch {
                expected: 10,
                got: 9,
                ..
            })
        ));
        // A 15-state EKF, whose last entry is the z gyro bias.
        assert!(matches!(
            resolve_bias_index(15, gravity),
            Err(StrapdownError::DimensionMismatch {
                expected: 10,
                got: 15,
                ..
            })
        ));
        // Not recoverable: a caller cannot sensibly skip the fix and carry on, the wiring is
        // wrong.
        assert!(
            !resolve_bias_index(15, gravity)
                .unwrap_err()
                .is_recoverable()
        );

        // No bias declared: nothing to resolve, any width accepted.
        assert_eq!(resolve_bias_index(9, None).unwrap(), None);
        assert_eq!(resolve_bias_index(15, None).unwrap(), None);
    }

    /// The UKF never asks for a Jacobian, so `get_measurement` has to carry the check.
    ///
    /// `UnscentedKalmanFilter::update` maps sigma points through `get_expected_measurement`
    /// and takes its innovation from `get_measurement`; it calls `get_jacobian` nowhere. A
    /// contract enforced only on the Jacobian would therefore leave every UKF silently
    /// dropping a bias its state could not carry, which is the one thing the loud error is
    /// meant to prevent.
    #[test]
    fn test_gravity_get_measurement_reports_a_missing_bias_state() {
        let bias = GeoBiasLayout::appended(NAVIGATION_STATE_DIM, true, false)
            .unwrap()
            .unwrap()
            .gravity_bias();
        let measurement = gravity_measurement_with_bias(bias);
        let err = measurement
            .get_measurement(&state_with_extras(&[]))
            .unwrap_err();
        assert!(matches!(err, StrapdownError::DimensionMismatch { .. }));
        assert!(!err.is_recoverable());

        // Carried: no error, and the anomaly still comes back.
        assert!(
            measurement
                .get_measurement(&state_with_extras(&[7.0]))
                .is_ok()
        );
        // Not declared: any width is fine.
        assert!(
            gravity_measurement_with_bias(None)
                .get_measurement(&state_with_extras(&[]))
                .is_ok()
        );
    }

    /// A hand-built `BiasState` of the right width but a nonsense index is still rejected.
    ///
    /// [`GeoBiasLayout`] cannot produce one -- it validates at construction -- but
    /// `BiasState`'s fields are public, so the resolution has to stand on its own rather
    /// than trusting that every caller came through the layout.
    #[test]
    fn test_resolve_bias_index_rejects_an_index_outside_the_state() {
        // Inside the navigation states, at the width it claims.
        assert!(matches!(
            resolve_bias_index(
                11,
                Some(BiasState {
                    state_dim: 11,
                    index: 8
                })
            ),
            Err(StrapdownError::DimensionMismatch { .. })
        ));
        // Past the end of the vector it claims to describe.
        assert!(matches!(
            resolve_bias_index(
                11,
                Some(BiasState {
                    state_dim: 11,
                    index: 11
                })
            ),
            Err(StrapdownError::DimensionMismatch { .. })
        ));
    }

    /// The Jacobian is where a layout disagreement surfaces, because every filter in the
    /// workspace evaluates it before the expected measurement.
    #[test]
    fn test_gravity_jacobian_reports_a_missing_bias_state() {
        let bias = GeoBiasLayout::appended(NAVIGATION_STATE_DIM, true, false)
            .unwrap()
            .unwrap()
            .gravity_bias();
        let measurement = gravity_measurement_with_bias(bias);
        let err = measurement
            .get_jacobian(&state_with_extras(&[]))
            .unwrap_err();
        assert!(matches!(err, StrapdownError::DimensionMismatch { .. }));
        assert!(!err.is_recoverable());
    }

    /// A carried bias gets a unit column, because it enters the prediction additively.
    #[test]
    fn test_gravity_jacobian_carries_a_column_for_the_declared_bias() {
        let bias = GeoBiasLayout::appended(NAVIGATION_STATE_DIM, true, false)
            .unwrap()
            .unwrap()
            .gravity_bias();
        let h = gravity_measurement_with_bias(bias)
            .get_jacobian(&state_with_extras(&[7.0]))
            .unwrap();
        assert_eq!(h.nrows(), 1);
        assert_eq!(h.ncols(), 10, "the Jacobian must match the caller's width");
        assert_approx_eq!(h[(0, 9)], 1.0, 1e-12);
        for column in 2..9 {
            assert_approx_eq!(h[(0, column)], 0.0, 1e-12);
        }

        // No bias declared: nine columns, as before, and no unit entry anywhere.
        let h = gravity_measurement_with_bias(None)
            .get_jacobian(&state_with_extras(&[]))
            .unwrap();
        assert_eq!(h.ncols(), 9);
        for column in 2..9 {
            assert_approx_eq!(h[(0, column)], 0.0, 1e-12);
        }
    }

    /// The predicted measurement reads the bias state, not a navigation state.
    #[test]
    fn test_gravity_expected_measurement_reads_the_declared_bias() {
        let bias = GeoBiasLayout::appended(NAVIGATION_STATE_DIM, true, false)
            .unwrap()
            .unwrap()
            .gravity_bias();
        let unbiased = gravity_measurement_with_bias(None)
            .get_expected_measurement(&state_with_extras(&[]))[0];

        // yaw is 0.9 and the bias is 7.0, so the two readings are far apart.
        let biased = gravity_measurement_with_bias(bias)
            .get_expected_measurement(&state_with_extras(&[7.0]))[0];
        assert_approx_eq!(biased, unbiased + 7.0, 1e-9);

        // A state of the wrong width drops the bias rather than reading yaw. The loud report
        // is `get_jacobian`; this path only has to not invent a bias of 0.9.
        let dropped = gravity_measurement_with_bias(bias)
            .get_expected_measurement(&state_with_extras(&[]))[0];
        assert_approx_eq!(dropped, unbiased, 1e-9);
    }

    /// The WMM reference comes back in nanotesla, not in uom's base unit.
    ///
    /// This is the regression. Both anomaly paths read `magnetic_field.f().value`, and uom's
    /// `.value` is the *base* unit -- tesla. At this location the field is 5.0869e-5 T, so
    /// subtracting it from an observation of tens of thousands changed nothing to eleven
    /// significant figures: the reference removal silently did not happen and the filter was
    /// handed the raw field magnitude to difference against a map of anomalies.
    ///
    /// The band is the real invariant and it is what makes this unit-diagnostic: Earth's total
    /// field spans roughly 22,000-67,000 nT everywhere on the surface, so a value in tesla
    /// (5e-5), microtesla (51) or gauss (0.51) all fail it by orders of magnitude.
    #[test]
    fn test_magnetic_reference_field_is_nanotesla() {
        let measurement = magnetic_measurement_with_bias(None);
        let reference = measurement
            .reference_field_nt(40.05, -75.95, 100.0)
            .expect("the WMM must serve a position in Pennsylvania");

        assert!(
            (22_000.0..=67_000.0).contains(&reference),
            "the reference field must be a physical total field in nT, got {reference}"
        );
        // The value the current coefficient set gives at this position and epoch. Loose enough
        // to survive a WMM coefficient update, which moves this by nanoteslas; the defect moved
        // it by nine orders of magnitude.
        assert_approx_eq!(reference, 50_869.4, 50.0);
    }

    /// And the anomaly is the observation with that reference actually taken off it.
    ///
    /// `test_magnetic_anomaly_measurement` asserts the two anomaly paths agree with each other,
    /// which is exactly the shape of assertion that could not catch this: both paths were wrong
    /// in the same way, so they agreed. This one pins the value.
    #[test]
    fn test_magnetic_anomaly_removes_the_reference_field() {
        let mut measurement = magnetic_measurement_with_bias(None);
        measurement.latitude = 40.05;
        measurement.longitude = -75.95;
        measurement.altitude = 100.0;

        let reference = measurement
            .reference_field_nt(40.05, -75.95, 100.0)
            .unwrap();

        // An observation 120 nT above the reference is a 120 nT anomaly, which is the scale the
        // maps and `build_event_stream`'s `magnetic_noise_std` default of 150 are written in.
        measurement.mag_obs = reference + 120.0;
        assert_approx_eq!(measurement.get_anomaly().unwrap(), 120.0, 1e-6);

        // The per-state path takes its position from the state and must agree.
        let state = DVector::from_vec(vec![
            40.05_f64.to_radians(),
            (-75.95_f64).to_radians(),
            100.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]);
        assert_approx_eq!(measurement.get_measurement(&state).unwrap()[0], 120.0, 1e-6);

        // The defect's signature: the anomaly coming back as the raw observation, because
        // subtracting a tesla-valued reference from a nanotesla observation is a no-op.
        assert!(
            (measurement.get_anomaly().unwrap() - measurement.mag_obs).abs() > 1.0,
            "the reference field was not removed from the observation"
        );
    }

    /// A 2023 recording is referenced to WMM2020, and `analysis/geostats.py` agrees.
    ///
    /// Five of this repository's trajectories were recorded in 2023 and 2024. The crate carries
    /// WMM2020 for them and selects it by date. `geostats` loaded only pygeomag's default,
    /// WMM2025, which refuses those dates, and then dropped each such trajectory whole -- gravity
    /// included -- from the statistics `geo-adopt` writes into the configs. Its `self_check`
    /// asserts this same literal, so the two sides cannot pick different models again unnoticed.
    #[test]
    fn a_2023_recording_is_referenced_to_wmm2020() {
        let mut measurement = magnetic_measurement_with_bias(None);
        measurement.year = 2023;
        measurement.day = 182; // 1 July

        let reference = measurement
            .reference_field_nt(40.05, -75.95, 100.0)
            .unwrap();

        assert_approx_eq!(reference, 51_069.8, 1.0);
    }

    /// The event stream hands the model nanotesla, because the record is microtesla.
    #[test]
    fn test_event_stream_converts_the_magnetometer_to_nanotesla() {
        let records = create_test_records();
        let expected =
            (records[1].mag_x.powi(2) + records[1].mag_y.powi(2) + records[1].mag_z.powi(2)).sqrt()
                * MICROTESLA_TO_NANOTESLA;

        let stream = build_event_stream(
            &records,
            &AidingConfig::default(),
            false,
            &GeophysicalAiding {
                gravity_map: None,
                gravity_noise_std: None,
                magnetic_map: Some(Rc::new(create_test_magnetic_map())),
                magnetic_noise_std: None,
                interval_s: None,
                bias_layout: None,
            },
        )
        .unwrap();

        let observed = stream
            .events
            .iter()
            .find_map(|event| match event {
                Event::Measurement { meas, .. } => meas
                    .as_any()
                    .downcast_ref::<MagneticAnomalyMeasurement>()
                    .map(|m| m.mag_obs),
                Event::Imu { .. } => None,
            })
            .expect("the stream must carry a magnetic measurement");

        assert_approx_eq!(observed, expected, 1e-9);
        // A 49.5 uT record reaches the model as ~49,500 nT, not ~49.5.
        assert!(
            (20_000.0..=70_000.0).contains(&observed),
            "the observation must reach the model in nT, got {observed}"
        );
    }

    /// A stream built for a filter with no map-bias states must not declare one.
    ///
    /// The measurements used to take their bias declaration from which maps were loaded, so
    /// every geophysical stream promised bias states whether or not the filter consuming it
    /// had any. Passing `None` is how a caller says its filter carries none, and this is the
    /// assertion that the promise now follows the caller rather than the maps.
    #[test]
    fn test_event_stream_without_a_layout_declares_no_bias() {
        let records = create_test_records();
        let config = AidingConfig::default();
        let gravity = Rc::new(create_test_gravity_map());
        let magnetic = Rc::new(create_test_magnetic_map());

        let no_bias = build_event_stream(
            &records,
            &config,
            false,
            &GeophysicalAiding {
                gravity_map: Some(Rc::clone(&gravity)),
                gravity_noise_std: None,
                magnetic_map: Some(Rc::clone(&magnetic)),
                magnetic_noise_std: None,
                interval_s: None,
                bias_layout: None,
            },
        )
        .unwrap();
        let mut combined_seen = 0;
        for event in &no_bias.events {
            if let Event::Measurement { meas, .. } = event
                && let Some(combined) = meas
                    .as_any()
                    .downcast_ref::<CombinedGeophysicalMeasurement>()
            {
                combined_seen += 1;
                assert_eq!(combined.gravity.bias, None);
                assert_eq!(combined.magnetic.bias, None);
            }
        }
        assert!(
            combined_seen > 0,
            "the stream should carry geophysical measurements to check"
        );

        // The same stream built for a filter that does carry them declares exactly those.
        let layout = GeoBiasLayout::appended(NAVIGATION_STATE_DIM, true, true)
            .unwrap()
            .unwrap();
        let with_bias = build_event_stream(
            &records,
            &config,
            false,
            &GeophysicalAiding {
                gravity_map: Some(gravity),
                gravity_noise_std: None,
                magnetic_map: Some(magnetic),
                magnetic_noise_std: None,
                interval_s: None,
                bias_layout: Some(layout),
            },
        )
        .unwrap();
        for event in &with_bias.events {
            if let Event::Measurement { meas, .. } = event
                && let Some(combined) = meas
                    .as_any()
                    .downcast_ref::<CombinedGeophysicalMeasurement>()
            {
                assert_eq!(combined.gravity.bias, layout.gravity_bias());
                assert_eq!(combined.magnetic.bias, layout.magnetic_bias());
            }
        }
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
            bias: None,
        };

        let measurement2 = GravityMeasurement {
            map,
            noise_std: 150.0,
            gravity_observed: 9.8,
            latitude: f64::NAN,
            altitude: f64::NAN,
            north_velocity: 3.5,
            east_velocity: 3.5,
            bias: None,
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
        let config = {
            let mut built = AidingConfig::default();
            built.scheduler = MeasurementScheduler::PassThrough;
            built.fault = GnssFaultModel::None;
            built.seed = 42;
            built
        };
        let geomap = Rc::new(create_test_gravity_map());

        // Test with custom noise standard deviation
        let event_stream = build_event_stream(
            &records,
            &config,
            false,
            &GeophysicalAiding {
                gravity_map: Some(geomap),
                gravity_noise_std: Some(25.0),
                magnetic_map: None,
                magnetic_noise_std: None,
                interval_s: None,
                bias_layout: None,
            },
        )
        .unwrap();

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
        let config = {
            let mut built = AidingConfig::default();
            built.scheduler = MeasurementScheduler::PassThrough;
            built.fault = GnssFaultModel::None;
            built.seed = 42;
            built
        };
        let geomap = Rc::new(create_test_gravity_map());

        // Test with geophysical measurement frequency of 2 seconds
        let event_stream = build_event_stream(
            &records,
            &config,
            false,
            &GeophysicalAiding {
                gravity_map: Some(geomap.clone()),
                gravity_noise_std: Some(25.0),
                magnetic_map: None,
                magnetic_noise_std: None,
                interval_s: Some(2.0),
                bias_layout: None,
            },
        )
        .unwrap();

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
            false,
            &GeophysicalAiding {
                gravity_map: Some(geomap),
                gravity_noise_std: Some(25.0),
                magnetic_map: None,
                magnetic_noise_std: None,
                interval_s: None,
                bias_layout: None,
            },
        )
        .unwrap();

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
            bias: None,
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
            bias: None,
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
                bias: None,
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
                bias: None,
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
