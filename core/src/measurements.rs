//! Measurement-related code for the strapdown navigation system.
//!
//! This module defines generic measurement models and specific implementations
//! for GPS position, GPS velocity, combined GPS position and velocity, relative
//! altitude, and magnetometer-based yaw measurements. These models are used in
//! inertial navigation systems to process sensor data.

use crate::StrapdownError;
use crate::StrapdownState;
use crate::earth::METERS_TO_DEGREES;

use std::any::Any;
use std::fmt::{self, Debug, Display};

use nalgebra::{DMatrix, DVector, Rotation3, Vector3};
use world_magnetic_model::GeomagneticField;
use world_magnetic_model::time::Date;
use world_magnetic_model::uom::si::angle::degree;
use world_magnetic_model::uom::si::f32::{Angle, Length};
use world_magnetic_model::uom::si::length::meter;

pub const MAG_YAW_NOISE: f64 = 0.2; // radians

/// Date substituted when a record carries an unusable year/day-of-year pair.
///
/// The declination lookup degrades rather than failing here: a wrong date shifts declination
/// by a fraction of a degree, whereas refusing the measurement loses the heading aid
/// entirely. Built once, with a single documented allow, instead of an `unwrap` per call.
#[expect(
    clippy::expect_used,
    reason = "1 January 2025 is a valid ordinal date; this cannot fail"
)]
fn fallback_wmm_date() -> Date {
    Date::from_ordinal_date(2025, 1).expect("2025-001 is a valid ordinal date")
}

/// Build a [`StrapdownState`] from a 9-element state vector without range validation.
///
/// Measurement Jacobians are evaluated on whatever the filter's current estimate
/// happens to be, including estimates that have wandered outside the range the
/// mechanization is valid over. Constructing through `StrapdownState::new` would
/// assert in that situation, turning a filter-quality problem into a panic inside
/// the update step. Linearization itself is well defined for any finite state, so
/// build the value directly and leave range enforcement to state construction from
/// user input.
///
/// # Errors
/// [`StrapdownError::DimensionMismatch`] if `state` has fewer than 9 elements. The indexing
/// below is unconditional, so a short vector panicked here before #254.
fn jacobian_state(state: &DVector<f64>) -> Result<StrapdownState, StrapdownError> {
    if state.len() < 9 {
        return Err(StrapdownError::DimensionMismatch {
            what: "measurement Jacobian state vector",
            expected: 9,
            got: state.len(),
        });
    }
    Ok(StrapdownState {
        latitude: state[0],
        longitude: state[1],
        altitude: state[2],
        velocity_north: state[3],
        velocity_east: state[4],
        velocity_vertical: state[5],
        attitude: Rotation3::from_euler_angles(state[6], state[7], state[8]),
        // Frame follows the crate default (NED) rather than restating a bare `false`.
        // The filter state vector carries no frame tag, so this is the only convention
        // available here -- it agreed with nothing while the crate defaulted to ENU.
        ..StrapdownState::default()
    })
}

/// Generic measurement model trait for all types of measurements.
///
/// This trait defines the interface for measurement models used in Kalman-style
/// navigation filters. Measurement models define the relationship between the
/// state vector and observable measurements, following the probabilistic notation
/// $p(z|x)$ where $z$ is the measurement and $x$ is the state.
///
/// # State Vector Layout
///
/// The standard state vector ordering is:
/// ```text
/// x = [lat, lon, alt, v_n, v_e, v_d, roll, pitch, yaw, ...]
///      [0]  [1]  [2]  [3]  [4]   [5]  [6]   [7]    [8]
/// ```
///
/// For 15-state filters with IMU biases:
/// ```text
/// x = [lat, lon, alt, v_n, v_e, v_d, roll, pitch, yaw, b_ax, b_ay, b_az, b_gx, b_gy, b_gz]
/// ```
pub trait MeasurementModel: Any {
    /// Downcast helper method to allow for type-safe downcasting
    fn as_any(&self) -> &dyn Any;
    /// Downcast helper method for mutable references
    fn as_any_mut(&mut self) -> &mut dyn Any;
    /// Get the dimension of the measurement vector
    fn get_dimension(&self) -> usize;
    /// Get the measurement in a vector format given the current state estimate.
    ///
    /// This method returns the actual measurement value(s) as a vector. The state
    /// parameter allows state-dependent measurements (e.g., tilt-compensated
    /// magnetometer yaw requires roll and pitch from the state).
    ///
    /// # Arguments
    ///
    /// * `state` - Current state estimate vector
    ///
    /// # Returns
    ///
    /// A `DVector<f64>` containing the measurement value(s)
    /// The observed measurement `z` for the given state.
    ///
    /// # Errors
    /// Returns an error when no measurement can be formed. Analytic models in this crate are
    /// infallible, but a geophysical anomaly is `observed - model(position)`, so if the
    /// underlying model rejects the query there is no `z` to return — and substituting one
    /// would corrupt the innovation rather than report the gap.
    fn get_measurement(&self, state: &DVector<f64>) -> Result<DVector<f64>, StrapdownError>;
    /// Get the measurement noise characteristics in a matrix format
    fn get_noise(&self) -> DMatrix<f64>;
    /// Get the expected measurements from the state. Measurement model function
    /// that maps the state values to measurement space.
    ///
    /// # Arguments
    ///
    /// * `state` - State vector (may be a sigma point or particle state)
    ///
    /// # Returns
    ///
    /// Expected measurement given the state
    fn get_expected_measurement(&self, state: &DVector<f64>) -> DVector<f64>;

    /// Provide the measurement Jacobian (H matrix) for EKF updates.
    ///
    /// This method returns the linearized measurement model (Jacobian matrix H) for
    /// Extended Kalman Filter updates. All measurements must implement this method
    /// to support EKF-based navigation filters.
    ///
    /// The Jacobian H is the partial derivative of the measurement function with respect
    /// to the state: H = ∂h/∂x, where h(x) maps state to expected measurement.
    ///
    /// For standard measurements (GPS, barometric altitude), the Jacobian is typically
    /// sparse with identity elements. For geophysical measurements (gravity/magnetic
    /// anomaly), the Jacobian includes numerical gradients from the geophysical map.
    ///
    /// # Arguments
    ///
    /// * `state` - Current state estimate vector [lat, lon, alt, v_n, v_e, v_d, roll, pitch, yaw]
    ///
    /// # Returns
    ///
    /// Jacobian matrix H (measurement_dim × state_dim) for EKF updates
    ///
    /// # Example
    ///
    /// ```rust
    /// use strapdown::measurements::{MeasurementModel, GPSPositionMeasurement};
    /// use nalgebra::DVector;
    ///
    /// let gps_meas = GPSPositionMeasurement {
    ///     latitude: 45.0,
    ///     longitude: -122.0,
    ///     altitude: 100.0,
    ///     horizontal_noise_std: 5.0,
    ///     vertical_noise_std: 10.0,
    /// };
    ///
    /// let state = DVector::from_vec(vec![0.7854, -2.1293, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    /// let h_matrix = gps_meas.get_jacobian(&state).unwrap();
    /// assert_eq!(h_matrix.nrows(), 3);
    /// assert_eq!(h_matrix.ncols(), 9);
    /// ```
    ///
    /// # Errors
    /// Returns an error when the Jacobian cannot be evaluated at `state`. The models in this
    /// crate are analytic and fail only on a malformed state vector, but geophysical models
    /// read a loaded map and legitimately fail when the estimate leaves its bounds — see
    /// [`StrapdownError::is_recoverable`], which tells a caller that skipping the measurement
    /// is the correct response rather than aborting.
    fn get_jacobian(&self, state: &DVector<f64>) -> Result<DMatrix<f64>, StrapdownError>;

    /// Wrap angular components of an innovation vector into [-π, π).
    ///
    /// The default is the identity: most measurements live in R^n and need no
    /// wrapping. Angular measurements must override this. A `z`/`z_hat` pair
    /// straddling the branch cut otherwise produces phantom ±2π innovations;
    /// an error-state injection cannot survive those because the small-angle
    /// quaternion approximation is invalid at 350° (see #286).
    fn wrap_residual(&self, _residual: &mut DVector<f64>) {}
}

/// GPS position measurement model
#[derive(Clone, Debug, Default)]
pub struct GPSPositionMeasurement {
    pub latitude: f64,
    pub longitude: f64,
    pub altitude: f64,
    pub horizontal_noise_std: f64,
    pub vertical_noise_std: f64,
}
impl Display for GPSPositionMeasurement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "GPSPositionMeasurement(lat: {}, lon: {}, alt: {}, horiz_noise: {}, vert_noise: {})",
            self.latitude,
            self.longitude,
            self.altitude,
            self.horizontal_noise_std,
            self.vertical_noise_std
        )
    }
}
impl MeasurementModel for GPSPositionMeasurement {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn get_dimension(&self) -> usize {
        3
    }
    fn get_measurement(&self, _state: &DVector<f64>) -> Result<DVector<f64>, StrapdownError> {
        // GPS position measurement is state-independent
        Ok(DVector::from_vec(vec![
            self.latitude.to_radians(),
            self.longitude.to_radians(),
            self.altitude,
        ]))
    }
    fn get_noise(&self) -> DMatrix<f64> {
        // Convert horizontal noise from meters to radians for position covariance
        let horizontal_noise_rad = (self.horizontal_noise_std * METERS_TO_DEGREES).to_radians();
        DMatrix::from_diagonal(&DVector::from_vec(vec![
            horizontal_noise_rad.powi(2),
            horizontal_noise_rad.powi(2),
            self.vertical_noise_std.powi(2),
        ]))
    }
    fn get_expected_measurement(&self, state: &DVector<f64>) -> DVector<f64> {
        DVector::from_vec(vec![state[0], state[1], state[2]])
    }

    fn get_jacobian(&self, state: &DVector<f64>) -> Result<DMatrix<f64>, StrapdownError> {
        let nav_state = jacobian_state(state)?;
        Ok(crate::linearize::gps_position_jacobian(&nav_state))
    }
}
/// GPS Velocity measurement model
#[derive(Clone, Debug, Default)]
pub struct GPSVelocityMeasurement {
    pub northward_velocity: f64,
    pub eastward_velocity: f64,
    pub vertical_velocity: f64,
    pub horizontal_noise_std: f64,
    pub vertical_noise_std: f64,
}
impl Display for GPSVelocityMeasurement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "GPSVelocityMeasurement(north: {}, east: {}, down: {}, horiz_noise: {}, vert_noise: {})",
            self.northward_velocity,
            self.eastward_velocity,
            self.vertical_velocity,
            self.horizontal_noise_std,
            self.vertical_noise_std
        )
    }
}
impl MeasurementModel for GPSVelocityMeasurement {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn get_dimension(&self) -> usize {
        3
    }
    fn get_measurement(&self, _state: &DVector<f64>) -> Result<DVector<f64>, StrapdownError> {
        // GPS velocity measurement is state-independent
        Ok(DVector::from_vec(vec![
            self.northward_velocity,
            self.eastward_velocity,
            self.vertical_velocity,
        ]))
    }
    fn get_noise(&self) -> DMatrix<f64> {
        DMatrix::from_diagonal(&DVector::from_vec(vec![
            self.horizontal_noise_std.powi(2),
            self.horizontal_noise_std.powi(2),
            self.vertical_noise_std.powi(2),
        ]))
    }
    fn get_expected_measurement(&self, state: &DVector<f64>) -> DVector<f64> {
        DVector::from_vec(vec![state[3], state[4], state[5]])
    }

    fn get_jacobian(&self, state: &DVector<f64>) -> Result<DMatrix<f64>, StrapdownError> {
        let nav_state = jacobian_state(state)?;
        Ok(crate::linearize::gps_velocity_jacobian(&nav_state))
    }
}
/// GPS Position and Velocity measurement model
#[derive(Clone, Debug, Default)]
pub struct GPSPositionAndVelocityMeasurement {
    pub latitude: f64,
    pub longitude: f64,
    pub altitude: f64,
    pub northward_velocity: f64,
    pub eastward_velocity: f64,
    pub horizontal_noise_std: f64,
    pub vertical_noise_std: f64,
    pub velocity_noise_std: f64,
}
impl MeasurementModel for GPSPositionAndVelocityMeasurement {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn get_dimension(&self) -> usize {
        5
    }
    fn get_measurement(&self, _state: &DVector<f64>) -> Result<DVector<f64>, StrapdownError> {
        // GPS position and velocity measurement is state-independent
        Ok(DVector::from_vec(vec![
            self.latitude.to_radians(),
            self.longitude.to_radians(),
            self.altitude,
            self.northward_velocity,
            self.eastward_velocity,
        ]))
    }
    fn get_noise(&self) -> DMatrix<f64> {
        // Convert horizontal noise from meters to radians for position covariance
        let horizontal_noise_rad = (self.horizontal_noise_std * METERS_TO_DEGREES).to_radians();
        DMatrix::from_diagonal(&DVector::from_vec(vec![
            horizontal_noise_rad.powi(2),
            horizontal_noise_rad.powi(2),
            self.vertical_noise_std.powi(2),
            self.velocity_noise_std.powi(2),
            self.velocity_noise_std.powi(2),
        ]))
    }
    fn get_expected_measurement(&self, state: &DVector<f64>) -> DVector<f64> {
        // Measurement includes latitude, longitude, altitude, north and east velocities
        // (five elements). Do not include vertical velocity here.
        DVector::from_vec(vec![state[0], state[1], state[2], state[3], state[4]])
    }

    fn get_jacobian(&self, state: &DVector<f64>) -> Result<DMatrix<f64>, StrapdownError> {
        let nav_state = jacobian_state(state)?;
        Ok(crate::linearize::gps_position_velocity_jacobian(&nav_state))
    }
    //fn get_sigma_points(&self, state_sigma_points: &DMatrix<f64>) -> DMatrix<f64> {
    //    let mut measurement_sigma_points = DMatrix::<f64>::zeros(5, state_sigma_points.ncols());
    //    for (i, sigma_point) in state_sigma_points.column_iter().enumerate() {
    //        measurement_sigma_points[(0, i)] = sigma_point[0];
    //        measurement_sigma_points[(1, i)] = sigma_point[1];
    //        measurement_sigma_points[(2, i)] = sigma_point[2];
    //        measurement_sigma_points[(3, i)] = sigma_point[3];
    //        measurement_sigma_points[(4, i)] = sigma_point[4];
    //    }
    //    measurement_sigma_points
    //}
}

/// Relative altitude measurement (barometric)
#[derive(Clone, Debug, Default)]
pub struct RelativeAltitudeMeasurement {
    pub relative_altitude: f64,
    pub reference_altitude: f64,
}
impl Display for RelativeAltitudeMeasurement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "RelativeAltitudeMeasurement(rel_alt: {}, ref_alt: {})",
            self.relative_altitude, self.reference_altitude
        )
    }
}
impl MeasurementModel for RelativeAltitudeMeasurement {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn get_dimension(&self) -> usize {
        1
    }
    fn get_measurement(&self, _state: &DVector<f64>) -> Result<DVector<f64>, StrapdownError> {
        // Barometric altitude measurement is state-independent
        Ok(DVector::from_vec(vec![
            self.relative_altitude + self.reference_altitude,
        ]))
    }
    fn get_noise(&self) -> DMatrix<f64> {
        DMatrix::from_diagonal(&DVector::from_vec(vec![5.0]))
    }
    fn get_expected_measurement(&self, state: &DVector<f64>) -> DVector<f64> {
        DVector::from_vec(vec![state[2]])
    }

    fn get_jacobian(&self, state: &DVector<f64>) -> Result<DMatrix<f64>, StrapdownError> {
        let nav_state = jacobian_state(state)?;
        Ok(crate::linearize::relative_altitude_jacobian(&nav_state))
    }
    // fn get_sigma_points(&self, state_sigma_points: &DMatrix<f64>) -> DMatrix<f64> {
    //     let mut measurement_sigma_points = DMatrix::<f64>::zeros(self.get_dimension(), state_sigma_points.ncols());
    //     for (i, sigma_point) in state_sigma_points.column_iter().enumerate() {
    //         measurement_sigma_points[(0, i)] = sigma_point[2];
    //     }
    //     measurement_sigma_points
    // }
}

/// Magnetometer-based yaw measurement model.
///
/// This measurement model uses body-frame magnetometer data to derive a tilt-compensated
/// yaw (heading) measurement. The measurement applies tilt compensation using roll and
/// pitch from the state vector, then optionally corrects for magnetic declination using
/// the World Magnetic Model (WMM) to obtain true heading.
///
/// # Mathematical Background
///
/// ## Tilt Compensation
///
/// Raw magnetometer readings in the body frame must be projected onto the horizontal plane
/// to compute magnetic heading. Given body-frame magnetic field components $(m_x, m_y, m_z)$
/// and attitude angles $(\phi, \theta)$ (roll, pitch), the horizontal components are:
///
/// $$
/// \begin{aligned}
/// m_{x,h} &= m_x \cos\theta + m_y \sin\phi \sin\theta + m_z \cos\phi \sin\theta \\\\
/// m_{y,h} &= m_y \cos\phi - m_z \sin\phi
/// \end{aligned}
/// $$
///
/// ## Magnetic Heading
///
/// The magnetic heading (yaw relative to magnetic north) is:
///
/// $$
/// \psi_m = \arctan2(m_{y,h}, m_{x,h})
/// $$
///
/// ## True Heading
///
/// If declination correction is enabled, the true heading is:
///
/// $$
/// \psi = \psi_m + \delta
/// $$
///
/// where $\delta$ is the magnetic declination (positive east) obtained from WMM.
///
/// # State Vector Requirements
///
/// This measurement model requires the following state indices:
/// - `state[0]`: latitude (radians) - for WMM lookup
/// - `state[1]`: longitude (radians) - for WMM lookup  
/// - `state[2]`: altitude (meters) - for WMM lookup
/// - `state[6]`: roll (radians) - for tilt compensation
/// - `state[7]`: pitch (radians) - for tilt compensation
/// - `state[8]`: yaw (radians) - expected measurement
///
/// # Example
///
/// ```rust
/// use strapdown::measurements::MagnetometerYawMeasurement;
/// use strapdown::measurements::MeasurementModel;
/// use nalgebra::DVector;
///
/// // Create measurement from magnetometer data
/// let mag_meas = MagnetometerYawMeasurement {
///     mag_x: 20.0,  // µT
///     mag_y: 5.0,   // µT
///     mag_z: -45.0, // µT
///     noise_std: 0.05, // radians (~3 degrees)
///     apply_declination: true,
///     year: 2025,
///     day_of_year: 1,
/// };
///
/// // State vector with position and attitude
/// let state = DVector::from_vec(vec![
///     0.7854,   // lat (45 deg in rad)
///     -2.1293,  // lon (-122 deg in rad)
///     100.0,    // alt (m)
///     0.0, 0.0, 0.0,  // velocities
///     0.0,      // roll
///     0.0,      // pitch
///     0.5,      // yaw
/// ]);
///
/// // Get tilt-compensated yaw measurement
/// let z = mag_meas.get_measurement(&state).unwrap();
/// assert_eq!(z.len(), 1);
/// ```
#[derive(Clone, Debug)]
pub struct MagnetometerYawMeasurement {
    /// Body-frame magnetic field x-component (forward) in micro teslas
    pub mag_x: f64,
    /// Body-frame magnetic field y-component (right) in micro teslas
    pub mag_y: f64,
    /// Body-frame magnetic field z-component (down) in micro teslas
    pub mag_z: f64,
    /// Measurement noise standard deviation in radians
    pub noise_std: f64,
    /// Whether to apply WMM declination correction for true heading
    pub apply_declination: bool,
    /// Year for WMM calculation (e.g., 2025)
    pub year: i32,
    /// Day of year for WMM calculation (1-366)
    pub day_of_year: u16,
}

impl Default for MagnetometerYawMeasurement {
    fn default() -> Self {
        Self {
            mag_x: 0.0,
            mag_y: 0.0,
            mag_z: 0.0,
            noise_std: 0.05, // ~3 degrees
            apply_declination: false,
            year: 2025,
            day_of_year: 1,
        }
    }
}

impl Display for MagnetometerYawMeasurement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MagnetometerYawMeasurement(mag: [{:.2}, {:.2}, {:.2}] µT, noise: {:.4} rad, decl: {})",
            self.mag_x, self.mag_y, self.mag_z, self.noise_std, self.apply_declination
        )
    }
}

impl MagnetometerYawMeasurement {
    /// Get magnetic declination at the given position using WMM.
    ///
    /// # Arguments
    ///
    /// * `lat_deg` - Latitude in degrees
    /// * `lon_deg` - Longitude in degrees
    /// * `alt_m` - Altitude in meters
    ///
    /// # Returns
    ///
    /// Magnetic declination in radians (positive east)
    pub fn get_declination(&self, lat_deg: f64, lon_deg: f64, alt_m: f64) -> f64 {
        let date = Date::from_ordinal_date(self.year, self.day_of_year)
            .unwrap_or_else(|_| fallback_wmm_date());

        let field = GeomagneticField::new(
            Length::new::<meter>(alt_m as f32),
            Angle::new::<degree>(lat_deg as f32),
            Angle::new::<degree>(lon_deg as f32),
            date,
        );

        match field {
            Ok(f) => f64::from(f.declination().get::<degree>()) * std::f64::consts::PI / 180.0,
            Err(_) => 0.0, // Return 0 declination if WMM fails (e.g., position out of range)
        }
    }
}

impl MeasurementModel for MagnetometerYawMeasurement {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn get_dimension(&self) -> usize {
        1 // Single yaw measurement
    }

    fn get_measurement(&self, state: &DVector<f64>) -> Result<DVector<f64>, StrapdownError> {
        // Extract roll and pitch from state for tilt compensation
        let roll = if state.len() > 6 { state[6] } else { 0.0 };
        let pitch = if state.len() > 7 { state[7] } else { 0.0 };

        // Compute tilt-compensated magnetic vector.
        //
        // Only roll and pitch enter here: levelling the sensor must not use
        // yaw, otherwise the "measurement" becomes a function of the estimated
        // yaw and the innovation double-counts yaw error (the estimated yaw
        // rotates the vector one way in `z` and appears again in `z_hat`,
        // destabilising the attitude/bias loop -- see #286). Yaw is observed
        // through `get_expected_measurement`, not through the sensor rotation.
        let attitude = Rotation3::from_euler_angles(roll, pitch, 0.0);
        let mag_vector = attitude * Vector3::new(self.mag_x, self.mag_y, self.mag_z);
        let mut heading = mag_vector.y.atan2(mag_vector.x);

        // Apply declination correction if enabled
        if self.apply_declination && state.len() >= 3 {
            let lat_deg = state[0].to_degrees();
            let lon_deg = state[1].to_degrees();
            let alt_m = state[2];
            let declination = self.get_declination(lat_deg, lon_deg, alt_m);
            heading += declination;

            // Re-wrap to [0, 2π) after adding declination
            heading = heading.rem_euclid(2.0 * std::f64::consts::PI);
        }

        Ok(DVector::from_vec(vec![heading]))
    }

    fn get_noise(&self) -> DMatrix<f64> {
        DMatrix::from_diagonal(&DVector::from_vec(vec![self.noise_std.powi(2)]))
    }

    fn get_expected_measurement(&self, state: &DVector<f64>) -> DVector<f64> {
        // Expected measurement is the yaw from the state (index 8)
        let yaw = if state.len() > 8 { state[8] } else { 0.0 };
        DVector::from_vec(vec![yaw])
    }

    fn get_jacobian(&self, state: &DVector<f64>) -> Result<DMatrix<f64>, StrapdownError> {
        let nav_state = jacobian_state(state)?;
        Ok(crate::linearize::magnetometer_yaw_jacobian(
            &nav_state, self.mag_x, self.mag_y, self.mag_z,
        ))
    }

    fn wrap_residual(&self, residual: &mut DVector<f64>) {
        // Single yaw component: keep the innovation on the circle so a
        // z/z_hat pair straddling 0/2π does not inject a phantom ±2π kick.
        if !residual.is_empty() {
            residual[0] = (residual[0] + std::f64::consts::PI)
                .rem_euclid(2.0 * std::f64::consts::PI)
                - std::f64::consts::PI;
        }
    }
}

/// Default pseudo-measurement noise for [`ZuptMeasurement`], m/s per axis.
///
/// A ZUPT is not a sensor reading, so its "noise" is really a statement of how
/// literally the filter should take the constraint. 0.01 m/s says the platform is
/// stationary to within a centimetre per second, which is tight enough to arrest
/// velocity drift within a few updates and loose enough to absorb the residual
/// vibration of a running engine. Tightening it much further makes the update
/// nearly deterministic, which is a good way to collapse the velocity covariance
/// and then have the filter refuse the GNSS fix that follows the stop.
pub const DEFAULT_ZUPT_NOISE_MPS: f64 = 0.01;

/// Default pseudo-measurement noise for [`ZaruMeasurement`], rad/s per axis.
///
/// Sized for the angle-random-walk floor of a consumer MEMS gyro over a one-second
/// window. As with ZUPT this is a confidence statement rather than a sensor spec:
/// it should be at or above the gyro's own noise over the averaging interval, or
/// the filter will read that noise as bias and chase it.
pub const DEFAULT_ZARU_NOISE_RPS: f64 = 1.0e-3;

/// Zero-velocity update (ZUPT) pseudo-measurement.
///
/// Asserts that the local-level-frame velocity is zero. Valid only while the
/// platform is genuinely stationary -- see [`StationaryDetector`] for the decision,
/// which this type deliberately does not make for itself: the detector needs a
/// window of IMU history, and a measurement model is handed one state at a time.
///
/// # Why it works
///
/// Velocity error is the integral of accelerometer error, and position error the
/// integral of that. Pinning velocity to zero during a stop does not merely stop
/// the position drift for the duration; because the filter's velocity error is
/// correlated with its accelerometer bias error, the correction propagates back
/// into the bias estimate and the solution is better *after* the stop than it was
/// before. That is the whole reason to bother: an unaided stop is the cheapest
/// observability a strapdown system ever gets.
///
/// # Example
///
/// ```rust
/// use nalgebra::DVector;
/// use strapdown::measurements::{MeasurementModel, ZuptMeasurement};
///
/// let zupt = ZuptMeasurement::default();
/// // The state thinks it is moving north at 0.3 m/s; the pseudo-measurement says 0.
/// let state = DVector::from_vec(vec![0.79, -2.13, 100.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0]);
/// let innovation =
///     zupt.get_measurement(&state).unwrap() - zupt.get_expected_measurement(&state);
/// assert!((innovation[0] + 0.3).abs() < 1e-12);
/// ```
///
/// # References
///
/// - Groves 2nd ed., Section 15.2.1
///
/// [`StationaryDetector`]: crate::stationary::StationaryDetector
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ZuptMeasurement {
    /// Per-axis pseudo-measurement standard deviation, m/s.
    pub velocity_noise_std: f64,
}

impl Default for ZuptMeasurement {
    fn default() -> Self {
        Self {
            velocity_noise_std: DEFAULT_ZUPT_NOISE_MPS,
        }
    }
}

impl ZuptMeasurement {
    /// Build a ZUPT with an explicit per-axis standard deviation.
    ///
    /// # Errors
    /// [`StrapdownError::OutOfRange`] if `velocity_noise_std` is not finite and
    /// strictly positive. A zero standard deviation gives a singular `R`, and the
    /// resulting Kalman gain is not something to discover at runtime.
    pub fn new(velocity_noise_std: f64) -> Result<Self, StrapdownError> {
        if !velocity_noise_std.is_finite() || velocity_noise_std <= 0.0 {
            return Err(StrapdownError::OutOfRange {
                what: "ZUPT velocity noise std (m/s)",
                value: velocity_noise_std,
                min: f64::MIN_POSITIVE,
                max: f64::INFINITY,
            });
        }
        Ok(Self { velocity_noise_std })
    }
}

impl Display for ZuptMeasurement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ZuptMeasurement(noise: {} m/s)", self.velocity_noise_std)
    }
}

impl MeasurementModel for ZuptMeasurement {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn get_dimension(&self) -> usize {
        3
    }
    fn get_measurement(&self, _state: &DVector<f64>) -> Result<DVector<f64>, StrapdownError> {
        // The constraint itself: the platform is not moving.
        Ok(DVector::zeros(3))
    }
    fn get_noise(&self) -> DMatrix<f64> {
        DMatrix::from_diagonal(&DVector::from_vec(vec![
            self.velocity_noise_std.powi(2),
            self.velocity_noise_std.powi(2),
            self.velocity_noise_std.powi(2),
        ]))
    }
    fn get_expected_measurement(&self, state: &DVector<f64>) -> DVector<f64> {
        if state.len() < 9 {
            // Short states cannot occur through any filter in this crate; returning
            // the constraint value makes the innovation zero rather than panicking
            // in a method with no way to report. `get_jacobian` rejects the same
            // state, which is where the caller sees the problem.
            return DVector::zeros(3);
        }
        DVector::from_vec(vec![state[3], state[4], state[5]])
    }
    fn get_jacobian(&self, state: &DVector<f64>) -> Result<DMatrix<f64>, StrapdownError> {
        let nav_state = jacobian_state(state)?;
        Ok(crate::linearize::zupt_jacobian(&nav_state))
    }
}

/// Zero-angular-rate update (ZARU) pseudo-measurement.
///
/// Asserts that a stationary platform's gyroscopes read nothing but their own bias
/// plus the Earth rate, making the measurement a direct observation of gyro bias.
///
/// # State requirement
///
/// ZARU observes states 12..15 and therefore **requires a filter that carries gyro
/// bias states**: the 15-state [`ErrorStateKalmanFilter`] (the crate default) or an
/// [`ExtendedKalmanFilter`] built with `use_biases = true`. Applied to a 9-state
/// filter it returns [`StrapdownError::DimensionMismatch`] rather than quietly
/// correcting nothing, because "the update ran and changed no state" is
/// indistinguishable from "the update worked" in a log.
///
/// # Measurement model
///
/// ```text
/// z    = omega_measured                          (raw body-frame gyro, rad/s)
/// h(x) = b_g + C_n^b(attitude) * omega_ie^n(lat)
/// ```
///
/// The Earth-rate term is carried in `h` rather than subtracted from `z` so that
/// `z` stays a raw sensor reading. Its attitude dependence is real but of order the
/// Earth rate itself (7.3e-5 rad/s), well under the noise floor of the MEMS
/// hardware this crate targets, so the Jacobian's attitude block is left at zero;
/// [`set_earth_rate_compensation`] turns the term off entirely for a unit whose
/// noise swamps it.
///
/// # Example
///
/// ```rust
/// use nalgebra::DVector;
/// use strapdown::measurements::{MeasurementModel, ZaruMeasurement};
///
/// // A stationary gyro reading 0.002 rad/s about z is reading its own bias.
/// let zaru = ZaruMeasurement::from_gyro([0.0, 0.0, 0.002]);
/// let mut state = DVector::zeros(15);
/// let innovation =
///     zaru.get_measurement(&state).unwrap() - zaru.get_expected_measurement(&state);
/// // The filter currently estimates zero bias, so the whole reading is innovation.
/// assert!((innovation[2] - 0.002).abs() < 1e-9);
///
/// // Only the gyro-bias block is observable.
/// let h = zaru.get_jacobian(&state).unwrap();
/// assert_eq!((h.nrows(), h.ncols()), (3, 15));
/// ```
///
/// # References
///
/// - Groves 2nd ed., Section 15.2.2
///
/// [`ErrorStateKalmanFilter`]: crate::kalman::ErrorStateKalmanFilter
/// [`ExtendedKalmanFilter`]: crate::kalman::ExtendedKalmanFilter
/// [`set_earth_rate_compensation`]: ZaruMeasurement::set_earth_rate_compensation
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ZaruMeasurement {
    /// Measured body-frame angular rate, rad/s.
    pub angular_rate: Vector3<f64>,
    /// Per-axis pseudo-measurement standard deviation, rad/s.
    pub angular_rate_noise_std: f64,
    /// Whether `h(x)` includes the sensed Earth rate.
    pub compensate_earth_rate: bool,
}

impl Default for ZaruMeasurement {
    fn default() -> Self {
        Self {
            angular_rate: Vector3::zeros(),
            angular_rate_noise_std: DEFAULT_ZARU_NOISE_RPS,
            compensate_earth_rate: true,
        }
    }
}

impl ZaruMeasurement {
    /// Build a ZARU from a measured body-frame angular rate, with default noise.
    #[must_use]
    pub fn from_gyro(angular_rate: [f64; 3]) -> Self {
        Self {
            angular_rate: Vector3::from_column_slice(&angular_rate),
            ..Self::default()
        }
    }

    /// Build a ZARU with an explicit per-axis standard deviation.
    ///
    /// # Errors
    /// [`StrapdownError::OutOfRange`] if `angular_rate_noise_std` is not finite and
    /// strictly positive, or [`StrapdownError::NonFinite`] if any rate component is
    /// not finite.
    pub fn new(
        angular_rate: Vector3<f64>,
        angular_rate_noise_std: f64,
    ) -> Result<Self, StrapdownError> {
        if !angular_rate.iter().all(|v| v.is_finite()) {
            return Err(StrapdownError::NonFinite {
                what: "ZARU angular rate",
            });
        }
        if !angular_rate_noise_std.is_finite() || angular_rate_noise_std <= 0.0 {
            return Err(StrapdownError::OutOfRange {
                what: "ZARU angular rate noise std (rad/s)",
                value: angular_rate_noise_std,
                min: f64::MIN_POSITIVE,
                max: f64::INFINITY,
            });
        }
        Ok(Self {
            angular_rate,
            angular_rate_noise_std,
            compensate_earth_rate: true,
        })
    }

    /// Include or omit the sensed Earth rate in `h(x)`.
    ///
    /// Leave it on for tactical-grade hardware, where 7.3e-5 rad/s is a resolvable
    /// quantity and omitting it biases the gyro-bias estimate by that much. Turn it
    /// off when reproducing a reference implementation that ignores it.
    #[must_use]
    pub const fn set_earth_rate_compensation(mut self, compensate: bool) -> Self {
        self.compensate_earth_rate = compensate;
        self
    }

    /// Earth rate resolved into the body frame for the given state, rad/s.
    ///
    /// Returns zero when compensation is disabled or the state is too short to
    /// carry an attitude.
    fn sensed_earth_rate(&self, state: &DVector<f64>) -> Vector3<f64> {
        if !self.compensate_earth_rate || state.len() < 9 {
            return Vector3::zeros();
        }
        // `earth_rate_lla` takes degrees and returns the NED local-level vector;
        // the state carries latitude in radians.
        let earth_rate_ned = crate::earth::earth_rate_lla(&state[0].to_degrees());
        // `from_euler_angles` builds C_b^n (body to nav), so its transpose resolves
        // a nav-frame vector into the body frame.
        let body_to_nav = Rotation3::from_euler_angles(state[6], state[7], state[8]);
        body_to_nav.inverse() * earth_rate_ned
    }

    /// Reject a state that cannot carry gyro biases.
    fn require_bias_states(state: &DVector<f64>) -> Result<(), StrapdownError> {
        if state.len() < 15 {
            return Err(StrapdownError::DimensionMismatch {
                what: "ZARU state vector (requires gyro bias states 12..15)",
                expected: 15,
                got: state.len(),
            });
        }
        Ok(())
    }
}

impl Display for ZaruMeasurement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ZaruMeasurement(rate: [{:.3e}, {:.3e}, {:.3e}] rad/s, noise: {} rad/s)",
            self.angular_rate[0],
            self.angular_rate[1],
            self.angular_rate[2],
            self.angular_rate_noise_std
        )
    }
}

impl MeasurementModel for ZaruMeasurement {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn get_dimension(&self) -> usize {
        3
    }
    /// The raw body-frame gyro reading.
    ///
    /// # Errors
    /// [`StrapdownError::DimensionMismatch`] if `state` is shorter than 15 elements.
    /// The reading itself does not depend on the state; the check lives here because
    /// this is the one `Result`-returning method every filter calls, and a 9-state
    /// filter applying ZARU has to be told rather than silently corrected by nothing.
    fn get_measurement(&self, state: &DVector<f64>) -> Result<DVector<f64>, StrapdownError> {
        Self::require_bias_states(state)?;
        Ok(DVector::from_vec(vec![
            self.angular_rate[0],
            self.angular_rate[1],
            self.angular_rate[2],
        ]))
    }
    fn get_noise(&self) -> DMatrix<f64> {
        DMatrix::from_diagonal(&DVector::from_vec(vec![
            self.angular_rate_noise_std.powi(2),
            self.angular_rate_noise_std.powi(2),
            self.angular_rate_noise_std.powi(2),
        ]))
    }
    fn get_expected_measurement(&self, state: &DVector<f64>) -> DVector<f64> {
        let earth_rate_body = self.sensed_earth_rate(state);
        if state.len() < 15 {
            // Rejected by `get_measurement`/`get_jacobian`; predict the Earth-rate
            // term alone rather than index past the end of a short state.
            return DVector::from_vec(vec![
                earth_rate_body[0],
                earth_rate_body[1],
                earth_rate_body[2],
            ]);
        }
        DVector::from_vec(vec![
            state[12] + earth_rate_body[0],
            state[13] + earth_rate_body[1],
            state[14] + earth_rate_body[2],
        ])
    }
    /// The 3x15 gyro-bias Jacobian.
    ///
    /// # Errors
    /// [`StrapdownError::DimensionMismatch`] if `state` is shorter than 15 elements.
    fn get_jacobian(&self, state: &DVector<f64>) -> Result<DMatrix<f64>, StrapdownError> {
        Self::require_bias_states(state)?;
        let nav_state = jacobian_state(state)?;
        Ok(crate::linearize::zaru_jacobian(&nav_state))
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use assert_approx_eq::assert_approx_eq;

    const EPS: f64 = 1e-12;

    /// #286: the tilt-compensated yaw must not depend on estimated yaw.
    ///
    /// Levelling the sensor uses roll/pitch only; rotating by yaw would make
    /// the measurement a function of the estimate and double-count yaw error
    /// in the innovation.
    #[test]
    fn mag_yaw_measurement_ignores_estimated_yaw() {
        let m = MagnetometerYawMeasurement {
            mag_x: 20.0,
            mag_y: 5.0,
            mag_z: -45.0,
            noise_std: 0.2,
            apply_declination: false,
            year: 2025,
            day_of_year: 1,
        };
        let z_at = |yaw: f64| {
            m.get_measurement(&DVector::from_vec(vec![
                0.7, -1.3, 100.0, 0.0, 0.0, 0.0, 0.16, -0.4, yaw,
            ]))
            .unwrap()[0]
        };
        let z0 = z_at(0.2);
        for yaw in [0.0, 1.0, 2.5, 5.0, -1.2] {
            assert_approx_eq!(z_at(yaw), z0, 1e-12);
        }
    }

    /// #286: angular residuals wrap onto the circle.
    #[test]
    fn mag_wrap_residual_keeps_innovation_on_circle() {
        let m = MagnetometerYawMeasurement {
            mag_x: 20.0,
            mag_y: 5.0,
            mag_z: -45.0,
            noise_std: 0.2,
            apply_declination: false,
            year: 2025,
            day_of_year: 1,
        };
        // z/z_hat straddling the branch cut must not produce a ±2π kick.
        let mut r = DVector::from_vec(vec![6.1]);
        m.wrap_residual(&mut r);
        assert_approx_eq!(r[0], 6.1 - 2.0 * std::f64::consts::PI, 1e-12);
        let mut r = DVector::from_vec(vec![-5.0]);
        m.wrap_residual(&mut r);
        assert_approx_eq!(r[0], -5.0 + 2.0 * std::f64::consts::PI, 1e-12);
        // Small residuals pass through untouched.
        let mut r = DVector::from_vec(vec![0.5]);
        m.wrap_residual(&mut r);
        assert_approx_eq!(r[0], 0.5, EPS);
    }

    #[test]
    fn gps_position_vector_noise_and_sigma_points() {
        let meas = GPSPositionMeasurement {
            latitude: 37.0,
            longitude: -122.0,
            altitude: 12.34,
            horizontal_noise_std: 3.0,
            vertical_noise_std: 2.0,
        };

        // Dummy state for get_measurement (GPS position is state-independent)
        let dummy_state = DVector::from_vec(vec![0.0; 9]);

        // Vector in radians for lat/lon
        let vec = meas.get_measurement(&dummy_state).unwrap();
        assert_eq!(vec.len(), 3);
        assert!((vec[0] - 37.0_f64.to_radians()).abs() < EPS);
        assert!((vec[1] - (-122.0_f64).to_radians()).abs() < EPS);
        assert!((vec[2] - 12.34).abs() < EPS);

        // Noise diagonal entries - should be in radians squared for lat/lon
        let noise = meas.get_noise();
        let expected_h = (3.0 * METERS_TO_DEGREES).to_radians().powi(2);
        let expected_v = 2.0_f64.powi(2);
        assert_eq!(noise.nrows(), 3);
        assert!((noise[(0, 0)] - expected_h).abs() < EPS);
        assert!((noise[(1, 1)] - expected_h).abs() < EPS);
        assert!((noise[(2, 2)] - expected_v).abs() < EPS);

        let state_sigma: DVector<f64> = DVector::from_vec(vec![
            0.1, // lat
            1.1, // lon
            2.1, // alt
            3.0, // v_n
            4.0, // v_e
            5.0, // v_d
        ]);
        let z = meas.get_expected_measurement(&state_sigma);
        assert_eq!(z.len(), 3);
        assert_approx_eq!(z[0], 0.1, EPS);
        assert_approx_eq!(z[1], 1.1, EPS);
        assert_approx_eq!(z[2], 2.1, EPS);
    }

    #[test]
    fn gps_velocity_vector_noise_and_sigma_points() {
        let meas = GPSVelocityMeasurement {
            northward_velocity: 1.5,
            eastward_velocity: -0.5,
            vertical_velocity: 0.25,
            horizontal_noise_std: 0.2,
            vertical_noise_std: 0.1,
        };

        // Dummy state for get_measurement (GPS velocity is state-independent)
        let dummy_state = DVector::from_vec(vec![0.0; 9]);

        let vec = meas.get_measurement(&dummy_state).unwrap();
        assert_eq!(vec.len(), 3);
        assert!((vec[0] - 1.5).abs() < EPS);
        assert!((vec[1] - (-0.5)).abs() < EPS);
        assert!((vec[2] - 0.25).abs() < EPS);

        let noise = meas.get_noise();
        assert!((noise[(0, 0)] - 0.2_f64.powi(2)).abs() < EPS);
        assert!((noise[(2, 2)] - 0.1_f64.powi(2)).abs() < EPS);

        let state_sigma: DVector<f64> = DVector::from_vec(vec![
            0.1, // lat
            1.1, // lon
            2.1, // alt
            3.0, // v_n
            4.0, // v_e
            5.0, // v_d
        ]);
        let z = meas.get_expected_measurement(&state_sigma);
        assert_eq!(z.len(), 3);
        assert_approx_eq!(z[0], 3.0, EPS);
        assert_approx_eq!(z[1], 4.0, EPS);
        assert_approx_eq!(z[2], 5.0, EPS);
    }

    #[test]
    fn position_and_velocity_measurement_behaviour() {
        let meas = GPSPositionAndVelocityMeasurement {
            latitude: 10.0,
            longitude: 20.0,
            altitude: 100.0,
            northward_velocity: 2.0,
            eastward_velocity: -1.0,
            horizontal_noise_std: 1.0,
            vertical_noise_std: 4.0,
            velocity_noise_std: 0.5,
        };

        // Dummy state for get_measurement (GPS measurement is state-independent)
        let dummy_state = DVector::from_vec(vec![0.0; 9]);

        let vec = meas.get_measurement(&dummy_state).unwrap();
        assert_eq!(vec.len(), 5);
        assert!((vec[0] - 10.0_f64.to_radians()).abs() < EPS);
        assert!((vec[3] - 2.0).abs() < EPS);

        let noise = meas.get_noise();
        assert_eq!(noise.nrows(), 5);

        let state_sigma: DVector<f64> = DVector::from_vec(vec![
            0.1, // lat
            1.1, // lon
            2.1, // alt
            3.0, // v_n
            4.0, // v_e
            5.0, // v_d
        ]);
        let z = meas.get_expected_measurement(&state_sigma);
        assert_eq!(z.len(), 5);
        assert_approx_eq!(z[0], 0.1, EPS);
        assert_approx_eq!(z[1], 1.1, EPS);
        assert_approx_eq!(z[2], 2.1, EPS);
        assert_approx_eq!(z[3], 3.0, EPS);
        assert_approx_eq!(z[4], 4.0, EPS);
    }

    #[test]
    fn relative_altitude_measurement_and_display_and_sigma() {
        let meas = RelativeAltitudeMeasurement {
            relative_altitude: -5.0,
            reference_altitude: 100.0,
        };

        // Dummy state for get_measurement (barometric altitude is state-independent)
        let dummy_state = DVector::from_vec(vec![0.0; 9]);

        let vec = meas.get_measurement(&dummy_state).unwrap();
        assert_eq!(vec.len(), 1);
        assert!((vec[0] - 95.0).abs() < EPS);

        let noise = meas.get_noise();
        assert_eq!(noise.nrows(), 1);
        assert!((noise[(0, 0)] - 5.0).abs() < EPS);

        // sigma points should extract altitude (index 2)
        let state_sigma: DVector<f64> = DVector::from_vec(vec![
            0.1,  // lat
            1.1,  // lon
            50.0, // alt
            3.0,  // v_n
            4.0,  // v_e
            5.0,  // v_d
        ]);
        let z = meas.get_expected_measurement(&state_sigma);
        assert_eq!(z.len(), 1);
        assert_approx_eq!(z[0], 50.0, EPS);

        // Display string
        let s = format!("{meas}");
        assert!(s.contains("rel_alt") && s.contains("ref_alt"));
    }

    #[test]
    fn downcast_trait_object_and_display() {
        let pos = GPSPositionMeasurement::default();
        // pos.latitude = 1.0;
        // pos.longitude = 2.0;
        // pos.altitude = 3.0;
        let boxed: Box<dyn MeasurementModel> = Box::new(pos);
        // downcast via as_any
        let any = boxed.as_any();
        let down = any
            .downcast_ref::<GPSPositionMeasurement>()
            .expect("downcast failed");
        assert!((down.latitude).abs() < EPS);

        // Display formatting
        let s = format!("{down}");
        assert!(s.contains("GPSPositionMeasurement"));
    }

    #[test]
    fn negative_and_zero_std_are_handled() {
        let meas = GPSVelocityMeasurement {
            northward_velocity: 0.0,
            eastward_velocity: 0.0,
            vertical_velocity: 0.0,
            horizontal_noise_std: -2.0,
            vertical_noise_std: 0.0,
        };
        let noise = meas.get_noise();
        // negative std should be squared, resulting positive variance
        assert!((noise[(0, 0)] - 4.0).abs() < EPS);
        // zero std -> zero variance
        assert!((noise[(2, 2)] - 0.0).abs() < EPS);
    }

    #[test]
    fn magnetometer_yaw_measurement_level_attitude() {
        // Test magnetometer yaw with level attitude (no tilt)
        // Magnetic field pointing north (positive x) should give ~0 heading
        let meas = MagnetometerYawMeasurement {
            mag_x: 20.0, // pointing north
            mag_y: 0.0,
            mag_z: -45.0, // typical downward component
            noise_std: 0.05,
            apply_declination: false,
            year: 2025,
            day_of_year: 1,
        };

        // State with zero roll/pitch (level)
        let state = DVector::from_vec(vec![
            std::f64::consts::FRAC_PI_4, // lat (45 deg)
            -2.1293,                     // lon (-122 deg)
            100.0,                       // alt
            0.0,
            0.0,
            0.0, // velocities
            0.0, // roll = 0
            0.0, // pitch = 0
            0.0, // yaw
        ]);

        let z = meas.get_measurement(&state).unwrap();
        assert_eq!(z.len(), 1);

        // With mag pointing north and level attitude, heading should be ~0
        assert!(z[0].abs() < 0.01, "Expected heading near 0, got {}", z[0]);
    }

    #[test]
    fn magnetometer_yaw_measurement_east_heading() {
        // Test magnetometer pointing east (positive y)
        let meas = MagnetometerYawMeasurement {
            mag_x: 0.0,
            mag_y: 20.0, // pointing east
            mag_z: -45.0,
            noise_std: 0.05,
            apply_declination: false,
            year: 2025,
            day_of_year: 1,
        };

        let state = DVector::from_vec(vec![
            std::f64::consts::FRAC_PI_4,
            -2.1293,
            100.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0, // level attitude
        ]);

        let z = meas.get_measurement(&state).unwrap();

        // With mag pointing east and level attitude, heading should be ~π/2 (90 deg)
        let expected = std::f64::consts::FRAC_PI_2;
        assert!(
            (z[0] - expected).abs() < 0.01,
            "Expected heading near π/2, got {}",
            z[0]
        );
    }

    #[test]
    fn magnetometer_yaw_tilt_compensation() {
        // Test that tilt compensation changes the result
        let meas = MagnetometerYawMeasurement {
            mag_x: 20.0,
            mag_y: 5.0,
            mag_z: -45.0,
            noise_std: 0.05,
            apply_declination: false,
            year: 2025,
            day_of_year: 1,
        };

        // Level state
        let level_state = DVector::from_vec(vec![
            std::f64::consts::FRAC_PI_4,
            -2.1293,
            100.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]);

        // Tilted state (10 degrees roll)
        let tilted_state = DVector::from_vec(vec![
            std::f64::consts::FRAC_PI_4,
            -2.1293,
            100.0,
            0.0,
            0.0,
            0.0,
            0.1745,
            0.0,
            0.0, // ~10 deg roll
        ]);

        let z_level = meas.get_measurement(&level_state).unwrap();
        let z_tilted = meas.get_measurement(&tilted_state).unwrap();

        // Heading should be different when tilted (tilt compensation effect)
        assert!(
            (z_level[0] - z_tilted[0]).abs() > 0.001,
            "Tilt compensation should change heading"
        );
    }

    #[test]
    fn magnetometer_yaw_expected_measurement() {
        // Test that expected measurement extracts yaw from state
        let meas = MagnetometerYawMeasurement::default();

        let state = DVector::from_vec(vec![
            std::f64::consts::FRAC_PI_4,
            -2.1293,
            100.0,
            0.0,
            0.0,
            0.0,
            0.1,
            0.2,
            std::f64::consts::FRAC_PI_2, // yaw = π/2
        ]);

        let expected = meas.get_expected_measurement(&state);
        assert_eq!(expected.len(), 1);
        assert!(
            (expected[0] - std::f64::consts::FRAC_PI_2).abs() < EPS,
            "Expected measurement should be state yaw"
        );
    }

    #[test]
    fn magnetometer_yaw_noise_matrix() {
        let meas = MagnetometerYawMeasurement {
            noise_std: 0.1,
            ..Default::default()
        };

        let noise = meas.get_noise();
        assert_eq!(noise.nrows(), 1);
        assert_eq!(noise.ncols(), 1);
        assert!(
            (noise[(0, 0)] - 0.01).abs() < EPS,
            "Noise variance should be 0.1^2 = 0.01"
        );
    }

    #[test]
    fn magnetometer_yaw_display() {
        let meas = MagnetometerYawMeasurement {
            mag_x: 20.0,
            mag_y: 5.0,
            mag_z: -45.0,
            noise_std: 0.05,
            apply_declination: true,
            year: 2025,
            day_of_year: 1,
        };

        let s = format!("{meas}");
        assert!(s.contains("MagnetometerYawMeasurement"));
        assert!(s.contains("20.00"));
        assert!(s.contains("true")); // apply_declination
    }

    #[test]
    fn magnetometer_yaw_downcast() {
        let meas = MagnetometerYawMeasurement::default();
        let boxed: Box<dyn MeasurementModel> = Box::new(meas);

        let any = boxed.as_any();
        let down = any.downcast_ref::<MagnetometerYawMeasurement>();
        assert!(
            down.is_some(),
            "Should be able to downcast MagnetometerYawMeasurement"
        );
    }

    // ---------------------------------------------------------------- ZUPT / ZARU

    /// A 15-state vector with the navigation block filled in and zero biases.
    fn state_15(velocity: [f64; 3], attitude: [f64; 3]) -> DVector<f64> {
        let mut state = DVector::zeros(15);
        state[0] = std::f64::consts::FRAC_PI_4; // 45 deg N, radians
        state[1] = -2.1293;
        state[2] = 100.0;
        state[3] = velocity[0];
        state[4] = velocity[1];
        state[5] = velocity[2];
        state[6] = attitude[0];
        state[7] = attitude[1];
        state[8] = attitude[2];
        state
    }

    #[test]
    fn zupt_innovation_is_the_negated_velocity() {
        let zupt = ZuptMeasurement::default();
        let state = state_15([0.3, -0.7, 0.05], [0.0, 0.0, 0.0]);
        let innovation =
            zupt.get_measurement(&state).unwrap() - zupt.get_expected_measurement(&state);
        assert_approx_eq!(innovation[0], -0.3, EPS);
        assert_approx_eq!(innovation[1], 0.7, EPS);
        assert_approx_eq!(innovation[2], -0.05, EPS);
    }

    #[test]
    fn zupt_constrains_all_three_velocity_axes() {
        // The vertical channel is the one that most needs the constraint, so a ZUPT
        // that quietly skipped v_d would be worse than useless.
        let zupt = ZuptMeasurement::default();
        let h = zupt.get_jacobian(&state_15([0.0; 3], [0.0; 3])).unwrap();
        assert_eq!((h.nrows(), h.ncols()), (3, 9));
        for (row, column) in (3..6).enumerate() {
            assert_approx_eq!(h[(row, column)], 1.0, EPS);
        }
        // Nothing but velocity is observed.
        assert_approx_eq!(h.view((0, 0), (3, 3)).iter().sum::<f64>(), 0.0, EPS);
        assert_approx_eq!(h.view((0, 6), (3, 3)).iter().sum::<f64>(), 0.0, EPS);
    }

    #[test]
    fn zupt_pseudo_measurement_is_zero_regardless_of_state() {
        let zupt = ZuptMeasurement::default();
        for velocity in [[0.0; 3], [10.0, -4.0, 1.0], [-100.0, 100.0, -100.0]] {
            let z = zupt.get_measurement(&state_15(velocity, [0.0; 3])).unwrap();
            assert_approx_eq!(z.norm(), 0.0, EPS);
        }
    }

    #[test]
    fn zupt_noise_is_the_configured_variance() {
        let zupt = ZuptMeasurement::new(0.05).unwrap();
        let r = zupt.get_noise();
        assert_eq!((r.nrows(), r.ncols()), (3, 3));
        for axis in 0..3 {
            assert_approx_eq!(r[(axis, axis)], 0.0025, EPS);
        }
    }

    #[test]
    fn zupt_rejects_a_degenerate_noise_value() {
        // A zero standard deviation gives a singular R; the resulting gain is not
        // something to discover at runtime.
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(ZuptMeasurement::new(bad).is_err(), "accepted std {bad}");
        }
    }

    #[test]
    fn zupt_reports_a_short_state_through_the_jacobian() {
        let zupt = ZuptMeasurement::default();
        let short = DVector::zeros(5);
        assert!(matches!(
            zupt.get_jacobian(&short).unwrap_err(),
            StrapdownError::DimensionMismatch { .. }
        ));
        // The infallible predictor must not index past the end while doing so.
        assert_approx_eq!(zupt.get_expected_measurement(&short).norm(), 0.0, EPS);
    }

    #[test]
    fn zaru_innovation_is_the_uncompensated_bias_error() {
        // A stationary gyro reading 0.002 rad/s is reading its own bias. With the
        // filter estimating zero bias, the whole reading is innovation.
        let zaru = ZaruMeasurement::from_gyro([0.0, 0.0, 0.002]).set_earth_rate_compensation(false);
        let state = state_15([0.0; 3], [0.0; 3]);
        let innovation =
            zaru.get_measurement(&state).unwrap() - zaru.get_expected_measurement(&state);
        assert_approx_eq!(innovation[2], 0.002, EPS);

        // Once the filter has learned that bias, the innovation vanishes -- which is
        // what "the update has converged" has to look like.
        let mut learned = state;
        learned[14] = 0.002;
        let converged =
            zaru.get_measurement(&learned).unwrap() - zaru.get_expected_measurement(&learned);
        assert_approx_eq!(converged.norm(), 0.0, EPS);
    }

    #[test]
    fn zaru_observes_only_the_gyro_bias_block() {
        let zaru = ZaruMeasurement::default();
        let h = zaru.get_jacobian(&state_15([0.0; 3], [0.0; 3])).unwrap();
        assert_eq!((h.nrows(), h.ncols()), (3, 15));
        for (row, column) in (12..15).enumerate() {
            assert_approx_eq!(h[(row, column)], 1.0, EPS);
        }
        // Position, velocity, attitude and accelerometer bias are all unobserved:
        // a stationary gyro reading says nothing about where the platform is.
        assert_approx_eq!(h.view((0, 0), (3, 12)).iter().sum::<f64>(), 0.0, EPS);
    }

    #[test]
    fn zaru_requires_a_filter_that_carries_gyro_biases() {
        // Against a 9-state filter ZARU would correct nothing at all, and "the update
        // ran and changed no state" is indistinguishable from "the update worked" in
        // a log. Both Result-returning entry points have to say so.
        let zaru = ZaruMeasurement::default();
        let nine_state = DVector::zeros(9);
        assert!(matches!(
            zaru.get_measurement(&nine_state).unwrap_err(),
            StrapdownError::DimensionMismatch {
                expected: 15,
                got: 9,
                ..
            }
        ));
        assert!(matches!(
            zaru.get_jacobian(&nine_state).unwrap_err(),
            StrapdownError::DimensionMismatch {
                expected: 15,
                got: 9,
                ..
            }
        ));
    }

    #[test]
    fn zaru_earth_rate_term_is_resolved_into_the_body_frame() {
        // At 45 deg N with a level, north-facing platform, Earth rate in NED is
        // (w cos lat, 0, -w sin lat) and the body frame coincides with it.
        let zaru = ZaruMeasurement::default();
        let state = state_15([0.0; 3], [0.0, 0.0, 0.0]);
        let predicted = zaru.get_expected_measurement(&state);
        let rate = crate::earth::RATE;
        let latitude = state[0];
        assert_approx_eq!(predicted[0], rate * latitude.cos(), 1e-15);
        assert_approx_eq!(predicted[1], 0.0, 1e-15);
        assert_approx_eq!(predicted[2], -rate * latitude.sin(), 1e-15);
    }

    #[test]
    fn zaru_earth_rate_term_rotates_with_attitude() {
        // Yawed 90 deg east, the north component of Earth rate should appear on the
        // body y axis instead of x. This is the term's only attitude dependence.
        let zaru = ZaruMeasurement::default();
        let level_north = zaru.get_expected_measurement(&state_15([0.0; 3], [0.0; 3]));
        let yawed = zaru
            .get_expected_measurement(&state_15([0.0; 3], [0.0, 0.0, std::f64::consts::FRAC_PI_2]));
        assert_approx_eq!(yawed[1], -level_north[0], 1e-15);
        assert_approx_eq!(yawed[0], 0.0, 1e-15);
        // Rotating about the down axis cannot change the down component.
        assert_approx_eq!(yawed[2], level_north[2], 1e-15);
    }

    #[test]
    fn zaru_earth_rate_compensation_can_be_disabled() {
        let compensated = ZaruMeasurement::default();
        let plain = ZaruMeasurement::default().set_earth_rate_compensation(false);
        let state = state_15([0.0; 3], [0.0; 3]);
        assert!(compensated.get_expected_measurement(&state).norm() > 0.0);
        assert_approx_eq!(plain.get_expected_measurement(&state).norm(), 0.0, EPS);
    }

    #[test]
    fn zaru_earth_rate_term_is_small_enough_to_ignore_in_the_jacobian() {
        // The Jacobian leaves the attitude block at zero even though the Earth-rate
        // term does depend on attitude. That is only defensible while the term is far
        // below the noise floor it is being compared against -- check the premise
        // rather than trusting the comment.
        let zaru = ZaruMeasurement::default();
        let earth_rate_magnitude = zaru
            .get_expected_measurement(&state_15([0.0; 3], [0.0; 3]))
            .norm();
        assert!(
            earth_rate_magnitude < 0.1 * DEFAULT_ZARU_NOISE_RPS,
            "Earth rate {earth_rate_magnitude} is no longer negligible against the \
             {DEFAULT_ZARU_NOISE_RPS} rad/s noise floor; the attitude block now matters"
        );
    }

    #[test]
    fn zaru_noise_is_the_configured_variance() {
        let zaru = ZaruMeasurement::new(Vector3::zeros(), 0.002).unwrap();
        let r = zaru.get_noise();
        for axis in 0..3 {
            assert_approx_eq!(r[(axis, axis)], 4.0e-6, EPS);
        }
    }

    #[test]
    fn zaru_rejects_degenerate_construction() {
        for bad in [0.0, -1.0, f64::NAN] {
            assert!(ZaruMeasurement::new(Vector3::zeros(), bad).is_err());
        }
        assert!(matches!(
            ZaruMeasurement::new(Vector3::new(f64::NAN, 0.0, 0.0), 1e-3).unwrap_err(),
            StrapdownError::NonFinite { .. }
        ));
    }

    #[test]
    fn zupt_and_zaru_report_their_dimensions() {
        assert_eq!(ZuptMeasurement::default().get_dimension(), 3);
        assert_eq!(ZaruMeasurement::default().get_dimension(), 3);
    }
}
