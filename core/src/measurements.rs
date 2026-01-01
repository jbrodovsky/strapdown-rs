//! Measurement-related code for the strapdown navigation system.
//!
//! This module defines generic measurement models and specific implementations
//! for GPS position, GPS velocity, combined GPS position and velocity, relative
//! altitude, and magnetometer-based yaw measurements. These models are used in
//! inertial navigation systems to process sensor data.

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
    fn get_measurement(&self, state: &DVector<f64>) -> DVector<f64>;
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
    fn get_measurement(&self, _state: &DVector<f64>) -> DVector<f64> {
        // GPS position measurement is state-independent
        DVector::from_vec(vec![
            self.latitude.to_radians(),
            self.longitude.to_radians(),
            self.altitude,
        ])
    }
    fn get_noise(&self) -> DMatrix<f64> {
        DMatrix::from_diagonal(&DVector::from_vec(vec![
            (self.horizontal_noise_std * METERS_TO_DEGREES).powi(2),
            (self.horizontal_noise_std * METERS_TO_DEGREES).powi(2),
            self.vertical_noise_std.powi(2),
        ]))
    }
    fn get_expected_measurement(&self, state: &DVector<f64>) -> DVector<f64> {
        DVector::from_vec(vec![state[0], state[1], state[2]])
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
    fn get_measurement(&self, _state: &DVector<f64>) -> DVector<f64> {
        // GPS velocity measurement is state-independent
        DVector::from_vec(vec![
            self.northward_velocity,
            self.eastward_velocity,
            self.vertical_velocity,
        ])
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
    fn get_measurement(&self, _state: &DVector<f64>) -> DVector<f64> {
        // GPS position and velocity measurement is state-independent
        DVector::from_vec(vec![
            self.latitude.to_radians(),
            self.longitude.to_radians(),
            self.altitude,
            self.northward_velocity,
            self.eastward_velocity,
        ])
    }
    fn get_noise(&self) -> DMatrix<f64> {
        DMatrix::from_diagonal(&DVector::from_vec(vec![
            (self.horizontal_noise_std * METERS_TO_DEGREES).powi(2),
            (self.horizontal_noise_std * METERS_TO_DEGREES).powi(2),
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
    fn get_measurement(&self, _state: &DVector<f64>) -> DVector<f64> {
        // Barometric altitude measurement is state-independent
        DVector::from_vec(vec![self.relative_altitude + self.reference_altitude])
    }
    fn get_noise(&self) -> DMatrix<f64> {
        DMatrix::from_diagonal(&DVector::from_vec(vec![5.0]))
    }
    fn get_expected_measurement(&self, state: &DVector<f64>) -> DVector<f64> {
        DVector::from_vec(vec![state[2]])
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
/// let z = mag_meas.get_measurement(&state);
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
            .unwrap_or_else(|_| Date::from_ordinal_date(2025, 1).unwrap());

        let field = GeomagneticField::new(
            Length::new::<meter>(alt_m as f32),
            Angle::new::<degree>(lat_deg as f32),
            Angle::new::<degree>(lon_deg as f32),
            date,
        );

        match field {
            Ok(f) => f.declination().get::<degree>() as f64 * std::f64::consts::PI / 180.0,
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

    fn get_measurement(&self, state: &DVector<f64>) -> DVector<f64> {
        // Extract roll and pitch from state for tilt compensation
        let roll = if state.len() > 6 { state[6] } else { 0.0 };
        let pitch = if state.len() > 7 { state[7] } else { 0.0 };
        let yaw = if state.len() > 8 { state[8] } else { 0.0 };

        // Compute tilt-compensated magnetic vector
        //let mut heading = self.compute_tilt_compensated_heading(roll, pitch);
        let attitude = Rotation3::from_euler_angles(roll, pitch, yaw);
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

        DVector::from_vec(vec![heading])
    }

    fn get_noise(&self) -> DMatrix<f64> {
        DMatrix::from_diagonal(&DVector::from_vec(vec![self.noise_std.powi(2)]))
    }

    fn get_expected_measurement(&self, state: &DVector<f64>) -> DVector<f64> {
        // Expected measurement is the yaw from the state (index 8)
        let yaw = if state.len() > 8 { state[8] } else { 0.0 };
        DVector::from_vec(vec![yaw])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert_approx_eq::assert_approx_eq;

    const EPS: f64 = 1e-12;

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
        let vec = meas.get_measurement(&dummy_state);
        assert_eq!(vec.len(), 3);
        assert!((vec[0] - 37.0_f64.to_radians()).abs() < EPS);
        assert!((vec[1] - (-122.0_f64).to_radians()).abs() < EPS);
        assert!((vec[2] - 12.34).abs() < EPS);

        // Noise diagonal entries
        let noise = meas.get_noise();
        let expected_h = (3.0 * METERS_TO_DEGREES).powi(2);
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

        let vec = meas.get_measurement(&dummy_state);
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

        let vec = meas.get_measurement(&dummy_state);
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

        let vec = meas.get_measurement(&dummy_state);
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
        let s = format!("{}", meas);
        assert!(s.contains("rel_alt") && s.contains("ref_alt"));
    }

    #[test]
    fn downcast_trait_object_and_display() {
        let pos = GPSPositionMeasurement::default();
        // pos.latitude = 1.0;
        // pos.longitude = 2.0;
        // pos.altitude = 3.0;
        let boxed: Box<dyn MeasurementModel> = Box::new(pos.clone());
        // downcast via as_any
        let any = boxed.as_any();
        let down = any
            .downcast_ref::<GPSPositionMeasurement>()
            .expect("downcast failed");
        assert!((down.latitude).abs() < EPS);

        // Display formatting
        let s = format!("{}", down);
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
            0.7854,  // lat (45 deg)
            -2.1293, // lon (-122 deg)
            100.0,   // alt
            0.0, 0.0, 0.0, // velocities
            0.0, // roll = 0
            0.0, // pitch = 0
            0.0, // yaw
        ]);

        let z = meas.get_measurement(&state);
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
            0.7854, -2.1293, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // level attitude
        ]);

        let z = meas.get_measurement(&state);

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
        let level_state =
            DVector::from_vec(vec![0.7854, -2.1293, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

        // Tilted state (10 degrees roll)
        let tilted_state = DVector::from_vec(vec![
            0.7854, -2.1293, 100.0, 0.0, 0.0, 0.0, 0.1745, 0.0, 0.0, // ~10 deg roll
        ]);

        let z_level = meas.get_measurement(&level_state);
        let z_tilted = meas.get_measurement(&tilted_state);

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
            0.7854, -2.1293, 100.0, 0.0, 0.0, 0.0, 0.1, 0.2, 1.5708, // yaw = π/2
        ]);

        let expected = meas.get_expected_measurement(&state);
        assert_eq!(expected.len(), 1);
        assert!(
            (expected[0] - 1.5708).abs() < EPS,
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

        let s = format!("{}", meas);
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
}
