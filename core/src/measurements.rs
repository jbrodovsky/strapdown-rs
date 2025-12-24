//! Measurement-related code for the strapdown navigation system.
//!
//! This module defines generic measurement models and specific implementations
//! for GPS position, GPS velocity, combined GPS position and velocity, and relative
//! altitude measurements. These models are used in inertial navigation systems to
//! process sensor data.

use crate::earth::METERS_TO_DEGREES;

use std::any::Any;
use std::fmt::{self, Debug, Display};

use nalgebra::{DMatrix, DVector, Matrix3, Rotation3, Vector3};

/// Generic measurement model trait for all types of measurements
pub trait MeasurementModel: Any {
    /// Downcast helper method to allow for type-safe downcasting
    fn as_any(&self) -> &dyn Any;
    /// Downcast helper method for mutable references
    fn as_any_mut(&mut self) -> &mut dyn Any;
    /// Get the dimension of the measurement vector
    fn get_dimension(&self) -> usize;
    /// Get the measurement in a vector format
    fn get_vector(&self) -> DVector<f64>;
    /// Get the measurement noise characteristics in a matrix format
    fn get_noise(&self) -> DMatrix<f64>;
    //fn get_sigma_points(&self, state_sigma_points: &DMatrix<f64>) -> DMatrix<f64>
    /// Get the expected measurements from the state. Measurement model function
    /// that maps the state values to measurement space.
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
    fn get_vector(&self) -> DVector<f64> {
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
    // fn get_sigma_points(&self, state_sigma_points: &DMatrix<f64>) -> DMatrix<f64> {
    //     let mut measurement_sigma_points = DMatrix::<f64>::zeros(3, state_sigma_points.ncols());
    //     for (i, sigma_point) in state_sigma_points.column_iter().enumerate() {
    //         measurement_sigma_points[(0, i)] = sigma_point[0];
    //         measurement_sigma_points[(1, i)] = sigma_point[1];
    //         measurement_sigma_points[(2, i)] = sigma_point[2];
    //     }
    //     measurement_sigma_points
    // }
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
    fn get_vector(&self) -> DVector<f64> {
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
    // fn get_sigma_points(&self, state_sigma_points: &DMatrix<f64>) -> DMatrix<f64> {
    //     let mut measurement_sigma_points = DMatrix::<f64>::zeros(3, state_sigma_points.ncols());
    //     for (i, sigma_point) in state_sigma_points.column_iter().enumerate() {
    //         measurement_sigma_points[(0, i)] = sigma_point[3];
    //         measurement_sigma_points[(1, i)] = sigma_point[4];
    //         measurement_sigma_points[(2, i)] = sigma_point[5];
    //     }
    //     measurement_sigma_points
    // }
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
    fn get_vector(&self) -> DVector<f64> {
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
    fn get_vector(&self) -> DVector<f64> {
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

/// Magnetometer measurement model
///
/// This measurement model maps the vehicle attitude state to expected magnetic field
/// measurements in the body frame. It supports calibration parameters for hard-iron
/// offset correction and soft-iron correction (scale and misalignment).
///
/// The measurement is the three-axis magnetic field vector (mag_x, mag_y, mag_z) in
/// microteslas (μT) measured in the body frame of the vehicle.
///
/// # Reference Field
///
/// The reference magnetic field can be provided in two ways:
/// 1. **Manual**: Set `reference_field_ned` directly with known values
/// 2. **Automatic (WMM)**: Set `use_wmm` to true and provide date information. The reference
///    field will be computed from the state's position using the World Magnetic Model.
///
/// # Calibration Parameters
///
/// - **Hard-iron offset**: A constant bias in the magnetometer readings caused by
///   ferromagnetic materials on the vehicle. This is represented as a 3D vector
///   (offset_x, offset_y, offset_z) in μT that is subtracted from raw measurements.
///
/// - **Soft-iron correction**: A 3x3 matrix that corrects for scale factors and
///   cross-axis misalignments caused by nearby magnetic materials. Applied as:
///   `corrected = soft_iron_matrix * (raw - hard_iron_offset)`
///
/// # Measurement Model
///
/// The expected measurement is computed by:
/// 1. Extract roll, pitch, yaw from the state vector (indices 6, 7, 8)
/// 2. If `use_wmm` is true, compute the reference field from position (indices 0, 1, 2)
/// 3. Compute rotation matrix from NED frame to body frame using attitude
/// 4. Rotate the local NED magnetic field reference vector into the body frame
/// 5. Apply soft-iron and hard-iron calibration corrections
///
/// # Example
/// ```rust
/// use nalgebra::{Vector3, Matrix3};
/// use strapdown::measurements::MagnetometerMeasurement;
///
/// // Manual reference field (e.g., from prior calibration)
/// let mut meas = MagnetometerMeasurement {
///     mag_x: 20.0,
///     mag_y: 5.0,
///     mag_z: 40.0,
///     reference_field_ned: Vector3::new(25.0, 0.0, 45.0),
///     use_wmm: false,
///     wmm_year: 2025,
///     wmm_day_of_year: 1,
///     hard_iron_offset: Vector3::zeros(),
///     soft_iron_matrix: Matrix3::identity(),
///     noise_std: 3.0,
/// };
///
/// // Or use WMM to compute reference field automatically
/// meas.use_wmm = true;
/// meas.wmm_year = 2025;
/// meas.wmm_day_of_year = 180;  // Mid-year
/// ```
#[derive(Clone, Debug)]
pub struct MagnetometerMeasurement {
    /// Measured magnetic field in body frame X-axis (μT)
    pub mag_x: f64,
    /// Measured magnetic field in body frame Y-axis (μT)
    pub mag_y: f64,
    /// Measured magnetic field in body frame Z-axis (μT)
    pub mag_z: f64,
    /// Reference magnetic field vector in NED frame (μT)
    /// [North, East, Down] components
    /// If `use_wmm` is true, this is computed automatically from the state position
    pub reference_field_ned: Vector3<f64>,
    /// Use World Magnetic Model to compute reference field from position
    pub use_wmm: bool,
    /// Year for WMM calculation (e.g., 2025)
    pub wmm_year: i32,
    /// Day of year for WMM calculation (1-365 or 1-366)
    pub wmm_day_of_year: u16,
    /// Hard-iron offset correction vector (μT)
    /// This offset is subtracted from raw measurements
    pub hard_iron_offset: Vector3<f64>,
    /// Soft-iron correction matrix (dimensionless)
    /// Applied to correct for scale and misalignment
    pub soft_iron_matrix: Matrix3<f64>,
    /// Measurement noise standard deviation (μT)
    pub noise_std: f64,
}

impl Default for MagnetometerMeasurement {
    fn default() -> Self {
        Self {
            mag_x: 0.0,
            mag_y: 0.0,
            mag_z: 0.0,
            // Typical mid-latitude northern hemisphere magnetic field
            // Users should set this to match their geographic location
            // using Earth magnetic models (WMM, IGRF) or local calibration,
            // or enable use_wmm for automatic calculation
            reference_field_ned: Vector3::new(25.0, 0.0, 40.0),
            use_wmm: false,
            wmm_year: 2025,
            wmm_day_of_year: 1,
            hard_iron_offset: Vector3::zeros(),
            soft_iron_matrix: Matrix3::identity(),
            noise_std: 5.0, // Default 5 μT noise std
        }
    }
}

impl Display for MagnetometerMeasurement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ref_source = if self.use_wmm {
            format!("WMM({}/{})", self.wmm_year, self.wmm_day_of_year)
        } else {
            "Manual".to_string()
        };
        write!(
            f,
            "MagnetometerMeasurement(mag: [{:.2}, {:.2}, {:.2}] μT, ref: [{:.2}, {:.2}, {:.2}] μT ({}), noise: {:.2} μT)",
            self.mag_x,
            self.mag_y,
            self.mag_z,
            self.reference_field_ned[0],
            self.reference_field_ned[1],
            self.reference_field_ned[2],
            ref_source,
            self.noise_std
        )
    }
}

impl MeasurementModel for MagnetometerMeasurement {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn get_dimension(&self) -> usize {
        3
    }

    fn get_vector(&self) -> DVector<f64> {
        // Apply calibration corrections to raw measurements
        let raw_mag = Vector3::new(self.mag_x, self.mag_y, self.mag_z);
        let corrected_mag = self.soft_iron_matrix * (raw_mag - self.hard_iron_offset);
        DVector::from_vec(vec![corrected_mag[0], corrected_mag[1], corrected_mag[2]])
    }

    fn get_noise(&self) -> DMatrix<f64> {
        // Isotropic noise model: same noise in all three axes
        DMatrix::from_diagonal(&DVector::from_vec(vec![
            self.noise_std.powi(2),
            self.noise_std.powi(2),
            self.noise_std.powi(2),
        ]))
    }

    fn get_expected_measurement(&self, state: &DVector<f64>) -> DVector<f64> {
        // Extract attitude from state: roll (6), pitch (7), yaw (8)
        let roll = state[6];
        let pitch = state[7];
        let yaw = state[8];

        // Get reference field - either from WMM or use the provided value
        let reference_field = if self.use_wmm {
            // Extract position from state: lat (0), lon (1), alt (2)
            let latitude = state[0].to_degrees();
            let longitude = state[1].to_degrees();
            let altitude = state[2];
            
            // Compute magnetic field using WMM
            if let Some(field) = crate::earth::magnetic_field_wmm(
                &latitude,
                &longitude,
                &altitude,
                self.wmm_year,
                self.wmm_day_of_year,
            ) {
                field
            } else {
                // Fallback to stored reference field if WMM fails
                self.reference_field_ned
            }
        } else {
            self.reference_field_ned
        };

        // Compute rotation matrix from body to NED frame (C_bn)
        // This is what from_euler_angles gives us
        let rot_body_to_ned = Rotation3::from_euler_angles(roll, pitch, yaw);

        // To rotate from NED to body, we need the transpose (inverse for rotation matrices)
        let rot_ned_to_body = rot_body_to_ned.transpose();

        // Rotate reference field from NED to body frame
        let mag_body = rot_ned_to_body * reference_field;

        // Apply calibration transform to get expected measurement in calibrated space
        // This matches the calibration applied in get_vector() to the raw measurements
        // Both the expected and actual measurements are in the same calibrated space
        let expected_mag = self.soft_iron_matrix * (mag_body - self.hard_iron_offset);

        DVector::from_vec(vec![expected_mag[0], expected_mag[1], expected_mag[2]])
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

        // Vector in radians for lat/lon
        let vec = meas.get_vector();
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

        let vec = meas.get_vector();
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
        let vec = meas.get_vector();
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
        let vec = meas.get_vector();
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
    fn magnetometer_measurement_default() {
        let meas = MagnetometerMeasurement::default();
        assert_eq!(meas.get_dimension(), 3);
        assert!(meas.noise_std > 0.0);
        assert_eq!(meas.hard_iron_offset, Vector3::zeros());
        assert_eq!(meas.soft_iron_matrix, Matrix3::identity());
    }

    #[test]
    fn magnetometer_measurement_vector_and_noise() {
        let meas = MagnetometerMeasurement {
            mag_x: 20.0,
            mag_y: 5.0,
            mag_z: 40.0,
            reference_field_ned: Vector3::new(25.0, 0.0, 45.0),
            use_wmm: false,
            wmm_year: 2025,
            wmm_day_of_year: 1,
            hard_iron_offset: Vector3::zeros(),
            soft_iron_matrix: Matrix3::identity(),
            noise_std: 3.0,
        };

        // Get measurement vector (should be same as input with identity calibration)
        let vec = meas.get_vector();
        assert_eq!(vec.len(), 3);
        assert_approx_eq!(vec[0], 20.0, EPS);
        assert_approx_eq!(vec[1], 5.0, EPS);
        assert_approx_eq!(vec[2], 40.0, EPS);

        // Check noise matrix
        let noise = meas.get_noise();
        assert_eq!(noise.nrows(), 3);
        assert_approx_eq!(noise[(0, 0)], 9.0, EPS); // 3.0^2
        assert_approx_eq!(noise[(1, 1)], 9.0, EPS);
        assert_approx_eq!(noise[(2, 2)], 9.0, EPS);
    }

    #[test]
    fn magnetometer_hard_iron_correction() {
        let hard_iron = Vector3::new(5.0, -2.0, 3.0);
        let meas = MagnetometerMeasurement {
            mag_x: 25.0,
            mag_y: 3.0,
            mag_z: 43.0,
            reference_field_ned: Vector3::new(25.0, 0.0, 45.0),
            use_wmm: false,
            wmm_year: 2025,
            wmm_day_of_year: 1,
            hard_iron_offset: hard_iron,
            soft_iron_matrix: Matrix3::identity(),
            noise_std: 3.0,
        };

        // After correction: raw - offset
        let vec = meas.get_vector();
        assert_approx_eq!(vec[0], 20.0, EPS); // 25 - 5
        assert_approx_eq!(vec[1], 5.0, EPS); // 3 - (-2)
        assert_approx_eq!(vec[2], 40.0, EPS); // 43 - 3
    }

    #[test]
    fn magnetometer_soft_iron_correction() {
        // Simple scale matrix (scales X by 2, Y by 0.5, Z unchanged)
        let soft_iron = Matrix3::new(2.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0);
        let meas = MagnetometerMeasurement {
            mag_x: 10.0,
            mag_y: 10.0,
            mag_z: 40.0,
            reference_field_ned: Vector3::new(25.0, 0.0, 45.0),
            use_wmm: false,
            wmm_year: 2025,
            wmm_day_of_year: 1,
            hard_iron_offset: Vector3::zeros(),
            soft_iron_matrix: soft_iron,
            noise_std: 3.0,
        };

        let vec = meas.get_vector();
        assert_approx_eq!(vec[0], 20.0, EPS); // 10 * 2
        assert_approx_eq!(vec[1], 5.0, EPS); // 10 * 0.5
        assert_approx_eq!(vec[2], 40.0, EPS); // 40 * 1
    }

    #[test]
    fn magnetometer_expected_measurement_zero_attitude() {
        // State with zero attitude (aligned with NED)
        let state = DVector::from_vec(vec![
            0.0, // lat
            0.0, // lon
            0.0, // alt
            0.0, // v_n
            0.0, // v_e
            0.0, // v_d
            0.0, // roll
            0.0, // pitch
            0.0, // yaw
        ]);

        let reference = Vector3::new(25.0, 5.0, 40.0);
        let meas = MagnetometerMeasurement {
            mag_x: 0.0,
            mag_y: 0.0,
            mag_z: 0.0,
            reference_field_ned: reference,
            use_wmm: false,
            wmm_year: 2025,
            wmm_day_of_year: 1,
            hard_iron_offset: Vector3::zeros(),
            soft_iron_matrix: Matrix3::identity(),
            noise_std: 3.0,
        };

        // With zero attitude, body frame = NED frame
        let expected = meas.get_expected_measurement(&state);
        assert_eq!(expected.len(), 3);
        assert_approx_eq!(expected[0], 25.0, 1e-6);
        assert_approx_eq!(expected[1], 5.0, 1e-6);
        assert_approx_eq!(expected[2], 40.0, 1e-6);
    }

    #[test]
    fn magnetometer_expected_measurement_yaw_rotation() {
        // State with 90 degree yaw rotation (pointing East)
        let state = DVector::from_vec(vec![
            0.0,                         // lat
            0.0,                         // lon
            0.0,                         // alt
            0.0,                         // v_n
            0.0,                         // v_e
            0.0,                         // v_d
            0.0,                         // roll
            0.0,                         // pitch
            std::f64::consts::FRAC_PI_2, // yaw = 90 degrees
        ]);

        // Reference field: 25 μT North, 0 East, 40 Down
        let reference = Vector3::new(25.0, 0.0, 40.0);
        let meas = MagnetometerMeasurement {
            mag_x: 0.0,
            mag_y: 0.0,
            mag_z: 0.0,
            reference_field_ned: reference,
            use_wmm: false,
            wmm_year: 2025,
            wmm_day_of_year: 1,
            hard_iron_offset: Vector3::zeros(),
            soft_iron_matrix: Matrix3::identity(),
            noise_std: 3.0,
        };

        // After 90° yaw: body_x points East, body_y points South
        // So North component (25) -> -body_y, East (0) -> body_x
        let expected = meas.get_expected_measurement(&state);
        assert_eq!(expected.len(), 3);
        assert_approx_eq!(expected[0], 0.0, 1e-6); // East component
        assert_approx_eq!(expected[1], -25.0, 1e-6); // -North component
        assert_approx_eq!(expected[2], 40.0, 1e-6); // Down unchanged
    }

    #[test]
    fn magnetometer_expected_measurement_with_calibration() {
        let state = DVector::from_vec(vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // roll
            0.0, // pitch
            0.0, // yaw
        ]);

        let reference = Vector3::new(20.0, 5.0, 40.0);
        let hard_iron = Vector3::new(1.0, -1.0, 2.0);
        let soft_iron = Matrix3::identity() * 1.1; // 10% scale factor

        let meas = MagnetometerMeasurement {
            mag_x: 0.0,
            mag_y: 0.0,
            mag_z: 0.0,
            reference_field_ned: reference,
            use_wmm: false,
            wmm_year: 2025,
            wmm_day_of_year: 1,
            hard_iron_offset: hard_iron,
            soft_iron_matrix: soft_iron,
            noise_std: 3.0,
        };

        let expected = meas.get_expected_measurement(&state);
        // Expected: soft_iron * (reference - hard_iron)
        // = 1.1 * ([20, 5, 40] - [1, -1, 2])
        // = 1.1 * [19, 6, 38]
        assert_approx_eq!(expected[0], 20.9, 1e-6); // 19 * 1.1
        assert_approx_eq!(expected[1], 6.6, 1e-6); // 6 * 1.1
        assert_approx_eq!(expected[2], 41.8, 1e-6); // 38 * 1.1
    }

    #[test]
    fn magnetometer_display() {
        let meas = MagnetometerMeasurement {
            mag_x: 20.0,
            mag_y: 5.0,
            mag_z: 40.0,
            reference_field_ned: Vector3::new(25.0, 0.0, 45.0),
            use_wmm: false,
            wmm_year: 2025,
            wmm_day_of_year: 1,
            hard_iron_offset: Vector3::zeros(),
            soft_iron_matrix: Matrix3::identity(),
            noise_std: 3.0,
        };
        let s = format!("{}", meas);
        assert!(s.contains("MagnetometerMeasurement"));
        assert!(s.contains("20.00"));
        assert!(s.contains("μT"));
    }

    #[test]
    fn magnetometer_downcast() {
        let meas = MagnetometerMeasurement::default();
        let boxed: Box<dyn MeasurementModel> = Box::new(meas.clone());

        let any = boxed.as_any();
        let down = any
            .downcast_ref::<MagnetometerMeasurement>()
            .expect("downcast failed");

        assert_eq!(down.get_dimension(), 3);
        assert_eq!(down.noise_std, meas.noise_std);
    }

    #[test]
    fn magnetometer_wmm_disabled() {
        // Test that WMM is disabled by default and uses manual reference field
        let meas = MagnetometerMeasurement {
            mag_x: 20.0,
            mag_y: 5.0,
            mag_z: 40.0,
            reference_field_ned: Vector3::new(25.0, 0.0, 45.0),
            use_wmm: false,
            wmm_year: 2025,
            wmm_day_of_year: 1,
            hard_iron_offset: Vector3::zeros(),
            soft_iron_matrix: Matrix3::identity(),
            noise_std: 3.0,
        };

        // State with zero attitude and position at Philadelphia
        let state = DVector::from_vec(vec![
            39.95_f64.to_radians(), // lat
            -75.16_f64.to_radians(), // lon
            100.0,                    // alt
            0.0,                      // v_n
            0.0,                      // v_e
            0.0,                      // v_d
            0.0,                      // roll
            0.0,                      // pitch
            0.0,                      // yaw
        ]);

        let expected = meas.get_expected_measurement(&state);
        
        // With zero attitude, body frame = NED frame, so expected should equal reference field
        assert_approx_eq!(expected[0], 25.0, 1e-6);
        assert_approx_eq!(expected[1], 0.0, 1e-6);
        assert_approx_eq!(expected[2], 45.0, 1e-6);
    }

    #[test]
    fn magnetometer_wmm_enabled() {
        // Test that WMM computes reference field from state position
        let meas = MagnetometerMeasurement {
            mag_x: 20.0,
            mag_y: 5.0,
            mag_z: 40.0,
            reference_field_ned: Vector3::new(999.0, 999.0, 999.0), // Should be ignored
            use_wmm: true,
            wmm_year: 2025,
            wmm_day_of_year: 1,
            hard_iron_offset: Vector3::zeros(),
            soft_iron_matrix: Matrix3::identity(),
            noise_std: 3.0,
        };

        // State with zero attitude and position at Philadelphia
        let state = DVector::from_vec(vec![
            39.95_f64.to_radians(), // lat
            -75.16_f64.to_radians(), // lon
            100.0,                    // alt
            0.0,                      // v_n
            0.0,                      // v_e
            0.0,                      // v_d
            0.0,                      // roll
            0.0,                      // pitch
            0.0,                      // yaw
        ]);

        let expected = meas.get_expected_measurement(&state);
        
        // With zero attitude, expected should be WMM field at this location
        // We can't check exact values but can verify reasonableness
        assert!(expected[0].abs() < 100.0, "North component should be reasonable");
        assert!(expected[1].abs() < 100.0, "East component should be reasonable");
        assert!(expected[2] > 0.0, "Down component should be positive in northern hemisphere");
        
        // Should not be the default reference field
        assert!((expected[0] - 999.0).abs() > 1.0, "Should use WMM, not default reference");
    }

    #[test]
    fn magnetometer_wmm_fallback() {
        // Test that WMM falls back to manual reference field on error
        let meas = MagnetometerMeasurement {
            mag_x: 20.0,
            mag_y: 5.0,
            mag_z: 40.0,
            reference_field_ned: Vector3::new(25.0, 0.0, 45.0),
            use_wmm: true,
            wmm_year: 2025,
            wmm_day_of_year: 366, // Invalid day for non-leap year
            hard_iron_offset: Vector3::zeros(),
            soft_iron_matrix: Matrix3::identity(),
            noise_std: 3.0,
        };

        let state = DVector::from_vec(vec![
            39.95_f64.to_radians(), // lat
            -75.16_f64.to_radians(), // lon
            100.0,                    // alt
            0.0, 0.0, 0.0,            // velocities
            0.0, 0.0, 0.0,            // attitude
        ]);

        let expected = meas.get_expected_measurement(&state);
        
        // Should fall back to manual reference field
        assert_approx_eq!(expected[0], 25.0, 1e-6);
        assert_approx_eq!(expected[1], 0.0, 1e-6);
        assert_approx_eq!(expected[2], 45.0, 1e-6);
    }

    #[test]
    fn magnetometer_wmm_with_rotation() {
        // Test WMM with non-zero attitude
        let meas = MagnetometerMeasurement {
            mag_x: 0.0,
            mag_y: 0.0,
            mag_z: 0.0,
            reference_field_ned: Vector3::zeros(), // Unused with WMM
            use_wmm: true,
            wmm_year: 2025,
            wmm_day_of_year: 180,
            hard_iron_offset: Vector3::zeros(),
            soft_iron_matrix: Matrix3::identity(),
            noise_std: 3.0,
        };

        // State with 90° yaw rotation
        let state = DVector::from_vec(vec![
            39.95_f64.to_radians(),              // lat
            -75.16_f64.to_radians(),             // lon
            100.0,                                // alt
            0.0, 0.0, 0.0,                        // velocities
            0.0,                                  // roll
            0.0,                                  // pitch
            std::f64::consts::FRAC_PI_2,         // yaw = 90°
        ]);

        let expected = meas.get_expected_measurement(&state);
        
        // All components should be rotated, none should be exactly zero
        assert!(expected[0].abs() > 0.1 || expected[1].abs() > 0.1, 
                "Rotation should affect expected measurement");
    }

    #[test]
    fn magnetometer_display_shows_wmm_status() {
        // Test manual mode display
        let meas_manual = MagnetometerMeasurement {
            mag_x: 20.0,
            mag_y: 5.0,
            mag_z: 40.0,
            reference_field_ned: Vector3::new(25.0, 0.0, 45.0),
            use_wmm: false,
            wmm_year: 2025,
            wmm_day_of_year: 1,
            hard_iron_offset: Vector3::zeros(),
            soft_iron_matrix: Matrix3::identity(),
            noise_std: 3.0,
        };
        let s = format!("{}", meas_manual);
        assert!(s.contains("Manual"), "Display should show manual mode");

        // Test WMM mode display
        let meas_wmm = MagnetometerMeasurement {
            use_wmm: true,
            wmm_year: 2025,
            wmm_day_of_year: 180,
            ..meas_manual
        };
        let s = format!("{}", meas_wmm);
        assert!(s.contains("WMM"), "Display should show WMM mode");
        assert!(s.contains("2025"), "Display should show WMM year");
        assert!(s.contains("180"), "Display should show WMM day");
    }
}
