//! Measurement-related code for the strapdown navigation system.
//! 
//! This module defines generic measurement models and specific implementations
//! for GPS position, GPS velocity, combined GPS position and velocity, and relative 
//! altitude measurements. These models are used in inertial navigation systems to 
//! process sensor data.

use crate::earth::METERS_TO_DEGREES;

use std::any::Any;
use std::fmt::{self, Debug, Display};

use nalgebra::{DMatrix, DVector};

/// Generic measurement model trait for all types of measurements
pub trait MeasurementModel: Any {
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
    fn get_dimension(&self) -> usize;
    fn get_vector(&self) -> DVector<f64>;
    fn get_noise(&self) -> DMatrix<f64>;
    fn get_sigma_points(&self, state_sigma_points: &DMatrix<f64>) -> DMatrix<f64>;
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
    fn as_any(&self) -> &dyn Any { self }
    fn as_any_mut(&mut self) -> &mut dyn Any { self }
    fn get_dimension(&self) -> usize { 3 }
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
    fn get_sigma_points(&self, state_sigma_points: &DMatrix<f64>) -> DMatrix<f64> {
        let mut measurement_sigma_points = DMatrix::<f64>::zeros(3, state_sigma_points.ncols());
        for (i, sigma_point) in state_sigma_points.column_iter().enumerate() {
            measurement_sigma_points[(0, i)] = sigma_point[0];
            measurement_sigma_points[(1, i)] = sigma_point[1];
            measurement_sigma_points[(2, i)] = sigma_point[2];
        }
        measurement_sigma_points
    }
}

/// GPS Velocity measurement model
#[derive(Clone, Debug, Default)]
pub struct GPSVelocityMeasurement {
    pub northward_velocity: f64,
    pub eastward_velocity: f64,
    pub downward_velocity: f64,
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
            self.downward_velocity,
            self.horizontal_noise_std,
            self.vertical_noise_std
        )
    }
}
impl MeasurementModel for GPSVelocityMeasurement {
    fn as_any(&self) -> &dyn Any { self }
    fn as_any_mut(&mut self) -> &mut dyn Any { self }
    fn get_dimension(&self) -> usize { 3 }
    fn get_vector(&self) -> DVector<f64> {
        DVector::from_vec(vec![
            self.northward_velocity,
            self.eastward_velocity,
            self.downward_velocity,
        ])
    }
    fn get_noise(&self) -> DMatrix<f64> {
        DMatrix::from_diagonal(&DVector::from_vec(vec![
            self.horizontal_noise_std.powi(2),
            self.horizontal_noise_std.powi(2),
            self.vertical_noise_std.powi(2),
        ]))
    }
    fn get_sigma_points(&self, state_sigma_points: &DMatrix<f64>) -> DMatrix<f64> {
        let mut measurement_sigma_points = DMatrix::<f64>::zeros(3, state_sigma_points.ncols());
        for (i, sigma_point) in state_sigma_points.column_iter().enumerate() {
            measurement_sigma_points[(0, i)] = sigma_point[3];
            measurement_sigma_points[(1, i)] = sigma_point[4];
            measurement_sigma_points[(2, i)] = sigma_point[5];
        }
        measurement_sigma_points
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
    fn as_any(&self) -> &dyn Any { self }
    fn as_any_mut(&mut self) -> &mut dyn Any { self }
    fn get_dimension(&self) -> usize { 5 }
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
    fn get_sigma_points(&self, state_sigma_points: &DMatrix<f64>) -> DMatrix<f64> {
        let mut measurement_sigma_points = DMatrix::<f64>::zeros(5, state_sigma_points.ncols());
        for (i, sigma_point) in state_sigma_points.column_iter().enumerate() {
            measurement_sigma_points[(0, i)] = sigma_point[0];
            measurement_sigma_points[(1, i)] = sigma_point[1];
            measurement_sigma_points[(2, i)] = sigma_point[2];
            measurement_sigma_points[(3, i)] = sigma_point[3];
            measurement_sigma_points[(4, i)] = sigma_point[4];
        }
        measurement_sigma_points
    }
}

/// Relative altitude measurement (barometric)
#[derive(Clone, Debug, Default)]
pub struct RelativeAltitudeMeasurement {
    pub relative_altitude: f64,
    pub reference_altitude: f64,
}
impl Display for RelativeAltitudeMeasurement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RelativeAltitudeMeasurement(rel_alt: {}, ref_alt: {})", self.relative_altitude, self.reference_altitude)
    }
}
impl MeasurementModel for RelativeAltitudeMeasurement {
    fn as_any(&self) -> &dyn Any { self }
    fn as_any_mut(&mut self) -> &mut dyn Any { self }
    fn get_dimension(&self) -> usize { 1 }
    fn get_vector(&self) -> DVector<f64> { DVector::from_vec(vec![self.relative_altitude + self.reference_altitude]) }
    fn get_noise(&self) -> DMatrix<f64> { DMatrix::from_diagonal(&DVector::from_vec(vec![5.0])) }
    fn get_sigma_points(&self, state_sigma_points: &DMatrix<f64>) -> DMatrix<f64> {
        let mut measurement_sigma_points = DMatrix::<f64>::zeros(self.get_dimension(), state_sigma_points.ncols());
        for (i, sigma_point) in state_sigma_points.column_iter().enumerate() {
            measurement_sigma_points[(0, i)] = sigma_point[2];
        }
        measurement_sigma_points
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DMatrix;

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

        // Sigma points: prepare a state sigma matrix with 3 rows (lat,lon,alt) and 2 columns
        let state_sigma = DMatrix::from_vec(6, 2, vec![
            0.1, 0.2, // lat
            1.1, 1.2, // lon
            2.1, 2.2, // alt
            3.0, 3.1, // v_n
            4.0, 4.1, // v_e
            5.0, 5.1, // v_d
        ]);
        let z = meas.get_sigma_points(&state_sigma);
        assert_eq!(z.nrows(), 3);
        assert_eq!(z.ncols(), 2);
        assert!((z[(0, 0)] - 0.1).abs() < EPS);
        assert!((z[(1, 1)] - 1.2).abs() < EPS);
        assert!((z[(2, 0)] - 2.1).abs() < EPS);
    }

    #[test]
    fn gps_velocity_vector_noise_and_sigma_points() {
        let meas = GPSVelocityMeasurement {
            northward_velocity: 1.5,
            eastward_velocity: -0.5,
            downward_velocity: 0.25,
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

        // Build a state sigma matrix where velocity entries are at indices 3..5
        let state_sigma = DMatrix::from_vec(6, 3, vec![
            0.0, 0.0, 0.0, // lat,lon,alt
            1.5, 1.6, 1.7, // v_n
            -0.5, -0.6, -0.7, // v_e
            0.25, 0.35, 0.45, // v_d
            9.0, 9.0, 9.0, // extra
            8.0, 8.0, 8.0, // extra
        ]);
        let z = meas.get_sigma_points(&state_sigma);
        assert_eq!(z.nrows(), 3);
        assert_eq!(z.ncols(), 3);
        assert!((z[(0, 2)] - 1.7).abs() < EPS);
        assert!((z[(1, 0)] - (-0.5)).abs() < EPS);
        assert!((z[(2, 1)] - 0.35).abs() < EPS);
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
        // position noises use METERS_TO_DEGREES
        assert!((noise[(0, 0)] - (1.0 * METERS_TO_DEGREES).powi(2)).abs() < EPS);
        assert!((noise[(3, 3)] - 0.5_f64.powi(2)).abs() < EPS);
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
        let state_sigma = DMatrix::from_vec(4, 2, vec![0.0, 0.0, 50.0, 60.0, 0.0, 0.0, 0.0, 0.0]);
        let z = meas.get_sigma_points(&state_sigma);
        assert_eq!(z.nrows(), 1);
        assert_eq!(z.ncols(), 2);
        assert!((z[(0, 0)] - 50.0).abs() < EPS);

        // Display string
        let s = format!("{}", meas);
        assert!(s.contains("rel_alt") && s.contains("ref_alt"));
    }

    #[test]
    fn downcast_trait_object_and_display() {
        let mut pos = GPSPositionMeasurement::default();
        pos.latitude = 1.0; pos.longitude = 2.0; pos.altitude = 3.0;
        let boxed: Box<dyn MeasurementModel> = Box::new(pos.clone());
        // downcast via as_any
        let any = boxed.as_any();
        let down = any.downcast_ref::<GPSPositionMeasurement>().expect("downcast failed");
        assert!((down.latitude - 1.0).abs() < EPS);

        // Display formatting
        let s = format!("{}", down);
        assert!(s.contains("GPSPositionMeasurement"));
    }

    #[test]
    fn negative_and_zero_std_are_handled() {
        let meas = GPSVelocityMeasurement {
            northward_velocity: 0.0,
            eastward_velocity: 0.0,
            downward_velocity: 0.0,
            horizontal_noise_std: -2.0,
            vertical_noise_std: 0.0,
        };
        let noise = meas.get_noise();
        // negative std should be squared, resulting positive variance
        assert!((noise[(0, 0)] - 4.0).abs() < EPS);
        // zero std -> zero variance
        assert!((noise[(2, 2)] - 0.0).abs() < EPS);
    }
}