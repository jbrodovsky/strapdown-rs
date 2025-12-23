//! Kalman-style navigation filters (UKF/EKF)
//!
//! This module contains the traditional Kalman filter style implementation of strapdown
//! inertial navigation systems. These filter build on the dead-reckoning functions
//! provided in the top-level [lib] module.

use crate::linalg::{matrix_square_root, robust_spd_solve, symmetrize};
use crate::measurements::MeasurementModel;
use crate::{
    IMUData, NavigationFilter, StrapdownState, forward, wrap_to_2pi, wrap_to_180, wrap_to_360,
};

use std::fmt::{self, Debug, Display};

use nalgebra::{DMatrix, DVector, Rotation3};

/// Basic strapdown initial state struct
#[derive(Clone, Debug, Default)]
pub struct InitialState {
    pub latitude: f64,
    pub longitude: f64,
    pub altitude: f64,
    pub northward_velocity: f64,
    pub eastward_velocity: f64,
    pub vertical_velocity: f64,
    pub roll: f64,
    pub pitch: f64,
    pub yaw: f64,
    pub in_degrees: bool,
    pub is_enu: bool,
}
impl InitialState {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        latitude: f64,
        longitude: f64,
        altitude: f64,
        northward_velocity: f64,
        eastward_velocity: f64,
        vertical_velocity: f64,
        mut roll: f64,
        mut pitch: f64,
        mut yaw: f64,
        in_degrees: bool,
        is_enu: Option<bool>,
    ) -> Self {
        let latitude = if in_degrees {
            latitude
        } else {
            latitude.to_degrees()
        };
        let longitude = if in_degrees {
            wrap_to_180(longitude)
        } else {
            longitude
        };
        let is_enu = is_enu.unwrap_or(true);
        if in_degrees {
            roll = wrap_to_360(roll).to_radians();
            pitch = wrap_to_360(pitch).to_radians();
            yaw = wrap_to_360(yaw).to_radians();
        } else {
            roll = wrap_to_2pi(roll);
            pitch = wrap_to_2pi(pitch);
            yaw = wrap_to_2pi(yaw);
        }
        InitialState {
            latitude,
            longitude,
            altitude,
            northward_velocity,
            eastward_velocity,
            vertical_velocity,
            roll,
            pitch,
            yaw,
            in_degrees,
            is_enu,
        }
    }
}
/// Unscented Kalman Filter implementation
#[derive(Clone)]
pub struct UnscentedKalmanFilter {
    mean_state: DVector<f64>,
    covariance: DMatrix<f64>,
    process_noise: DMatrix<f64>,
    lambda: f64,
    state_size: usize,
    weights_mean: DVector<f64>,
    weights_cov: DVector<f64>,
    is_enu: bool,
}
impl Debug for UnscentedKalmanFilter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UKF")
            .field("mean_state", &self.mean_state)
            .field("covariance", &self.covariance)
            .field("process_noise", &self.process_noise)
            .field("lambda", &self.lambda)
            .field("state_size", &self.state_size)
            .finish()
    }
}
impl Display for UnscentedKalmanFilter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("UnscentedKalmanFilter")
            .field("mean_state", &self.mean_state)
            .field("covariance", &self.covariance)
            .field("process_noise", &self.process_noise)
            .field("lambda", &self.lambda)
            .field("state_size", &self.state_size)
            .finish()
    }
}
impl UnscentedKalmanFilter {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        initial_state: InitialState,
        imu_biases: Vec<f64>,
        other_states: Option<Vec<f64>>,
        covariance_diagonal: Vec<f64>,
        process_noise: DMatrix<f64>,
        alpha: f64,
        beta: f64,
        kappa: f64,
    ) -> UnscentedKalmanFilter {
        let mut mean = if initial_state.in_degrees {
            vec![
                initial_state.latitude.to_radians(),
                initial_state.longitude.to_radians(),
                initial_state.altitude,
                initial_state.northward_velocity,
                initial_state.eastward_velocity,
                initial_state.vertical_velocity,
                initial_state.roll,
                initial_state.pitch,
                initial_state.yaw,
            ]
        } else {
            vec![
                initial_state.latitude,
                initial_state.longitude,
                initial_state.altitude,
                initial_state.northward_velocity,
                initial_state.eastward_velocity,
                initial_state.vertical_velocity,
                initial_state.roll,
                initial_state.pitch,
                initial_state.yaw,
            ]
        };
        mean.extend(imu_biases);
        if let Some(ref other_states) = other_states {
            mean.extend(other_states.iter().cloned());
        }
        let state_size = mean.len();
        let mean_state = DVector::from_vec(mean);
        let covariance = DMatrix::<f64>::from_diagonal(&DVector::from_vec(covariance_diagonal));
        let lambda = alpha * alpha * (state_size as f64 + kappa) - state_size as f64;
        let mut weights_mean = DVector::zeros(2 * state_size + 1);
        let mut weights_cov = DVector::zeros(2 * state_size + 1);
        weights_mean[0] = lambda / (state_size as f64 + lambda);
        weights_cov[0] = lambda / (state_size as f64 + lambda) + (1.0 - alpha * alpha + beta);
        for i in 1..(2 * state_size + 1) {
            let w = 1.0 / (2.0 * (state_size as f64 + lambda));
            weights_mean[i] = w;
            weights_cov[i] = w;
        }
        UnscentedKalmanFilter {
            mean_state,
            covariance,
            process_noise,
            lambda,
            state_size,
            weights_mean,
            weights_cov,
            is_enu: initial_state.is_enu,
        }
    }
    pub fn get_sigma_points(&self) -> DMatrix<f64> {
        let p = (self.state_size as f64 + self.lambda) * self.covariance.clone();
        let sqrt_p = matrix_square_root(&p);
        let mu = self.mean_state.clone();
        let mut pts = DMatrix::<f64>::zeros(self.state_size, 2 * self.state_size + 1);
        pts.column_mut(0).copy_from(&mu);
        for i in 0..sqrt_p.ncols() {
            pts.column_mut(i + 1).copy_from(&(&mu + sqrt_p.column(i)));
            pts.column_mut(i + 1 + self.state_size)
                .copy_from(&(&mu - sqrt_p.column(i)));
        }
        pts
    }
    fn robust_kalman_gain(
        &mut self,
        cross_covariance: &DMatrix<f64>,
        s: &DMatrix<f64>,
    ) -> DMatrix<f64> {
        let kt = robust_spd_solve(&symmetrize(s), &cross_covariance.transpose());
        kt.transpose()
    }
}
impl NavigationFilter for UnscentedKalmanFilter {
    fn predict<C: crate::InputModel>(&mut self, control_input: &C, dt: f64) {
        let imu_input = control_input
            .as_any()
            .downcast_ref::<IMUData>()
            .expect("UnscentedKalmanFilter.predict expects an IMUData InputModel");

        let mut sigma_points = self.get_sigma_points();
        for i in 0..sigma_points.ncols() {
            let mut sigma_point_vec = sigma_points.column(i).clone_owned();
            let mut state = StrapdownState {
                latitude: sigma_point_vec[0],
                longitude: sigma_point_vec[1],
                altitude: sigma_point_vec[2],
                velocity_north: sigma_point_vec[3],
                velocity_east: sigma_point_vec[4],
                velocity_vertical: sigma_point_vec[5],
                attitude: Rotation3::from_euler_angles(
                    sigma_point_vec[6],
                    sigma_point_vec[7],
                    sigma_point_vec[8],
                ),
                is_enu: self.is_enu,
            };
            let accel_biases = if self.state_size >= 15 {
                DVector::from_vec(vec![
                    sigma_point_vec[9],
                    sigma_point_vec[10],
                    sigma_point_vec[11],
                ])
            } else {
                DVector::from_vec(vec![0.0, 0.0, 0.0])
            };
            let gyro_biases = if self.state_size >= 15 {
                DVector::from_vec(vec![
                    sigma_point_vec[12],
                    sigma_point_vec[13],
                    sigma_point_vec[14],
                ])
            } else {
                DVector::from_vec(vec![0.0, 0.0, 0.0])
            };
            let imu_data = IMUData {
                accel: imu_input.accel - &accel_biases,
                gyro: imu_input.gyro - &gyro_biases,
            };
            forward(&mut state, imu_data, dt);
            sigma_point_vec[0] = state.latitude;
            sigma_point_vec[1] = state.longitude;
            sigma_point_vec[2] = state.altitude;
            sigma_point_vec[3] = state.velocity_north;
            sigma_point_vec[4] = state.velocity_east;
            sigma_point_vec[5] = state.velocity_vertical;
            sigma_point_vec[6] = state.attitude.euler_angles().0;
            sigma_point_vec[7] = state.attitude.euler_angles().1;
            sigma_point_vec[8] = state.attitude.euler_angles().2;
            sigma_points.set_column(i, &sigma_point_vec);
        }
        let mut mu_bar = DVector::<f64>::zeros(self.state_size);
        for (i, sigma_point) in sigma_points.column_iter().enumerate() {
            mu_bar += self.weights_mean[i] * sigma_point;
        }
        let mut p_bar = DMatrix::<f64>::zeros(self.state_size, self.state_size);
        for (i, sigma_point) in sigma_points.column_iter().enumerate() {
            let diff = sigma_point - &mu_bar;
            p_bar += self.weights_cov[i] * &diff * &diff.transpose();
        }
        p_bar += &self.process_noise;
        self.mean_state = mu_bar;
        self.covariance = symmetrize(&p_bar);
    }
    fn update<M: MeasurementModel + ?Sized>(&mut self, measurement: &M) {
        //let measurement_sigma_points = measurement.get_sigma_points(&self.get_sigma_points());
        let mut measurement_sigma_points =
            DMatrix::<f64>::zeros(measurement.get_dimension(), 2 * self.state_size + 1);
        let mut z_hat = DVector::<f64>::zeros(measurement.get_dimension());
        for (i, sigma_point) in self.get_sigma_points().column_iter().enumerate() {
            //let sigma_point_vec = sigma_point.clone_owned();
            let sigma_point = measurement.get_expected_measurement(&sigma_point.clone_owned());
            measurement_sigma_points.set_column(i, &sigma_point);
            z_hat += self.weights_mean[i] * sigma_point;
        }
        let mut s = DMatrix::<f64>::zeros(measurement.get_dimension(), measurement.get_dimension());
        for (i, sigma_point) in measurement_sigma_points.column_iter().enumerate() {
            let diff = sigma_point - &z_hat;
            s += self.weights_cov[i] * &diff * &diff.transpose();
        }
        s += measurement.get_noise();
        let sigma_points = self.get_sigma_points();
        let mut cross_covariance =
            DMatrix::<f64>::zeros(self.state_size, measurement.get_dimension());
        for (i, measurement_sigma_point) in measurement_sigma_points.column_iter().enumerate() {
            let measurement_diff = measurement_sigma_point - &z_hat;
            let state_diff = sigma_points.column(i) - &self.mean_state;
            cross_covariance += self.weights_cov[i] * state_diff * measurement_diff.transpose();
        }
        let k = self.robust_kalman_gain(&cross_covariance, &s);
        self.mean_state += &k * (measurement.get_vector() - &z_hat);
        self.mean_state[6] = wrap_to_2pi(self.mean_state[6]);
        self.mean_state[7] = wrap_to_2pi(self.mean_state[7]);
        self.mean_state[8] = wrap_to_2pi(self.mean_state[8]);
        self.covariance -= &k * &s * &k.transpose();
        // Ensure covariance remains positive semi-definite with gentle regularization
        self.covariance = symmetrize(&self.covariance);
        // Add small diagonal regularization to prevent negative eigenvalues
        let eps = 1e-9;
        for i in 0..self.state_size {
            self.covariance[(i, i)] += eps;
        }
    }
    fn get_estimate(&self) -> DVector<f64> {
        self.mean_state.clone()
    }
    fn get_certainty(&self) -> DMatrix<f64> {
        self.covariance.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::earth;
    use crate::measurements::{
        GPSPositionAndVelocityMeasurement, GPSPositionMeasurement, GPSVelocityMeasurement,
        RelativeAltitudeMeasurement,
    };
    use assert_approx_eq::assert_approx_eq;
    use nalgebra::Vector3;

    const IMU_BIASES: [f64; 6] = [0.0; 6];
    const N: usize = 15;
    const COVARIANCE_DIAGONAL: [f64; N] = [1e-9; N];
    const PROCESS_NOISE_DIAGONAL: [f64; N] = [1e-9; N];

    const ALPHA: f64 = 1e-3;
    const BETA: f64 = 2.0;
    const KAPPA: f64 = 0.0;
    const UKF_PARAMS: InitialState = InitialState {
        latitude: 0.0,
        longitude: 0.0,
        altitude: 0.0,
        northward_velocity: 0.0,
        eastward_velocity: 0.0,
        vertical_velocity: 0.0,
        roll: 0.0,
        pitch: 0.0,
        yaw: 0.0,
        in_degrees: false,
        is_enu: true,
    };

    #[test]
    fn ukf_construction() {
        let measurement_bias = vec![0.0; 3]; // Example measurement bias
        let ukf = UnscentedKalmanFilter::new(
            UKF_PARAMS,
            IMU_BIASES.to_vec(),
            Some(measurement_bias.clone()),
            vec![1e-3; 18],
            DMatrix::from_diagonal(&DVector::from_vec(vec![1e-3; 18])),
            ALPHA,
            BETA,
            KAPPA,
        );
        assert_eq!(ukf.mean_state.len(), 18);
        let wms = ukf.weights_mean;
        let wcs = ukf.weights_cov;
        assert_eq!(wms.len(), (2 * ukf.state_size) + 1);
        assert_eq!(wcs.len(), (2 * ukf.state_size) + 1);
        // Check that the weights are correct
        let lambda = ALPHA.powi(2) * (18.0 + KAPPA) - 18.0;
        assert_eq!(lambda, ukf.lambda);
        let wm_0 = lambda / (18.0 + lambda);
        let wc_0 = wm_0 + (1.0 - ALPHA.powi(2)) + BETA;
        let w_i = 1.0 / (2.0 * (18.0 + lambda));
        assert_approx_eq!(wms[0], wm_0, 1e-6);
        assert_approx_eq!(wcs[0], wc_0, 1e-6);
        for i in 1..wms.len() {
            assert_approx_eq!(wms[i], w_i, 1e-6);
            assert_approx_eq!(wcs[i], w_i, 1e-6);
        }
    }

    #[test]
    fn ukf_get_sigma_points() {
        let ukf = UnscentedKalmanFilter::new(
            UKF_PARAMS,
            IMU_BIASES.to_vec(),
            None,
            COVARIANCE_DIAGONAL.to_vec(),
            DMatrix::from_diagonal(&DVector::from_vec(PROCESS_NOISE_DIAGONAL.to_vec())),
            ALPHA,
            BETA,
            KAPPA,
        );
        let sigma_points = ukf.get_sigma_points();
        assert_eq!(sigma_points.ncols(), (2 * ukf.state_size) + 1);

        let mu = ukf.get_sigma_points() * ukf.weights_mean;
        assert_eq!(mu.nrows(), ukf.state_size);
        assert_eq!(mu.ncols(), 1);
        assert_approx_eq!(mu[0], 0.0, 1e-6);
        assert_approx_eq!(mu[1], 0.0, 1e-6);
        assert_approx_eq!(mu[2], 0.0, 1e-6);
        assert_approx_eq!(mu[3], 0.0, 1e-6);
        assert_approx_eq!(mu[4], 0.0, 1e-6);
        assert_approx_eq!(mu[5], 0.0, 1e-6);
        assert_approx_eq!(mu[6], 0.0, 1e-6);
        assert_approx_eq!(mu[7], 0.0, 1e-6);
        assert_approx_eq!(mu[8], 0.0, 1e-6);
    }

    #[test]
    fn ukf_propagate() {
        let mut ukf = UnscentedKalmanFilter::new(
            UKF_PARAMS,
            vec![0.0; 6],
            None,         //Some(measurement_bias.clone()),
            vec![0.0; N], // Absolute certainty use for testing the process
            DMatrix::from_diagonal(&DVector::from_vec(PROCESS_NOISE_DIAGONAL.to_vec())),
            1e-3,
            2.0,
            0.0,
        );
        let dt = 1.0;
        let imu_data = IMUData {
            accel: Vector3::new(0.0, 0.0, earth::gravity(&0.0, &0.0)),
            gyro: Vector3::new(0.0, 0.0, 0.0), // No rotation
        };
        ukf.predict(&imu_data, dt);
        assert!(
            ukf.mean_state.len() == 15 //+ measurement_bias.len()
        );
        let measurement = GPSPositionMeasurement {
            latitude: 0.0,
            longitude: 0.0,
            altitude: 0.0,
            horizontal_noise_std: 1e-3,
            vertical_noise_std: 1e-3,
        };
        ukf.update(&measurement);
        // Check that the state has not changed
        assert_approx_eq!(ukf.mean_state[0], 0.0, 1e-3);
        assert_approx_eq!(ukf.mean_state[1], 0.0, 1e-3);
        assert_approx_eq!(ukf.mean_state[2], 0.0, 0.1);
        assert_approx_eq!(ukf.mean_state[3], 0.0, 0.1);
        assert_approx_eq!(ukf.mean_state[4], 0.0, 0.1);
        assert_approx_eq!(ukf.mean_state[5], 0.0, 0.1);
    }

    #[test]
    fn ukf_debug_display() {
        // Test Debug and Display implementations for UKF
        let ukf = UnscentedKalmanFilter::new(
            UKF_PARAMS,
            IMU_BIASES.to_vec(),
            None,
            COVARIANCE_DIAGONAL.to_vec(),
            DMatrix::from_diagonal(&DVector::from_vec(PROCESS_NOISE_DIAGONAL.to_vec())),
            ALPHA,
            BETA,
            KAPPA,
        );

        // Test Debug
        let debug_str = format!("{:?}", ukf);
        assert!(debug_str.contains("UKF"));
        assert!(debug_str.contains("mean_state"));

        // Test Display
        let display_str = format!("{}", ukf);
        assert!(display_str.contains("UnscentedKalmanFilter"));
        assert!(display_str.contains("covariance"));
    }

    #[test]
    fn ukf_predict_with_biases() {
        // Test UKF predict with non-zero biases
        let mut ukf = UnscentedKalmanFilter::new(
            UKF_PARAMS,
            vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], // non-zero biases
            None,
            COVARIANCE_DIAGONAL.to_vec(),
            DMatrix::from_diagonal(&DVector::from_vec(PROCESS_NOISE_DIAGONAL.to_vec())),
            ALPHA,
            BETA,
            KAPPA,
        );

        let imu_data = IMUData {
            accel: Vector3::new(0.0, 0.0, -9.81),
            gyro: Vector3::new(0.0, 0.0, 0.0),
        };

        ukf.predict(&imu_data, 0.1);

        // Just verify prediction completed without panic
        assert_eq!(ukf.mean_state.len(), 15);
    }

    #[test]
    fn ukf_update_with_cross_covariance() {
        // Test UKF update to cover cross-covariance calculation
        let mut ukf = UnscentedKalmanFilter::new(
            UKF_PARAMS,
            IMU_BIASES.to_vec(),
            None,
            COVARIANCE_DIAGONAL.to_vec(),
            DMatrix::from_diagonal(&DVector::from_vec(PROCESS_NOISE_DIAGONAL.to_vec())),
            ALPHA,
            BETA,
            KAPPA,
        );

        // First predict to move state
        let imu_data = IMUData {
            accel: Vector3::new(0.0, 0.0, -9.81),
            gyro: Vector3::new(0.0, 0.0, 0.0),
        };
        ukf.predict(&imu_data, 0.1);

        // Update with GPS position measurement
        let measurement = GPSPositionMeasurement {
            latitude: 0.001,
            longitude: 0.001,
            altitude: 10.0,
            horizontal_noise_std: 5.0,
            vertical_noise_std: 2.0,
        };

        ukf.update(&measurement);

        // Verify update completed
        assert!(!ukf.mean_state.is_empty());
    }

    #[test]
    fn ukf_with_additional_states() {
        // Test UKF construction with additional states beyond 15
        let measurement_bias = vec![1.0, 2.0, 3.0];
        let total_states = 15 + measurement_bias.len();

        let ukf = UnscentedKalmanFilter::new(
            UKF_PARAMS,
            IMU_BIASES.to_vec(),
            Some(measurement_bias),
            vec![1e-6; total_states],
            DMatrix::from_diagonal(&DVector::from_vec(vec![1e-9; total_states])),
            ALPHA,
            BETA,
            KAPPA,
        );

        assert_eq!(ukf.state_size, total_states);
        assert_eq!(ukf.mean_state.len(), total_states);
    }

    #[test]
    fn ukf_with_velocity_measurement() {
        // Test UKF with velocity measurement
        let mut ukf = UnscentedKalmanFilter::new(
            UKF_PARAMS,
            IMU_BIASES.to_vec(),
            None,
            COVARIANCE_DIAGONAL.to_vec(),
            DMatrix::from_diagonal(&DVector::from_vec(PROCESS_NOISE_DIAGONAL.to_vec())),
            ALPHA,
            BETA,
            KAPPA,
        );

        let vel_meas = GPSVelocityMeasurement {
            northward_velocity: 0.0,
            eastward_velocity: 0.0,
            vertical_velocity: 0.0,
            horizontal_noise_std: 0.5,
            vertical_noise_std: 0.5,
        };

        ukf.update(&vel_meas);

        // Verify update completed
        assert_eq!(ukf.mean_state.len(), 15);
    }

    #[test]
    fn ukf_with_position_velocity_measurement() {
        // Test UKF with combined position and velocity measurement
        let mut ukf = UnscentedKalmanFilter::new(
            UKF_PARAMS,
            IMU_BIASES.to_vec(),
            None,
            COVARIANCE_DIAGONAL.to_vec(),
            DMatrix::from_diagonal(&DVector::from_vec(PROCESS_NOISE_DIAGONAL.to_vec())),
            ALPHA,
            BETA,
            KAPPA,
        );

        let meas = GPSPositionAndVelocityMeasurement {
            latitude: 0.0,
            longitude: 0.0,
            altitude: 0.0,
            northward_velocity: 0.0,
            eastward_velocity: 0.0,
            horizontal_noise_std: 5.0,
            vertical_noise_std: 2.0,
            velocity_noise_std: 0.5,
        };

        ukf.update(&meas);

        // Verify update completed
        assert_eq!(ukf.mean_state.len(), 15);
    }

    #[test]
    fn ukf_with_altitude_measurement() {
        // Test UKF with relative altitude measurement
        let initial_state = InitialState {
            latitude: 0.0,
            longitude: 0.0,
            altitude: 100.0,
            northward_velocity: 0.0,
            eastward_velocity: 0.0,
            vertical_velocity: 0.0,
            roll: 0.0,
            pitch: 0.0,
            yaw: 0.0,
            in_degrees: false,
            is_enu: true,
        };

        let mut ukf = UnscentedKalmanFilter::new(
            initial_state,
            IMU_BIASES.to_vec(),
            None,
            COVARIANCE_DIAGONAL.to_vec(),
            DMatrix::from_diagonal(&DVector::from_vec(PROCESS_NOISE_DIAGONAL.to_vec())),
            ALPHA,
            BETA,
            KAPPA,
        );

        let alt_meas = RelativeAltitudeMeasurement {
            relative_altitude: 5.0,
            reference_altitude: 95.0,
        };

        ukf.update(&alt_meas);

        // Should pull altitude toward 100m
        assert!(ukf.mean_state[2] > 90.0 && ukf.mean_state[2] < 110.0);
    }

    #[test]
    fn ukf_free_fall_motion() {
        // Test UKF with free fall motion profile
        let initial_state = InitialState {
            latitude: 0.0,
            longitude: 0.0,
            altitude: 100.0,
            northward_velocity: 0.0,
            eastward_velocity: 0.0,
            vertical_velocity: 0.0,
            roll: 0.0,
            pitch: 0.0,
            yaw: 0.0,
            in_degrees: false,
            is_enu: true,
        };

        let mut ukf = UnscentedKalmanFilter::new(
            initial_state,
            IMU_BIASES.to_vec(),
            None,
            vec![
                1e-6, 1e-6, 1.0, 0.1, 0.1, 0.1, 1e-4, 1e-4, 1e-4, 1e-6, 1e-6, 1e-6, 1e-8, 1e-8,
                1e-8,
            ],
            DMatrix::from_diagonal(&DVector::from_vec(vec![
                1e-9, 1e-9, 1e-6, 1e-6, 1e-6, 1e-6, 1e-9, 1e-9, 1e-9, 1e-9, 1e-9, 1e-9, 1e-9, 1e-9,
                1e-9,
            ])),
            ALPHA,
            BETA,
            KAPPA,
        );

        let dt = 0.1;
        let num_steps = 10;

        // Simulate free fall with only gravity (no vertical acceleration resistance)
        for _ in 0..num_steps {
            let imu_data = IMUData {
                accel: Vector3::new(0.0, 0.0, 0.0), // Free fall - no measured acceleration
                gyro: Vector3::new(0.0, 0.0, 0.0),
            };
            ukf.predict(&imu_data, dt);
        }

        // After 1 second of free fall, should have accumulated vertical velocity
        // v = g*t = 9.81 * 1.0 = 9.81 m/s (downward is negative in ENU)
        let final_vd = ukf.mean_state[5];
        assert!(
            final_vd < -5.0,
            "Expected significant vertical velocity, got {}",
            final_vd
        );

        // Altitude should have decreased
        let final_altitude = ukf.mean_state[2];
        assert!(
            final_altitude < 100.0,
            "Expected altitude decrease, got {}",
            final_altitude
        );

        // Apply measurement update with GPS position
        let measurement = GPSPositionMeasurement {
            latitude: 0.0,
            longitude: 0.0,
            altitude: final_altitude,
            horizontal_noise_std: 5.0,
            vertical_noise_std: 2.0,
        };
        ukf.update(&measurement);

        // After measurement update, estimate should remain close to measurement
        assert_approx_eq!(ukf.mean_state[2], final_altitude, 5.0);
    }

    #[test]
    fn ukf_hover_motion() {
        // Test UKF with hover (stationary vertical) motion profile
        let initial_state = InitialState {
            latitude: 0.0,
            longitude: 0.0,
            altitude: 100.0,
            northward_velocity: 0.0,
            eastward_velocity: 0.0,
            vertical_velocity: 0.0,
            roll: 0.0,
            pitch: 0.0,
            yaw: 0.0,
            in_degrees: false,
            is_enu: true,
        };

        let mut ukf = UnscentedKalmanFilter::new(
            initial_state,
            IMU_BIASES.to_vec(),
            None,
            vec![
                1e-6, 1e-6, 1.0, 0.1, 0.1, 0.1, 1e-4, 1e-4, 1e-4, 1e-6, 1e-6, 1e-6, 1e-8, 1e-8,
                1e-8,
            ],
            DMatrix::from_diagonal(&DVector::from_vec(vec![
                1e-9, 1e-9, 1e-6, 1e-6, 1e-6, 1e-6, 1e-9, 1e-9, 1e-9, 1e-9, 1e-9, 1e-9, 1e-9, 1e-9,
                1e-9,
            ])),
            ALPHA,
            BETA,
            KAPPA,
        );

        let dt = 0.1;
        let num_steps = 10;

        // Simulate hover with upward acceleration exactly canceling gravity
        for _ in 0..num_steps {
            let imu_data = IMUData {
                accel: Vector3::new(0.0, 0.0, earth::gravity(&0.0, &0.0)),
                gyro: Vector3::new(0.0, 0.0, 0.0),
            };
            ukf.predict(&imu_data, dt);
        }

        // Velocity should remain near zero
        let final_vn = ukf.mean_state[3];
        let final_ve = ukf.mean_state[4];
        let final_vd = ukf.mean_state[5];
        assert_approx_eq!(final_vn, 0.0, 0.5);
        assert_approx_eq!(final_ve, 0.0, 0.5);
        assert_approx_eq!(final_vd, 0.0, 0.5);

        // Altitude should remain approximately constant
        let final_altitude = ukf.mean_state[2];
        assert_approx_eq!(final_altitude, 100.0, 1.0);

        // Apply GPS velocity measurement to verify zero velocity state
        let vel_measurement = GPSVelocityMeasurement {
            northward_velocity: 0.0,
            eastward_velocity: 0.0,
            vertical_velocity: 0.0,
            horizontal_noise_std: 0.5,
            vertical_noise_std: 0.5,
        };
        ukf.update(&vel_measurement);

        // After update, velocities should remain near zero
        assert_approx_eq!(ukf.mean_state[3], 0.0, 0.5);
        assert_approx_eq!(ukf.mean_state[4], 0.0, 0.5);
        assert_approx_eq!(ukf.mean_state[5], 0.0, 0.5);
    }

    #[test]
    fn ukf_northward_motion() {
        // Test UKF with constant northward velocity motion profile
        let initial_state = InitialState {
            latitude: 0.0,
            longitude: 0.0,
            altitude: 100.0,
            northward_velocity: 10.0, // 10 m/s northward
            eastward_velocity: 0.0,
            vertical_velocity: 0.0,
            roll: 0.0,
            pitch: 0.0,
            yaw: 0.0,
            in_degrees: false,
            is_enu: true,
        };

        let mut ukf = UnscentedKalmanFilter::new(
            initial_state,
            IMU_BIASES.to_vec(),
            None,
            vec![
                1e-6, 1e-6, 1.0, 0.1, 0.1, 0.1, 1e-4, 1e-4, 1e-4, 1e-6, 1e-6, 1e-6, 1e-8, 1e-8,
                1e-8,
            ],
            DMatrix::from_diagonal(&DVector::from_vec(vec![
                1e-9, 1e-9, 1e-6, 1e-6, 1e-6, 1e-6, 1e-9, 1e-9, 1e-9, 1e-9, 1e-9, 1e-9, 1e-9, 1e-9,
                1e-9,
            ])),
            ALPHA,
            BETA,
            KAPPA,
        );

        let dt = 0.1;
        let num_steps = 10;
        let initial_lat = ukf.mean_state[0];

        // Simulate constant northward motion with gravity compensation
        for _ in 0..num_steps {
            let imu_data = IMUData {
                accel: Vector3::new(0.0, 0.0, earth::gravity(&0.0, &0.0)),
                gyro: Vector3::new(0.0, 0.0, 0.0),
            };
            ukf.predict(&imu_data, dt);
        }

        // Latitude should have increased (moving north)
        let final_lat = ukf.mean_state[0];
        assert!(
            final_lat > initial_lat,
            "Expected latitude increase, got initial: {} final: {}",
            initial_lat,
            final_lat
        );

        // Northward velocity should remain approximately constant
        let final_vn = ukf.mean_state[3];
        assert_approx_eq!(final_vn, 10.0, 2.0);

        // Eastward velocity should remain near zero
        let final_ve = ukf.mean_state[4];
        assert_approx_eq!(final_ve, 0.0, 0.5);

        // Apply GPS position and velocity measurement
        let meas = GPSPositionAndVelocityMeasurement {
            latitude: final_lat.to_degrees(),
            longitude: 0.0,
            altitude: 100.0,
            northward_velocity: 10.0,
            eastward_velocity: 0.0,
            horizontal_noise_std: 5.0,
            vertical_noise_std: 2.0,
            velocity_noise_std: 0.5,
        };
        ukf.update(&meas);

        // After measurement, velocities should be close to measured values
        assert_approx_eq!(ukf.mean_state[3], 10.0, 1.0);
        assert_approx_eq!(ukf.mean_state[4], 0.0, 0.5);
    }

    #[test]
    fn ukf_eastward_motion() {
        // Test UKF with constant eastward velocity motion profile
        let initial_state = InitialState {
            latitude: 0.0,
            longitude: 0.0,
            altitude: 100.0,
            northward_velocity: 0.0,
            eastward_velocity: 15.0, // 15 m/s eastward
            vertical_velocity: 0.0,
            roll: 0.0,
            pitch: 0.0,
            yaw: 0.0,
            in_degrees: false,
            is_enu: true,
        };

        let mut ukf = UnscentedKalmanFilter::new(
            initial_state,
            IMU_BIASES.to_vec(),
            None,
            vec![
                1e-6, 1e-6, 1.0, 0.1, 0.1, 0.1, 1e-4, 1e-4, 1e-4, 1e-6, 1e-6, 1e-6, 1e-8, 1e-8,
                1e-8,
            ],
            DMatrix::from_diagonal(&DVector::from_vec(vec![
                1e-9, 1e-9, 1e-6, 1e-6, 1e-6, 1e-6, 1e-9, 1e-9, 1e-9, 1e-9, 1e-9, 1e-9, 1e-9, 1e-9,
                1e-9,
            ])),
            ALPHA,
            BETA,
            KAPPA,
        );

        let dt = 0.1;
        let num_steps = 10;
        let initial_lon = ukf.mean_state[1];

        // Simulate constant eastward motion with gravity compensation
        for _ in 0..num_steps {
            let imu_data = IMUData {
                accel: Vector3::new(0.0, 0.0, earth::gravity(&0.0, &0.0)),
                gyro: Vector3::new(0.0, 0.0, 0.0),
            };
            ukf.predict(&imu_data, dt);
        }

        // Longitude should have increased (moving east)
        let final_lon = ukf.mean_state[1];
        assert!(
            final_lon > initial_lon,
            "Expected longitude increase, got initial: {} final: {}",
            initial_lon,
            final_lon
        );

        // Eastward velocity should remain approximately constant
        let final_ve = ukf.mean_state[4];
        assert_approx_eq!(final_ve, 15.0, 2.0);

        // Northward velocity should remain near zero
        let final_vn = ukf.mean_state[3];
        assert_approx_eq!(final_vn, 0.0, 0.5);

        // Vertical velocity should remain near zero
        let final_vd = ukf.mean_state[5];
        assert_approx_eq!(final_vd, 0.0, 0.5);

        // Apply GPS position measurement
        let pos_meas = GPSPositionMeasurement {
            latitude: 0.0,
            longitude: final_lon.to_degrees(),
            altitude: 100.0,
            horizontal_noise_std: 5.0,
            vertical_noise_std: 2.0,
        };
        ukf.update(&pos_meas);

        // Position should remain close to measurement
        assert_approx_eq!(ukf.mean_state[1], final_lon, 0.01);
        assert_approx_eq!(ukf.mean_state[2], 100.0, 5.0);

        // Apply velocity measurement
        let vel_meas = GPSVelocityMeasurement {
            northward_velocity: 0.0,
            eastward_velocity: 15.0,
            vertical_velocity: 0.0,
            horizontal_noise_std: 0.5,
            vertical_noise_std: 0.5,
        };
        ukf.update(&vel_meas);

        // After measurement, velocities should be close to measured values
        assert_approx_eq!(ukf.mean_state[3], 0.0, 0.5);
        assert_approx_eq!(ukf.mean_state[4], 15.0, 1.0);
        assert_approx_eq!(ukf.mean_state[5], 0.0, 0.5);
    }

    #[test]
    fn ukf_combined_horizontal_motion() {
        // Test UKF with combined northward and eastward motion
        let initial_state = InitialState {
            latitude: 0.0,
            longitude: 0.0,
            altitude: 100.0,
            northward_velocity: 10.0,
            eastward_velocity: 10.0,
            vertical_velocity: 0.0,
            roll: 0.0,
            pitch: 0.0,
            yaw: 0.0,
            in_degrees: false,
            is_enu: true,
        };

        let mut ukf = UnscentedKalmanFilter::new(
            initial_state,
            IMU_BIASES.to_vec(),
            None,
            vec![
                1e-6, 1e-6, 1.0, 0.1, 0.1, 0.1, 1e-4, 1e-4, 1e-4, 1e-6, 1e-6, 1e-6, 1e-8, 1e-8,
                1e-8,
            ],
            DMatrix::from_diagonal(&DVector::from_vec(vec![
                1e-9, 1e-9, 1e-6, 1e-6, 1e-6, 1e-6, 1e-9, 1e-9, 1e-9, 1e-9, 1e-9, 1e-9, 1e-9, 1e-9,
                1e-9,
            ])),
            ALPHA,
            BETA,
            KAPPA,
        );

        let dt = 0.1;
        let num_steps = 10;
        let initial_lat = ukf.mean_state[0];
        let initial_lon = ukf.mean_state[1];

        // Simulate combined motion
        for _ in 0..num_steps {
            let imu_data = IMUData {
                accel: Vector3::new(0.0, 0.0, earth::gravity(&0.0, &0.0)),
                gyro: Vector3::new(0.0, 0.0, 0.0),
            };
            ukf.predict(&imu_data, dt);
        }

        // Both latitude and longitude should have increased
        let final_lat = ukf.mean_state[0];
        let final_lon = ukf.mean_state[1];
        assert!(final_lat > initial_lat, "Expected latitude increase");
        assert!(final_lon > initial_lon, "Expected longitude increase");

        // Both velocities should remain approximately constant
        assert_approx_eq!(ukf.mean_state[3], 10.0, 2.0);
        assert_approx_eq!(ukf.mean_state[4], 10.0, 2.0);

        // Apply combined position and velocity measurement
        let meas = GPSPositionAndVelocityMeasurement {
            latitude: final_lat.to_degrees(),
            longitude: final_lon.to_degrees(),
            altitude: 100.0,
            northward_velocity: 10.0,
            eastward_velocity: 10.0,
            horizontal_noise_std: 5.0,
            vertical_noise_std: 2.0,
            velocity_noise_std: 0.5,
        };
        ukf.update(&meas);

        // After measurement, state should be well-constrained
        assert_approx_eq!(ukf.mean_state[3], 10.0, 1.0);
        assert_approx_eq!(ukf.mean_state[4], 10.0, 1.0);
        assert_approx_eq!(ukf.mean_state[2], 100.0, 5.0);
    }
}
