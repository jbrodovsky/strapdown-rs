//! Kalman-style navigation filters (UKF/EKF)
//!
//! This module contains the traditional Kalman filter style implementation of strapdown
//! inertial navigation systems. These filter build on the dead-reckoning functions
//! provided in the top-level [lib] module.

use crate::linalg::{matrix_square_root, robust_spd_solve, symmetrize};
use crate::measurements::MeasurementModel;
use crate::{IMUData, StrapdownState, forward, wrap_to_2pi, wrap_to_180, wrap_to_360};

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
    pub downward_velocity: f64,
    pub roll: f64,
    pub pitch: f64,
    pub yaw: f64,
    pub in_degrees: bool,
    pub is_enu: bool,
}
impl InitialState {
    pub fn new(
        latitude: f64,
        longitude: f64,
        altitude: f64,
        northward_velocity: f64,
        eastward_velocity: f64,
        downward_velocity: f64,
        mut roll: f64,
        mut pitch: f64,
        mut yaw: f64,
        in_degrees: bool,
        is_enu: Option<bool>,
    ) -> Self {
        let latitude = if in_degrees { latitude } else { latitude };
        let longitude = if in_degrees {
            wrap_to_180(longitude)
        } else {
            longitude
        };
        let is_enu = is_enu.unwrap_or(true);
        let altitude = altitude;
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
            downward_velocity,
            roll,
            pitch,
            yaw,
            in_degrees,
            is_enu,
        }
    }
}

/// Generic Navigation (Kalman?) filter trait
pub trait NavigationFilter {
    fn predict(&mut self, imu_data: IMUData, dt: f64);
    fn update<M: MeasurementModel + ?Sized>(&mut self, measurement: &M);
    fn get_estimate(&self) -> DVector<f64>;
    fn get_certainty(&self) -> DMatrix<f64>;
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
                initial_state.downward_velocity,
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
                initial_state.downward_velocity,
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
    fn predict(&mut self, imu_data: IMUData, dt: f64) {
        let mut sigma_points = self.get_sigma_points();
        for i in 0..sigma_points.ncols() {
            let mut sigma_point_vec = sigma_points.column(i).clone_owned();
            let mut state = StrapdownState {
                latitude: sigma_point_vec[0],
                longitude: sigma_point_vec[1],
                altitude: sigma_point_vec[2],
                velocity_north: sigma_point_vec[3],
                velocity_east: sigma_point_vec[4],
                velocity_down: sigma_point_vec[5],
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
                accel: imu_data.accel - &accel_biases,
                gyro: imu_data.gyro - &gyro_biases,
            };
            forward(&mut state, imu_data, dt);
            sigma_point_vec[0] = state.latitude;
            sigma_point_vec[1] = state.longitude;
            sigma_point_vec[2] = state.altitude;
            sigma_point_vec[3] = state.velocity_north;
            sigma_point_vec[4] = state.velocity_east;
            sigma_point_vec[5] = state.velocity_down;
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
        self.covariance = 0.5 * (&self.covariance + self.covariance.transpose());
    }
    fn get_estimate(&self) -> DVector<f64> {
        self.mean_state.clone()
    }
    fn get_certainty(&self) -> DMatrix<f64> {
        self.covariance.clone()
    }
}
