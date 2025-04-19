//! Simulation module for running navigation filters as an INS
//!
//! This module provides basic functionality for running a navigation filter (e.g., UKF) as an INS
//! with a generic normally distributed measurement function. The measurement function computes the
//! difference between a simulated measurement (e.g., GPS) and the current filter state.

use crate::filter::UKF;
use crate::{IMUData, StrapdownState};
use nalgebra::{DVector, DMatrix};
use rand_distr::{Normal, Distribution};

/// Simulate a single INS step with the UKF filter
///
/// # Arguments
/// * `ukf` - The UKF filter instance (mutable)
/// * `imu_data` - The IMU data for propagation
/// * `dt` - Time step
/// * `measurement` - The simulated measurement vector (e.g., GPS position)
/// * `measurement_noise_std` - Standard deviation for measurement noise (applied to each measurement dimension)
/// * `measurement_model` - Function that computes the expected measurement from the filter state
/// * `measurement_jacobian` - Function that computes the measurement Jacobian at the current state
pub fn ukf_ins_step<F, G>(
    ukf: &mut UKF,
    imu_data: &IMUData,
    dt: f64,
    measurement: &DVector<f64>,
    measurement_noise_std: f64,
    measurement_model: F,
    measurement_jacobian: G,
) where
    F: Fn(&DVector<f64>) -> DVector<f64>,
    G: Fn(&DVector<f64>) -> DMatrix<f64>,
{
    // Propagate the filter
    ukf.propagate(imu_data, dt);

    // Simulate measurement noise
    let mut noisy_measurement = measurement.clone();
    let normal = Normal::new(0.0, measurement_noise_std).unwrap();
    for i in 0..noisy_measurement.len() {
        noisy_measurement[i] += normal.sample(&mut rand::thread_rng());
    }

    // Compute expected measurement and innovation
    let mean_state = ukf.get_mean();
    let expected_measurement = measurement_model(&mean_state);
    let innovation = &noisy_measurement - &expected_measurement;

    // Compute measurement sigma points (for UKF update)
    // For simplicity, use the mean state as the only sigma point (not a true UKF update, but placeholder)
    let measurement_sigma_points = vec![expected_measurement.clone(); 2 * mean_state.len() + 1];
    let innovation_matrix = DMatrix::identity(noisy_measurement.len(), noisy_measurement.len()) * measurement_noise_std.powi(2);

    // Update the filter
    ukf.update(noisy_measurement, measurement_sigma_points, innovation_matrix);
}
