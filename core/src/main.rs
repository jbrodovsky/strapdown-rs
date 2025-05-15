use nalgebra::{Vector3, DVector};

use strapdown::earth;
use strapdown::{StrapdownState, IMUData};
use strapdown::filter::{UKF, self};

fn main() {
    let imu_data = IMUData::new_from_vec(
        vec![0.0, 0.0, -earth::gravity(&0.0, &0.0)], 
        vec![0.0, 0.0, 0.0]
    );
    let position = vec![0.0, 0.0, 0.0];
    let velocity = vec![0.0, 0.0, 0.0];
    let attitude = vec![0.0, 0.0, 0.0];
    let imu_biases = vec![0.0, 0.0, 0.0];
    let measurement_bias = vec![0.0, 0.0, 0.0];
    let covariance_diagonal = vec![1e-12; 9 + imu_biases.len() + measurement_bias.len()];
    let process_noise_diagonal = vec![1e-12; 9 + imu_biases.len() + measurement_bias.len()];
    let measurement_noise_diagonal = vec![1e-12; 3];
    let alpha = 1e-3;
    let beta = 2.0;
    let kappa = 1e-3;
    let mut ukf = UKF::new(
        position.clone(),
        velocity.clone(),
        attitude.clone(),
        imu_biases.clone(),
        measurement_bias.clone(),
        covariance_diagonal,
        process_noise_diagonal,
        measurement_noise_diagonal,
        alpha,
        beta,
        kappa,
    );
    let dt = 1.0;
    let measurement_sigmas = filter::ukf_position_measurement_model(&ukf.get_sigma_points());
    let measurement = DVector::from_vec(position.clone());
    for i in 0..60 {
        ukf.propagate(&imu_data, dt);
        ukf.update(&measurement, &measurement_sigmas);
        let mean = ukf.get_mean();
        println!("Iteration: {}", i);
        println!(
            "Position: [{:.2}, {:.2}, {:.2}]",
            mean[0], mean[1], mean[2]
        );
        println!(
            "Velocity: [{:.2}, {:.2}, {:.2}]",
            mean[3], mean[4], mean[5]
        );
        let covariance_diagonal = ukf.get_covariance().diagonal();
        println!("Covariance: {:.2?}", &covariance_diagonal.as_slice()[0..6]);
    }
}
