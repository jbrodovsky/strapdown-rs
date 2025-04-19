use nalgebra::Vector3;

use strapdown::earth;
use strapdown::{StrapdownState, IMUData};

fn main() {
    let mut state: StrapdownState = StrapdownState::new_from(
        Vector3::new(0.0, 0.0, 0.0),
        Vector3::new(0.0, 0.0, 0.0),
        Vector3::new(0.0, 0.0, 0.0),
    );
    let imu_data = IMUData {
        accel: Vector3::new(0.0, 0.0, -earth::gravity(&0.0,&0.0)), // Currently configured as relative body-frame acceleration
        gyro: Vector3::new(0.0, 0.0, 0.0),
    };
    println!("Initial state: {:?}", state);
    println!("Initial IMU data: {:?}", imu_data);
    // Simulate a time step
    let dt = 1.0; // 1s time step
    // Update the strapdown state with the IMU data
    state.forward(&imu_data, dt);
    // Print the updated state
    println!("Updated state: {:?}", state);
    // Update the strapdown state with the IMU data
    state.forward(&imu_data, 1.0);
    // Print the updated state
    println!("Updated state: {:?}", state);
    // Update the strapdown state with the IMU data
    state.forward(&imu_data, 1.0);
    // Print the updated state
    println!("Updated state: {:?}", state);
    // Update the strapdown state with the IMU data
    state.forward(&imu_data, 1.0);
    // Print the updated state
    println!("Updated state: {:?}", state);
}
