//! From raw IMU samples to an aided solution: calibration, coarse alignment, a P0 from the
//! IMU grade, and an antenna lever arm.
//!
//! Run it with:
//!
//! ```bash
//! cargo run -p strapdown-core --example lever_arm
//! ```
//!
//! A tactical-grade IMU sits still for 30 s, tilted and pointing 60 degrees east of north,
//! with a GNSS antenna 1.5 m ahead of it and 0.8 m above it. The example
//!
//! 1. removes a known accelerometer bias and scale-factor error with [`ImuCalibration`];
//! 2. levels and gyrocompasses from the stationary window, and shows the same window being
//!    refused for a consumer-grade gyro;
//! 3. builds two engines seeded from that alignment with a P0 from
//!    [`IMUQuality::auto_covariance`], one told about the lever arm and one not, and feeds both
//!    the same antenna-referred fixes;
//! 4. shows a malformed fix being reported as a *recoverable* error and skipped.
//!
//! The scenario is synthetic and noise-free so that the numbers can be checked by hand.

use std::error::Error;

use nalgebra::{Rotation3, Vector3};

use strapdown::alignment::{
    GyrocompassConfig, attitude_from_level_and_heading, average_imu, coarse_leveling,
    gyrocompassing,
};
use strapdown::calibration::{ImuCalibration, SensorCalibration};
use strapdown::earth::{self, METERS_TO_DEGREES, haversine_distance};
use strapdown::engine::{GnssFix, InsEngine};
use strapdown::kalman::InitialState;
use strapdown::stationary::{StationaryConfig, StationaryDetector};
use strapdown::{IMUData, IMUQuality, ImuSample, InitialUncertainty, StrapdownError};

const LATITUDE_DEG: f64 = 39.95;
const LONGITUDE_DEG: f64 = -75.16;
const ALTITUDE_M: f64 = 12.0;
/// IMU sample interval, seconds.
const DT: f64 = 0.01;
/// Antenna phase centre relative to the IMU, body frame (x forward, y right, z down), metres.
const LEVER_ARM_M: [f64; 3] = [1.5, 0.0, -0.8];

fn main() -> Result<(), Box<dyn Error>> {
    // Truth: tilted 2 degrees right-wing-down and 1 degree nose-down, heading 060.
    let truth_attitude = Rotation3::from_euler_angles(
        2.0_f64.to_radians(),
        -1.0_f64.to_radians(),
        60.0_f64.to_radians(),
    );
    // What a perfect stationary IMU reads in NED: the reaction to gravity, and Earth rate.
    let perfect = IMUData {
        accel: truth_attitude.inverse()
            * Vector3::new(0.0, 0.0, -earth::gravity(&LATITUDE_DEG, &ALTITUDE_M)),
        gyro: truth_attitude.inverse() * earth::earth_rate_lla(&LATITUDE_DEG),
    };

    // ANCHOR: calibration
    // The accelerometer's forward error model, as a calibration report would quote it:
    // 0.05 m/s^2 of bias on x and a 1% scale-factor error on y. The gyro is taken as perfect.
    let accelerometer = SensorCalibration::new(
        [0.05, 0.0, 0.0], // bias, m/s^2
        [0.0, 0.01, 0.0], // scale-factor error, fractional
        [[0.0; 3]; 3],    // misalignment (off-diagonal only)
    )?;
    let calibration = ImuCalibration::new(accelerometer, SensorCalibration::identity());
    // ANCHOR_END: calibration

    // What the sensor actually reports: b + (I + M) f.
    let raw = IMUData {
        accel: Vector3::new(
            perfect.accel.x + 0.05,
            perfect.accel.y * 1.01,
            perfect.accel.z,
        ),
        gyro: perfect.gyro,
    };

    // ANCHOR: alignment
    // Keep only the samples the detector vouches for, corrected before anything uses them.
    let mut detector = StationaryDetector::new(StationaryConfig::default());
    let mut window = Vec::new();
    for _ in 0..3000 {
        let corrected = calibration.correct_rates(&raw);
        if detector.push(&corrected) {
            window.push(corrected);
        }
    }
    let averaged = average_imu(&window)?;
    let level = coarse_leveling(&averaged.accel)?;

    // A consumer gyro's bias is several times Earth rate, so it cannot gyrocompass, and the
    // estimator says so instead of returning a heading.
    if let Err(refusal) = gyrocompassing(
        &averaged,
        LATITUDE_DEG,
        IMUQuality::Consumer,
        &GyrocompassConfig::default(),
    ) {
        println!("consumer grade: {refusal}");
    }
    let heading = gyrocompassing(
        &averaged,
        LATITUDE_DEG,
        IMUQuality::Tactical,
        &GyrocompassConfig::default(),
    )?;
    let attitude = attitude_from_level_and_heading(level, heading.heading_radians);
    // ANCHOR_END: alignment

    let (roll, pitch, yaw) = attitude.euler_angles();
    println!(
        "aligned from {} stationary samples: roll {:.3}, pitch {:.3}, heading {:.3} deg \
         (claimed 1-sigma {:.1} deg)",
        window.len(),
        roll.to_degrees(),
        pitch.to_degrees(),
        yaw.to_degrees(),
        heading.uncertainty_radians.to_degrees()
    );

    // ANCHOR: engine
    let initial_state = InitialState::new(
        LATITUDE_DEG.to_radians(),
        LONGITUDE_DEG.to_radians(),
        ALTITUDE_M,
        0.0,
        0.0,
        0.0,
        roll,
        pitch,
        yaw,
        false, // everything above is in radians
        None,  // NED
    );
    // P0 from the grade and the first fix, then widened on yaw by the heading uncertainty the
    // gyrocompass reported -- `auto_covariance` treats heading like roll and pitch.
    let mut initial_covariance = IMUQuality::Tactical.auto_covariance(
        InitialUncertainty::new(3.0, 5.0, 0.1),
        LATITUDE_DEG,
        ALTITUDE_M,
    )?;
    initial_covariance[8] += heading.uncertainty_radians.powi(2);

    let mut engine = InsEngine::builder()
        .with_initial_state(initial_state.clone())
        .with_initial_covariance(initial_covariance.to_vec())
        .with_lever_arm(LEVER_ARM_M)
        .build()?;
    // ANCHOR_END: engine
    let mut uncompensated = InsEngine::builder()
        .with_initial_state(initial_state)
        .with_initial_covariance(initial_covariance.to_vec())
        .build()?;

    // Where the antenna really is: the IMU position plus the lever arm rotated into NED.
    let offset = truth_attitude * Vector3::from(LEVER_ARM_M);
    let antenna = GnssFix::position(
        LATITUDE_DEG + offset.x * METERS_TO_DEGREES,
        LONGITUDE_DEG + offset.y * METERS_TO_DEGREES / LATITUDE_DEG.to_radians().cos(),
        ALTITUDE_M - offset.z, // altitude is positive up; the NED offset is positive down
        3.0,
        5.0,
    );

    // ANCHOR: run
    let sample = ImuSample::from_rates(&raw, DT);
    for step in 1..=6000 {
        let corrected = calibration.correct(&sample);
        engine.predict(&corrected)?;
        uncompensated.predict(&corrected)?;
        if step % 100 == 0 {
            engine.update_gnss(&antenna)?;
            uncompensated.update_gnss(&antenna)?;
        }
    }
    // ANCHOR_END: run

    for (name, solution) in [
        ("with lever arm   ", engine.nav_solution()),
        ("without lever arm", uncompensated.nav_solution()),
    ] {
        let horizontal_m = haversine_distance(
            solution.latitude.to_radians(),
            solution.longitude.to_radians(),
            LATITUDE_DEG.to_radians(),
            LONGITUDE_DEG.to_radians(),
        );
        println!(
            "{name}: IMU position off by {horizontal_m:.2} m horizontally, {:.2} m vertically",
            solution.altitude - ALTITUDE_M
        );
    }

    // ANCHOR: recoverable
    // A fix with a NaN in it is reported, not applied. `is_recoverable` says whether the
    // filter is still usable -- skip the measurement and carry on -- or the run must stop.
    let broken = GnssFix::position(f64::NAN, LONGITUDE_DEG, ALTITUDE_M, 3.0, 5.0);
    match engine.update_gnss(&broken) {
        Ok(outcome) => println!("applied, NIS {:.2}", outcome.nis),
        Err(error) if error.is_recoverable() => println!("skipped: {error}"),
        Err(error) => return Err(error.into()),
    }
    // ANCHOR_END: recoverable

    // The offset can change on a running engine, and is validated like the builder's.
    let refused: Result<(), StrapdownError> = engine.set_lever_arm([150.0, 0.0, 0.0]);
    if let Err(error) = refused {
        println!("set_lever_arm: {error}");
    }

    Ok(())
}
