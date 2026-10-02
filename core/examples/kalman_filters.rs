//! The three Kalman-family filters, built two ways and driven through one trait.
//!
//! Run it with:
//!
//! ```bash
//! cargo run -p strapdown-core --example kalman_filters
//! ```
//!
//! [`strapdown::NavigationFilter`] is object-safe, so the error-state, extended and unscented
//! filters can sit in one `Vec<Box<dyn NavigationFilter>>` and be driven by one loop. This
//! example builds each of them twice:
//!
//! 1. through the `sim::initialize_*` helpers, which is what `strapdown-sim cl` does; and
//! 2. through the filters' own constructors, with an initial covariance derived from an IMU
//!    grade by `IMUQuality::auto_covariance` and the crate's default process-noise density.
//!
//! All six then run over the same two-minute synthetic drive with a GNSS position fix once a
//! second, and the example prints each filter's final horizontal error against the exact
//! trajectory and the last
//! [`UpdateOutcome`](strapdown::UpdateOutcome) it returned. The point is the shape of the API,
//! not a ranking: one short run of one trajectory says nothing general about which filter is
//! more accurate.

use std::error::Error;

use nalgebra::{DMatrix, DVector, Vector3};
use rand::SeedableRng;
use rand::rngs::StdRng;

use strapdown::earth::haversine_distance;
use strapdown::kalman::{ErrorStateKalmanFilter, ExtendedKalmanFilter, UnscentedKalmanFilter};
use strapdown::measurements::GPSPositionMeasurement;
use strapdown::sim::{
    DEFAULT_PROCESS_NOISE_DENSITY, DEFAULT_UKF_ALPHA, EkfConfig, EskfConfig, NavigationResult,
    SyntheticConfig, SyntheticInitialState, TestDataRecord, UkfConfig, generate_synthetic,
    initialize_ekf, initialize_eskf, initialize_ukf,
};
use strapdown::{IMUData, IMUQuality, ImuSample, InitialUncertainty, NavigationFilter};

/// How often a GNSS fix is applied, in records. The synthetic log is 10 Hz.
const GNSS_EVERY: usize = 10;

/// A two-minute synthetic drive due north at 15 m/s, NED, consumer-grade IMU: the exact
/// trajectory and the noisy sensor log derived from it.
fn synthetic_drive() -> Result<(Vec<NavigationResult>, Vec<TestDataRecord>), Box<dyn Error>> {
    // ANCHOR: synthetic
    let mut config = SyntheticConfig::default();
    config.initial_state = SyntheticInitialState {
        latitude_deg: 39.95,
        longitude_deg: -75.16,
        altitude_m: 50.0,
        velocity_north_mps: 15.0,
        velocity_east_mps: 0.0,
        velocity_down_mps: 0.0,
        roll_deg: 0.0,
        pitch_deg: 0.0,
        yaw_deg: 0.0,
        angular_velocity_x_dps: 0.0,
        angular_velocity_y_dps: 0.0,
        angular_velocity_z_dps: 0.0,
        is_enu: false,
    };
    config.duration_s = 120.0;
    config.sample_rate_hz = 10.0;
    config.imu_quality = IMUQuality::Consumer;
    config.seed = 42;

    // Returns the exact trajectory and the noisy sensor log derived from it.
    let mut rng = StdRng::seed_from_u64(config.seed);
    let (truth, records) = generate_synthetic(&config, &mut rng)?;
    // ANCHOR_END: synthetic
    Ok((truth, records))
}

fn main() -> Result<(), Box<dyn Error>> {
    let (truth, records) = synthetic_drive()?;
    let first = &records[0];

    // ANCHOR: helpers
    // The route `strapdown-sim cl` takes: seed everything from the first record. The configs
    // are `#[non_exhaustive]`, so start from `default()` and assign; the defaults are NED.
    let eskf = initialize_eskf(first, EskfConfig::default())?;
    let ekf = initialize_ekf(first, EkfConfig::default())?;
    let ukf = initialize_ukf(first, UkfConfig::default())?;
    // ANCHOR_END: helpers

    // ANCHOR: initial_covariance
    // The same seed state the helpers use: radians, NED (`false`).
    let initial_state = first.initial_state(false);

    // P0 from an IMU grade and how well the first fix placed the vehicle. The fifteen entries
    // come back in each state's own units: rad^2 for latitude and longitude, m^2 for
    // altitude, (m/s)^2, rad^2, (m/s^2)^2 and (rad/s)^2 -- never one literal for all of them.
    let uncertainty = InitialUncertainty::new(
        first.horizontal_accuracy, // horizontal, m (1-sigma)
        first.vertical_accuracy,   // vertical, m (1-sigma)
        0.5,                       // velocity, m/s (1-sigma)
    );
    let initial_covariance =
        IMUQuality::Consumer.auto_covariance(uncertainty, first.latitude, first.altitude)?;

    // Q as a spectral density: a variance per second on each state. Each filter forms
    // Q_k = q * dt itself, so the same matrix is right at 1 Hz and at 100 Hz.
    let process_noise_density =
        DMatrix::from_diagonal(&DVector::from_row_slice(&DEFAULT_PROCESS_NOISE_DENSITY));
    // ANCHOR_END: initial_covariance

    // ANCHOR: constructors
    let no_bias = [0.0; 6];
    // ANCHOR: eskf_new
    let eskf_direct = ErrorStateKalmanFilter::new(
        &initial_state,
        &no_bias,
        initial_covariance.to_vec(),
        process_noise_density.clone(),
    );
    // ANCHOR_END: eskf_new
    // ANCHOR: ekf_new
    let ekf_direct = ExtendedKalmanFilter::new(
        &initial_state,
        &no_bias,
        initial_covariance.to_vec(),
        process_noise_density.clone(),
        true, // carry the six IMU-bias states: 15 states, not 9
    );
    // ANCHOR_END: ekf_new
    // ANCHOR: ukf_new
    let ukf_direct = UnscentedKalmanFilter::new(
        &initial_state,
        &no_bias,
        None, // no extra states beyond the fifteen
        initial_covariance.to_vec(),
        process_noise_density,
        DEFAULT_UKF_ALPHA, // 0.1
        2.0,               // beta: optimal for a Gaussian prior
        0.0,               // kappa
    );
    // ANCHOR_END: ukf_new
    // ANCHOR_END: constructors

    // ANCHOR: trait_objects
    let mut filters: Vec<(&str, Box<dyn NavigationFilter>)> = vec![
        ("ESKF, initialize_eskf", Box::new(eskf)),
        ("EKF,  initialize_ekf", Box::new(ekf)),
        ("UKF,  initialize_ukf", Box::new(ukf)),
        ("ESKF, constructor", Box::new(eskf_direct)),
        ("EKF,  constructor", Box::new(ekf_direct)),
        ("UKF,  constructor", Box::new(ukf_direct)),
    ];
    // ANCHOR_END: trait_objects

    let final_truth = &truth[truth.len() - 1];
    for (name, filter) in &mut filters {
        // ANCHOR: loop
        let mut outcome = None;
        for (index, pair) in records.windows(2).enumerate() {
            let (previous, record) = (&pair[0], &pair[1]);
            let dt = (record.time - previous.time).num_milliseconds() as f64 / 1000.0;

            // Rates in, increments mechanized: `from_rates` integrates over `dt`.
            let imu = IMUData {
                accel: Vector3::new(record.acc_x, record.acc_y, record.acc_z),
                gyro: Vector3::new(record.gyro_x, record.gyro_y, record.gyro_z),
            };
            filter.predict(&ImuSample::from_rates(&imu, dt), dt)?;

            if (index + 1) % GNSS_EVERY == 0 {
                let fix = GPSPositionMeasurement {
                    latitude: record.latitude, // degrees
                    longitude: record.longitude,
                    altitude: record.altitude,
                    horizontal_noise_std: record.horizontal_accuracy, // metres, 1-sigma
                    vertical_noise_std: record.vertical_accuracy,
                };
                // `Ok` carries the NIS and whether the correction was applied.
                outcome = Some(filter.update(&fix)?);
            }
        }
        // ANCHOR_END: loop

        // The estimate is in the filter's native units: radians for latitude and longitude.
        let estimate = filter.get_estimate();
        let error_m = haversine_distance(
            estimate[0],
            estimate[1],
            final_truth.latitude.to_radians(),
            final_truth.longitude.to_radians(),
        );
        if let Some(outcome) = outcome {
            println!(
                "{name}: {} states, {error_m:.3} m from truth at the end, last update NIS {:.3} \
                 ({} dof, accepted: {})",
                estimate.len(),
                outcome.nis,
                outcome.dof,
                outcome.accepted
            );
        }
    }

    Ok(())
}
