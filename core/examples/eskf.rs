//! The default filter end to end: build an ESKF, run it through a GNSS outage, score it.
//!
//! Run it with:
//!
//! ```bash
//! cargo run -p strapdown-core --example eskf
//! ```
//!
//! This is the library route to what `strapdown-sim cl` does, minus the file I/O:
//!
//! 1. generate a five-minute synthetic drive and its exact truth trajectory;
//! 2. build the 15-state error-state Kalman filter with `initialize_eskf`, plus the
//!    barometric-bias state `strapdown-sim` turns on by default;
//! 3. turn the sensor log into an event stream with a 60 s GNSS outage in the middle;
//! 4. run the closed loop and score the result against truth with `metrics::evaluate`.
//!
//! It prints the filter's own horizontal uncertainty beside its actual error at the end of the
//! outage, and the accelerometer and gyroscope bias estimates it finishes with.

use std::error::Error;

use rand::SeedableRng;
use rand::rngs::StdRng;

use strapdown::earth::{haversine_distance, principal_radii};
use strapdown::messages::{AidingConfig, GnssFaultModel, MeasurementScheduler, build_event_stream};
use strapdown::metrics::{MetricOptions, evaluate, truth_from_trajectory};
use strapdown::sim::{
    EskfConfig, NavigationResult, SyntheticConfig, SyntheticInitialState, generate_synthetic,
    initialize_eskf, run_closed_loop,
};
use strapdown::{IMUQuality, NavigationFilter};

/// GNSS is available for this long, then denied for `OUTAGE_S`, then available again.
const AIDED_S: f64 = 120.0;
/// Length of the GNSS outage, seconds.
const OUTAGE_S: f64 = 60.0;

fn main() -> Result<(), Box<dyn Error>> {
    let mut synthetic = SyntheticConfig::default();
    synthetic.initial_state = SyntheticInitialState {
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
    synthetic.duration_s = 300.0;
    synthetic.sample_rate_hz = 10.0;
    synthetic.imu_quality = IMUQuality::Consumer;
    synthetic.seed = 7;
    let mut rng = StdRng::seed_from_u64(synthetic.seed);
    let (truth, records) = generate_synthetic(&synthetic, &mut rng)?;

    // ANCHOR: build
    // `EskfConfig` is `#[non_exhaustive]`: start from the default (NED, 15 error states, bias
    // priors from the consumer IMU grade) and assign what differs.
    let mut config = EskfConfig::default();
    config.imu_quality = IMUQuality::Consumer;
    config.estimate_baro_bias = true; // a 16th error state, as `strapdown-sim cl` uses
    let baro_bias_index = config.baro_bias_index(); // where that state landed: Some(15)
    let mut eskf = initialize_eskf(&records[0], config)?;
    // ANCHOR_END: build

    // ANCHOR: stream
    // GNSS on for 120 s, off for 60 s, on again. A duty cycle runs `start_phase_s` of ON,
    // then `off_s` OFF and `on_s` ON repeating -- so with a zero phase it starts OFF. The
    // barometer and magnetometer keep their own 1 Hz schedules, so the outage removes
    // position and velocity aiding only.
    let mut aiding = AidingConfig::default();
    aiding.scheduler = MeasurementScheduler::DutyCycle {
        on_s: synthetic.duration_s, // longer than what is left: one outage, not a cycle
        off_s: OUTAGE_S,
        start_phase_s: AIDED_S,
    };
    aiding.fault = GnssFaultModel::None;
    // The barometer model has to be told which state is its bias; the filter carries it but
    // does not observe it unless the measurement names it.
    aiding.baro_bias_index = baro_bias_index;
    let stream = build_event_stream(&records, &aiding, false)?; // false = NED
    // ANCHOR_END: stream

    // ANCHOR: run
    // No health or wall-clock limits: `None, None`. `strapdown-sim` passes its defaults here.
    let results = run_closed_loop(&mut eskf, stream, None, None)?;
    // ANCHOR_END: run

    // ANCHOR: score
    let metrics = evaluate(
        &results,
        &truth_from_trajectory(&truth),
        MetricOptions::default(),
    )?;
    println!("scored {} samples against truth", metrics.sample_count);
    if let (Some(rmse), Some(max)) = (metrics.horizontal_rmse_m, metrics.horizontal_max_m) {
        println!("horizontal error: RMSE {rmse:.2} m, max {max:.2} m");
    }
    // ANCHOR_END: score

    // The filter's claim against its error, at the last sample of the outage.
    // The last sample before GNSS returns: the fix at exactly `AIDED_S + OUTAGE_S` is applied
    // in that row, so the row before it is the one that has coasted the whole outage.
    let outage_end = AIDED_S + OUTAGE_S;
    let start = truth[0].timestamp;
    let at_outage_end = |row: &&NavigationResult| {
        (row.timestamp - start).num_milliseconds() as f64 / 1000.0 >= outage_end - 0.15
    };
    if let (Some(estimate), Some(actual)) = (
        results.iter().find(at_outage_end),
        truth.iter().find(at_outage_end),
    ) {
        print_outage_end(estimate, actual);
    }

    // ANCHOR: biases
    // Fifteen navigation and IMU-bias states, then the barometric bias: [lat, lon, alt,
    // v_n, v_e, v_v, roll, pitch, yaw, b_a (3), b_g (3), baro bias].
    let state = eskf.get_estimate();
    println!(
        "final accelerometer bias estimate: [{:.4}, {:.4}, {:.4}] m/s^2",
        state[9], state[10], state[11]
    );
    println!(
        "final gyroscope bias estimate:     [{:.5}, {:.5}, {:.5}] rad/s",
        state[12], state[13], state[14]
    );
    if let Some(index) = eskf.baro_bias_index() {
        println!("final barometric bias estimate:    {:.3} m", state[index]);
    }
    // ANCHOR_END: biases

    Ok(())
}

/// Print the horizontal error at one epoch beside the filter's own 1-sigma.
fn print_outage_end(estimate: &NavigationResult, truth: &NavigationResult) {
    let error_m = haversine_distance(
        estimate.latitude.to_radians(),
        estimate.longitude.to_radians(),
        truth.latitude.to_radians(),
        truth.longitude.to_radians(),
    );
    // The covariance columns stay in the filter's units: rad^2 for latitude and longitude.
    // Back to metres through the WGS84 radii of curvature at the estimate.
    let (meridian_m, transverse_m, _) = principal_radii(&estimate.latitude, &estimate.altitude);
    let north_sigma_m = estimate.latitude_cov.sqrt() * (meridian_m + estimate.altitude);
    let east_sigma_m = estimate.longitude_cov.sqrt()
        * (transverse_m + estimate.altitude)
        * estimate.latitude.to_radians().cos();
    println!(
        "end of outage: error {error_m:.2} m, filter 1-sigma {north_sigma_m:.2} m north, \
         {east_sigma_m:.2} m east"
    );
}
