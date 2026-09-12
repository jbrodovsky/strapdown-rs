//! Minimal `InsEngine` usage: propagate an IMU, fuse a GNSS fix, read the solution.
//!
//! Run it with:
//!
//! ```bash
//! cargo run -p strapdown-core --example basic_ins
//! ```
//!
//! The trajectory is synthetic and deliberately trivial -- a vehicle driving due north at a
//! constant 10 m/s on a level road -- so that the numbers printed can be checked by hand.
//! What the example is really showing is the shape of the loop, which is the same whether the
//! samples come from a file, a driver, or a generator like this one:
//!
//! 1. Build an engine from an initial state.
//! 2. For every IMU sample, `predict`.
//! 3. Whenever a GNSS fix arrives, `update_gnss`.
//! 4. Read `nav_solution` whenever you want the current estimate.
//!
//! For a worked GNSS-denial scenario, see `gnss_outage.rs`.

use nalgebra::Vector3;

use strapdown::earth::METERS_TO_DEGREES;
use strapdown::engine::{GnssFix, InsEngine, InsEngineConfig};
use strapdown::kalman::InitialState;
use strapdown::{IMUData, ImuSample, StrapdownError};

/// Sample interval, seconds. 100 Hz is a typical MEMS IMU rate.
const DT: f64 = 0.01;
/// How long to drive, in seconds.
const DURATION_S: f64 = 60.0;
/// Constant northward ground speed, m/s.
const SPEED_MPS: f64 = 10.0;
/// How often a GNSS fix arrives, in seconds.
const GNSS_INTERVAL_S: f64 = 1.0;

/// One-sigma GNSS accuracies for the synthetic receiver, metres and m/s.
const GNSS_HORIZONTAL_STD_M: f64 = 3.0;
const GNSS_VERTICAL_STD_M: f64 = 5.0;
const GNSS_VELOCITY_STD_MPS: f64 = 0.2;

fn main() -> Result<(), StrapdownError> {
    // Start level, stationary-ish, heading north, somewhere in Philadelphia.
    let start_latitude_deg = 39.95;
    let start_longitude_deg = -75.16;
    let start_altitude_m = 12.0;

    let initial_state = InitialState::new(
        start_latitude_deg,
        start_longitude_deg,
        start_altitude_m,
        SPEED_MPS, // north
        0.0,       // east
        0.0,       // vertical
        0.0,       // roll
        0.0,       // pitch
        0.0,       // yaw: heading north
        true,      // the angles above are in degrees
        None,      // no initial bias estimate
    );

    // NED, no antenna lever arm. `InsEngineConfig::default()` is NED already; naming it here
    // is documentation rather than necessity.
    let mut engine = InsEngine::builder()
        .with_config(InsEngineConfig {
            is_enu: false,
            ..InsEngineConfig::default()
        })
        .with_initial_state(initial_state)
        .build()?;

    println!("start: {}", engine.nav_solution());

    // A level, non-accelerating vehicle senses only the reaction to gravity, and the sign is
    // the thing to get right. An accelerometer measures *specific force*, not acceleration:
    // at rest the ground pushes the vehicle up, so the sensed vector points up. In NED the
    // vertical axis points down, which makes that reading **negative**: -9.81 m/s^2 on the
    // down axis. It has to be, for the mechanization to cancel it -- `velocity_update` adds
    // gravity (positive down in NED) to the sensed increment, and a stationary vehicle's
    // velocity must not change.
    //
    // Getting this backwards does not fail loudly. It drives the filter to conclude the
    // vehicle is upside down: roll converges to 180 degrees and the vertical channel walks
    // away even while GNSS is holding altitude fixed.
    //
    // In ENU the same reading is +9.81 on the up axis, which is what the Sensor Logger
    // exports in `core/tests/test_data.csv` look like.
    let stationary_imu = IMUData {
        accel: Vector3::new(0.0, 0.0, -9.81),
        gyro: Vector3::zeros(),
    };

    let steps = (DURATION_S / DT) as usize;
    let gnss_every = (GNSS_INTERVAL_S / DT) as usize;

    for step in 1..=steps {
        engine.predict(&ImuSample::from_rates(&stationary_imu, DT))?;

        if step % gnss_every == 0 {
            let elapsed_s = step as f64 * DT;
            // Truth for this trajectory: distance north is speed * time, converted to degrees
            // of latitude. A real fix would carry receiver noise; this one does not, which is
            // why the estimate below tracks it so closely.
            let north_m = SPEED_MPS * elapsed_s;
            let fix = GnssFix::position(
                start_latitude_deg + north_m * METERS_TO_DEGREES,
                start_longitude_deg,
                start_altitude_m,
                GNSS_HORIZONTAL_STD_M,
                GNSS_VERTICAL_STD_M,
            )
            .with_velocity([SPEED_MPS, 0.0, 0.0], GNSS_VELOCITY_STD_MPS);

            let outcome = engine.update_gnss(&fix)?;
            if !outcome.accepted {
                // With no gate installed this never fires. It is worth handling anyway: once
                // you call `set_innovation_gate`, a rejected fix is reported here rather than
                // raised as an error, because a rejection is a normal event and not a fault.
                println!(
                    "  t={elapsed_s:6.1}s  fix rejected by the gate (NIS {:.1})",
                    outcome.nis
                );
            }
        }
    }

    let solution = engine.nav_solution();
    println!("after {DURATION_S:.0} s: {solution}");

    let expected_north_m = SPEED_MPS * DURATION_S;
    let actual_north_m = (solution.latitude - start_latitude_deg) / METERS_TO_DEGREES;
    println!(
        "travelled {actual_north_m:.1} m north; expected {expected_north_m:.1} m \
         (difference {:.2} m)",
        actual_north_m - expected_north_m
    );
    println!(
        "position uncertainty (1-sigma): {:.2} m north, {:.2} m east, {:.2} m vertical",
        solution.position_std_m[0], solution.position_std_m[1], solution.position_std_m[2]
    );

    Ok(())
}
