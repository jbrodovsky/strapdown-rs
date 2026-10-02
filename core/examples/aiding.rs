//! Measurement models beyond GNSS, and innovation gating.
//!
//! Run it with:
//!
//! ```bash
//! cargo run -p strapdown-core --example aiding
//! ```
//!
//! Three short scenes, each on a fresh [`InsEngine`] (the default 15-state ESKF):
//!
//! 1. **A parked vehicle.** A [`StationaryDetector`] decides when the IMU is still, and
//!    while it is, zero-velocity (ZUPT) and zero-angular-rate (ZARU) pseudo-measurements are
//!    applied, with a barometric altitude once a second. The gyro carries a bias the filter
//!    is not told about; ZARU is what lets it find it.
//! 2. **A magnetometer heading.** A body-frame field is synthesised for a known heading and
//!    turned back into a yaw measurement, with WMM declination applied.
//! 3. **A spoofed fix.** A chi-squared innovation gate rejects a GNSS fix 200 m off, and the
//!    default [`GateRecovery`] rejects four fixes in a row and forces the fifth through.
//!
//! None of the pseudo-measurements here are applied by `strapdown-sim`: the CLI's event
//! stream carries GNSS position and velocity, barometric altitude and magnetometer yaw. ZUPT,
//! ZARU and the stationary detector are library building blocks, called as below.

use std::error::Error;

use nalgebra::{Rotation3, Vector3};

use strapdown::earth::{METERS_TO_DEGREES, earth_rate_lla};
use strapdown::engine::{GnssFix, InsEngine};
use strapdown::gating::{GateRecovery, InnovationGate, chi_squared_quantile};
use strapdown::kalman::InitialState;
use strapdown::measurements::{
    BAROMETRIC_ALTITUDE_NOISE_M, MagnetometerYawMeasurement, MeasurementModel,
    RelativeAltitudeMeasurement, ZaruMeasurement, ZuptMeasurement,
};
use strapdown::stationary::{StationaryConfig, StationaryDetector};
use strapdown::{IMUData, StrapdownError};

const LATITUDE_DEG: f64 = 39.95;
const LONGITUDE_DEG: f64 = -75.16;
const ALTITUDE_M: f64 = 12.0;
/// IMU sample interval, seconds (100 Hz, what `StationaryConfig::default()` is tuned for).
const DT: f64 = 0.01;

fn main() -> Result<(), Box<dyn Error>> {
    parked()?;
    magnetometer_heading();
    spoofed_fix()?;
    Ok(())
}

/// Scene 1: ZUPT, ZARU and barometric altitude on a parked vehicle.
fn parked() -> Result<(), StrapdownError> {
    let mut engine = InsEngine::builder()
        .with_initial_state(InitialState::new(
            LATITUDE_DEG,
            LONGITUDE_DEG,
            ALTITUDE_M,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            true, // degrees
            None, // NED, the default
        ))
        .build()?;

    // Level, heading north, NED: the accelerometer reads the reaction to gravity (-g on the
    // down axis) and the gyro reads the Earth's rotation plus a bias the filter has not been
    // given. Level and north-pointing means the body and navigation axes coincide, so the
    // Earth rate resolves into the body frame unchanged.
    let gyro_bias = Vector3::new(0.0, 0.0, 0.002); // rad/s, about z
    let imu = IMUData {
        accel: Vector3::new(0.0, 0.0, -9.81),
        gyro: earth_rate_lla(&LATITUDE_DEG) + gyro_bias,
    };

    // ANCHOR: stationary
    let mut detector = StationaryDetector::new(StationaryConfig::default());
    let mut pseudo_measurements = 0;
    for step in 1..=3000 {
        engine.predict_rates(&imu, DT)?;

        // The detector needs a full window, then a dwell, before it latches: 1.5 s at the
        // defaults. Pseudo-measurements are applied at 10 Hz once it has.
        if detector.push(&imu) && step % 10 == 0 {
            engine.update(&ZuptMeasurement::default())?;
            // ZARU reads the raw gyro: on a stationary platform that is bias plus Earth rate,
            // and the model predicts the Earth rate itself.
            engine.update(&ZaruMeasurement::from_gyro([
                imu.gyro.x, imu.gyro.y, imu.gyro.z,
            ]))?;
            pseudo_measurements += 1;
        }

        // A barometer that has not moved since the reference epoch.
        if step % 100 == 0 {
            engine.update(&RelativeAltitudeMeasurement {
                relative_altitude: 0.0,
                reference_altitude: ALTITUDE_M,
                noise_std: BAROMETRIC_ALTITUDE_NOISE_M,
                bias_index: None, // no barometric-bias state in this filter
            })?;
        }
    }
    // ANCHOR_END: stationary

    let solution = engine.nav_solution();
    println!(
        "parked for {:.0} s, {pseudo_measurements} ZUPT/ZARU pairs",
        solution.elapsed_s
    );
    println!(
        "  gyro bias estimate z: {:.5} rad/s (injected {:.5})",
        solution.gyro_bias[2], gyro_bias.z
    );
    println!(
        "  speed: {:.4} m/s, altitude {:.2} m",
        solution.velocity_north.hypot(solution.velocity_east),
        solution.altitude
    );
    Ok(())
}

/// Scene 2: a tilt-compensated magnetometer heading with WMM declination.
fn magnetometer_heading() {
    // ANCHOR: magnetometer
    let mut magnetometer = MagnetometerYawMeasurement::default(); // NED, 0.05 rad noise
    magnetometer.apply_declination = true;
    magnetometer.year = 2025;
    magnetometer.day_of_year = 1;
    let declination = magnetometer.get_declination(LATITUDE_DEG, LONGITUDE_DEG, ALTITUDE_M);

    // A field whose horizontal part points to magnetic north, `declination` east of true
    // north, seen from a level vehicle heading 30 degrees true.
    let true_heading = 30.0_f64.to_radians();
    let field_ned = Vector3::new(20.0 * declination.cos(), 20.0 * declination.sin(), 45.0); // uT
    let field_body = Rotation3::from_euler_angles(0.0, 0.0, true_heading).inverse() * field_ned;
    magnetometer.mag_x = field_body.x;
    magnetometer.mag_y = field_body.y;
    magnetometer.mag_z = field_body.z;

    // Roll and pitch come from the state; latitude, longitude and altitude feed the WMM.
    let state = nalgebra::DVector::from_vec(vec![
        LATITUDE_DEG.to_radians(),
        LONGITUDE_DEG.to_radians(),
        ALTITUDE_M,
        0.0,
        0.0,
        0.0,
        0.0, // roll
        0.0, // pitch
        0.0, // yaw: not used to form the measurement
    ]);
    let measured = magnetometer.get_measurement(&state);
    // ANCHOR_END: magnetometer

    if let Ok(measured) = measured {
        println!(
            "magnetometer: declination {:.2} deg, true heading {:.2} deg, measured {:.2} deg",
            declination.to_degrees(),
            true_heading.to_degrees(),
            measured[0].to_degrees()
        );
    }
}

/// Scene 3: a chi-squared gate, a spoofed fix, and the recovery that bounds a rejection run.
fn spoofed_fix() -> Result<(), StrapdownError> {
    const SPEED_MPS: f64 = 10.0;
    let mut engine = InsEngine::builder()
        .with_initial_state(InitialState::new(
            LATITUDE_DEG,
            LONGITUDE_DEG,
            ALTITUDE_M,
            SPEED_MPS,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            true,
            None,
        ))
        .build()?;

    // ANCHOR: gate
    // Reject what the filter's own model says should happen less than 0.1% of the time. The
    // threshold depends on the measurement's dimension, so one confidence serves a 1-D
    // barometer and a 3-D position fix alike.
    let gate = InnovationGate::chi_squared(0.999)?;
    engine.set_innovation_gate(Some(gate));
    // Already the default; written out to show the knobs: double the covariance in the
    // observed directions on every rejection, force an update after five in a row.
    engine.set_gate_recovery(GateRecovery::new(2.0, Some(5))?);
    // ANCHOR_END: gate

    for dof in [1, 3, 5] {
        println!(
            "chi-squared 0.999 threshold, {dof} dof: {:.2}",
            chi_squared_quantile(0.999, dof)
        );
    }

    let level = IMUData {
        accel: Vector3::new(0.0, 0.0, -9.81),
        gyro: Vector3::zeros(),
    };
    for second in 1..=70 {
        for _ in 0..100 {
            engine.predict_rates(&level, DT)?;
        }
        let north_m = SPEED_MPS * f64::from(second);
        // From t = 61 s on, the receiver reports a position 200 m east of the truth.
        let spoof_m = if second > 60 { 200.0 } else { 0.0 };
        let fix = GnssFix::position(
            LATITUDE_DEG + north_m * METERS_TO_DEGREES,
            LONGITUDE_DEG + spoof_m * METERS_TO_DEGREES / LATITUDE_DEG.to_radians().cos(),
            ALTITUDE_M,
            3.0,
            5.0,
        );
        // ANCHOR: outcome
        let outcome = engine.update_gnss(&fix)?;
        if second > 58 {
            println!(
                "t = {second:2} s  NIS {:>10.1}  accepted {:<5}  forced {}",
                outcome.nis, outcome.accepted, outcome.forced
            );
        }
        // ANCHOR_END: outcome
    }
    Ok(())
}
