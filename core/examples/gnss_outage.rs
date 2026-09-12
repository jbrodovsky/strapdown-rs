//! Dead-reckoning performance during GNSS denial, and recovery once the signal returns.
//!
//! Run it with:
//!
//! ```bash
//! cargo run -p strapdown-core --example gnss_outage
//! ```
//!
//! The scenario is the one every GNSS-denied study starts from: a vehicle tracking normally,
//! a stretch with no fixes (a tunnel, an urban canyon, jamming), then aiding returns. What it
//! shows is the shape of the error -- flat while aided, growing quadratically while coasting,
//! and collapsing again within a few fixes of recovery -- and, more usefully, why the growth
//! is several times smaller than double-integrating the sensor's bias would predict.
//!
//! Two knobs are worth turning after the first run:
//!
//! * `OUTAGE_DURATION_S` -- drift grows with the *square* of this, so doubling it roughly
//!   quadruples the peak error.
//! * `ACCEL_BIAS_MPS2` -- the accelerometer bias present in the sensor. Set it to zero and the
//!   vehicle coasts almost perfectly. The drift does *not* scale with it as `0.5 * a * t^2`
//!   would suggest, and the output explains why: the filter absorbs most of the bias into a
//!   small pitch error, because the two are not separately observable from position and
//!   velocity aiding.
//!
//! For the same scenario driven from a config file rather than in code, see
//! `examples/configs/simple_dropout.yaml` and `extended_gnss_denied.yaml`.

use nalgebra::Vector3;

use strapdown::earth::{METERS_TO_DEGREES, haversine_distance};
use strapdown::engine::{GnssFix, InsEngine, InsEngineConfig};
use strapdown::kalman::InitialState;
use strapdown::{IMUData, ImuSample, StrapdownError};

const DT: f64 = 0.01;
const SPEED_MPS: f64 = 20.0;
const GNSS_INTERVAL_S: f64 = 1.0;

/// When the outage starts and how long it lasts, in seconds.
const AIDED_BEFORE_S: f64 = 60.0;
const OUTAGE_DURATION_S: f64 = 120.0;
const AIDED_AFTER_S: f64 = 120.0;

/// Uncorrected accelerometer bias on the body x-axis, m/s^2.
///
/// Representative of a consumer MEMS part after turn-on. This is the term that makes the
/// coasting error grow quadratically: the filter has no way to distinguish it from real
/// acceleration once GNSS stops arriving.
const ACCEL_BIAS_MPS2: f64 = 0.05;

const GNSS_HORIZONTAL_STD_M: f64 = 3.0;
const GNSS_VERTICAL_STD_M: f64 = 5.0;
const GNSS_VELOCITY_STD_MPS: f64 = 0.2;

const START_LATITUDE_DEG: f64 = 39.95;
const START_LONGITUDE_DEG: f64 = -75.16;
const START_ALTITUDE_M: f64 = 12.0;

fn main() -> Result<(), StrapdownError> {
    let mut engine = InsEngine::builder()
        .with_config(InsEngineConfig {
            is_enu: false,
            ..InsEngineConfig::default()
        })
        .with_initial_state(InitialState::new(
            START_LATITUDE_DEG,
            START_LONGITUDE_DEG,
            START_ALTITUDE_M,
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

    // Level and non-accelerating, so the accelerometer senses only the reaction to gravity:
    // -9.81 on the down axis in NED. See `basic_ins.rs` for why the sign is what it is. The
    // bias is added on the body x-axis (forward), where it looks exactly like the vehicle
    // speeding up.
    let imu = IMUData {
        accel: Vector3::new(ACCEL_BIAS_MPS2, 0.0, -9.81),
        gyro: Vector3::zeros(),
    };

    let total_s = AIDED_BEFORE_S + OUTAGE_DURATION_S + AIDED_AFTER_S;
    let steps = (total_s / DT) as usize;
    let gnss_every = (GNSS_INTERVAL_S / DT) as usize;

    println!(
        "aided for {AIDED_BEFORE_S:.0} s, dark for {OUTAGE_DURATION_S:.0} s, \
         aided again for {AIDED_AFTER_S:.0} s"
    );
    println!("accelerometer bias: {ACCEL_BIAS_MPS2} m/s^2 on the forward axis\n");
    println!("{:>8}  {:>12}  {:>10}", "t (s)", "error (m)", "GNSS");

    let mut peak_error_m = 0.0_f64;
    let mut error_at_outage_end_m = 0.0_f64;
    let mut estimated_bias_at_outage_start = [0.0_f64; 3];
    let mut pitch_at_outage_start_deg = 0.0_f64;

    for step in 1..=steps {
        let elapsed_s = step as f64 * DT;
        engine.predict(&ImuSample::from_rates(&imu, DT))?;

        // Truth: constant speed due north from the start point. The vehicle really is doing
        // this; the bias only exists in what the IMU reports.
        let truth_latitude_deg = START_LATITUDE_DEG + SPEED_MPS * elapsed_s * METERS_TO_DEGREES;

        let in_outage =
            elapsed_s > AIDED_BEFORE_S && elapsed_s <= AIDED_BEFORE_S + OUTAGE_DURATION_S;

        // Snapshot the bias estimate at the moment the fixes stop; it is what the filter has
        // to coast on, and the figure that explains the drift below.
        if (elapsed_s - AIDED_BEFORE_S).abs() < DT {
            let solution = engine.nav_solution();
            estimated_bias_at_outage_start = solution.accel_bias;
            pitch_at_outage_start_deg = solution.pitch;
        }

        if !in_outage && step % gnss_every == 0 {
            let fix = GnssFix::position(
                truth_latitude_deg,
                START_LONGITUDE_DEG,
                START_ALTITUDE_M,
                GNSS_HORIZONTAL_STD_M,
                GNSS_VERTICAL_STD_M,
            )
            .with_velocity([SPEED_MPS, 0.0, 0.0], GNSS_VELOCITY_STD_MPS);
            engine.update_gnss(&fix)?;
        }

        if step % gnss_every == 0 {
            let solution = engine.nav_solution();
            let error_m = haversine_distance(
                solution.latitude.to_radians(),
                solution.longitude.to_radians(),
                truth_latitude_deg.to_radians(),
                START_LONGITUDE_DEG.to_radians(),
            );
            peak_error_m = peak_error_m.max(error_m);
            if (elapsed_s - (AIDED_BEFORE_S + OUTAGE_DURATION_S)).abs() < DT {
                error_at_outage_end_m = error_m;
            }

            // Print every 10 s so the table stays readable.
            if (elapsed_s % 10.0).abs() < DT {
                println!(
                    "{elapsed_s:8.0}  {error_m:12.1}  {:>10}",
                    if in_outage { "denied" } else { "ok" }
                );
            }
        }
    }

    let final_solution = engine.nav_solution();
    let final_error_m = haversine_distance(
        final_solution.latitude.to_radians(),
        final_solution.longitude.to_radians(),
        (START_LATITUDE_DEG + SPEED_MPS * total_s * METERS_TO_DEGREES).to_radians(),
        START_LONGITUDE_DEG.to_radians(),
    );

    // What an unaided INS carrying this bias would do: double-integrate it over the outage.
    let uncorrected_drift_m = 0.5 * ACCEL_BIAS_MPS2 * OUTAGE_DURATION_S.powi(2);

    // The observed drift is far smaller than that, and the reason is worth reading off the
    // state rather than guessing at. The filter did *not* estimate the bias -- that state
    // barely moved. It absorbed the bias into **attitude** instead: it is pitched nose-up by
    // a fraction of a degree, and the gravity that tilt leaks into the forward axis very
    // nearly cancels the bias.
    //
    // That is not a defect. A constant forward specific-force bias and a small pitch error
    // produce the same horizontal acceleration signature, so position and velocity aiding
    // cannot separate them -- the pair is unobservable from this measurement set alone. The
    // filter found an explanation that fits every fix it was given, and it happens to be the
    // one that also coasts well.
    let gravity_leak = 9.81 * pitch_at_outage_start_deg.to_radians().sin();

    println!("\n--- why the drift is smaller than the textbook figure ---");
    println!("accelerometer bias present:        {ACCEL_BIAS_MPS2:.4} m/s^2 (forward axis)");
    println!(
        "bias the filter estimated:         {:.4} m/s^2  <- it barely moved",
        estimated_bias_at_outage_start[0]
    );
    println!("pitch error at outage start:       {pitch_at_outage_start_deg:.4} deg");
    println!(
        "gravity that tilt leaks forward:   {gravity_leak:.4} m/s^2  <- nearly cancels the bias\n"
    );

    println!("error at end of outage:    {error_at_outage_end_m:.1} m");
    println!("0.5 * full bias * t^2:     {uncorrected_drift_m:.1} m   <- an unaided INS");
    println!("peak error over the run:   {peak_error_m:.1} m");
    println!("error after recovery:      {final_error_m:.1} m");

    println!(
        "\nThe lesson is not the quadratic -- it is that accelerometer bias and tilt are not\n\
         separately observable from position and velocity aiding. Which of the two the filter\n\
         'blames' is arbitrary, and it changes how well the solution coasts. Aiding that does\n\
         separate them (ZUPT, a second antenna, a magnetometer for heading) is what makes\n\
         dead-reckoning performance predictable rather than lucky."
    );

    Ok(())
}
