//! Comprehensive integration tests for INS filters using real data
#![allow(
    clippy::unwrap_used,
    clippy::panic,
    reason = "integration tests are a separate crate target, so `clippy.toml`'s \
              allow-unwrap-in-tests -- which only covers `#[cfg(test)]` items -- does not \
              reach them; unwrapping is how these assert"
)]
//!
//! This module contains end-to-end integration tests for the strapdown inertial navigation
//! filters using real data recorded from a MEMS-grade IMU. See [mems-nav-dataset](www.github.com/jbrodovsky/mems-nav-dataset).
//! These tests ensure that the entire navigation system works as expected in realistic scenarios, not just
//! at the API level but with actual IMU and GNSS data.
//!
//! ## Error Metrics
//!
//! The tests use the following error metrics to validate filter performance:
//! - **Horizontal position error**: Haversine distance between estimated and GNSS positions (meters)
//! - **Altitude error**: Simple difference between estimated and GNSS altitude (meters)
//! - **Velocity error**: Component-wise differences for north, east, and down velocities (m/s)
//! - **Orientation error**: Component-wise differences for roll, pitch, and yaw (radians)
//!
//! The specific performance numbers given in the assertions in the test are not theoretical
//! or design goals, but rather empirically derived from running the filters on the dataset and observing
//! performance on the test data set. They serve as regression checks to ensure that future changes
//! do not degrade performance.
//!
//! ## Test Structure
//!
//! Tests load real data from CSV files, run the filters, and compute error metrics against
//! GNSS measurements. The tests verify that:
//! 1. Filters complete without errors
//! 2. Position errors remain within reasonable bounds
//! 3. Velocity and orientation estimates are stable
//! 4. The closed-loop filter outperforms dead reckoning
//!
//! ## Theoretical bounds
//!
//! Most limits in this file are regression guards: empirical levels chosen to catch a change
//! for the worse. A few are not, and the difference matters -- a bound derived from the
//! measurement setup stays valid when the tuning changes, an empirical one does not. The four
//! below are derived, and they explain why the empirical ones sit where they do.
//!
//! **The error floor is set by the reference, not by the filter.** "Truth" here is the GNSS
//! fix, which is also the filters' aiding source, so these metrics measure agreement with the
//! aiding signal rather than agreement with an independent truth. `test_data.csv` carries the
//! receiver's own accuracy estimate: 3.81 m horizontal and 1.38 m vertical, averaged over the
//! run (1 sigma). No filter scored against this reference can show an RMSE below roughly
//! 3.8 m however good it is, and one that did would be reporting the reference's noise rather
//! than its own accuracy. Independent ground truth would need a different dataset.
//!
//! **The ~23 m horizontal RMSE this file used to report was a labelling defect, and it is
//! gone.** `sim::run_closed_loop` emitted each row *after* applying the first event of the
//! following epoch, so a row stamped `t_k` held a state already propagated to `t_{k+1}`. At
//! 1 Hz and 21.19 m/s that is 21.2 m of along-track error on its own -- very nearly the whole
//! of the figure, and the reason all four filters used to land within 0.3 m of each other
//! rather than spreading out by tuning. This header previously attributed it to the harness's
//! timestamp matching; the direction was right and the cause was not, and it was generated in
//! the runner rather than inherited from the 1 Hz reference. Fixed in #367.
//!
//! **What that uncovered underneath it, and how it was resolved.** With the label corrected,
//! a row at `t_k` contains `t_k`'s GNSS update -- and on this full-rate stream the UKF and EKF
//! then scored *below* the reference's own 3.81 m, at 0.01 m and 0.0001 m. That is not
//! accuracy: a filter whose Kalman gain is ~1 reproduces the fix it was given, and scoring it
//! against that fix is circular. Both carried an absolute `eps = 1e-9` covariance floor
//! against a latitude variance in rad^2, which is a 201 m horizontal sigma. The ESKF, which
//! used a relative floor (#266), sat at 5 m and was unaffected.
//!
//! **#373 has since landed** and given all three the relative floor, so they now read 4.78,
//! 5.22 and 5.12 m -- above the reference's own noise rather than beneath it. The horizontal
//! limits below were deliberately left un-re-derived while that was in flight, on the grounds
//! that a bound fitted to a transient is worse than a loose one (#288). They are still the
//! derived physical bounds they always were, and tightening them to match today's numbers
//! would make them a baseline, which is what `core/tests/perf_baseline.rs` is for.
//!
//! **Each attitude axis is bounded by the sensor that observes it.** Roll and pitch are
//! observable through gravity: the accelerometer senses a 9.81 m/s^2 vector whose direction
//! in the body frame fixes two of the three angles, so [`MAX_LEVEL_ATTITUDE_RMSE_RAD`] is
//! derived from gravity and they hold 2.55-3.48 deg. Yaw has no such anchor -- position-only
//! aiding leaves it unobservable, and stripping the magnetometer events from the stream
//! confirms it: the ESKF reaches 102.9 deg, against the 103.9 deg (= 180/sqrt(3)) of an error
//! uniform on the circle. What makes it observable is the magnetometer, which this dataset
//! carries and `build_event_stream` has always emitted, so [`MAX_YAW_RMSE_RAD`] is derived
//! from *that* sensor's own measured error instead.
//!
//! Yaw was previously reported unasserted, described here as unobservable. That description
//! was half right and the half that was wrong mattered: the aiding was present the whole
//! time, but [`MagnetometerYawMeasurement`] computed its heading on one hardcoded branch
//! regardless of the frame of the state it was updating. On this ENU dataset that returns
//! pi/2 - psi -- the heading reflected about the 45 deg line -- and the filters converged on
//! the reflection, which is why the yaw column read 86-98 deg and why the EKF was *better*
//! with the magnetometer stripped (57.6 deg) than with it (88.2 deg). An aid that makes a
//! filter worse than no aid is a defect, not an observability limit. Fixed in #305; the
//! numbers are now ESKF 16.4, EKF 22.7, UKF 22.8 deg.
//!
//! The convention-free geodesic attitude error is reported alongside the per-axis figures
//! because it is the quantity that does not depend on a choice of Euler sequence, and on this
//! dataset it still tracks the yaw error closely, which is what shows the other two axes are
//! healthy.
//!
//! **Two filters are exempt from the yaw bound**, for reasons that are about their attitude
//! representations rather than about aiding, and both are recorded at the assertion: the UKF
//! averages sigma-point Euler angles linearly (#336, so its 22.8 deg is not evidence of
//! anything), and the RBPF linearly averages unwrapped per-particle Euler error states and
//! sits at 65.9 deg.
//!
//! **The dead-reckoning baseline is a 240-sample window, not the whole recording.** Unaided
//! dead reckoning over all 5,366 samples ends 6.57e6 m from truth, so asserting that a filter
//! at 23.5 m beats it is a 280,000x ratio asserted as `<`: it cannot fail for any reason to do
//! with navigation, and an EKF 14,707 km from truth passed it (#307, #299). It also reaches
//! that figure by way of -6.14e6 m of altitude, where the `r_e + altitude` denominator in
//! `earth::transport_rate` is within 3.7% of zero and the attitude update's finiteness is a
//! floating-point accident -- the Windows/Linux split that #299 was filed for. The baseline is
//! therefore truncated to a window bounded from below by the point where dead reckoning leaves
//! the band a healthy filter occupies, and from above by the altitude band the mechanization is
//! documented over, and placed where the margins against those two are equal. See
//! [`DEAD_RECKONING_BASELINE_SAMPLES`] for the arithmetic. Exactly one test still runs the full
//! 89-minute arc, `test_dead_reckoning_on_real_data`, and it accepts either outcome.
use chrono::Datelike;
use std::path::Path;

use strapdown::earth::haversine_distance;
use strapdown::engine::{GnssFix, InsEngine, InsEngineConfig};
use strapdown::gating::InnovationGate;
use strapdown::kalman::{
    ErrorStateKalmanFilter, ExtendedKalmanFilter, InitialState, UnscentedKalmanFilter,
};
use strapdown::measurements::MAG_YAW_NOISE;
use strapdown::measurements::{MagnetometerYawMeasurement, MeasurementModel, ZuptMeasurement};
use strapdown::messages::{
    AidingConfig, Event, GnssFaultModel, MeasurementScheduler, build_event_stream,
};
use strapdown::rbpf::{RaoBlackwellizedParticleFilter, RbpfConfig};
// `DEFAULT_PROCESS_NOISE_DENSITY` is imported, not copied. This file used to keep its own array of
// literals carrying a comment claiming it matched the crate's -- it did not (its altitude
// entry was `1e-6` where the crate's was `1e-4`), and both copies carried #308's units
// defect: latitude and longitude written in rad^2 with values chosen as though they were
// metres, i.e. a 6.4 km per-step standard deviation. A suite that keeps its own copy of the
// tuning it is validating cannot notice when that tuning is wrong, which is most of why the
// defect survived as long as it did. What these tests exercise is now, by construction, what
// the library ships.
use strapdown::sim::{
    DEFAULT_INITIAL_POSITION_UNCERTAINTY_M, DEFAULT_PROCESS_NOISE_DENSITY, EskfConfig,
    NavigationResult, TestDataRecord, check_declared_frame, dead_reckoning, initialize_eskf,
    run_closed_loop,
};
use strapdown::stationary::{StationaryConfig, StationaryDetector};
use strapdown::{
    IMUData, IMUQuality, ImuSample, InitialUncertainty, NavigationFilter, StrapdownError,
    StrapdownState, wrap_to_pi,
};

use nalgebra::{DMatrix, DVector, Rotation3, Vector3};

/// Default initial covariance for testing (15-state).
///
/// The position block comes from the crate's [`DEFAULT_INITIAL_POSITION_UNCERTAINTY_M`], for
/// the reason `DEFAULT_PROCESS_NOISE_DENSITY` is imported rather than copied. The literals it replaces
/// -- `1e-6, 1e-6, 1.0`, commented "(lat, lon, alt in meters)" -- were #308 in $P_0$: latitude
/// and longitude are radians, so `1e-6 rad^2` is a 6367 m claim and only the altitude entry
/// was ever the metre it said. Every filter in this file therefore started each run believing
/// it might be an Earth radius from its own seed.
const DEFAULT_INITIAL_COVARIANCE: [f64; 15] = [
    INITIAL_HORIZONTAL_VARIANCE_RAD2, // latitude covariance, rad^2
    INITIAL_HORIZONTAL_VARIANCE_RAD2, // longitude covariance, rad^2
    DEFAULT_INITIAL_POSITION_UNCERTAINTY_M * DEFAULT_INITIAL_POSITION_UNCERTAINTY_M, // alt, m^2
    0.1,
    0.1,
    0.1, // velocity covariance (m/s)
    0.01,
    0.01,
    0.01, // attitude covariance (radians)
    0.01,
    0.01,
    0.01, // accelerometer bias covariance (m/s²)
    0.001,
    0.001,
    0.001, // gyroscope bias covariance (rad/s)
];

/// [`DEFAULT_INITIAL_POSITION_UNCERTAINTY_M`] as a latitude/longitude variance, rad^2.
const INITIAL_HORIZONTAL_VARIANCE_RAD2: f64 = {
    let radians = DEFAULT_INITIAL_POSITION_UNCERTAINTY_M * strapdown::earth::METERS_TO_RADIANS;
    radians * radians
};

/// The local-level frame `test_data.csv` is recorded in: ENU.
///
/// Sensor Logger writes the ENU convention, so at rest the recording's specific force rotated
/// through its own quaternion lands on up at +9.72 m/s^2 (`TestDataRecord::attitude` quotes
/// the same figure for sample 0, and `assert_test_data_is_enu` re-derives it from the file on
/// every run). NED, which is the library's and the CLI's default since #296, expects -9.78,
/// so every entry point in this file that mechanizes this recording has to say ENU
/// explicitly. Named rather than spelled `true` at five call sites because the bare boolean
/// is exactly the kind of argument that gets copied into a sixth call site on a different
/// dataset without anyone rechecking it.
const IS_ENU_TEST_DATA: bool = true;

/// Mean 1-sigma horizontal accuracy the receiver reports across `test_data.csv` (meters).
///
/// The error floor described in the module documentation. Held as a constant so the bounds
/// below can refer to it; `assert_reference_accuracy_matches_dataset` checks it against the
/// data on every run so it cannot quietly go stale if the dataset is replaced.
const GNSS_REPORTED_HORIZONTAL_ACCURACY_M: f64 = 3.81;

/// Mean 1-sigma vertical accuracy the receiver reports across `test_data.csv` (meters).
const GNSS_REPORTED_VERTICAL_ACCURACY_M: f64 = 1.38;

/// Mean ground speed over `test_data.csv` (m/s).
///
/// At the recording's 1 Hz fix rate this is also, in metres, the apparent along-track error
/// produced by one sample of misalignment between an estimate and the record it is scored
/// against -- the term that dominates the horizontal RMSE. See the module documentation.
const MEAN_GROUND_SPEED_MPS: f64 = 21.19;

/// Horizontal RMSE ceiling applied to every healthy filter in the benchmark (meters).
///
/// The three healthy filters sit within 0.3 m of each other at ~23.5 m, which is the sample
/// alignment term described in the module documentation rather than filter error. 40 m is
/// ~1.7x that, tight enough to catch a filter that stops tracking and loose enough that it is
/// not measuring the harness.
const MAX_HORIZONTAL_RMSE_M: f64 = 40.0;

/// Vertical RMSE ceiling applied to every healthy filter in the benchmark (meters).
///
/// Observed at 2.4-4.6 m; 10 m leaves ~2x margin on the worst of them. The vertical channel
/// is the one that fails first when a filter regresses -- it is unstable without aiding and
/// was the signature of both #266 and #286 -- so this is worth asserting per filter rather
/// than folding into the horizontal check.
const MAX_VERTICAL_RMSE_M: f64 = 10.0;

/// Roll and pitch RMSE ceiling applied to every healthy filter in the benchmark (radians).
///
/// Only the *level* axes. Roll and pitch are gravity-observable and observed at 2.55-3.48 deg;
/// 10 deg leaves ~2.6x margin. Yaw is bounded separately by [`MAX_YAW_RMSE_RAD`], because it
/// is observable through a different sensor and its bound is derived from that sensor rather
/// than from gravity.
const MAX_LEVEL_ATTITUDE_RMSE_RAD: f64 = 10.0 * std::f64::consts::PI / 180.0;

/// The local-level frame `core/tests/test_data.csv` is expressed in.
///
/// One constant rather than a `true` repeated at each call site, because more than one thing
/// now reads it -- the filters' initial states and the event stream's magnetometer heading --
/// and they must agree. #305 is what happens when they do not: the magnetometer model assumed
/// a frame instead of being told one, disagreed with the state it was updating, and drove yaw
/// to a reflection of the truth for 89 minutes without anything noticing.
///
/// ENU because this is a Sensor Logger export, whose accelerometer reads +g along the
/// device's up-axis at rest; `check_declared_frame` asserts that against the data.
const TEST_DATA_IS_ENU: bool = true;

/// Yaw RMSE ceiling applied to the magnetometer-aided filters in the benchmark (radians).
///
/// # Where this number comes from
///
/// Yaw is observable on this dataset **only** through the magnetometer. Stripping the
/// magnetometer events from the stream leaves the ESKF at 102.9 deg RMSE, against the 103.9
/// deg (= 180/sqrt(3)) of an error distributed uniformly around the circle: statistically
/// indistinguishable from carrying no heading information at all, which is what the
/// position-only observability argument in #305 predicts. So the bound has to be derived from
/// the aiding source, not from the trajectory or from gravity.
///
/// The source's own error is directly measurable, with no filter involved: evaluate
/// [`MagnetometerYawMeasurement`] at the *reference* attitude for every record and compare
/// with the reference yaw. On this recording that is **17.06 deg RMS**, with a 3.07 deg mean
/// bias -- residual hard iron from the vehicle, whose own field puts the implied magnetic
/// north at -8.21 deg against a WMM declination of -11.40 deg.
/// `magnetometer_yaw_aiding_source_error_matches_derivation` measures that figure and fails if
/// the dataset stops supporting it, the same discipline
/// `assert_reference_accuracy_matches_dataset` applies to the GNSS accuracies.
///
/// A filter whose only yaw observation is that source cannot do better than it except by
/// smoothing, and how much smoothing an 89-minute drive buys is not something to predict in
/// advance -- so the ceiling is **1.5x the source's own RMS**, 25.6 deg. The factor is margin
/// for the filter being *worse* than its sensor (gyro drift between updates, the tilt
/// coupling the Jacobian deliberately neglects), not a fit: bounding at the source's 17.06 deg
/// itself is equally derivable but leaves 4% of headroom on the ESKF's measured 16.42 deg and
/// would be flaky rather than informative.
///
/// **Margins differ sharply between the filters this is asserted against**, and the ESKF's is
/// not the one to watch. ESKF 16.42 deg is 1.56x under the bound; **EKF 22.67 deg is 1.13x**,
/// using 89% of it. A z-gyro bias of 1e-2 rad/s puts the EKF at 24.85 deg -- 97% -- while the
/// ESKF does not move. So this bound is, in practice, a live assertion on the EKF and a loose
/// one on the ESKF, and a failure here should be read as an EKF finding first.
///
/// This number moves if `core/tests/test_data.csv` is replaced -- it describes that vehicle's
/// magnetic environment, not the filters.
const MAX_YAW_RMSE_RAD: f64 = 1.5 * MAG_YAW_SOURCE_RMSE_RAD;

/// Measured RMS error of the magnetometer heading itself on `core/tests/test_data.csv`.
///
/// Evaluated at the reference attitude, so no filter is involved: this is what the aiding
/// source knows, and the floor any filter reading it is working against. See
/// [`MAX_YAW_RMSE_RAD`], which derives the benchmark's ceiling from it, and
/// `magnetometer_yaw_aiding_source_error_matches_derivation`, which re-measures it.
const MAG_YAW_SOURCE_RMSE_RAD: f64 = 17.06 * std::f64::consts::PI / 180.0;

/// Length of the unaided dead-reckoning baseline arc, in samples.
///
/// The three `*_outperforms_dead_reckoning` tests score an aided filter against unaided dead
/// reckoning. They used to run that baseline over the whole 5,366-sample recording, which
/// breaks the comparison in both directions (#299):
///
/// * Dead reckoning ends the full recording 6.57e6 m from truth, against filters at 23.5 m.
///   Asserting `filter < baseline` on a 280,000x ratio cannot fail for any reason connected
///   to navigation -- a filter on the far side of the planet passes it, which is how #307's
///   14,707 km EKF sat here undetected.
/// * The arc reaches that figure by way of -6.14e6 m of altitude, where the `r_e + altitude`
///   denominator in `earth::transport_rate` is within 3.7% of zero. Whether the attitude
///   matrix stays finite through that is a floating-point accident, not a property: it is
///   finite on Linux and macOS and was not on Windows until #302 removed nine orders of
///   magnitude of vertical divergence from the initial attitude.
///
/// So the baseline is truncated, and the window is bounded from both sides by quantities that
/// do not depend on what the run currently prints.
///
/// **Lower bound: the baseline must sit outside the band a healthy filter occupies.** Below
/// [`DEAD_RECKONING_DIVERGENCE_FLOOR_M`] = 400 m the baseline is inside the range a degraded
/// but not broken filter can reach, so comparing against it measures the harness rather than
/// the navigation -- at a 105 m baseline the ratio would be demanding better than 10.5 m,
/// which is the one-sample alignment floor over this window (mean ground speed across the
/// first 240 samples is 10.47 m/s at 1 Hz). Unaided, this recording crosses 400 m of
/// horizontal RMSE between samples 130 (381.1 m) and 140 (427.9 m).
///
/// **Upper bound: the baseline must stay inside the mechanization's stated domain.** The crate
/// documentation gives the local-level mechanization's altitude validity as
/// [-11,000 m, 30,000 m] -- the deepest ocean trench to the top of the band where a
/// local-level frame is still the right tool -- and `sim::health::HealthLimits` names the same
/// range. `StrapdownState::new` and `IMUQuality::auto_covariance` hard-refuse anything outside
/// +/-30,000 m. Unaided, this recording's dead-reckoned altitude passes -11,000 m at sample
/// 485 and -30,000 m at sample 766.
///
/// **The derived quantity is the range `[135, 485]` samples; 240 is a choice inside it.** Both
/// endpoints are measured against quantities that do not involve a filter, and the interior is
/// not further determined -- a log midpoint gives 256 and an arithmetic one 310, and nothing
/// distinguishes them. 240 is a round four minutes at this recording's 1 Hz, and holds 5.5x on
/// the divergence floor (2,210.8 m against 400 m) and 4.7x on the documented domain
/// (-2,361.7 m against -11,000 m), with 12.7x against the +/-30 km the code enforces. An
/// earlier draft claimed the point was determined by equating the two margins; it is not --
/// those are a horizontal-RMSE ratio and an altitude ratio, and where two incommensurable
/// ratios cross depends on the units each is written in.
///
/// Note what is absent from the derivation: the filters' own numbers. Over this window they
/// sit at 13.6-15.5 m and beat the baseline by 143-162x, and neither endpoint was read off
/// that. What the window *does* have to respect, once chosen, is that
/// [`DEAD_RECKONING_BEAT_FACTOR`] stays live at the resulting baseline -- see its derivation.
///
/// [`assert_baseline_window_is_1hz`] re-derives the sample rate from the data on every run, so
/// a dataset swap cannot silently turn 240 samples into a window of some other duration.
const DEAD_RECKONING_BASELINE_SAMPLES: usize = 240;

/// Altitude band, in metres, the crate documents the local-level mechanization to be valid over.
///
/// The crate documentation gives it as [-11,000 m, 30,000 m]: the deepest ocean trenches are
/// about 11 km below mean sea level, and above 30 km an Earth-centred frame is the usual choice
/// instead of a local-level one. `sim::health::HealthLimits` names the same band. This is what
/// the crate *documents*; [`STATE_CONSTRUCTOR_ALTITUDE_LIMIT_M`] is the looser band it actually
/// enforces, which is why a solution can be inside the second and outside the first.
const MECHANIZATION_VALID_ALTITUDE_M: (f64, f64) = (-11_000.0, 30_000.0);

/// Altitude magnitude, in metres, outside which the crate refuses to build a state at all.
///
/// `StrapdownState::new` and `IMUQuality::auto_covariance` both return
/// `StrapdownError::OutOfRange` for an altitude outside +/-30,000 m. `mechanize` has no such
/// check and will propagate anywhere, which is the gap `test_dead_reckoning_on_real_data`
/// documents at the end (#299).
const STATE_CONSTRUCTOR_ALTITUDE_LIMIT_M: f64 = 30_000.0;

/// Horizontal RMSE, in metres, past which unaided dead reckoning has diverged far enough for
/// a comparison against it to say anything.
///
/// A tripwire on the *baseline*, not on the filter. It must sit above
/// [`MAX_HORIZONTAL_RMSE_M`], or the comparison is asking a filter to beat a number inside the
/// band healthy filters already occupy; one order of magnitude above it is the round figure.
/// The two harness terms -- the receiver's own 3.81 m reported accuracy
/// ([`GNSS_REPORTED_HORIZONTAL_ACCURACY_M`]) and the ~10.5 m one-sample alignment error over
/// this window -- put 400 m at ~38x the larger, which no plausible timestamp misalignment
/// produces.
///
/// Its job is to fail when dead reckoning stops diverging, so that
/// [`DEAD_RECKONING_BASELINE_SAMPLES`] gets re-derived rather than the comparison quietly
/// going vacuous the way #299 describes.
const DEAD_RECKONING_DIVERGENCE_FLOOR_M: f64 = 10.0 * MAX_HORIZONTAL_RMSE_M;

/// Factor by which an aided filter must beat unaided dead reckoning over the baseline window.
///
/// **Separate from [`DEAD_RECKONING_DIVERGENCE_FLOOR_M`] on purpose, and the reason is the
/// whole point of #299.** One constant used for both jobs makes the ratio assertion dead code:
/// if the same factor `k` defines the floor as `k * MAX_HORIZONTAL_RMSE_M` *and* the required
/// ratio, then `baseline > k * ceiling` and `filter < ceiling` together imply
/// `filter * k < baseline` arithmetically, and the ratio's failure region is empty. The first
/// draft of this fix did exactly that, replacing a threshold that could not fail at 6.57e6 m
/// with one that could not fail at 221 m.
///
/// So the ratio is derived from where it becomes *live* instead. It has content only when it
/// is stricter than the ceiling standing beside it, i.e. when
/// `baseline / factor < MAX_HORIZONTAL_RMSE_M`. The baseline over this window is 2,210.81 m
/// -- a property of unaided dead reckoning on this recording, measured with no filter involved
/// -- so the factor must exceed 2210.81 / 40 = 55.3 to assert anything at all. Above that it
/// is bounded by not writing the observation down: the filters achieve 143-162x.
///
/// 75 sits above the first bound and less than half of the second. Concretely it fires when a
/// filter passes 29.5 m, where the ceiling alone would not fire until 40 m, and the healthy
/// filters sit at 13.6-15.5 m -- so roughly 1.9x of margin on a live assertion.
const DEAD_RECKONING_BEAT_FACTOR: f64 = 75.0;

/// Factor by which the ESKF tests inflate the crate default's velocity, attitude and bias
/// process noise.
///
/// Historically described as "tuned to balance stability and accuracy; higher values prevent
/// divergence". It is kept at its historical value because nothing in #308 bears on it: the
/// velocity, attitude and bias entries were always in the states' own units.
const ESKF_PROCESS_NOISE_SCALE: f64 = 8.0;

/// ESKF-specific process noise covariance (15-state): the crate default with its velocity,
/// attitude and bias entries scaled by [`ESKF_PROCESS_NOISE_SCALE`].
///
/// **The position block is deliberately *not* scaled.** The array this replaces was written
/// as `8e-6, 8e-6, 8e-6, 8e-3, ...`, i.e. eight times every entry of the old default -- which
/// means its horizontal terms were eight times #308's 6.4 km per-step standard deviation. The
/// question the fix raises is whether that 8x should follow the position block into a regime
/// where the position block finally matters, and it should not, because it was never tuning
/// it. Measured on `test_data.csv` through `initialize_eskf`, in the **old** regime:
///
/// | position block | rest | NIS median | NIS mean |
/// |---|---|---|---|
/// | 8x | 8x | 4.174 | 11.482 |
/// | 1x | 8x | 4.174 | 11.482 |
///
/// Identical to three decimals: with the horizontal terms already at kilometre scale, eight
/// times more made no difference any measurement could see, so whatever the 8x was balancing,
/// it was not position. In the **new** regime the same pair differs (NIS median 4.441 against
/// 5.129, final horizontal uncertainty 1.076 m against 0.795 m), which is exactly why
/// propagating the multiplier onto the position block would be importing a number into a
/// place it was never measured in -- the thing #288 closed on.
///
/// Both tables were taken through `initialize_eskf` while it still built the pre-#308 $P_0$;
/// the $P_0$ fix moves them slightly. It does not touch the argument, which turns on the two
/// process-noise rows being *equal* in the old regime and unequal in the new one, and $P_0$
/// is held fixed within each comparison.
const ESKF_PROCESS_NOISE: [f64; 15] = {
    let mut scaled = DEFAULT_PROCESS_NOISE_DENSITY;
    let mut i = 3;
    while i < scaled.len() {
        scaled[i] *= ESKF_PROCESS_NOISE_SCALE;
        i += 1;
    }
    scaled
};

/// ESKF-specific initial covariance (15-state): [`DEFAULT_INITIAL_COVARIANCE`] with its
/// velocity, attitude and bias entries scaled by [`ESKF_PROCESS_NOISE_SCALE`].
///
/// Historically "higher uncertainty (8x default) for stability", written as `8e-6, 8e-6, 8.0,
/// 0.8, ...` -- eight times every entry of the old default. **The position block is
/// deliberately no longer scaled**, for exactly the reason [`ESKF_PROCESS_NOISE`] gives at
/// length: the 8x cannot have been tuning a horizontal term that was already 6367 m, so
/// carrying it onto a horizontal term that finally means something would be importing a
/// multiplier into a place it was never measured in (#288). What the 8x was tuning was the
/// velocity, attitude and bias block, and that is where it stays.
const ESKF_INITIAL_COVARIANCE: [f64; 15] = {
    let mut scaled = DEFAULT_INITIAL_COVARIANCE;
    let mut i = 3;
    while i < scaled.len() {
        scaled[i] *= ESKF_PROCESS_NOISE_SCALE;
        i += 1;
    }
    scaled
};
/// Anti-windup caps the ESKF clamps its bias estimates to (`kalman.rs`, #286).
///
/// Orders of magnitude above legitimate consumer-MEMS turn-on biases (~0.1 m/s^2,
/// ~0.01 rad/s) and far below the runaway values a persistently faulty aiding sensor
/// otherwise produces -- 9.2 m/s^2 and 6.5 rad/s on this very dataset before #286.
const MAX_ACCEL_BIAS_MPS2: f64 = 2.0;
const MAX_GYRO_BIAS_RPS: f64 = 0.05;

/// Assert the bias estimates stayed bounded at *every* sample, not just the last one.
///
/// #258 asks for boundedness across the whole run, and the distinction matters: checking
/// only the final estimate cannot tell a filter whose biases never moved from one that
/// wound up to a runaway value mid-run and was dragged back by a later fix. The clamp in
/// `inject_error_state` guarantees the final value regardless, so a final-sample assertion
/// tests the clamp rather than the filter.
fn assert_bias_estimates_bounded(results: &[NavigationResult], context: &str) {
    let accel_peak = |pick: fn(&NavigationResult) -> f64| {
        results
            .iter()
            .map(|r| pick(r).abs())
            .fold(0.0_f64, f64::max)
    };
    println!(
        "{context}: peak |bias| accel=[{:.4}, {:.4}, {:.4}] m/s^2, gyro=[{:.5}, {:.5}, {:.5}] rad/s",
        accel_peak(|r| r.acc_bias_x),
        accel_peak(|r| r.acc_bias_y),
        accel_peak(|r| r.acc_bias_z),
        accel_peak(|r| r.gyro_bias_x),
        accel_peak(|r| r.gyro_bias_y),
        accel_peak(|r| r.gyro_bias_z),
    );

    for (i, result) in results.iter().enumerate() {
        for (axis, bias) in [
            ("x", result.acc_bias_x),
            ("y", result.acc_bias_y),
            ("z", result.acc_bias_z),
        ] {
            assert!(
                bias.is_finite() && bias.abs() <= MAX_ACCEL_BIAS_MPS2,
                "{context}: accel bias {axis} left its physical bound at sample {i} of {}: \
                 {bias:.3} m/s^2 exceeds {MAX_ACCEL_BIAS_MPS2} (see #258, #286)",
                results.len()
            );
        }
        for (axis, bias) in [
            ("x", result.gyro_bias_x),
            ("y", result.gyro_bias_y),
            ("z", result.gyro_bias_z),
        ] {
            assert!(
                bias.is_finite() && bias.abs() <= MAX_GYRO_BIAS_RPS,
                "{context}: gyro bias {axis} left its physical bound at sample {i} of {}: \
                 {bias:.4} rad/s exceeds {MAX_GYRO_BIAS_RPS} (see #258, #286)",
                results.len()
            );
        }
    }
}

/// Reference attitude for a record as (roll, pitch, yaw) in radians, nalgebra XYZ convention.
///
/// Delegates to [`TestDataRecord::attitude`], which reads the quaternion rather than the
/// record's `roll`/`pitch`/`yaw` columns -- those are a different convention and disagree by
/// more than a sign (#302). Scoring an estimate against this means the reference it is
/// *measured* against is the same quantity [`create_initial_state`] *seeded* it from.
fn reference_attitude(record: &TestDataRecord) -> (f64, f64, f64) {
    record.attitude().euler_angles()
}

/// Geodesic attitude error between an estimate and its reference, in radians.
///
/// The rotation angle of `R_est^T * R_ref`: the single rotation that carries one attitude
/// onto the other, and the only attitude error metric that does not depend on a choice of
/// Euler sequence. Per-axis errors are reported alongside it because they say *which* axis
/// is at fault, but this is the quantity that is convention-free.
fn attitude_error_angle(result: &NavigationResult, record: &TestDataRecord) -> f64 {
    let estimate = Rotation3::from_euler_angles(result.roll, result.pitch, result.yaw);
    (estimate.inverse() * record.attitude()).angle()
}

/// Mean, min, median, max and RMS of one error channel, in that channel's own units.
#[derive(Debug, Clone, Copy, Default)]
struct Summary {
    mean: f64,
    min: f64,
    median: f64,
    max: f64,
    rms: f64,
}

/// Reduce one channel of per-sample errors to its summary statistics.
///
/// Every error channel reduces identically, so the reduction lives here rather than being
/// written out once per channel. An empty sample yields all zeros, which is what the
/// callers' `ErrorStats::new()` default already encoded.
fn summarize(samples: &[f64]) -> Summary {
    if samples.is_empty() {
        return Summary::default();
    }
    let count = samples.len() as f64;
    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mid = sorted.len() / 2;
    Summary {
        mean: samples.iter().sum::<f64>() / count,
        min: sorted[0],
        median: if sorted.len().is_multiple_of(2) {
            f64::midpoint(sorted[mid - 1], sorted[mid])
        } else {
            sorted[mid]
        },
        max: sorted[sorted.len() - 1],
        rms: (samples.iter().map(|e| e.powi(2)).sum::<f64>() / count).sqrt(),
    }
}

/// Error statistics for a navigation solution
#[allow(
    clippy::struct_field_names,
    reason = "every field is an error metric; dropping the `_error` suffix would make `mean_horizontal` and `rms_altitude` ambiguous against the non-error quantities in scope"
)]
#[derive(Debug, Clone)]
struct ErrorStats {
    /// Mean horizontal position error (meters)
    mean_horizontal_error: f64,
    /// Minimum horizontal position error (meters)
    min_horizontal_error: f64,
    /// Median horizontal position error (meters)
    median_horizontal_error: f64,
    /// Maximum horizontal position error (meters)
    max_horizontal_error: f64,
    /// Root mean square horizontal position error (meters)
    rms_horizontal_error: f64,
    /// Mean altitude error (meters)
    mean_altitude_error: f64,
    /// Minimum altitude error (meters)
    min_altitude_error: f64,
    /// Median altitude error (meters)
    median_altitude_error: f64,
    /// Maximum altitude error (meters)
    max_altitude_error: f64,
    /// Root mean square altitude error (meters)
    rms_altitude_error: f64,
    /// Mean velocity north error (m/s)
    mean_velocity_north_error: f64,
    /// Mean velocity east error (m/s)
    mean_velocity_east_error: f64,
    /// Mean velocity down error (m/s)
    mean_velocity_vertical_error: f64,
    /// Root mean square roll error (radians)
    rms_roll_error: f64,
    /// Root mean square pitch error (radians)
    rms_pitch_error: f64,
    /// Root mean square yaw error (radians)
    rms_yaw_error: f64,
    /// Maximum absolute roll error (radians)
    max_roll_error: f64,
    /// Maximum absolute pitch error (radians)
    max_pitch_error: f64,
    /// Maximum absolute yaw error (radians)
    max_yaw_error: f64,
    /// Root mean square geodesic attitude error (radians)
    rms_attitude_error: f64,
}

impl ErrorStats {
    /// Create a new `ErrorStats` with all zeros
    const fn new() -> Self {
        Self {
            mean_horizontal_error: 0.0,
            min_horizontal_error: 0.0,
            median_horizontal_error: 0.0,
            max_horizontal_error: 0.0,
            rms_horizontal_error: 0.0,
            mean_altitude_error: 0.0,
            min_altitude_error: 0.0,
            median_altitude_error: 0.0,
            max_altitude_error: 0.0,
            rms_altitude_error: 0.0,
            mean_velocity_north_error: 0.0,
            mean_velocity_east_error: 0.0,
            mean_velocity_vertical_error: 0.0,
            rms_roll_error: 0.0,
            rms_pitch_error: 0.0,
            rms_yaw_error: 0.0,
            max_roll_error: 0.0,
            max_pitch_error: 0.0,
            max_yaw_error: 0.0,
            rms_attitude_error: 0.0,
        }
    }
}

/// Compute error metrics between navigation results and GNSS truth data
///
/// This function calculates various error metrics by comparing the filter's navigation
/// solution against GNSS measurements treated as ground truth. It computes:
/// - Horizontal position error using haversine distance
/// - Altitude error as simple difference
/// - Velocity component errors
///
/// # Arguments
/// - `results` - Navigation results from filter (estimated state)
/// - `records` - Test data records containing GNSS measurements (truth)
///
/// # Returns
/// `ErrorStats` containing mean, max, and RMS errors for various quantities
fn compute_error_metrics(results: &[NavigationResult], records: &[TestDataRecord]) -> ErrorStats {
    let mut horizontal_errors = Vec::new();
    let mut altitude_errors = Vec::new();
    let mut velocity_north_errors = Vec::new();
    let mut velocity_east_errors = Vec::new();
    let mut velocity_vertical_errors = Vec::new();
    let mut roll_errors = Vec::new();
    let mut pitch_errors = Vec::new();
    let mut yaw_errors = Vec::new();
    let mut attitude_errors = Vec::new();

    // Match navigation results to GNSS measurements by timestamp
    for (i, result) in results.iter().enumerate() {
        // Find matching record by timestamp
        if let Some(record) = records.iter().find(|r| r.time == result.timestamp) {
            // Skip if GNSS data is invalid (NaN)
            if record.latitude.is_nan()
                || record.longitude.is_nan()
                || record.altitude.is_nan()
                || record.horizontal_accuracy.is_nan()
            {
                continue;
            }

            // Debug first few values
            if i < 3 {
                println!(
                    "Record {}: result.lat={:.6}, result.lon={:.6}, result.alt={:.2}",
                    i, result.latitude, result.longitude, result.altitude
                );
                println!(
                    "Record {}: record.lat={:.6}, record.lon={:.6}, record.alt={:.2}",
                    i, record.latitude, record.longitude, record.altitude
                );
            }

            // Compute horizontal position error using haversine distance
            // NavigationResult stores lat/lon in degrees, TestDataRecord also in degrees
            let horizontal_error = haversine_distance(
                result.latitude.to_radians(),
                result.longitude.to_radians(),
                record.latitude.to_radians(),
                record.longitude.to_radians(),
            );

            if i < 3 {
                println!("Record {i}: horizontal_error={horizontal_error:.2}m");
            }

            // Skip invalid errors (NaN or Inf)
            if !horizontal_error.is_finite() {
                if i < 10 || horizontal_errors.len() < 10 {
                    println!("WARNING: Skipping non-finite horizontal_error at index {i}");
                }
                continue;
            }

            horizontal_errors.push(horizontal_error);

            // Compute altitude error
            let altitude_error = (result.altitude - record.altitude).abs();
            if altitude_error.is_finite() {
                altitude_errors.push(altitude_error);
            }

            // Compute velocity errors
            // Note: GNSS provides speed and bearing, need to convert to N-E components
            let gnss_vel_north = record.speed * record.bearing.to_radians().cos();
            let gnss_vel_east = record.speed * record.bearing.to_radians().sin();

            let vn_err = (result.velocity_north - gnss_vel_north).abs();
            let ve_err = (result.velocity_east - gnss_vel_east).abs();
            let vd_err = result.velocity_vertical.abs();

            if vn_err.is_finite() {
                velocity_north_errors.push(vn_err);
            }
            if ve_err.is_finite() {
                velocity_east_errors.push(ve_err);
            }
            if vd_err.is_finite() {
                velocity_vertical_errors.push(vd_err);
            }

            // Attitude errors, wrapped so a branch-cut crossing is not counted as a full turn.
            let (reference_roll, reference_pitch, reference_yaw) = reference_attitude(record);
            let roll_err = wrap_to_pi(result.roll - reference_roll).abs();
            let pitch_err = wrap_to_pi(result.pitch - reference_pitch).abs();
            let yaw_err = wrap_to_pi(result.yaw - reference_yaw).abs();

            if roll_err.is_finite() && pitch_err.is_finite() && yaw_err.is_finite() {
                roll_errors.push(roll_err);
                pitch_errors.push(pitch_err);
                yaw_errors.push(yaw_err);
            }

            let attitude_err = attitude_error_angle(result, record);
            if attitude_err.is_finite() {
                attitude_errors.push(attitude_err);
            }
        }
    }

    // Compute statistics
    let mut stats = ErrorStats::new();

    println!("Collected {} horizontal errors", horizontal_errors.len());
    if horizontal_errors.len() > 10 {
        println!(
            "Last 10 horizontal errors: {:?}",
            &horizontal_errors[horizontal_errors.len() - 10..]
        );
    }

    let horizontal = summarize(&horizontal_errors);
    stats.mean_horizontal_error = horizontal.mean;
    stats.min_horizontal_error = horizontal.min;
    stats.median_horizontal_error = horizontal.median;
    stats.max_horizontal_error = horizontal.max;
    stats.rms_horizontal_error = horizontal.rms;

    let altitude = summarize(&altitude_errors);
    stats.mean_altitude_error = altitude.mean;
    stats.min_altitude_error = altitude.min;
    stats.median_altitude_error = altitude.median;
    stats.max_altitude_error = altitude.max;
    stats.rms_altitude_error = altitude.rms;

    stats.mean_velocity_north_error = summarize(&velocity_north_errors).mean;
    stats.mean_velocity_east_error = summarize(&velocity_east_errors).mean;
    stats.mean_velocity_vertical_error = summarize(&velocity_vertical_errors).mean;

    let roll = summarize(&roll_errors);
    let pitch = summarize(&pitch_errors);
    let yaw = summarize(&yaw_errors);
    stats.rms_roll_error = roll.rms;
    stats.rms_pitch_error = pitch.rms;
    stats.rms_yaw_error = yaw.rms;
    stats.max_roll_error = roll.max;
    stats.max_pitch_error = pitch.max;
    stats.max_yaw_error = yaw.max;
    stats.rms_attitude_error = summarize(&attitude_errors).rms;

    stats
}

/// Load test data from the provided CSV file
///
/// # Arguments
/// - `path` - Path to the CSV file containing test data
///
/// # Returns
/// Vector of `TestDataRecord` instances
fn load_test_data(path: &Path) -> Vec<TestDataRecord> {
    TestDataRecord::from_csv(path)
        .unwrap_or_else(|_| panic!("Failed to load test data from CSV: {}", path.display()))
}

/// Check that the baseline window is the duration [`DEAD_RECKONING_BASELINE_SAMPLES`] assumes.
///
/// That constant is derived in *seconds* -- both of its bounds are statements about how far an
/// unaided solution drifts in a given amount of time -- but it is applied in *samples*, and
/// the two are only interchangeable because this recording is 1 Hz. Nothing else in the file
/// enforces that. Replace `test_data.csv` with a 50 Hz log and 240 samples silently becomes a
/// 4.8 s window, which sits far under the alignment floor and would flip the comparison
/// without any assertion firing. So re-derive the rate from the data, the way
/// [`assert_reference_accuracy_matches_dataset`] re-derives the reported accuracies.
///
/// The 5% tolerance is there to survive a dropped sample or a timestamp rounded to the second,
/// not to accommodate a different rate: the nearest other plausible rate is 2 Hz, which is a
/// factor of two away.
fn assert_baseline_window_is_1hz(records: &[TestDataRecord], window: usize) {
    let span_s = (records[window - 1].time - records[0].time).num_seconds() as f64;
    let expected_s = (window - 1) as f64;
    assert!(
        (span_s - expected_s).abs() <= 0.05 * expected_s,
        "DEAD_RECKONING_BASELINE_SAMPLES is derived in seconds and applied in samples, which \
         only works at this recording's 1 Hz: {window} samples should span ~{expected_s:.0} s \
         but span {span_s:.0} s. If the dataset has been replaced, re-derive the window length \
         from the new rate rather than keeping the sample count."
    );
}

/// Check that this recording really is the ENU [`IS_ENU_TEST_DATA`] claims it is.
///
/// Re-derived from the file on every run, the way [`assert_reference_accuracy_matches_dataset`]
/// re-derives the reported accuracies, so the constant cannot go stale if `test_data.csv` is
/// replaced with a NED log -- which would otherwise turn every number in this file into a 2 g
/// integration that still produced a plausible-looking CSV (#296).
///
/// Both halves matter. The first says the guard accepts the frame this file declares; the
/// second says it *rejects* the other one, which is what distinguishes a working discriminator
/// from one that fails open on everything.
fn assert_test_data_is_enu(records: &[TestDataRecord]) {
    assert!(
        check_declared_frame(records, IS_ENU_TEST_DATA).is_ok(),
        "test_data.csv no longer reads as ENU, which every dead-reckoning and filter bound in \
         this file assumes. If the dataset has been replaced, set IS_ENU_TEST_DATA to match \
         it and re-baseline the error statistics -- they are not comparable across a frame \
         change."
    );
    assert!(
        check_declared_frame(records, !IS_ENU_TEST_DATA).is_err(),
        "sim::check_declared_frame accepted test_data.csv as both ENU and NED, so it is not \
         discriminating and the frame pinning in this file is vacuous."
    );
}

/// Dead-reckon the first [`DEAD_RECKONING_BASELINE_SAMPLES`] records and score them.
///
/// Returns the solution, its error statistics and the window length actually used, so a caller
/// can index the solution (the gravity-cancellation guard in `test_dead_reckoning_on_real_data`
/// needs `results[1]`) and can slice its own filter output to the same window. Every comparison
/// against unaided dead reckoning in this file goes through here, so there is exactly one
/// baseline definition to re-derive if the dataset or the mechanization changes.
fn dead_reckoning_baseline(
    records: &[TestDataRecord],
) -> (Vec<NavigationResult>, ErrorStats, usize) {
    let window = records.len().min(DEAD_RECKONING_BASELINE_SAMPLES);
    assert_baseline_window_is_1hz(records, window);
    assert_test_data_is_enu(records);
    // `test_data.csv` is a Sensor Logger export, so ENU: at rest its specific force lands on
    // the device's up-axis at +9.72 m/s^2. NED here would be rejected by
    // `sim::check_declared_frame` rather than silently mechanized at 2 g (#296).
    let results = dead_reckoning(&records[..window], IS_ENU_TEST_DATA).unwrap();
    let stats = compute_error_metrics(&results, &records[..window]);
    (results, stats, window)
}

/// Assert that an aided filter beats the unaided baseline by [`DEAD_RECKONING_BEAT_FACTOR`].
///
/// Both arguments must be scored over the **same** window. That is not pedantry: over the
/// 240-sample baseline window the filters score 13.6-15.5 m, and over the full 5,366-sample
/// recording they score 23.5 m, so scoring a filter on the full run against a truncated
/// baseline would inflate the filter's number by ~60% and compare two different things.
///
/// Three assertions, in the order a failure is easiest to read:
///
/// 1. The baseline really did diverge. This is the guard that keeps the comparison from
///    becoming vacuous again, and it is deliberately a *tripwire*: it fires if unaided dead
///    reckoning on this dataset ever gets more than ~5.5x better, which is exactly what a
///    vertical-channel damping change or a frame fix in `sim::dead_reckoning` would do. That
///    is a library improvement, not a regression -- see the failure message.
/// 2. The filter is inside the absolute operating bound. Necessary because "better than dead
///    reckoning" is not evidence of a working filter (#307).
/// 3. The ratio itself.
fn assert_beats_dead_reckoning(name: &str, filter: &ErrorStats, baseline: &ErrorStats) {
    let divergence_floor = DEAD_RECKONING_DIVERGENCE_FLOOR_M;
    assert!(
        baseline.rms_horizontal_error > divergence_floor,
        "unaided dead reckoning over the {DEAD_RECKONING_BASELINE_SAMPLES}-sample baseline \
         window reached only {:.2} m, under the {divergence_floor:.0} m this comparison needs \
         to say anything a filter's own {MAX_HORIZONTAL_RMSE_M} m ceiling does not. If dead \
         reckoning has legitimately improved, lengthen DEAD_RECKONING_BASELINE_SAMPLES using \
         the derivation in its documentation -- do not delete this assertion, it is what stops \
         the comparison going vacuous again (#299).",
        baseline.rms_horizontal_error
    );
    assert!(
        filter.rms_horizontal_error < MAX_HORIZONTAL_RMSE_M,
        "{name} RMS horizontal error over the baseline window is {:.2} m, past the \
         {MAX_HORIZONTAL_RMSE_M} m ceiling every healthy filter on this dataset holds",
        filter.rms_horizontal_error
    );
    assert!(
        filter.rms_horizontal_error * DEAD_RECKONING_BEAT_FACTOR < baseline.rms_horizontal_error,
        "{name} should beat unaided dead reckoning by {DEAD_RECKONING_BEAT_FACTOR}x over the \
         same window; {name}: {:.2} m, dead reckoning: {:.2} m ({:.1}x)",
        filter.rms_horizontal_error,
        baseline.rms_horizontal_error,
        baseline.rms_horizontal_error / filter.rms_horizontal_error
    );
}

/// Create an initial state from the first test data record
///
/// # Arguments
/// - `first_record` - The first test data record
///
/// # Returns
/// `InitialState` for filter initialization
fn create_initial_state(first_record: &TestDataRecord) -> InitialState {
    // NOTE: Test data from Sensor Logger has latitude/longitude in degrees and roll/pitch/yaw
    // in a different Euler convention than nalgebra's XYZ, so `reference_attitude` reads the
    // quaternion instead -- see its documentation.
    let (roll, pitch, yaw) = reference_attitude(first_record);
    // Through the record's own accessor rather than re-deriving it: `bearing` is degrees, and
    // open-coding the conversion here is how this helper could drift away from the production
    // initialiser it is supposed to mirror.
    let (northward_velocity, eastward_velocity) = first_record.ground_track_velocity();

    InitialState {
        latitude: first_record.latitude.to_radians(),
        longitude: first_record.longitude.to_radians(),
        altitude: first_record.altitude,
        northward_velocity,
        eastward_velocity,
        vertical_velocity: 0.0,
        roll,
        pitch,
        yaw,
        in_degrees: false, // All angles now in radians
        // ENU on purpose; see `TEST_DATA_IS_ENU`, which is also what the event stream's
        // magnetometer heading is built against. The two must agree (#305).
        is_enu: TEST_DATA_IS_ENU,
    }
}

/// Create a nominal `StrapdownState` from the first test data record
fn create_nominal_state(first_record: &TestDataRecord) -> StrapdownState {
    let (roll, pitch, yaw) = reference_attitude(first_record);
    let (velocity_north, velocity_east) = first_record.ground_track_velocity();

    StrapdownState {
        latitude: first_record.latitude.to_radians(),
        longitude: first_record.longitude.to_radians(),
        altitude: first_record.altitude,
        velocity_north,
        velocity_east,
        velocity_vertical: 0.0,
        attitude: Rotation3::from_euler_angles(roll, pitch, yaw),
        // ENU on purpose; see `TEST_DATA_IS_ENU`, which is also what the event stream's
        // magnetometer heading is built against. The two must agree (#305).
        is_enu: TEST_DATA_IS_ENU,
    }
}

/// Particle count for RBPF tests running against undegraded GNSS.
///
/// This matches `RbpfConfig::default()`. A sweep over `core/tests/test_data.csv`
/// (seed 42) shows accuracy here is flat in particle count, so the previous value
/// of 5000 bought nothing but runtime:
///
/// | particles | median horiz | rms horiz | wall  |
/// |-----------|--------------|-----------|-------|
/// | 250       | 23.86 m      | 24.41 m   | 12 s  |
/// | 500       | 23.67 m      | 24.19 m   | 23 s  |
/// | 5000      | 23.50 m      | 23.98 m   | 226 s |
///
/// The assertions below clear their thresholds by roughly 9x at this count.
const RBPF_PARTICLES: usize = 500;

/// Particle count for the degraded-GNSS RBPF test.
///
/// Kept at 5000 as the reference configuration: with the #267 fixes (wrapped
/// angular likelihoods, proposal matched to the fault scale) the error
/// decreases with particle count and then plateaus (250→2000: 150→125→109→111 m
/// median; the 2000-vs-1000 wiggle is within seed noise, measured at 15% spread
/// across seeds) and is robust across seeds (117-136 m at 500 particles), so
/// this asserts a bound rather than the old 5000-or-bust coincidence.
const RBPF_DEGRADED_PARTICLES: usize = 5000;

fn run_rbpf_with_cfg(
    records: &[TestDataRecord],
    cfg: &AidingConfig,
    rbpf_config: RbpfConfig,
) -> Vec<NavigationResult> {
    let stream = build_event_stream(records, cfg, TEST_DATA_IS_ENU).unwrap();

    let nominal = create_nominal_state(&records[0]);
    let mut rbpf = RaoBlackwellizedParticleFilter::new(nominal, rbpf_config).unwrap();

    let start_time = stream.start_time;
    let mut results: Vec<NavigationResult> = Vec::with_capacity(stream.events.len());
    let mut last_ts: Option<chrono::DateTime<chrono::Utc>> = None;

    for event in stream.events {
        let elapsed_s = match &event {
            Event::Imu { elapsed_s, .. } | Event::Measurement { elapsed_s, .. } => *elapsed_s,
        };
        let ts = start_time + chrono::Duration::milliseconds((elapsed_s * 1000.0).round() as i64);

        // Emit the row for the epoch that just ended before applying anything from this one,
        // matching `sim::run_closed_loop` and `sim::dead_reckoning` (#367). This loop is a
        // copy of the former and carried the same defect: the push sat below the `match`, so
        // a row labelled `t_k` held the state after `t_{k+1}`'s first event.
        if Some(ts) != last_ts {
            if let Some(prev_ts) = last_ts {
                let (mean, cov) = rbpf.estimate();
                results.push(NavigationResult::from_particle_filter(
                    &prev_ts, &mean, &cov,
                ));
            }
            last_ts = Some(ts);
        }

        match event {
            Event::Imu { dt_s, imu, .. } => rbpf.predict(&imu, dt_s).unwrap(),
            // `update` now reports an `UpdateOutcome`; this loop does not gate, so the
            // statistic is discarded rather than the arms being forced to agree on `()`.
            Event::Measurement { meas, .. } => {
                rbpf.update(meas.as_ref()).unwrap();
            }
        }
    }

    // Flush the final epoch; the boundary push only fires when a later timestamp arrives.
    if let Some(final_ts) = last_ts {
        let (mean, cov) = rbpf.estimate();
        results.push(NavigationResult::from_particle_filter(
            &final_ts, &mean, &cov,
        ));
    }

    results
}

fn run_rbpf(records: &[TestDataRecord]) -> Vec<NavigationResult> {
    let cfg = {
        let mut built = AidingConfig::default();
        built.scheduler = MeasurementScheduler::PassThrough;
        built.fault = GnssFaultModel::None;
        built
    };
    run_rbpf_with_cfg(records, &cfg, {
        let mut built = RbpfConfig::default();
        built.num_particles = RBPF_PARTICLES;
        built.seed = 42;
        built
    })
}

/// Test dead reckoning on real data to establish baseline
///
/// This test runs pure INS dead reckoning (no GNSS corrections) on real data and verifies that
/// it completes, that gravity cancels on the first step, and that the solution over the
/// baseline window the three comparison tests use is a usable one. It then makes the one
/// statement about the *full* 89-minute arc that holds on every platform -- see the bottom of
/// the test.
#[test]
fn test_dead_reckoning_on_real_data() {
    // Load test data
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let test_data_path = Path::new(manifest_dir).join("tests/test_data.csv");
    let records = load_test_data(&test_data_path);

    assert!(
        !records.is_empty(),
        "Test data should contain at least one record"
    );

    // Run dead reckoning over the baseline window -- the same arc the three
    // `*_outperforms_dead_reckoning` tests compare against. See
    // `DEAD_RECKONING_BASELINE_SAMPLES` for why it is not the whole recording.
    let (results, stats, window) = dead_reckoning_baseline(&records);

    // Verify results
    assert_eq!(
        results.len(),
        window,
        "Dead reckoning should produce one result per input record"
    );

    // Print statistics for reference
    println!("\n=== Dead Reckoning Error Statistics ({window}-sample baseline window) ===");
    println!(
        "Horizontal Error: mean={:.2}m, min={:.2}m, median={:.2}m, max={:.2}m, rms={:.2}m",
        stats.mean_horizontal_error,
        stats.min_horizontal_error,
        stats.median_horizontal_error,
        stats.max_horizontal_error,
        stats.rms_horizontal_error
    );
    println!(
        "Altitude Error: mean={:.2}m, min={:.2}m, median={:.2}m, max={:.2}m, rms={:.2}m",
        stats.mean_altitude_error,
        stats.min_altitude_error,
        stats.median_altitude_error,
        stats.max_altitude_error,
        stats.rms_altitude_error
    );
    println!(
        "Velocity Error: N={:.3}m/s, E={:.3}m/s, D={:.3}m/s",
        stats.mean_velocity_north_error,
        stats.mean_velocity_east_error,
        stats.mean_velocity_vertical_error
    );

    // Gravity must cancel on the very first propagation step.
    //
    // This is the one dead-reckoning quantity worth bounding, and the only reason it needs
    // bounding is that it was wrong. `dead_reckoning` built its initial attitude by feeding
    // the record's Euler fields to `Rotation3::from_euler_angles`, but those fields are a
    // different convention (see `TestDataRecord::attitude`), so gravity was rotated into the
    // horizontal axes and the vertical channel had nothing to cancel. One 1 s step then left
    // 9.31 m/s of vertical velocity -- a full uncancelled g -- and compounded from there to
    // 1.7e16 m of altitude by the end of the run, finite on Linux and over the edge into
    // `mechanize`'s non-finite check on Windows.
    //
    // The bound is physical: this recording starts with a near-stationary vehicle, so once
    // gravity is removed the residual vertical specific force is the vehicle's own motion
    // plus sensor error, far under 1 m/s^2. After one second that is well under 1 m/s. The
    // run sits at 0.075 m/s, so the limit carries ~13x margin while still catching the
    // 9.31 m/s failure by a factor of 9.
    //
    // Nothing else over the window is bounded by value on purpose. Unaided dead reckoning on
    // consumer-MEMS data genuinely diverges -- the vertical channel is unstable without aiding
    // and accel bias integrates as t^2 -- and inventing a ceiling for that would be fitting a
    // number, not deriving one. What the window is required to be is *usable*: finite, and
    // inside the altitude band the mechanization is documented over. Those two are asserted
    // below, and they are what the three `*_outperforms_dead_reckoning` comparisons need from
    // it. The full arc is a separate question, taken up at the end of this test.
    let first_step_vertical_velocity = results[1].velocity_vertical.abs();
    assert!(
        first_step_vertical_velocity < 1.0,
        "gravity should cancel on the first step, leaving |v_vertical| well under 1 m/s; \
         got {first_step_vertical_velocity:.4} m/s. A value near 9.8 means the initial \
         attitude is in the wrong Euler convention again."
    );

    // Over the baseline window the solution must be usable, and "usable" is more than finite:
    // `mechanize` carries no altitude guard of its own, so a solution can be perfectly finite
    // and still be somewhere the local-level mechanization is not defined. Assert both, and
    // report the index of the first offender -- a divergence that starts at sample 200 reads
    // very differently from one that starts at sample 1.
    let (min_valid_altitude, max_valid_altitude) = MECHANIZATION_VALID_ALTITUDE_M;
    for (index, result) in results.iter().enumerate() {
        assert!(
            result.latitude.is_finite()
                && result.longitude.is_finite()
                && result.altitude.is_finite(),
            "dead reckoning went non-finite at sample {index} of the {window}-sample baseline \
             window: lat={}, lon={}, alt={}",
            result.latitude,
            result.longitude,
            result.altitude
        );
        assert!(
            (min_valid_altitude..=max_valid_altitude).contains(&result.altitude),
            "the baseline window must stay inside the [{min_valid_altitude}, \
             {max_valid_altitude}] m altitude band this mechanization is documented over, or \
             the filters are being compared against a solution the model does not define; \
             sample {index} of {window} is at {:.1} m. Shorten DEAD_RECKONING_BASELINE_SAMPLES \
             using the derivation in its documentation.",
            result.altitude
        );
    }

    // The full 89-minute arc, and the one thing about it that is true on every platform.
    //
    // #299 was filed because these tests failed on windows-latest while passing on Linux and
    // macOS, all five with `NonFinite { what: "propagated attitude matrix" }`. The cause was
    // not platform-specific code. Unaided, this recording used to reach 1.7e16 m of altitude,
    // and whether an intermediate product of the attitude update overflows at that magnitude
    // comes down to evaluation order. #302 removed the uncancelled gravity responsible for
    // nine orders of magnitude of it -- that is the guard above -- and the arc now completes
    // finitely on all three platforms. But it completes by way of -6.14e6 m of altitude, and
    // at sample 3456 the `r_e + altitude` denominator in `earth::transport_rate` bottoms out
    // at 2.35e5 m against an `r_e` of 6.38e6 m: 3.7% of the way to a division by zero, which
    // another 3.7% of downward drift would close. Finiteness there is still an
    // arithmetic accident, so neither outcome is asserted, and both are accepted below.
    //
    // What holds regardless is #299's second observation. An unaided arc of this length does
    // not stay inside the mechanization's domain, and nothing in `mechanize` says so:
    // `StrapdownState::new` and `IMUQuality::auto_covariance` both refuse an altitude outside
    // +/-30 km as outside the band the model is valid over, while `mechanize` propagates
    // straight through it -- first at sample 766 of 5,366, with 4,600 samples still to run.
    // That is the reason the comparison tests score a truncated window rather than this arc.
    match dead_reckoning(&records, IS_ENU_TEST_DATA) {
        Ok(full) => {
            let first_out_of_domain = full.iter().position(|result| {
                !(-STATE_CONSTRUCTOR_ALTITUDE_LIMIT_M..=STATE_CONSTRUCTOR_ALTITUDE_LIMIT_M)
                    .contains(&result.altitude)
            });
            let index = first_out_of_domain.unwrap_or_else(|| {
                panic!(
                    "the full unaided arc stayed within +/-{STATE_CONSTRUCTOR_ALTITUDE_LIMIT_M} m \
                     of altitude for all {} samples. That is an improvement to \
                     `sim::dead_reckoning`, not a regression -- but it removes the upper bound \
                     DEAD_RECKONING_BASELINE_SAMPLES is derived from, so re-derive the window \
                     and rewrite this paragraph before relaxing the assertion.",
                    full.len()
                )
            });
            println!(
                "Full unaided arc: left the +/-{STATE_CONSTRUCTOR_ALTITUDE_LIMIT_M:.0} m \
                 mechanization domain at sample {index} of {} ({:.1} m), finishing at {:.3e} m",
                full.len(),
                full[index].altitude,
                full.last().unwrap().altitude
            );
        }
        // The other half of "true on every platform": nothing here requires the arc to
        // complete. This is the arm windows-latest took before #302, and the arm any platform
        // takes the moment `r_e + altitude` crosses zero.
        Err(StrapdownError::NonFinite { what }) => {
            println!(
                "Full unaided arc: mechanization reported a non-finite {what}, as it did on \
                 windows-latest before #302"
            );
        }
        Err(other) => panic!("unexpected dead-reckoning failure on the full arc: {other}"),
    }
}

/// Test UKF closed-loop filter on real data
///
/// This test runs a closed-loop UKF with GNSS measurements on real data and verifies that:
/// 1. The filter completes without errors
/// 2. Position errors remain bounded
/// 3. The filter performs better than dead reckoning
#[test]
fn test_ukf_closed_loop_on_real_data() {
    // Load test data
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let test_data_path = Path::new(manifest_dir).join("tests/test_data.csv");
    let records = load_test_data(&test_data_path);

    assert!(
        !records.is_empty(),
        "Test data should contain at least one record"
    );

    // Create initial state from first record
    let initial_state = create_initial_state(&records[0]);

    // Initialize UKF
    let imu_biases = vec![0.0; 6]; // Zero initial bias estimates
    let initial_covariance = DEFAULT_INITIAL_COVARIANCE.to_vec();

    let process_noise =
        DMatrix::from_diagonal(&DVector::from_vec(DEFAULT_PROCESS_NOISE_DENSITY.to_vec()));

    let mut ukf = UnscentedKalmanFilter::new(
        &initial_state,
        &imu_biases,
        None, // No measurement bias
        initial_covariance,
        process_noise,
        1e-3, // alpha
        2.0,  // beta
        0.0,  // kappa
    );

    // Create event stream with passthrough scheduler (all GNSS measurements used)
    let cfg = {
        let mut built = AidingConfig::default();
        built.scheduler = MeasurementScheduler::PassThrough;
        built.fault = GnssFaultModel::None;
        built
    };

    let stream = build_event_stream(&records, &cfg, TEST_DATA_IS_ENU).unwrap();

    // Run closed-loop filter
    let results =
        run_closed_loop(&mut ukf, stream, None, None).expect("Closed-loop filter should complete");

    // Verify results
    assert!(
        !results.is_empty(),
        "Closed-loop filter should produce results"
    );

    // Compute error metrics
    let stats = compute_error_metrics(&results, &records);

    // Print statistics
    println!("\n=== UKF Closed-Loop Error Statistics ===");
    println!(
        "Horizontal Error: mean={:.2}m, min={:.2}m, median={:.2}m, max={:.2}m, rms={:.2}m",
        stats.mean_horizontal_error,
        stats.min_horizontal_error,
        stats.median_horizontal_error,
        stats.max_horizontal_error,
        stats.rms_horizontal_error
    );
    println!(
        "Altitude Error: mean={:.2}m, min={:.2}m, median={:.2}m, max={:.2}m, rms={:.2}m",
        stats.mean_altitude_error,
        stats.min_altitude_error,
        stats.median_altitude_error,
        stats.max_altitude_error,
        stats.rms_altitude_error
    );
    println!(
        "Velocity Error: N={:.3}m/s, E={:.3}m/s, D={:.3}m/s",
        stats.mean_velocity_north_error,
        stats.mean_velocity_east_error,
        stats.mean_velocity_vertical_error
    );

    // Assert error bounds: they should hold for a healthy aided filter on this
    // data. The three filters agree at ~24 m horizontal rms, an order of
    // magnitude above the ~4.7 m fix noise floor because of dynamics, so the
    // bounds below are ~1.6x the observed healthy value: tight enough that any
    // divergence trips them instantly (dead reckoning is at 5e6 m), loose
    // enough that floating-point codegen differences between platforms cannot
    // cross them (see #288: the old 39.0 m bound carried only 2% margin).
    let rms_horizontal_limit = 40.0;
    let max_horizontal_limit = 60.0;
    let rms_altitude_limit = 50.0;
    let max_altitude_limit = 250.0;

    assert!(
        stats.rms_horizontal_error < rms_horizontal_limit,
        "UKF RMS horizontal error should be less than {:.2}m, got {:.2}m",
        rms_horizontal_limit,
        stats.rms_horizontal_error
    );
    assert!(
        stats.max_horizontal_error < max_horizontal_limit,
        "UKF maximum horizontal error should be less than {:.2}m, got {:.2}m",
        max_horizontal_limit,
        stats.max_horizontal_error
    );
    assert!(
        stats.rms_altitude_error < rms_altitude_limit,
        "UKF RMS altitude error should be less than {:.2}m, got {:.2}m",
        rms_altitude_limit,
        stats.rms_altitude_error
    );
    assert!(
        stats.max_altitude_error < max_altitude_limit,
        "UKF maximum altitude error should be less than {:.2}m, got {:.2}m",
        max_altitude_limit,
        stats.max_altitude_error
    );

    // Verify no NaN or infinite values in results
    for result in &results {
        assert!(
            result.latitude.is_finite(),
            "Latitude should be finite: {}",
            result.latitude
        );
        assert!(
            result.longitude.is_finite(),
            "Longitude should be finite: {}",
            result.longitude
        );
        assert!(
            result.altitude.is_finite(),
            "Altitude should be finite: {}",
            result.altitude
        );
        assert!(
            result.velocity_north.is_finite(),
            "Velocity north should be finite: {}",
            result.velocity_north
        );
        assert!(
            result.velocity_east.is_finite(),
            "Velocity east should be finite: {}",
            result.velocity_east
        );
        assert!(
            result.velocity_vertical.is_finite(),
            "Velocity down should be finite: {}",
            result.velocity_vertical
        );
    }
}

/// Test UKF with degraded GNSS (reduced update rate)
///
/// This test simulates degraded GNSS conditions with reduced update rate and verifies
/// that the filter still performs reasonably well, though with higher errors than full-rate GNSS.
#[test]
fn test_ukf_with_degraded_gnss() {
    // Load test data
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let test_data_path = Path::new(manifest_dir).join("tests/test_data.csv");
    let records = load_test_data(&test_data_path);

    assert!(
        !records.is_empty(),
        "Test data should contain at least one record"
    );

    // Create initial state from first record
    let initial_state = create_initial_state(&records[0]);

    // Initialize UKF
    let imu_biases = vec![0.0; 6];
    let initial_covariance = DEFAULT_INITIAL_COVARIANCE.to_vec();

    let process_noise =
        DMatrix::from_diagonal(&DVector::from_vec(DEFAULT_PROCESS_NOISE_DENSITY.to_vec()));

    let mut ukf = UnscentedKalmanFilter::new(
        &initial_state,
        &imu_biases,
        None,
        initial_covariance,
        process_noise,
        1e-3, // alpha
        2.0,  // beta
        0.0,  // kappa
    );

    // Create event stream with periodic scheduler (e.g., every 5 seconds)
    let cfg = {
        let mut built = AidingConfig::default();
        built.scheduler = MeasurementScheduler::FixedInterval {
            interval_s: 5.0,
            phase_s: 0.0,
        };
        built.fault = GnssFaultModel::None;
        built
    };

    let stream = build_event_stream(&records, &cfg, TEST_DATA_IS_ENU).unwrap();

    // Run closed-loop filter
    let results = run_closed_loop(&mut ukf, stream, None, None)
        .expect("Closed-loop filter with degraded GNSS should complete");

    // Compute error metrics
    let stats = compute_error_metrics(&results, &records);

    // Print statistics
    println!("\n=== UKF with Degraded GNSS (5s updates) Error Statistics ===");
    println!(
        "Horizontal Error: mean={:.2}m, min={:.2}m, median={:.2}m, max={:.2}m, rms={:.2}m",
        stats.mean_horizontal_error,
        stats.min_horizontal_error,
        stats.median_horizontal_error,
        stats.max_horizontal_error,
        stats.rms_horizontal_error
    );
    println!(
        "Altitude Error: mean={:.2}m, min={:.2}m, median={:.2}m, max={:.2}m, rms={:.2}m",
        stats.mean_altitude_error,
        stats.min_altitude_error,
        stats.median_altitude_error,
        stats.max_altitude_error,
        stats.rms_altitude_error
    );

    // Error bounds should be looser than full-rate GNSS but still reasonable
    assert!(
        stats.rms_horizontal_error < 50.0,
        "RMS horizontal error with degraded GNSS should be less than 50m, got {:.2}m",
        stats.rms_horizontal_error
    );

    assert!(
        stats.max_horizontal_error < 400.0,
        "Maximum horizontal error with degraded GNSS should be less than 400m, got {:.2}m",
        stats.max_horizontal_error
    );

    // Verify no invalid values
    for result in &results {
        assert!(result.latitude.is_finite());
        assert!(result.longitude.is_finite());
        assert!(result.altitude.is_finite());
    }
}

/// Test that closed-loop UKF outperforms dead reckoning
///
/// This test runs both dead reckoning and UKF on the same data and verifies that
/// the UKF produces lower errors than dead reckoning, demonstrating the benefit
/// of GNSS-aided navigation.
#[test]
fn test_ukf_outperforms_dead_reckoning() {
    // Load test data
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let test_data_path = Path::new(manifest_dir).join("tests/test_data.csv");
    let records = load_test_data(&test_data_path);

    assert!(
        !records.is_empty(),
        "Test data should contain at least one record"
    );

    // Run dead reckoning over the baseline window (see `DEAD_RECKONING_BASELINE_SAMPLES`)
    let (_, dr_stats, window) = dead_reckoning_baseline(&records);

    // Run UKF
    let initial_state = create_initial_state(&records[0]);
    let imu_biases = vec![0.0; 6];
    // The same diagonal every other test in this file uses. It was written out here as a
    // second copy of the literals, which is how it kept #308's `1e-6` horizontal entries
    // after the named constant above was corrected.
    let initial_covariance = DEFAULT_INITIAL_COVARIANCE.to_vec();
    let process_noise =
        DMatrix::from_diagonal(&DVector::from_vec(DEFAULT_PROCESS_NOISE_DENSITY.to_vec()));

    let mut ukf = UnscentedKalmanFilter::new(
        &initial_state,
        &imu_biases,
        None,
        initial_covariance,
        process_noise,
        1e-3,
        2.0,
        0.0,
    );

    let scheduler = MeasurementScheduler::PassThrough;
    let fault_model = GnssFaultModel::None;
    let cfg = {
        let mut built = AidingConfig::default();
        built.scheduler = scheduler;
        built.fault = fault_model;
        built
    };
    let stream = build_event_stream(&records, &cfg, TEST_DATA_IS_ENU).unwrap();

    // The filter runs the whole stream -- that coverage is worth keeping, and the absolute
    // ceiling below is asserted on it -- but the *comparison* is scored over the baseline
    // window, on the same samples as the baseline. Scoring the filter over the full run
    // against a truncated baseline would compare two different things, and not in the
    // filter's favour: these filters score 13.6-15.5 m over the head window and 23.5 m over
    // the full run, because the full run includes dynamics the head window does not.
    let ukf_results = run_closed_loop(&mut ukf, stream, None, None).expect("UKF should complete");
    let ukf_stats = compute_error_metrics(&ukf_results, &records);
    let ukf_window_stats = compute_error_metrics(&ukf_results[..window], &records[..window]);

    // Print comparison
    println!("\n=== Performance Comparison ({window}-sample baseline window) ===");
    println!(
        "Dead Reckoning RMS Horizontal Error: {:.2}m",
        dr_stats.rms_horizontal_error
    );
    println!(
        "UKF RMS Horizontal Error: {:.2}m over the window, {:.2}m over the full run",
        ukf_window_stats.rms_horizontal_error, ukf_stats.rms_horizontal_error
    );
    println!(
        "Improvement over the window: {:.1}x",
        dr_stats.rms_horizontal_error / ukf_window_stats.rms_horizontal_error
    );

    assert_beats_dead_reckoning("UKF", &ukf_window_stats, &dr_stats);

    // And hold the full run to the same absolute ceiling as every other healthy filter. The
    // relative check above is necessary but not sufficient, and on its own it is what let #307
    // sit: an EKF 14,707 km from truth still "beat" a dead-reckoning baseline that was further
    // out still. Asserted over the whole stream, because that is where a filter that tracks
    // for four minutes and then walks away shows up.
    assert!(
        ukf_stats.rms_horizontal_error < MAX_HORIZONTAL_RMSE_M,
        "UKF horizontal RMSE over the full run should be under the {MAX_HORIZONTAL_RMSE_M} m \
         operating bound, got {:.2} m",
        ukf_stats.rms_horizontal_error
    );
}

// ==================== Extended Kalman Filter Integration Tests ====================

/// Test EKF closed-loop filter on real data
///
/// This test runs a closed-loop EKF with GNSS measurements on real data and verifies that:
/// 1. The filter completes without errors
/// 2. Position errors remain bounded
/// 3. The filter performs comparably to UKF
#[test]
fn test_ekf_closed_loop_on_real_data() {
    // Load test data
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let test_data_path = Path::new(manifest_dir).join("tests/test_data.csv");
    let records = load_test_data(&test_data_path);

    assert!(
        !records.is_empty(),
        "Test data should contain at least one record"
    );

    // Create initial state from first record
    let initial_state = create_initial_state(&records[0]);

    // Initialize EKF with 15-state configuration (with biases)
    let initial_covariance = DEFAULT_INITIAL_COVARIANCE.to_vec();

    let process_noise =
        DMatrix::from_diagonal(&DVector::from_vec(DEFAULT_PROCESS_NOISE_DENSITY.to_vec()));

    // Initialize EKF (note: EKF constructor differs from UKF - no measurement bias parameter,
    // uses use_biases flag instead of optional measurement_bias)
    let mut ekf = ExtendedKalmanFilter::new(
        &initial_state,
        &[0.0; 6], // Zero initial bias estimates
        initial_covariance,
        process_noise,
        true, // use_biases (15-state configuration)
    );

    // Create event stream with passthrough scheduler (all GNSS measurements used)
    let cfg = {
        let mut built = AidingConfig::default();
        built.scheduler = MeasurementScheduler::PassThrough;
        built.fault = GnssFaultModel::None;
        built
    };

    let stream = build_event_stream(&records, &cfg, TEST_DATA_IS_ENU).unwrap();

    // Run closed-loop filter
    let results = run_closed_loop(&mut ekf, stream, None, None)
        .expect("Closed-loop EKF filter should complete");

    // Verify results
    assert!(
        !results.is_empty(),
        "Closed-loop EKF filter should produce results"
    );

    // Compute error metrics
    let stats = compute_error_metrics(&results, &records);

    // Print statistics
    println!("\n=== EKF Closed-Loop Error Statistics ===");
    println!(
        "Horizontal Error: mean={:.2}m, min={:.2}m, median={:.2}m, max={:.2}m, rms={:.2}m",
        stats.mean_horizontal_error,
        stats.min_horizontal_error,
        stats.median_horizontal_error,
        stats.max_horizontal_error,
        stats.rms_horizontal_error
    );
    println!(
        "Altitude Error: mean={:.2}m, min={:.2}m, median={:.2}m, max={:.2}m, rms={:.2}m",
        stats.mean_altitude_error,
        stats.min_altitude_error,
        stats.median_altitude_error,
        stats.max_altitude_error,
        stats.rms_altitude_error
    );
    println!(
        "Velocity Error: N={:.3}m/s, E={:.3}m/s, D={:.3}m/s",
        stats.mean_velocity_north_error,
        stats.mean_velocity_east_error,
        stats.mean_velocity_vertical_error
    );

    // Assert error bounds - these should be reasonable for a working filter with GNSS
    // With good GNSS, horizontal error should be within a few meters RMS
    // EKF may have slightly higher errors than UKF due to linearization

    // Same healthy-filter rationale as the UKF test (see #288): the three
    // filters agree at ~24-27 m horizontal rms, so hold the EKF to ~1.7x that.
    // Altitude bounds stay wide deliberately: the EKF vertical channel can
    // excursion under sparse aiding (#290), and with 1 s fixes that stays
    // reined in (max 173.5 m observed) but is not bit-stable across platforms.
    let rms_horizontal_limit = 45.0;
    let max_horizontal_limit = 175.0;
    let rms_altitude_limit = 150.0;
    let max_altitude_limit = 1230.0;

    assert!(
        stats.rms_horizontal_error < rms_horizontal_limit,
        "EKF RMS horizontal error should be less than {:.2}m, got {:.2}m",
        rms_horizontal_limit,
        stats.rms_horizontal_error
    );
    assert!(
        stats.max_horizontal_error < max_horizontal_limit,
        "EKF maximum horizontal error should be less than {:.2}m, got {:.2}m",
        max_horizontal_limit,
        stats.max_horizontal_error
    );
    assert!(
        stats.rms_altitude_error < rms_altitude_limit,
        "EKF RMS altitude error should be less than {:.2}m, got {:.2}m",
        rms_altitude_limit,
        stats.rms_altitude_error
    );
    assert!(
        stats.max_altitude_error < max_altitude_limit,
        "EKF maximum altitude error should be less than {:.2}m, got {:.2}m",
        max_altitude_limit,
        stats.max_altitude_error
    );

    // Verify no NaN or infinite values in results
    for result in &results {
        assert!(
            result.latitude.is_finite(),
            "Latitude should be finite: {}",
            result.latitude
        );
        assert!(
            result.longitude.is_finite(),
            "Longitude should be finite: {}",
            result.longitude
        );
        assert!(
            result.altitude.is_finite(),
            "Altitude should be finite: {}",
            result.altitude
        );
        assert!(
            result.velocity_north.is_finite(),
            "Velocity north should be finite: {}",
            result.velocity_north
        );
        assert!(
            result.velocity_east.is_finite(),
            "Velocity east should be finite: {}",
            result.velocity_east
        );
        assert!(
            result.velocity_vertical.is_finite(),
            "Velocity vertical should be finite: {}",
            result.velocity_vertical
        );
    }
}

/// Test EKF with degraded GNSS (reduced update rate)
///
/// This test simulates degraded GNSS conditions with reduced update rate and verifies
/// that the filter still performs reasonably well, though with higher errors than full-rate GNSS.
///
/// Degradation profile: `FixedInterval { interval_s: 5.0 }` with `fault: None`
/// (uncorrupted fixes, dataset accuracies: horizontal sigma ~4.7 m, vertical
/// sigma ~1.4 m), plus the per-sample baro/mag aiding present in every stream.
#[test]
fn test_ekf_with_degraded_gnss() {
    // Load test data
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let test_data_path = Path::new(manifest_dir).join("tests/test_data.csv");
    let records = load_test_data(&test_data_path);

    assert!(
        !records.is_empty(),
        "Test data should contain at least one record"
    );

    // Create initial state from first record
    let initial_state = create_initial_state(&records[0]);

    // Initialize EKF
    let initial_covariance = DEFAULT_INITIAL_COVARIANCE.to_vec();

    let process_noise =
        DMatrix::from_diagonal(&DVector::from_vec(DEFAULT_PROCESS_NOISE_DENSITY.to_vec()));

    let mut ekf = ExtendedKalmanFilter::new(
        &initial_state,
        &[0.0; 6],
        initial_covariance,
        process_noise,
        true, // 15-state with biases
    );

    // Create event stream with periodic scheduler (e.g., every 5 seconds)
    let cfg = {
        let mut built = AidingConfig::default();
        built.scheduler = MeasurementScheduler::FixedInterval {
            interval_s: 5.0,
            phase_s: 0.0,
        };
        built.fault = GnssFaultModel::None;
        built
    };

    let stream = build_event_stream(&records, &cfg, TEST_DATA_IS_ENU).unwrap();

    // Run closed-loop filter
    let results = run_closed_loop(&mut ekf, stream, None, None)
        .expect("Closed-loop EKF filter with degraded GNSS should complete");

    // Compute error metrics
    let stats = compute_error_metrics(&results, &records);

    // Print statistics
    println!("\n=== EKF with Degraded GNSS (5s updates) Error Statistics ===");
    println!(
        "Horizontal Error: mean={:.2}m, min={:.2}m, median={:.2}m, max={:.2}m, rms={:.2}m",
        stats.mean_horizontal_error,
        stats.min_horizontal_error,
        stats.median_horizontal_error,
        stats.max_horizontal_error,
        stats.rms_horizontal_error
    );
    println!(
        "Altitude Error: mean={:.2}m, min={:.2}m, median={:.2}m, max={:.2}m, rms={:.2}m",
        stats.mean_altitude_error,
        stats.min_altitude_error,
        stats.median_altitude_error,
        stats.max_altitude_error,
        stats.rms_altitude_error
    );

    // Error bounds should be looser than full-rate GNSS but still reasonable.
    // EKF may have slightly higher errors than UKF due to linearization.
    //
    // Two kinds of bound are used here, and they must not be confused (see #288).
    // Typical (median) accuracy is governed by dead-reckoning drift between the
    // 5 s fixes: rate error x 5 s plus the fix noise floor (~4.7 m horizontal,
    // ~1.4 m vertical). With a 10 m/s credible horizontal rate error and a
    // 5 m/s credible vertical rate error that gives 50 m / 25 m; observed
    // medians on Linux are 29.5 m / 8.8 m, so both carry real margin.
    // The median is used (rather than the mean) because it is insensitive to
    // the excursion tail and hence stable across floating-point codegen.
    //
    // The rms/max asserts are anti-divergence guards, not accuracy bounds: the
    // EKF vertical channel suffers a large excursion at ~29 m/s with 5 s fixes
    // (rms 81 m Linux / 186 m macOS, max ~945 m; UKF holds 5.9 m on the same
    // stream), tracked by #290. They are set at ~1.6-2x the worst observed
    // cross-platform value so a genuine divergence (1e8 m scale, cf. #266)
    // still trips them while codegen jitter cannot. Do not tighten these to
    // observed values without fixing #290 first.
    let median_horizontal_limit = 50.0;
    let median_altitude_limit = 25.0;
    let rms_horizontal_limit = 150.0;
    let max_horizontal_limit = 1700.0;
    let rms_altitude_limit = 300.0;
    let max_altitude_limit = 2000.0;

    assert!(
        stats.median_horizontal_error < median_horizontal_limit,
        "EKF median horizontal error with degraded GNSS should be less than {:.2}m, got {:.2}m",
        median_horizontal_limit,
        stats.median_horizontal_error
    );
    assert!(
        stats.median_altitude_error < median_altitude_limit,
        "EKF median altitude error with degraded GNSS should be less than {:.2}m, got {:.2}m",
        median_altitude_limit,
        stats.median_altitude_error
    );

    assert!(
        stats.rms_horizontal_error < rms_horizontal_limit,
        "EKF RMS horizontal error with degraded GNSS should be less than {:.2}m, got {:.2}m",
        rms_horizontal_limit,
        stats.rms_horizontal_error
    );
    assert!(
        stats.max_horizontal_error < max_horizontal_limit,
        "EKF maximum horizontal error with degraded GNSS should be less than {:.2}m, got {:.2}m",
        max_horizontal_limit,
        stats.max_horizontal_error
    );
    assert!(
        stats.rms_altitude_error < rms_altitude_limit,
        "EKF RMS altitude error with degraded GNSS should be less than {:.2}m, got {:.2}m",
        rms_altitude_limit,
        stats.rms_altitude_error
    );
    assert!(
        stats.max_altitude_error < max_altitude_limit,
        "EKF maximum altitude error with degraded GNSS should be less than {:.2}m, got {:.2}m",
        max_altitude_limit,
        stats.max_altitude_error
    );

    // Verify no invalid values
    for result in &results {
        assert!(result.latitude.is_finite());
        assert!(result.longitude.is_finite());
        assert!(result.altitude.is_finite());
    }
}

/// Test that closed-loop EKF outperforms dead reckoning
///
/// This test runs both dead reckoning and EKF on the same data and verifies that
/// the EKF produces lower errors than dead reckoning, demonstrating the benefit
/// of GNSS-aided navigation.
#[test]
fn test_ekf_outperforms_dead_reckoning() {
    // Load test data
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let test_data_path = Path::new(manifest_dir).join("tests/test_data.csv");
    let records = load_test_data(&test_data_path);

    assert!(
        !records.is_empty(),
        "Test data should contain at least one record"
    );

    // Run dead reckoning over the baseline window (see `DEAD_RECKONING_BASELINE_SAMPLES`)
    let (_, dr_stats, window) = dead_reckoning_baseline(&records);

    // Run EKF
    let initial_state = create_initial_state(&records[0]);
    let initial_covariance = DEFAULT_INITIAL_COVARIANCE.to_vec();
    let process_noise =
        DMatrix::from_diagonal(&DVector::from_vec(DEFAULT_PROCESS_NOISE_DENSITY.to_vec()));

    let mut ekf = ExtendedKalmanFilter::new(
        &initial_state,
        &[0.0; 6],
        initial_covariance,
        process_noise,
        true, // 15-state
    );

    let scheduler = MeasurementScheduler::PassThrough;
    let fault_model = GnssFaultModel::None;
    let cfg = {
        let mut built = AidingConfig::default();
        built.scheduler = scheduler;
        built.fault = fault_model;
        built
    };
    let stream = build_event_stream(&records, &cfg, TEST_DATA_IS_ENU).unwrap();

    // Full stream for the absolute ceiling below, baseline window for the comparison; see the
    // equivalent block in `test_ukf_outperforms_dead_reckoning` for why both are needed.
    let ekf_results = run_closed_loop(&mut ekf, stream, None, None).expect("EKF should complete");
    let ekf_stats = compute_error_metrics(&ekf_results, &records);
    let ekf_window_stats = compute_error_metrics(&ekf_results[..window], &records[..window]);

    // Print comparison
    println!("\n=== Performance Comparison (EKF vs Dead Reckoning, {window}-sample window) ===");
    println!(
        "Dead Reckoning RMS Horizontal Error: {:.2}m",
        dr_stats.rms_horizontal_error
    );
    println!(
        "EKF RMS Horizontal Error: {:.2}m over the window, {:.2}m over the full run",
        ekf_window_stats.rms_horizontal_error, ekf_stats.rms_horizontal_error
    );
    println!(
        "Improvement over the window: {:.1}x",
        dr_stats.rms_horizontal_error / ekf_window_stats.rms_horizontal_error
    );

    assert_beats_dead_reckoning("EKF", &ekf_window_stats, &dr_stats);

    // The relative check above is necessary but nowhere near sufficient, and on its own it is
    // what let #307 sit: dead reckoning ends this recording ~6,600 km out, so "better than
    // dead reckoning" was satisfied by an EKF 14,707 km from truth on the far side of the
    // planet. Hold it to the same absolute ceiling as every other healthy filter -- and hold
    // it over the **full** run, not the baseline window, or the guard shrinks from 89 minutes
    // to the first four and an EKF that loses the solution afterwards passes.
    assert!(
        ekf_stats.rms_horizontal_error < MAX_HORIZONTAL_RMSE_M,
        "EKF horizontal RMSE over the full run should be under the {MAX_HORIZONTAL_RMSE_M} m \
         operating bound, got {:.2} m. Beating dead reckoning is not evidence of a working \
         filter when dead reckoning is at {:.0} m",
        ekf_stats.rms_horizontal_error,
        dr_stats.rms_horizontal_error
    );
}

// ==================== Error-State Kalman Filter Integration Tests ====================

/// Test ESKF closed-loop filter on real data
///
/// This test runs a closed-loop ESKF with GNSS measurements on real data and verifies that:
/// 1. The filter completes without errors
/// 2. Position errors remain bounded
/// 3. The filter performs comparably to UKF/EKF
/// 4. Quaternion normalization is maintained
#[test]
fn test_eskf_closed_loop_on_real_data() {
    // Load test data
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let test_data_path = Path::new(manifest_dir).join("tests/test_data.csv");
    let records = load_test_data(&test_data_path);

    assert!(
        !records.is_empty(),
        "Test data should contain at least one record"
    );

    // Create initial state from first record
    let initial_state = create_initial_state(&records[0]);

    // Initialize ESKF with 15-state configuration (error-state representation)
    // Use ESKF-specific covariance and process noise to prevent divergence
    let initial_error_covariance = ESKF_INITIAL_COVARIANCE.to_vec();

    let process_noise = DMatrix::from_diagonal(&DVector::from_vec(ESKF_PROCESS_NOISE.to_vec()));

    // Initialize ESKF
    let mut eskf = ErrorStateKalmanFilter::new(
        &initial_state,
        &[0.0; 6], // Zero initial bias estimates
        initial_error_covariance,
        process_noise,
    );

    // Create event stream with passthrough scheduler (all GNSS measurements used)
    let cfg = {
        let mut built = AidingConfig::default();
        built.scheduler = MeasurementScheduler::PassThrough;
        built.fault = GnssFaultModel::None;
        built
    };

    let stream = build_event_stream(&records, &cfg, TEST_DATA_IS_ENU).unwrap();

    // Run closed-loop filter
    let results = run_closed_loop(&mut eskf, stream, None, None)
        .expect("Closed-loop ESKF filter should complete");

    // Verify results
    assert!(
        !results.is_empty(),
        "Closed-loop ESKF filter should produce results"
    );

    // Compute error metrics
    let stats = compute_error_metrics(&results, &records);

    // Print statistics
    println!("\n=== ESKF Closed-Loop Error Statistics ===");
    println!(
        "Horizontal Error: mean={:.2}m, min={:.2}m, median={:.2}m, max={:.2}m, rms={:.2}m",
        stats.mean_horizontal_error,
        stats.min_horizontal_error,
        stats.median_horizontal_error,
        stats.max_horizontal_error,
        stats.rms_horizontal_error
    );
    println!(
        "Altitude Error: mean={:.2}m, min={:.2}m, median={:.2}m, max={:.2}m, rms={:.2}m",
        stats.mean_altitude_error,
        stats.min_altitude_error,
        stats.median_altitude_error,
        stats.max_altitude_error,
        stats.rms_altitude_error
    );
    println!(
        "Velocity Error: N={:.3}m/s, E={:.3}m/s, D={:.3}m/s",
        stats.mean_velocity_north_error,
        stats.mean_velocity_east_error,
        stats.mean_velocity_vertical_error
    );

    // Assert error bounds - ESKF with 5x process noise tuning
    // Performance reflects trade-off between stability (no divergence) and accuracy
    // These bounds are based on empirical performance with real MEMS-grade IMU data

    // Horizontal bounds are physical, not fitted. With continuous GNSS aiding at a
    // few metres of position noise, a correctly closed loosely-coupled filter must
    // stay in the tens of metres; the UKF and EKF sit at 23.54 m and 23.58 m rms on this
    // dataset and the ESKF is at 23.54 m. The limits below match the UKF test's
    // (~1.7x observed) so all three filters are held to the same standard: they
    // still fail loudly if the horizontal loop opens again (before #266 this run
    // produced 1734 m rms).
    let rms_horizontal_limit = 40.0;
    let max_horizontal_limit = 60.0;

    // Vertical bounds, tightened when #286 landed. The UKF achieves 2.7 m rms /
    // 13.0 m peak on this data; the ESKF is at 2.4 m / 9.2 m. Limits carry ~4x
    // margin: any return of the vertical-channel divergence (previously 119 m
    // rms / 385 m peak) trips them immediately, while healthy-filter codegen
    // jitter across platforms cannot.
    let rms_altitude_limit = 10.0;
    let max_altitude_limit = 40.0;

    assert!(
        stats.rms_horizontal_error < rms_horizontal_limit,
        "ESKF RMS horizontal error should be less than {:.2}m, got {:.2}m",
        rms_horizontal_limit,
        stats.rms_horizontal_error
    );
    assert!(
        stats.max_horizontal_error < max_horizontal_limit,
        "ESKF maximum horizontal error should be less than {:.2}m, got {:.2}m",
        max_horizontal_limit,
        stats.max_horizontal_error
    );
    assert!(
        stats.rms_altitude_error < rms_altitude_limit,
        "ESKF RMS altitude error should be less than {:.2}m, got {:.2}m",
        rms_altitude_limit,
        stats.rms_altitude_error
    );
    assert!(
        stats.max_altitude_error < max_altitude_limit,
        "ESKF maximum altitude error should be less than {:.2}m, got {:.2}m",
        max_altitude_limit,
        stats.max_altitude_error
    );

    // Bias plausibility (#286, #258): estimates must stay within the anti-windup caps
    // that bound them at every sample of the run, not merely at the end. Before the
    // #286 fixes these reached 9.2 m/s² and 6.5 rad/s on this same data.
    assert_bias_estimates_bounded(&results, "ESKF closed-loop");
    let final_est = eskf.get_estimate();
    println!(
        "Final biases: accel=[{:.4}, {:.4}, {:.4}] m/s², gyro=[{:.5}, {:.5}, {:.5}] rad/s",
        final_est[9], final_est[10], final_est[11], final_est[12], final_est[13], final_est[14]
    );

    // Verify no NaN or infinite values in results
    for result in &results {
        assert!(
            result.latitude.is_finite(),
            "Latitude should be finite: {}",
            result.latitude
        );
        assert!(
            result.longitude.is_finite(),
            "Longitude should be finite: {}",
            result.longitude
        );
        assert!(
            result.altitude.is_finite(),
            "Altitude should be finite: {}",
            result.altitude
        );
        assert!(
            result.velocity_north.is_finite(),
            "Velocity north should be finite: {}",
            result.velocity_north
        );
        assert!(
            result.velocity_east.is_finite(),
            "Velocity east should be finite: {}",
            result.velocity_east
        );
        assert!(
            result.velocity_vertical.is_finite(),
            "Velocity vertical should be finite: {}",
            result.velocity_vertical
        );
    }
}

/// Test ESKF with degraded GNSS (reduced update rate)
///
/// This test simulates degraded GNSS conditions with reduced update rate (2s intervals).
#[test]
fn test_eskf_with_degraded_gnss() {
    // Load test data
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let test_data_path = Path::new(manifest_dir).join("tests/test_data.csv");
    let records = load_test_data(&test_data_path);

    assert!(
        !records.is_empty(),
        "Test data should contain at least one record"
    );

    // Create initial state from first record
    let initial_state = create_initial_state(&records[0]);

    // Initialize ESKF with ESKF-specific covariance and process noise
    let initial_error_covariance = ESKF_INITIAL_COVARIANCE.to_vec();

    let process_noise = DMatrix::from_diagonal(&DVector::from_vec(ESKF_PROCESS_NOISE.to_vec()));

    let mut eskf = ErrorStateKalmanFilter::new(
        &initial_state,
        &[0.0; 6],
        initial_error_covariance,
        process_noise,
    );

    // Create event stream with periodic scheduler (e.g., every 5 seconds)
    let cfg = {
        let mut built = AidingConfig::default();
        built.scheduler = MeasurementScheduler::FixedInterval {
            interval_s: 2.0,
            phase_s: 0.0,
        };
        built.fault = GnssFaultModel::None;
        built
    };

    let stream = build_event_stream(&records, &cfg, TEST_DATA_IS_ENU).unwrap();

    // Run closed-loop filter
    let results = run_closed_loop(&mut eskf, stream, None, None)
        .expect("Closed-loop ESKF filter with degraded GNSS should complete");

    // Compute error metrics
    let stats = compute_error_metrics(&results, &records);

    // Print statistics
    println!("\n=== ESKF with Degraded GNSS (2s updates) Error Statistics ===");
    println!(
        "Horizontal Error: mean={:.2}m, min={:.2}m, median={:.2}m, max={:.2}m, rms={:.2}m",
        stats.mean_horizontal_error,
        stats.min_horizontal_error,
        stats.median_horizontal_error,
        stats.max_horizontal_error,
        stats.rms_horizontal_error
    );
    println!(
        "Altitude Error: mean={:.2}m, min={:.2}m, median={:.2}m, max={:.2}m, rms={:.2}m",
        stats.mean_altitude_error,
        stats.min_altitude_error,
        stats.median_altitude_error,
        stats.max_altitude_error,
        stats.rms_altitude_error
    );

    // Error bounds for degraded GNSS (2s update intervals).
    //
    // Re-enabled and tightened when #286 landed: with 2 s fixes the healthy
    // ESKF sits at 23.6 m horizontal rms / 42.4 m peak and 3.6 m altitude rms /
    // 12.9 m peak -- barely above the full-rate numbers (23.5 / 2.4 m), since
    // 2 s of MEMS dead-reckoning drift is small next to the fix noise floor.
    // Limits carry ~2.5-4.5x margin: the previous 1000/3500/400/3000 m ceilings
    // were vacuous (any non-divergent filter passed) and are replaced with
    // bounds that actually fail if the vertical channel regresses.
    assert!(
        stats.rms_horizontal_error < 60.0,
        "RMS horizontal error with degraded GNSS should be less than 60m, got {:.2}m",
        stats.rms_horizontal_error
    );

    assert!(
        stats.max_horizontal_error < 150.0,
        "Maximum horizontal error with degraded GNSS should be less than 150m, got {:.2}m",
        stats.max_horizontal_error
    );

    assert!(
        stats.rms_altitude_error < 15.0,
        "RMS altitude error with degraded GNSS should be less than 15m, got {:.2}m",
        stats.rms_altitude_error
    );

    assert!(
        stats.max_altitude_error < 60.0,
        "Maximum altitude error with degraded GNSS should be less than 60m, got {:.2}m",
        stats.max_altitude_error
    );

    // Verify no invalid values
    for result in &results {
        assert!(result.latitude.is_finite());
        assert!(result.longitude.is_finite());
        assert!(result.altitude.is_finite());
    }

    // Degraded aiding is the condition bias windup showed up under (#286): halving the
    // fix rate doubles the interval a mis-scaled correction has to accumulate over before
    // the next measurement pulls it back.
    assert_bias_estimates_bounded(&results, "ESKF degraded GNSS");
}

/// Test that closed-loop ESKF outperforms dead reckoning
///
/// This test runs both dead reckoning and ESKF on the same data and verifies that
/// the ESKF produces lower errors than dead reckoning, demonstrating the benefit
/// of GNSS-aided navigation with error-state formulation.
#[test]
fn test_eskf_outperforms_dead_reckoning() {
    // Load test data
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let test_data_path = Path::new(manifest_dir).join("tests/test_data.csv");
    let records = load_test_data(&test_data_path);

    assert!(
        !records.is_empty(),
        "Test data should contain at least one record"
    );

    // Run dead reckoning over the baseline window (see `DEAD_RECKONING_BASELINE_SAMPLES`)
    let (_, dr_stats, window) = dead_reckoning_baseline(&records);

    // Run ESKF
    let initial_state = create_initial_state(&records[0]);
    let initial_error_covariance = ESKF_INITIAL_COVARIANCE.to_vec();
    let process_noise = DMatrix::from_diagonal(&DVector::from_vec(ESKF_PROCESS_NOISE.to_vec()));

    let mut eskf = ErrorStateKalmanFilter::new(
        &initial_state,
        &[0.0; 6], // Zero initial bias estimates
        initial_error_covariance,
        process_noise,
    );

    let cfg = {
        let mut built = AidingConfig::default();
        built.scheduler = MeasurementScheduler::PassThrough;
        built.fault = GnssFaultModel::None;
        built
    };
    let stream = build_event_stream(&records, &cfg, TEST_DATA_IS_ENU).unwrap();

    // Full stream for the absolute ceiling below, baseline window for the comparison; see the
    // equivalent block in `test_ukf_outperforms_dead_reckoning` for why both are needed.
    let eskf_results =
        run_closed_loop(&mut eskf, stream, None, None).expect("ESKF should complete");
    let eskf_stats = compute_error_metrics(&eskf_results, &records);
    let eskf_window_stats = compute_error_metrics(&eskf_results[..window], &records[..window]);

    // Print comparison
    println!("\n=== Performance Comparison (ESKF vs Dead Reckoning, {window}-sample window) ===");
    println!(
        "Dead Reckoning RMS Horizontal Error: {:.2}m",
        dr_stats.rms_horizontal_error
    );
    println!(
        "ESKF RMS Horizontal Error: {:.2}m over the window, {:.2}m over the full run",
        eskf_window_stats.rms_horizontal_error, eskf_stats.rms_horizontal_error
    );
    println!(
        "Improvement over the window: {:.1}x",
        dr_stats.rms_horizontal_error / eskf_window_stats.rms_horizontal_error
    );

    assert_beats_dead_reckoning("ESKF", &eskf_window_stats, &dr_stats);

    // As in the EKF test: the ratio alone is not evidence of a working filter (#307), and the
    // absolute ceiling is asserted over the full run so a filter that tracks for the baseline
    // window and then walks away cannot pass.
    assert!(
        eskf_stats.rms_horizontal_error < MAX_HORIZONTAL_RMSE_M,
        "ESKF horizontal RMSE over the full run should be under the {MAX_HORIZONTAL_RMSE_M} m \
         operating bound, got {:.2} m",
        eskf_stats.rms_horizontal_error
    );
}

/// Every sample the ESKF emits over the full run is a usable navigation solution.
///
/// This is the per-sample validity sweep for the shipped tuning: position, velocity *and*
/// attitude, at all 5,366 samples, reported with the index of the first sample that fails.
/// Accuracy and bias plausibility for this same run are asserted in
/// `test_eskf_closed_loop_on_real_data`; what is unique here is the attitude channel, which
/// no other test in this suite looks at. `get_estimate` builds roll/pitch/yaw by converting
/// the nominal quaternion to a rotation and decomposing it, so a nominal quaternion that
/// stopped being unit-length -- the invariant `ErrorStateKalmanFilter`'s documentation leads
/// with -- surfaces here as a non-finite angle before it is large enough to move the
/// position error. The bounds are `euler_angles`'s own codomain; `asin` and `atan2` pass NaN
/// through unchanged, which is why finiteness is asserted separately.
///
/// ## This test used to claim it exercised high dynamics. It never did.
///
/// It ran `test_data.csv` through the same tuning, the same `PassThrough` scheduler and the
/// same code path as `test_eskf_closed_loop_on_real_data`, down to identical metrics. A "5x
/// default" initial covariance and a "10x default" process noise sat in the body
/// `_`-prefixed and unused, implying a tuning that was never applied; they are gone.
///
/// A genuine high-dynamics variant was investigated and is not currently derivable:
///
/// - **The dataset has no high-dynamics segment to cut.** `test_data.csv` is a road drive:
///   angular rate is 0.54 deg/s median and 9.8 deg/s at the 99th percentile, with 2 of its
///   5,366 samples above 50 deg/s and a peak of 54.5 deg/s, and horizontal specific force
///   -- sqrt(|f|^2 - g^2) -- peaks at 4.2 m/s^2 (0.43 g). There is no
///   window in it that is dynamic by any useful definition, and the full run -- which
///   contains whatever dynamics exist -- is already covered.
/// - **`sim::generate_synthetic` cannot produce linear dynamics at all.** It propagates
///   constant nav-frame velocity by construction (`compute_perfect_imu` solves for the
///   specific force that holds `v_dot = 0`), so the only stressor it offers is body angular
///   rate.
/// - **Its angular-rate runs are not clean enough to bound.** Measured against the generator's
///   own truth trajectory with 1 Hz GNSS at 2.5 m noise: at a navigation-grade IMU and *zero*
///   angular rate the ESKF already sits at 12.7 m rms / 79.4 m peak horizontal at 50 Hz, and
///   error is not monotone in angular rate: at 10 Hz a 30/15/20 deg/s run comes in *under*
///   the zero-rate run, 5.1 m rms against 5.9 m. A bound drawn
///   from those runs would be pinned to an unexplained baseline rather than to the dynamics,
///   which is the practice this file's bounds exist to avoid. At consumer grade the bias
///   estimates sit on the anti-windup clamp even at zero angular rate, so the clamp, not the
///   estimator, would be what such a test measured.
///
/// Making the synthetic generator produce accelerating trajectories, and explaining its
/// zero-dynamics baseline, are both prerequisites for a real high-dynamics test.
#[test]
fn test_eskf_output_stays_valid_across_full_run() {
    // Load test data
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let test_data_path = Path::new(manifest_dir).join("tests/test_data.csv");
    let records = load_test_data(&test_data_path);

    assert!(
        !records.is_empty(),
        "Test data should contain at least one record"
    );

    // Create initial state from first record
    let initial_state = create_initial_state(&records[0]);

    // The shipped tuning, the same one `strapdown-sim` and every other ESKF test here use.
    let initial_error_covariance = ESKF_INITIAL_COVARIANCE.to_vec();
    let process_noise = DMatrix::from_diagonal(&DVector::from_vec(ESKF_PROCESS_NOISE.to_vec()));

    let mut eskf = ErrorStateKalmanFilter::new(
        &initial_state,
        &[0.0; 6], // Zero initial bias estimates
        initial_error_covariance,
        process_noise,
    );

    // Use passthrough GNSS to help constrain the solution
    let cfg = {
        let mut built = AidingConfig::default();
        built.scheduler = MeasurementScheduler::PassThrough;
        built.fault = GnssFaultModel::None;
        built
    };

    let stream = build_event_stream(&records, &cfg, TEST_DATA_IS_ENU).unwrap();

    // Run closed-loop filter
    let results = run_closed_loop(&mut eskf, stream, None, None).expect("ESKF should complete");

    // Everything below this line is per-sample, and every per-sample assertion in a `for`
    // loop is vacuously true over an empty series. `compute_error_metrics` fails the same
    // way from the other side: it starts its accumulators at zero and returns them
    // untouched when nothing matches, so the bounds at the end of this test would pass on
    // no data at all. Pin the length first, and pin it to `records.len()` rather than to
    // "non-empty" -- a filter that emitted one solution and stopped is exactly the
    // regression that would otherwise slip through here. This is also the only absolute
    // length check the ESKF has: `test_filter_comparison` asserts only that the three
    // filters agree with *each other*, which all three being short would satisfy.
    assert_eq!(
        results.len(),
        records.len(),
        "the ESKF must emit one solution per input record; got {} for {} records",
        results.len(),
        records.len()
    );

    // Verify all results are valid (no NaN or Inf)
    for (i, result) in results.iter().enumerate() {
        assert!(
            result.latitude.is_finite(),
            "Latitude should be finite at step {}: {}",
            i,
            result.latitude
        );
        assert!(
            result.longitude.is_finite(),
            "Longitude should be finite at step {}: {}",
            i,
            result.longitude
        );
        assert!(
            result.altitude.is_finite(),
            "Altitude should be finite at step {}: {}",
            i,
            result.altitude
        );
        assert!(
            result.velocity_north.is_finite(),
            "Velocity north should be finite at step {}: {}",
            i,
            result.velocity_north
        );
        assert!(
            result.velocity_east.is_finite(),
            "Velocity east should be finite at step {}: {}",
            i,
            result.velocity_east
        );
        assert!(
            result.velocity_vertical.is_finite(),
            "Velocity vertical should be finite at step {}: {}",
            i,
            result.velocity_vertical
        );

        // Attitude channel. `get_estimate` reads roll/pitch/yaw straight off the nominal
        // quaternion's `euler_angles`, so a nominal quaternion that stopped being
        // unit-length lands here as NaN -- `asin` and `atan2` neither reject nor normalise
        // it -- rather than as an out-of-range angle. Hence: finite first, then inside the
        // decomposition's own codomain, which is `atan2`'s [-pi, pi] for roll and yaw and
        // `asin`'s [-pi/2, pi/2] for pitch (#314).
        for (name, angle, bound) in [
            ("roll", result.roll, std::f64::consts::PI),
            ("pitch", result.pitch, std::f64::consts::FRAC_PI_2),
            ("yaw", result.yaw, std::f64::consts::PI),
        ] {
            assert!(
                angle.is_finite(),
                "{name} should be finite at step {i}: {angle} (a non-finite Euler angle here \
                 means the nominal quaternion lost unit length)"
            );
            assert!(
                (-bound..=bound).contains(&angle),
                "{name} should lie in `euler_angles`'s principal branch at step {i}, got \
                 {angle}"
            );
        }
    }

    // Compute error metrics to verify reasonable performance
    let stats = compute_error_metrics(&results, &records);

    println!("\n=== ESKF Full-Run Validity ===");
    println!(
        "RMS Horizontal Error: {:.2}m, Max: {:.2}m",
        stats.rms_horizontal_error, stats.max_horizontal_error
    );
    println!(
        "RMS Altitude Error: {:.2}m, Max: {:.2}m",
        stats.rms_altitude_error, stats.max_altitude_error
    );

    // Physical bounds, not fitted. Before #266 the ESKF's horizontal loop was open --
    // corrections were rescaled by ~1/6.4e6 -- and this run produced ~1900 m rms, which
    // is what the previous 1905.0 / 2494.0 limits were pinned to. Those numbers were
    // tight enough to the observed value that macOS and Windows failed them at 2121.97 m
    // purely on floating-point code generation. With the loop closed the run sits at
    // 24 m rms, so bound it where a GNSS-aided filter physically belongs.
    //
    // These are `test_eskf_closed_loop_on_real_data`'s limits, not looser ones. This is
    // that same run -- same tuning, same scheduler, same records -- so a second, weaker
    // ceiling on the same numbers would only record that this test was once believed to be
    // doing something harder. The tripwire it is here to trip is divergence, and the
    // shared limits trip it with room to spare.
    let rms_horizontal_limit = 40.0;
    let max_horizontal_limit = 60.0;
    assert!(
        stats.rms_horizontal_error < rms_horizontal_limit,
        "ESKF RMS horizontal error should stay under {:.2}m across the full run, got {:.2}m",
        rms_horizontal_limit,
        stats.rms_horizontal_error
    );

    assert!(
        stats.max_horizontal_error < max_horizontal_limit,
        "ESKF maximum horizontal error should stay under {:.2}m across the full run, got {:.2}m",
        max_horizontal_limit,
        stats.max_horizontal_error
    );
}

/// The construction path a user of the default filter actually takes (#258).
///
/// Every other ESKF test in this file builds the filter from `ESKF_INITIAL_COVARIANCE` and
/// `ESKF_PROCESS_NOISE`, which are test-local tuning constants. Now that `FilterType`
/// defaults to `Eskf`, the tuning a `strapdown-sim closed-loop` run gets is
/// `initialize_eskf`'s -- five orders of magnitude tighter on the bias states -- and until
/// this test existed nothing exercised it end to end. Promoting a filter to the default
/// without covering the default's own initialisation would ship the untested path.
#[test]
fn test_eskf_default_initialization_on_real_data() {
    /// Samples the vertical channel is allowed to settle over: 30 s at this recording's 1 Hz.
    const SETTLING_SAMPLES: usize = 30;

    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let test_data_path = Path::new(manifest_dir).join("tests/test_data.csv");
    let records = load_test_data(&test_data_path);
    assert!(!records.is_empty(), "test data should not be empty");

    // Every option left at its default, with only the frame set: exactly what
    // `strapdown-sim cl --enu` passes. Note the CLI's default is NED since #296; `--enu` is
    // what a Sensor Logger recording like this one needs.
    let mut eskf = initialize_eskf(&records[0], {
        let mut built = EskfConfig::default();
        built.is_enu = IS_ENU_TEST_DATA;
        built
    })
    .expect("the default ESKF initialisation must succeed on real data");

    let cfg = {
        let mut built = AidingConfig::default();
        built.scheduler = MeasurementScheduler::PassThrough;
        built.fault = GnssFaultModel::None;
        built
    };
    let results = run_closed_loop(
        &mut eskf,
        build_event_stream(&records, &cfg, TEST_DATA_IS_ENU).unwrap(),
        None,
        None,
    )
    .expect("the default ESKF must complete the full run");
    assert_eq!(
        results.len(),
        records.len(),
        "the filter must emit one solution per input record"
    );

    let stats = compute_error_metrics(&results, &records);
    println!("\n=== ESKF Default Initialization ===");
    println!(
        "Horizontal Error: rms={:.2}m, max={:.2}m",
        stats.rms_horizontal_error, stats.max_horizontal_error
    );
    println!(
        "Altitude Error: rms={:.2}m, max={:.2}m",
        stats.rms_altitude_error, stats.max_altitude_error
    );

    // Held to the same standard as `test_eskf_closed_loop_on_real_data`, so the default
    // tuning cannot quietly be the worse of the two by a margin that matters. The two now
    // sit on top of each other: 23.66 m rms / 41.84 m peak horizontal against that test's
    // 23.54 m / 41.84 m. Before #308 the default path was the better of the two by 2 m of
    // peak, which was the default's 1x position process noise against this file's 8x of the
    // same broken number -- a difference between two wrong values, not a result.
    assert!(
        stats.rms_horizontal_error < 40.0,
        "default-initialised ESKF RMS horizontal error should be under 40m, got {:.2}m",
        stats.rms_horizontal_error
    );
    assert!(
        stats.max_horizontal_error < 60.0,
        "default-initialised ESKF max horizontal error should be under 60m, got {:.2}m",
        stats.max_horizontal_error
    );
    assert!(
        stats.rms_altitude_error < 10.0,
        "default-initialised ESKF RMS altitude error should be under 10m, got {:.2}m",
        stats.rms_altitude_error
    );

    // The vertical channel is unobservable at t=0: the filter starts with zero vertical
    // velocity and no knowledge of the accelerometer bias, and needs a few GNSS fixes before
    // it can separate the two. There is no startup transient left to bound, though -- the
    // peak is 11.97 m and falls after `SETTLING_SAMPLES`, so it equals the settled peak
    // below.
    //
    // It took two fixes to get there, and neither was about the unobservable vertical
    // channel. #308 gave `initialize_eskf` a vertical P0 of 10 m instead of 1 cm, so the
    // first fixes could move the altitude estimate instead of being argued down by it: that
    // took the peak from 42.55 m to 28.20 m. The rest of it was the initial attitude, seeded
    // from the record's radian Euler columns with `in_degrees: true` and so scaled by
    // pi/180 -- this recording's 77-degree roll reached the filter as a 0.16-degree one, and
    // the first samples removed gravity along the wrong body axes. Seeding from the
    // quaternion drops the peak to 11.97 m.
    //
    // The two ceilings below now measure the same sample. They are kept separate because a
    // reintroduced transient is exactly the regression they exist to catch.
    let settled_max_altitude_error = results
        .iter()
        .zip(records.iter())
        .skip(SETTLING_SAMPLES)
        .map(|(result, record)| (result.altitude - record.altitude).abs())
        .fold(0.0_f64, f64::max);
    println!("Altitude Error after settling: max={settled_max_altitude_error:.2}m");
    assert!(
        stats.max_altitude_error < 40.0,
        "default-initialised ESKF max altitude error over the full run should be under 40m, got {:.2}m",
        stats.max_altitude_error
    );
    // ~3x margin over the 11.97 m observed across the remaining 5,336 samples, and far
    // below the 385 m peak the pre-#286 vertical divergence produced.
    assert!(
        settled_max_altitude_error < 40.0,
        "default-initialised ESKF max altitude error after settling should be under 40m, got {settled_max_altitude_error:.2}m"
    );

    // On this tuning the anti-windup clamp never engages -- unlike the looser test-local
    // covariance, where it fires on 76 of the 5,366 samples. So here the bound below is a
    // statement about the estimator rather than about the clamp.
    assert_bias_estimates_bounded(&results, "ESKF default initialization");
    for (i, result) in results.iter().enumerate() {
        for (axis, bias) in [
            ("x", result.gyro_bias_x),
            ("y", result.gyro_bias_y),
            ("z", result.gyro_bias_z),
        ] {
            assert!(
                bias.abs() < MAX_GYRO_BIAS_RPS,
                "gyro bias {axis} reached the anti-windup clamp at sample {i} \
                 ({bias:.5} rad/s): the default tuning is no longer estimating the bias, \
                 it is being held by the clamp (#258)"
            );
        }
    }
}

/// Test comparison of all three filter types (UKF, EKF, ESKF)
///
/// This test runs all three filter types on the same data and compares their performance.
/// It verifies that all filters produce reasonable results and helps understand their
/// relative strengths.
#[test]
// #[ignore = "ESKF diverges on extended real-world datasets - requires further tuning"]
fn test_filter_comparison() {
    // Load test data
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let test_data_path = Path::new(manifest_dir).join("tests/test_data.csv");
    let records = load_test_data(&test_data_path);

    assert!(
        !records.is_empty(),
        "Test data should contain at least one record"
    );

    let initial_state = create_initial_state(&records[0]);
    let cfg = {
        let mut built = AidingConfig::default();
        built.scheduler = MeasurementScheduler::PassThrough;
        built.fault = GnssFaultModel::None;
        built
    };

    // Run UKF
    let initial_covariance = DEFAULT_INITIAL_COVARIANCE.to_vec();
    let process_noise =
        DMatrix::from_diagonal(&DVector::from_vec(DEFAULT_PROCESS_NOISE_DENSITY.to_vec()));

    let mut ukf = UnscentedKalmanFilter::new(
        &initial_state,
        &[0.0; 6],
        None,
        initial_covariance.clone(),
        process_noise.clone(),
        1e-3,
        2.0,
        0.0,
    );
    let stream_ukf = build_event_stream(&records, &cfg, TEST_DATA_IS_ENU).unwrap();
    let ukf_results =
        run_closed_loop(&mut ukf, stream_ukf, None, None).expect("UKF should complete");
    let ukf_stats = compute_error_metrics(&ukf_results, &records);

    // Run EKF
    let mut ekf = ExtendedKalmanFilter::new(
        &initial_state,
        &[0.0; 6],
        initial_covariance,
        process_noise,
        true,
    );
    let stream_ekf = build_event_stream(&records, &cfg, TEST_DATA_IS_ENU).unwrap();
    let ekf_results =
        run_closed_loop(&mut ekf, stream_ekf, None, None).expect("EKF should complete");
    let ekf_stats = compute_error_metrics(&ekf_results, &records);

    // Run ESKF
    let initial_error_covariance = ESKF_INITIAL_COVARIANCE.to_vec();
    let process_noise = DMatrix::from_diagonal(&DVector::from_vec(ESKF_PROCESS_NOISE.to_vec()));

    let mut eskf = ErrorStateKalmanFilter::new(
        &initial_state,
        &[0.0; 6], // Zero initial bias estimates
        initial_error_covariance,
        process_noise,
    );

    let stream_eskf = build_event_stream(&records, &cfg, TEST_DATA_IS_ENU).unwrap();
    let eskf_results =
        run_closed_loop(&mut eskf, stream_eskf, None, None).expect("ESKF should complete");
    let eskf_stats = compute_error_metrics(&eskf_results, &records);

    // Print comparison
    println!("\n=== Filter Performance Comparison ===");
    println!(
        "UKF  - RMS Horizontal: {:.2}m, RMS Altitude: {:.2}m, Max Horizontal: {:.2}m",
        ukf_stats.rms_horizontal_error,
        ukf_stats.rms_altitude_error,
        ukf_stats.max_horizontal_error
    );
    println!(
        "EKF  - RMS Horizontal: {:.2}m, RMS Altitude: {:.2}m, Max Horizontal: {:.2}m",
        ekf_stats.rms_horizontal_error,
        ekf_stats.rms_altitude_error,
        ekf_stats.max_horizontal_error
    );
    println!(
        "ESKF - RMS Horizontal: {:.2}m, RMS Altitude: {:.2}m, Max Horizontal: {:.2}m",
        eskf_stats.rms_horizontal_error,
        eskf_stats.rms_altitude_error,
        eskf_stats.max_horizontal_error
    );

    // All filters should produce reasonable results. Bounds are ~1.6x the
    // observed healthy-filter rms (~24-27 m for all three; dead reckoning is
    // at 5e6 m), not fitted to observed values -- see #288.
    assert!(
        ukf_stats.rms_horizontal_error < 40.0,
        "UKF RMS horizontal error should be reasonable"
    );
    assert!(
        ekf_stats.rms_horizontal_error < 45.0,
        "EKF RMS horizontal error should be reasonable"
    );
    // The 1905.0 m tolerance this used to carry was the signature of #266: the ESKF's
    // horizontal corrections were divided by the principal radii, leaving that channel
    // open loop. With the units fixed the ESKF tracks the other two filters, so hold it
    // to the same standard as the EKF.
    assert!(
        eskf_stats.rms_horizontal_error < 45.0,
        "ESKF RMS horizontal error should be comparable to UKF/EKF, got {:.2}m",
        eskf_stats.rms_horizontal_error
    );

    // Verify all filters completed without producing invalid values
    assert_eq!(ukf_results.len(), ekf_results.len());
    assert_eq!(ekf_results.len(), eskf_results.len());
}

/// Test RBPF on real data with GNSS measurements
#[test]
fn test_rbpf_closed_loop_on_real_data() {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let test_data_path = Path::new(manifest_dir).join("tests/test_data.csv");
    let records = load_test_data(&test_data_path);

    assert!(
        !records.is_empty(),
        "Test data should contain at least one record"
    );

    let results = run_rbpf(&records);
    assert!(!results.is_empty(), "RBPF should produce results");

    let stats = compute_error_metrics(&results, &records);

    println!("\n=== RBPF Error Statistics ===");
    println!(
        "Horizontal Error: mean={:.2}m, min={:.2}m, median={:.2}m, max={:.2}m, rms={:.2}m",
        stats.mean_horizontal_error,
        stats.min_horizontal_error,
        stats.median_horizontal_error,
        stats.max_horizontal_error,
        stats.rms_horizontal_error
    );
    println!(
        "Altitude Error: mean={:.2}m, min={:.2}m, median={:.2}m, max={:.2}m, rms={:.2}m",
        stats.mean_altitude_error,
        stats.min_altitude_error,
        stats.median_altitude_error,
        stats.max_altitude_error,
        stats.rms_altitude_error
    );
    println!(
        "Velocity Error: N={:.3}m/s, E={:.3}m/s, D={:.3}m/s",
        stats.mean_velocity_north_error,
        stats.mean_velocity_east_error,
        stats.mean_velocity_vertical_error
    );

    assert!(
        stats.rms_horizontal_error < 2200.0,
        "RBPF RMS horizontal error should be less than 2200m, got {:.2}m",
        stats.rms_horizontal_error
    );
    assert!(
        stats.median_horizontal_error < 210.0,
        "RBPF median horizontal error should be less than 210m, got {:.2}m",
        stats.median_horizontal_error
    );
    assert!(
        stats.rms_altitude_error < 100.0,
        "RBPF RMS altitude error should be less than 100m, got {:.2}m",
        stats.rms_altitude_error
    );

    for result in &results {
        assert!(result.latitude.is_finite());
        assert!(result.longitude.is_finite());
        assert!(result.altitude.is_finite());
    }
}

/// Test RBPF with degraded GNSS measurements.
///
/// Re-enabled when #267 landed. Note the ~200 s runtime: this is the suite's
/// long pole by design (5000 particles over 10.7k events), kept because it is
/// the only test exercising the particle filter under faulted, sparse aiding.
#[test]
fn test_rbpf_with_degraded_gnss() {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let test_data_path = Path::new(manifest_dir).join("tests/test_data.csv");
    let records = load_test_data(&test_data_path);

    assert!(
        !records.is_empty(),
        "Test data should contain at least one record"
    );

    let cfg = {
        let mut built = AidingConfig::default();
        built.scheduler = MeasurementScheduler::FixedInterval {
            interval_s: 5.0,
            phase_s: 0.0,
        };
        built.fault = GnssFaultModel::Degraded {
            rho_pos: 0.99,
            sigma_pos_m: 3.0,
            rho_vel: 0.95,
            sigma_vel_mps: 0.3,
            r_scale: 5.0,
            tau_pos_s: None,
            tau_vel_s: None,
        };
        built
    };

    let results = run_rbpf_with_cfg(&records, &cfg, {
        let mut built = RbpfConfig::default();
        built.num_particles = RBPF_DEGRADED_PARTICLES;
        built.seed = 42;
        // Proposal matched to the fault scale: the AR(1) wander
        // (sigma_pos_m 3.0, quasi-bias ±20 m) over 5 s fixes starves the
        // default 1 m proposal cloud (see #267). Explicit here rather
        // than in the default: a wider default proposal measurably
        // degrades clean stationary tracking.
        built.position_process_noise_std_m = Vector3::new(3.0, 3.0, 3.0);
        built
    });
    assert!(!results.is_empty(), "RBPF should produce results");

    let stats = compute_error_metrics(&results, &records);

    println!("\n=== RBPF Degraded GNSS Error Statistics ===");
    println!(
        "Horizontal Error: mean={:.2}m, min={:.2}m, median={:.2}m, max={:.2}m, rms={:.2}m",
        stats.mean_horizontal_error,
        stats.min_horizontal_error,
        stats.median_horizontal_error,
        stats.max_horizontal_error,
        stats.rms_horizontal_error
    );
    println!(
        "Altitude Error: mean={:.2}m, min={:.2}m, median={:.2}m, max={:.2}m, rms={:.2}m",
        stats.mean_altitude_error,
        stats.min_altitude_error,
        stats.median_altitude_error,
        stats.max_altitude_error,
        stats.rms_altitude_error
    );

    // Bounds carry ~3-3.5x margin over the observed healthy values
    // (rms_h 204 m, median_h 70 m, rms_alt 8.4 m at seed 42; medians 117-136 m
    // across seeds 1,2,3,7,123 at 500 particles). The old 2200 m rms ceiling
    // was vacuous -- any non-divergent filter passed -- and is replaced with a
    // guard that still trips on the pre-#267 behaviour (km-scale medians).
    assert!(
        stats.rms_horizontal_error < 600.0,
        "RBPF RMS horizontal error with degraded GNSS should be less than 600m, got {:.2}m",
        stats.rms_horizontal_error
    );
    assert!(
        stats.median_horizontal_error < 250.0,
        "RBPF median horizontal error with degraded GNSS should be less than 250m, got {:.2}m",
        stats.median_horizontal_error
    );
    assert!(
        stats.rms_altitude_error < 30.0,
        "RBPF RMS altitude error with degraded GNSS should be less than 30m, got {:.2}m",
        stats.rms_altitude_error
    );

    for result in &results {
        assert!(result.latitude.is_finite());
        assert!(result.longitude.is_finite());
        assert!(result.altitude.is_finite());
    }
}

/// Test that filter output length matches input data length
///
/// This test verifies that all filter implementations (UKF, EKF, ESKF, dead reckoning)
/// produce output with the same number of records as the input data. This is critical for
/// downstream analysis tools that expect aligned data streams.
#[test]
fn test_filter_output_length_matches_input() {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let test_data_path = Path::new(manifest_dir).join("tests/test_data.csv");
    let records = load_test_data(&test_data_path);

    assert!(
        !records.is_empty(),
        "Test data should contain at least one record"
    );

    let input_length = records.len();
    println!("Testing with {input_length} input records");

    // Test dead reckoning. One output per input is a property of the loop, not of the arc's
    // length, so this uses the baseline window: exactly one test in this file runs the
    // 89-minute unaided arc (`test_dead_reckoning_on_real_data`), and it is the one written to
    // tolerate either arithmetic outcome. See `DEAD_RECKONING_BASELINE_SAMPLES`.
    let (dr_results, _, window) = dead_reckoning_baseline(&records);
    assert_eq!(
        dr_results.len(),
        window,
        "Dead reckoning output length {} should match input length {window}",
        dr_results.len()
    );
    println!(
        "✓ Dead reckoning: {} outputs for {window} inputs",
        dr_results.len()
    );

    // Create initial state from first record
    let initial_state = create_initial_state(&records[0]);
    let imu_biases = vec![0.0; 6]; // Zero initial bias estimates
    let initial_covariance = DEFAULT_INITIAL_COVARIANCE.to_vec();
    let process_noise =
        DMatrix::from_diagonal(&DVector::from_vec(DEFAULT_PROCESS_NOISE_DENSITY.to_vec()));
    let degradation = AidingConfig::default();

    // Test UKF
    let mut ukf = UnscentedKalmanFilter::new(
        &initial_state,
        &imu_biases,
        None, // No measurement bias
        initial_covariance.clone(),
        process_noise.clone(),
        1e-3, // alpha
        2.0,  // beta
        0.0,  // kappa
    );

    let event_stream = build_event_stream(&records, &degradation, TEST_DATA_IS_ENU).unwrap();
    let ukf_results = run_closed_loop(&mut ukf, event_stream, None, None)
        .expect("UKF closed loop should complete successfully");

    assert_eq!(
        ukf_results.len(),
        input_length,
        "UKF output length {} should match input length {}",
        ukf_results.len(),
        input_length
    );
    println!(
        "✓ UKF: {} outputs for {} inputs",
        ukf_results.len(),
        input_length
    );

    // Test EKF
    let mut ekf = ExtendedKalmanFilter::new(
        &initial_state,
        &imu_biases,
        initial_covariance,
        process_noise,
        true,
    );

    let event_stream = build_event_stream(&records, &degradation, TEST_DATA_IS_ENU).unwrap();
    let ekf_results = run_closed_loop(&mut ekf, event_stream, None, None)
        .expect("EKF closed loop should complete successfully");

    assert_eq!(
        ekf_results.len(),
        input_length,
        "EKF output length {} should match input length {}",
        ekf_results.len(),
        input_length
    );
    println!(
        "✓ EKF: {} outputs for {} inputs",
        ekf_results.len(),
        input_length
    );

    // ESKF length coverage lives in test_eskf_output_stays_valid_across_full_run, which
    // asserts one solution per input record on the same data. This comment previously
    // named `test_eskf_output_length_matches_input`, which is not in this file and does
    // not appear to have ever been -- so until that assertion was added the ESKF had no
    // absolute length coverage anywhere, only the relative check in
    // `test_filter_comparison`.

    println!("\n✅ All filters produce output length matching input length: {input_length}");
}

// ===================== v1.0 validation suite (#264) =========================================
//
// The tests below are the queue 8 deliverable: a benchmark across filters, a reproducibility
// check, outage recovery, and the full operational lifecycle driven through `InsEngine`. The
// older single-filter tests above remain the per-filter regression guards.

/// Run a filter over the whole undegraded stream and return its results.
///
/// Every benchmark leg differs only in the filter it drives, so the stream construction and
/// the closed-loop call live here rather than being repeated per filter.
fn run_filter_on_clean_stream<F: NavigationFilter>(
    filter: &mut F,
    records: &[TestDataRecord],
) -> Vec<NavigationResult> {
    let cfg = {
        let mut built = AidingConfig::default();
        built.scheduler = MeasurementScheduler::PassThrough;
        built.fault = GnssFaultModel::None;
        built
    };
    let stream = build_event_stream(records, &cfg, TEST_DATA_IS_ENU).unwrap();
    run_closed_loop(filter, stream, None, None)
        .unwrap_or_else(|error| panic!("filter should complete the clean stream: {error}"))
}

/// Build a UKF on the shared default tuning.
fn build_ukf(initial_state: &InitialState) -> UnscentedKalmanFilter {
    UnscentedKalmanFilter::new(
        initial_state,
        &[0.0; 6],
        None,
        DEFAULT_INITIAL_COVARIANCE.to_vec(),
        DMatrix::from_diagonal(&DVector::from_vec(DEFAULT_PROCESS_NOISE_DENSITY.to_vec())),
        1e-3,
        2.0,
        0.0,
    )
}

/// Build an EKF on the shared default tuning.
fn build_ekf(initial_state: &InitialState) -> ExtendedKalmanFilter {
    ExtendedKalmanFilter::new(
        initial_state,
        &[0.0; 6],
        DEFAULT_INITIAL_COVARIANCE.to_vec(),
        DMatrix::from_diagonal(&DVector::from_vec(DEFAULT_PROCESS_NOISE_DENSITY.to_vec())),
        true,
    )
}

/// Build an ESKF on its own tuning.
fn build_eskf(initial_state: &InitialState) -> ErrorStateKalmanFilter {
    ErrorStateKalmanFilter::new(
        initial_state,
        &[0.0; 6],
        ESKF_INITIAL_COVARIANCE.to_vec(),
        DMatrix::from_diagonal(&DVector::from_vec(ESKF_PROCESS_NOISE.to_vec())),
    )
}

/// The magnetometer heading's own error on this dataset, measured without a filter (#305).
///
/// [`MAX_YAW_RMSE_RAD`] is derived from this number, so this test is what stops that
/// derivation becoming fiction if `test_data.csv` is ever replaced -- the same job
/// [`assert_reference_accuracy_matches_dataset`] does for the GNSS accuracies.
///
/// The measurement is evaluated at the *reference* attitude rather than at any filter's
/// estimate, which is what makes it a property of the sensor and the vehicle's magnetic
/// environment rather than of the estimator. A filter reading only this source cannot beat it
/// except by smoothing.
///
/// It also pins the frame, and that half is the actual #305 regression guard: the same records
/// evaluated on the NED branch give 110.8 deg rather than 17.1 deg. That gap is the whole
/// defect -- the model computed one fixed branch regardless of the state it was updating, so
/// on this ENU dataset it returned pi/2 - psi and every filter converged on the reflection.
/// A single `assert!(rms < something)` would not have caught it, because 110.8 deg and the
/// 95.1 deg the original code produced are both just "large"; asserting that the *matching*
/// branch is small and the *mismatched* branch is large is what makes the orientation
/// explicit.
#[test]
fn magnetometer_yaw_aiding_source_error_matches_derivation() {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let test_data_path = Path::new(manifest_dir).join("tests/test_data.csv");
    let records = load_test_data(&test_data_path);
    assert!(!records.is_empty(), "test data should not be empty");

    let mut squared_error: [f64; 2] = [0.0, 0.0];
    let mut signed_error_sum = 0.0;
    let mut count = 0usize;

    for record in &records {
        if [record.mag_x, record.mag_y, record.mag_z]
            .iter()
            .any(|component| component.is_nan())
        {
            continue;
        }
        let (roll, pitch, reference_yaw) = reference_attitude(record);
        let state = DVector::from_vec(vec![
            record.latitude.to_radians(),
            record.longitude.to_radians(),
            record.altitude,
            0.0,
            0.0,
            0.0,
            roll,
            pitch,
            reference_yaw,
        ]);
        // Index 0 is the frame the dataset is actually in, index 1 the other one.
        for (slot, is_enu) in [TEST_DATA_IS_ENU, !TEST_DATA_IS_ENU]
            .into_iter()
            .enumerate()
        {
            let measurement = MagnetometerYawMeasurement {
                mag_x: record.mag_x,
                mag_y: record.mag_y,
                mag_z: record.mag_z,
                noise_std: MAG_YAW_NOISE,
                apply_declination: true,
                year: record.time.year(),
                day_of_year: record.time.ordinal() as u16,
                is_enu,
            };
            let heading = measurement
                .get_measurement(&state)
                .expect("the magnetometer model is analytic and cannot fail on a valid state")[0];
            let error = wrap_to_pi(heading - reference_yaw);
            squared_error[slot] += error * error;
            if slot == 0 {
                signed_error_sum += error;
            }
        }
        count += 1;
    }

    assert!(
        count > 5000,
        "expected the full recording, got {count} usable magnetometer records"
    );
    let matching_rms = (squared_error[0] / count as f64).sqrt();
    let mismatched_rms = (squared_error[1] / count as f64).sqrt();
    let mean_bias = signed_error_sum / count as f64;

    println!(
        "magnetometer heading vs reference attitude over {count} records: matching frame \
         {:.2} deg RMS (bias {:.2} deg), mismatched frame {:.2} deg RMS",
        matching_rms.to_degrees(),
        mean_bias.to_degrees(),
        mismatched_rms.to_degrees()
    );

    assert!(
        (matching_rms - MAG_YAW_SOURCE_RMSE_RAD).abs() < 0.5_f64.to_radians(),
        "MAG_YAW_SOURCE_RMSE_RAD is documented as {:.2} deg but the dataset now measures \
         {:.2} deg; MAX_YAW_RMSE_RAD is derived from it and needs revisiting",
        MAG_YAW_SOURCE_RMSE_RAD.to_degrees(),
        matching_rms.to_degrees()
    );

    // The mean bias is residual hard iron from the vehicle, not noise: the field rotated into
    // the navigation frame implies a magnetic north of -8.21 deg against a WMM declination of
    // -11.40 deg. Recorded because it is the reason the filters cannot do much better than
    // 16 deg, and because a bias that grows means the heading aid has acquired an offset.
    assert!(
        mean_bias.abs() < 6.0_f64.to_radians(),
        "magnetometer heading bias is {:.2} deg, up from the 3.07 deg this dataset showed; \
         the heading aid has acquired an offset",
        mean_bias.to_degrees()
    );

    // The orientation check. Not a tuned threshold: 45 deg is the midpoint of the reflection
    // the wrong branch performs, so anything at or above it cannot be a matching frame.
    assert!(
        mismatched_rms > 45.0_f64.to_radians(),
        "evaluating the magnetometer on the {} branch should be badly wrong on {} data, but it \
         measures {:.2} deg -- either the frames have stopped differing or TEST_DATA_IS_ENU no \
         longer describes this dataset",
        if TEST_DATA_IS_ENU { "NED" } else { "ENU" },
        if TEST_DATA_IS_ENU { "ENU" } else { "NED" },
        mismatched_rms.to_degrees()
    );
}

/// Check the documented reference accuracies still describe the dataset.
///
/// The module documentation derives the error floor and the sample-alignment term from three
/// numbers measured off `test_data.csv`. If the dataset is ever replaced those numbers become
/// fiction, and the bounds derived from them stop meaning what they say. This fails loudly
/// instead.
fn assert_reference_accuracy_matches_dataset(records: &[TestDataRecord]) {
    let mean_of = |pick: fn(&TestDataRecord) -> f64| {
        let values: Vec<f64> = records.iter().map(pick).filter(|v| v.is_finite()).collect();
        summarize(&values).mean
    };

    for (label, measured, documented) in [
        (
            "horizontal accuracy",
            mean_of(|r| r.horizontal_accuracy),
            GNSS_REPORTED_HORIZONTAL_ACCURACY_M,
        ),
        (
            "vertical accuracy",
            mean_of(|r| r.vertical_accuracy),
            GNSS_REPORTED_VERTICAL_ACCURACY_M,
        ),
        ("ground speed", mean_of(|r| r.speed), MEAN_GROUND_SPEED_MPS),
    ] {
        assert!(
            (measured - documented).abs() < 0.05,
            "documented mean {label} is {documented:.2} but the dataset now measures              {measured:.2}; the derived bounds in the module documentation need revisiting"
        );
    }
}

/// Benchmark horizontal, vertical and attitude RMSE across the filters (#264).
///
/// Reports all three channels the v1.0 criteria name, side by side, for every filter that
/// tracks on this dataset. The per-filter tests above each assert on one filter in isolation;
/// what this adds is the comparison, which is where a filter that has quietly regressed
/// relative to its peers shows up.
///
/// All four filters are covered. The EKF was excluded when this benchmark was written because
/// it diverged to ~14,707 km on this dataset (#307); with the Euler-angle correction to
/// `state_transition_jacobian` it tracks the others and is back in the table.
#[test]
fn test_rmse_benchmark_across_filters() {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let test_data_path = Path::new(manifest_dir).join("tests/test_data.csv");
    let records = load_test_data(&test_data_path);
    assert!(!records.is_empty(), "test data should not be empty");
    assert_reference_accuracy_matches_dataset(&records);

    let initial_state = create_initial_state(&records[0]);

    let mut ukf = build_ukf(&initial_state);
    let ukf_stats =
        compute_error_metrics(&run_filter_on_clean_stream(&mut ukf, &records), &records);

    let mut eskf = build_eskf(&initial_state);
    let eskf_stats =
        compute_error_metrics(&run_filter_on_clean_stream(&mut eskf, &records), &records);

    let mut ekf = build_ekf(&initial_state);
    let ekf_stats =
        compute_error_metrics(&run_filter_on_clean_stream(&mut ekf, &records), &records);

    let rbpf_stats = compute_error_metrics(&run_rbpf(&records), &records);

    let benchmark = [
        ("UKF", &ukf_stats),
        ("EKF", &ekf_stats),
        ("ESKF", &eskf_stats),
        ("RBPF", &rbpf_stats),
    ];

    println!("\n=== RMSE benchmark (#264) ===");
    println!("reference: GNSS fix, 3.81 m horizontal / 1.38 m vertical 1-sigma (see module docs)");
    println!(
        "{:<6} {:>9} {:>9} {:>9} {:>9} {:>9} {:>10}",
        "filter", "horiz_m", "vert_m", "roll_deg", "pitch_deg", "yaw_deg", "geodesic_deg"
    );
    for (name, stats) in benchmark {
        println!(
            "{:<6} {:>9.2} {:>9.2} {:>9.3} {:>9.3} {:>9.3} {:>10.3}",
            name,
            stats.rms_horizontal_error,
            stats.rms_altitude_error,
            stats.rms_roll_error.to_degrees(),
            stats.rms_pitch_error.to_degrees(),
            stats.rms_yaw_error.to_degrees(),
            stats.rms_attitude_error.to_degrees()
        );
    }

    for (name, stats) in benchmark {
        assert!(
            stats.rms_horizontal_error < MAX_HORIZONTAL_RMSE_M,
            "{name} horizontal RMSE should be under {MAX_HORIZONTAL_RMSE_M} m, got {:.2} m",
            stats.rms_horizontal_error
        );
        assert!(
            stats.rms_altitude_error < MAX_VERTICAL_RMSE_M,
            "{name} vertical RMSE should be under {MAX_VERTICAL_RMSE_M} m, got {:.2} m",
            stats.rms_altitude_error
        );
        for (axis, rms) in [
            ("roll", stats.rms_roll_error),
            ("pitch", stats.rms_pitch_error),
        ] {
            assert!(
                rms < MAX_LEVEL_ATTITUDE_RMSE_RAD,
                "{name} {axis} RMSE should be under {:.1} deg, got {:.2} deg",
                MAX_LEVEL_ATTITUDE_RMSE_RAD.to_degrees(),
                rms.to_degrees()
            );
        }
    }

    // Yaw, on the filters whose attitude representation can carry it (#305).
    //
    // One filter is excluded, and the exclusion is the point of this being a separate loop
    // rather than a third entry in the one above:
    //
    // * **UKF** -- #371: it averages sigma-point Euler angles linearly under non-convex
    //   weights, so its yaw is not the mean rotation and cannot be relied on regardless of
    //   what the aiding does. It measures 22.77 deg here, which *would* pass; asserting it
    //   would be asserting that this recording happens to be kind to that defect, not that
    //   the filter holds heading. (#336, the branch-cut half of the same area, is closed --
    //   the unwrap it added is in `kalman.rs` and is not what this exclusion waits on.)
    //   Restore it to the list when #371 lands.
    //
    // The **RBPF** was excluded here too until #341, at 65.89 deg. That exclusion attributed
    // the number to #336's defect in RBPF form -- unwrapped Euler error states and a linear
    // weighted mean over a cloud straddling the cut -- and measuring it did not bear that out.
    // The cloud does cross the cut: on 140 of 21,460 steps at least one particle's assembled
    // yaw left [-pi, pi], and on 18 of them the linear mean did. But every particle shares one
    // nominal attitude and carries only a small error state on top of it, so the cloud crosses
    // as a body, and over the whole run the linear and circular means of its yaw never
    // differed by more than 2e-4 deg. A defect worth 2e-4 deg is not a 49 deg gap.
    //
    // The cause was that the magnetometer reached the filter through the particle weights
    // alone, which cannot carry a heading in a filter where yaw is a shared linear state:
    // every particle predicted the same heading, so 5,365 fixes moved the effective sample
    // size from 500 to a median of 492.6 and the estimate not at all. Routing it through the
    // Kalman branch (`RaoBlackwellizedParticleFilter::update_yaw_only`) took it to 15.66 deg,
    // at the aiding source's own floor and the best of the four. The wrap handling was fixed
    // alongside it as the latent defect it is, worth 2e-4 deg here.
    //
    // That 15.66 deg is not a seed: across seeds 1, 2, 3, 7, 42, 123 and 999 at 250, 500 and
    // 2000 particles it spans 15.65-15.69 deg. It should not vary, and this is the check that
    // it does not -- yaw now comes from the deterministic Kalman branch rather than from the
    // sampled cloud, so a result that moved with the seed would mean the heading had found
    // its way back into the weights.
    for (name, stats) in benchmark {
        if matches!(name, "UKF") {
            continue;
        }
        assert!(
            stats.rms_yaw_error < MAX_YAW_RMSE_RAD,
            "{name} yaw RMSE should be under {:.1} deg -- 1.5x the {:.1} deg the magnetometer \
             itself achieves on this recording, see MAX_YAW_RMSE_RAD -- got {:.2} deg. Yaw is \
             observable here only through the magnetometer, so a regression in this number is \
             most likely in the heading aid: check that the frame passed to `build_event_stream` \
             still matches the filters' own (#305).",
            MAX_YAW_RMSE_RAD.to_degrees(),
            MAG_YAW_SOURCE_RMSE_RAD.to_degrees(),
            stats.rms_yaw_error.to_degrees()
        );
    }

    // No filter may beat the reference it is scored against. Tripping this does not mean the
    // filter got better than GNSS -- it means the metric stopped measuring what it claims to,
    // most likely by scoring a result against the record it was derived from.
    //
    // This briefly excluded the UKF and EKF. #367 corrected the row labelling, which removed
    // the 21.2 m of along-track offset that had been standing between every estimate and the
    // fix it was scored against -- and underneath that offset both filters turned out to be
    // *copying* their fixes: 0.01 m and 0.0001 m against a reference specified at 3.81 m, a
    // Kalman gain of ~1. #373 found why (an absolute `1e-9` covariance floor against position
    // variances in rad^2, worth a 201 m horizontal sigma) and fixed it, so all four filters
    // are asserted again.
    for (name, stats) in benchmark {
        assert!(
            stats.rms_horizontal_error > GNSS_REPORTED_HORIZONTAL_ACCURACY_M,
            "{name} horizontal RMSE of {:.2} m is below the {GNSS_REPORTED_HORIZONTAL_ACCURACY_M} m \
             accuracy of the reference itself, which means the comparison is no longer valid",
            stats.rms_horizontal_error
        );
    }
}

/// Every filter must reproduce its output exactly when re-run on the same input (#264).
///
/// Reproducibility is a stated v1.0 requirement, and a published error metric is only
/// meaningful if the same configuration reproduces it. The comparison is exact rather than
/// approximate on purpose: these filters are deterministic code driven by a seeded RNG, so any
/// run-to-run difference is a defect -- an unseeded generator, iteration over a hash
/// container, uninitialised state -- and not numerical noise to be tolerated.
///
/// Comparing the serialised results rather than a chosen set of fields means covariances and
/// bias estimates are covered too, and that a field added later is covered without this test
/// being updated.
#[test]
fn test_filters_are_deterministic_across_runs() {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let test_data_path = Path::new(manifest_dir).join("tests/test_data.csv");
    let records = load_test_data(&test_data_path);
    let initial_state = create_initial_state(&records[0]);

    let mut ukf_a = build_ukf(&initial_state);
    let mut ukf_b = build_ukf(&initial_state);
    let mut eskf_a = build_eskf(&initial_state);
    let mut eskf_b = build_eskf(&initial_state);

    let runs: [(&str, Vec<NavigationResult>, Vec<NavigationResult>); 3] = [
        (
            "UKF",
            run_filter_on_clean_stream(&mut ukf_a, &records),
            run_filter_on_clean_stream(&mut ukf_b, &records),
        ),
        (
            "ESKF",
            run_filter_on_clean_stream(&mut eskf_a, &records),
            run_filter_on_clean_stream(&mut eskf_b, &records),
        ),
        // The RBPF is the one that could plausibly differ: it draws from an RNG every step.
        // `run_rbpf` pins the seed, so two runs must still agree exactly.
        ("RBPF", run_rbpf(&records), run_rbpf(&records)),
    ];

    for (name, first, second) in runs {
        assert_eq!(
            first.len(),
            second.len(),
            "{name} produced different result counts across runs"
        );

        let first_json = serde_json::to_string(&first).expect("results should serialize");
        let second_json = serde_json::to_string(&second).expect("results should serialize");

        if first_json != second_json {
            // Name the first sample that differs; a bare "not equal" on an 89-minute run is
            // not enough to start debugging from.
            let divergence = first
                .iter()
                .zip(&second)
                .position(|(a, b)| serde_json::to_string(a).ok() != serde_json::to_string(b).ok())
                .unwrap_or(0);
            panic!(
                "{name} is not reproducible: runs first differ at sample {divergence} \
                 (t = {}). Deterministic output is a v1.0 requirement.",
                first[divergence].timestamp
            );
        }
        println!("{name}: {} samples reproduced exactly", first.len());
    }
}

/// Stationary detector tuning for this dataset's 1 Hz sampling.
///
/// [`StationaryConfig::default`] is written for 100 Hz: a 100-sample window and 50 latching
/// samples are 1 s and 0.5 s there, but 100 s and 50 s at 1 Hz, which is longer than any stop
/// in this recording (the longest is 36 s). Scaled down to a 5 s window latching after 3 s,
/// the detector sees the stops that are actually present. The thresholds are left at their
/// defaults -- they are about sensor noise, not sample rate.
fn stationary_config_for_1hz() -> StationaryConfig {
    StationaryConfig {
        window: 5,
        min_stationary_samples: 3,
        ..StationaryConfig::default()
    }
}

/// The full operational lifecycle through `InsEngine` (#264).
///
/// Exercises, in one run over the real recording: levelling from the first record, inertial
/// dead reckoning between fixes, GNSS fusion with an innovation gate, ZUPT applied whenever
/// the stationary detector fires, a deliberate outage, and recovery once fixes resume. This
/// is the assembly the per-component tests do not cover -- each piece has its own unit tests
/// in `engine`, `gating` and `stationary`; what can only be tested here is that they compose
/// over an 89-minute drive without one of them undoing another.
///
/// Coarse alignment proper is not part of this: it lands with queue 103 (#282, still open),
/// so the engine is levelled from the first record's attitude the same way the rest of this
/// suite seeds its filters. When #282 merges this test should start from a genuine alignment.
///
/// One number here needs reading carefully. The aided error of ~2.3 m is *below* the 3.81 m
/// accuracy of the reference, which is impossible for a real accuracy figure. It happens
/// because this loop scores each solution against the very fix it has just consumed, so it
/// measures how tightly the engine follows its aiding rather than how close it is to truth.
/// That is the right quantity for the comparison this test makes -- aided against coasting
/// against recovered, all measured the same way -- but it is not an accuracy result.
/// `test_rmse_benchmark_across_filters` is where accuracy is reported.
///
/// # Gating history: quarantined by #308, reinstated by #340
///
/// This test passed originally only because the process noise was wrong. With the horizontal
/// terms at #308's 6.4 km per-step standard deviation (18 km here, after the old 8x), the
/// innovation covariance was so large that a chi-squared gate could not reject anything it
/// was shown: NIS sat at a median of 0.27 against a 3-dof gate at 16.27, and 39 of 5,365
/// fixes were gated across the whole drive. Correcting the units made the gate work, and the
/// first thing it did was prove itself unusable here.
///
/// | run | accepted | rejected | NIS median | final horizontal error |
/// |---|---|---|---|---|
/// | ungated | 5,365 | 0 | 0.553 | 2.29 m |
/// | gated, no recovery (#308 units, #340 defect) | 116 | 5,249 | 242.8 | 4.2e6 m |
/// | gated, with recovery (today) | 5,155 | 90 | -- | 2.29 m aided, 2.16 m rms recovered |
///
/// The middle row is #340. The first rejection was fix #110, on a genuine 19.3 m innovation
/// that a filter claiming 0.8 m of its own uncertainty against a 3.81 m fix is right to
/// disbelieve. What followed had no bottom: pre-update innovation 19.3, 32.0, 46.3, 61.8,
/// 77.3, 92.1, 105.9, 117.9 m over the next eight fixes, every one rejected, forever, because
/// nothing in the gating path re-inflated the covariance after a rejection. The old `Q`
/// cascaded the same way at fix #1487 and *escaped*, because adding (18 km)^2 to the
/// covariance once per step is an accidental covariance reset -- so the units defect was
/// supplying the recovery path the gating code did not have, and removing it is the point of
/// #308.
///
/// It could not be fixed from here, which is why this test was quarantined rather than tuned
/// around, per the #267 precedent: sweeping the position process noise showed the only values
/// that kept the run green were 10 m and above, where NIS collapses to a median of 0.017 and
/// then 0.002 -- #308 restored under another name. The fix is `GateRecovery`: the covariance
/// is inflated on every rejection and an update is forced through after five consecutive
/// ones, so a rejection can no longer be permanent. The third row is this test with it.
///
/// It remains the only end-to-end exercise of `InsEngine` together with innovation gating --
/// every other filter test calls `run_closed_loop(&mut f, stream, None, None)` with no gate
/// and no health monitor -- which is why the assertions below check that fixes were both
/// accepted *and* rejected. A run with nothing rejected is not evidence that the gate works.
#[test]
fn test_full_lifecycle_through_ins_engine() {
    // The outage: two minutes without fixes in the middle of the drive.
    //
    // Two minutes, not ten. Unaided MEMS dead reckoning diverges quadratically, and on this
    // recording ten minutes puts the solution ~167 km out -- a number that says nothing about
    // whether the engine works, only that consumer inertial sensors are what they are. Two
    // minutes is both a realistic denial (a tunnel, an underpass) and short enough that the
    // drift stays within a bound that can be derived rather than observed.
    const OUTAGE_START_S: f64 = 1800.0;
    const OUTAGE_DURATION_S: f64 = 120.0;
    const OUTAGE_END_S: f64 = OUTAGE_START_S + OUTAGE_DURATION_S;
    // Settled-aiding window, ending where the outage begins.
    const SETTLED_FROM_S: f64 = 1500.0;
    // Recovery window, starting well after fixes resume so reconvergence has had time.
    const RECOVERY_FROM_S: f64 = 2700.0;
    const RECOVERY_TO_S: f64 = 3000.0;

    // Cap on the residual horizontal specific-force error used to bound unaided drift below.
    const MAX_RESIDUAL_ACCELERATION_MPS2: f64 = 4.0;

    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let test_data_path = Path::new(manifest_dir).join("tests/test_data.csv");
    let records = load_test_data(&test_data_path);
    assert!(
        records.len() > 3000,
        "lifecycle test needs the full recording"
    );

    let config = {
        let mut built = InsEngineConfig::default();
        built.is_enu = true;
        // A plausible vehicle geometry: antenna 1.5 m forward of the IMU and 1 m above it.
        // Non-zero so the lever-arm path is actually exercised rather than short-circuited.
        built.lever_arm = [1.5, 0.0, 1.0];
        built.process_noise_diagonal = Some(ESKF_PROCESS_NOISE.to_vec());
        built.initial_covariance_diagonal = Some(ESKF_INITIAL_COVARIANCE.to_vec());
        built
    };

    let mut engine = InsEngine::builder()
        .with_config(config)
        .with_initial_state(create_initial_state(&records[0]))
        .build()
        .expect("engine should build from a valid configuration");

    // Reject the worst 0.1% of fixes the filter's own model predicts.
    assert!(
        engine.set_innovation_gate(Some(
            InnovationGate::chi_squared(0.999).expect("0.999 is a valid confidence")
        )),
        "the default ESKF should honour an innovation gate"
    );

    let mut detector = StationaryDetector::new(stationary_config_for_1hz());
    let zupt = ZuptMeasurement::default();

    let start = records[0].time;
    let mut zupt_applications = 0_usize;
    let mut gnss_accepted = 0_usize;
    let mut gnss_rejected = 0_usize;
    let mut errors_while_aided = Vec::new();
    let mut errors_during_outage = Vec::new();
    let mut errors_after_recovery = Vec::new();

    for pair in records.windows(2) {
        let (previous, record) = (&pair[0], &pair[1]);
        let dt = (record.time - previous.time).num_milliseconds() as f64 / 1000.0;
        if !(dt > 0.0 && dt.is_finite()) {
            continue;
        }
        let elapsed = (record.time - start).num_milliseconds() as f64 / 1000.0;

        let imu = IMUData {
            accel: Vector3::new(record.acc_x, record.acc_y, record.acc_z),
            gyro: Vector3::new(record.gyro_x, record.gyro_y, record.gyro_z),
        };
        let sample = ImuSample::from_rates(&imu, dt);

        engine.predict(&sample).expect("propagation should succeed");

        // ZUPT whenever the vehicle is judged stationary, independent of GNSS. This is the
        // aiding that keeps the solution bounded through the outage.
        if detector.push(&imu) {
            engine.update(&zupt).expect("ZUPT should apply");
            zupt_applications += 1;
        }

        let in_outage = (OUTAGE_START_S..OUTAGE_END_S).contains(&elapsed);
        if !in_outage && record.latitude.is_finite() && record.longitude.is_finite() {
            let (velocity_north, velocity_east) = record.ground_track_velocity();
            let fix = GnssFix::position(
                record.latitude,
                record.longitude,
                record.altitude,
                record.horizontal_accuracy,
                record.vertical_accuracy,
            )
            .with_velocity([velocity_north, velocity_east, 0.0], 1.0);

            let outcome = engine
                .update_gnss(&fix)
                .expect("GNSS update should succeed");
            if outcome.accepted {
                gnss_accepted += 1;
            } else {
                gnss_rejected += 1;
            }
        }

        let solution = engine.nav_solution();
        if record.latitude.is_finite() && record.longitude.is_finite() {
            let error = haversine_distance(
                solution.latitude.to_radians(),
                solution.longitude.to_radians(),
                record.latitude.to_radians(),
                record.longitude.to_radians(),
            );
            if error.is_finite() {
                if in_outage {
                    errors_during_outage.push(error);
                } else if (SETTLED_FROM_S..OUTAGE_START_S).contains(&elapsed) {
                    errors_while_aided.push(error);
                } else if (RECOVERY_FROM_S..RECOVERY_TO_S).contains(&elapsed) {
                    errors_after_recovery.push(error);
                }
            }
        }
    }

    let aided = summarize(&errors_while_aided);
    let outage = summarize(&errors_during_outage);
    let recovered = summarize(&errors_after_recovery);

    println!("\n=== InsEngine full lifecycle ===");
    println!("ZUPT applications: {zupt_applications}");
    println!("GNSS fixes: {gnss_accepted} accepted, {gnss_rejected} gated out");
    for (label, stats) in [
        (
            format!("aided     ({SETTLED_FROM_S:.0}-{OUTAGE_START_S:.0} s)"),
            aided,
        ),
        (
            format!("outage    ({OUTAGE_START_S:.0}-{OUTAGE_END_S:.0} s)"),
            outage,
        ),
        (
            format!("recovered ({RECOVERY_FROM_S:.0}-{RECOVERY_TO_S:.0} s)"),
            recovered,
        ),
    ] {
        println!(
            "{label}: mean={:.2} m median={:.2} m max={:.2} m rms={:.2} m",
            stats.mean, stats.median, stats.max, stats.rms
        );
    }

    // Each stage must actually have happened. Without these the error assertions below could
    // pass on a run where the detector never fired or the gate swallowed every fix.
    assert!(
        zupt_applications > 0,
        "the stationary detector never fired, so ZUPT was never exercised; this recording \
         has 218 samples under 0.5 m/s and a 36 s stop, so a detector that sees none of them \
         is mistuned"
    );
    assert!(
        gnss_accepted > 1000,
        "only {gnss_accepted} GNSS fixes were accepted; the gate is rejecting fixes it should \
         not, and the lifecycle is not being exercised as intended"
    );
    // The other direction, and the reason this test exists in this file at all: a gate that
    // rejects nothing on 89 minutes of consumer GNSS is not a gate. Measured here, 90 of
    // 5,245 fixes are gated out and the rest are used, which is the behaviour #340 restored
    // -- before it, the same configuration accepted 116 and rejected 5,249.
    assert!(
        gnss_rejected > 0,
        "no GNSS fix was gated out across the whole drive; the gate is installed but is not \
         deciding anything, so every gating assertion here is vacuous"
    );
    assert!(
        !errors_during_outage.is_empty() && !errors_after_recovery.is_empty(),
        "the outage and recovery windows must both contain samples"
    );

    // Unaided drift is bounded by the dead-reckoning model rather than by an observed number.
    // A residual horizontal specific-force error `a` -- accelerometer bias plus the component
    // of gravity that a tilt error leaks into the horizontal axes -- integrates twice into
    // `0.5 * a * t^2`. Measured on this recording the effective `a` is about 1.8 m/s^2 over
    // this window; a 600 s outage reached 167 km, implying 0.93 m/s^2. The growth is faster
    // than quadratic because the attitude error is itself growing, which is why the shorter
    // window shows the larger effective figure. Capping `a` at 4.0 m/s^2 keeps roughly 2x
    // headroom on the worse of the two while staying far below what a filter that has come
    // apart produces, and unlike a fitted ceiling it rescales correctly if the outage length
    // is changed.
    let drift_bound_m = 0.5 * MAX_RESIDUAL_ACCELERATION_MPS2 * OUTAGE_DURATION_S.powi(2);
    assert!(
        outage.max < drift_bound_m,
        "coasting for {OUTAGE_DURATION_S:.0} s should leave the solution within \
         {drift_bound_m:.0} m (0.5 * {MAX_RESIDUAL_ACCELERATION_MPS2} m/s^2 * t^2), but it \
         reached {:.0} m; the engine is diverging rather than dead reckoning",
        outage.max
    );

    assert!(
        outage.max > aided.max,
        "peak error during the outage ({:.2} m) was no worse than while aided ({:.2} m), so \
         fixes are still reaching the filter and the outage is not being simulated",
        outage.max,
        aided.max
    );

    assert!(
        recovered.rms < 2.0 * aided.rms.max(1.0),
        "after the outage the engine should reconverge to within 2x its settled error \
         ({:.2} m), but it sits at {:.2} m rms",
        aided.rms,
        recovered.rms
    );
}

/// The ESKF must recover after GNSS returns, not just survive the outage (#264).
///
/// Reinstated with the `DutyCycle` fix (#312). It was written during queue 8 and removed
/// again when the scheduler turned out to deliver two fixes across the whole recording
/// regardless of configuration, which made every window here meaningless. It is the
/// end-to-end check that the fix actually schedules: if `DutyCycle` regresses to emitting
/// only at window boundaries, the aided windows below stop being aided and this fails.
///
/// Surviving an outage is the easy half, and the degraded-GNSS tests above already cover it.
/// What this adds is that error comes back *down* once fixes resume. A filter whose
/// covariance has collapsed, or whose biases have wound up, keeps coasting after aiding
/// returns and never reconverges -- and its whole-run RMSE can still look acceptable, because
/// most of the run is aided.
#[test]
fn test_eskf_recovers_from_gnss_outage() {
    // Two cycles of 25 minutes aided then 5 minutes dark, so recovery is shown to repeat
    // rather than being one lucky window. Five minutes rather than the lifecycle test's two:
    // `run_closed_loop` also feeds per-sample baro and mag aiding, which are not scheduled
    // and keep the vertical and heading channels from running open loop, so the horizontal
    // drift here is milder than pure coasting.
    const ON_S: f64 = 1500.0;
    const OFF_S: f64 = 300.0;

    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let test_data_path = Path::new(manifest_dir).join("tests/test_data.csv");
    let records = load_test_data(&test_data_path);

    let initial_state = create_initial_state(&records[0]);
    let mut eskf = build_eskf(&initial_state);

    let cfg = {
        let mut built = AidingConfig::default();
        built.scheduler = MeasurementScheduler::DutyCycle {
            on_s: ON_S,
            off_s: OFF_S,
            start_phase_s: ON_S,
        };
        built.fault = GnssFaultModel::None;
        built
    };
    let stream = build_event_stream(&records, &cfg, TEST_DATA_IS_ENU).unwrap();
    let results = run_closed_loop(&mut eskf, stream, None, None)
        .expect("ESKF should complete the duty-cycled stream");

    let start = records[0].time;
    // Horizontal error of every result whose elapsed time falls in `[from, to)`.
    let window = |from: f64, to: f64| -> Vec<f64> {
        results
            .iter()
            .filter(|result| {
                let elapsed = (result.timestamp - start).num_milliseconds() as f64 / 1000.0;
                elapsed >= from && elapsed < to
            })
            .filter_map(|result| {
                records
                    .iter()
                    .find(|record| record.time == result.timestamp)
                    .map(|record| {
                        haversine_distance(
                            result.latitude.to_radians(),
                            result.longitude.to_radians(),
                            record.latitude.to_radians(),
                            record.longitude.to_radians(),
                        )
                    })
            })
            .filter(|error| error.is_finite())
            .collect()
    };

    // Timeline: ON [0, 1500), OFF [1500, 1800), ON [1800, 3300), OFF [3300, 3600),
    // ON [3600, 5100), OFF [5100, end].
    //
    // Each "recovered" window is sampled at the same offset into its ON window as the
    // "aided" baseline is into the first one -- the last 300 s of it. Comparing a window
    // 300 s after aiding resumed against a baseline measured 1200 s in would penalise the
    // filter for settling time the baseline was given and the comparison was not, which is
    // a property of the windows rather than of the filter.
    let before = summarize(&window(1200.0, 1500.0));
    let outage = summarize(&window(1500.0, 1800.0));
    let recovered = summarize(&window(3000.0, 3300.0));
    let second_outage = summarize(&window(3300.0, 3600.0));
    let after_second = summarize(&window(4800.0, 5100.0));

    println!("\n=== ESKF GNSS outage recovery ===");
    for (label, stats) in [
        ("aided     (1200-1500 s)", before),
        ("outage 1  (1500-1800 s)", outage),
        ("recovered (3000-3300 s)", recovered),
        ("outage 2  (3300-3600 s)", second_outage),
        ("recovered (4800-5100 s)", after_second),
    ] {
        println!(
            "{label}: mean={:.2} m median={:.2} m max={:.2} m rms={:.2} m",
            stats.mean, stats.median, stats.max, stats.rms
        );
    }

    assert!(
        outage.max > before.max,
        "the outage should actually deprive the filter: peak error during the outage \
         ({:.2} m) was no worse than while aided ({:.2} m). Before #312 this failed because \
         `DutyCycle` withheld GNSS everywhere, making the aided windows the drifting ones",
        outage.max,
        before.max
    );

    // Recovery is the point, and it is asserted against the normal operating bound rather
    // than against the pre-outage window.
    //
    // The settled error is not the same in every ON window -- 15 m in the first, 31 m in the
    // second, 22 m in the third -- because the dominant term is the sample-alignment one
    // described in the module documentation, which scales with ground speed. The windows
    // cover different stretches of driving, so a ratio against one of them would be
    // measuring the route rather than the filter. `MAX_HORIZONTAL_RMSE_M` is the ceiling the
    // healthy filters are held to across the whole run, so returning beneath it *is* the
    // statement that the filter is operating normally again.
    for (label, stats) in [("first", recovered), ("second", after_second)] {
        assert!(
            stats.rms < MAX_HORIZONTAL_RMSE_M,
            "after the {label} outage the ESKF should be back under the {MAX_HORIZONTAL_RMSE_M} m \
             operating bound, but it sits at {:.2} m rms -- it is still coasting",
            stats.rms
        );
    }

    // And the recovery must be a collapse, not a drift back. Paired with the bound above --
    // recovered error back under the operating ceiling -- the statement that makes is: the
    // filter left the band a healthy filter occupies entirely, and came back inside it.
    //
    // This used to be a ratio, `recovered * 100 < outage`, justified as "far below the ~650x
    // actually observed". #305 is what that kind of threshold costs. Fixing the magnetometer's
    // frame gave the filter a heading to coast on, which cut the second outage's drift from
    // 6297 m to 1718 m while the recovered window stayed at 21.7 m -- so the ratio fell from
    // 290x to 79x and the assertion failed *because the filter got better*. The ratio was
    // never measuring recovery; it was measuring how far the filter flew off during the
    // outage, which is a property of the route, the outage length and the quality of the
    // coast. That is the same objection the comment above already raises against ratios
    // taken over a single window, and it applies here too.
    //
    // Both bounds are quantities the suite already derives:
    // [`DEAD_RECKONING_DIVERGENCE_FLOOR_M`] (400 m) is the point where unaided propagation
    // leaves the range a degraded-but-not-broken filter can reach, and
    // [`MAX_HORIZONTAL_RMSE_M`] (40 m) is the ceiling healthy filters hold. A filter that
    // merely stopped getting worse fails the second; one that was never actually deprived
    // fails the first.
    //
    // **Be honest about what the floor is**: it is a lower bound on how badly the filter
    // coasts, so it is exactly the kind of assertion that fails when the filter *improves* --
    // the same character as the ratio it replaced, and the same character as the floor in
    // `assert_beats_dead_reckoning`. It is not immune, it is only further away: outage 2 sits
    // at 1718 m against 400 m, and #305 just moved that number by 3.7x, so another improvement
    // of that size trips it. It earns its place anyway, because "the outage actually deprived
    // the filter" is a premise the recovery assertion needs and cannot check for itself, and
    // the failure message says what to do. The genuinely improvement-proof half of the pair is
    // `outage.max > before.max` above, which compares two windows of the same run.
    for (label, outage_stats) in [("first", outage), ("second", second_outage)] {
        assert!(
            outage_stats.rms > DEAD_RECKONING_DIVERGENCE_FLOOR_M,
            "the {label} outage should push the ESKF clear of the \
             {DEAD_RECKONING_DIVERGENCE_FLOOR_M} m band a healthy filter occupies, but it only \
             reached {:.0} m rms. Either GNSS is still reaching the filter, or coasting has \
             genuinely improved -- check `outage.max > before.max` above, which is the \
             improvement-proof form of the same question, and if that still passes then \
             lengthen the outage windows rather than lowering this floor",
            outage_stats.rms
        );
    }
}

/// Gating through `run_closed_loop`, the loop a simulation run actually goes through (#340).
///
/// Before #340 nothing tested this path with a gate installed: every other closed-loop test in
/// this file calls `run_closed_loop(&mut f, stream, None, None)` -- no gate -- so the gating
/// path was exercised end to end only by `test_full_lifecycle_through_ins_engine`, which drives
/// `InsEngine` directly and was itself quarantined on this very defect. This is the other half,
/// and it covers what the lifecycle test cannot: `run_closed_loop` feeds barometric altitude
/// and magnetometer yaw on *every* sample alongside 1 Hz GNSS, so it is where a recovery policy
/// has to get the interaction between several sensors right rather than just one.
///
/// `HealthMonitor` is live here -- `run_closed_loop` builds one from `HealthLimits::default()`
/// whether or not limits are passed -- so completing the run is itself an assertion: 20
/// consecutive NIS exceedances, a 500 m/s speed bound or a diverging covariance all abort.
///
/// # Quarantined: two compounding problems, neither of them the cascade
///
/// The #340 cascade *is* fixed here -- the run no longer walks off to 4.2e6 m -- but the run
/// still ends early, failed by the health monitor at 667.72 m/s, and the reasons sit outside
/// what a recovery policy can reach.
///
/// **This filter is over-confident for the aiding it is given here.** Ungated it is healthy --
/// 23.59 m rms, 41.71 m peak, tracking truth for 89 minutes -- and yet the 5-dof GNSS
/// position+velocity NIS has a median of 7.42 and a 90th percentile of 24.30 against a 0.999
/// threshold of 20.52 and an expected median of 4.35, while the 1-dof baro and magnetometer
/// updates sit at a median of 1.06 with a 90th percentile of 13.85 against a threshold of 10.83
/// and an expected median of 0.45. The innovations are about 1.7x larger than the covariance
/// says they should be, so an *honest* gate rejects roughly one measurement in seven on a run
/// that is doing fine. That is a covariance-consistency defect of the same family as #303, and
/// the fix is a retune measured against the validation suite, not a looser gate.
///
/// **The consecutive-rejection escape cannot be reached on this stream.** `GatePolicy` counts
/// one rejection streak across all sensors, and here two 1-dof updates arrive per sample and
/// are mostly accepted, so the streak is cleared before the 1 Hz GNSS channel can accumulate
/// the five rejections that would force an update. GNSS is left recovering on covariance
/// inflation alone, which is slower than the drift, and the vertical channel is what runs away
/// first.
///
/// Keying the streak per measurement type instead -- so the GNSS channel reaches its own escape
/// -- does fix *this* test (23.68 m rms against 23.59 m ungated, health monitor satisfied) and
/// breaks the lifecycle one, which is the acceptance criterion #340 is written against: gated
/// fixes go from 43 to 2,139 and post-outage reconvergence from 2.20 m rms to 102.14 m. The
/// reason is visible in the sweep -- forcing earlier means forcing with less accumulated
/// inflation behind it, so the forced update's gain is too small to correct the state that
/// caused the rejections. Getting both almost certainly means inflating *in proportion to the
/// streak* at the moment an update is forced, rather than choosing between the two counters,
/// and that is a design change with its own measurements to take. Quarantined rather than
/// tuned around, per the #267 precedent.
#[ignore = "gating through run_closed_loop needs two things #340 does not deliver: a filter \
            whose covariance matches its innovations on this stream (it over-states confidence \
            by ~1.7x, #303 family), and a consecutive-rejection escape that a 100 Hz accepted \
            sensor cannot starve. The #340 cascade itself is fixed; see the doc comment."]
#[test]
fn gating_through_the_closed_loop_no_longer_cascades() {
    // Sized against the table above, not fitted to it: the gated run sits at 1.03x the
    // ungated rms with a 94 m peak, while the #340 cascade reached 4.2e6 m and
    // whole-covariance inflation reached 719 m/s and a health-monitor abort.
    const MAX_GATED_RMSE_RATIO: f64 = 1.5;
    const MAX_GATED_PEAK_M: f64 = 2000.0;

    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let records = load_test_data(&Path::new(manifest_dir).join("tests/test_data.csv"));
    let initial_state = create_initial_state(&records[0]);
    let degradation = AidingConfig::default();

    let mut summaries = Vec::new();
    for gate in [
        None,
        Some(InnovationGate::chi_squared(0.999).expect("0.999 is a valid confidence")),
    ] {
        let mut eskf = build_eskf(&initial_state);
        assert!(
            eskf.set_innovation_gate(gate),
            "the ESKF must honour an innovation gate for this test to mean anything"
        );
        let stream = build_event_stream(&records, &degradation, TEST_DATA_IS_ENU)
            .expect("the clean stream should build");
        let results = run_closed_loop(&mut eskf, stream, None, None).unwrap_or_else(|error| {
            panic!(
                "the {} run did not survive its health monitor: {error}",
                if gate.is_some() { "gated" } else { "ungated" }
            );
        });

        let errors: Vec<f64> = results
            .iter()
            .filter_map(|result| {
                records
                    .iter()
                    .find(|record| record.time == result.timestamp)
                    .map(|record| {
                        haversine_distance(
                            result.latitude.to_radians(),
                            result.longitude.to_radians(),
                            record.latitude.to_radians(),
                            record.longitude.to_radians(),
                        )
                    })
            })
            .filter(|error| error.is_finite())
            .collect();
        assert!(
            !errors.is_empty(),
            "no comparable results came back from the closed-loop run"
        );
        summaries.push(summarize(&errors));
    }

    let (ungated, gated) = (summaries[0], summaries[1]);
    println!("\n=== gated vs ungated closed loop ===");
    println!(
        "ungated: rms={:.2} m max={:.2} m\ngated:   rms={:.2} m max={:.2} m",
        ungated.rms, ungated.max, gated.rms, gated.max
    );

    assert!(
        gated.rms < MAX_GATED_RMSE_RATIO * ungated.rms,
        "gating cost {:.2} m rms against {:.2} m ungated. A gate whose rejections compound is \
         #340, and inflating the whole covariance rather than the observed subspace is the \
         other way to get here; check `GateRecovery` and `GateDecision::inflate_observed` \
         before adjusting this bound",
        gated.rms,
        ungated.rms
    );
    assert!(
        gated.max < MAX_GATED_PEAK_M,
        "the gated run peaked at {:.0} m. The recovery path bounds how long a rejection can \
         stand; a peak this large means it is not bounding it",
        gated.max
    );
}

/// The opt-in `IMUQuality::auto_covariance` initialisation on the same recording (#257).
///
/// `auto_covariance` derives P0 from the IMU grade and the fix that positioned the vehicle
/// instead of the hand-picked constant `initialize_eskf` uses, so it deserves the same
/// end-to-end coverage as the default path in
/// `test_eskf_default_initialization_on_real_data`. `test_data.csv` is a phone recording, so
/// `IMUQuality::Consumer` is the honest grade and the receiver's own reported accuracy is the
/// honest position uncertainty.
///
/// Measured against the default constant, holding process noise and everything else fixed:
/// horizontal rms and peak are identical to two decimal places (23.72 m / 41.87 m), altitude
/// rms is identical at 2.72 m, and the altitude peak moves 11.97 m -> 12.36 m, which is also
/// the settled peak for both. The derived P0 is not the default -- changing that is a
/// separate decision with its own blast radius -- but these are the numbers the #266 retune
/// starts from.
///
/// They have now moved three times. Before #308's process-noise units fix the two paths
/// agreed to two decimals at 23.53 m / 37.88 m, because both were discarding their own
/// prediction and landing on the same fixes, so P0 could not tell them apart. The $P_0$ half
/// of that same fix then opened a gap in the vertical transient: `initialize_eskf`'s was
/// 42.55 m against this path's 21.41 m while it claimed a 1 cm initial altitude uncertainty,
/// and 28.20 m once it claimed 10 m.
///
/// What closed the gap again was neither P0 nor process noise. Both sides were seeding the
/// initial attitude from the record's radian Euler columns with `in_degrees: true`, scaling
/// it by pi/180, so both filters began level when the phone was at 77 degrees of roll -- and
/// a tighter P0 simply took longer to be talked out of it. With the attitude seeded from the
/// quaternion there is no transient left for P0 to halve, and the two initialisations are
/// separated by 0.39 m of altitude peak rather than by a startup artefact.
#[test]
fn test_eskf_auto_covariance_initialization_on_real_data() {
    /// Samples the vertical channel is allowed to settle over: 30 s at this recording's 1 Hz.
    const SETTLING_SAMPLES: usize = 30;

    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let test_data_path = Path::new(manifest_dir).join("tests/test_data.csv");
    let records = load_test_data(&test_data_path);
    assert!(!records.is_empty(), "test data should not be empty");

    let first = &records[0];
    // Exactly the state `initialize_eskf` builds, so P0 is the only thing that differs.
    //
    // This used to be a hand-copied struct literal, which is how it kept the seeding defect
    // `initialize_eskf` had -- Euler columns that are radians passed with `in_degrees: true`,
    // scaling the initial attitude by pi/180 -- for as long as the initialiser did.
    // `create_initial_state` builds the same state the initialiser now does, field for
    // field, so the claim above stays true by construction rather than by copying.
    let initial_state = create_initial_state(first);

    // The receiver's own reported horizontal accuracy across this recording, with the usual
    // vertical-is-twice-horizontal ratio and a half-metre-per-second velocity accuracy.
    let uncertainty = InitialUncertainty::new(
        GNSS_REPORTED_HORIZONTAL_ACCURACY_M,
        2.0 * GNSS_REPORTED_HORIZONTAL_ACCURACY_M,
        0.5,
    );
    let initial_covariance = IMUQuality::Consumer
        .auto_covariance(uncertainty, first.latitude, first.altitude)
        .expect("a phone-grade IMU and a reported GNSS accuracy must yield a covariance");

    let mut eskf = ErrorStateKalmanFilter::new(
        &initial_state,
        &[0.0; 6],
        initial_covariance.to_vec(),
        DMatrix::from_diagonal(&DVector::from_vec(DEFAULT_PROCESS_NOISE_DENSITY.to_vec())),
    );

    // The "P0 is the only thing that differs" claim above, made checkable rather than
    // asserted in prose. The two paths are
    // compared through the state the filters actually start from. Without this, a future
    // change to the production initialiser would leave this test quietly comparing two
    // different initial states while still describing itself as a P0 comparison -- which is
    // exactly what happened when this test hand-copied the initialiser's struct literal and
    // kept its pi/180 attitude defect alive.
    let reference = initialize_eskf(first, EskfConfig::default())
        .expect("the default ESKF initialisation must succeed on real data");
    let (reference_mean, this_mean) = (reference.get_estimate(), eskf.get_estimate());
    for (i, label) in [
        "latitude",
        "longitude",
        "altitude",
        "velocity north",
        "velocity east",
        "velocity vertical",
        "roll",
        "pitch",
        "yaw",
    ]
    .iter()
    .enumerate()
    {
        assert!(
            (reference_mean[i] - this_mean[i]).abs() < 1e-12,
            "this test and `initialize_eskf` disagree on the initial {label} \
             ({} vs {}), so P0 is no longer the only difference between them",
            reference_mean[i],
            this_mean[i]
        );
    }

    let cfg = {
        let mut built = AidingConfig::default();
        built.scheduler = MeasurementScheduler::PassThrough;
        built.fault = GnssFaultModel::None;
        built
    };
    let results = run_closed_loop(
        &mut eskf,
        build_event_stream(&records, &cfg, TEST_DATA_IS_ENU).unwrap(),
        None,
        None,
    )
    .expect("the auto-covariance ESKF must complete the full run");
    assert_eq!(results.len(), records.len());

    let stats = compute_error_metrics(&results, &records);
    println!("\n=== ESKF auto_covariance Initialization ===");
    println!(
        "Horizontal Error: rms={:.2}m, max={:.2}m",
        stats.rms_horizontal_error, stats.max_horizontal_error
    );
    println!(
        "Altitude Error: rms={:.2}m, max={:.2}m",
        stats.rms_altitude_error, stats.max_altitude_error
    );

    // Held to exactly the bounds the default initialisation is held to. A principled P0 that
    // navigates worse than the constant it replaces would not be an improvement.
    assert!(
        stats.rms_horizontal_error < 40.0,
        "auto-covariance ESKF RMS horizontal error should be under 40m, got {:.2}m",
        stats.rms_horizontal_error
    );
    assert!(
        stats.max_horizontal_error < 60.0,
        "auto-covariance ESKF max horizontal error should be under 60m, got {:.2}m",
        stats.max_horizontal_error
    );
    assert!(
        stats.rms_altitude_error < 10.0,
        "auto-covariance ESKF RMS altitude error should be under 10m, got {:.2}m",
        stats.rms_altitude_error
    );
    assert!(
        stats.max_altitude_error < 40.0,
        "auto-covariance ESKF max altitude error over the full run should be under 40m, got {:.2}m",
        stats.max_altitude_error
    );

    let settled_max_altitude_error = results
        .iter()
        .zip(records.iter())
        .skip(SETTLING_SAMPLES)
        .map(|(result, record)| (result.altitude - record.altitude).abs())
        .fold(0.0_f64, f64::max);
    println!("Altitude Error after settling: max={settled_max_altitude_error:.2}m");
    assert!(
        settled_max_altitude_error < 40.0,
        "auto-covariance ESKF settled altitude error should be under 40m, got          {settled_max_altitude_error:.2}m"
    );
}
