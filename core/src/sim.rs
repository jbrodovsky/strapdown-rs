//! Simulation utilities and data serialization for strapdown inertial navigation.
//!
//! This module provides tools for simulating and evaluating strapdown inertial navigation systems.
//! It is primarily designed to work with data produced from the [Sensor Logger](https://www.tszheichoi.com/sensorlogger)
//! app, as such it makes assumptions about the data format and structure that that corresponds to
//! how that app records data.
//!
//! ## Data Formats
//!
//! Data is represented by the `TestDataRecord` struct (sensor measurements) and
//! `NavigationResult` struct (navigation solutions). Both support several formats:
//!
//! - **CSV**: human-readable, suitable for quick inspection and editing. Always available.
//! - **netCDF**: binary, optimized for large datasets and archival. Requires the `netcdf` feature.
//! - **HDF5**: binary, good compression and fast I/O. Requires the `hdf5` feature.
//! - **MCAP**: robotics log format. Requires the `mcap` feature.
//!
//! The binary formats each depend on a system library or a large dependency tree,
//! so they are opt-in; see the crate README for the feature table. Format-specific
//! examples live on the individual methods (`to_netcdf`, `from_hdf5`, and so on).
//!
//! ### Example: Working with CSV files
//!
//! ```no_run
//! use strapdown::sim::{TestDataRecord, NavigationResult};
//!
//! // Read test data
//! let test_data = TestDataRecord::from_csv("input_data.csv")
//!     .expect("Failed to read input data");
//!
//! // ... perform navigation simulation ...
//!
//! // Write navigation results
//! # let nav_results: Vec<NavigationResult> = vec![];
//! NavigationResult::to_csv(&nav_results, "output_results.csv")
//!     .expect("Failed to write navigation results");
//! ```
//!
//! ## Simulation Functions
//!
//! This module also provides basic functionality for analyzing canonical strapdown inertial navigation
//! systems via the `dead_reckoning` and `closed_loop` functions. The `closed_loop` function in particular
//! can also be used to simulate various types of GNSS-denied scenarios, such as intermittent, degraded,
//! or intermittent and degraded GNSS via the measurement models provided in this module. You can install
//! the programs that execute this generic simulation by installing the binary via `cargo install strapdown-rs`.
use core::f64;
use log::{debug, info, warn};
use std::fmt::{Debug, Display};
use std::io::{self, Read, Write};
use std::path::Path;
use std::time::{Duration as StdDuration, Instant};

use anyhow::{Result, bail};

use crate::StrapdownError;
use chrono::{DateTime, Datelike, Duration, Utc};
use nalgebra::{DMatrix, DVector, Vector3};
use serde::{Deserialize, Deserializer, Serialize};

#[cfg(feature = "clap")]
use clap::{Args, ValueEnum};

use crate::NavigationFilter;
use crate::earth::{METERS_TO_DEGREES, METERS_TO_RADIANS, principal_radii};
use crate::gating::{GateRecovery, InnovationGate};
use crate::kalman::{InitialState, UnscentedKalmanFilter};
use crate::messages::{Event, EventStream, GnssFaultModel, MeasurementScheduler};

use crate::{IMUData, ImuSample, StrapdownState, mechanize};
use health::HealthMonitor;

// Re-export execution and health types for easier access in tests and external users
pub use execution::{ExecutionLimits, ExecutionMonitor};
pub use health::HealthLimits;

/// Position process-noise density for [`DEFAULT_PROCESS_NOISE_DENSITY`], as the standard
/// deviation accumulated in **one second**, in **metres** (so m/sqrt(s)).
///
/// # Per second, not per step (#374)
///
/// This used to be a *per-step* standard deviation, added once per IMU sample with no `dt`
/// scaling, which made the process noise a trajectory actually saw a function of its sample
/// rate rather than of its physics. Across the data this repository ships that was a **50x
/// spread** from one constant: `core/tests/test_data.csv` is 1 Hz, `generate_synthetic`
/// defaults to 10 Hz, and the `syn_*` baseline scenarios run at 50 Hz. Resampling a log
/// silently retuned the filter.
///
/// The numbers below are unchanged, and are now read as densities. That reinterpretation is
/// exact rather than approximate for the recording they were tuned on: `test_data.csv` steps
/// at **1.0000 s**, every step, so `q * dt == q` there and its behaviour is bit-identical
/// across #374. What changes is everything sampled faster, which stops receiving one full
/// second of process noise per step.
///
/// Every position quantity in the default diagonal is written here, in one unit, and converted
/// to each state's own unit exactly once at the point of use. That is the whole of the fix for
/// #308: latitude and longitude are held in radians and altitude in metres, so three literals
/// chosen to look alike on the page are three different physical claims, and the crate shipped
/// `1e-6, 1e-6, 1e-4` -- a 6367 m horizontal standard deviation next to a 1 cm vertical one --
/// for exactly that reason.
///
/// # Why 0.1 m
///
/// The value is not new: [`crate::sim`]'s own aiding acceptance tests (`core/tests/aiding.rs`)
/// already define `POSITION_PROCESS_NOISE_M_PER_ROOT_S = 0.1` and build their diagonal this way, having
/// hit the same trap. Adopting it here gives the workspace one number for this quantity instead
/// of a fourth.
///
/// What bounds it is the Kalman gain it implies. For a scalar random walk of standard
/// deviation $q$ per step observed with measurement standard deviation $r$, the steady-state prior
/// variance solves $P^2 - q^2 P - q^2 r^2 = 0$, so for $q \ll r$ it is $P \approx qr$ and the
/// steady-state gain is
///
/// $$ K = \frac{P}{P + r^2} \approx \frac{q}{q + r}. $$
///
/// The reference recording's GNSS reports a 3.81 m horizontal 1-sigma, and that sets both ends
/// of the admissible band:
///
/// - **Upper.** $K$ is what decides whether the filter filters at all. Requiring it to average
///   at least ten fixes ($K \le 0.1$) caps $q$ at $r/9 \approx 0.42$ m. Above that the filter
///   increasingly discards its own prediction, continuously, all the way up to the $K = 0.999$
///   of the defect -- which is why the old value produced a final solution sitting $10^{-8}$ m
///   from the fix it had just consumed.
/// - **Lower.** As $q \to 0$ the position block of $P$ collapses, $K \to 0$, and fixes stop
///   being able to correct inertial drift that is really there. There is no clean closed form
///   for this end, because what it trades against is unmodelled dynamics rather than a quantity
///   in the filter; empirically on the reference recording the whole-run statistics are flat
///   from 0.01 m to 1 m and the bias estimates stay inside their anti-windup clamps throughout.
///
/// 0.1 m sits an order of magnitude inside the derived upper bound, not against it: it gives
/// $K = 0.026$, so the filter averages roughly forty fixes and settles at a horizontal standard
/// deviation of $\sqrt{qr} = 0.62$ m against a 3.81 m fix. None of that is read off what the
/// suite currently prints.
///
/// # The derivation survives the change of units, and gets better away from 1 Hz
///
/// That argument is stated per step, and the step it was made on is the reference recording's
/// -- which is exactly 1 s, so at 1 Hz the per-step standard deviation and the per-root-second
/// density are the same number and the bound is unchanged.
///
/// Away from 1 Hz the density is the quantity that keeps the argument honest. A step of
/// $\Delta t$ contributes $q\sqrt{\Delta t}$, so at the 50 Hz the `syn_*` scenarios run at the
/// per-step figure is $0.1\sqrt{0.02} = 0.014$ m and $K = 0.0037$: the filter averages roughly
/// 270 fixes rather than 40. That is the *right* answer -- sampling the same trajectory more
/// often should let a filter average more, not inject fifty times the random walk -- and it is
/// what the per-step form got backwards. Under the old convention those scenarios received a
/// full second of process noise every 20 ms.
pub const POSITION_PROCESS_NOISE_M_PER_ROOT_S: f64 = 0.1;

/// [`POSITION_PROCESS_NOISE_M_PER_ROOT_S`] as a latitude/longitude variance density,
/// rad^2 per second.
///
/// The filters hold latitude and longitude in radians, so a metric horizontal uncertainty has
/// to pass through [`crate::earth::METERS_TO_RADIANS`] before it can sit on a covariance
/// diagonal -- the same conversion [`initialize_ukf`] spells out as
/// `(position_accuracy * METERS_TO_DEGREES).to_radians()` when it builds $P_0$.
const HORIZONTAL_POSITION_PROCESS_NOISE_RAD2_PER_S: f64 = {
    let radians = POSITION_PROCESS_NOISE_M_PER_ROOT_S * METERS_TO_RADIANS;
    radians * radians
};

/// Initial position uncertainty the default 15-state filters claim, as a standard deviation
/// in **metres**.
///
/// The $P_0$ counterpart of [`POSITION_PROCESS_NOISE_M_PER_ROOT_S`], and it exists for the same reason:
/// [`initialize_eskf`] and [`crate::engine`]'s `DEFAULT_INITIAL_COVARIANCE` both wrote their
/// position block as three literals -- `1e-6, 1e-6, 1e-4`, one of them commented "(m^2)" --
/// when latitude and longitude are radians and altitude is metres. Read correctly that is a
/// 6367 m horizontal claim beside a 1 cm vertical one, which is #308 in $P_0$ rather than in
/// $Q$; #303 had already noticed it in passing. Written in metres and converted at the point
/// of use, the units are checkable by reading them.
///
/// # Why 10 m
///
/// A coarse GNSS initialisation, and deliberately conservative: the reference recording's
/// receiver reports 3.81 m horizontal and 1.38 m vertical 1-sigma, so this is about 2.6x what
/// the fix that positions the vehicle actually claims.
///
/// Erring large is the safe direction, and the asymmetry is the derivation. A $P_0$ that is
/// too large costs only a short transient: with a 5 m fix the scalar Riccati recursion
/// $1/P_k = 1/P_0 + k/R$ pulls $(10 \text{ m})^2$ down to the metre level inside about twenty
/// fixes, twenty seconds at 1 Hz. A $P_0$ that is too small does not self-correct -- the
/// filter reports an uncertainty it has not earned, weights its own prediction accordingly,
/// and rejects or discounts the fixes that would have corrected it, which is the failure #260
/// gating turns from a slow drift into an outright rejection.
///
/// The same number is what `core/tests/aiding.rs` already uses for this quantity, so adopting
/// it gives the workspace one value rather than a fourth. Callers holding a fix's own reported
/// accuracy should prefer it -- [`initialize_ukf`] and [`initialize_ekf`] build $P_0$ from
/// `TestDataRecord::horizontal_accuracy`, and [`crate::IMUQuality::auto_covariance`] derives
/// the whole diagonal from an IMU grade and an [`crate::InitialUncertainty`].
pub const DEFAULT_INITIAL_POSITION_UNCERTAINTY_M: f64 = 10.0;

/// [`DEFAULT_INITIAL_POSITION_UNCERTAINTY_M`] as a latitude/longitude variance, rad^2.
pub(crate) const INITIAL_HORIZONTAL_POSITION_VARIANCE_RAD2: f64 = {
    let radians = DEFAULT_INITIAL_POSITION_UNCERTAINTY_M * METERS_TO_RADIANS;
    radians * radians
};

/// [`DEFAULT_INITIAL_POSITION_UNCERTAINTY_M`] as an altitude variance, m^2.
pub(crate) const INITIAL_VERTICAL_POSITION_VARIANCE_M2: f64 =
    DEFAULT_INITIAL_POSITION_UNCERTAINTY_M * DEFAULT_INITIAL_POSITION_UNCERTAINTY_M;

/// Per-step altitude process noise for [`DEFAULT_PROCESS_NOISE_DENSITY`], m^2.
///
/// Its own constant because it is the one position entry in metres rather than radians, and
/// tied to [`POSITION_PROCESS_NOISE_M_PER_ROOT_S`] because nothing ever justified it being anything else.
///
/// # Why it is no longer 1e-4
///
/// #308 was a units defect in the *horizontal* entries -- rad^2 written as though they were
/// m^2 -- and left this one at its historical `1e-4` on the explicit grounds that a units fix
/// is not the place to retune the vertical channel, the one channel this crate's history says
/// cannot take a quiet retune (#266, #286, #295). That was right, and it recorded the open
/// question: whether `1e-4` -- a **1 cm** per-step standard deviation, a tenth of the
/// horizontal `0.1 m`, an asymmetry no derivation was ever offered for -- is the right tuning.
///
/// It is not, and the symptom is consistency rather than accuracy. At `1e-4` the vertical
/// channel is **over-confident**: three-sigma containment of the altitude error is 0.88 on the
/// synthetic trajectory, where truth is exact, and 0.44 on `core/tests/test_data.csv`, against
/// the 0.9973 a correct covariance would give. Better than half the altitude errors on the
/// reference recording fall outside the uncertainty the filter reports for them. A filter that
/// claims sub-metre altitude while sitting 2.7 m out is not merely mistuned -- every consumer
/// of that covariance, innovation gating included, is being told something false.
///
/// # Why this knob and not another
///
/// Three entries could plausibly be blamed for an over-confident vertical channel. Sweeping
/// each alone over four decades through `initialize_eskf` on the synthetic cruise, reading
/// three-sigma altitude containment (ideal 0.9973) and `npes_position` (ideal 3.0):
///
/// | entry swept | containment | `npes_position` |
/// |---|---|---|
/// | altitude position, this constant | 0.881 -> 0.949, monotone | 5.65 -> 4.50, toward 3 |
/// | vertical velocity, index 5 | 0.881 -> 0.878, flat | 5.65 -> 5.84, away from 3 |
/// | accelerometer bias z, index 11 | 0.881 -> 0.870, worse | 5.65 -> 5.96, away from 3 |
///
/// Only this one moves the vertical channel toward consistency. The velocity entry buys
/// nothing and costs a vertical-velocity RMSE that grows from 0.34 to 4.37 m/s across the
/// sweep; the bias entry is actively harmful, taking the 60 s-outage horizontal RMSE on the
/// reference recording from 291 m to 443 m.
///
/// # Why 1e-2 specifically
///
/// It is [`POSITION_PROCESS_NOISE_M_PER_ROOT_S`] squared, which makes the position block isotropic in
/// per-step standard deviation. That is an argument from symmetry rather than from the
/// vertical sensor -- $Q$ models unmodelled dynamics, not fix quality, so the vertical fix
/// being *better* than the horizontal one (1.38 m against 3.81 m) is not a reason for a
/// tighter vertical process model -- and it happens to land on the knee of the measured curve.
/// Containment gains 0.05 going from `1e-4` to `1e-2` and only 0.016 more for the next decade,
/// while synthetic vertical RMSE, which is flat from `1e-6` to `1e-3`, starts climbing:
/// 0.665 m at `1e-4`, 0.836 m at `1e-2`, 1.352 m at `1e-1`.
///
/// # What it costs, stated plainly
///
/// Vertical RMSE against exact synthetic truth worsens by about a quarter, 0.665 m to 0.836 m.
/// Against the reference recording it does not: 2.724 m to 2.692 m, slightly better, because
/// there the filter was over-confident about an error it was not correcting. Horizontal
/// accuracy is unchanged everywhere to four significant figures except on the 60 s-outage
/// scenarios, where the ESKF improves sharply -- synthetic 138 m to 20 m -- and the UKF
/// worsens, 222 m to 267 m. That UKF row should not be read as a cost of this change: its yaw
/// error over the same sweep runs 105, 98, 21 and 34 degrees, so position degrades exactly
/// where attitude improves fivefold. That is #371 -- a linear Euler mean over sigma-point
/// attitudes -- coasting a GNSS outage, and it responds monotonically to no knob at all.
///
/// All of these numbers are gated in `core/tests/perf_baseline.json`, so the next change to
/// them has to be deliberate.
const VERTICAL_POSITION_PROCESS_NOISE_M2_PER_S: f64 =
    POSITION_PROCESS_NOISE_M_PER_ROOT_S * POSITION_PROCESS_NOISE_M_PER_ROOT_S;

/// Default process-noise **spectral density** used when a caller supplies none.
///
/// Each entry is a variance **per second**. The filters build $Q_k$ from it as
/// $Q_k = q \, \Delta t$ and add that to the propagated covariance, so the noise a trajectory
/// accumulates is a function of elapsed time rather than of how often it was sampled.
///
/// That was not true until #374. These entries were per-step variances added once per IMU
/// sample with no $\Delta t$ anywhere, which made the effective process noise proportional to
/// the sample rate: 1 Hz on `core/tests/test_data.csv`, 10 Hz from `generate_synthetic`'s
/// default and 50 Hz on the `syn_*` baseline scenarios -- **a 50x spread from one constant**,
/// tuned on exactly one of them. The values are unchanged and reinterpreted, which is exact
/// for the reference recording: it steps at 1.0000 s, so $q \Delta t = q$ there.
///
/// Ordering matches the 15-state vector
/// \[lat, lon, alt, v_n, v_e, v_d, roll, pitch, yaw, accel bias x/y/z, gyro bias x/y/z\], with
/// the states in the crate's native units (angles in radians, altitude in metres, velocities in
/// m/s). Nine-state filters take only the leading nine entries.
///
/// The three position entries are built from named constants rather than written as literals,
/// because they are *not* in the same unit as each other: latitude and longitude are radians
/// and altitude is metres. Until #308 they were written as `1e-6, 1e-6, 1e-4`, three literals
/// picked as though they were, which made the horizontal terms a 6.4 km per-step standard
/// deviation sitting next to a 1 cm one.
///
/// #308 changed only the horizontal pair, leaving [`VERTICAL_POSITION_PROCESS_NOISE_M2_PER_S`] at
/// its historical `1e-4` because altitude never carried the units defect and a units fix is
/// not the place to retune a channel. That retune is now done, separately and on its own
/// evidence: the altitude entry is `1e-2`, derived from the same
/// [`POSITION_PROCESS_NOISE_M_PER_ROOT_S`] as the horizontal pair, because `1e-4` left the vertical
/// channel reporting an uncertainty roughly half its actual error. See that constant for the
/// measurements. The remaining entries are hand-picked tuning values rather than values
/// derived from any particular sensor.
///
/// Callers in this crate have also reused the array verbatim as an initial error covariance
/// $P_0$; [`crate::IMUQuality::auto_covariance`] derives that fifteen-element diagonal from an
/// IMU grade and an initial fix accuracy instead.
pub const DEFAULT_PROCESS_NOISE_DENSITY: [f64; 15] = [
    HORIZONTAL_POSITION_PROCESS_NOISE_RAD2_PER_S, // latitude, rad^2/s
    HORIZONTAL_POSITION_PROCESS_NOISE_RAD2_PER_S, // longitude, rad^2/s
    VERTICAL_POSITION_PROCESS_NOISE_M2_PER_S,     // altitude, m^2/s
    1e-3,                                         // velocity north, (m/s)^2/s
    1e-3,                                         // velocity east, (m/s)^2/s
    1e-3,                                         // velocity down, (m/s)^2/s
    1e-5,                                         // roll, rad^2/s
    1e-5,                                         // pitch, rad^2/s
    1e-5,                                         // yaw, rad^2/s
    1e-6,                                         // acc bias x, (m/s^2)^2/s
    1e-6,                                         // acc bias y, (m/s^2)^2/s
    1e-6,                                         // acc bias z, (m/s^2)^2/s
    1e-8,                                         // gyro bias x, (rad/s)^2/s
    1e-8,                                         // gyro bias y, (rad/s)^2/s
    1e-8,                                         // gyro bias z, (rad/s)^2/s
];

/// Default [`ExecutionLimits::max_wall_clock_ratio`]: a run may burn at most a quarter of a
/// second of wall-clock time per second of trajectory it simulates.
pub const DEFAULT_MAX_WALL_CLOCK_RATIO: f64 = 0.25;
/// Default [`ExecutionLimits::max_wall_clock_s`]: hard ceiling of 1200 wall-clock seconds per
/// trajectory, whichever of it and the ratio budget is smaller.
pub const DEFAULT_MAX_WALL_CLOCK_S: f64 = 1200.0;
/// Default [`ExecutionLimits::max_no_progress_s`]: 600 wall-clock seconds without a call to
/// [`ExecutionMonitor::mark_progress`] before the run is treated as hung.
pub const DEFAULT_MAX_NO_PROGRESS_S: f64 = 600.0;

fn de_f64_nan<'de, D>(deserializer: D) -> Result<f64, D::Error>
where
    D: Deserializer<'de>,
{
    // Read whatever the CSV cell was as an Option<String>.
    // Missing field -> None; present but empty -> Some(""), etc.
    let opt = Option::<String>::deserialize(deserializer)?;
    match opt {
        None => Ok(f64::NAN),
        Some(s) => {
            let t = s.trim();
            if t.is_empty() || t.eq_ignore_ascii_case("nan") || t.eq_ignore_ascii_case("null") {
                return Ok(f64::NAN);
            }
            t.parse::<f64>().map_err(serde::de::Error::custom)
        }
    }
}
/// Struct representing a single row of test data from the CSV file.
///
/// Fields correspond to columns in the CSV, with appropriate renaming for Rust style.
/// This struct is setup to capture the data recorded from the [Sensor Logger](https://www.tszheichoi.com/sensorlogger) app.
/// Primarily, this represents IMU data as (relative to the device) and GPS data.
#[allow(
    clippy::unsafe_derive_deserialize,
    reason = "the only `unsafe` here is `Mmap::map` on a file handle; it relies on no invariant of this record"
)]
#[derive(Debug, Default, Deserialize, Serialize, Clone)]
pub struct TestDataRecord {
    /// Date-time string: YYYY-MM-DD hh:mm:ss+UTCTZ
    //#[serde(with = "ts_seconds")]
    pub time: DateTime<Utc>,
    /// accuracy of the bearing (magnetic heading) in degrees
    #[serde(rename = "bearingAccuracy", deserialize_with = "de_f64_nan")]
    pub bearing_accuracy: f64,
    /// accuracy of the speed in m/s
    #[serde(rename = "speedAccuracy", deserialize_with = "de_f64_nan")]
    pub speed_accuracy: f64,
    /// accuracy of the altitude in meters
    #[serde(rename = "verticalAccuracy", deserialize_with = "de_f64_nan")]
    pub vertical_accuracy: f64,
    /// accuracy of the horizontal position in meters
    #[serde(rename = "horizontalAccuracy", deserialize_with = "de_f64_nan")]
    pub horizontal_accuracy: f64,
    /// Speed in m/s
    #[serde(deserialize_with = "de_f64_nan")]
    pub speed: f64,
    /// Bearing in degrees
    #[serde(deserialize_with = "de_f64_nan")]
    pub bearing: f64,
    /// Altitude in meters
    #[serde(deserialize_with = "de_f64_nan")]
    pub altitude: f64,
    /// Longitude in degrees
    #[serde(deserialize_with = "de_f64_nan")]
    pub longitude: f64,
    /// Latitude in degrees
    #[serde(deserialize_with = "de_f64_nan")]
    pub latitude: f64,
    /// Quaternion component representing the rotation around the z-axis
    #[serde(deserialize_with = "de_f64_nan")]
    pub qz: f64,
    /// Quaternion component representing the rotation around the y-axis
    #[serde(deserialize_with = "de_f64_nan")]
    pub qy: f64,
    /// Quaternion component representing the rotation around the x-axis
    #[serde(deserialize_with = "de_f64_nan")]
    pub qx: f64,
    /// Quaternion component representing the rotation around the w-axis
    #[serde(deserialize_with = "de_f64_nan")]
    pub qw: f64,
    /// Roll angle in radians
    #[serde(deserialize_with = "de_f64_nan")]
    pub roll: f64,
    /// Pitch angle in radians
    #[serde(deserialize_with = "de_f64_nan")]
    pub pitch: f64,
    /// Yaw angle in radians
    #[serde(deserialize_with = "de_f64_nan")]
    pub yaw: f64,
    /// Z-acceleration in m/s^2
    #[serde(deserialize_with = "de_f64_nan")]
    pub acc_z: f64,
    /// Y-acceleration in m/s^2
    #[serde(deserialize_with = "de_f64_nan")]
    pub acc_y: f64,
    /// X-acceleration in m/s^2
    #[serde(deserialize_with = "de_f64_nan")]
    pub acc_x: f64,
    /// Rotation rate around the z-axis in radians/s
    #[serde(deserialize_with = "de_f64_nan")]
    pub gyro_z: f64,
    /// Rotation rate around the y-axis in radians/s
    #[serde(deserialize_with = "de_f64_nan")]
    pub gyro_y: f64,
    /// Rotation rate around the x-axis in radians/s
    #[serde(deserialize_with = "de_f64_nan")]
    pub gyro_x: f64,
    /// Magnetic field strength in the z-direction in micro teslas
    #[serde(deserialize_with = "de_f64_nan")]
    pub mag_z: f64,
    /// Magnetic field strength in the y-direction in micro teslas
    #[serde(deserialize_with = "de_f64_nan")]
    pub mag_y: f64,
    /// Magnetic field strength in the x-direction in micro teslas
    #[serde(deserialize_with = "de_f64_nan")]
    pub mag_x: f64,
    /// Change in altitude in meters
    #[serde(rename = "relativeAltitude", deserialize_with = "de_f64_nan")]
    pub relative_altitude: f64,
    /// pressure in millibars
    #[serde(deserialize_with = "de_f64_nan")]
    pub pressure: f64,
    /// Acceleration due to gravity in the z-direction in m/s^2
    #[serde(deserialize_with = "de_f64_nan")]
    pub grav_z: f64,
    /// Acceleration due to gravity in the y-direction in m/s^2
    #[serde(deserialize_with = "de_f64_nan")]
    pub grav_y: f64,
    /// Acceleration due to gravity in the x-direction in m/s^2
    #[serde(deserialize_with = "de_f64_nan")]
    pub grav_x: f64,
}
impl TestDataRecord {
    /// The record's attitude as a rotation matrix, taken from its quaternion.
    ///
    /// Use this rather than feeding `roll`/`pitch`/`yaw` to
    /// [`nalgebra::Rotation3::from_euler_angles`]. Those three fields are radians, but they
    /// are *not* nalgebra's intrinsic XYZ sequence: the Sensor Logger app this format comes
    /// from reports them in its own convention, and the two disagree by more than a sign.
    /// On the first sample of `core/tests/test_data.csv` the record's
    /// `(roll, pitch, yaw)` is `(0.163, -1.340, 0.179)` while the same attitude recovered
    /// from `(qw, qx, qy, qz)` is `(1.343, 0.037, -0.021)`: roll and pitch have effectively
    /// traded places.
    ///
    /// The consequence is not cosmetic. Rotating that sample's accelerometer reading into
    /// the navigation frame gives `(-0.000, 0.005, 9.725)` m/s^2 through the quaternion --
    /// specific force almost exactly along ENU up, which is what a near-stationary start
    /// must produce -- against `(-5.226, 8.187, 0.496)` through the raw Euler angles, which
    /// smears a full gravity across the horizontal axes and leaves the vertical channel with
    /// nothing to cancel. Propagated, that is an uncancelled ~9.8 m/s^2 that compounds: it
    /// took `dead_reckoning` to -1.7e16 m of altitude over this recording's 5,366 samples,
    /// finite on Linux and over the edge into the non-finite check on Windows.
    ///
    /// The quaternion is the authoritative attitude in this format. A record whose
    /// quaternion is all-NaN or zero-norm yields the identity rotation, matching how the
    /// rest of this module treats absent fields.
    #[must_use]
    pub fn attitude(&self) -> nalgebra::Rotation3<f64> {
        let q = nalgebra::Quaternion::new(self.qw, self.qx, self.qy, self.qz);
        if !q.coords.iter().all(|c| c.is_finite()) || q.norm() < f64::EPSILON {
            return nalgebra::Rotation3::identity();
        }
        nalgebra::UnitQuaternion::from_quaternion(q).into()
    }

    /// The record's GNSS ground track as north/east velocity components, in m/s.
    ///
    /// `bearing` is stored in **degrees**; converting it is the caller's job and was twice
    /// forgotten, so it is done here once. `speed` or `bearing` being NaN yields `(0.0, 0.0)`
    /// rather than propagating the NaN into a filter's initial state.
    #[must_use]
    pub fn ground_track_velocity(&self) -> (f64, f64) {
        if !self.speed.is_finite() || !self.bearing.is_finite() {
            return (0.0, 0.0);
        }
        let bearing_rad = self.bearing.to_radians();
        (
            self.speed * bearing_rad.cos(),
            self.speed * bearing_rad.sin(),
        )
    }

    /// The record as an [`InitialState`] for seeding a filter, in the caller's declared frame.
    ///
    /// This is the one place a `TestDataRecord` is turned into a filter seed:
    /// [`initialize_ukf`], [`initialize_ekf`] and [`initialize_eskf`] all call it, so the
    /// three cannot drift apart in their unit handling the way they had (#337).
    ///
    /// Every angular field leaves here in **radians**, tagged `in_degrees: false`, because a
    /// single flag cannot describe this record: its `latitude`/`longitude` are degrees while
    /// its `roll`/`pitch`/`yaw` are radians. Building the struct as a literal with
    /// `in_degrees: true` -- what all three did -- was therefore right for the position pair
    /// and wrong for the attitude triple, which the filter constructors then converted a
    /// second time: a 0.5 rad (28.6 deg) roll reached the filter as 0.0087 rad, shrunk by
    /// 57.3x. Converting once here and declaring radians removes the ambiguity rather than
    /// re-sharing the flag.
    ///
    /// Attitude comes from the record's quaternion via [`TestDataRecord::attitude`], not from
    /// its Euler columns, for the reason documented there: those columns are radians but not
    /// nalgebra's intrinsic XYZ sequence, so they are not interchangeable with the rotation
    /// the rest of the crate uses. [`dead_reckoning`] was switched to the quaternion in #302;
    /// this puts the closed-loop seed on the same footing, so `cl` and `open-loop` now start
    /// from the same attitude.
    ///
    /// Horizontal velocity comes from [`TestDataRecord::ground_track_velocity`], which does
    /// the degrees-to-radians conversion on `bearing` that [`initialize_ukf`] was missing
    /// entirely -- seeding a due-east track (bearing 90) as north-west at half speed. Vertical
    /// velocity is seeded at zero: the format reports no vertical rate, in either frame.
    #[must_use]
    pub fn initial_state(&self, is_enu: bool) -> InitialState {
        let (northward_velocity, eastward_velocity) = self.ground_track_velocity();
        let (roll, pitch, yaw) = self.attitude().euler_angles();
        InitialState::new(
            self.latitude.to_radians(),
            self.longitude.to_radians(),
            self.altitude,
            northward_velocity,
            eastward_velocity,
            0.0,
            roll,
            pitch,
            yaw,
            false,
            Some(is_enu),
        )
    }

    /// Reads a CSV file and returns a vector of `TestDataRecord` structs.
    ///
    /// # Arguments
    /// * `path` - Path to the CSV file to read.
    ///
    /// # Returns
    /// * `Ok(Vec<TestDataRecord>)` if successful.
    /// * `Err` if the file cannot be read or parsed.
    /// # Errors
    /// If the file cannot be read, or its contents are not valid CSV.
    pub fn from_csv<P: AsRef<std::path::Path>>(
        path: P,
    ) -> Result<Vec<Self>, Box<dyn std::error::Error>> {
        let mut rdr = csv::ReaderBuilder::new()
            .has_headers(true)
            .flexible(true)
            .trim(csv::Trim::All)
            .from_path(path)?;

        let mut records = Vec::new();
        for (i, result) in rdr.deserialize::<Self>().enumerate() {
            match result {
                Ok(r) => records.push(r),
                Err(e) => {
                    // Skip only this row; keep going.
                    warn!("Skipping row {} due to parse error: {e}", i + 1);
                }
            }
        }
        Ok(records)
    }
    /// Writes a vector of `TestDataRecord` structs to a CSV file.
    ///
    /// # Arguments
    /// * `records` - Vector of `TestDataRecord` structs to write
    /// * `path` - Path where the CSV file will be saved
    ///
    /// # Returns
    /// * `io::Result<()>` - Ok if successful, Err otherwise
    ///
    /// # Example
    ///
    /// ```
    /// use strapdown::sim::TestDataRecord;
    /// use std::path::Path;
    ///
    /// let record = TestDataRecord {
    ///     time: chrono::Utc::now(),
    ///     bearing_accuracy: 0.1,
    ///     speed_accuracy: 0.1,
    ///     vertical_accuracy: 0.1,
    ///     horizontal_accuracy: 0.1,
    ///     speed: 1.0,
    ///     bearing: 90.0,
    ///     altitude: 100.0,
    ///     longitude: -122.0,
    ///     latitude: 37.0,
    ///     qz: 0.0,
    ///     qy: 0.0,
    ///     qx: 0.0,
    ///     qw: 1.0,
    ///     roll: 0.0,
    ///     pitch: 0.0,
    ///     yaw: 0.0,
    ///     acc_z: 9.81,
    ///     acc_y: 0.0,
    ///     acc_x: 0.0,
    ///     gyro_z: 0.01,
    ///     gyro_y: 0.01,
    ///     gyro_x: 0.01,
    ///     mag_z: 50.0,
    ///     mag_y: -30.0,
    ///     mag_x: -20.0,
    ///     relative_altitude: 0.0,
    ///     pressure: 1013.25,
    ///     grav_z: 9.81,
    ///     grav_y: 0.0,
    ///     grav_x: 0.0,
    /// };
    /// let records = vec![record];
    /// TestDataRecord::to_csv(&records, "data.csv")
    ///    .expect("Failed to write test data to CSV");
    /// // doctest cleanup
    /// std::fs::remove_file("data.csv").unwrap();
    /// ```
    /// # Errors
    /// If the file cannot be created or written, or the records cannot be
    /// serialised as CSV.
    pub fn to_csv<P: AsRef<Path>>(records: &[Self], path: P) -> io::Result<()> {
        let mut writer = csv::Writer::from_path(path)?;
        for record in records {
            writer.serialize(record)?;
        }
        writer.flush()?;
        Ok(())
    }

    /// Writes a vector of `TestDataRecord` structs to an HDF5 file.
    ///
    /// # Arguments
    /// * `records` - Vector of `TestDataRecord` structs to write
    /// * `path` - Path where the HDF5 file will be saved
    ///
    /// # Returns
    /// * `Result<()>` - Ok if successful, Err otherwise
    ///
    /// # Example
    ///
    /// ```no_run
    /// use strapdown::sim::TestDataRecord;
    /// use std::path::Path;
    ///
    /// let record = TestDataRecord::default();
    /// let records = vec![record];
    /// TestDataRecord::to_hdf5(&records, "data.h5")
    ///    .expect("Failed to write test data to HDF5");
    /// ```
    #[cfg(feature = "hdf5")]
    /// # Errors
    /// If the file cannot be created or written, or the records cannot be
    /// serialised as HDF5.
    pub fn to_hdf5<P: AsRef<Path>>(records: &[Self], path: P) -> Result<()> {
        use hdf5::File;

        let file = File::create(path)?;
        let n = records.len();

        // Handle empty datasets
        if n == 0 {
            // Create group to indicate structure even for empty datasets
            let _group = file.create_group("test_data")?;
            return Ok(());
        }

        // Create a group for test data records
        let group = file.create_group("test_data")?;

        // Write timestamps as strings
        let timestamps: Result<Vec<hdf5::types::VarLenAscii>> = records
            .iter()
            .map(|r| {
                hdf5::types::VarLenAscii::from_ascii(&r.time.to_rfc3339())
                    .map_err(|e| anyhow::anyhow!("Failed to encode timestamp as ASCII: {e}"))
            })
            .collect();
        let timestamps = timestamps?;
        let ds_time = group
            .new_dataset::<hdf5::types::VarLenAscii>()
            .shape([n])
            .create("time")?;
        ds_time.write(&timestamps)?;

        // Helper macro to write f64 arrays
        macro_rules! write_f64_field {
            ($field_name:literal, $field:ident) => {{
                let data: Vec<f64> = records.iter().map(|r| r.$field).collect();
                let ds = group.new_dataset::<f64>().shape([n]).create($field_name)?;
                ds.write(&data)?;
            }};
        }

        write_f64_field!("bearing_accuracy", bearing_accuracy);
        write_f64_field!("speed_accuracy", speed_accuracy);
        write_f64_field!("vertical_accuracy", vertical_accuracy);
        write_f64_field!("horizontal_accuracy", horizontal_accuracy);
        write_f64_field!("speed", speed);
        write_f64_field!("bearing", bearing);
        write_f64_field!("altitude", altitude);
        write_f64_field!("longitude", longitude);
        write_f64_field!("latitude", latitude);
        write_f64_field!("qz", qz);
        write_f64_field!("qy", qy);
        write_f64_field!("qx", qx);
        write_f64_field!("qw", qw);
        write_f64_field!("roll", roll);
        write_f64_field!("pitch", pitch);
        write_f64_field!("yaw", yaw);
        write_f64_field!("acc_z", acc_z);
        write_f64_field!("acc_y", acc_y);
        write_f64_field!("acc_x", acc_x);
        write_f64_field!("gyro_z", gyro_z);
        write_f64_field!("gyro_y", gyro_y);
        write_f64_field!("gyro_x", gyro_x);
        write_f64_field!("mag_z", mag_z);
        write_f64_field!("mag_y", mag_y);
        write_f64_field!("mag_x", mag_x);
        write_f64_field!("relative_altitude", relative_altitude);
        write_f64_field!("pressure", pressure);
        write_f64_field!("grav_z", grav_z);
        write_f64_field!("grav_y", grav_y);
        write_f64_field!("grav_x", grav_x);
        Ok(())
    }
    /// Writes a vector of `TestDataRecord` structs to an MCAP file.
    ///
    /// **Note**: This method uses `MessagePack` encoding. Due to CSV-specific field deserializers\
    /// in `TestDataRecord`, direct MCAP deserialization may have limitations. For production use,
    /// consider converting to `NavigationResult` or using CSV format for `TestDataRecord`.
    ///
    /// # Arguments
    /// * `records` - Vector of `TestDataRecord` structs to write
    /// * `path` - Path where the MCAP file will be saved
    ///
    /// # Returns
    /// * `io::Result<()>` - Ok if successful, Err otherwise
    #[cfg(feature = "mcap")]
    /// # Errors
    /// If the file cannot be created or written, or the records cannot be
    /// serialised as MCAP.
    pub fn to_mcap<P: AsRef<Path>>(records: &[Self], path: P) -> io::Result<()> {
        use mcap::{Writer, records::MessageHeader};
        use std::collections::BTreeMap;
        use std::fs::File;
        use std::io::BufWriter;

        let file = File::create(path)?;
        let buf_writer = BufWriter::new(file);
        let mut writer = Writer::new(buf_writer).map_err(io::Error::other)?;

        // Add schema for TestDataRecord (using MessagePack encoding)
        let schema_name = "TestDataRecord";
        let schema_encoding = "msgpack";
        let schema_data = b"TestDataRecord struct serialized with MessagePack";

        let schema_id = writer
            .add_schema(schema_name, schema_encoding, schema_data)
            .map_err(io::Error::other)?;

        // Add channel for TestDataRecord messages
        let metadata = BTreeMap::new();
        let channel_id = writer
            .add_channel(schema_id, "sensor_data", "msgpack", &metadata)
            .map_err(io::Error::other)?;

        // Write each record as a message
        for (seq, record) in records.iter().enumerate() {
            let data = rmp_serde::to_vec(record)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

            let timestamp_nanos = record.time.timestamp_nanos_opt().ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "Timestamp out of range")
            })?;

            let header = MessageHeader {
                channel_id,
                sequence: seq as u32,
                log_time: timestamp_nanos as u64,
                publish_time: timestamp_nanos as u64,
            };

            writer
                .write_to_known_channel(&header, &data)
                .map_err(io::Error::other)?;
        }

        writer.finish().map_err(io::Error::other)?;

        Ok(())
    }

    /// Reads an HDF5 file and returns a vector of `TestDataRecord` structs.
    ///
    /// # Arguments
    /// * `path` - Path to the HDF5 file to read.
    ///
    /// # Returns
    /// * `Ok(Vec<TestDataRecord>)` if successful.
    /// * `Err` if the file cannot be read or parsed.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use strapdown::sim::TestDataRecord;
    ///
    /// let records = TestDataRecord::from_hdf5("data.h5")
    ///     .expect("Failed to read test data from HDF5");
    /// ```
    #[cfg(feature = "hdf5")]
    /// # Errors
    /// If the file cannot be read, or its contents are not valid HDF5.
    pub fn from_hdf5<P: AsRef<Path>>(path: P) -> Result<Vec<Self>> {
        use hdf5::File;

        let file = File::open(path)?;
        let group = file.group("test_data")?;

        // Check if time dataset exists (might be empty dataset)
        if let Ok(ds_time) = group.dataset("time") {
            // Read timestamps
            let timestamps: Vec<hdf5::types::VarLenAscii> = ds_time.read_raw()?;
            let n = timestamps.len();

            // Handle empty dataset
            if n == 0 {
                return Ok(Vec::new());
            }

            // Helper macro to read f64 arrays
            macro_rules! read_f64_field {
                ($field_name:literal) => {{
                    let ds = group.dataset($field_name)?;
                    let data: Vec<f64> = ds.read_raw()?;
                    data
                }};
            }

            let bearing_accuracy = read_f64_field!("bearing_accuracy");
            let speed_accuracy = read_f64_field!("speed_accuracy");
            let vertical_accuracy = read_f64_field!("vertical_accuracy");
            let horizontal_accuracy = read_f64_field!("horizontal_accuracy");
            let speed = read_f64_field!("speed");
            let bearing = read_f64_field!("bearing");
            let altitude = read_f64_field!("altitude");
            let longitude = read_f64_field!("longitude");
            let latitude = read_f64_field!("latitude");
            let qz = read_f64_field!("qz");
            let qy = read_f64_field!("qy");
            let qx = read_f64_field!("qx");
            let qw = read_f64_field!("qw");
            let roll = read_f64_field!("roll");
            let pitch = read_f64_field!("pitch");
            let yaw = read_f64_field!("yaw");
            let acc_z = read_f64_field!("acc_z");
            let acc_y = read_f64_field!("acc_y");
            let acc_x = read_f64_field!("acc_x");
            let gyro_z = read_f64_field!("gyro_z");
            let gyro_y = read_f64_field!("gyro_y");
            let gyro_x = read_f64_field!("gyro_x");
            let mag_z = read_f64_field!("mag_z");
            let mag_y = read_f64_field!("mag_y");
            let mag_x = read_f64_field!("mag_x");
            let relative_altitude = read_f64_field!("relative_altitude");
            let pressure = read_f64_field!("pressure");
            let grav_z = read_f64_field!("grav_z");
            let grav_y = read_f64_field!("grav_y");
            let grav_x = read_f64_field!("grav_x");

            let mut records = Vec::with_capacity(n);
            for i in 0..n {
                let time = DateTime::parse_from_rfc3339(timestamps[i].as_str())
                    .map_err(|e| anyhow::anyhow!("Failed to parse timestamp: {e}"))?
                    .with_timezone(&Utc);

                records.push(Self {
                    time,
                    bearing_accuracy: bearing_accuracy[i],
                    speed_accuracy: speed_accuracy[i],
                    vertical_accuracy: vertical_accuracy[i],
                    horizontal_accuracy: horizontal_accuracy[i],
                    speed: speed[i],
                    bearing: bearing[i],
                    altitude: altitude[i],
                    longitude: longitude[i],
                    latitude: latitude[i],
                    qz: qz[i],
                    qy: qy[i],
                    qx: qx[i],
                    qw: qw[i],
                    roll: roll[i],
                    pitch: pitch[i],
                    yaw: yaw[i],
                    acc_z: acc_z[i],
                    acc_y: acc_y[i],
                    acc_x: acc_x[i],
                    gyro_z: gyro_z[i],
                    gyro_y: gyro_y[i],
                    gyro_x: gyro_x[i],
                    mag_z: mag_z[i],
                    mag_y: mag_y[i],
                    mag_x: mag_x[i],
                    relative_altitude: relative_altitude[i],
                    pressure: pressure[i],
                    grav_z: grav_z[i],
                    grav_y: grav_y[i],
                    grav_x: grav_x[i],
                });
            }

            Ok(records)
        } else {
            // No time dataset means empty file
            Ok(Vec::new())
        }
    }

    /// Writes a vector of `TestDataRecord` structs to a netCDF file.
    ///
    /// # Arguments
    /// * `records` - Vector of `TestDataRecord` structs to write
    /// * `path` - Path where the netCDF file will be saved
    ///
    /// # Returns
    /// * `Result<()>` - Ok if successful, Err otherwise
    #[cfg(feature = "netcdf")]
    /// # Errors
    /// If the file cannot be created or written, or the records cannot be
    /// serialised as `NetCDF`.
    pub fn to_netcdf<P: AsRef<Path>>(records: &[Self], path: P) -> Result<()> {
        if records.is_empty() {
            bail!("Cannot write empty records to netCDF");
        }

        let n = records.len();
        let mut file = netcdf::create(path)?;

        // Define dimensions
        file.add_dimension("time", n)?;

        // Helper macro to add a variable and write data
        macro_rules! add_and_write {
            ($file:expr, $name:expr, $data:expr) => {{
                let mut var = $file.add_variable::<f64>($name, &["time"])?;
                var.put_values(&$data, ..)?;
            }};
        }

        // Prepare all data arrays first
        let times: Vec<f64> = records.iter().map(|r| r.time.timestamp() as f64).collect();
        let bearing_accuracy: Vec<f64> = records.iter().map(|r| r.bearing_accuracy).collect();
        let speed_accuracy: Vec<f64> = records.iter().map(|r| r.speed_accuracy).collect();
        let vertical_accuracy: Vec<f64> = records.iter().map(|r| r.vertical_accuracy).collect();
        let horizontal_accuracy: Vec<f64> = records.iter().map(|r| r.horizontal_accuracy).collect();
        let speed: Vec<f64> = records.iter().map(|r| r.speed).collect();
        let bearing: Vec<f64> = records.iter().map(|r| r.bearing).collect();
        let altitude: Vec<f64> = records.iter().map(|r| r.altitude).collect();
        let longitude: Vec<f64> = records.iter().map(|r| r.longitude).collect();
        let latitude: Vec<f64> = records.iter().map(|r| r.latitude).collect();
        let qz: Vec<f64> = records.iter().map(|r| r.qz).collect();
        let qy: Vec<f64> = records.iter().map(|r| r.qy).collect();
        let qx: Vec<f64> = records.iter().map(|r| r.qx).collect();
        let qw: Vec<f64> = records.iter().map(|r| r.qw).collect();
        let roll: Vec<f64> = records.iter().map(|r| r.roll).collect();
        let pitch: Vec<f64> = records.iter().map(|r| r.pitch).collect();
        let yaw: Vec<f64> = records.iter().map(|r| r.yaw).collect();
        let acc_z: Vec<f64> = records.iter().map(|r| r.acc_z).collect();
        let acc_y: Vec<f64> = records.iter().map(|r| r.acc_y).collect();
        let acc_x: Vec<f64> = records.iter().map(|r| r.acc_x).collect();
        let gyro_z: Vec<f64> = records.iter().map(|r| r.gyro_z).collect();
        let gyro_y: Vec<f64> = records.iter().map(|r| r.gyro_y).collect();
        let gyro_x: Vec<f64> = records.iter().map(|r| r.gyro_x).collect();
        let mag_z: Vec<f64> = records.iter().map(|r| r.mag_z).collect();
        let mag_y: Vec<f64> = records.iter().map(|r| r.mag_y).collect();
        let mag_x: Vec<f64> = records.iter().map(|r| r.mag_x).collect();
        let relative_altitude: Vec<f64> = records.iter().map(|r| r.relative_altitude).collect();
        let pressure: Vec<f64> = records.iter().map(|r| r.pressure).collect();
        let grav_z: Vec<f64> = records.iter().map(|r| r.grav_z).collect();
        let grav_y: Vec<f64> = records.iter().map(|r| r.grav_y).collect();
        let grav_x: Vec<f64> = records.iter().map(|r| r.grav_x).collect();

        // Add variables and write data
        add_and_write!(file, "time", times);
        add_and_write!(file, "bearingAccuracy", bearing_accuracy);
        add_and_write!(file, "speedAccuracy", speed_accuracy);
        add_and_write!(file, "verticalAccuracy", vertical_accuracy);
        add_and_write!(file, "horizontalAccuracy", horizontal_accuracy);
        add_and_write!(file, "speed", speed);
        add_and_write!(file, "bearing", bearing);
        add_and_write!(file, "altitude", altitude);
        add_and_write!(file, "longitude", longitude);
        add_and_write!(file, "latitude", latitude);
        add_and_write!(file, "qz", qz);
        add_and_write!(file, "qy", qy);
        add_and_write!(file, "qx", qx);
        add_and_write!(file, "qw", qw);
        add_and_write!(file, "roll", roll);
        add_and_write!(file, "pitch", pitch);
        add_and_write!(file, "yaw", yaw);
        add_and_write!(file, "acc_z", acc_z);
        add_and_write!(file, "acc_y", acc_y);
        add_and_write!(file, "acc_x", acc_x);
        add_and_write!(file, "gyro_z", gyro_z);
        add_and_write!(file, "gyro_y", gyro_y);
        add_and_write!(file, "gyro_x", gyro_x);
        add_and_write!(file, "mag_z", mag_z);
        add_and_write!(file, "mag_y", mag_y);
        add_and_write!(file, "mag_x", mag_x);
        add_and_write!(file, "relativeAltitude", relative_altitude);
        add_and_write!(file, "pressure", pressure);
        add_and_write!(file, "grav_z", grav_z);
        add_and_write!(file, "grav_y", grav_y);
        add_and_write!(file, "grav_x", grav_x);

        Ok(())
    }

    /// Reads a netCDF file and returns a vector of `TestDataRecord` structs.
    ///
    /// # Arguments
    /// * `path` - Path to the netCDF file to read.
    ///
    /// # Returns
    /// * `Ok(Vec<TestDataRecord>)` if successful.
    /// * `Err` if the file cannot be read or parsed.
    #[cfg(feature = "netcdf")]
    /// # Errors
    /// If the file cannot be read, or its contents are not valid `NetCDF`.
    pub fn from_netcdf<P: AsRef<Path>>(path: P) -> Result<Vec<Self>> {
        let file = netcdf::open(path)?;

        // Read time variable
        let time_var = file
            .variable("time")
            .ok_or_else(|| anyhow::anyhow!("time variable not found"))?;
        let times: Vec<f64> = time_var.get_values(..)?;
        let n = times.len();

        // Helper macro to read a variable
        macro_rules! read_var {
            ($file:expr, $name:expr) => {{
                let var = $file
                    .variable($name)
                    .ok_or_else(|| anyhow::anyhow!(concat!($name, " variable not found")))?;
                let data: Vec<f64> = var.get_values(..)?;
                data
            }};
        }

        // Read all variables
        let bearing_accuracy = read_var!(file, "bearingAccuracy");
        let speed_accuracy = read_var!(file, "speedAccuracy");
        let vertical_accuracy = read_var!(file, "verticalAccuracy");
        let horizontal_accuracy = read_var!(file, "horizontalAccuracy");
        let speed = read_var!(file, "speed");
        let bearing = read_var!(file, "bearing");
        let altitude = read_var!(file, "altitude");
        let longitude = read_var!(file, "longitude");
        let latitude = read_var!(file, "latitude");
        let qz = read_var!(file, "qz");
        let qy = read_var!(file, "qy");
        let qx = read_var!(file, "qx");
        let qw = read_var!(file, "qw");
        let roll = read_var!(file, "roll");
        let pitch = read_var!(file, "pitch");
        let yaw = read_var!(file, "yaw");
        let acc_z = read_var!(file, "acc_z");
        let acc_y = read_var!(file, "acc_y");
        let acc_x = read_var!(file, "acc_x");
        let gyro_z = read_var!(file, "gyro_z");
        let gyro_y = read_var!(file, "gyro_y");
        let gyro_x = read_var!(file, "gyro_x");
        let mag_z = read_var!(file, "mag_z");
        let mag_y = read_var!(file, "mag_y");
        let mag_x = read_var!(file, "mag_x");
        let relative_altitude = read_var!(file, "relativeAltitude");
        let pressure = read_var!(file, "pressure");
        let grav_z = read_var!(file, "grav_z");
        let grav_y = read_var!(file, "grav_y");
        let grav_x = read_var!(file, "grav_x");

        // Build records
        let mut records = Vec::with_capacity(n);
        for i in 0..n {
            let time = DateTime::from_timestamp(times[i] as i64, 0)
                .ok_or_else(|| anyhow::anyhow!("Invalid timestamp"))?
                .with_timezone(&Utc);

            records.push(Self {
                time,
                bearing_accuracy: bearing_accuracy[i],
                speed_accuracy: speed_accuracy[i],
                vertical_accuracy: vertical_accuracy[i],
                horizontal_accuracy: horizontal_accuracy[i],
                speed: speed[i],
                bearing: bearing[i],
                altitude: altitude[i],
                longitude: longitude[i],
                latitude: latitude[i],
                qz: qz[i],
                qy: qy[i],
                qx: qx[i],
                qw: qw[i],
                roll: roll[i],
                pitch: pitch[i],
                yaw: yaw[i],
                acc_z: acc_z[i],
                acc_y: acc_y[i],
                acc_x: acc_x[i],
                gyro_z: gyro_z[i],
                gyro_y: gyro_y[i],
                gyro_x: gyro_x[i],
                mag_z: mag_z[i],
                mag_y: mag_y[i],
                mag_x: mag_x[i],
                relative_altitude: relative_altitude[i],
                pressure: pressure[i],
                grav_z: grav_z[i],
                grav_y: grav_y[i],
                grav_x: grav_x[i],
            });
        }

        Ok(records)
    }

    /// Reads an MCAP file and returns a vector of `TestDataRecord` structs.
    ///
    /// **Note**: Due to CSV-specific field deserializers in `TestDataRecord`, MCAP deserialization\
    /// may fail. For production use, consider using CSV format for `TestDataRecord` or convert\
    /// to `NavigationResult` which fully supports MCAP.
    ///
    /// # Arguments
    /// * `path` - Path to the MCAP file to read.
    #[cfg(feature = "mcap")]
    /// # Errors
    /// If the file cannot be read, or its contents are not valid MCAP.
    pub fn from_mcap<P: AsRef<Path>>(path: P) -> Result<Vec<Self>, Box<dyn std::error::Error>> {
        use mcap::MessageStream;
        use std::fs::File;

        let file = File::open(path)?;

        // Memory-map the file for efficient reading
        let mapped = unsafe { memmap2::Mmap::map(&file)? };

        let message_stream = MessageStream::new(&mapped)?;
        let mut records = Vec::new();

        for message_result in message_stream {
            let message = message_result?;
            let record: Self = rmp_serde::from_slice(&message.data)?;
            records.push(record);
        }

        Ok(records)
    }
}
impl Display for TestDataRecord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "TestDataRecord(time: {}, latitude: {}, longitude: {}, altitude: {}, speed: {}, bearing: {})",
            self.time, self.latitude, self.longitude, self.altitude, self.speed, self.bearing
        )
    }
}
// ==== Helper structs for navigation simulations ====
/// Struct representing the covariance diagonal of a navigation solution in NED coordinates.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct NEDCovariance {
    /// Variance of the latitude estimate.
    pub latitude_cov: f64,
    /// Variance of the longitude estimate.
    pub longitude_cov: f64,
    /// Variance of the altitude estimate.
    pub altitude_cov: f64,
    /// Variance of the north velocity estimate.
    pub velocity_n_cov: f64,
    /// Variance of the east velocity estimate.
    pub velocity_e_cov: f64,
    /// Variance of the vertical velocity estimate.
    pub velocity_v_cov: f64,
    /// Variance of the roll estimate.
    pub roll_cov: f64,
    /// Variance of the pitch estimate.
    pub pitch_cov: f64,
    /// Variance of the yaw estimate.
    pub yaw_cov: f64,
    /// Variance of the accelerometer x-axis bias estimate.
    pub acc_bias_x_cov: f64,
    /// Variance of the accelerometer y-axis bias estimate.
    pub acc_bias_y_cov: f64,
    /// Variance of the accelerometer z-axis bias estimate.
    pub acc_bias_z_cov: f64,
    /// Variance of the gyroscope x-axis bias estimate.
    pub gyro_bias_x_cov: f64,
    /// Variance of the gyroscope y-axis bias estimate.
    pub gyro_bias_y_cov: f64,
    /// Variance of the gyroscope z-axis bias estimate.
    pub gyro_bias_z_cov: f64,
}
/// Where a filter carries its geophysical map-bias states, for labelling the solution.
///
/// A state vector cannot describe this on its own: a 16-element state is gravity-only or
/// magnetic-only depending on which maps the run was given, and reading the wrong label off it
/// would put a milligal figure in a nanotesla column. So the layout travels with the run rather
/// than being inferred from a length.
///
/// Indices rather than flags, and a declared `state_dim` rather than an assumed one, because a
/// filter need not append its map biases at the end -- `strapdown-geonav`'s `GeoBiasLayout` is
/// the authority on where they actually live, and this is its counterpart on the `core` side of
/// the dependency edge, which cannot name that type. `strapdown-sim` builds one from the other
/// so there is a single source of truth for the placement.
///
/// [`ExtraStateLayout::NONE`] is the ordinary, non-geophysical case and is what
/// [`run_closed_loop`] uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExtraStateLayout {
    state_dim: usize,
    gravity_index: Option<usize>,
    magnetic_index: Option<usize>,
    baro_index: Option<usize>,
}

impl Default for ExtraStateLayout {
    fn default() -> Self {
        Self::NONE
    }
}

impl ExtraStateLayout {
    /// No geophysical states: the fifteen-element solution every other path produces.
    pub const NONE: Self = Self {
        state_dim: NAVIGATION_STATES,
        gravity_index: None,
        magnetic_index: None,
        baro_index: None,
    };

    /// No map biases, on the particle filter's nine-state estimate.
    ///
    /// [`Self::NONE`] says the same thing for the Kalman filters and is fifteen wide, because
    /// that is *their* unaided shape. The particle filter carries no IMU-bias block, so an
    /// unaided particle estimate is nine, and handing the Kalman constant to
    /// [`NavigationResult::from_particle_filter_with_geo`] would fail its width assertion on
    /// the first row of every ordinary particle run.
    pub const PARTICLE_NONE: Self = Self {
        state_dim: PARTICLE_FILTER_STATES,
        gravity_index: None,
        magnetic_index: None,
        baro_index: None,
    };

    /// A layout over a state of `state_dim` entries, with the biases at the given indices.
    ///
    /// The indices are taken on trust: the caller that knows the filter has already validated
    /// them -- `GeoBiasLayout::new` rejects an index inside the navigation states or past the
    /// end -- and duplicating that here would be a second, drifting copy of the same rule. What
    /// is checked, at the point it matters, is that the state handed over is `state_dim` wide;
    /// see the conversion into [`NavigationResult`].
    #[must_use]
    pub const fn new(
        state_dim: usize,
        gravity_index: Option<usize>,
        magnetic_index: Option<usize>,
    ) -> Self {
        Self {
            state_dim,
            gravity_index,
            magnetic_index,
            baro_index: None,
        }
    }

    /// The same layout, with a barometric bias at `index`.
    ///
    /// A builder rather than a fourth parameter on [`Self::new`], so the geophysical callers --
    /// which are every existing one -- do not have to say "no barometer" to keep compiling.
    ///
    /// The barometric bias is not a map bias and this type's name is now narrower than what it
    /// holds: it is the layout of *every* state past the navigation block, whatever put them
    /// there. Renaming it belongs with the other two deferred renames on the 1.0 API-freeze
    /// list, not in a change that adds a state.
    #[must_use]
    pub const fn with_baro_bias(self, index: usize) -> Self {
        Self {
            baro_index: Some(index),
            ..self
        }
    }

    /// Index of the barometric bias in the state vector, if the filter carries one.
    #[must_use]
    pub const fn baro_index(self) -> Option<usize> {
        self.baro_index
    }

    /// Width of the state vector this layout describes.
    #[must_use]
    pub const fn state_dim(self) -> usize {
        self.state_dim
    }

    /// How many extra bias states the filter carries past the navigation block.
    #[must_use]
    pub const fn len(self) -> usize {
        self.gravity_index.is_some() as usize
            + self.magnetic_index.is_some() as usize
            + self.baro_index.is_some() as usize
    }

    /// Whether the layout carries no extra bias states at all.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.len() == 0
    }

    /// Index of the gravity bias in the state vector, if the filter carries one.
    #[must_use]
    pub const fn gravity_index(self) -> Option<usize> {
        self.gravity_index
    }

    /// Index of the magnetic bias in the state vector, if the filter carries one.
    #[must_use]
    pub const fn magnetic_index(self) -> Option<usize> {
        self.magnetic_index
    }
}

/// Reference-pressure drift a barometer is assumed to accumulate over an hour, metres.
///
/// One hectopascal, which is about 8.3 m near sea level. That is the everyday scale of
/// synoptic pressure change -- a front moving through over a few hours -- and it is what a
/// barometric altimeter reports as altitude when nothing corrects its reference.
///
/// This is the single physical quantity both barometric-bias constants below are derived
/// from, so there is one number to argue with rather than two.
pub const BARO_BIAS_DRIFT_M_PER_HOUR: f64 = 8.3;

/// Random-walk process noise on the barometric bias state, m^2 per second.
///
/// A random walk of spectral density $q$ reaches a standard deviation of $\sqrt{q t}$ after
/// $t$ seconds, so [`BARO_BIAS_DRIFT_M_PER_HOUR`] over 3,600 s gives
/// $q = 8.3^2 / 3600 = 0.0191$.
///
/// # It was derived and then measured, in that order
///
/// Sweeping $q$ on the reference recording through a UKF carrying this state, the vertical
/// channel reads:
///
/// | $q$ | 3-sigma containment | vertical bias | vertical RMSE |
/// |---|---:|---:|---:|
/// | none (no bias state) | 0.400 | +0.395 m | 2.575 m |
/// | 1e-6 | 0.554 | +0.729 m | 2.276 m |
/// | 1e-4 | 0.785 | -0.002 m | 1.579 m |
/// | 1e-2 | 0.836 | -0.025 m | 1.413 m |
///
/// The best measured value is `1e-2` and this constant is `0.0191`: **the derivation and the
/// measurement agree to within a factor of two**, on a knob swept over four decades. That is
/// the reason to take the derived value rather than the fitted one -- a constant that comes
/// from 1 hPa of pressure drift can be argued with by a meteorologist, and one that comes from
/// a sweep can only be re-swept (#288).
pub const BARO_BIAS_PROCESS_NOISE_M2_PER_S: f64 = {
    let hour = 3600.0;
    BARO_BIAS_DRIFT_M_PER_HOUR * BARO_BIAS_DRIFT_M_PER_HOUR / hour
};

/// Initial variance of the barometric bias state, m^2.
///
/// [`BARO_BIAS_DRIFT_M_PER_HOUR`] squared: the filter opens believing the barometer's
/// reference is off by something on the order of an hour's drift, which is what an
/// uncalibrated turn-on offset is.
///
/// The measurement above says this one barely matters -- initial variances of 1, 25 and 100
/// give the same vertical metrics to four significant figures, because the state is observable
/// and converges within the first minutes. Only $q$ moves the answer. That insensitivity is
/// itself the evidence the state is correctly identified rather than absorbing something else.
pub const INITIAL_BARO_BIAS_VARIANCE_M2: f64 =
    BARO_BIAS_DRIFT_M_PER_HOUR * BARO_BIAS_DRIFT_M_PER_HOUR;

/// Length of the full Kalman state vector: nine navigation states plus three accelerometer and
/// three gyroscope biases.
///
/// This is the shape the UKF, EKF and ESKF carry and what [`NavigationResult`]'s conversions
/// index, not a property of every filter in the crate: the particle filter reports a nine-state
/// navigation estimate and appends its own extra linear states, with no IMU-bias block.
pub const NAVIGATION_STATES: usize = 15;

/// Length of the particle filter's navigation estimate: the nine navigation states alone.
///
/// The Rao-Blackwellized particle filter carries no IMU-bias block -- its linear state is
/// velocity, attitude and whatever [`RbpfConfig::extra_state_dim`](crate::rbpf::RbpfConfig)
/// asks for -- so its estimate is six states shorter than [`NAVIGATION_STATES`] and its map
/// biases begin here rather than at 15. It is the bound
/// [`NavigationResult::from_particle_filter_with_geo`] checks a declared bias index against,
/// where the Kalman conversion checks [`NAVIGATION_STATES`].
///
/// The same nine as `geonav`'s `NAVIGATION_STATE_DIM`, which is what `GeoBiasLayout::appended`
/// is given as the base for an RBPF run. `geonav` depends on this crate rather than the other
/// way round, so the two are stated separately; `rbpf` holds the const assertion that keeps
/// this one in step with the width its estimator actually reports.
pub const PARTICLE_FILTER_STATES: usize = 9;

/// Read an on-disk geophysical column back into an `Option`.
///
/// `to_hdf5` and `to_netcdf` write NaN where the run carried no such map, because neither
/// format has an option type and both write flat f64 tables. NaN is not a value the filter
/// can produce for a bias it is actually estimating -- a NaN there would have failed the
/// health monitor long before the writer -- so it round-trips unambiguously.
///
/// Gated to match its only callers, `from_hdf5` and `from_netcdf`. Without this the default
/// build -- which has neither feature, and is what `cargo build -p strapdown-core` gives you --
/// carries it as dead code, and the CI lint job runs `-D warnings`.
#[cfg(any(feature = "hdf5", feature = "netcdf"))]
const fn none_if_nan(value: f64) -> Option<f64> {
    if value.is_nan() { None } else { Some(value) }
}

/// Generic result struct for navigation simulations.
///
/// This structure contains a single row of position, velocity, and attitude vectors
/// representing the navigation solution at a specific timestamp, along with the covariance diagonal,
/// input IMU measurements, and derived geophysical values.
///
/// It can be used across different types of navigation simulations such as dead reckoning,
/// Kalman filtering, or any other navigation algorithm.
#[allow(
    clippy::unsafe_derive_deserialize,
    reason = "the only `unsafe` here is `Mmap::map` on a file handle; it relies on no invariant of this record"
)]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[non_exhaustive]
pub struct NavigationResult {
    /// Timestamp corresponding to the state
    pub timestamp: DateTime<Utc>,
    // ---- Navigation solution states ----
    /// Latitude in **degrees** (WGS84).
    ///
    /// Every constructor converts on the way in -- see the `state[0].to_degrees()` in the
    /// `From<(&DateTime<Utc>, &DVector<f64>, &DMatrix<f64>, ExtraStateLayout)>` impl below --
    /// while [`Self::latitude_cov`] is the raw filter variance and stays in rad^2. The two
    /// fields are deliberately in different units; anything scoring a position error against
    /// its covariance has to convert the error to radians rather than the variance to degrees.
    pub latitude: f64,
    /// Longitude in **degrees** (WGS84). See [`Self::latitude`] for the units note.
    pub longitude: f64,
    /// Altitude in meters
    pub altitude: f64,
    /// Northward velocity in m/s
    pub velocity_north: f64,
    /// Eastward velocity in m/s
    pub velocity_east: f64,
    /// Vertical velocity in m/s
    pub velocity_vertical: f64,
    /// Roll angle in radians, on -pi..pi.
    ///
    /// All three angles are written on the branch `Rotation3::euler_angles` returns, whether
    /// the row came from a filter's `get_estimate` or from the dead-reckoning writer; before
    /// #314 the closed-loop rows used 0..2*pi and the open-loop rows did not, so one CSV
    /// schema carried two conventions.
    pub roll: f64,
    /// Pitch angle in radians. On -pi/2..pi/2 whenever the row came from a rotation -- that
    /// is the range the Euler decomposition produces -- and on -pi..pi in general, since the
    /// EKF and UKF carry pitch as a plain state element that an update can move.
    pub pitch: f64,
    /// Yaw angle in radians, on -pi..pi; negative is west of north.
    pub yaw: f64,
    /// IMU accelerometer x-axis bias in m/s^2
    pub acc_bias_x: f64,
    /// IMU accelerometer y-axis bias in m/s^2
    pub acc_bias_y: f64,
    /// IMU accelerometer z-axis bias in m/s^2
    pub acc_bias_z: f64,
    /// IMU gyroscope x-axis bias in radians/s
    pub gyro_bias_x: f64,
    /// IMU gyroscope y-axis bias in radians/s
    pub gyro_bias_y: f64,
    /// IMU gyroscope z-axis bias in radians/s
    pub gyro_bias_z: f64,
    // ---- Covariance values for the navigation solution ----
    /// Latitude variance, rad^2.
    pub latitude_cov: f64,
    /// Longitude variance, rad^2.
    pub longitude_cov: f64,
    /// Altitude variance, m^2.
    pub altitude_cov: f64,
    // ---- Position off-diagonals: the rest of the 3x3 block (#376) ----
    //
    // Everything else on this struct is a covariance *diagonal*, because the filters' full
    // matrices were reduced to `covariance.diagonal()` on the way in. For the position block
    // that reduction cost a real capability: `e^T P^-1 e` -- the actual NEES, the statistic
    // that says whether a filter believes the right thing -- is not computable from a
    // diagonal. `metrics::npes_position` is the diagonal-only stand-in, and it is
    // *optimistic*: it equals the NEES only when the block is genuinely diagonal, and after a
    // GNSS update the position states are correlated.
    //
    // Six numbers describe a symmetric 3x3; three are the variances above, and these are the
    // other three. The velocity, attitude and bias blocks keep their diagonals only -- nothing
    // scores them against a covariance, so carrying 30 more columns would be schema for its
    // own sake.
    /// Latitude-longitude covariance, rad^2.
    pub latitude_longitude_cov: f64,
    /// Latitude-altitude covariance, rad*m.
    pub latitude_altitude_cov: f64,
    /// Longitude-altitude covariance, rad*m.
    pub longitude_altitude_cov: f64,
    /// Northward velocity covariance
    pub velocity_n_cov: f64,
    /// Eastward velocity covariance
    pub velocity_e_cov: f64,
    /// Vertical velocity covariance
    pub velocity_v_cov: f64,
    /// Roll covariance
    pub roll_cov: f64,
    /// Pitch covariance
    pub pitch_cov: f64,
    /// Yaw covariance
    pub yaw_cov: f64,
    /// Accelerometer x-axis bias covariance
    pub acc_bias_x_cov: f64,
    /// Accelerometer y-axis bias covariance
    pub acc_bias_y_cov: f64,
    /// Accelerometer z-axis bias covariance
    pub acc_bias_z_cov: f64,
    /// Gyroscope x-axis bias covariance
    pub gyro_bias_x_cov: f64,
    /// Gyroscope y-axis bias covariance
    pub gyro_bias_y_cov: f64,
    /// Gyroscope z-axis bias covariance
    pub gyro_bias_z_cov: f64,
    // ---- Extra bias states: two geophysical pairs and the barometer ----
    //
    // `Option` rather than a sentinel because "this run carried no gravity map" and "this run
    // estimated a bias of exactly zero" are different facts, and a reader has to be able to
    // tell them apart. Serde writes `None` as an empty CSV cell and keeps the column, so the
    // schema is the same width for every run and a solution carrying none of these states
    // simply leaves the last *six* cells blank.
    //
    // `#[serde(default)]` on all six is what makes an older file still readable. CSV does not
    // need it -- the `csv` crate fills a column the header does not mention with `None` -- but
    // **MCAP does**: `rmp_serde::to_vec` writes a struct as a positional array, so a file
    // written before a column was added decodes as `invalid length 38, expected struct
    // NavigationResult with 40 elements`. That was already true of the four geophysical
    // columns when they were added; it is fixed for all six here rather than for the two that
    // prompted it, because half a rule is the one that drifts.
    /// Estimated gravity-anomaly measurement bias in mGal, when the run carried a gravity map.
    #[serde(default)]
    pub gravity_bias: Option<f64>,
    /// Covariance of [`NavigationResult::gravity_bias`].
    #[serde(default)]
    pub gravity_bias_cov: Option<f64>,
    /// Estimated magnetic-anomaly measurement bias in nT, when the run carried a magnetic map.
    #[serde(default)]
    pub magnetic_bias: Option<f64>,
    /// Covariance of [`NavigationResult::magnetic_bias`].
    #[serde(default)]
    pub magnetic_bias_cov: Option<f64>,
    /// Estimated barometric altitude bias, metres, when the filter carries that state.
    ///
    /// `None` on a filter that models the barometer as unbiased, which is every filter before
    /// #372 and any 15-state one after it. Same convention as the two geophysical pairs above,
    /// and for the same reason: "this run did not estimate a barometric bias" and "this run
    /// estimated a bias of exactly zero" are different facts.
    ///
    /// A barometer's reference pressure drifts by metres over an hour -- 1 hPa is about 8.3 m
    /// -- and with no state for it that drift lands in the altitude estimate under a tight $R$.
    /// That is what took three-sigma vertical containment to 0.40 on the reference recording
    /// against an ideal of 0.9973.
    #[serde(default)]
    pub baro_bias: Option<f64>,
    /// Covariance of [`NavigationResult::baro_bias`], m^2.
    #[serde(default)]
    pub baro_bias_cov: Option<f64>,
}
impl Default for NavigationResult {
    fn default() -> Self {
        Self {
            latitude_longitude_cov: 0.0,
            latitude_altitude_cov: 0.0,
            longitude_altitude_cov: 0.0,
            timestamp: Utc::now(),
            latitude: 0.0,
            longitude: 0.0,
            altitude: 0.0,
            velocity_north: 0.0,
            velocity_east: 0.0,
            velocity_vertical: 0.0,
            roll: 0.0,
            pitch: 0.0,
            yaw: 0.0,
            acc_bias_x: 0.0,
            acc_bias_y: 0.0,
            acc_bias_z: 0.0,
            gyro_bias_x: 0.0,
            gyro_bias_y: 0.0,
            gyro_bias_z: 0.0,
            latitude_cov: 1e-6, // default covariance values
            longitude_cov: 1e-6,
            altitude_cov: 1e-6,
            velocity_n_cov: 1e-6,
            velocity_e_cov: 1e-6,
            velocity_v_cov: 1e-6,
            roll_cov: 1e-6,
            pitch_cov: 1e-6,
            yaw_cov: 1e-6,
            acc_bias_x_cov: 1e-6,
            acc_bias_y_cov: 1e-6,
            acc_bias_z_cov: 1e-6,
            gyro_bias_x_cov: 1e-6,
            gyro_bias_y_cov: 1e-6,
            gyro_bias_z_cov: 1e-6,
            gravity_bias: None,
            gravity_bias_cov: None,
            magnetic_bias: None,
            magnetic_bias_cov: None,
            baro_bias: None,
            baro_bias_cov: None,
        }
    }
}
impl NavigationResult {
    /// Creates a new `NavigationResult` with default values.
    pub fn new() -> Self {
        Self::default() // add in validation
    }

    /// Writes the `NavigationResult` to a CSV file.
    ///
    /// # Arguments
    /// * `records` - Vector of `NavigationResult` structs to write
    /// * `path` - Path where the CSV file will be saved
    ///
    /// # Returns
    /// * `io::Result<()>` - Ok if successful, Err otherwise
    /// # Errors
    /// If the file cannot be created or written, or the records cannot be
    /// serialised as CSV.
    pub fn to_csv<P: AsRef<Path>>(records: &[Self], path: P) -> io::Result<()> {
        let mut writer = csv::Writer::from_path(path)?;
        for record in records {
            writer.serialize(record)?;
        }
        writer.flush()?;
        Ok(())
    }
    /// Reads a CSV file and returns a vector of `NavigationResult` structs.
    ///
    /// # Arguments
    /// * `path` - Path to the CSV file to read.
    ///
    /// # Returns
    /// * `Ok(Vec<NavigationResult>)` if successful.
    /// * `Err` if the file cannot be read or parsed.
    /// # Errors
    /// If the file cannot be read, or its contents are not valid CSV.
    pub fn from_csv<P: AsRef<std::path::Path>>(
        path: P,
    ) -> Result<Vec<Self>, Box<dyn std::error::Error>> {
        let mut rdr = csv::Reader::from_path(path)?;
        let mut records = Vec::new();
        for result in rdr.deserialize() {
            let record: Self = result?;
            records.push(record);
        }
        Ok(records)
    }

    /// Writes a vector of `NavigationResult` structs to an HDF5 file.
    ///
    /// # Arguments
    /// * `records` - Vector of `NavigationResult` structs to write
    /// * `path` - Path where the HDF5 file will be saved
    ///    
    ///
    /// # Example
    ///
    /// ```no_run
    /// use strapdown::sim::NavigationResult;
    ///
    /// let result = NavigationResult::default();
    /// let results = vec![result];
    /// NavigationResult::to_hdf5(&results, "nav_results.h5")
    ///     .expect("Failed to write navigation results to HDF5");
    /// ```
    #[cfg(feature = "hdf5")]
    /// # Errors
    /// If the file cannot be created or written, or the records cannot be
    /// serialised as HDF5.
    pub fn to_hdf5<P: AsRef<Path>>(records: &[Self], path: P) -> Result<()> {
        use hdf5::File;

        let file = File::create(path)?;
        let n = records.len();

        // Handle empty datasets
        if n == 0 {
            // Create group to indicate structure even for empty datasets
            let _group = file.create_group("navigation_results")?;
            return Ok(());
        }

        // Create a group for navigation results
        let group = file.create_group("navigation_results")?;

        // Write timestamps as strings
        let timestamps: Result<Vec<hdf5::types::VarLenAscii>> = records
            .iter()
            .map(|r| {
                hdf5::types::VarLenAscii::from_ascii(&r.timestamp.to_rfc3339())
                    .map_err(|e| anyhow::anyhow!("Failed to encode timestamp as ASCII: {e}"))
            })
            .collect();
        let timestamps = timestamps?;
        let ds_time = group
            .new_dataset::<hdf5::types::VarLenAscii>()
            .shape([n])
            .create("timestamp")?;
        ds_time.write(&timestamps)?;

        // Helper macro to write f64 arrays
        macro_rules! write_f64_field {
            ($field_name:literal, $field:ident) => {{
                let data: Vec<f64> = records.iter().map(|r| r.$field).collect();
                let ds = group.new_dataset::<f64>().shape([n]).create($field_name)?;
                ds.write(&data)?;
            }};
        }
        // The same, for a column that is absent on runs that carried no such map.
        macro_rules! write_optional_f64_field {
            ($field_name:literal, $field:ident) => {{
                let data: Vec<f64> = records
                    .iter()
                    .map(|r| r.$field.unwrap_or(f64::NAN))
                    .collect();
                let ds = group.new_dataset::<f64>().shape([n]).create($field_name)?;
                ds.write(&data)?;
            }};
        }

        // Navigation solution states
        write_f64_field!("latitude", latitude);
        write_f64_field!("longitude", longitude);
        write_f64_field!("altitude", altitude);
        write_f64_field!("velocity_north", velocity_north);
        write_f64_field!("velocity_east", velocity_east);
        write_f64_field!("velocity_vertical", velocity_vertical);
        write_f64_field!("roll", roll);
        write_f64_field!("pitch", pitch);
        write_f64_field!("yaw", yaw);
        write_f64_field!("acc_bias_x", acc_bias_x);
        write_f64_field!("acc_bias_y", acc_bias_y);
        write_f64_field!("acc_bias_z", acc_bias_z);
        write_f64_field!("gyro_bias_x", gyro_bias_x);
        write_f64_field!("gyro_bias_y", gyro_bias_y);
        write_f64_field!("gyro_bias_z", gyro_bias_z);

        // Covariance values
        write_f64_field!("latitude_cov", latitude_cov);
        write_f64_field!("latitude_longitude_cov", latitude_longitude_cov);
        write_f64_field!("latitude_altitude_cov", latitude_altitude_cov);
        write_f64_field!("longitude_altitude_cov", longitude_altitude_cov);
        write_f64_field!("longitude_cov", longitude_cov);
        write_f64_field!("altitude_cov", altitude_cov);
        write_f64_field!("velocity_n_cov", velocity_n_cov);
        write_f64_field!("velocity_e_cov", velocity_e_cov);
        write_f64_field!("velocity_v_cov", velocity_v_cov);
        write_f64_field!("roll_cov", roll_cov);
        write_f64_field!("pitch_cov", pitch_cov);
        write_f64_field!("yaw_cov", yaw_cov);
        write_f64_field!("acc_bias_x_cov", acc_bias_x_cov);
        write_f64_field!("acc_bias_y_cov", acc_bias_y_cov);
        write_f64_field!("acc_bias_z_cov", acc_bias_z_cov);
        write_f64_field!("gyro_bias_x_cov", gyro_bias_x_cov);
        write_f64_field!("gyro_bias_y_cov", gyro_bias_y_cov);
        write_f64_field!("gyro_bias_z_cov", gyro_bias_z_cov);
        // The geophysical columns are `Option` in memory but plain f64 on disk, with NaN for
        // "this run carried no such map". HDF5 has no native option type and the rest of this
        // writer is a flat f64 table; NaN is what the other absent-value paths in this module
        // already use, and `from_hdf5` maps it back to `None`.
        write_optional_f64_field!("gravity_bias", gravity_bias);
        write_optional_f64_field!("gravity_bias_cov", gravity_bias_cov);
        write_optional_f64_field!("magnetic_bias", magnetic_bias);
        write_optional_f64_field!("magnetic_bias_cov", magnetic_bias_cov);
        write_optional_f64_field!("baro_bias", baro_bias);
        write_optional_f64_field!("baro_bias_cov", baro_bias_cov);

        Ok(())
    }
    /// Reads an HDF5 file and returns a vector of `NavigationResult` structs.
    ///
    /// # Arguments
    /// * `path` - Path to the HDF5 file to read.
    ///
    /// # Returns
    /// * `Ok(Vec<NavigationResult>)` if successful.
    /// * `Err` if the file cannot be read or parsed.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use strapdown::sim::NavigationResult;
    ///
    /// let results = NavigationResult::from_hdf5("nav_results.h5")
    ///     .expect("Failed to read navigation results from HDF5");
    /// ```
    #[cfg(feature = "hdf5")]
    /// # Errors
    /// If the file cannot be read, or its contents are not valid HDF5.
    pub fn from_hdf5<P: AsRef<Path>>(path: P) -> Result<Vec<Self>> {
        use hdf5::File;

        let file = File::open(path)?;
        let group = file.group("navigation_results")?;

        // Check if timestamp dataset exists (might be empty dataset)
        if let Ok(ds_time) = group.dataset("timestamp") {
            // Read timestamps
            let timestamps: Vec<hdf5::types::VarLenAscii> = ds_time.read_raw()?;
            let n = timestamps.len();

            // Handle empty dataset
            if n == 0 {
                return Ok(Vec::new());
            }

            // Helper macro to read f64 arrays
            macro_rules! read_f64_field {
                ($field_name:literal) => {{
                    let ds = group.dataset($field_name)?;
                    let data: Vec<f64> = ds.read_raw()?;
                    data
                }};
            }
            /// A column that a file written before it existed will not have.
            ///
            /// The geophysical columns are additive: every result file written before they
            /// existed is still a valid navigation solution, and reading one must not fail just
            /// because it predates the schema. A *missing* dataset reads as all-absent; a
            /// dataset that is present but unreadable still propagates its error, so this does
            /// not paper over a corrupt file.
            macro_rules! read_optional_f64_field {
                ($field_name:literal) => {{
                    match group.dataset($field_name) {
                        Ok(ds) => ds.read_raw::<f64>()?,
                        Err(_) => vec![f64::NAN; n],
                    }
                }};
            }

            // Read navigation solution states
            let latitude = read_f64_field!("latitude");
            let longitude = read_f64_field!("longitude");
            let altitude = read_f64_field!("altitude");
            let velocity_north = read_f64_field!("velocity_north");
            let velocity_east = read_f64_field!("velocity_east");
            let velocity_vertical = read_f64_field!("velocity_vertical");
            let roll = read_f64_field!("roll");
            let pitch = read_f64_field!("pitch");
            let yaw = read_f64_field!("yaw");
            let acc_bias_x = read_f64_field!("acc_bias_x");
            let acc_bias_y = read_f64_field!("acc_bias_y");
            let acc_bias_z = read_f64_field!("acc_bias_z");
            let gyro_bias_x = read_f64_field!("gyro_bias_x");
            let gyro_bias_y = read_f64_field!("gyro_bias_y");
            let gyro_bias_z = read_f64_field!("gyro_bias_z");

            // Read covariance values
            let latitude_cov = read_f64_field!("latitude_cov");
            let latitude_longitude_cov = read_f64_field!("latitude_longitude_cov");
            let latitude_altitude_cov = read_f64_field!("latitude_altitude_cov");
            let longitude_altitude_cov = read_f64_field!("longitude_altitude_cov");
            let longitude_cov = read_f64_field!("longitude_cov");
            let altitude_cov = read_f64_field!("altitude_cov");
            let velocity_n_cov = read_f64_field!("velocity_n_cov");
            let velocity_e_cov = read_f64_field!("velocity_e_cov");
            let velocity_v_cov = read_f64_field!("velocity_v_cov");
            let roll_cov = read_f64_field!("roll_cov");
            let pitch_cov = read_f64_field!("pitch_cov");
            let yaw_cov = read_f64_field!("yaw_cov");
            let acc_bias_x_cov = read_f64_field!("acc_bias_x_cov");
            let acc_bias_y_cov = read_f64_field!("acc_bias_y_cov");
            let acc_bias_z_cov = read_f64_field!("acc_bias_z_cov");
            let gyro_bias_x_cov = read_f64_field!("gyro_bias_x_cov");
            let gyro_bias_y_cov = read_f64_field!("gyro_bias_y_cov");
            let gyro_bias_z_cov = read_f64_field!("gyro_bias_z_cov");
            let gravity_bias = read_optional_f64_field!("gravity_bias");
            let gravity_bias_cov = read_optional_f64_field!("gravity_bias_cov");
            let magnetic_bias = read_optional_f64_field!("magnetic_bias");
            let magnetic_bias_cov = read_optional_f64_field!("magnetic_bias_cov");
            let baro_bias = read_optional_f64_field!("baro_bias");
            let baro_bias_cov = read_optional_f64_field!("baro_bias_cov");

            let mut records = Vec::with_capacity(n);
            for i in 0..n {
                let timestamp = DateTime::parse_from_rfc3339(timestamps[i].as_str())
                    .map_err(|e| anyhow::anyhow!("Failed to parse timestamp: {e}"))?
                    .with_timezone(&Utc);

                records.push(Self {
                    latitude_longitude_cov: latitude_longitude_cov[i],
                    latitude_altitude_cov: latitude_altitude_cov[i],
                    longitude_altitude_cov: longitude_altitude_cov[i],
                    timestamp,
                    latitude: latitude[i],
                    longitude: longitude[i],
                    altitude: altitude[i],
                    velocity_north: velocity_north[i],
                    velocity_east: velocity_east[i],
                    velocity_vertical: velocity_vertical[i],
                    roll: roll[i],
                    pitch: pitch[i],
                    yaw: yaw[i],
                    acc_bias_x: acc_bias_x[i],
                    acc_bias_y: acc_bias_y[i],
                    acc_bias_z: acc_bias_z[i],
                    gyro_bias_x: gyro_bias_x[i],
                    gyro_bias_y: gyro_bias_y[i],
                    gyro_bias_z: gyro_bias_z[i],
                    latitude_cov: latitude_cov[i],
                    longitude_cov: longitude_cov[i],
                    altitude_cov: altitude_cov[i],
                    velocity_n_cov: velocity_n_cov[i],
                    velocity_e_cov: velocity_e_cov[i],
                    velocity_v_cov: velocity_v_cov[i],
                    roll_cov: roll_cov[i],
                    pitch_cov: pitch_cov[i],
                    yaw_cov: yaw_cov[i],
                    acc_bias_x_cov: acc_bias_x_cov[i],
                    acc_bias_y_cov: acc_bias_y_cov[i],
                    acc_bias_z_cov: acc_bias_z_cov[i],
                    gyro_bias_x_cov: gyro_bias_x_cov[i],
                    gyro_bias_y_cov: gyro_bias_y_cov[i],
                    gyro_bias_z_cov: gyro_bias_z_cov[i],
                    gravity_bias: none_if_nan(gravity_bias[i]),
                    gravity_bias_cov: none_if_nan(gravity_bias_cov[i]),
                    magnetic_bias: none_if_nan(magnetic_bias[i]),
                    magnetic_bias_cov: none_if_nan(magnetic_bias_cov[i]),
                    baro_bias: none_if_nan(baro_bias[i]),
                    baro_bias_cov: none_if_nan(baro_bias_cov[i]),
                });
            }

            Ok(records)
        } else {
            // No timestamp dataset means empty file
            Ok(Vec::new())
        }
    }

    /// Writes a vector of `NavigationResult` structs to a netCDF file.
    ///
    /// # Arguments
    /// * `records` - Vector of `NavigationResult` structs to write
    /// * `path` - Path where the netCDF file will be saved
    ///
    /// # Returns
    /// * `Result<()>` - Ok if successful, Err otherwise
    #[cfg(feature = "netcdf")]
    /// # Errors
    /// If the file cannot be created or written, or the records cannot be
    /// serialised as `NetCDF`.
    pub fn to_netcdf<P: AsRef<Path>>(records: &[Self], path: P) -> Result<()> {
        if records.is_empty() {
            bail!("Cannot write empty records to netCDF");
        }

        let n = records.len();
        let mut file = netcdf::create(path)?;

        // Define dimensions
        file.add_dimension("time", n)?;

        // Helper macro to add a variable and write data
        macro_rules! add_and_write {
            ($file:expr, $name:expr, $data:expr) => {{
                let mut var = $file.add_variable::<f64>($name, &["time"])?;
                var.put_values(&$data, ..)?;
            }};
        }

        // Prepare all data arrays
        let timestamps: Vec<f64> = records
            .iter()
            .map(|r| r.timestamp.timestamp() as f64)
            .collect();
        let latitude: Vec<f64> = records.iter().map(|r| r.latitude).collect();
        let longitude: Vec<f64> = records.iter().map(|r| r.longitude).collect();
        let altitude: Vec<f64> = records.iter().map(|r| r.altitude).collect();
        let velocity_north: Vec<f64> = records.iter().map(|r| r.velocity_north).collect();
        let velocity_east: Vec<f64> = records.iter().map(|r| r.velocity_east).collect();
        let velocity_vertical: Vec<f64> = records.iter().map(|r| r.velocity_vertical).collect();
        let roll: Vec<f64> = records.iter().map(|r| r.roll).collect();
        let pitch: Vec<f64> = records.iter().map(|r| r.pitch).collect();
        let yaw: Vec<f64> = records.iter().map(|r| r.yaw).collect();
        let acc_bias_x: Vec<f64> = records.iter().map(|r| r.acc_bias_x).collect();
        let acc_bias_y: Vec<f64> = records.iter().map(|r| r.acc_bias_y).collect();
        let acc_bias_z: Vec<f64> = records.iter().map(|r| r.acc_bias_z).collect();
        let gyro_bias_x: Vec<f64> = records.iter().map(|r| r.gyro_bias_x).collect();
        let gyro_bias_y: Vec<f64> = records.iter().map(|r| r.gyro_bias_y).collect();
        let gyro_bias_z: Vec<f64> = records.iter().map(|r| r.gyro_bias_z).collect();
        let latitude_cov: Vec<f64> = records.iter().map(|r| r.latitude_cov).collect();
        let latitude_longitude_cov: Vec<f64> =
            records.iter().map(|r| r.latitude_longitude_cov).collect();
        let latitude_altitude_cov: Vec<f64> =
            records.iter().map(|r| r.latitude_altitude_cov).collect();
        let longitude_altitude_cov: Vec<f64> =
            records.iter().map(|r| r.longitude_altitude_cov).collect();
        let longitude_cov: Vec<f64> = records.iter().map(|r| r.longitude_cov).collect();
        let altitude_cov: Vec<f64> = records.iter().map(|r| r.altitude_cov).collect();
        let velocity_n_cov: Vec<f64> = records.iter().map(|r| r.velocity_n_cov).collect();
        let velocity_e_cov: Vec<f64> = records.iter().map(|r| r.velocity_e_cov).collect();
        let velocity_v_cov: Vec<f64> = records.iter().map(|r| r.velocity_v_cov).collect();
        let roll_cov: Vec<f64> = records.iter().map(|r| r.roll_cov).collect();
        let pitch_cov: Vec<f64> = records.iter().map(|r| r.pitch_cov).collect();
        let yaw_cov: Vec<f64> = records.iter().map(|r| r.yaw_cov).collect();
        let acc_bias_x_cov: Vec<f64> = records.iter().map(|r| r.acc_bias_x_cov).collect();
        let acc_bias_y_cov: Vec<f64> = records.iter().map(|r| r.acc_bias_y_cov).collect();
        let acc_bias_z_cov: Vec<f64> = records.iter().map(|r| r.acc_bias_z_cov).collect();
        let gyro_bias_x_cov: Vec<f64> = records.iter().map(|r| r.gyro_bias_x_cov).collect();
        let gyro_bias_y_cov: Vec<f64> = records.iter().map(|r| r.gyro_bias_y_cov).collect();
        let gyro_bias_z_cov: Vec<f64> = records.iter().map(|r| r.gyro_bias_z_cov).collect();
        // NaN on disk for an absent geophysical column; see the note in `to_hdf5`.
        let gravity_bias: Vec<f64> = records
            .iter()
            .map(|r| r.gravity_bias.unwrap_or(f64::NAN))
            .collect();
        let gravity_bias_cov: Vec<f64> = records
            .iter()
            .map(|r| r.gravity_bias_cov.unwrap_or(f64::NAN))
            .collect();
        let magnetic_bias: Vec<f64> = records
            .iter()
            .map(|r| r.magnetic_bias.unwrap_or(f64::NAN))
            .collect();
        let magnetic_bias_cov: Vec<f64> = records
            .iter()
            .map(|r| r.magnetic_bias_cov.unwrap_or(f64::NAN))
            .collect();
        let baro_bias: Vec<f64> = records
            .iter()
            .map(|r| r.baro_bias.unwrap_or(f64::NAN))
            .collect();
        let baro_bias_cov: Vec<f64> = records
            .iter()
            .map(|r| r.baro_bias_cov.unwrap_or(f64::NAN))
            .collect();

        // Add variables and write data
        add_and_write!(file, "timestamp", timestamps);
        add_and_write!(file, "latitude", latitude);
        add_and_write!(file, "longitude", longitude);
        add_and_write!(file, "altitude", altitude);
        add_and_write!(file, "velocity_north", velocity_north);
        add_and_write!(file, "velocity_east", velocity_east);
        add_and_write!(file, "velocity_vertical", velocity_vertical);
        add_and_write!(file, "roll", roll);
        add_and_write!(file, "pitch", pitch);
        add_and_write!(file, "yaw", yaw);
        add_and_write!(file, "acc_bias_x", acc_bias_x);
        add_and_write!(file, "acc_bias_y", acc_bias_y);
        add_and_write!(file, "acc_bias_z", acc_bias_z);
        add_and_write!(file, "gyro_bias_x", gyro_bias_x);
        add_and_write!(file, "gyro_bias_y", gyro_bias_y);
        add_and_write!(file, "gyro_bias_z", gyro_bias_z);
        add_and_write!(file, "latitude_cov", latitude_cov);
        add_and_write!(file, "latitude_longitude_cov", latitude_longitude_cov);
        add_and_write!(file, "latitude_altitude_cov", latitude_altitude_cov);
        add_and_write!(file, "longitude_altitude_cov", longitude_altitude_cov);
        add_and_write!(file, "longitude_cov", longitude_cov);
        add_and_write!(file, "altitude_cov", altitude_cov);
        add_and_write!(file, "velocity_n_cov", velocity_n_cov);
        add_and_write!(file, "velocity_e_cov", velocity_e_cov);
        add_and_write!(file, "velocity_v_cov", velocity_v_cov);
        add_and_write!(file, "roll_cov", roll_cov);
        add_and_write!(file, "pitch_cov", pitch_cov);
        add_and_write!(file, "yaw_cov", yaw_cov);
        add_and_write!(file, "acc_bias_x_cov", acc_bias_x_cov);
        add_and_write!(file, "acc_bias_y_cov", acc_bias_y_cov);
        add_and_write!(file, "acc_bias_z_cov", acc_bias_z_cov);
        add_and_write!(file, "gyro_bias_x_cov", gyro_bias_x_cov);
        add_and_write!(file, "gyro_bias_y_cov", gyro_bias_y_cov);
        add_and_write!(file, "gyro_bias_z_cov", gyro_bias_z_cov);
        add_and_write!(file, "gravity_bias", gravity_bias);
        add_and_write!(file, "gravity_bias_cov", gravity_bias_cov);
        add_and_write!(file, "magnetic_bias", magnetic_bias);
        add_and_write!(file, "magnetic_bias_cov", magnetic_bias_cov);
        add_and_write!(file, "baro_bias", baro_bias);
        add_and_write!(file, "baro_bias_cov", baro_bias_cov);

        Ok(())
    }

    /// Reads a netCDF file and returns a vector of `NavigationResult` structs.
    ///
    /// # Arguments
    /// * `path` - Path to the netCDF file to read.
    ///
    /// # Returns
    /// * `Ok(Vec<NavigationResult>)` if successful.
    /// * `Err` if the file cannot be read or parsed.
    #[cfg(feature = "netcdf")]
    /// # Errors
    /// If the file cannot be read, or its contents are not valid `NetCDF`.
    pub fn from_netcdf<P: AsRef<Path>>(path: P) -> Result<Vec<Self>> {
        let file = netcdf::open(path)?;

        // Read timestamp variable
        let time_var = file
            .variable("timestamp")
            .ok_or_else(|| anyhow::anyhow!("timestamp variable not found"))?;
        let timestamps: Vec<f64> = time_var.get_values(..)?;
        let n = timestamps.len();

        // Helper macro to read a variable
        macro_rules! read_var {
            ($file:expr, $name:expr) => {{
                let var = $file
                    .variable($name)
                    .ok_or_else(|| anyhow::anyhow!(concat!($name, " variable not found")))?;
                let data: Vec<f64> = var.get_values(..)?;
                data
            }};
        }
        /// A variable that a file written before it existed will not have.
        ///
        /// Same reasoning as `read_optional_f64_field` in `from_hdf5`: the geophysical columns
        /// are additive, so a result file that predates them must still read. An absent
        /// variable yields all-absent; one that is present but unreadable still fails.
        macro_rules! read_optional_var {
            ($file:expr, $name:expr, $len:expr) => {{
                match $file.variable($name) {
                    Some(var) => var.get_values(..)?,
                    None => vec![f64::NAN; $len],
                }
            }};
        }

        // Read all variables
        let latitude = read_var!(file, "latitude");
        let longitude = read_var!(file, "longitude");
        let altitude = read_var!(file, "altitude");
        let velocity_north = read_var!(file, "velocity_north");
        let velocity_east = read_var!(file, "velocity_east");
        let velocity_vertical = read_var!(file, "velocity_vertical");
        let roll = read_var!(file, "roll");
        let pitch = read_var!(file, "pitch");
        let yaw = read_var!(file, "yaw");
        let acc_bias_x = read_var!(file, "acc_bias_x");
        let acc_bias_y = read_var!(file, "acc_bias_y");
        let acc_bias_z = read_var!(file, "acc_bias_z");
        let gyro_bias_x = read_var!(file, "gyro_bias_x");
        let gyro_bias_y = read_var!(file, "gyro_bias_y");
        let gyro_bias_z = read_var!(file, "gyro_bias_z");
        let latitude_cov = read_var!(file, "latitude_cov");
        let latitude_longitude_cov = read_var!(file, "latitude_longitude_cov");
        let latitude_altitude_cov = read_var!(file, "latitude_altitude_cov");
        let longitude_altitude_cov = read_var!(file, "longitude_altitude_cov");
        let longitude_cov = read_var!(file, "longitude_cov");
        let altitude_cov = read_var!(file, "altitude_cov");
        let velocity_n_cov = read_var!(file, "velocity_n_cov");
        let velocity_e_cov = read_var!(file, "velocity_e_cov");
        let velocity_v_cov = read_var!(file, "velocity_v_cov");
        let roll_cov = read_var!(file, "roll_cov");
        let pitch_cov = read_var!(file, "pitch_cov");
        let yaw_cov = read_var!(file, "yaw_cov");
        let acc_bias_x_cov = read_var!(file, "acc_bias_x_cov");
        let acc_bias_y_cov = read_var!(file, "acc_bias_y_cov");
        let acc_bias_z_cov = read_var!(file, "acc_bias_z_cov");
        let gyro_bias_x_cov = read_var!(file, "gyro_bias_x_cov");
        let gyro_bias_y_cov = read_var!(file, "gyro_bias_y_cov");
        let gyro_bias_z_cov = read_var!(file, "gyro_bias_z_cov");
        let gravity_bias = read_optional_var!(file, "gravity_bias", latitude.len());
        let gravity_bias_cov = read_optional_var!(file, "gravity_bias_cov", latitude.len());
        let magnetic_bias = read_optional_var!(file, "magnetic_bias", latitude.len());
        let magnetic_bias_cov = read_optional_var!(file, "magnetic_bias_cov", latitude.len());
        let baro_bias = read_optional_var!(file, "baro_bias", latitude.len());
        let baro_bias_cov = read_optional_var!(file, "baro_bias_cov", latitude.len());

        // Build records
        let mut records = Vec::with_capacity(n);
        for i in 0..n {
            let timestamp = DateTime::from_timestamp(timestamps[i] as i64, 0)
                .ok_or_else(|| anyhow::anyhow!("Invalid timestamp"))?
                .with_timezone(&Utc);

            records.push(Self {
                latitude_longitude_cov: latitude_longitude_cov[i],
                latitude_altitude_cov: latitude_altitude_cov[i],
                longitude_altitude_cov: longitude_altitude_cov[i],
                timestamp,
                latitude: latitude[i],
                longitude: longitude[i],
                altitude: altitude[i],
                velocity_north: velocity_north[i],
                velocity_east: velocity_east[i],
                velocity_vertical: velocity_vertical[i],
                roll: roll[i],
                pitch: pitch[i],
                yaw: yaw[i],
                acc_bias_x: acc_bias_x[i],
                acc_bias_y: acc_bias_y[i],
                acc_bias_z: acc_bias_z[i],
                gyro_bias_x: gyro_bias_x[i],
                gyro_bias_y: gyro_bias_y[i],
                gyro_bias_z: gyro_bias_z[i],
                latitude_cov: latitude_cov[i],
                longitude_cov: longitude_cov[i],
                altitude_cov: altitude_cov[i],
                velocity_n_cov: velocity_n_cov[i],
                velocity_e_cov: velocity_e_cov[i],
                velocity_v_cov: velocity_v_cov[i],
                roll_cov: roll_cov[i],
                pitch_cov: pitch_cov[i],
                yaw_cov: yaw_cov[i],
                acc_bias_x_cov: acc_bias_x_cov[i],
                acc_bias_y_cov: acc_bias_y_cov[i],
                acc_bias_z_cov: acc_bias_z_cov[i],
                gyro_bias_x_cov: gyro_bias_x_cov[i],
                gyro_bias_y_cov: gyro_bias_y_cov[i],
                gyro_bias_z_cov: gyro_bias_z_cov[i],
                gravity_bias: none_if_nan(gravity_bias[i]),
                gravity_bias_cov: none_if_nan(gravity_bias_cov[i]),
                magnetic_bias: none_if_nan(magnetic_bias[i]),
                magnetic_bias_cov: none_if_nan(magnetic_bias_cov[i]),
                baro_bias: none_if_nan(baro_bias[i]),
                baro_bias_cov: none_if_nan(baro_bias_cov[i]),
            });
        }

        Ok(records)
    }
    /// Writes a vector of `NavigationResult` structs to an MCAP file.
    ///
    /// # Arguments
    /// * `records` - Vector of `NavigationResult` structs to write
    /// * `path` - Path where the MCAP file will be saved
    ///
    /// # Returns
    /// * `io::Result<()>` - Ok if successful, Err otherwise
    ///
    /// # Example
    /// ```no_run
    /// use strapdown::sim::NavigationResult;
    /// use std::path::Path;
    ///
    /// let mut result = NavigationResult::default();
    /// result.latitude = 37.0;
    /// result.longitude = -122.0;
    /// result.altitude = 100.0;
    /// let results = vec![result];
    /// NavigationResult::to_mcap(&results, "results.mcap")
    ///    .expect("Failed to write navigation results to MCAP");
    /// ```
    #[cfg(feature = "mcap")]
    /// # Errors
    /// If the file cannot be created or written, or the records cannot be
    /// serialised as MCAP.
    pub fn to_mcap<P: AsRef<Path>>(records: &[Self], path: P) -> io::Result<()> {
        use mcap::{Writer, records::MessageHeader};
        use std::collections::BTreeMap;
        use std::fs::File;
        use std::io::BufWriter;

        let file = File::create(path)?;
        let buf_writer = BufWriter::new(file);
        let mut writer = Writer::new(buf_writer).map_err(io::Error::other)?;

        // Add schema for NavigationResult (using MessagePack encoding)
        let schema_name = "NavigationResult";
        let schema_encoding = "msgpack";
        let schema_data = b"NavigationResult struct serialized with MessagePack";

        let schema_id = writer
            .add_schema(schema_name, schema_encoding, schema_data)
            .map_err(io::Error::other)?;

        // Add channel for NavigationResult messages
        let metadata = BTreeMap::new();
        let channel_id = writer
            .add_channel(schema_id, "navigation_results", "msgpack", &metadata)
            .map_err(io::Error::other)?;

        // Write each record as a message
        for (seq, record) in records.iter().enumerate() {
            let data = rmp_serde::to_vec(record)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

            let timestamp_nanos = record.timestamp.timestamp_nanos_opt().ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "Timestamp out of range")
            })?;

            let header = MessageHeader {
                channel_id,
                sequence: seq as u32,
                log_time: timestamp_nanos as u64,
                publish_time: timestamp_nanos as u64,
            };

            writer
                .write_to_known_channel(&header, &data)
                .map_err(io::Error::other)?;
        }

        writer.finish().map_err(io::Error::other)?;

        Ok(())
    }

    /// Reads an MCAP file and returns a vector of `NavigationResult` structs.
    ///
    /// # Example
    /// ```no_run
    /// use strapdown::sim::NavigationResult;
    ///
    /// let results = NavigationResult::from_mcap("results.mcap")
    ///     .expect("Failed to read navigation results from MCAP");
    /// println!("Read {} navigation results", results.len());
    /// ```
    #[cfg(feature = "mcap")]
    /// # Errors
    /// If the file cannot be read, or its contents are not valid MCAP.
    pub fn from_mcap<P: AsRef<Path>>(path: P) -> Result<Vec<Self>, Box<dyn std::error::Error>> {
        use mcap::MessageStream;
        use std::fs::File;

        let file = File::open(path)?;

        // Memory-map the file for efficient reading
        let mapped = unsafe { memmap2::Mmap::map(&file)? };

        let message_stream = MessageStream::new(&mapped)?;
        let mut records = Vec::new();

        for message_result in message_stream {
            let message = message_result?;
            let record: Self = rmp_serde::from_slice(&message.data)?;
            records.push(record);
        }

        Ok(records)
    }
}
/// Convert `DVectors` containing the navigation state mean and covariance into a `NavigationResult`
/// struct.
///
/// This implementation is useful for converting the output of a Kalman filter or UKF into a
/// `NavigationResult`, which can then be used for further processing or analysis.
///
/// # Arguments
/// - `timestamp`: The timestamp of the navigation solution.
/// - `state`: A `DVector` containing the navigation state mean.
/// - `covariance`: A `DMatrix` containing the covariance of the state.
/// - `imu_data`: An `IMUData` struct containing the IMU measurements.
/// - `mag_x`, `mag_y`, `mag_z`: Magnetic field strength in micro teslas.
/// - `pressure`: Pressure in millibars.
/// - `freeair`: Free-air gravity anomaly in mGal.
///
/// # Returns
/// A `NavigationResult` struct containing the navigation solution.
impl From<(&DateTime<Utc>, &DVector<f64>, &DMatrix<f64>)> for NavigationResult {
    /// # Panics
    /// If the state is not 15 elements or the covariance is not 15x15.
    ///
    /// Deliberately an assertion rather than a [`StrapdownError`], unlike the rest of the
    /// #254 conversion: this is fed exclusively by `filter.get_estimate()` /
    /// `get_certainty()`, so a wrong shape is a crate invariant violation rather than bad
    /// user input. Converting it to `TryFrom` would push `?` into `run_closed_loop`'s result
    /// assembly and the integration tests for no reachable failure.
    ///
    /// A filter carrying geophysical bias states is longer than 15 and must go through the
    /// [`ExtraStateLayout`] form below, which knows what those extra states are; this one would
    /// otherwise reject it. That was the regression: the geophysical closed loop built a 16-state filter
    /// and died here on its first result, for every filter and every map.
    fn from(
        (timestamp, state, covariance): (&DateTime<Utc>, &DVector<f64>, &DMatrix<f64>),
    ) -> Self {
        Self::from((timestamp, state, covariance, ExtraStateLayout::NONE))
    }
}

/// The same conversion for a filter that carries geophysical bias states after its
/// navigation states.
///
/// The layout has to be supplied because the state vector cannot describe itself: a
/// 16-element state is gravity-only or magnetic-only depending on the run's flags. See
/// [`ExtraStateLayout`].
///
/// A layout narrower than [`NAVIGATION_STATES`] is a particle layout and is forwarded to
/// [`NavigationResult::from_particle_filter_with_geo`], which is what lets
/// [`run_closed_loop_with_geo`] drive a particle filter at [`ExtraStateLayout::PARTICLE_NONE`].
///
/// That is the no-extra-states particle path and only that path. A particle layout carrying
/// map biases is ten or eleven wide, while the runner reads its estimate through
/// [`NavigationFilter::get_estimate`](crate::NavigationFilter::get_estimate), which the
/// Rao-Blackwellized particle filter implements with its nine-state `estimate()` -- the wider
/// vector lives behind `estimate_with_extra_states`, which the trait has no way to ask for.
/// Such a layout still reaches the width assertion in the particle constructor and fails it,
/// by design rather than by running off the end of the vector. Geophysical particle runs
/// therefore keep their own event loop.
impl
    From<(
        &DateTime<Utc>,
        &DVector<f64>,
        &DMatrix<f64>,
        ExtraStateLayout,
    )> for NavigationResult
{
    /// # Panics
    /// If the state length or covariance shape disagrees with `layout.state_dim()`, or if a
    /// declared bias index falls outside the state. Same reasoning as the three-tuple form:
    /// this is fed by `filter.get_estimate()` / `get_certainty()`, so a mismatch is a crate
    /// invariant violation rather than user input.
    fn from(
        (timestamp, state, covariance, layout): (
            &DateTime<Utc>,
            &DVector<f64>,
            &DMatrix<f64>,
            ExtraStateLayout,
        ),
    ) -> Self {
        let expected = layout.state_dim();
        // A layout narrower than the fifteen Kalman states describes a particle estimate,
        // which carries no IMU-bias block. Without this dispatch the width assertion below
        // passes for `ExtraStateLayout::PARTICLE_NONE` -- nine states, nine given -- and the
        // bias reads at `state[9]..state[14]` then index off the end. That made
        // `run_closed_loop_with_geo` unusable for the plain particle layout, which is why
        // every particle event loop in this workspace is a hand-rolled copy of the others.
        // It does not make the *geophysical* particle layouts usable through the runner; see
        // the impl documentation above for why that needs an accessor the trait lacks.
        if expected < NAVIGATION_STATES {
            return Self::from_particle_filter_with_geo(timestamp, state, covariance, layout);
        }
        assert!(
            state.len() == expected,
            "State vector must have {expected} elements; got {}",
            state.len()
        );
        assert!(
            covariance.nrows() == expected && covariance.ncols() == expected,
            "Covariance matrix must be {expected}x{expected}"
        );
        // Capture the position block's off-diagonals before the diagonal reduction throws the
        // rest away. Without them `e^T P^-1 e` -- the actual NEES -- is not computable
        // downstream, and `metrics::npes_position` has to stand in for it optimistically,
        // assuming a correlation of zero that a GNSS update does not leave behind (#376).
        let latitude_longitude_cov = covariance[(0, 1)];
        let latitude_altitude_cov = covariance[(0, 2)];
        let longitude_altitude_cov = covariance[(1, 2)];
        let covariance = DVector::from_vec(covariance.diagonal().iter().copied().collect());
        // `layout` says which of the states past `NAVIGATION_STATES` is which; an absent map
        // leaves its column `None` rather than zero, so a reader can tell "no gravity map" from
        // "gravity bias estimated at zero".
        // Indices are checked here rather than at construction: this is the point where a
        // wrong one would silently read a navigation state as a map bias.
        let checked = |index: Option<usize>| {
            if let Some(i) = index {
                assert!(
                    i >= NAVIGATION_STATES && i < expected,
                    "a map bias lives after the {NAVIGATION_STATES} navigation states and \
                     inside the {expected}-element state; got index {i}"
                );
            }
            index
        };
        let geo_state = |index: Option<usize>| checked(index).map(|i| state[i]);
        let geo_cov = |index: Option<usize>| checked(index).map(|i| covariance[i]);
        let gravity_bias = geo_state(layout.gravity_index());
        let gravity_bias_cov = geo_cov(layout.gravity_index());
        let magnetic_bias = geo_state(layout.magnetic_index());
        let magnetic_bias_cov = geo_cov(layout.magnetic_index());
        let baro_bias = geo_state(layout.baro_index());
        let baro_bias_cov = geo_cov(layout.baro_index());
        // let wmm_date: Date = Date::from_calendar_date(
        //     timestamp.year(),
        //     Month::try_from(timestamp.month() as u8).unwrap(),
        //     timestamp.day() as u8,
        // )
        // .expect("Invalid date for world magnetic model");
        // let magnetic_field = GeomagneticField::new(
        //     Length::new::<meter>(state[2] as f32),
        //     Angle::new::<radian>(state[0] as f32),
        //     Angle::new::<radian>(state[1] as f32),
        //     wmm_date,
        // );
        Self {
            latitude_longitude_cov,
            latitude_altitude_cov,
            longitude_altitude_cov,
            timestamp: *timestamp,
            latitude: state[0].to_degrees(),
            longitude: state[1].to_degrees(),
            altitude: state[2],
            velocity_north: state[3],
            velocity_east: state[4],
            velocity_vertical: state[5],
            roll: state[6],
            pitch: state[7],
            yaw: state[8],
            acc_bias_x: state[9],
            acc_bias_y: state[10],
            acc_bias_z: state[11],
            gyro_bias_x: state[12],
            gyro_bias_y: state[13],
            gyro_bias_z: state[14],
            latitude_cov: covariance[0],
            longitude_cov: covariance[1],
            altitude_cov: covariance[2],
            velocity_n_cov: covariance[3],
            velocity_e_cov: covariance[4],
            velocity_v_cov: covariance[5],
            roll_cov: covariance[6],
            pitch_cov: covariance[7],
            yaw_cov: covariance[8],
            acc_bias_x_cov: covariance[9],
            acc_bias_y_cov: covariance[10],
            acc_bias_z_cov: covariance[11],
            gyro_bias_x_cov: covariance[12],
            gyro_bias_y_cov: covariance[13],
            gyro_bias_z_cov: covariance[14],
            gravity_bias,
            gravity_bias_cov,
            magnetic_bias,
            magnetic_bias_cov,
            baro_bias,
            baro_bias_cov,
        }
    }
}
/// Convert NED UKF to `NavigationResult`.
///
/// This implementation is useful for converting the output of a UKF into a
/// `NavigationResult`, which can then be used for further processing or analysis.
///
/// # Arguments
/// - `timestamp`: The timestamp of the navigation solution.
/// - `ukf`: A reference to the UKF instance containing the navigation state mean and covariance.
/// - `imu_data`: An `IMUData` struct containing the IMU measurements.
/// - `magnetic_vector`: Magnetic field strength measurement in micro teslas (body frame x, y, z).
/// - `pressure`: Pressure in millibars.
///
/// # Returns
/// A `NavigationResult` struct containing the navigation solution.
/// Converts the navigation states, the IMU biases, and the barometric bias if there is one.
///
/// A filter carrying *geophysical* bias states has them past index 14, and this conversion
/// leaves [`NavigationResult`]'s two map columns `None` rather than reading them: it is handed
/// a filter, not a [`ExtraStateLayout`], and the state vector cannot say which of its extra
/// states is gravity and which is magnetic. Geophysical runs therefore go through
/// [`run_closed_loop_with_geo`], which carries the layout.
///
/// The **barometric** bias is different, and is read here. Since #372 the filter answers
/// [`NavigationFilter::baro_bias_index`](crate::NavigationFilter::baro_bias_index) for itself,
/// so the one question this conversion could not previously answer -- which extra state is
/// which -- now has an answer for that state. Writing `None` regardless would drop an estimate
/// the filter demonstrably holds.
impl From<(&DateTime<Utc>, &UnscentedKalmanFilter)> for NavigationResult {
    fn from((timestamp, ukf): (&DateTime<Utc>, &UnscentedKalmanFilter)) -> Self {
        let state = &ukf.get_estimate();
        let covariance = ukf.get_certainty();
        // `None` when the filter carries no barometric bias, which is every 15-state one.
        let baro = ukf.baro_bias_index();
        Self {
            latitude_longitude_cov: covariance[(0, 1)],
            latitude_altitude_cov: covariance[(0, 2)],
            longitude_altitude_cov: covariance[(1, 2)],
            timestamp: *timestamp,
            latitude: state[0].to_degrees(),
            longitude: state[1].to_degrees(),
            altitude: state[2],
            velocity_north: state[3],
            velocity_east: state[4],
            velocity_vertical: state[5],
            roll: state[6],
            pitch: state[7],
            yaw: state[8],
            acc_bias_x: state[9],
            acc_bias_y: state[10],
            acc_bias_z: state[11],
            gyro_bias_x: state[12],
            gyro_bias_y: state[13],
            gyro_bias_z: state[14],
            latitude_cov: covariance[(0, 0)],
            longitude_cov: covariance[(1, 1)],
            altitude_cov: covariance[(2, 2)],
            velocity_n_cov: covariance[(3, 3)],
            velocity_e_cov: covariance[(4, 4)],
            velocity_v_cov: covariance[(5, 5)],
            roll_cov: covariance[(6, 6)],
            pitch_cov: covariance[(7, 7)],
            yaw_cov: covariance[(8, 8)],
            acc_bias_x_cov: covariance[(9, 9)],
            acc_bias_y_cov: covariance[(10, 10)],
            acc_bias_z_cov: covariance[(11, 11)],
            gyro_bias_x_cov: covariance[(12, 12)],
            gyro_bias_y_cov: covariance[(13, 13)],
            gyro_bias_z_cov: covariance[(14, 14)],
            gravity_bias: None,
            gravity_bias_cov: None,
            magnetic_bias: None,
            magnetic_bias_cov: None,
            baro_bias: baro.map(|index| state[index]),
            baro_bias_cov: baro.map(|index| covariance[(index, index)]),
        }
    }
}

/// Converts the navigation states, the IMU biases, and the barometric bias if there is one.
///
/// A filter carrying *geophysical* bias states has them past index 14, and this conversion
/// leaves [`NavigationResult`]'s two map columns `None` rather than reading them: it is handed
/// a filter, not a [`ExtraStateLayout`], and the state vector cannot say which of its extra
/// states is gravity and which is magnetic. Geophysical runs therefore go through
/// [`run_closed_loop_with_geo`], which carries the layout.
///
/// The **barometric** bias is different, and is read here. Since #372 the filter answers
/// [`NavigationFilter::baro_bias_index`](crate::NavigationFilter::baro_bias_index) for itself,
/// so the one question this conversion could not previously answer -- which extra state is
/// which -- now has an answer for that state. Writing `None` regardless would drop an estimate
/// the filter demonstrably holds.
impl From<(&DateTime<Utc>, &crate::kalman::ExtendedKalmanFilter)> for NavigationResult {
    fn from((timestamp, ekf): (&DateTime<Utc>, &crate::kalman::ExtendedKalmanFilter)) -> Self {
        let state = &ekf.get_estimate();
        let covariance = ekf.get_certainty();
        // `None` when the filter carries no barometric bias, which is every 15-state one.
        let baro = ekf.baro_bias_index();
        Self {
            latitude_longitude_cov: covariance[(0, 1)],
            latitude_altitude_cov: covariance[(0, 2)],
            longitude_altitude_cov: covariance[(1, 2)],
            timestamp: *timestamp,
            latitude: state[0].to_degrees(),
            longitude: state[1].to_degrees(),
            altitude: state[2],
            velocity_north: state[3],
            velocity_east: state[4],
            velocity_vertical: state[5],
            roll: state[6],
            pitch: state[7],
            yaw: state[8],
            acc_bias_x: if state.len() > 9 { state[9] } else { 0.0 },
            acc_bias_y: if state.len() > 10 { state[10] } else { 0.0 },
            acc_bias_z: if state.len() > 11 { state[11] } else { 0.0 },
            gyro_bias_x: if state.len() > 12 { state[12] } else { 0.0 },
            gyro_bias_y: if state.len() > 13 { state[13] } else { 0.0 },
            gyro_bias_z: if state.len() > 14 { state[14] } else { 0.0 },
            latitude_cov: covariance[(0, 0)],
            longitude_cov: covariance[(1, 1)],
            altitude_cov: covariance[(2, 2)],
            velocity_n_cov: covariance[(3, 3)],
            velocity_e_cov: covariance[(4, 4)],
            velocity_v_cov: covariance[(5, 5)],
            roll_cov: covariance[(6, 6)],
            pitch_cov: covariance[(7, 7)],
            yaw_cov: covariance[(8, 8)],
            acc_bias_x_cov: if covariance.nrows() > 9 {
                covariance[(9, 9)]
            } else {
                0.0
            },
            acc_bias_y_cov: if covariance.nrows() > 10 {
                covariance[(10, 10)]
            } else {
                0.0
            },
            acc_bias_z_cov: if covariance.nrows() > 11 {
                covariance[(11, 11)]
            } else {
                0.0
            },
            gyro_bias_x_cov: if covariance.nrows() > 12 {
                covariance[(12, 12)]
            } else {
                0.0
            },
            gyro_bias_y_cov: if covariance.nrows() > 13 {
                covariance[(13, 13)]
            } else {
                0.0
            },
            gyro_bias_z_cov: if covariance.nrows() > 14 {
                covariance[(14, 14)]
            } else {
                0.0
            },
            gravity_bias: None,
            gravity_bias_cov: None,
            magnetic_bias: None,
            magnetic_bias_cov: None,
            baro_bias: baro.map(|index| state[index]),
            baro_bias_cov: baro.map(|index| covariance[(index, index)]),
        }
    }
}

/// Convert `StrapdownState` to `NavigationResult`.
///
/// This implementation is useful for converting the output of a `StrapdownState` into a
/// `NavigationResult`, which can then be used for further processing or analysis.
///
/// # Arguments
/// - `timestamp`: The timestamp of the navigation solution.
/// - `state`: A reference to the `StrapdownState` instance containing the navigation state.
///
/// # Returns
/// A `NavigationResult` struct containing the navigation solution.
impl From<(&DateTime<Utc>, &StrapdownState)> for NavigationResult {
    fn from((timestamp, state): (&DateTime<Utc>, &StrapdownState)) -> Self {
        //let wmm_date: Date = Date::from_calendar_date(
        //    timestamp.year(),
        //    Month::try_from(timestamp.month() as u8).unwrap(),
        //    timestamp.day() as u8,
        //)
        //.expect("Invalid date for world magnetic model");
        //let magnetic_field = GeomagneticField::new(
        //    Length::new::<meter>(state.altitude as f32),
        //    Angle::new::<radian>(state.latitude as f32),
        //    Angle::new::<radian>(state.longitude as f32),
        //    wmm_date,
        //);
        Self {
            latitude_longitude_cov: f64::NAN,
            latitude_altitude_cov: f64::NAN,
            longitude_altitude_cov: f64::NAN,
            timestamp: *timestamp,
            latitude: state.latitude.to_degrees(),
            longitude: state.longitude.to_degrees(),
            altitude: state.altitude,
            velocity_north: state.velocity_north,
            velocity_east: state.velocity_east,
            velocity_vertical: state.velocity_vertical,
            roll: state.attitude.euler_angles().0,
            pitch: state.attitude.euler_angles().1,
            yaw: state.attitude.euler_angles().2,
            acc_bias_x: 0.0, // StrapdownState does not store biases
            acc_bias_y: 0.0,
            acc_bias_z: 0.0,
            gyro_bias_x: 0.0,
            gyro_bias_y: 0.0,
            gyro_bias_z: 0.0,
            latitude_cov: f64::NAN, // default covariance values
            longitude_cov: f64::NAN,
            altitude_cov: f64::NAN,
            velocity_n_cov: f64::NAN,
            velocity_e_cov: f64::NAN,
            velocity_v_cov: f64::NAN,
            roll_cov: f64::NAN,
            pitch_cov: f64::NAN,
            yaw_cov: f64::NAN,
            acc_bias_x_cov: f64::NAN,
            acc_bias_y_cov: f64::NAN,
            acc_bias_z_cov: f64::NAN,
            gyro_bias_x_cov: f64::NAN,
            gyro_bias_y_cov: f64::NAN,
            gyro_bias_z_cov: f64::NAN,
            gravity_bias: None,
            gravity_bias_cov: None,
            magnetic_bias: None,
            magnetic_bias_cov: None,
            baro_bias: None,
            baro_bias_cov: None,
        }
    }
}

impl NavigationResult {
    /// Create `NavigationResult` from particle filter state
    ///
    /// Creates a navigation result from a 9-element state vector (position, velocity, attitude)
    /// and covariance matrix produced by particle filter averaging. Since particle filters don't
    /// estimate IMU biases, those fields are set to zero.
    ///
    /// A geophysically aided run carries one bias state per active map after those nine, and
    /// must go through [`Self::from_particle_filter_with_geo`], which knows what they are.
    /// This one would reject it on the length assertion rather than silently drop them.
    ///
    /// # Arguments
    /// * `timestamp` - Timestamp for this navigation solution
    /// * `mean` - 9-element state vector [lat, lon, alt, vn, ve, vd, roll, pitch, yaw] in radians/meters
    /// * `cov` - 9x9 covariance matrix
    /// # Panics
    /// If `mean` is not 9 elements or `cov` is not 9x9. Same reasoning as the `From` impl
    /// above: the inputs come from `rbpf.estimate()`, not from user input.
    pub fn from_particle_filter(
        timestamp: &DateTime<Utc>,
        mean: &DVector<f64>,
        cov: &DMatrix<f64>,
    ) -> Self {
        // Not [`ExtraStateLayout::NONE`]: that one is fifteen states wide, because it describes
        // the Kalman filters' unaided shape. An unaided particle estimate is nine.
        Self::from_particle_filter_with_geo(timestamp, mean, cov, ExtraStateLayout::PARTICLE_NONE)
    }

    /// [`Self::from_particle_filter`] for a cloud that carries geophysical bias states.
    ///
    /// The particle filter appends one extra linear state per active map after its nine
    /// navigation states -- see
    /// [`RbpfConfig::extra_state_dim`](crate::rbpf::RbpfConfig::extra_state_dim) -- and
    /// [`RaoBlackwellizedParticleFilter::estimate_with_extra_states`](crate::rbpf::RaoBlackwellizedParticleFilter::estimate_with_extra_states)
    /// is the accessor that returns them with their covariance. `layout` says which of those
    /// extra states is which, exactly as it does for the Kalman paths: a ten-element particle
    /// estimate is gravity-only or magnetic-only depending on the run's flags, and the vector
    /// cannot say which.
    ///
    /// This is the particle-filter counterpart of the four-tuple `From` impl above, and the
    /// reason it is a separate constructor rather than a fourth argument on
    /// `from_particle_filter` is the same reason [`run_closed_loop_with_geo`] is separate from
    /// [`run_closed_loop`]: the geophysical paths need the layout and the ordinary ones do not.
    ///
    /// # Arguments
    /// * `timestamp` - Timestamp for this navigation solution
    /// * `mean` - `layout.state_dim()` element state vector
    /// * `cov` - Covariance of `mean`, square and in the same ordering
    /// * `layout` - Where the filter carries its map biases, past its nine navigation states
    ///
    /// # Panics
    /// If `mean` or `cov` disagrees with `layout.state_dim()`, or if a declared bias index
    /// falls outside the nine navigation states and the end of the vector. Same reasoning as
    /// [`Self::from_particle_filter`]: these come from the filter, not from user input, so a
    /// mismatch is a crate invariant violation.
    pub fn from_particle_filter_with_geo(
        timestamp: &DateTime<Utc>,
        mean: &DVector<f64>,
        cov: &DMatrix<f64>,
        layout: ExtraStateLayout,
    ) -> Self {
        let expected = layout.state_dim();
        assert_eq!(
            mean.len(),
            expected,
            "Particle filter state must have {expected} elements"
        );
        assert_eq!(
            cov.shape(),
            (expected, expected),
            "Particle filter covariance must be {expected}x{expected}"
        );
        // The same index check the four-tuple `From` makes, against this filter's own base:
        // the particle filter has no IMU-bias block, so its map biases start at
        // `PARTICLE_FILTER_STATES` and not at `NAVIGATION_STATES`. Reusing the Kalman bound
        // here would reject every genuine particle layout; dropping the check would let a
        // wrong index read a navigation state as a map bias, which is the failure the whole
        // layout exists to prevent.
        let checked = |index: Option<usize>| {
            if let Some(i) = index {
                assert!(
                    i >= PARTICLE_FILTER_STATES && i < expected,
                    "a map bias lives after the {PARTICLE_FILTER_STATES} navigation states \
                     and inside the {expected}-element state; got index {i}"
                );
            }
            index
        };
        // An absent map leaves its column `None` rather than zero, so a reader can tell "this
        // run carried no gravity map" from "this run estimated a gravity bias of zero". The
        // covariance travels with the bias: an estimate whose uncertainty was dropped on the
        // way out is not one anybody can use.
        let gravity_index = checked(layout.gravity_index());
        let magnetic_index = checked(layout.magnetic_index());
        let gravity_bias = gravity_index.map(|i| mean[i]);
        let gravity_bias_cov = gravity_index.map(|i| cov[(i, i)]);
        let magnetic_bias = magnetic_index.map(|i| mean[i]);
        let magnetic_bias_cov = magnetic_index.map(|i| cov[(i, i)]);
        let baro_index = checked(layout.baro_index());
        let baro_bias = baro_index.map(|i| mean[i]);
        let baro_bias_cov = baro_index.map(|i| cov[(i, i)]);

        Self {
            latitude_longitude_cov: cov[(0, 1)],
            latitude_altitude_cov: cov[(0, 2)],
            longitude_altitude_cov: cov[(1, 2)],
            timestamp: *timestamp,
            latitude: mean[0].to_degrees(),
            longitude: mean[1].to_degrees(),
            altitude: mean[2],
            velocity_north: mean[3],
            velocity_east: mean[4],
            velocity_vertical: mean[5],
            roll: mean[6],
            pitch: mean[7],
            yaw: mean[8],
            acc_bias_x: 0.0, // Particle filter doesn't estimate biases
            acc_bias_y: 0.0,
            acc_bias_z: 0.0,
            gyro_bias_x: 0.0,
            gyro_bias_y: 0.0,
            gyro_bias_z: 0.0,
            latitude_cov: cov[(0, 0)],
            longitude_cov: cov[(1, 1)],
            altitude_cov: cov[(2, 2)],
            velocity_n_cov: cov[(3, 3)],
            velocity_e_cov: cov[(4, 4)],
            velocity_v_cov: cov[(5, 5)],
            roll_cov: cov[(6, 6)],
            pitch_cov: cov[(7, 7)],
            yaw_cov: cov[(8, 8)],
            acc_bias_x_cov: f64::NAN,
            acc_bias_y_cov: f64::NAN,
            acc_bias_z_cov: f64::NAN,
            gyro_bias_x_cov: f64::NAN,
            gyro_bias_y_cov: f64::NAN,
            gyro_bias_z_cov: f64::NAN,
            gravity_bias,
            gravity_bias_cov,
            magnetic_bias,
            magnetic_bias_cov,
            baro_bias,
            baro_bias_cov,
        }
    }
}

/// Number of leading records averaged when checking a file against its declared frame.
///
/// Ten samples is a compromise between two failure modes. One sample is what
/// [`initialize_ukf`] and friends have available -- they are handed a single pose -- and it
/// carries the full per-sample accelerometer noise; averaging ten suppresses that by
/// $\sqrt{10}$ without reaching far enough into the recording to average over a manoeuvre.
/// At the 1 Hz of `core/tests/test_data.csv` that is ten seconds, and at the 10 Hz of
/// `strapdown-sim syn` it is one.
pub const FRAME_CHECK_SAMPLES: usize = 10;

/// Fraction of local gravity by which sensed specific force must contradict the declared
/// frame before [`check_declared_frame`] rejects it.
///
/// The decision variable is the sensed vertical specific force multiplied by the sign the
/// declared frame expects, so at rest it reads $+g$ when the declaration is right and $-g$
/// when it is wrong -- the two conventions are a full $2g$ apart. Rejecting at $-0.5g$
/// rather than at the $0$ midpoint is a deliberate asymmetry: a false rejection stops a
/// legitimate run, while a false acceptance only reproduces the behaviour this crate shipped
/// before #296, so the guard is biased towards believing the caller.
///
/// What that buys, in physical terms: firing on a *correctly* declared file needs the
/// windowed mean vertical acceleration to exceed $1.5g$ **downward** -- past free fall
/// ($1g$, which reads as exactly zero specific force and is accepted), and so requiring
/// sustained downward thrust or a near-inverted platform. Firing on a *wrongly* declared
/// file at rest has $1g$ of margin, twice the threshold. Sensor noise is nowhere near
/// either bound: consumer-grade accelerometer bias instability is 0.1 m/s^2 and the
/// velocity random walk contributes ~5e-3 m/s^2 per 0.1 s sample, three orders of magnitude
/// under $0.5g$.
pub const FRAME_CHECK_MARGIN_G: f64 = 0.5;

/// Reject records whose sensed specific force contradicts the declared local-level frame.
///
/// At rest an accelerometer senses the reaction to gravity, so rotating its reading into the
/// navigation frame gives $+g$ on ENU up and $-g$ on NED down (Groves 5.54 with zero
/// inertial acceleration). [`TestDataRecord`] carries no frame tag -- a Sensor Logger export
/// and [`generate_synthetic`] output are indistinguishable once loaded -- so this quantity is
/// the only thing that can tell the two apart. It is the same quantity
/// [`TestDataRecord::attitude`] documents and that `test_attitude_cancels_gravity_in_enu`
/// already asserts: $+9.72$ m/s^2 on `core/tests/test_data.csv`, $-9.78$ m/s^2 on `syn`
/// output.
///
/// This exists because getting the frame wrong is not a small error. Mechanizing NED records
/// as ENU adds the gravity model to the sensed specific force instead of cancelling it, and
/// the solution falls at $2g$: 35 km and 1174 m/s of vertical velocity in 60 s of stationary
/// truth (#296). Before the frame was selectable that was the only thing these entry points
/// could do; now that it is, the wrong answer must be an error rather than a plausible-looking
/// CSV.
///
/// The check fails **open**, never closed. An empty window, a non-finite gravity (which a NaN
/// latitude or altitude produces), or a record whose rotated specific force is not finite all
/// return `Ok`: this guard's job is to catch the overwhelming case, not to adjudicate
/// marginal ones, and every comparison against NaN is false anyway.
///
/// # Arguments
/// * `records` - The records about to be mechanized; only the first [`FRAME_CHECK_SAMPLES`]
///   are read.
/// * `is_enu` - The frame the caller declared: `false` for NED, `true` for ENU.
///
/// # Errors
/// [`StrapdownError::InvalidConfiguration`] on `is_enu` when the windowed mean vertical
/// specific force is more than [`FRAME_CHECK_MARGIN_G`] of local gravity the wrong way for
/// the declared frame. The message names the flag to pass.
pub fn check_declared_frame(
    records: &[TestDataRecord],
    is_enu: bool,
) -> Result<(), StrapdownError> {
    let Some(first) = records.first() else {
        return Ok(());
    };
    let window = &records[..records.len().min(FRAME_CHECK_SAMPLES)];
    let mut sum = 0.0;
    // `u32` rather than `usize` so the mean below is `f64::from(used)`, which is exact and
    // needs no cast: the window is at most `FRAME_CHECK_SAMPLES` long.
    let mut used = 0_u32;
    for record in window {
        // The quaternion, not the Euler angles: see `TestDataRecord::attitude` for why the
        // two are not interchangeable in this format, and what feeding the raw angles here
        // would cost (it smears a full gravity across the horizontal axes, which would make
        // this check read ~0 and fail open on every record).
        let specific_force_nav =
            record.attitude().matrix() * Vector3::new(record.acc_x, record.acc_y, record.acc_z);
        if specific_force_nav[2].is_finite() {
            sum += specific_force_nav[2];
            used += 1;
        }
    }
    let gravity = crate::earth::gravity(&first.latitude, &first.altitude);
    if used == 0 || !gravity.is_finite() {
        return Ok(());
    }
    let sensed = sum / f64::from(used);
    let expected_sign = if is_enu { 1.0 } else { -1.0 };
    let frame = if is_enu { "ENU" } else { "NED" };
    if sensed * expected_sign < -FRAME_CHECK_MARGIN_G * gravity {
        let other = if is_enu { "NED" } else { "ENU" };
        return Err(StrapdownError::InvalidConfiguration {
            field: "is_enu",
            reason: format!(
                "mean vertical specific force over the first {used} record(s) is \
                 {sensed:+.2} m/s^2, but {frame} mechanization expects {:+.2} m/s^2 at rest: \
                 these look like {other} records. Mechanizing them as {frame} would \
                 double-count gravity and integrate at 2 g. Declare the frame that matches \
                 the data ({}), or re-record it in {frame}.",
                expected_sign * gravity,
                if is_enu {
                    "drop `--enu`, or set `is_enu = false` in the config file"
                } else {
                    "pass `--enu`, or set `is_enu = true` in the config file"
                },
            ),
        });
    }
    info!(
        "Mechanizing input as {frame}: mean vertical specific force over the first {used} \
         record(s) is {sensed:+.3} m/s^2 against a local gravity of {gravity:.3} m/s^2"
    );
    Ok(())
}

/// Run dead reckoning or "open-loop" simulation using test data.
///
/// This function processes a sequence of sensor records through a `StrapdownState`, using
/// the "forward" method to propagate the state based on IMU measurements. It initializes
/// the `StrapdownState` with position, velocity, and attitude from the first record, and
/// then applies the IMU measurements from subsequent records. It does not record the
/// errors or confidence values, as this is a simple dead reckoning simulation and in testing
/// these values would be used as a baseline for comparison. Keep in mind that this toolbox
/// is designed for the local level frame of reference and the forward mechanization is typically
/// only valid at lower latitude (e.g. < 60 degrees) and at low altitudes (e.g. < 1000m). With
/// that, remember that dead reckoning is subject to drift and errors accumulate over time relative
/// to the quality of the IMU data. Poor quality IMU data (e.g. MEMS grade IMUs) will lead to
/// significant drift very quickly which may cause this function to produce unrealistic results,
/// hang, or crash.
///
/// # Arguments
/// * `records` - Vector of test data records containing IMU measurements and other sensor data
/// * `is_enu` - The local-level frame the records are expressed in: `false` for NED (the
///   library default, and what `strapdown-sim syn` emits), `true` for ENU (the convention
///   Sensor Logger exports). [`TestDataRecord`] carries no frame tag, so this cannot be
///   inferred; it is checked against the data by [`check_declared_frame`] rather than guessed.
///
/// # Returns
/// * `Vec<NavigationResult>` containing the sequence of `StrapdownState` instances over time,
///   along with timestamps and time differences.
/// # Errors
/// [`StrapdownError::InvalidConfiguration`] if the records' sensed specific force contradicts
/// `is_enu` -- see [`check_declared_frame`]. Otherwise propagated from [`crate::mechanize`] --
/// chiefly a non-positive `dt`, which duplicate or out-of-order record timestamps produce.
pub fn dead_reckoning(
    records: &[TestDataRecord],
    is_enu: bool,
) -> Result<Vec<NavigationResult>, StrapdownError> {
    if records.is_empty() {
        return Ok(Vec::new());
    }
    check_declared_frame(records, is_enu)?;
    // Initialize the result vector
    let mut results = Vec::with_capacity(records.len());
    // Initialize the StrapdownState with the first record
    let first_record = &records[0];
    // Attitude comes from the record's quaternion, not its Euler angles -- see
    // `TestDataRecord::attitude` for why the two are not interchangeable and what feeding
    // the raw angles here used to cost.
    let attitude = first_record.attitude();
    let (velocity_north, velocity_east) = first_record.ground_track_velocity();
    let mut state = StrapdownState {
        latitude: first_record.latitude.to_radians(),
        longitude: first_record.longitude.to_radians(),
        altitude: first_record.altitude,
        velocity_north,
        velocity_east,
        velocity_vertical: 0.0, // initial velocities
        attitude,
        is_enu,
    };
    // Store the initial state and metadata
    results.push(NavigationResult::from((&first_record.time, &state)));
    let mut previous_time = records[0].time;
    // Process each subsequent record
    for record in records.iter().skip(1) {
        // Try to calculate time difference from timestamps, default to 1 second if parsing fails
        let current_time = record.time;
        let dt = (current_time - previous_time).as_seconds_f64();
        // Create IMU data from the record
        let imu_data = IMUData {
            accel: Vector3::new(record.acc_x, record.acc_y, record.acc_z),
            gyro: Vector3::new(record.gyro_x, record.gyro_y, record.gyro_z),
        };
        mechanize(&mut state, &ImuSample::from_rates(&imu_data, dt))?;
        results.push(NavigationResult::from((&current_time, &state)));
        previous_time = record.time;
    }
    Ok(results)
}
/// Abort after this many consecutive rejected measurements.
///
/// Skipping unusable measurements keeps a run alive through a map edge; skipping *every*
/// measurement silently degrades the run to dead reckoning, which would still pass an
/// accuracy assertion by coincidence. This is the circuit breaker that distinguishes the
/// two. At typical 1 Hz aiding it is roughly 100 s without a usable fix.
const MAX_CONSECUTIVE_REJECTIONS: usize = 100;

/// Generic closed-loop simulation runner for any `NavigationFilter`
///
/// This function implements the core simulation loop for navigation filter architectures.
/// It iterates through the event stream, performs prediction and update steps, checks health limits,
/// and records navigation results. While generic, this is really only intended for Kalman-filter
/// family navigation filters because particle filter style navigation filters have the additional
/// step of resampling. For particle filter type filters, use the particle-filter loop instead.
///
/// # Innovation gating
///
/// Whether measurements are gated is the filter's business, not this function's:
/// install a gate with
/// [`NavigationFilter::set_innovation_gate`]
/// before calling. This loop counts and logs what the gate rejected, and feeds every
/// update's NIS to the [`HealthMonitor`] so a run that is gating *everything* -- the
/// signature of a diverged filter rather than an unlucky one -- trips the
/// consecutive-exceedance limit instead of silently degrading to dead reckoning.
///
/// # Arguments
/// * `filter` - Mutable reference to a type implementing `NavigationFilter`
/// * `stream` - Event stream containing IMU and measurement events
/// * `health_limits` - Optional health limits for monitoring
/// * `execution_limits` - Optional wall-clock and no-progress limits
///
/// # Returns
/// * `Vec<NavigationResult>` - A vector of navigation results
/// # Errors
/// If propagation fails, if a non-recoverable measurement error occurs, if the health or
/// execution monitor trips, or if more than `MAX_CONSECUTIVE_REJECTIONS` measurements are
/// rejected in a row. Recoverable measurement failures are skipped and counted instead.
pub fn run_closed_loop<F: NavigationFilter>(
    filter: &mut F,
    stream: EventStream,
    health_limits: Option<HealthLimits>,
    execution_limits: Option<ExecutionLimits>,
) -> anyhow::Result<Vec<NavigationResult>> {
    // `ExtraStateLayout::NONE` is fifteen wide, and the conversion into `NavigationResult`
    // asserts the state matches it. A filter estimating a barometric bias is sixteen, so
    // taking the layout from the filter is what keeps this -- the documented runner -- working
    // when `estimate_baro_bias` is on, instead of panicking deep in a `From` impl (#372).
    // Only the barometric bias is reachable this way: map biases still need
    // `run_closed_loop_with_geo`, because the filter cannot say which of its extra states is
    // a gravity anomaly and which a magnetic one.
    let layout = match filter.baro_bias_index() {
        Some(index) => {
            ExtraStateLayout::new(filter.get_estimate().len(), None, None).with_baro_bias(index)
        }
        None => ExtraStateLayout::NONE,
    };
    run_closed_loop_with_geo(filter, stream, health_limits, execution_limits, layout)
}

/// [`run_closed_loop`] for a filter carrying geophysical bias states.
///
/// Identical in every respect except that the extra states are labelled on the way into
/// [`NavigationResult`], using `layout` -- which the state vector cannot supply itself, since
/// a 16-element state is gravity-only or magnetic-only depending on the run's flags.
///
/// This exists as a separate entry point rather than a fifth parameter on `run_closed_loop`
/// because only the geophysical paths need it and thirty call sites do not.
///
/// # Arguments
/// * `filter` - Mutable reference to a type implementing `NavigationFilter`
/// * `stream` - Event stream containing IMU and measurement events
/// * `health_limits` - Optional health limits for monitoring
/// * `execution_limits` - Optional wall-clock and no-progress limits
/// * `layout` - Which geophysical bias states the filter carries past its navigation states
///
/// # Returns
/// * `Vec<NavigationResult>` - A vector of navigation results
///
/// # Errors
/// As [`run_closed_loop`].
pub fn run_closed_loop_with_geo<F: NavigationFilter>(
    filter: &mut F,
    stream: EventStream,
    health_limits: Option<HealthLimits>,
    execution_limits: Option<ExecutionLimits>,
    layout: ExtraStateLayout,
) -> anyhow::Result<Vec<NavigationResult>> {
    let start_time = stream.start_time;
    let mut results: Vec<NavigationResult> = Vec::with_capacity(stream.events.len());
    let total = stream.events.len();
    let mut monitor = HealthMonitor::new(health_limits.unwrap_or_default());
    let mut rejected_measurements: usize = 0;
    let mut gated_measurements: usize = 0;
    let mut forced_measurements: usize = 0;
    let mut consecutive_rejections: usize = 0;
    let sim_duration_s = stream.events.last().map_or(0.0, |event| match event {
        Event::Imu { elapsed_s, .. } | Event::Measurement { elapsed_s, .. } => *elapsed_s,
    });
    let mut execution_monitor =
        execution_limits.map(|limits| ExecutionMonitor::new(&limits, sim_duration_s));

    info!("Starting closed-loop navigation filter with {total} events");

    // Store the initial state (before processing any events)
    let mean = filter.get_estimate();
    let cov = filter.get_certainty();
    results.push(NavigationResult::from((&start_time, &mean, &cov, layout)));
    debug!("Initial filter state at {start_time}: {mean:?}");
    let mut last_ts = Some(start_time);

    for (i, event) in stream.events.into_iter().enumerate() {
        // Print detailed progress every 100 iterations or at key milestones
        if i % 10 == 0 || i == total {
            let mean = filter.get_estimate();
            let cov = filter.get_certainty();

            // Extract position and covariance diagonal
            let lat = mean[0].to_degrees();
            let lon = mean[1].to_degrees();
            let alt = mean[2];

            // Get position uncertainty (diagonal elements), in the state's native units
            // (radians for lat/lon, metres for altitude)
            let pos_std_lat_rad = cov[(0, 0)].sqrt();
            let pos_std_lon_rad = cov[(1, 1)].sqrt();
            let pos_std_alt = cov[(2, 2)].sqrt();

            let pos_rms =
                position_rms_meters(lat, alt, pos_std_lat_rad, pos_std_lon_rad, pos_std_alt);
            info!(
                "[{:.1}%] Event {}/{} | Pos: ({:.6}°, {:.6}°, {:.1}m) | Vel: ({:.2} m/s, {:.2} m/s, {:.2} m/s) | σ: ({:.2e}°, {:.2e}°, {:.2}m) | RMS: {:.2e}m",
                (i as f64 / total as f64) * 100.0,
                i,
                total,
                lat,
                lon,
                alt,
                mean[3],
                mean[4],
                mean[5],
                pos_std_lat_rad.to_degrees(),
                pos_std_lon_rad.to_degrees(),
                pos_std_alt,
                pos_rms
            );
        }

        // Compute wall-clock time for this event
        let elapsed_s = match &event {
            Event::Imu { elapsed_s, .. } | Event::Measurement { elapsed_s, .. } => *elapsed_s,
        };
        let ts = start_time + Duration::milliseconds((elapsed_s * 1000.0).round() as i64);

        // Emit the row for the epoch that just ended, *before* applying anything from this
        // one. This is the only moment at which a row stamped `last_ts` means what it says:
        // every event at or before it has been applied and none after it has (#367).
        //
        // It used to sit below the `match`, which made the row labelled `t_k` hold the state
        // after `t_{k+1}`'s first event -- always an `Event::Imu`, by `build_event_stream`'s
        // fixed per-epoch ordering -- so every interior row was one propagation step ahead of
        // its own label. On the 1 Hz reference recording at 21.19 m/s that was 21.2 m of
        // along-track error, very nearly the whole of the ~23.5 m horizontal RMSE all three
        // healthy filters reported, which is why they landed within 0.2 m of each other.
        if Some(ts) != last_ts {
            if let Some(prev_ts) = last_ts {
                // The seed row above already covers `start_time`, whose epoch is empty:
                // `build_event_stream` walks `windows(2)`, so record 0 produces no events.
                if prev_ts != start_time {
                    let mean = filter.get_estimate();
                    let cov = filter.get_certainty();
                    results.push(NavigationResult::from((&prev_ts, &mean, &cov, layout)));
                    debug!("Filter state at {prev_ts}: {mean:?}");
                }
            }
            last_ts = Some(ts);
        }

        // Checked before the event rather than after it, so the recoverable-measurement
        // `continue` below cannot skip `mark_progress`. It could, and a run legitimately
        // rejecting off-map samples -- the case that `continue` exists to support -- could
        // therefore trip `max_no_progress_s` while making perfectly good progress (#367).
        if let Some(ref mut monitor) = execution_monitor {
            monitor.check("closed-loop")?;
            monitor.mark_progress();
        }

        // Apply event
        match event {
            Event::Imu { dt_s, imu, .. } => {
                // Propagation failures are always fatal: a failed mechanization or a
                // singular covariance means the state is undefined, and continuing would
                // emit numbers that look like a trajectory and are not.
                filter.predict(&imu, dt_s)?;
                let mean = filter.get_estimate();
                let cov = filter.get_certainty();
                if let Err(e) = monitor.check(mean.as_slice(), &cov, None) {
                    log::error!("Health fail after propagate at {ts} (#{i}): {e}");
                    bail!(e);
                }
            }
            Event::Measurement { meas, .. } => {
                let outcome = match filter.update(meas.as_ref()) {
                    Ok(outcome) => {
                        consecutive_rejections = 0;
                        outcome
                    }
                    // A measurement the filter cannot use -- an off-map geophysical sample,
                    // an unavailable external model -- leaves the state untouched and valid.
                    // Aborting on it would make geophysical aiding unusable at map edges,
                    // which is the condition it exists to handle (#254).
                    Err(e) if e.is_recoverable() => {
                        rejected_measurements += 1;
                        consecutive_rejections += 1;
                        log::warn!("Measurement rejected at {ts} (#{i}): {e}");
                        if consecutive_rejections > MAX_CONSECUTIVE_REJECTIONS {
                            bail!(
                                "aborting: {consecutive_rejections} consecutive measurements \
                                 rejected, most recently at {ts} (#{i}): {e}"
                            );
                        }
                        // State is unchanged, so the health check has nothing new to judge.
                        continue;
                    }
                    Err(e) => {
                        log::error!("Filter update failed at {ts} (#{i}): {e}");
                        bail!(e);
                    }
                };
                if !outcome.accepted {
                    gated_measurements += 1;
                    log::debug!(
                        "Measurement gated out at {ts} (#{i}): NIS = {:.3} on {} dof",
                        outcome.nis,
                        outcome.dof
                    );
                }
                if outcome.forced {
                    forced_measurements += 1;
                    log::debug!(
                        "Measurement forced through the gate at {ts} (#{i}): NIS = {:.3} on \
                         {} dof",
                        outcome.nis,
                        outcome.dof
                    );
                }
                let mean = filter.get_estimate();
                let cov = filter.get_certainty();
                // The real NIS, at last. `HealthMonitor` counts consecutive
                // exceedances, which is the check the per-measurement gate cannot
                // make: gating rejects outliers one at a time and would happily
                // reject every fix of a diverged run without ever saying so.
                if let Err(e) = monitor.check(mean.as_slice(), &cov, Some(outcome.nis)) {
                    log::error!("Health fail after measurement update at {ts} (#{i}): {e}");
                    bail!(e);
                }
            }
        }
    }

    // Flush the final epoch. Nothing inside the loop can emit it -- the boundary push only
    // fires when a *later* timestamp arrives, and there is none -- so this is where the last
    // row comes from, with every event applied.
    //
    // The `i == total - 1` push this replaces ran *in addition to* the boundary push when the
    // last event happened to be the first at its timestamp, emitting one state under two
    // labels. It never fired on `test_data.csv`, where each epoch carries up to four events,
    // which is why it survived.
    if let Some(final_ts) = last_ts.filter(|ts| *ts != start_time) {
        let mean = filter.get_estimate();
        let cov = filter.get_certainty();
        results.push(NavigationResult::from((&final_ts, &mean, &cov, layout)));
        debug!("Filter state at {final_ts}: {mean:?}");
    }

    debug!("Closed-loop simulation complete");
    // Report the total even when it is zero: a silent run and a run that rejected every
    // measurement look identical from the outside otherwise.
    if rejected_measurements > 0 {
        log::warn!(
            "closed-loop run completed with {rejected_measurements} of {total} events rejected as unusable measurements"
        );
    }
    // Reported separately from the line above: "unusable" means the measurement could
    // not be evaluated, "gated out" means it was evaluated and disbelieved. Folding
    // them together would hide a filter that is quietly refusing every valid fix.
    if gated_measurements > 0 {
        log::warn!(
            "closed-loop run completed with {gated_measurements} of {total} events gated out by the innovation test"
        );
    }
    // Forced updates are the recovery path doing its job (#340), but they are also the
    // filter being overruled: a run with a lot of them is one whose covariance or
    // process noise does not describe the trajectory it was handed.
    if forced_measurements > 0 {
        log::warn!(
            "closed-loop run completed with {forced_measurements} of {total} events applied \
             despite failing the innovation test, after repeated consecutive rejections"
        );
    }
    Ok(results)
}
/// Print the Unscented Kalman Filter state and covariance for debugging purposes.
///
/// The reference each error is measured against is the same quantity
/// [`TestDataRecord::initial_state`] seeds the filter from: the ground track through
/// [`TestDataRecord::ground_track_velocity`] and the attitude through
/// [`TestDataRecord::attitude`]. Taking `speed * bearing.cos()` and the raw
/// `roll`/`pitch`/`yaw` columns instead -- degrees fed to a radian trig call, and Euler
/// angles in a convention that is not nalgebra's -- made this diagnostic report a large
/// error for a correctly initialised filter.
pub fn print_ukf(ukf: &UnscentedKalmanFilter, record: &TestDataRecord) {
    let (reference_north, reference_east) = record.ground_track_velocity();
    let (reference_roll, reference_pitch, reference_yaw) = record.attitude().euler_angles();
    debug!(
        "UKF position: ({:.4}, {:.4}, {:.4})  |  Covariance: {:.4e}, {:.4e}, {:.4}  |  Error: {:.4e}, {:.4e}, {:.4}",
        ukf.get_estimate()[0].to_degrees(),
        ukf.get_estimate()[1].to_degrees(),
        ukf.get_estimate()[2],
        ukf.get_certainty()[(0, 0)],
        ukf.get_certainty()[(1, 1)],
        ukf.get_certainty()[(2, 2)],
        ukf.get_estimate()[0].to_degrees() - record.latitude,
        ukf.get_estimate()[1].to_degrees() - record.longitude,
        ukf.get_estimate()[2] - record.altitude
    );
    debug!(
        "UKF velocity: ({:.4}, {:.4}, {:.4})  | Covariance: {:.4}, {:.4}, {:.4}  | Error: {:.4}, {:.4}, {:.4}",
        ukf.get_estimate()[3],
        ukf.get_estimate()[4],
        ukf.get_estimate()[5],
        ukf.get_certainty()[(3, 3)],
        ukf.get_certainty()[(4, 4)],
        ukf.get_certainty()[(5, 5)],
        ukf.get_estimate()[3] - reference_north,
        ukf.get_estimate()[4] - reference_east,
        ukf.get_estimate()[5] - 0.0 // Assuming no vertical velocity
    );
    debug!(
        "UKF attitude: ({:.4}, {:.4}, {:.4})  | Covariance: {:.4}, {:.4}, {:.4}  | Error: {:.4}, {:.4}, {:.4}",
        ukf.get_estimate()[6],
        ukf.get_estimate()[7],
        ukf.get_estimate()[8],
        ukf.get_certainty()[(6, 6)],
        ukf.get_certainty()[(7, 7)],
        ukf.get_certainty()[(8, 8)],
        ukf.get_estimate()[6] - reference_roll,
        ukf.get_estimate()[7] - reference_pitch,
        ukf.get_estimate()[8] - reference_yaw
    );
    debug!(
        "UKF accel biases: ({:.4}, {:.4}, {:.4})  | Covariance: {:.4e}, {:.4e}, {:.4e}",
        ukf.get_estimate()[9],
        ukf.get_estimate()[10],
        ukf.get_estimate()[11],
        ukf.get_certainty()[(9, 9)],
        ukf.get_certainty()[(10, 10)],
        ukf.get_certainty()[(11, 11)]
    );
    debug!(
        "UKF gyro biases: ({:.4}, {:.4}, {:.4})  | Covariance: {:.4e}, {:.4e}, {:.4e}",
        ukf.get_estimate()[12],
        ukf.get_estimate()[13],
        ukf.get_estimate()[14],
        ukf.get_certainty()[(12, 12)],
        ukf.get_certainty()[(13, 13)],
        ukf.get_certainty()[(14, 14)]
    );
}

/// Configuration parameters for UKF initialization.
///
/// This struct groups together optional parameters for initializing an Unscented Kalman Filter,
/// reducing the number of function arguments and making it easier to specify custom configurations.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct UkfConfig {
    /// Optional vector of f64 representing the initial attitude covariance (default is a small value).
    pub attitude_covariance: Option<Vec<f64>>,
    /// Optional initial IMU bias **estimate**, 6 elements: 3 accelerometer, 3 gyroscope.
    ///
    /// Defaults to **zero** on all six -- a filter that is uncertain about its biases, not one
    /// asserting it has them. This used to default to `1e-3`, which was the *covariance* on
    /// the line below copied into the estimate, so every UKF built without an explicit value
    /// opened by claiming a 1 mrad/s (0.0573 deg/s) rate bias on every gyroscope axis and
    /// subtracting it from every sample (#392).
    pub imu_biases: Option<Vec<f64>>,
    /// Optional initial IMU bias covariance diagonal, 6 elements in the same order.
    ///
    /// Defaults to [`crate::IMUQuality::initial_bias_covariance`] for [`Self::imu_quality`],
    /// not to a constant. Read independently of [`Self::imu_biases`]: setting one without the
    /// other used to discard this silently (#392).
    pub imu_biases_covariance: Option<Vec<f64>>,
    /// Optional vector of f64 for any additional states (not used in the canonical UKF, but can be useful for custom implementations).
    pub other_states: Option<Vec<f64>>,
    /// Optional vector of f64 for other states covariance.
    pub other_states_covariance: Option<Vec<f64>>,
    /// Optional process noise diagonal vector.
    pub process_noise_diagonal: Option<Vec<f64>>,
    /// Optional UKF alpha parameter (sigma-point spread).
    pub ukf_alpha: Option<f64>,
    /// Optional UKF beta parameter (prior distribution).
    pub ukf_beta: Option<f64>,
    /// Optional UKF kappa parameter (secondary spread control).
    pub ukf_kappa: Option<f64>,
    /// IMU grade the initial bias covariance is derived from when
    /// `imu_biases_covariance` is not given.
    ///
    /// [`crate::IMUQuality::initial_bias_covariance`] turns the grade's bias instability into the
    /// six-entry diagonal, so a filter opens believing its biases to within about one
    /// instability of zero. Before this field the three constructors each hard-coded a
    /// different answer and none of them modelled any hardware -- see that method for the
    /// measurement, and for why it made the UKF-versus-ESKF comparison in #371 meaningless.
    pub imu_quality: crate::IMUQuality,
    /// Estimate a barometric altitude bias as an extra state (#372).
    ///
    /// `false` -- the default -- models the barometer as unbiased, which is what every filter
    /// did before #372. `true` appends one state **after** any [`Self::other_states`], so it
    /// cannot collide with the map-bias indices geonav computes from a base of fifteen; read
    /// the resulting index off [`UkfConfig::baro_bias_index`] rather than assuming it.
    ///
    /// The state opens at [`INITIAL_BARO_BIAS_VARIANCE_M2`] and walks at
    /// [`BARO_BIAS_PROCESS_NOISE_M2_PER_S`], both derived from one hectopascal of
    /// reference-pressure drift per hour.
    pub estimate_baro_bias: bool,
    /// Local-level frame of the records: `false` (the default) is NED, `true` is ENU.
    ///
    /// Sensor Logger exports are ENU -- at rest their specific force lands on the device's
    /// up-axis at $+g$ -- while anything from [`generate_synthetic`] or `strapdown-sim syn`
    /// is NED. [`TestDataRecord`] carries no frame tag, so the caller has to say which, and
    /// [`check_declared_frame`] rejects a declaration the data contradicts rather than
    /// silently mechanizing at 2 g (#296).
    pub is_enu: bool,
}

/// Reject an invalid configuration value.
///
/// Every call below guards a length that comes from a user-authored TOML/YAML/JSON config,
/// so the failure is a report-and-exit condition rather than a crate invariant (#254).
fn require_config(ok: bool, field: &'static str, reason: String) -> Result<(), StrapdownError> {
    if ok {
        Ok(())
    } else {
        Err(StrapdownError::InvalidConfiguration { field, reason })
    }
}

/// Helper function to initialize a UKF for closed-loop mode.
///
/// This function sets up the Unscented Kalman Filter (UKF) with initial pose and configuration parameters.
/// It initializes the UKF with position, velocity, attitude, and covariance matrices.
///
/// # Arguments
///
/// * `initial_pose` - A `TestDataRecord` containing the initial pose information.
/// * `config` - A `UkfConfig` struct containing optional configuration parameters.
///
/// # Returns
///
/// * `UnscentedKalmanFilter` - An instance of the Unscented Kalman Filter initialized with the provided parameters.
/// # Errors
/// [`StrapdownError::InvalidConfiguration`] if any configured vector length disagrees with
/// the filter's state size.
///
/// **This function does not validate `config.is_enu` against the data**, and deliberately so:
/// it is handed one [`TestDataRecord`], and a single sample cannot tell a frame error from a
/// motion transient. Checking it here was worse than not checking it -- one NaN or accelerating
/// first sample disabled the guard entirely, while one ordinary sample above 1.5 g rejected a
/// correctly declared file and advised the caller to flip the flag. Call
/// [`check_declared_frame`] over the whole record slice instead, as `strapdown-sim` does on
/// every path (#296).
pub fn initialize_ukf(
    initial_pose: &TestDataRecord,
    config: UkfConfig,
) -> Result<UnscentedKalmanFilter, StrapdownError> {
    // Units, attitude source and frame all live in `TestDataRecord::initial_state`; see it
    // for why this is not a struct literal any more (#337). The frame is the caller's
    // declared one, checked against the data by `check_declared_frame` rather than assumed
    // here: `TestDataRecord` carries no frame tag -- Sensor Logger exports (ENU-convention:
    // +g along the device's up-axis at rest) and `generate_synthetic` output (NED) are
    // indistinguishable once loaded -- so this has to be supplied, and supplying it wrongly
    // is what that guard is for (#296).
    let initial_state = initial_pose.initial_state(config.is_enu);
    // Read before `config` is consumed field by field below. The barometric bias goes last,
    // after any `other_states`, so its index is not a constant and `UkfConfig::baro_bias_index`
    // is the one place that arithmetic lives (#372).
    let baro_bias_index = config.baro_bias_index();
    let process_noise_diagonal = match config.process_noise_diagonal {
        Some(pn) => pn,
        None => DEFAULT_PROCESS_NOISE_DENSITY.to_vec(),
    };
    // Covariance parameters
    let position_accuracy = initial_pose.horizontal_accuracy; //.sqrt();
    let position_std_rad = (position_accuracy * METERS_TO_DEGREES).to_radians();
    let mut covariance_diagonal = vec![
        position_std_rad.powf(2.0),
        position_std_rad.powf(2.0),
        initial_pose.vertical_accuracy.powf(2.0),
        initial_pose.speed_accuracy.powf(2.0),
        initial_pose.speed_accuracy.powf(2.0),
        initial_pose.speed_accuracy.powf(2.0),
    ];
    // extend the covariance diagonal if attitude covariance is provided
    match config.attitude_covariance {
        Some(att_cov) => covariance_diagonal.extend(att_cov),
        None => covariance_diagonal.extend(vec![1e-9; 3]), // Default values if not provided
    }
    // The IMU bias estimate and the uncertainty in that estimate are two independent
    // settings, and are read independently. They were not: `imu_biases_covariance` used to be
    // consulted only inside the `if let Some(imu_biases)` arm, so a caller who set the
    // covariance without also setting the estimate had it **silently discarded** in favour of
    // the hard-coded default -- a `pub` config field that did nothing, which is the failure
    // mode `book/src/user-guide/configuration.md` calls "the single most common way to get a
    // wrong result out of the simulator".
    covariance_diagonal.extend(match config.imu_biases_covariance {
        Some(imu_cov) => {
            // Length-checked here, like the EKF and ESKF already do. Without it a malformed
            // vector could still pass the whole-diagonal check below whenever `other_states`
            // made the total come out right, sliding an extra-state variance into a bias slot
            // instead of returning `InvalidConfiguration`.
            require_config(
                imu_cov.len() == 6,
                "imu_biases_covariance",
                format!("expected 6 elements, got {}", imu_cov.len()),
            )?;
            imu_cov
        }
        // Derived from the IMU grade, not a constant. See
        // `IMUQuality::initial_bias_covariance`: the `1e-3` this replaces was a gyro-bias
        // sigma of 1.81 deg/s against a consumer part's 0.028, and the ESKF's own hard-coded
        // answer disagreed with it by five orders of magnitude.
        None => config.imu_quality.initial_bias_covariance().to_vec(),
    });
    // Zero, and the history is worth keeping because it is the defect this whole block
    // exists to not repeat. This used to read `vec![1e-3; 6]` -- the *covariance* that the
    // line above used to carry, copied down into the estimate. The filter opened by asserting
    // a 1 mrad/s rate bias on every gyroscope axis and a 1 mm/s^2 bias on every accelerometer
    // axis, as a point estimate rather than an uncertainty, and subtracted it from every
    // sample. 1e-3 rad/s is 0.057 deg/s, and that is exactly the rate at which the UKF's
    // attitude walked away from `dead_reckoning` on an IMU-only stream. `initialize_ekf` and
    // `initialize_eskf` both wrote `vec![0.0; 6]` here; the UKF was the only one of the three
    // that did not (#392).
    //
    // The `1e-3` is gone from both places now -- the covariance above is derived from the IMU
    // grade (#393) -- so this comment describes what was fixed rather than what is there.
    let imu_biases = config.imu_biases.unwrap_or_else(|| vec![0.0; 6]);
    // extend the covariance diagonal if other states are provided
    let other_states = match config.other_states {
        Some(other_states) => {
            covariance_diagonal.extend(match config.other_states_covariance {
                Some(other_cov) => other_cov,
                None => vec![1e-3; other_states.len()], // Default covariance if not provided
            });
            Some(other_states)
        }
        None => None,
    };
    // The barometric bias goes last, after any map biases, so it cannot collide with the
    // indices geonav derives from a base of fifteen.
    let other_states = if config.estimate_baro_bias {
        covariance_diagonal.push(INITIAL_BARO_BIAS_VARIANCE_M2);
        let mut extra = other_states.unwrap_or_default();
        extra.push(0.0);
        Some(extra)
    } else {
        other_states
    };
    let mut process_noise_diagonal = process_noise_diagonal;
    if config.estimate_baro_bias
        && process_noise_diagonal.len() + 1 == 15 + other_states.as_ref().map_or(0, Vec::len)
    {
        // The caller gave a diagonal sized for the state without this one; extend it rather
        // than making every caller that turns the flag on also hand-build a longer vector.
        process_noise_diagonal.push(BARO_BIAS_PROCESS_NOISE_M2_PER_S);
    }
    let expected = 15 + other_states.as_ref().map_or(0, Vec::len);
    require_config(
        covariance_diagonal.len() == expected,
        "covariance_diagonal",
        format!(
            "expected {expected} elements, got {}",
            covariance_diagonal.len()
        ),
    )?;
    require_config(
        process_noise_diagonal.len() == expected,
        "process_noise_diagonal",
        format!(
            "expected {expected} elements, got {}",
            process_noise_diagonal.len()
        ),
    )?;
    require_config(
        process_noise_diagonal.len() == covariance_diagonal.len(),
        "process_noise_diagonal",
        format!(
            "must match covariance_diagonal: {} vs {}",
            process_noise_diagonal.len(),
            covariance_diagonal.len()
        ),
    )?;
    let process_noise = DMatrix::from_diagonal(&DVector::from_vec(process_noise_diagonal));
    //DVector::from_vec(vec![0.0; 15]);
    let mut filter = UnscentedKalmanFilter::new(
        &initial_state,
        &imu_biases,
        other_states.as_deref(),
        covariance_diagonal,
        process_noise,
        config.ukf_alpha.unwrap_or(1e-3),
        config.ukf_beta.unwrap_or(2.0),
        config.ukf_kappa.unwrap_or(0.0),
    );
    // This function sized the state, so it is the one thing that knows where the bias landed.
    // Telling the filter is what lets `run_closed_loop` label the column without being handed
    // a `ExtraStateLayout` (#372).
    filter.set_baro_bias_index(baro_bias_index)?;
    Ok(filter)
}

impl UkfConfig {
    /// Where [`Self::estimate_baro_bias`] puts the barometric bias, if it is on.
    ///
    /// One source of truth for the index, because three places need it and they must agree:
    /// the filter's state layout, the
    /// [`AidingConfig::baro_bias_index`](crate::messages::AidingConfig)
    /// that tells the measurement which state to read, and the [`ExtraStateLayout`] that labels
    /// it on the way out. A caller computing `15 + n` by hand in each of those is how the two
    /// drift apart, and a measurement pointed at the wrong state reads a *map* bias as a
    /// barometric one -- which is the hazard `strapdown-geonav`'s `BiasState` documentation
    /// describes for index 14.
    ///
    /// After the other states, not before them: geonav derives its map-bias indices from a
    /// base of [`NAVIGATION_STATES`], so taking index 15 for the barometer would collide with
    /// a gravity bias on any geophysical run.
    #[must_use]
    pub fn baro_bias_index(&self) -> Option<usize> {
        self.estimate_baro_bias
            .then(|| NAVIGATION_STATES + self.other_states.as_ref().map_or(0, Vec::len))
    }
}

/// Configuration parameters for EKF initialization.
///
/// Mirrors [`UkfConfig`]: the alternative is a seventh positional argument on
/// [`initialize_ekf`], which already carried five `Option`s whose order the compiler cannot
/// check for you.
///
/// Note that [`Default`] is written by hand rather than derived, because a derived one would
/// give `use_biases: false` and silently demote every caller from the 15-state EKF to the
/// 9-state one -- a retune disguised as a struct literal.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct EkfConfig {
    /// Optional initial attitude covariance (3 elements, rad^2).
    pub attitude_covariance: Option<Vec<f64>>,
    /// Optional initial IMU bias **estimate**, 6 elements: 3 accelerometer, 3 gyroscope.
    ///
    /// Defaults to zero on all six -- an uncertainty about the biases, never a claim about
    /// them. See [`UkfConfig::imu_biases`] for why that is worth stating (#392).
    pub imu_biases: Option<Vec<f64>>,
    /// Optional initial IMU bias covariance diagonal, 6 elements in the same order.
    ///
    /// Defaults to [`crate::IMUQuality::initial_bias_covariance`] for [`Self::imu_quality`].
    /// Read **independently** of [`Self::imu_biases`]: setting one without the other used to
    /// discard this silently on this constructor, which is #392 on the one filter it was not
    /// fixed on.
    pub imu_biases_covariance: Option<Vec<f64>>,
    /// Optional process noise diagonal (9 or 15 elements, matching `use_biases`).
    pub process_noise_diagonal: Option<Vec<f64>>,
    /// 15-state (navigation states plus IMU biases) when `true`, 9-state otherwise.
    pub use_biases: bool,
    /// Estimate a barometric altitude bias as a sixteenth state (#372).
    ///
    /// `false` -- the default -- models the barometer as unbiased. `true` requires
    /// [`Self::use_biases`], because the state is appended after the IMU bias block; read the
    /// index off [`EkfConfig::baro_bias_index`] rather than assuming it.
    ///
    /// The state opens at [`INITIAL_BARO_BIAS_VARIANCE_M2`] and walks at
    /// [`BARO_BIAS_PROCESS_NOISE_M2_PER_S`], both derived from one hectopascal of
    /// reference-pressure drift per hour.
    pub estimate_baro_bias: bool,
    /// IMU grade the initial bias covariance is derived from when
    /// `imu_biases_covariance` is not given.
    ///
    /// [`crate::IMUQuality::initial_bias_covariance`] turns the grade's bias instability into the
    /// six-entry diagonal, so a filter opens believing its biases to within about one
    /// instability of zero. Before this field the three constructors each hard-coded a
    /// different answer and none of them modelled any hardware -- see that method for the
    /// measurement, and for why it made the UKF-versus-ESKF comparison in #371 meaningless.
    ///
    /// **On this filter the value is currently inert**, and measurably so: any value produces
    /// bit-identical output, because the EKF's state-transition Jacobian has no
    /// $\partial(\text{nav})/\partial(\text{bias})$ block, so `P[0..9, 9..15]` starts at zero
    /// and stays there and the gain over the bias rows is always zero (#394). It is set
    /// correctly here anyway: the field is what the filter *claims*, the claim should be true
    /// whether or not anything reads it today, and #394's fix makes it load-bearing without
    /// touching this line.
    pub imu_quality: crate::IMUQuality,
    /// Local-level frame of the records: `false` (the default) is NED, `true` is ENU.
    ///
    /// See [`UkfConfig::is_enu`]; the same reasoning and the same guard apply.
    pub is_enu: bool,
}

impl Default for EkfConfig {
    fn default() -> Self {
        Self {
            attitude_covariance: None,
            imu_biases: None,
            imu_biases_covariance: None,
            process_noise_diagonal: None,
            // Every caller in this workspace asked for the 15-state filter before this
            // struct existed, and estimating the IMU biases is the whole reason to prefer
            // the EKF over dead reckoning on a drifting sensor. Deriving `Default` here
            // would flip that to 9-state without a diff anyone would read as a retune.
            use_biases: true,
            estimate_baro_bias: false,
            imu_quality: crate::IMUQuality::default(),
            is_enu: false,
        }
    }
}

/// Initialize an Extended Kalman Filter for simulation.
///
/// This function creates and initializes an `ExtendedKalmanFilter` with the given parameters,
/// providing a linearized Gaussian approximation for navigation state estimation.
///
/// # Arguments
///
/// * `initial_pose` - A `TestDataRecord` containing the initial pose information.
/// * `config` - An [`EkfConfig`] carrying the optional covariance, bias, process-noise,
///   state-size and frame settings.
///
/// # Returns
///
/// * `ExtendedKalmanFilter` - An instance of the Extended Kalman Filter.
///
/// # Errors
/// [`StrapdownError::InvalidConfiguration`] if any configured vector length disagrees with
/// the filter's state size.
///
/// **This function does not validate `config.is_enu` against the data**, and deliberately so:
/// it is handed one [`TestDataRecord`], and a single sample cannot tell a frame error from a
/// motion transient. Checking it here was worse than not checking it -- one NaN or accelerating
/// first sample disabled the guard entirely, while one ordinary sample above 1.5 g rejected a
/// correctly declared file and advised the caller to flip the flag. Call
/// [`check_declared_frame`] over the whole record slice instead, as `strapdown-sim` does on
/// every path (#296).
pub fn initialize_ekf(
    initial_pose: &TestDataRecord,
    config: EkfConfig,
) -> Result<crate::kalman::ExtendedKalmanFilter, StrapdownError> {
    use crate::kalman::ExtendedKalmanFilter;

    // Read before the destructure below consumes `config`, so the index comes from
    // `EkfConfig::baro_bias_index` rather than being spelled out a second time here (#372).
    let baro_bias_index = config.baro_bias_index();
    let EkfConfig {
        attitude_covariance,
        imu_biases,
        imu_biases_covariance,
        process_noise_diagonal,
        use_biases,
        estimate_baro_bias,
        imu_quality,
        is_enu,
    } = config;

    // Build initial state from sensor data. Shared with `initialize_ukf` and
    // `initialize_eskf` so the three agree on units and on where the attitude comes from
    // (#337); the frame is the caller's declared one, checked by `check_declared_frame`
    // (#296).
    let initial_state = initial_pose.initial_state(is_enu);

    // Determine state size based on use_biases flag, plus the barometric bias if asked for.
    //
    // The barometer's bias sits after the IMU bias block, so it needs that block to exist; a
    // 9-state EKF has nowhere to put it. Rejected rather than silently ignored -- a `pub`
    // config field that does nothing is #392's whole failure mode.
    require_config(
        !estimate_baro_bias || use_biases,
        "estimate_baro_bias",
        "requires use_biases: the barometric bias is appended after the IMU bias block".to_string(),
    )?;
    let state_size = if use_biases { 15 } else { 9 } + usize::from(estimate_baro_bias);

    // Build process noise diagonal
    let process_noise_diagonal = if let Some(pn) = process_noise_diagonal {
        require_config(
            pn.len() == state_size,
            "process_noise_diagonal",
            format!("expected {state_size} elements, got {}", pn.len()),
        )?;
        pn
    } else {
        let mut default = if use_biases {
            DEFAULT_PROCESS_NOISE_DENSITY.to_vec()
        } else {
            DEFAULT_PROCESS_NOISE_DENSITY[0..9].to_vec()
        };
        if estimate_baro_bias {
            default.push(BARO_BIAS_PROCESS_NOISE_M2_PER_S);
        }
        default
    };

    // Build covariance diagonal.
    //
    // The EKF holds latitude and longitude in radians, so the reported accuracy needs *both*
    // conversions, not just the metres-to-degrees one: this read
    // `(position_accuracy * METERS_TO_DEGREES).powf(2.0)` until #308, which is degrees
    // squared on a radian state -- 57.3x too large as a standard deviation, 3283x in
    // variance, so a 5 m fix was entered as a 286 m one. `initialize_ukf` above spells the
    // same conversion out as `(position_accuracy * METERS_TO_DEGREES).to_radians()`;
    // [`METERS_TO_RADIANS`] is that composition as a single constant.
    let position_accuracy = initial_pose.horizontal_accuracy;
    let position_std_rad = position_accuracy * METERS_TO_RADIANS;
    let mut covariance_diagonal = vec![
        position_std_rad.powf(2.0),
        position_std_rad.powf(2.0),
        initial_pose.vertical_accuracy.powf(2.0),
        initial_pose.speed_accuracy.powf(2.0),
        initial_pose.speed_accuracy.powf(2.0),
        initial_pose.speed_accuracy.powf(2.0),
    ];

    // Add attitude covariance
    match attitude_covariance {
        Some(att_cov) => {
            require_config(
                att_cov.len() == 3,
                "attitude_covariance",
                format!("expected 3 elements, got {}", att_cov.len()),
            )?;
            covariance_diagonal.extend(att_cov);
        }
        None => covariance_diagonal.extend(vec![1e-9; 3]),
    }

    // Add IMU bias covariance if using biases
    let imu_biases_vec = if use_biases {
        // The estimate and its uncertainty are two independent settings, read independently.
        // This arm used to read the covariance only when an estimate was also supplied, so a
        // caller who set `imu_biases_covariance` and left `imu_biases` at `None` -- wanting a
        // custom uncertainty about a zero bias, which is the ordinary case -- had it
        // **silently discarded** for the grade default. That is #392's defect exactly, on the
        // one constructor it was not fixed on: the UKF and ESKF both read it unconditionally.
        covariance_diagonal.extend(match imu_biases_covariance {
            Some(imu_cov) => {
                require_config(
                    imu_cov.len() == 6,
                    "imu_biases_covariance",
                    format!("expected 6 elements, got {}", imu_cov.len()),
                )?;
                imu_cov
            }
            None => imu_quality.initial_bias_covariance().to_vec(),
        });
        if let Some(biases) = imu_biases {
            require_config(
                biases.len() == 6,
                "imu_biases",
                format!("expected 6 elements, got {}", biases.len()),
            )?;
            biases
        } else {
            vec![0.0; 6]
        }
    } else {
        vec![0.0; 6] // Not used in 9-state, but required by constructor
    };

    if estimate_baro_bias {
        covariance_diagonal.push(INITIAL_BARO_BIAS_VARIANCE_M2);
    }

    require_config(
        covariance_diagonal.len() == state_size,
        "covariance_diagonal",
        format!(
            "expected {state_size} elements, got {}",
            covariance_diagonal.len()
        ),
    )?;

    let process_noise = DMatrix::from_diagonal(&DVector::from_vec(process_noise_diagonal));
    let mut filter = ExtendedKalmanFilter::new(
        &initial_state,
        &imu_biases_vec,
        covariance_diagonal,
        process_noise,
        use_biases,
    );
    // As in `initialize_ukf`: the constructor knows the layout, so it declares it.
    filter.set_baro_bias_index(baro_bias_index)?;
    Ok(filter)
}

impl EkfConfig {
    /// Where [`Self::estimate_baro_bias`] puts the barometric bias, if it is on.
    ///
    /// Always [`NAVIGATION_STATES`] here, because this constructor has no `other_states` to
    /// append after. It is still a method rather than a literal for the reason
    /// [`UkfConfig::baro_bias_index`] gives: three places have to agree on the index, and the
    /// one that computes it by hand is the one that drifts.
    #[must_use]
    pub const fn baro_bias_index(&self) -> Option<usize> {
        if self.estimate_baro_bias {
            Some(NAVIGATION_STATES)
        } else {
            None
        }
    }
}

/// Configuration parameters for ESKF initialization.
///
/// Mirrors [`EkfConfig`] minus `use_biases`: the error-state filter always carries the full
/// fifteen (position, velocity, attitude, accelerometer bias, gyroscope bias), so there is
/// nothing to select. [`Self::estimate_baro_bias`] adds a sixteenth, so a
/// [`Self::process_noise_diagonal`] or `imu_biases_covariance` handed in alongside it is
/// sized to sixteen and **not** fifteen; read [`Self::baro_bias_index`] for where it lands.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct EskfConfig {
    /// Optional initial attitude error covariance (3 elements, rad^2).
    pub attitude_covariance: Option<Vec<f64>>,
    /// Optional initial IMU bias **estimate**, 6 elements: `b_ax`, `b_ay`, `b_az`, `b_gx`,
    /// `b_gy`, `b_gz`.
    ///
    /// Defaults to zero on all six -- an uncertainty about the biases, never a claim about
    /// them. See [`UkfConfig::imu_biases`] for why that is worth stating (#392).
    pub imu_biases: Option<Vec<f64>>,
    /// Optional IMU bias error covariance (6 elements).
    pub imu_biases_covariance: Option<Vec<f64>>,
    /// Optional process noise diagonal (15 elements for the error state, 16 with a
    /// barometric bias).
    pub process_noise_diagonal: Option<Vec<f64>>,
    /// Estimate a barometric altitude bias as a sixteenth error state (#372).
    ///
    /// `false` -- the default -- is the fifteen-state error vector this filter has always
    /// carried. Read the index off [`EskfConfig::baro_bias_index`] rather than assuming it.
    ///
    /// The state opens at [`INITIAL_BARO_BIAS_VARIANCE_M2`] and walks at
    /// [`BARO_BIAS_PROCESS_NOISE_M2_PER_S`].
    pub estimate_baro_bias: bool,
    /// IMU grade the initial bias covariance is derived from when
    /// `imu_biases_covariance` is not given.
    ///
    /// [`crate::IMUQuality::initial_bias_covariance`] turns the grade's bias instability into the
    /// six-entry diagonal, so a filter opens believing its biases to within about one
    /// instability of zero. Before this field the three constructors each hard-coded a
    /// different answer and none of them modelled any hardware -- see that method for the
    /// measurement, and for why it made the UKF-versus-ESKF comparison in #371 meaningless.
    pub imu_quality: crate::IMUQuality,
    /// Local-level frame of the records: `false` (the default) is NED, `true` is ENU.
    ///
    /// See [`UkfConfig::is_enu`]; the same reasoning and the same guard apply.
    pub is_enu: bool,
}

impl EskfConfig {
    /// Where [`Self::estimate_baro_bias`] puts the barometric bias, if it is on.
    ///
    /// Always [`NAVIGATION_STATES`], as for the EKF. See [`UkfConfig::baro_bias_index`] for
    /// why this is a method.
    #[must_use]
    pub const fn baro_bias_index(&self) -> Option<usize> {
        if self.estimate_baro_bias {
            Some(NAVIGATION_STATES)
        } else {
            None
        }
    }
}

/// Initialize an Error-State Kalman Filter (ESKF) for simulation.
///
/// This function creates and initializes an `ErrorStateKalmanFilter` with the given parameters,
/// providing a robust error-state formulation that uses quaternions for nominal attitude and
/// small-angle representation for attitude errors.
///
/// The ESKF is the standard approach for strapdown INS, offering better numerical stability
/// and avoiding attitude singularities compared to full-state EKF implementations.
///
/// # Arguments
///
/// * `initial_pose` - A `TestDataRecord` containing the initial pose information.
/// * `config` - An [`EskfConfig`] carrying the optional covariance, bias, process-noise and
///   frame settings.
///
/// # Returns
///
/// * `ErrorStateKalmanFilter` - An instance of the Error-State Kalman Filter.
///
/// # Errors
/// [`StrapdownError::InvalidConfiguration`] if any configured vector length disagrees with
/// the filter's state size.
///
/// **This function does not validate `config.is_enu` against the data**, and deliberately so:
/// it is handed one [`TestDataRecord`], and a single sample cannot tell a frame error from a
/// motion transient. Checking it here was worse than not checking it -- one NaN or accelerating
/// first sample disabled the guard entirely, while one ordinary sample above 1.5 g rejected a
/// correctly declared file and advised the caller to flip the flag. Call
/// [`check_declared_frame`] over the whole record slice instead, as `strapdown-sim` does on
/// every path (#296).
///
/// # Example
///
/// ```no_run
/// use strapdown::sim::{EskfConfig, initialize_eskf, TestDataRecord};
/// use chrono::Utc;
///
/// let initial_pose = TestDataRecord {
///     time: Utc::now(),
///     latitude: 45.0,
///     longitude: -122.0,
///     altitude: 100.0,
///     // ... other fields ...
///     ..Default::default()
/// };
/// // `EskfConfig::default()` is NED. For a Sensor Logger export, take the default and set
/// // `is_enu = true` -- `EskfConfig` is `#[non_exhaustive]`, so a struct literal will not
/// // compile outside this crate.
/// let eskf = initialize_eskf(&initial_pose, EskfConfig::default()).unwrap();
/// ```
pub fn initialize_eskf(
    initial_pose: &TestDataRecord,
    config: EskfConfig,
) -> Result<crate::kalman::ErrorStateKalmanFilter, StrapdownError> {
    use crate::kalman::ErrorStateKalmanFilter;

    let EskfConfig {
        attitude_covariance,
        imu_biases,
        imu_biases_covariance,
        process_noise_diagonal,
        estimate_baro_bias,
        imu_quality,
        is_enu,
    } = config;

    // Build initial state from sensor data. Shared with `initialize_ukf` and
    // `initialize_ekf` so the three agree on units and on where the attitude comes from
    // (#337); the frame is the caller's declared one, checked by `check_declared_frame`
    // (#296).
    let initial_state = initial_pose.initial_state(is_enu);

    // Fifteen error states (pos, vel, att, accel bias, gyro bias), plus a barometric bias when
    // asked for. This filter used to hard-code fifteen in six places; it now takes its width
    // from the covariance it is handed, the way the EKF already did (#372).
    let state_size = NAVIGATION_STATES + usize::from(estimate_baro_bias);

    // Build process noise diagonal for the error state: fifteen, or sixteen with the bias.
    let process_noise_diagonal = if let Some(pn) = process_noise_diagonal {
        require_config(
            pn.len() == state_size,
            "process_noise_diagonal",
            format!("expected {state_size} elements, got {}", pn.len()),
        )?;
        pn
    } else {
        let mut default = DEFAULT_PROCESS_NOISE_DENSITY.to_vec();
        if estimate_baro_bias {
            default.push(BARO_BIAS_PROCESS_NOISE_M2_PER_S);
        }
        default
    };

    // Build IMU biases
    // `try_into` rather than `require_config` + a `Vec`: `ErrorStateKalmanFilter::new` takes
    // `&[f64; 6]`, so the length stops being a checked precondition and becomes the type. The
    // conversion is the check.
    let imu_biases: [f64; 6] = match imu_biases {
        Some(biases) => {
            let length = biases.len();
            biases
                .try_into()
                .map_err(|_| StrapdownError::InvalidConfiguration {
                    field: "imu_biases",
                    reason: format!("expected 6 elements, got {length}"),
                })?
        }
        None => [0.0; 6],
    };

    // Build error covariance diagonal.
    //
    // This represents initial uncertainty in the error state (NOT nominal state). The error
    // state's position block is carried in the *filter's* units rather than in metres:
    // radians for latitude and longitude, metres for altitude. `inject_error_state` adds
    // those corrections straight onto the nominal latitude and longitude with no conversion
    // (deliberately -- dividing by the principal radii there is what #266 removed), and the
    // GNSS position Jacobian is the identity against a radian-valued measurement.
    //
    // So the three entries are two different units, and the literals this used to carry --
    // `1e-6, 1e-6, 1e-4`, commented "(m²)" -- were #308's defect in P0 rather than in Q: a
    // 6367 m horizontal claim sitting beside a 1 cm vertical one, in the filter this crate
    // ships as its default. Written from one metric constant and converted once, the way
    // `initialize_ukf` above builds its own P0 and the way `DEFAULT_PROCESS_NOISE_DENSITY` is built.
    let mut error_covariance_diagonal = vec![
        INITIAL_HORIZONTAL_POSITION_VARIANCE_RAD2, // latitude error, rad^2
        INITIAL_HORIZONTAL_POSITION_VARIANCE_RAD2, // longitude error, rad^2
        INITIAL_VERTICAL_POSITION_VARIANCE_M2,     // altitude error, m^2
        1e-3,
        1e-3,
        1e-3, // velocity error covariance (m²/s²)
    ];

    // Add attitude error covariance
    error_covariance_diagonal.extend(match attitude_covariance {
        Some(att_cov) => {
            require_config(
                att_cov.len() == 3,
                "attitude_covariance",
                format!("expected 3 elements, got {}", att_cov.len()),
            )?;
            att_cov
        }
        None => vec![1e-5; 3], // Default: small attitude uncertainty (rad²)
    });

    // Add IMU bias error covariance
    error_covariance_diagonal.extend(match imu_biases_covariance {
        Some(bias_cov) => {
            require_config(
                bias_cov.len() == 6,
                "imu_biases_covariance",
                format!("expected 6 elements, got {}", bias_cov.len()),
            )?;
            bias_cov
        }
        // Derived from the IMU grade like the other two constructors, rather than the
        // `1e-6`/`1e-8` this replaces. That pair was the tightest of the three hard-coded
        // answers -- about 5x tighter than a consumer part's actual bias instability -- and
        // being the only one in the right order of magnitude is what made this filter look
        // like the well-behaved one in every cross-filter comparison.
        None => imu_quality.initial_bias_covariance().to_vec(),
    });

    if estimate_baro_bias {
        error_covariance_diagonal.push(INITIAL_BARO_BIAS_VARIANCE_M2);
    }

    require_config(
        error_covariance_diagonal.len() == state_size,
        "error_covariance_diagonal",
        format!(
            "expected {state_size} elements, got {}",
            error_covariance_diagonal.len()
        ),
    )?;

    let process_noise = DMatrix::from_diagonal(&DVector::from_vec(process_noise_diagonal));
    Ok(ErrorStateKalmanFilter::new(
        &initial_state,
        &imu_biases,
        error_covariance_diagonal,
        process_noise,
    ))
}

// ==== Simulation Helper functions ====

/// Root-sum-square of a position uncertainty, in metres.
///
/// The latitude and longitude standard deviations arrive in radians (the state's native unit);
/// this converts each to metres via [`principal_radii`] -- $(R_N + h) \sigma_{lat}$ north-south,
/// $(R_E + h) \cos(lat) \sigma_{lon}$ east-west -- before combining them with the (already
/// metric) altitude standard deviation, so the three terms being root-sum-squared are the same
/// unit. Combining a radian sigma with a metre sigma directly would make the result dominated by
/// whichever term happens to have the larger *number*, regardless of the physical uncertainty it
/// represents.
///
/// # Arguments
/// - `lat_deg` - latitude in degrees, used to evaluate the local radii of curvature
/// - `alt_m` - altitude in metres
/// - `pos_std_lat_rad` - latitude standard deviation in radians
/// - `pos_std_lon_rad` - longitude standard deviation in radians
/// - `pos_std_alt_m` - altitude standard deviation in metres
fn position_rms_meters(
    lat_deg: f64,
    alt_m: f64,
    pos_std_lat_rad: f64,
    pos_std_lon_rad: f64,
    pos_std_alt_m: f64,
) -> f64 {
    let (r_n, r_e, _) = principal_radii(&lat_deg, &alt_m);
    let pos_std_lat_m = pos_std_lat_rad * (r_n + alt_m);
    let pos_std_lon_m = pos_std_lon_rad * (r_e + alt_m) * lat_deg.to_radians().cos();
    (pos_std_lat_m.powi(2) + pos_std_lon_m.powi(2) + pos_std_alt_m.powi(2)).sqrt()
}

/// Logs a one-line summary of a filter's current position estimate and its uncertainty.
///
/// Reads the filter's mean state and covariance, converts latitude and longitude from radians
/// to degrees, and emits latitude, longitude, altitude (metres) and the three position standard
/// deviations $\sqrt{P_{ii}}$ (degrees, degrees, metres) at `debug` level -- despite the name,
/// nothing is written to stdout, so the message appears only when the logger is configured for
/// [`LogLevel::Debug`] or finer. The horizontal sigmas are also converted to metres (see
/// [`position_rms_meters`]) so they can be root-sum-squared with the (already-metric) altitude
/// sigma into a single, dimensionally meaningful RMS distance.
///
/// The filter must expose at least the three position states; any 9- or 15-state filter in this
/// crate does.
pub fn print_sim_status<F: NavigationFilter>(filter: &F) {
    let mean = filter.get_estimate();
    let cov = filter.get_certainty();

    // Extract position and covariance diagonal
    let lat = mean[0].to_degrees();
    let lon = mean[1].to_degrees();
    let alt = mean[2];

    // Get position uncertainty (diagonal elements), in the state's native units (radians for
    // lat/lon, metres for altitude)
    let pos_std_lat_rad = cov[(0, 0)].sqrt();
    let pos_std_lon_rad = cov[(1, 1)].sqrt();
    let pos_std_alt = cov[(2, 2)].sqrt();

    let pos_rms = position_rms_meters(lat, alt, pos_std_lat_rad, pos_std_lon_rad, pos_std_alt);

    let pos_std_lat_deg = pos_std_lat_rad.to_degrees();
    let pos_std_lon_deg = pos_std_lon_rad.to_degrees();
    debug!(
        "Pos: ({lat:.6}°, {lon:.6}°, {alt:.1}m) | σ: ({pos_std_lat_deg:.2e}°, {pos_std_lon_deg:.2e}°, {pos_std_alt:.2}m) | RMS: {pos_rms:.2e}m"
    );
}

/// Wall-clock guards that stop a simulation which is running too long or has stopped
/// progressing.
///
/// A diverging filter can take arbitrarily long per step without ever failing an arithmetic
/// check, so the simulation drivers pair the numerical guards in [`health`] with a time budget:
/// [`ExecutionLimits`] states the budget and [`ExecutionMonitor`] enforces it, failing the run
/// with a message naming the context it was checked from.
pub mod execution {
    use super::{
        DEFAULT_MAX_NO_PROGRESS_S, DEFAULT_MAX_WALL_CLOCK_RATIO, DEFAULT_MAX_WALL_CLOCK_S, Debug,
        Deserialize, Instant, Result, Serialize, StdDuration, bail, f64,
    };

    /// Configuration for execution timeout limits in simulations.
    ///
    /// This struct provides multiple timeout mechanisms to prevent runaway computations
    /// and detect performance issues during simulation execution:
    ///
    /// - **Wall-clock ratio timeout**: Limits execution time relative to simulation duration
    /// - **Absolute wall-clock timeout**: Enforces a hard limit in real-world seconds
    /// - **No-progress timeout**: Detects when simulation makes no progress (possible hang)
    ///
    /// All timeout values <= 0 are treated as disabled. The most restrictive active timeout
    /// will trigger first. These limits are enforced by [`ExecutionMonitor`] during simulation.
    #[derive(Clone, Debug, Serialize, Deserialize)]
    pub struct ExecutionLimits {
        /// Max wall-clock time as a ratio of simulated duration (<= 0 disables).
        #[serde(default = "default_max_wall_clock_ratio")]
        pub max_wall_clock_ratio: f64,
        /// Max wall-clock time per trajectory in seconds (<= 0 disables).
        #[serde(default = "default_max_wall_clock_s")]
        pub max_wall_clock_s: f64,
        /// Max wall-clock time without progress in seconds (<= 0 disables).
        #[serde(default = "default_max_no_progress_s")]
        pub max_no_progress_s: f64,
    }

    const fn default_max_wall_clock_ratio() -> f64 {
        DEFAULT_MAX_WALL_CLOCK_RATIO
    }

    const fn default_max_wall_clock_s() -> f64 {
        DEFAULT_MAX_WALL_CLOCK_S
    }

    const fn default_max_no_progress_s() -> f64 {
        DEFAULT_MAX_NO_PROGRESS_S
    }

    impl Default for ExecutionLimits {
        fn default() -> Self {
            Self {
                max_wall_clock_ratio: default_max_wall_clock_ratio(),
                max_wall_clock_s: default_max_wall_clock_s(),
                max_no_progress_s: default_max_no_progress_s(),
            }
        }
    }

    /// Monitors execution time and progress during simulation runs.
    ///
    /// This struct tracks wall-clock time and detects stalled simulations by monitoring
    /// when progress was last reported. It enforces timeout limits specified in [`ExecutionLimits`].
    ///
    /// The monitor maintains two types of timeouts:
    /// - **Wall-clock timeout**: Maximum total execution time (computed from simulation duration and limits)
    /// - **No-progress timeout**: Maximum time without calling `mark_progress()`
    ///
    /// Use [`check`](ExecutionMonitor::check) periodically during simulation to verify execution
    /// stays within limits, and call `mark_progress()` after each significant simulation step.
    #[derive(Clone, Debug)]
    pub struct ExecutionMonitor {
        start_time: Instant,
        last_progress: Instant,
        max_wall_clock: Option<StdDuration>,
        max_no_progress: Option<StdDuration>,
    }

    impl ExecutionMonitor {
        /// Creates a new execution monitor with the specified limits.
        ///
        /// # Arguments
        ///
        /// * `limits` - Timeout configuration including wall-clock and no-progress limits
        /// * `sim_duration_s` - Expected simulation duration in seconds, used to compute
        ///   wall-clock timeout when `max_wall_clock_ratio` is enabled
        ///
        /// # Returns
        ///
        /// A new `ExecutionMonitor` instance initialized with the current time.
        pub fn new(limits: &ExecutionLimits, sim_duration_s: f64) -> Self {
            Self::new_at(limits, sim_duration_s, Instant::now())
        }

        /// Construct with an explicit start instant.
        ///
        /// The timeout logic is a pure function of the instants it is given, so the
        /// only thing separating it from a deterministic test is where those instants
        /// come from. This constructor and the `*_at` methods below take them as
        /// arguments so tests can advance a clock instead of sleeping on one.
        ///
        /// `std::thread::sleep` guarantees a *minimum* duration, never a maximum, so a
        /// test that sleeps 10 ms and then asserts a 50 ms budget was not exceeded is
        /// asserting something the standard library does not promise. On a contended
        /// runner it fails. See #284.
        pub(crate) fn new_at(limits: &ExecutionLimits, sim_duration_s: f64, now: Instant) -> Self {
            let max_wall_clock = compute_max_wall_clock(
                sim_duration_s,
                limits.max_wall_clock_ratio,
                limits.max_wall_clock_s,
            );
            let max_no_progress = if limits.max_no_progress_s > 0.0 {
                Some(StdDuration::from_secs_f64(limits.max_no_progress_s))
            } else {
                None
            };

            Self {
                start_time: now,
                last_progress: now,
                max_wall_clock,
                max_no_progress,
            }
        }

        /// Checks if execution is within timeout limits.
        ///
        /// This method verifies that the simulation has not exceeded wall-clock or no-progress
        /// timeout limits. It should be called periodically during simulation execution.
        ///
        /// # Arguments
        ///
        /// * `context` - A string describing the current execution context, included in error messages
        ///
        /// # Returns
        ///
        /// * `Ok(())` if execution is within limits
        /// * `Err(...)` with a descriptive message if any timeout has been exceeded
        ///
        /// # Example
        ///
        /// ```no_run
        /// # use strapdown::sim::{ExecutionMonitor, ExecutionLimits};
        /// let limits = ExecutionLimits::default();
        /// let mut monitor = ExecutionMonitor::new(&limits, 100.0);
        ///
        /// // Check timeout before processing
        /// monitor.check("data processing")?;
        /// // ... do work ...
        /// monitor.mark_progress();
        /// # Ok::<(), anyhow::Error>(())
        /// ```
        /// # Errors
        /// If the wall-clock budget or the no-progress budget has been exceeded.
        pub fn check(&self, context: &str) -> Result<()> {
            self.check_at(context, Instant::now())
        }

        /// [`Self::check`] against an explicit instant. See [`Self::new_at`].
        pub(crate) fn check_at(&self, context: &str, now: Instant) -> Result<()> {
            if let Some(max_wall_clock) = self.max_wall_clock
                && now.duration_since(self.start_time) > max_wall_clock
            {
                bail!(
                    "Execution timeout ({context}): exceeded wall-clock limit of {:.2} s",
                    max_wall_clock.as_secs_f64()
                );
            }
            if let Some(max_no_progress) = self.max_no_progress {
                let since_progress = now.duration_since(self.last_progress);
                if since_progress > max_no_progress {
                    bail!(
                        "Execution timeout ({context}): no progress for {:.2} s (limit {:.2} s)",
                        since_progress.as_secs_f64(),
                        max_no_progress.as_secs_f64()
                    );
                }
            }
            Ok(())
        }

        /// Mark that progress has been made in the simulation.
        /// This should be called after successfully processing each event.
        pub fn mark_progress(&mut self) {
            self.mark_progress_at(Instant::now());
        }

        /// [`Self::mark_progress`] against an explicit instant. See [`Self::new_at`].
        pub(crate) const fn mark_progress_at(&mut self, now: Instant) {
            self.last_progress = now;
        }
    }

    fn compute_max_wall_clock(
        sim_duration_s: f64,
        max_ratio: f64,
        max_wall_clock_s: f64,
    ) -> Option<StdDuration> {
        let mut max_s = if sim_duration_s > 0.0 && max_ratio > 0.0 {
            Some(sim_duration_s * max_ratio)
        } else {
            None
        };
        if max_wall_clock_s > 0.0 {
            max_s = Some(match max_s {
                Some(current) => current.min(max_wall_clock_s),
                None => max_wall_clock_s,
            });
        }

        max_s.and_then(|s| {
            if s > 0.0 {
                Some(StdDuration::from_secs_f64(s))
            } else {
                None
            }
        })
    }
}

/// Divergence detection for a running filter.
///
/// [`HealthMonitor::check`] is called after every predict and update with the current mean and
/// covariance, and aborts the run as soon as the estimate stops being physically or numerically
/// meaningful -- a non-finite state or covariance, a position outside the bounds in
/// [`HealthLimits`], a negative or absurdly large variance on the covariance diagonal, or a run
/// of consecutive measurement updates whose normalised innovation squared (NIS) exceeds its
/// gate. That NIS streak is not GNSS-specific: the monitor is called once per
/// [`crate::messages::Event`] measurement, so barometric altitude, magnetometer-yaw and
/// geophysical updates increment the same counter that GNSS fixes do.
/// This is the circuit breaker behind the per-update gating in [`crate::gating`]: gating rejects
/// individual measurements, the monitor gives up on the whole trajectory.
pub mod health {
    use super::{Debug, Result, bail, f64};

    /// Bounds a filter estimate must stay inside for [`HealthMonitor`] to consider it healthy.
    ///
    /// [`Default`] is deliberately permissive -- in particular the altitude band is opened to
    /// +/-1e8 m so that vertical-channel instability shows up as a covariance or NIS failure
    /// rather than as an altitude bound trip.
    #[derive(Clone, Debug)]
    pub struct HealthLimits {
        /// Inclusive (min, max) latitude band in radians; defaults to the full +/-90 degrees.
        pub lat_rad: (f64, f64),
        /// Inclusive (min, max) longitude band in radians; defaults to the full +/-180 degrees.
        pub lon_rad: (f64, f64),
        /// Inclusive (min, max) altitude band in metres above the ellipsoid. Defaults to
        /// +/-1e8 -- deliberately far wider than the [-11,000 m, 30,000 m] over which the
        /// mechanization is documented to be valid, so that a diverging vertical channel is
        /// caught by the finiteness and covariance checks rather than by this band. Narrow
        /// it to the scenario's real altitude range to make it an effective gate.
        pub alt_m: (f64, f64),
        /// Maximum velocity vector magnitude in m/s -- north, east, *and* down combined, not
        /// ground speed alone (default 500, i.e. road or low-altitude aircraft). Checked
        /// against the NED velocity indices (3..=5), which are correct for every filter in
        /// this crate. Narrow this to the scenario's real speed range to make it an
        /// effective gate; unaided `dead_reckoning` never calls [`HealthMonitor`], so a run
        /// that deliberately drifts past this bound (see #299) is unaffected.
        pub speed_mps_max: f64,
        /// Largest variance allowed on the covariance diagonal before the run is failed
        /// (default 1e15).
        pub cov_diag_max: f64,
        /// NIS above which a measurement update counts as an outlier (default 100).
        ///
        /// Despite the name, the gate applies to **every** measurement update in the event
        /// stream, whatever the sensor: `run_closed_loop` calls [`HealthMonitor::check`] from
        /// the single measurement arm of the event loop, so GNSS position/velocity fixes,
        /// `RelativeAltitudeMeasurement`, magnetometer-yaw and geophysical updates are all
        /// tested against this one threshold. The `_pos` in the field name is historical.
        pub nis_pos_max: f64,
        /// Number of consecutive NIS exceedances that fails the run (default 20). A single
        /// update whose NIS is within [`Self::nis_pos_max`] resets the streak.
        pub nis_pos_consec_fail: usize,
    }

    impl Default for HealthLimits {
        fn default() -> Self {
            Self {
                lat_rad: (-std::f64::consts::FRAC_PI_2, std::f64::consts::FRAC_PI_2),
                lon_rad: (-std::f64::consts::PI, std::f64::consts::PI),
                alt_m: (-100000000.0, 100000000.0), // Very tolerant for vertical channel instability
                speed_mps_max: 500.0,
                cov_diag_max: 1e15,
                nis_pos_max: 100.0,
                nis_pos_consec_fail: 20,
            }
        }
    }

    /// Stateful divergence detector for one simulation run.
    ///
    /// Holds the [`HealthLimits`] to test against plus the only piece of history the tests need:
    /// how many measurement updates in a row have failed the NIS gate, counted across every
    /// sensor rather than GNSS alone. Construct one per trajectory and call
    /// [`check`](Self::check) after every predict and update.
    #[derive(Default, Clone, Debug)]
    pub struct HealthMonitor {
        limits: HealthLimits,
        consec_nis_pos_fail: usize,
    }

    impl HealthMonitor {
        /// Creates a monitor that enforces `limits`, with an empty NIS-failure streak.
        pub const fn new(limits: HealthLimits) -> Self {
            Self {
                limits,
                consec_nis_pos_fail: 0,
            }
        }

        /// Narrowest state [`HealthMonitor::check`] can read: three position and three
        /// velocity components. Every filter in this crate is at least nine wide, so this is a
        /// guard against a caller's mistake rather than a limit anything here runs into.
        pub(crate) const MINIMUM_MONITORED_STATE: usize = 6;

        /// Call after **every event** (predict or update). Provide the optional NIS whenever the
        /// event was a measurement update -- of any sensor, not only GNSS.
        ///
        /// # Errors
        /// If the state has left the configured physical bounds, the covariance diagonal has
        /// grown past its limit, or a supplied NIS exceeds its gate -- i.e. the filter has
        /// diverged and later results would be meaningless.
        pub fn check(
            &mut self,
            x: &[f64], // your mean_state slice
            p: &nalgebra::DMatrix<f64>,
            maybe_nis_pos: Option<f64>,
        ) -> Result<()> {
            // 0) Width. Everything below reads `x[0..=5]`, and `x` is a slice rather than a
            // fixed-size array because a state is 9, 15 or 16 wide depending on the filter.
            // Without this guard a short slice panics here, inside a `pub fn`, in a crate that
            // denies `panic`/`unwrap`/`expect` in library code -- and clippy does not flag
            // slice indexing, so nothing else catches it.
            if x.len() < Self::MINIMUM_MONITORED_STATE {
                bail!(
                    "Health check needs at least {} states \
                     (position and velocity); got {}",
                    Self::MINIMUM_MONITORED_STATE,
                    x.len()
                );
            }

            // 1) Finite checks
            if !x.iter().all(|v| v.is_finite()) {
                bail!("Non-finite state detected");
            }
            if !p.iter().all(|v| v.is_finite()) {
                bail!("Non-finite covariance detected");
            }

            // 2) Basic bounds (lat, lon, alt)
            let lat = x[0];
            let lon = x[1];
            let alt = x[2];
            if lat < self.limits.lat_rad.0 || lat > self.limits.lat_rad.1 {
                bail!("Latitude out of range: {lat}");
            }
            if lon < self.limits.lon_rad.0 || lon > self.limits.lon_rad.1 {
                bail!("Longitude out of range: {lon}");
            }
            if alt < self.limits.alt_m.0 || alt > self.limits.alt_m.1 {
                bail!("Altitude out of range: {alt} m");
            }

            // 3) Speed sanity (assumes NED velocities at indices 3..=5, true for every
            // filter in this crate). `hypot` rather than summing squares directly: x[3..6]
            // are already known finite from the check above, but a naive sum of squares can
            // still overflow to infinity for a merely large (not actually non-finite)
            // component, and `f64::is_finite` on that overflowed value would then read as
            // "no speed to check" and silently wave the divergence through.
            let speed = x[3].hypot(x[4]).hypot(x[5]);
            if speed > self.limits.speed_mps_max {
                bail!("Speed exceeded: {speed:.2} m/s");
            }

            // 4) Covariance sanity: diagonals only. A condition-number check was considered
            // (see #332) but dropped: this covariance's diagonal mixes units -- the position
            // variances are in radians^2 while the velocity and altitude variances are in
            // (m/s)^2 and m^2 -- so even the cheapest proxy, the ratio of the largest to the
            // smallest diagonal entry, is dominated by that unit mismatch rather than by
            // divergence. It fires on a perfectly healthy default P0: lat/lon variance is
            // ~1e-13 rad^2 against a ~4 m^2 altitude variance, a ratio in the 1e12-1e13 range
            // before a single sample has been processed. A true condition number needs a
            // matrix inverse, which this function cannot afford to run on every
            // predict/update.
            for i in 0..p.nrows().min(p.ncols()) {
                if p[(i, i)].is_sign_negative() {
                    bail!("Negative variance on diagonal: idx={i}, val={}", p[(i, i)]);
                }
                if p[(i, i)] > self.limits.cov_diag_max {
                    bail!("Variance too large on diagonal idx={i}: {}", p[(i, i)]);
                }
            }

            // 5) GNSS gating streak (if a NIS was computed at update time)
            if let Some(nis_pos) = maybe_nis_pos {
                if !nis_pos.is_finite() || nis_pos.is_sign_negative() {
                    bail!("Invalid NIS value: {nis_pos}");
                }
                if nis_pos > self.limits.nis_pos_max {
                    self.consec_nis_pos_fail += 1;
                    if self.consec_nis_pos_fail >= self.limits.nis_pos_consec_fail {
                        bail!(
                            "Consecutive NIS exceedances: {} (> {}), last NIS={}",
                            self.consec_nis_pos_fail,
                            self.limits.nis_pos_consec_fail,
                            nis_pos
                        );
                    }
                } else {
                    self.consec_nis_pos_fail = 0;
                }
            }

            Ok(())
        }
    }
}

//================= CLI Argument Structures for Simulation Programs =====================================

/// Scheduler configuration kind
#[derive(Copy, Clone, Debug)]
#[cfg_attr(feature = "clap", derive(ValueEnum))]
pub enum SchedKind {
    /// Deliver every GNSS fix unchanged ([`MeasurementScheduler::PassThrough`]).
    Passthrough,
    /// Deliver a fix every `interval_s` seconds ([`MeasurementScheduler::FixedInterval`]).
    Fixed,
    /// Alternate `on_s`/`off_s` availability windows ([`MeasurementScheduler::DutyCycle`]).
    Duty,
}

/// GNSS scheduler arguments for CLI
#[derive(Clone, Debug)]
#[cfg_attr(feature = "clap", derive(Args))]
pub struct SchedulerArgs {
    /// Scheduler kind: passthrough | fixed | duty
    #[cfg_attr(feature = "clap", arg(long, value_enum, default_value_t = SchedKind::Passthrough))]
    pub sched: SchedKind,
    /// Fixed-interval seconds (sched=fixed)
    #[cfg_attr(feature = "clap", arg(long, default_value_t = 1.0))]
    pub interval_s: f64,
    /// Initial phase seconds (sched=fixed)
    #[cfg_attr(feature = "clap", arg(long, default_value_t = 0.0))]
    pub phase_s: f64,
    /// Duty-cycle ON seconds (sched=duty)
    #[cfg_attr(feature = "clap", arg(long, default_value_t = 10.0))]
    pub on_s: f64,
    /// Duty-cycle OFF seconds (sched=duty)
    #[cfg_attr(feature = "clap", arg(long, default_value_t = 10.0))]
    pub off_s: f64,
    /// Duty-cycle start phase seconds (sched=duty)
    #[cfg_attr(feature = "clap", arg(long, default_value_t = 0.0))]
    pub duty_phase_s: f64,
}

/// Fault configuration kind
#[derive(Copy, Clone, Debug)]
#[cfg_attr(feature = "clap", derive(ValueEnum))]
pub enum FaultKind {
    /// No corruption; fixes reach the filter unchanged ([`GnssFaultModel::None`]).
    None,
    /// AR(1)-correlated position and velocity error plus an inflated measurement covariance
    /// ([`GnssFaultModel::Degraded`]).
    Degraded,
    /// Slowly drifting north/east offset, a soft spoof ([`GnssFaultModel::SlowBias`]).
    Slowbias,
    /// Constant north/east offset applied over a fixed window, a hard spoof
    /// ([`GnssFaultModel::Hijack`]).
    Hijack,
}

/// GNSS fault model arguments for CLI
#[derive(Clone, Debug)]
#[cfg_attr(feature = "clap", derive(Args))]
pub struct FaultArgs {
    /// Fault kind: none | degraded | slowbias | hijack
    #[cfg_attr(feature = "clap", arg(long, value_enum, default_value_t = FaultKind::None))]
    pub fault: FaultKind,
    /// Degraded: AR(1) correlation coefficient for the position error (0 to 1)
    #[cfg_attr(feature = "clap", arg(long, default_value_t = 0.99))]
    pub rho_pos: f64,
    /// Degraded: AR(1) innovation standard deviation for the position error, in meters
    #[cfg_attr(feature = "clap", arg(long, default_value_t = 3.0))]
    pub sigma_pos_m: f64,
    /// Degraded: AR(1) correlation coefficient for the velocity error (0 to 1)
    #[cfg_attr(feature = "clap", arg(long, default_value_t = 0.95))]
    pub rho_vel: f64,
    /// Degraded: AR(1) innovation standard deviation for the velocity error, in m/s
    #[cfg_attr(feature = "clap", arg(long, default_value_t = 0.3))]
    pub sigma_vel_mps: f64,
    /// Degraded: factor applied to the advertised 1-sigma measurement standard deviations
    /// (horizontal position in metres and velocity in m/s), NOT to the covariance.
    ///
    /// The scaled standard deviations reach the filter as
    /// `GPSPositionAndVelocityMeasurement::horizontal_noise_std` and `velocity_noise_std`, and
    /// the measurement model squares them to build R. **R is therefore inflated by `r_scale`
    /// squared**: the default 5.0 multiplies R by 25, not by 5.
    #[cfg_attr(feature = "clap", arg(long, default_value_t = 5.0))]
    pub r_scale: f64,
    /// Slow bias: northward drift rate of the injected offset, in m/s
    #[cfg_attr(feature = "clap", arg(long, default_value_t = 0.02))]
    pub drift_n_mps: f64,
    /// Slow bias: eastward drift rate of the injected offset, in m/s
    #[cfg_attr(feature = "clap", arg(long, default_value_t = 0.0))]
    pub drift_e_mps: f64,
    /// Slow bias: random-walk PSD of the drifting offset, in m^2/s.
    ///
    /// The offset itself is in metres and the driving noise adds variance `q_bias * dt` to it on
    /// every step, so the PSD carries units of metres squared per second -- not m^2/s^3, which
    /// would be the PSD of a random walk driving a velocity.
    #[cfg_attr(feature = "clap", arg(long, default_value_t = 1e-6))]
    pub q_bias: f64,
    /// Slow bias: rate at which the drift direction rotates, in rad/s (0 keeps it fixed)
    #[cfg_attr(feature = "clap", arg(long, default_value_t = 0.0))]
    pub rotate_omega_rps: f64,
    /// Hijack: constant northward offset applied during the window, in meters
    #[cfg_attr(feature = "clap", arg(long, default_value_t = 50.0))]
    pub hijack_offset_n_m: f64,
    /// Hijack: constant eastward offset applied during the window, in meters
    #[cfg_attr(feature = "clap", arg(long, default_value_t = 0.0))]
    pub hijack_offset_e_m: f64,
    /// Hijack: start of the spoofing window, in seconds from the start of the run
    #[cfg_attr(feature = "clap", arg(long, default_value_t = 120.0))]
    pub hijack_start_s: f64,
    /// Hijack: length of the spoofing window, in seconds
    #[cfg_attr(feature = "clap", arg(long, default_value_t = 60.0))]
    pub hijack_duration_s: f64,
}

/// Build GNSS scheduler from CLI arguments
pub const fn build_scheduler(a: &SchedulerArgs) -> MeasurementScheduler {
    match a.sched {
        SchedKind::Passthrough => MeasurementScheduler::PassThrough,
        SchedKind::Fixed => MeasurementScheduler::FixedInterval {
            interval_s: a.interval_s,
            phase_s: a.phase_s,
        },
        SchedKind::Duty => MeasurementScheduler::DutyCycle {
            on_s: a.on_s,
            off_s: a.off_s,
            start_phase_s: a.duty_phase_s,
        },
    }
}

/// Build GNSS fault model from CLI arguments
pub const fn build_fault(a: &FaultArgs) -> GnssFaultModel {
    match a.fault {
        FaultKind::None => GnssFaultModel::None,
        FaultKind::Degraded => GnssFaultModel::Degraded {
            rho_pos: a.rho_pos,
            sigma_pos_m: a.sigma_pos_m,
            rho_vel: a.rho_vel,
            sigma_vel_mps: a.sigma_vel_mps,
            r_scale: a.r_scale,
        },
        FaultKind::Slowbias => GnssFaultModel::SlowBias {
            drift_n_mps: a.drift_n_mps,
            drift_e_mps: a.drift_e_mps,
            q_bias: a.q_bias,
            rotate_omega_rps: a.rotate_omega_rps,
        },
        FaultKind::Hijack => GnssFaultModel::Hijack {
            offset_n_m: a.hijack_offset_n_m,
            offset_e_m: a.hijack_offset_e_m,
            start_s: a.hijack_start_s,
            duration_s: a.hijack_duration_s,
        },
    }
}

//================= Unified Simulation Configuration =====================================

/// Simulation mode selection
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "clap", derive(ValueEnum))]
#[serde(rename_all = "kebab-case")]
pub enum SimulationMode {
    /// Dead reckoning with no corrections
    DeadReckoning,
    /// Open-loop feed-forward INS
    OpenLoop,
    /// Closed-loop with Kalman filter corrections from GNSS/other sensors
    ClosedLoop,
    /// Particle filter based navigation
    ParticleFilter,
    /// Synthetic trajectory generation from initial kinematic state
    Synthetic,
}

/// Filter type for closed-loop mode
///
/// The default is [`FilterType::Eskf`] (#258). The 15-state error-state filter estimates
/// accelerometer and gyroscope biases online and carries attitude as a quaternion corrected
/// multiplicatively, so it neither accumulates the turn-on bias the 9-state filters have no
/// way to observe nor linearizes about Euler angles. The UKF and EKF remain selectable and
/// are unchanged by this default.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "clap", derive(ValueEnum))]
#[serde(rename_all = "kebab-case")]
#[derive(Default)]
pub enum FilterType {
    /// Error-State Kalman Filter (ESKF) with multiplicative attitude error. The default.
    #[default]
    Eskf,
    /// Unscented Kalman Filter
    Ukf,
    /// Extended Kalman Filter
    Ekf,
}

/// Particle filter type selection.
///
/// One variant, deliberately. This enum previously also advertised `Standard` (all states
/// as particles) and `Velocity` (position-only particles with externally supplied
/// velocities); neither was ever implemented, and `Standard` was the *default*, so
/// `strapdown-sim particle-filter` failed on its own defaults with "Only
/// Rao-Blackwellized particle filter is implemented in this mode". They were removed in
/// queue 5 (#259) rather than implemented: [`particle`](crate::particle) is documented as
/// a template-style module of building blocks -- resampling strategies and the
/// [`Particle`](crate::particle::Particle) trait -- for users assembling their own filter,
/// not as a filter itself.
///
/// The enum is kept rather than collapsed away so that adding a second concrete filter
/// stays a non-breaking change to the config schema.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "clap", derive(ValueEnum))]
#[serde(rename_all = "kebab-case")]
#[derive(Default)]
pub enum ParticleFilterType {
    /// Rao-Blackwellized particle filter (position as particles, velocity/attitude/extra
    /// states as per-particle Kalman filters). The default, and currently the only variant.
    #[default]
    RaoBlackwellized,
}

/// Closed-loop specific configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ClosedLoopConfig {
    /// Filter type; defaults to the 15-state ESKF.
    #[serde(default)]
    pub filter: FilterType,
    /// UKF alpha parameter (spread of sigma points)
    #[serde(default = "default_ukf_alpha")]
    pub ukf_alpha: f64,
    /// UKF beta parameter (prior knowledge of distribution, 2.0 is optimal for Gaussian)
    #[serde(default = "default_ukf_beta")]
    pub ukf_beta: f64,
    /// UKF kappa parameter (secondary spread control)
    #[serde(default = "default_ukf_kappa")]
    pub ukf_kappa: f64,
    /// Innovation gate applied to every measurement update.
    ///
    /// `None` -- the default -- accepts every measurement, which is the behaviour
    /// every run of this crate has had up to now. Gating is opt-in rather than on by
    /// default because switching it on changes the trajectory of every existing
    /// scenario, and that is a decision to make against the ground-truth validation
    /// suite rather than as a side effect of adding the capability.
    ///
    /// Deserializes from either form:
    /// ```yaml
    /// innovation_gate: { chi_squared: { confidence: 0.999 } }
    /// innovation_gate: { fixed: { threshold: 25.0 } }
    /// ```
    #[serde(default)]
    pub innovation_gate: Option<InnovationGate>,
    /// How the filter recovers from a measurement the gate rejected.
    ///
    /// Only consulted when [`Self::innovation_gate`] installs a gate, and defaulted
    /// rather than optional because a gate without a way back out is the defect in #340,
    /// not a configuration: the filter that rejects one fix keeps drifting while the
    /// covariance it judges the next fix against does not grow, so the rejection is
    /// self-reinforcing. Write it out only to tune it:
    ///
    /// ```yaml
    /// gate_recovery: { rejection_inflation: 4.0, forced_update_after: 3 }
    /// ```
    ///
    /// Either field may be omitted and keeps its default. `rejection_inflation: 1.0` with
    /// `forced_update_after: null` is both mechanisms off, i.e. the pre-#340 behaviour.
    #[serde(default)]
    pub gate_recovery: GateRecovery,
    /// Estimate a barometric altitude bias as an extra filter state (#372).
    ///
    /// A barometer's reference pressure drifts, and a filter that models the reading as
    /// unbiased pushes that drift into altitude. Measured on `core/tests/test_data.csv`,
    /// switching this on takes 3-sigma vertical containment from about 0.40 to 0.84 against
    /// an ideal of 0.9973, removes a systematic 0.4 m offset and improves vertical RMSE by
    /// 45%, in all three Kalman filters.
    ///
    /// **`true` by default as of the 1.0 API freeze**, which is the change the previous
    /// default's note promised. The state was held off by default when it landed so that it
    /// arrived separately from the decision to switch it on; the measurements above are that
    /// decision, and no measured case got worse -- position NEES moves *toward* its ideal of
    /// 3.0 on synthetic data, where the truth is exact.
    ///
    /// Turning it on also tells the barometer model which state to read; see
    /// [`AidingConfig::baro_bias_index`](crate::messages::AidingConfig),
    /// which `strapdown-sim` derives from this rather than making it a second thing to set.
    /// A library caller building a filter directly must set that index themselves: without it
    /// the barometer observes nothing and the extra state sits at its prior, which is why the
    /// `UkfConfig`/`EkfConfig`/`EskfConfig` defaults stay `false` -- flipping those would hand
    /// a direct caller a sixteenth state that nothing reads.
    ///
    /// Not available on the geophysical path, whose extra states are map biases.
    #[serde(default)]
    pub estimate_baro_bias: bool,
}

impl Default for ClosedLoopConfig {
    fn default() -> Self {
        Self {
            filter: FilterType::default(),
            ukf_alpha: default_ukf_alpha(),
            ukf_beta: default_ukf_beta(),
            ukf_kappa: default_ukf_kappa(),
            innovation_gate: None,
            gate_recovery: GateRecovery::default(),
            estimate_baro_bias: true,
        }
    }
}

/// Particle filter configuration (RBPF defaults).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ParticleFilterConfig {
    /// Number of particles in the filter.
    #[serde(default = "default_num_particles")]
    pub num_particles: usize,
    /// Initial position standard deviation as a ground extent in metres,
    /// [`north_m`, `east_m`, `up_m`]. Converted to the filter's radian position units
    /// at the starting latitude; see [`crate::rbpf::RbpfConfig::position_init_std_m`].
    #[serde(default = "default_position_init_std_m")]
    pub position_init_std_m: Vec<f64>,
    /// Initial velocity standard deviation (m/s).
    #[serde(default = "default_velocity_init_std_mps")]
    pub velocity_init_std_mps: f64,
    /// Initial attitude standard deviation (rad).
    #[serde(default = "default_attitude_init_std_rad")]
    pub attitude_init_std_rad: f64,
    /// Position random-walk rate as [`north`, `east`, `up`] in m/sqrt(s) -- the key name
    /// keeps its `_m` for compatibility with existing configuration files, and the value
    /// is unchanged at a 1 s step. The per-step standard deviation is this times
    /// `sqrt(dt)`; see [`crate::rbpf::RbpfConfig::position_process_noise_std_m`].
    #[serde(default = "default_position_process_noise_std_m")]
    pub position_process_noise_std_m: Vec<f64>,
    /// Velocity random-walk rate in m/s per sqrt(s) -- the key name keeps its `_mps` for
    /// compatibility with existing configuration files, and the value is unchanged at a 1 s
    /// step. The per-step standard deviation is this times `sqrt(dt)`; see
    /// [`crate::rbpf::RbpfConfig::velocity_process_noise_std_mps`].
    #[serde(default = "default_velocity_process_noise_std_mps")]
    pub velocity_process_noise_std_mps: f64,
    /// Attitude random-walk rate in rad per sqrt(s) -- the key name keeps its `_rad` for
    /// compatibility with existing configuration files, and the value is unchanged at a 1 s
    /// step. The per-step standard deviation is this times `sqrt(dt)`; see
    /// [`crate::rbpf::RbpfConfig::attitude_process_noise_std_rad`].
    #[serde(default = "default_attitude_process_noise_std_rad")]
    pub attitude_process_noise_std_rad: f64,
    /// Initial standard deviation for geophysical bias states.
    #[serde(default = "default_geo_bias_init_std")]
    pub geo_bias_init_std: f64,
    /// Random-walk rate for the geophysical bias states, in the bias's own units per
    /// sqrt(s): the variance it accumulates is `std^2 * elapsed_seconds`, independent of
    /// the log's sample rate. Supplies
    /// [`crate::rbpf::RbpfConfig::extra_state_process_noise_std`].
    #[serde(default = "default_geo_bias_process_noise_std")]
    pub geo_bias_process_noise_std: f64,
    /// Apply zero-vertical-velocity pseudo-measurement.
    #[serde(default = "default_zero_vertical_velocity")]
    pub zero_vertical_velocity: bool,
    /// Standard deviation for zero-vertical-velocity pseudo-measurement (m/s).
    #[serde(default = "default_zero_vertical_velocity_std_mps")]
    pub zero_vertical_velocity_std_mps: f64,
}

const fn default_zero_vertical_velocity() -> bool {
    true
}

const fn default_zero_vertical_velocity_std_mps() -> f64 {
    0.1
}

const fn default_ukf_alpha() -> f64 {
    1e-3
}

const fn default_ukf_beta() -> f64 {
    2.0
}

const fn default_ukf_kappa() -> f64 {
    0.0
}

const fn default_num_particles() -> usize {
    100
}

fn default_position_init_std_m() -> Vec<f64> {
    vec![10.0, 10.0, 5.0]
}

const fn default_velocity_init_std_mps() -> f64 {
    1.0
}

const fn default_attitude_init_std_rad() -> f64 {
    0.1
}

fn default_position_process_noise_std_m() -> Vec<f64> {
    vec![1.0, 1.0, 1.0]
}

const fn default_velocity_process_noise_std_mps() -> f64 {
    1e-3
}

const fn default_attitude_process_noise_std_rad() -> f64 {
    0.01
}

const fn default_geo_bias_init_std() -> f64 {
    1.0
}

const fn default_geo_bias_process_noise_std() -> f64 {
    1e-3
}

impl Default for ParticleFilterConfig {
    fn default() -> Self {
        Self {
            num_particles: default_num_particles(),
            position_init_std_m: default_position_init_std_m(),
            velocity_init_std_mps: default_velocity_init_std_mps(),
            attitude_init_std_rad: default_attitude_init_std_rad(),
            position_process_noise_std_m: default_position_process_noise_std_m(),
            velocity_process_noise_std_mps: default_velocity_process_noise_std_mps(),
            attitude_process_noise_std_rad: default_attitude_process_noise_std_rad(),
            geo_bias_init_std: default_geo_bias_init_std(),
            geo_bias_process_noise_std: default_geo_bias_process_noise_std(),
            zero_vertical_velocity: default_zero_vertical_velocity(),
            zero_vertical_velocity_std_mps: default_zero_vertical_velocity_std_mps(),
        }
    }
}
/// Log level options for simulation logging.
///
/// This enum is serialized/deserialized as lowercase strings to match existing
/// configuration files (e.g., `"info"`, `"debug"`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[cfg_attr(feature = "clap", derive(ValueEnum))]
pub enum LogLevel {
    /// Emit nothing at all (`"off"`).
    Off,
    /// Errors only (`"error"`).
    Error,
    /// Errors and warnings (`"warn"`).
    Warn,
    /// Progress and configuration messages and above (`"info"`); the default.
    Info,
    /// Per-step filter detail and above (`"debug"`).
    Debug,
    /// Everything, including the noisiest tracing (`"trace"`).
    Trace,
}

impl LogLevel {
    /// Convert `LogLevel` to string representation
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Error => "error",
            Self::Warn => "warn",
            Self::Info => "info",
            Self::Debug => "debug",
            Self::Trace => "trace",
        }
    }
}

/// Logging configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LoggingConfig {
    /// Log level (off, error, warn, info, debug, trace)
    #[serde(default = "default_log_level")]
    pub level: LogLevel,
    /// Optional log file path (if not specified, logs to stderr)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file: Option<String>,
}

const fn default_log_level() -> LogLevel {
    LogLevel::Info
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: default_log_level(),
            file: None,
        }
    }
}

/// Unified simulation configuration supporting all modes
#[derive(Clone, Debug, Serialize, Deserialize)]
#[non_exhaustive]
pub struct SimulationConfig {
    /// Input CSV file path (relative or absolute)
    #[serde(default = "default_input")]
    pub input: String,
    /// Output CSV file path (relative or absolute)
    #[serde(default = "default_output")]
    pub output: String,
    /// Simulation mode
    pub mode: SimulationMode,
    /// Random number generator seed
    #[serde(default = "default_seed")]
    pub seed: u64,
    /// Local-level frame the input records are expressed in: `false` (the default) is NED,
    /// `true` is ENU.
    ///
    /// Sensor Logger exports are ENU; `strapdown-sim syn` output and the rest of the library
    /// are NED. `serde(default)` is `false`, so every config file written before this field
    /// existed keeps parsing -- and any such file describing a Sensor Logger recording now
    /// fails loudly in `check_declared_frame` rather than mechanizing at 2 g (#296).
    #[serde(default)]
    pub is_enu: bool,
    /// Run simulations in parallel when processing multiple files
    #[serde(default)]
    pub parallel: bool,
    /// Generate performance plot comparing navigation output to GPS measurements
    #[serde(default)]
    pub generate_plot: bool,
    /// Execution time limits (wall-clock and no-progress)
    #[serde(default)]
    pub execution_limits: ExecutionLimits,
    /// Logging configuration
    #[serde(default)]
    pub logging: LoggingConfig,
    /// Closed-loop specific settings (only used if mode is `ClosedLoop`)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub closed_loop: Option<ClosedLoopConfig>,
    /// Particle filter settings (only used if mode is `ParticleFilter`)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub particle_filter: Option<ParticleFilterConfig>,
    /// Geophysical measurement configuration (optional, requires --features geonav)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub geophysical: Option<GeophysicalConfig>,
    /// Aiding-measurement configuration: the GNSS, barometer and magnetometer schedules, the
    /// GNSS fault model, and the barometer's noise and bias-state index.
    ///
    /// `#[serde(alias = "gnss_degradation")]` keeps every configuration file written before
    /// this field was renamed parsing unchanged. The old name described the type when it
    /// scheduled GNSS alone; it now carries `baro_scheduler`, `magnetometer_scheduler`,
    /// `baro_noise_std_m` and `baro_bias_index` as well, and only GNSS has a fault model at
    /// all. The alias is load-bearing -- the fifteen recipes under `conf/` all spell the old
    /// name -- so do not drop it.
    #[serde(default, alias = "gnss_degradation")]
    pub aiding: crate::messages::AidingConfig,
    /// Synthetic trajectory configuration (only used if mode is Synthetic)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub synthetic: Option<SyntheticConfig>,
}

fn default_input() -> String {
    "input.csv".to_string()
}

fn default_output() -> String {
    "output.csv".to_string()
}

const fn default_seed() -> u64 {
    42
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            input: "input.csv".to_string(),
            output: "output.csv".to_string(),
            mode: SimulationMode::ClosedLoop,
            seed: default_seed(),
            is_enu: false,
            parallel: false,
            generate_plot: false,
            execution_limits: ExecutionLimits::default(),
            logging: LoggingConfig::default(),
            closed_loop: Some(ClosedLoopConfig::default()),
            particle_filter: None,
            geophysical: None,
            aiding: crate::messages::AidingConfig::default(),
            synthetic: None,
        }
    }
}

impl SimulationConfig {
    /// Write the configuration to a JSON file (pretty-printed)
    /// # Errors
    /// If the file cannot be created or written, or the records cannot be
    /// serialised as JSON.
    pub fn to_json<P: AsRef<Path>>(&self, path: P) -> io::Result<()> {
        let file = std::fs::File::create(path)?;
        serde_json::to_writer_pretty(file, self).map_err(io::Error::other)
    }

    /// Read the configuration from a JSON file
    /// # Errors
    /// If the file cannot be read, or its contents are not valid JSON.
    pub fn from_json<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let file = std::fs::File::open(path)?;
        serde_json::from_reader(file).map_err(io::Error::other)
    }

    /// Write the configuration as YAML
    /// # Errors
    /// If the file cannot be created or written, or the records cannot be
    /// serialised as YAML.
    pub fn to_yaml<P: AsRef<Path>>(&self, path: P) -> io::Result<()> {
        let mut file = std::fs::File::create(path)?;
        let s = serde_yaml::to_string(self).map_err(io::Error::other)?;
        file.write_all(s.as_bytes())
    }

    /// Read the configuration from YAML
    /// # Errors
    /// If the file cannot be read, or its contents are not valid YAML.
    pub fn from_yaml<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let file = std::fs::File::open(path)?;
        serde_yaml::from_reader(file).map_err(io::Error::other)
    }

    /// Write the configuration as TOML
    /// # Errors
    /// If the file cannot be created or written, or the records cannot be
    /// serialised as TOML.
    pub fn to_toml<P: AsRef<Path>>(&self, path: P) -> io::Result<()> {
        let mut file = std::fs::File::create(path)?;
        let s = toml::to_string_pretty(self).map_err(io::Error::other)?;
        file.write_all(s.as_bytes())
    }

    /// Read the configuration from TOML
    /// # Errors
    /// If the file cannot be read, or its contents are not valid TOML.
    pub fn from_toml<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let mut s = String::new();
        let mut file = std::fs::File::open(path)?;
        file.read_to_string(&mut s)?;
        toml::from_str(&s).map_err(io::Error::other)
    }

    /// Generic write: choose format by file extension (.json/.yaml/.yml/.toml)
    /// # Errors
    /// If the file cannot be created or written, or the records cannot be
    /// serialised as the inferred format.
    pub fn to_file<P: AsRef<Path>>(&self, path: P) -> io::Result<()> {
        let p = path.as_ref();
        let ext = p
            .extension()
            .and_then(|s| s.to_str())
            .map(str::to_lowercase);
        match ext.as_deref() {
            Some("json") => self.to_json(p),
            Some("yaml" | "yml") => self.to_yaml(p),
            Some("toml") => self.to_toml(p),
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "unsupported file extension (expected .json, .yaml, .yml, or .toml)",
            )),
        }
    }

    /// Generic read: choose format by file extension (.json/.yaml/.yml/.toml)
    /// # Errors
    /// If the file cannot be read, or its contents are not valid the inferred format.
    pub fn from_file<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let p = path.as_ref();
        let ext = p
            .extension()
            .and_then(|s| s.to_str())
            .map(str::to_lowercase);
        match ext.as_deref() {
            Some("json") => Self::from_json(p),
            Some("yaml" | "yml") => Self::from_yaml(p),
            Some("toml") => Self::from_toml(p),
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "unsupported file extension (expected .json, .yaml, .yml, or .toml)",
            )),
        }
    }
}

/// Geophysical measurement type configuration
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "clap", derive(ValueEnum))]
#[serde(rename_all = "lowercase")]
#[derive(Default)]
pub enum GeoMeasurementType {
    /// Gravity anomaly measurements
    #[default]
    Gravity,
    /// Magnetic anomaly measurements
    Magnetic,
}

/// Geophysical map resolution configuration
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "clap", derive(ValueEnum))]
#[serde(rename_all = "snake_case")]
#[derive(Default)]
pub enum GeoResolution {
    /// 1 degree resolution
    OneDegree,
    /// 30 arcminute resolution
    ThirtyMinutes,
    /// 20 arcminute resolution
    TwentyMinutes,
    /// 15 arcminute resolution
    FifteenMinutes,
    /// 10 arcminute resolution
    TenMinutes,
    /// 6 arcminute resolution
    SixMinutes,
    /// 5 arcminute resolution
    FiveMinutes,
    /// 4 arcminute resolution
    FourMinutes,
    /// 3 arcminute resolution
    ThreeMinutes,
    /// 2 arcminute resolution
    TwoMinutes,
    /// 1 arcminute resolution
    #[default]
    OneMinute,
    /// 30 arcsecond resolution
    ThirtySeconds,
    /// 15 arcsecond resolution
    FifteenSeconds,
    /// 3 arcsecond resolution
    ThreeSeconds,
    /// 1 arcsecond resolution
    OneSecond,
}

/// Geophysical measurement configuration for geonav simulations
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[non_exhaustive]
pub struct GeophysicalConfig {
    // Gravity measurement configuration (all optional)
    /// Gravity map resolution (None = gravity not used)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gravity_resolution: Option<GeoResolution>,

    /// Gravity measurement bias (mGal)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gravity_bias: Option<f64>,

    /// Gravity measurement noise std dev (mGal)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gravity_noise_std: Option<f64>,

    /// Gravity map file path (auto-detected if None)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gravity_map_file: Option<String>,

    // Magnetic measurement configuration (all optional)
    /// Magnetic map resolution (None = magnetic not used)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub magnetic_resolution: Option<GeoResolution>,

    /// Magnetic measurement bias (nT)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub magnetic_bias: Option<f64>,

    /// Magnetic measurement noise std dev (nT)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub magnetic_noise_std: Option<f64>,

    /// Magnetic map file path (auto-detected if None)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub magnetic_map_file: Option<String>,

    // Common configuration
    /// Interval in **seconds between** geophysical measurements, applying to both measurement
    /// types when both are enabled.
    ///
    /// A period, not a frequency, despite the name it carried until the v1.0 freeze:
    /// `geonav`'s scheduler adds it to the time of the last measurement
    /// (`next_geo_time += interval`), so a larger value means *fewer* measurements.
    /// `#[serde(alias = "geo_frequency_s")]` keeps every configuration file written under the
    /// old name parsing -- the nine recipes under `conf/` among them -- so do not drop it.
    #[serde(
        default,
        alias = "geo_frequency_s",
        skip_serializing_if = "Option::is_none"
    )]
    pub geo_interval_s: Option<f64>,
}

// ==================== Synthetic Trajectory Generation ====================

/// Initial kinematic state for synthetic trajectory generation.
///
/// Defines the starting position, velocity, attitude, and angular velocity
/// for a synthetic trajectory. The trajectory maintains constant nav-frame
/// velocity and constant body-frame angular velocity (zero linear and angular
/// acceleration).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SyntheticInitialState {
    /// Starting latitude in degrees (WGS84)
    pub latitude_deg: f64,
    /// Starting longitude in degrees (WGS84)
    pub longitude_deg: f64,
    /// Starting altitude in meters (WGS84)
    pub altitude_m: f64,
    /// Initial northward velocity in m/s (NED frame)
    #[serde(default)]
    pub velocity_north_mps: f64,
    /// Initial eastward velocity in m/s (NED frame)
    #[serde(default)]
    pub velocity_east_mps: f64,
    /// Initial downward velocity in m/s (positive down in NED, positive up in ENU)
    #[serde(default)]
    pub velocity_down_mps: f64,
    /// Initial roll angle in degrees
    #[serde(default)]
    pub roll_deg: f64,
    /// Initial pitch angle in degrees
    #[serde(default)]
    pub pitch_deg: f64,
    /// Initial yaw (heading) angle in degrees
    #[serde(default)]
    pub yaw_deg: f64,
    /// Constant body-frame roll rate in degrees/s (angular velocity about x-axis)
    #[serde(default)]
    pub angular_velocity_x_dps: f64,
    /// Constant body-frame pitch rate in degrees/s (angular velocity about y-axis)
    #[serde(default)]
    pub angular_velocity_y_dps: f64,
    /// Constant body-frame yaw rate in degrees/s (angular velocity about z-axis)
    #[serde(default)]
    pub angular_velocity_z_dps: f64,
    /// Coordinate frame: true = ENU, false = NED (default)
    #[serde(default)]
    pub is_enu: bool,
}

impl Default for SyntheticInitialState {
    fn default() -> Self {
        Self {
            latitude_deg: 0.0,
            longitude_deg: 0.0,
            altitude_m: 0.0,
            velocity_north_mps: 0.0,
            velocity_east_mps: 0.0,
            velocity_down_mps: 0.0,
            roll_deg: 0.0,
            pitch_deg: 0.0,
            yaw_deg: 0.0,
            angular_velocity_x_dps: 0.0,
            angular_velocity_y_dps: 0.0,
            angular_velocity_z_dps: 0.0,
            is_enu: false,
        }
    }
}

const fn default_sample_rate_hz() -> f64 {
    10.0
}

const fn default_gnss_horizontal_noise_m() -> f64 {
    2.5
}

const fn default_gnss_vertical_noise_m() -> f64 {
    5.0
}

const fn default_baro_noise_std_pa() -> f64 {
    50.0
}

/// Per-axis magnetometer noise for [`SyntheticConfig`], microtesla.
///
/// A consumer phone magnetometer's own noise floor. Deliberately far smaller than the ~17 deg
/// RMS heading error `core/tests/test_data.csv` exhibits: that recording's error is dominated
/// by hard and soft iron in the vehicle, not by the sensor, and modelling the sensor is what
/// this constant is for.
const fn default_mag_noise_std_ut() -> f64 {
    0.5
}

/// Hard-iron offset magnitude for [`SyntheticConfig`], microtesla. **Zero by default.**
///
/// See the field's own documentation: a hard-iron offset biases heading unobservably, which is
/// realistic and is exactly what should not be switched on by default while #371 is being
/// diagnosed against this trajectory's yaw column.
const fn default_mag_hard_iron_std_ut() -> f64 {
    0.0
}

/// Altitude bounds the World Magnetic Model is defined over, metres.
///
/// Mirrors the clamp `strapdown-geonav` applies for the same reason: outside this band
/// `GeomagneticField::new` refuses, and a synthetic trajectory has no business failing because
/// its altitude wandered past a model boundary.
///
/// Shared with [`crate::measurements::MagnetometerYawMeasurement::get_declination`], which
/// must clamp identically. It did not, and the asymmetry was a silent heading bias: this
/// function wrote a field carrying the declination at the clamped altitude while the consumer
/// passed the raw altitude to a model that refused it and fell back to **zero** declination,
/// so the declination was put in at one value and taken out at another. The two clamps are one
/// constant for that reason.
pub(crate) const WMM_MIN_ALTITUDE_M: f64 = -1000.0;
/// Upper altitude bound of the World Magnetic Model, metres. See [`WMM_MIN_ALTITUDE_M`].
pub(crate) const WMM_MAX_ALTITUDE_M: f64 = 850_000.0;

/// Nanotesla per microtesla. `TestDataRecord`'s magnetic channels are microtesla; the WMM
/// reports nanotesla.
const NANOTESLA_PER_MICROTESLA: f64 = 1000.0;

/// Offset that separates the magnetometer's noise stream from the trajectory's.
///
/// Any fixed non-zero value would do; the point is only that the two streams differ, so that
/// giving a trajectory a magnetic field does not reshuffle its IMU, GNSS and barometer noise.
/// See the comment at its use in [`generate_synthetic`].
const MAGNETOMETER_NOISE_STREAM_OFFSET: u64 = 0x4d41_474e_4554_4f00;

/// Offset separating the hard-iron draw from the magnetometer's per-sample noise stream.
///
/// A third stream rather than the head of the second: hard iron is drawn once and held, so
/// sharing `mag_rng` made switching it on shift every per-sample noise value after it. See
/// [`MAGNETOMETER_NOISE_STREAM_OFFSET`] for the same argument one level up.
const MAGNETOMETER_HARD_IRON_STREAM_OFFSET: u64 = 0x4841_5244_4952_4f4e;

/// The true magnetic field at a point, in the navigation frame, microtesla.
///
/// Returns the field in whichever frame `is_enu` selects, so the caller can rotate it into the
/// body frame with the same `attitude.matrix().transpose()` it uses for gravity.
///
/// # The ENU form is `[east, north, -down]`, and that is not obvious
///
/// It disagrees with [`crate::vertical_flip`], which documents this crate's navigation frame
/// as ordered `(north, east, vertical)` in both conventions, differing only by `diag(1, 1, -1)`.
/// By that reading the horizontal axes should not swap here. They do, because the consumer
/// this field exists to feed reads them swapped: `MagnetometerYawMeasurement`'s ENU branch
/// recovers the heading as `atan2(m_x, m_y)`, which returns the ENU yaw -- counter-clockwise
/// from east -- only when `m_x` is *east* and `m_y` is *north*. That branch is not a guess;
/// #305 measured it against real ENU data (`core/tests/test_data.csv`), where the
/// unconditional NED form had the ESKF converge on the reflected heading at 96.8 deg RMSE
/// against 16.4 deg once corrected.
///
/// So the two conventions genuinely differ between `StrapdownState` and this measurement, and
/// this function matches the measurement, because matching the other one silently returns
/// `pi/2 - psi` -- the 45-degree reflection #305 is named for.
/// `the_enu_magnetic_field_round_trips_through_its_own_measurement` pins it end to end, in
/// both frames, rather than leaving the next reader to re-derive which convention wins.
/// Reconciling the two is worth doing, but it is a change to the measurement and the state
/// together, not to this function alone.
///
/// `date` must be the record's own date. [`MagnetometerYawMeasurement`] looks the declination
/// up again at consumption time from the record's timestamp, and if the two disagree the
/// declination is removed at a different value than it was put in.
///
/// # Errors
///
/// [`StrapdownError::ExternalModel`] when the date is not a valid ordinal date or the model
/// declines the position. Altitude is clamped into the model's own band first, with a warning
/// when the clamp moves it more than a metre -- an out-of-band altitude is a property of the
/// trajectory rather than an error in it.
fn magnetic_field_nav_ut(
    latitude_deg: f64,
    longitude_deg: f64,
    altitude_m: f64,
    date: world_magnetic_model::time::Date,
    is_enu: bool,
) -> Result<Vector3<f64>, StrapdownError> {
    use world_magnetic_model::GeomagneticField;
    use world_magnetic_model::uom::si::angle::degree;
    use world_magnetic_model::uom::si::f32::{Angle, Length};
    use world_magnetic_model::uom::si::length::meter;
    use world_magnetic_model::uom::si::magnetic_flux_density::nanotesla;

    let clamped = altitude_m.clamp(WMM_MIN_ALTITUDE_M, WMM_MAX_ALTITUDE_M);
    if (altitude_m - clamped).abs() > 1.0 {
        log::warn!("altitude {altitude_m} m is outside the WMM band; clamped to {clamped} m");
    }
    let field = GeomagneticField::new(
        Length::new::<meter>(clamped as f32),
        Angle::new::<degree>(latitude_deg as f32),
        Angle::new::<degree>(longitude_deg as f32),
        date,
    )
    .map_err(|e| StrapdownError::ExternalModel {
        model: "WMM",
        detail: format!(
            "no field at lat={latitude_deg}, lon={longitude_deg}, alt={altitude_m} \
             (clamped {clamped}): {e:?}"
        ),
    })?;

    let north = f64::from(field.x().get::<nanotesla>()) / NANOTESLA_PER_MICROTESLA;
    let east = f64::from(field.y().get::<nanotesla>()) / NANOTESLA_PER_MICROTESLA;
    let down = f64::from(field.z().get::<nanotesla>()) / NANOTESLA_PER_MICROTESLA;

    Ok(if is_enu {
        Vector3::new(east, north, -down)
    } else {
        Vector3::new(north, east, down)
    })
}

/// Configuration for the `syn` (synthetic trajectory) command.
///
/// Generates synthetic IMU, GNSS, and barometric sensor data from a defined
/// initial kinematic state. The trajectory propagates at constant nav-frame
/// velocity with constant body angular velocity.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[non_exhaustive]
pub struct SyntheticConfig {
    /// Output CSV file path
    pub output: String,
    /// Initial kinematic state (position, velocity, attitude, angular velocity)
    #[serde(default)]
    pub initial_state: SyntheticInitialState,
    /// Trajectory duration in seconds
    pub duration_s: f64,
    /// IMU sample rate in Hz
    #[serde(default = "default_sample_rate_hz")]
    pub sample_rate_hz: f64,
    /// IMU quality grade (controls noise and bias levels)
    #[serde(default)]
    pub imu_quality: crate::IMUQuality,
    /// Random number generator seed for reproducibility
    #[serde(default = "default_seed")]
    pub seed: u64,
    /// If true, output 9-state kinematic truth (`NavigationResult` format).
    /// If false (default), output noisy sensor measurements (`TestDataRecord` format).
    #[serde(default)]
    pub no_noise: bool,
    /// GNSS horizontal position noise standard deviation in meters
    #[serde(default = "default_gnss_horizontal_noise_m")]
    pub gnss_horizontal_noise_m: f64,
    /// GNSS vertical position noise standard deviation in meters
    #[serde(default = "default_gnss_vertical_noise_m")]
    pub gnss_vertical_noise_m: f64,
    /// Barometric pressure noise standard deviation in Pascals
    #[serde(default = "default_baro_noise_std_pa")]
    pub baro_noise_std_pa: f64,
    /// Magnetometer noise standard deviation per axis, microtesla.
    ///
    /// See [`default_mag_noise_std_ut`]. Sensor noise only.
    #[serde(default = "default_mag_noise_std_ut")]
    pub mag_noise_std_ut: f64,
    /// Hard-iron offset magnitude, microtesla, drawn once per trajectory and held constant.
    ///
    /// **Zero by default, deliberately.** A hard-iron offset is a constant field added in the
    /// *body* frame, so it biases the computed heading in a way no filtering can observe --
    /// realistic, and exactly what you do not want switched on while measuring whether a
    /// filter's attitude machinery works. Set it non-zero to study the effect on purpose.
    ///
    /// Soft iron is deliberately not modelled: it is a 3x3 distortion rather than an offset,
    /// so it needs a matrix in the configuration rather than a scalar, and hard iron is the
    /// dominant term in practice.
    #[serde(default = "default_mag_hard_iron_std_ut")]
    pub mag_hard_iron_std_ut: f64,
}

/// Every field's serde default, as a `Default` impl.
///
/// It was the one configuration type in the crate without one, which stopped mattering the
/// moment [`SyntheticConfig`] became `#[non_exhaustive]` at the v1.0 freeze: a
/// `#[non_exhaustive]` struct cannot be built from a struct literal outside its own crate, so
/// without a `Default` (or another constructor) it would be **impossible to construct at all**
/// from `strapdown-sim`, from the gated benchmarks, or by any user of the library.
///
/// The two fields serde treats as required get the only sensible standalone values: an empty
/// `output` path, and the 300 s the CLI's `--duration-s` already defaults to.
impl Default for SyntheticConfig {
    fn default() -> Self {
        Self {
            output: String::new(),
            initial_state: SyntheticInitialState::default(),
            duration_s: DEFAULT_SYNTHETIC_DURATION_S,
            sample_rate_hz: default_sample_rate_hz(),
            imu_quality: crate::IMUQuality::default(),
            seed: default_seed(),
            no_noise: false,
            gnss_horizontal_noise_m: default_gnss_horizontal_noise_m(),
            gnss_vertical_noise_m: default_gnss_vertical_noise_m(),
            baro_noise_std_pa: default_baro_noise_std_pa(),
            mag_noise_std_ut: default_mag_noise_std_ut(),
            mag_hard_iron_std_ut: default_mag_hard_iron_std_ut(),
        }
    }
}

/// Default synthetic trajectory length, seconds -- the same value `strapdown-sim syn`'s
/// `--duration-s` flag carries, so the library and the CLI agree.
const DEFAULT_SYNTHETIC_DURATION_S: f64 = 300.0;

impl SyntheticConfig {
    /// Write config to a file, choosing format by extension (.json, .yaml, .yml, .toml)
    /// # Errors
    /// If the file cannot be created or written, or the records cannot be
    /// serialised as the inferred format.
    pub fn to_file<P: AsRef<Path>>(&self, path: P) -> io::Result<()> {
        let p = path.as_ref();
        let ext = p
            .extension()
            .and_then(|s| s.to_str())
            .map(str::to_lowercase);
        match ext.as_deref() {
            Some("json") => {
                let json = serde_json::to_string_pretty(self)
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
                std::fs::write(p, json)
            }
            Some("yaml" | "yml") => {
                let yaml = serde_yaml::to_string(self)
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
                std::fs::write(p, yaml)
            }
            Some("toml") => {
                let toml = toml::to_string_pretty(self)
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
                std::fs::write(p, toml)
            }
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "unsupported file extension (expected .json, .yaml, .yml, or .toml)",
            )),
        }
    }

    /// Read config from a file, choosing format by extension (.json, .yaml, .yml, .toml)
    /// # Errors
    /// If the file cannot be read, or its contents are not valid the inferred format.
    pub fn from_file<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let p = path.as_ref();
        let ext = p
            .extension()
            .and_then(|s| s.to_str())
            .map(str::to_lowercase);
        let contents = std::fs::read_to_string(p)?;
        match ext.as_deref() {
            Some("json") => serde_json::from_str(&contents)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e)),
            Some("yaml" | "yml") => serde_yaml::from_str(&contents)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e)),
            Some("toml") => {
                toml::from_str(&contents).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
            }
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "unsupported file extension (expected .json, .yaml, .yml, or .toml)",
            )),
        }
    }
}

/// Compute the perfect (noise-free) IMU measurements required to maintain constant
/// nav-frame velocity with a specified constant body angular velocity.
///
/// The gyroscope reading is the sum of the desired body rotation rate and the
/// Earth/transport rate compensation (so the filter sees the full inertial angular rate).
/// The accelerometer reading is the specific force needed to keep nav-frame velocity
/// constant (zero linear acceleration), recomputed from the current attitude at each step.
///
/// # Arguments
/// - `state` - Current strapdown navigation state
/// - `angular_velocity_body_rps` - Desired body rotation rate relative to nav frame,
///   expressed in body coordinates, in radians/s. Zero gives constant-attitude motion.
fn compute_perfect_imu(
    state: &crate::StrapdownState,
    angular_velocity_body_rps: Vector3<f64>,
) -> crate::IMUData {
    if state.is_enu {
        // Same defect and same remedy as `velocity_update` (#321): the Earth-rate and
        // Coriolis terms below are NED, and only gravity used to consult the frame. Reflect
        // the commanded body rate in, solve in NED, reflect the sample back -- `mechanize`
        // reads it in the state's own convention.
        let ned = compute_perfect_imu_ned(
            &state.to_ned(),
            crate::flip_vertical_rate(&angular_velocity_body_rps),
        );
        return crate::IMUData {
            accel: crate::flip_vertical(&ned.accel),
            gyro: crate::flip_vertical_rate(&ned.gyro),
        };
    }
    compute_perfect_imu_ned(state, angular_velocity_body_rps)
}
/// The NED half of [`compute_perfect_imu`], with no frame branch in it.
fn compute_perfect_imu_ned(
    state: &crate::StrapdownState,
    angular_velocity_body_rps: Vector3<f64>,
) -> crate::IMUData {
    use crate::earth;

    let lat_deg = state.latitude.to_degrees();
    let velocity = Vector3::new(
        state.velocity_north,
        state.velocity_east,
        state.velocity_vertical,
    );

    // Earth rotation and transport rate in nav frame
    let omega_ie_nav = earth::earth_rate_lla(&lat_deg);
    let omega_en_nav = earth::transport_rate(&lat_deg, &state.altitude, &velocity);

    // Gyro: desired body rotation + Earth/transport compensation rotated into body frame
    let c_nb = state.attitude.matrix();
    let earth_body = c_nb.transpose() * (omega_ie_nav + omega_en_nav);
    let gyro = angular_velocity_body_rps + earth_body;

    // Accelerometer: specific force to hold v_nav constant (v_dot = 0)
    // From velocity_update: v_dot = f_nav + g_nav - (Ω_en + 2*Ω_ie)*v = 0
    // → f_nav = (Ω_en + 2*Ω_ie)*v - g_nav
    let omega_en_skew = earth::vector_to_skew_symmetric(&omega_en_nav);
    let omega_ie_skew = earth::vector_to_skew_symmetric(&omega_ie_nav);
    let coriolis = (omega_en_skew + 2.0 * omega_ie_skew) * velocity;

    // Down-positive: this is the NED half, and the ENU sign is the caller's reflection
    // rather than a branch beside a Coriolis term that has none (#321).
    let g_nav = Vector3::new(0.0, 0.0, earth::gravity(&lat_deg, &state.altitude));

    let f_nav = coriolis - g_nav;
    let accel = c_nb.transpose() * f_nav;

    crate::IMUData { accel, gyro }
}

/// Generate a synthetic trajectory from an initial kinematic state.
///
/// Propagates a constant nav-frame velocity / constant body angular velocity trajectory,
/// computes perfect IMU measurements at each step via inverse mechanization, and
/// optionally degrades them with IMU-grade noise.
///
/// Returns two parallel vectors:
/// - `Vec<NavigationResult>` — truth 9-state trajectory (write this for `--no-noise`)
/// - `Vec<TestDataRecord>` — sensor measurements with GNSS + baro (write this for noisy mode)
///
/// # Arguments
/// - `config` - Synthetic trajectory configuration
/// - `rng` - Seeded random number generator for reproducibility
/// # Errors
/// Propagated from [`crate::mechanize`] while propagating the truth trajectory.
pub fn generate_synthetic(
    config: &SyntheticConfig,
    rng: &mut rand::rngs::StdRng,
) -> Result<(Vec<NavigationResult>, Vec<TestDataRecord>), StrapdownError> {
    use crate::earth;
    use rand::Rng;
    use rand_distr::Normal;

    let s = &config.initial_state;

    // Build initial strapdown state
    let attitude = nalgebra::Rotation3::from_euler_angles(
        s.roll_deg.to_radians(),
        s.pitch_deg.to_radians(),
        s.yaw_deg.to_radians(),
    );
    let mut state = crate::StrapdownState {
        latitude: s.latitude_deg.to_radians(),
        longitude: s.longitude_deg.to_radians(),
        altitude: s.altitude_m,
        velocity_north: s.velocity_north_mps,
        velocity_east: s.velocity_east_mps,
        velocity_vertical: s.velocity_down_mps,
        attitude,
        is_enu: s.is_enu,
    };

    // Constant body angular velocity (rad/s)
    let angular_velocity_body_rps = Vector3::new(
        s.angular_velocity_x_dps.to_radians(),
        s.angular_velocity_y_dps.to_radians(),
        s.angular_velocity_z_dps.to_radians(),
    );

    let dt = 1.0 / config.sample_rate_hz;
    let n_steps = (config.duration_s * config.sample_rate_hz).round() as usize;
    let initial_alt = state.altitude;

    // Draw per-trajectory bias offsets (constant for the full run)
    let accel_bias = {
        let sigma = config.imu_quality.accel_bias_instability_mps2();
        let dist = Normal::new(0.0_f64, sigma).unwrap_or_else(|_| {
            log::warn!(
                "accel bias instability {sigma} is not a usable standard deviation; using 1e-6"
            );
            crate::normal_with_std(1e-6)
        });
        Vector3::new(rng.sample(dist), rng.sample(dist), rng.sample(dist))
    };
    let gyro_bias = {
        // `gyro_bias_instability_rad_per_hour` is radians per *hour* despite its name, and this bias is
        // added straight to `perfect_imu.gyro`, which is radians per second. Without the
        // conversion a consumer-grade run injects ~1.745 rad/s -- 100 deg/s -- of constant
        // gyro bias. The accelerometer block above needs no equivalent conversion because
        // `accel_bias_instability_mps2` is already in the units its sample is added to.
        let sigma =
            config.imu_quality.gyro_bias_instability_rad_per_hour() / crate::SECONDS_PER_HOUR;
        let dist = Normal::new(0.0_f64, sigma).unwrap_or_else(|_| {
            log::warn!(
                "gyro bias instability {sigma} is not a usable standard deviation; using 1e-9"
            );
            crate::normal_with_std(1e-9)
        });
        Vector3::new(rng.sample(dist), rng.sample(dist), rng.sample(dist))
    };

    // Per-sample noise standard deviations (ARW/VRW scaled to sample rate)
    let accel_noise_sigma = config.imu_quality.accel_velocity_random_walk()
        * (config.sample_rate_hz / crate::SECONDS_PER_HOUR).sqrt();
    let gyro_noise_sigma = config.imu_quality.gyro_angle_random_walk()
        * (config.sample_rate_hz / crate::SECONDS_PER_HOUR).sqrt();

    let accel_noise_dist =
        Normal::new(0.0_f64, accel_noise_sigma).unwrap_or_else(|_| crate::normal_with_std(1e-6));
    let gyro_noise_dist =
        Normal::new(0.0_f64, gyro_noise_sigma).unwrap_or_else(|_| crate::normal_with_std(1e-9));
    let gnss_h_dist = Normal::new(0.0_f64, config.gnss_horizontal_noise_m)
        .unwrap_or_else(|_| crate::normal_with_std(1.0));
    let gnss_v_dist = Normal::new(0.0_f64, config.gnss_vertical_noise_m)
        .unwrap_or_else(|_| crate::normal_with_std(1.0));
    let baro_dist = Normal::new(0.0_f64, config.baro_noise_std_pa)
        .unwrap_or_else(|_| crate::normal_with_std(1.0));
    let mag_dist = Normal::new(0.0_f64, config.mag_noise_std_ut)
        .unwrap_or_else(|_| crate::normal_with_std(0.5));
    // The magnetometer draws from its **own** stream rather than from `rng`, so that adding a
    // magnetic field to a trajectory leaves every other channel's realization bit-identical.
    // Sharing `rng` would consume three draws per epoch and shift the IMU, GNSS and barometer
    // noise on every record after the first -- which showed up as `syn_dead_reckoning`, a
    // scenario that takes no measurements at all, moving when the magnetometer was added. A
    // separate stream makes a re-bless attributable: a number that moves, moved because of the
    // heading aid.
    let mut mag_rng = {
        use rand::SeedableRng as _;
        rand::rngs::StdRng::seed_from_u64(
            config.seed.wrapping_add(MAGNETOMETER_NOISE_STREAM_OFFSET),
        )
    };
    // Hard iron is a constant field in the *body* frame, so it is drawn once and held, like
    // the IMU biases above rather than like the per-sample noise. Zero by default -- see
    // `SyntheticConfig::mag_hard_iron_std_ut`.
    let mag_hard_iron = if config.mag_hard_iron_std_ut > 0.0 {
        let dist = Normal::new(0.0_f64, config.mag_hard_iron_std_ut)
            .unwrap_or_else(|_| crate::normal_with_std(1.0));
        // Its own substream, for the same reason the magnetometer has one at all. Drawn from
        // `mag_rng` this consumed three values ahead of the per-sample noise, so turning hard
        // iron on moved every later noise sample as well -- and a hard-iron experiment would
        // then vary the bias and the realization together, which is exactly the confound the
        // separate stream was introduced to remove.
        let mut hard_iron_rng = {
            use rand::SeedableRng as _;
            rand::rngs::StdRng::seed_from_u64(
                config
                    .seed
                    .wrapping_add(MAGNETOMETER_HARD_IRON_STREAM_OFFSET),
            )
        };
        Vector3::new(
            hard_iron_rng.sample(dist),
            hard_iron_rng.sample(dist),
            hard_iron_rng.sample(dist),
        )
    } else {
        Vector3::zeros()
    };

    // Fixed epoch start time for reproducibility
    let start_time: chrono::DateTime<Utc> = "2025-01-01T00:00:00Z"
        .parse()
        .unwrap_or_else(|_| Utc::now());

    let mut truth_records: Vec<NavigationResult> = Vec::with_capacity(n_steps);
    let mut sensor_records: Vec<TestDataRecord> = Vec::with_capacity(n_steps);

    for i in 0..n_steps {
        let timestamp = start_time + Duration::milliseconds((i as f64 * dt * 1000.0) as i64);
        let perfect_imu = compute_perfect_imu(&state, angular_velocity_body_rps);

        // Build truth NavigationResult from current state
        let (roll, pitch, yaw) = state.attitude.euler_angles();
        let truth = NavigationResult {
            latitude_longitude_cov: 0.0,
            latitude_altitude_cov: 0.0,
            longitude_altitude_cov: 0.0,
            timestamp,
            latitude: state.latitude.to_degrees(),
            longitude: state.longitude.to_degrees(),
            altitude: state.altitude,
            velocity_north: state.velocity_north,
            velocity_east: state.velocity_east,
            velocity_vertical: state.velocity_vertical,
            roll,
            pitch,
            yaw,
            acc_bias_x: 0.0,
            acc_bias_y: 0.0,
            acc_bias_z: 0.0,
            gyro_bias_x: 0.0,
            gyro_bias_y: 0.0,
            gyro_bias_z: 0.0,
            latitude_cov: 0.0,
            longitude_cov: 0.0,
            altitude_cov: 0.0,
            velocity_n_cov: 0.0,
            velocity_e_cov: 0.0,
            velocity_v_cov: 0.0,
            roll_cov: 0.0,
            pitch_cov: 0.0,
            yaw_cov: 0.0,
            acc_bias_x_cov: 0.0,
            acc_bias_y_cov: 0.0,
            acc_bias_z_cov: 0.0,
            gyro_bias_x_cov: 0.0,
            gyro_bias_y_cov: 0.0,
            gyro_bias_z_cov: 0.0,
            gravity_bias: None,
            gravity_bias_cov: None,
            magnetic_bias: None,
            magnetic_bias_cov: None,
            baro_bias: None,
            baro_bias_cov: None,
        };
        truth_records.push(truth);

        // Build sensor TestDataRecord
        let (out_acc, out_gyro, out_lat, out_lon, out_alt) = if config.no_noise {
            (
                perfect_imu.accel,
                perfect_imu.gyro,
                state.latitude.to_degrees(),
                state.longitude.to_degrees(),
                state.altitude,
            )
        } else {
            let noisy_accel = perfect_imu.accel
                + accel_bias
                + Vector3::new(
                    rng.sample(accel_noise_dist),
                    rng.sample(accel_noise_dist),
                    rng.sample(accel_noise_dist),
                );
            let noisy_gyro = perfect_imu.gyro
                + gyro_bias
                + Vector3::new(
                    rng.sample(gyro_noise_dist),
                    rng.sample(gyro_noise_dist),
                    rng.sample(gyro_noise_dist),
                );
            let r_e = earth::EQUATORIAL_RADIUS;
            let lat_noise_rad = rng.sample(gnss_h_dist) / r_e;
            let lon_noise_rad =
                rng.sample(gnss_h_dist) / ((r_e + state.altitude) * state.latitude.cos().max(1e-6));
            (
                noisy_accel,
                noisy_gyro,
                (state.latitude + lat_noise_rad).to_degrees(),
                (state.longitude + lon_noise_rad).to_degrees(),
                state.altitude + rng.sample(gnss_v_dist),
            )
        };

        let true_pressure =
            earth::expected_barometric_pressure(state.altitude, earth::SEA_LEVEL_PRESSURE);
        let out_pressure = if config.no_noise {
            true_pressure
        } else {
            true_pressure + rng.sample(baro_dist)
        };

        let speed = state.velocity_north.hypot(state.velocity_east);
        let bearing = state.velocity_east.atan2(state.velocity_north).to_degrees();
        let attitude_quaternion = nalgebra::UnitQuaternion::from_rotation_matrix(&state.attitude);

        // Gravity vector in body frame (NED: [0,0,g])
        let g = earth::gravity(&state.latitude.to_degrees(), &state.altitude);
        let g_nav = if state.is_enu {
            Vector3::new(0.0, 0.0, -g)
        } else {
            Vector3::new(0.0, 0.0, g)
        };
        let grav_body = state.attitude.matrix().transpose() * g_nav;

        // The magnetic field, by the same route as gravity: evaluate it in the navigation
        // frame and rotate it into the body frame through the truth attitude (#369).
        //
        // Until now these three channels were `f64::NAN`, which meant `build_event_stream`
        // emitted no magnetometer event and the synthetic scenarios had no heading aid at all
        // -- the UKF's 42.7 deg yaw column was being read on a trajectory where yaw was
        // observable only through the GNSS velocity fix. Before #328 they were `0.0`, which
        // was worse: a zero field tilt-compensates to zero and `atan2(0.0, 0.0)` is `+0.0`, so
        // every epoch was aided by a *fabricated* heading of 0 rad. This is the third and
        // correct answer.
        //
        // The date is the record's own, because `MagnetometerYawMeasurement` looks the
        // declination up again from `r1.time` when it consumes the record; a different date
        // here would remove the declination at a different value than it was put in.
        let mag_body = {
            let date = world_magnetic_model::time::Date::from_ordinal_date(
                timestamp.year(),
                timestamp.ordinal() as u16,
            )
            .map_err(|e| StrapdownError::ExternalModel {
                model: "WMM",
                detail: format!("synthetic epoch {timestamp} is not a valid ordinal date: {e}"),
            })?;
            let field_nav = magnetic_field_nav_ut(
                state.latitude.to_degrees(),
                state.longitude.to_degrees(),
                state.altitude,
                date,
                state.is_enu,
            )?;
            let clean = state.attitude.matrix().transpose() * field_nav + mag_hard_iron;
            if config.no_noise {
                clean
            } else {
                clean
                    + Vector3::new(
                        mag_rng.sample(mag_dist),
                        mag_rng.sample(mag_dist),
                        mag_rng.sample(mag_dist),
                    )
            }
        };

        sensor_records.push(TestDataRecord {
            time: timestamp,
            latitude: out_lat,
            longitude: out_lon,
            altitude: out_alt,
            speed,
            bearing,
            bearing_accuracy: config.gnss_horizontal_noise_m,
            speed_accuracy: config.gnss_horizontal_noise_m,
            vertical_accuracy: config.gnss_vertical_noise_m,
            horizontal_accuracy: config.gnss_horizontal_noise_m,
            // `TestDataRecord` documents roll/pitch/yaw as radians, and the quaternion is
            // what every consumer now reads the attitude from (`TestDataRecord::attitude`).
            // Both were wrong here: the angles were written in degrees, and the quaternion
            // was `(cos(yaw), 0, 0, 0)` -- not a unit quaternion, and it discarded roll,
            // pitch and the sign of yaw. A synthetic CSV fed back through `cl` or
            // `dead_reckoning` therefore started from an attitude unrelated to its own truth.
            roll,
            pitch,
            yaw,
            qw: attitude_quaternion.w,
            qx: attitude_quaternion.i,
            qy: attitude_quaternion.j,
            qz: attitude_quaternion.k,
            acc_x: out_acc[0],
            acc_y: out_acc[1],
            acc_z: out_acc[2],
            gyro_x: out_gyro[0],
            gyro_y: out_gyro[1],
            gyro_z: out_gyro[2],
            mag_x: mag_body[0],
            mag_y: mag_body[1],
            mag_z: mag_body[2],
            relative_altitude: out_alt - initial_alt,
            pressure: out_pressure,
            grav_x: grav_body[0],
            grav_y: grav_body[1],
            grav_z: grav_body[2],
        });

        // Propagate truth state with perfect IMU (noise-free)
        crate::mechanize(&mut state, &crate::ImuSample::from_rates(&perfect_imu, dt))?;
    }

    Ok((truth_records, sensor_records))
}

#[cfg(test)]
mod tests {

    /// The synthetic gyro bias is drawn in the accessor's own units, which are radians per
    /// *hour* despite the `_dph` name, and then added to a rad/s gyro reading. Without the
    /// conversion a stationary consumer-grade run carries ~1.745 rad/s -- 100 deg/s -- of
    /// constant bias, which is not a consumer IMU, it is a spinning one.
    ///
    /// Bounded against the grade's own figure rather than a fitted number: a stationary
    /// platform senses Earth rate (~7.3e-5 rad/s) plus a bias whose sigma is the grade's bias
    /// instability in rad/s, so a few sigma either side is the whole budget.
    #[test]
    fn synthetic_gyro_bias_is_per_second_not_per_hour() {
        use rand::SeedableRng;

        let quality = crate::IMUQuality::Consumer;
        let config = SyntheticConfig {
            output: String::new(),
            initial_state: SyntheticInitialState::default(),
            duration_s: 10.0,
            sample_rate_hz: 10.0,
            imu_quality: quality,
            seed: 7,
            // Must be false: `no_noise` emits `perfect_imu` directly and never reaches the
            // bias at all, so a test with it set cannot see this bug.
            no_noise: false,
            gnss_horizontal_noise_m: 1.0,
            gnss_vertical_noise_m: 1.0,
            baro_noise_std_pa: 1.0,
            mag_noise_std_ut: default_mag_noise_std_ut(),
            mag_hard_iron_std_ut: default_mag_hard_iron_std_ut(),
        };

        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let (_, records) = generate_synthetic(&config, &mut rng).expect("generation");

        // Both error terms are in play, so both are in the budget: a constant bias whose sigma
        // is the grade's bias instability in rad/s, and per-sample angle random walk scaled to
        // the sample rate. With the bug the bias alone is ~1.745 rad/s, which overruns this by
        // more than two orders of magnitude.
        let bias_sigma_rps = quality.gyro_bias_instability_rad_per_hour() / crate::SECONDS_PER_HOUR;
        let arw_sigma_rps = quality.gyro_angle_random_walk()
            * (config.sample_rate_hz / crate::SECONDS_PER_HOUR).sqrt();
        let earth_rate_rps = 7.292_115e-5;
        let budget = earth_rate_rps + 6.0 * (bias_sigma_rps + arw_sigma_rps);

        for record in &records {
            for (axis, rate) in [
                ("x", record.gyro_x),
                ("y", record.gyro_y),
                ("z", record.gyro_z),
            ] {
                assert!(
                    rate.abs() < budget,
                    "stationary gyro_{axis} = {rate} rad/s exceeds the {budget} rad/s budget \
                     for {quality:?}; a per-hour bias added to a per-second reading is 3600x \
                     too large"
                );
            }
        }
    }

    /// A NED synthetic descent must actually lose altitude.
    ///
    /// `generate_synthetic` propagates a NED state (`is_enu: false`) through `mechanize`, so it
    /// rode directly on the vertical-channel sign bug in `position_update`: a positive
    /// `velocity_down_mps` used to make the trajectory *climb*. Nothing caught it because no
    /// test ran the generator with a non-zero vertical rate.
    #[test]
    fn synthetic_ned_descent_loses_altitude() {
        use rand::SeedableRng;

        let config = SyntheticConfig {
            output: String::new(),
            initial_state: SyntheticInitialState {
                altitude_m: 2000.0,
                velocity_north_mps: 50.0,
                velocity_down_mps: 5.0, // descending at 5 m/s, NED
                ..Default::default()
            },
            duration_s: 60.0,
            sample_rate_hz: 10.0,
            imu_quality: crate::IMUQuality::default(),
            seed: 42,
            no_noise: true,
            gnss_horizontal_noise_m: 1.0,
            gnss_vertical_noise_m: 1.0,
            baro_noise_std_pa: 1.0,
            mag_noise_std_ut: default_mag_noise_std_ut(),
            mag_hard_iron_std_ut: default_mag_hard_iron_std_ut(),
        };
        assert!(!config.initial_state.is_enu, "synthetic default is NED");

        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let (truth, _) = generate_synthetic(&config, &mut rng).unwrap();

        let first = truth.first().unwrap();
        let last = truth.last().unwrap();
        assert!(
            last.altitude < first.altitude,
            "descending in NED must lose altitude: {} -> {}",
            first.altitude,
            last.altitude
        );
        // ~5 m/s over ~60 s, allowing for the vertical channel's own dynamics.
        let drop = first.altitude - last.altitude;
        assert!(
            (200.0..400.0).contains(&drop),
            "expected roughly 300 m of descent, got {drop}"
        );
    }
    use super::*;
    use assert_approx_eq::assert_approx_eq;
    use chrono::Utc;
    use std::fs::File;
    use std::path::Path;
    use std::vec;
    /// Generate a test record for northward motion at constant velocity (1 knot = 1852 m/h).
    /// This helper returns a Vec<TestDataRecord> for 1 hour, sampled once per second.
    fn generate_northward_motion_records() -> Vec<TestDataRecord> {
        let mut records: Vec<TestDataRecord> = Vec::with_capacity(3601);
        let start_lat: f64 = 0.0;
        let start_lon: f64 = 0.0;
        let start_alt: f64 = 0.0;
        let velocity_mps: f64 = 1852.0 / 3600.0; // 1 knot in m/s
        let earth_radius: f64 = 6371000.0_f64; // meters

        for t in 0..3600 {
            // Each second, latitude increases by dlat = (v / R) * (180/pi)
            let dlat: f64 =
                (velocity_mps * f64::from(t)) / earth_radius * (180.0 / std::f64::consts::PI);
            let time_str: String = format!("2023-01-01 00:{:02}:{:02}+00:00", t / 60, t % 60);

            records.push(TestDataRecord {
                time: DateTime::parse_from_str(&time_str, "%Y-%m-%d %H:%M:%S%z")
                    .map(|dt| dt.with_timezone(&Utc))
                    .unwrap(),
                bearing_accuracy: 0.0,
                speed_accuracy: 0.0,
                vertical_accuracy: 0.0,
                horizontal_accuracy: 0.0,
                speed: velocity_mps,
                bearing: 0.0,
                altitude: start_alt,
                longitude: start_lon,
                latitude: start_lat + dlat,
                qz: 0.0,
                qy: 0.0,
                qx: 0.0,
                qw: 1.0,
                roll: 0.0,
                pitch: 0.0,
                yaw: 0.0,
                acc_z: 0.0,
                acc_y: 0.0,
                acc_x: 0.0,
                gyro_z: 0.0,
                gyro_y: 0.0,
                gyro_x: 0.0,
                mag_z: 0.0,
                mag_y: 0.0,
                mag_x: 0.0,
                relative_altitude: 0.0,
                pressure: 1000.0,
                grav_z: 9.81,
                grav_y: 0.0,
                grav_x: 0.0,
            });
        }
        records
    }
    /// A record with every field zeroed except the ones a test sets.
    fn blank_record() -> TestDataRecord {
        let mut r = generate_northward_motion_records().swap_remove(0);
        r.speed = 0.0;
        r.bearing = 0.0;
        r
    }

    #[test]
    fn test_attitude_uses_quaternion_not_euler_angles() {
        // The first sample of `core/tests/test_data.csv`. Its Euler fields and its quaternion
        // describe the same attitude in two different conventions, which is the whole reason
        // `attitude()` exists.
        let mut r = blank_record();
        r.roll = 0.162_628_241_479_396_78;
        r.pitch = -1.339_523_780_345_916_8;
        r.yaw = 0.179_200_585_931_539_5;
        r.qw = 0.782_8;
        r.qx = 0.622_0;
        r.qy = 0.008_1;
        r.qz = -0.019_7;

        let via_quaternion = r.attitude();
        let via_euler = nalgebra::Rotation3::from_euler_angles(r.roll, r.pitch, r.yaw);

        // The two disagree, and by much more than rounding: if they ever agree, the recording
        // convention changed and this helper's reason for existing needs re-checking.
        let disagreement = (via_quaternion.matrix() - via_euler.matrix()).abs().max();
        assert!(
            disagreement > 0.1,
            "quaternion- and Euler-derived attitudes should differ for this record, \
             got max element difference {disagreement}"
        );

        // `attitude()` must reproduce the quaternion exactly.
        let expected: nalgebra::Rotation3<f64> = nalgebra::UnitQuaternion::from_quaternion(
            nalgebra::Quaternion::new(r.qw, r.qx, r.qy, r.qz),
        )
        .into();
        let reproduction_error = (via_quaternion.matrix() - expected.matrix()).abs().max();
        assert!(
            reproduction_error < 1e-12,
            "attitude() should reproduce the quaternion exactly, off by {reproduction_error}"
        );
    }

    #[test]
    fn test_attitude_cancels_gravity_in_enu() {
        // This is the property the bug violated. Rotating the first sample's accelerometer
        // reading into the navigation frame must give specific force along ENU up and
        // essentially nothing horizontal, because the vehicle was near stationary.
        let mut r = blank_record();
        r.qw = 0.782_8;
        r.qx = 0.622_0;
        r.qy = 0.008_1;
        r.qz = -0.019_7;
        r.acc_x = -0.361_2;
        r.acc_y = 9.467_4;
        r.acc_z = 2.194_5;

        let f_nav = r.attitude().matrix() * Vector3::new(r.acc_x, r.acc_y, r.acc_z);
        assert!(
            f_nav[0].abs() < 0.1 && f_nav[1].abs() < 0.1,
            "horizontal specific force should be ~0 at a stationary start, got ({}, {})",
            f_nav[0],
            f_nav[1]
        );
        assert!(
            (f_nav[2] - 9.7).abs() < 0.2,
            "vertical specific force should be ~+g (ENU up), got {}",
            f_nav[2]
        );
    }

    /// The inverse mechanization has to hold the commanded velocity in either convention.
    ///
    /// `compute_perfect_imu` solves `v_dot = 0`, so feeding its output straight back into
    /// `mechanize` must leave the velocity where it started. Before #321 only its gravity
    /// term consulted `is_enu` while the Coriolis term and the Earth-rate gyro beside it did
    /// not, so the ENU branch commanded a force that did not cancel anything.
    #[test]
    fn perfect_imu_holds_velocity_in_both_conventions() {
        use crate::{ImuSample, StrapdownState, mechanize};
        use nalgebra::Rotation3;

        let initial = StrapdownState {
            latitude: 51.5_f64.to_radians(),
            longitude: (-0.12_f64).to_radians(),
            altitude: 2400.0,
            velocity_north: 120.0,
            velocity_east: -45.0,
            velocity_vertical: 0.0,
            attitude: Rotation3::from_euler_angles(0.0, 0.0, 2.4),
            is_enu: false,
        };
        let dt = 0.01;
        let steps = 2000;

        let run = |mut state: StrapdownState| {
            for _ in 0..steps {
                let imu = super::compute_perfect_imu(&state, Vector3::zeros());
                mechanize(&mut state, &ImuSample::from_rates(&imu, dt)).unwrap();
            }
            state
        };

        let ned = run(initial);
        let enu = run(initial.to_enu());
        assert!(enu.is_enu);

        // Velocity held, in both.
        for held in [ned, enu.to_ned()] {
            assert!(
                (held.velocity_north - 120.0).abs() < 1e-3,
                "north velocity drifted to {}",
                held.velocity_north
            );
            assert!(
                (held.velocity_east + 45.0).abs() < 1e-3,
                "east velocity drifted to {}",
                held.velocity_east
            );
            assert!(
                held.velocity_vertical.abs() < 1e-3,
                "vertical velocity drifted to {}",
                held.velocity_vertical
            );
        }
        // And the two conventions agree on where the vehicle ended up.
        let converted = enu.to_ned();
        assert!((converted.latitude - ned.latitude).abs() < 1e-12);
        assert!((converted.longitude - ned.longitude).abs() < 1e-12);
        assert!((converted.altitude - ned.altitude).abs() < 1e-6);
        // Non-vacuous: 20 s at 128 m/s has to have moved the vehicle.
        assert!((ned.latitude - initial.latitude).abs() > 1e-7);
    }

    #[test]
    fn test_attitude_falls_back_to_identity_on_unusable_quaternion() {
        let mut nan = blank_record();
        nan.qw = f64::NAN;
        nan.qx = f64::NAN;
        nan.qy = f64::NAN;
        nan.qz = f64::NAN;
        assert_eq!(nan.attitude(), nalgebra::Rotation3::identity());

        let mut zero = blank_record();
        zero.qw = 0.0;
        zero.qx = 0.0;
        zero.qy = 0.0;
        zero.qz = 0.0;
        assert_eq!(zero.attitude(), nalgebra::Rotation3::identity());
    }

    #[test]
    fn test_ground_track_velocity_treats_bearing_as_degrees() {
        let mut due_east = blank_record();
        due_east.speed = 10.0;
        due_east.bearing = 90.0;
        let (north, east) = due_east.ground_track_velocity();
        assert_approx_eq!(north, 0.0, 1e-12);
        assert_approx_eq!(east, 10.0, 1e-12);

        // 90 read as radians instead of degrees gives cos(90 rad) = -0.448, which is the
        // bug this helper removes.
        assert!(
            (north - 10.0 * 90.0_f64.cos()).abs() > 1.0,
            "bearing must be converted from degrees, not consumed as radians"
        );

        let mut due_south = blank_record();
        due_south.speed = 4.0;
        due_south.bearing = 180.0;
        let (north, east) = due_south.ground_track_velocity();
        assert_approx_eq!(north, -4.0, 1e-12);
        assert_approx_eq!(east, 0.0, 1e-12);
    }

    #[test]
    fn test_ground_track_velocity_is_zero_when_fields_are_nan() {
        let mut no_speed = blank_record();
        no_speed.speed = f64::NAN;
        no_speed.bearing = 45.0;
        assert_eq!(no_speed.ground_track_velocity(), (0.0, 0.0));

        let mut no_bearing = blank_record();
        no_bearing.speed = 5.0;
        no_bearing.bearing = f64::NAN;
        assert_eq!(no_bearing.ground_track_velocity(), (0.0, 0.0));
    }

    #[test]
    fn test_synthetic_attitude_round_trips_through_the_record() {
        // A synthetic CSV must be readable by the same consumers as a real one. Both
        // representations it writes were previously wrong -- Euler angles in degrees where
        // the struct documents radians, and a `(cos(yaw), 0, 0, 0)` placeholder quaternion --
        // so a trajectory fed back through `cl` or `dead_reckoning` began from an attitude
        // unrelated to the truth it was generated from. Rolling and pitching here, not just
        // yawing, is the point: the old placeholder could not represent either.
        let config = SyntheticConfig {
            output: String::new(),
            initial_state: SyntheticInitialState {
                latitude_deg: 40.0,
                longitude_deg: -76.0,
                altitude_m: 100.0,
                velocity_north_mps: 20.0,
                velocity_east_mps: 5.0,
                velocity_down_mps: 0.0,
                roll_deg: 12.0,
                pitch_deg: -7.0,
                yaw_deg: 143.0,
                angular_velocity_x_dps: 3.0,
                angular_velocity_y_dps: -2.0,
                angular_velocity_z_dps: 5.0,
                is_enu: false,
            },
            duration_s: 10.0,
            sample_rate_hz: 10.0,
            imu_quality: crate::IMUQuality::Navigation,
            seed: 42,
            no_noise: true,
            gnss_horizontal_noise_m: 2.5,
            gnss_vertical_noise_m: 5.0,
            baro_noise_std_pa: 50.0,
            mag_noise_std_ut: default_mag_noise_std_ut(),
            mag_hard_iron_std_ut: default_mag_hard_iron_std_ut(),
        };
        let mut rng = rand::SeedableRng::seed_from_u64(42);
        let (truth, records) = generate_synthetic(&config, &mut rng).expect("generation");
        assert_eq!(truth.len(), records.len());

        for (i, (t, r)) in truth.iter().zip(records.iter()).enumerate() {
            let expected = nalgebra::Rotation3::from_euler_angles(t.roll, t.pitch, t.yaw);
            let error = (r.attitude().matrix() - expected.matrix()).abs().max();
            assert!(
                error < 1e-9,
                "record {i} attitude should reproduce the truth attitude, off by {error}"
            );
            // And the Euler fields are radians, as the struct documents.
            assert_approx_eq!(r.roll, t.roll, 1e-12);
            assert_approx_eq!(r.pitch, t.pitch, 1e-12);
            assert_approx_eq!(r.yaw, t.yaw, 1e-12);
        }

        // The written quaternion is a real unit quaternion.
        for (i, r) in records.iter().enumerate() {
            let norm = (r.qw * r.qw + r.qx * r.qx + r.qy * r.qy + r.qz * r.qz).sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-12,
                "record {i} quaternion should be unit-length, got {norm}"
            );
        }
    }

    /// A stationary navigation-grade synthetic run in `frame`, exactly as `syn` writes it.
    ///
    /// Stationary is the point: the truth altitude never moves, so any altitude the solution
    /// accumulates is the mechanization's own error and needs no differencing against a
    /// moving reference to read.
    fn stationary_synthetic_records(is_enu: bool, duration_s: f64) -> Vec<TestDataRecord> {
        let config = SyntheticConfig {
            output: String::new(),
            initial_state: SyntheticInitialState {
                latitude_deg: 40.0,
                longitude_deg: -76.0,
                altitude_m: 100.0,
                is_enu,
                ..SyntheticInitialState::default()
            },
            duration_s,
            sample_rate_hz: 10.0,
            imu_quality: crate::IMUQuality::Navigation,
            seed: 42,
            no_noise: false,
            gnss_horizontal_noise_m: 2.5,
            gnss_vertical_noise_m: 5.0,
            baro_noise_std_pa: 50.0,
            mag_noise_std_ut: default_mag_noise_std_ut(),
            mag_hard_iron_std_ut: default_mag_hard_iron_std_ut(),
        };
        let mut rng = rand::SeedableRng::seed_from_u64(42);
        generate_synthetic(&config, &mut rng)
            .expect("synthetic generation must succeed")
            .1
    }

    /// Altitude a stationary navigation-grade solution may drift over 60 s, in metres.
    ///
    /// Derived, not fitted, from the three things that move the vertical channel over a
    /// minute:
    ///
    /// 1. Accelerometer bias. [`crate::IMUQuality::Navigation`] quotes 1e-4 m/s^2 of bias
    ///    instability; a 3-sigma draw of 3e-4 m/s^2 integrates to
    ///    $\tfrac{1}{2} a t^2 = 0.54$ m at $t = 60$ s.
    /// 2. Velocity random walk. 0.005 m/s/$\sqrt{\text{h}}$ gives
    ///    $\sigma_v(60\,\text{s}) = 6.5\times10^{-4}$ m/s, under 0.04 m of position.
    /// 3. The vertical channel's own instability, which grows as
    ///    $\cosh(t/\tau)$ with $\tau = \sqrt{R/g} \approx 806$ s -- a factor of 1.003 over
    ///    this interval, so it multiplies the 0.58 m above rather than adding to it.
    ///
    /// The budget is therefore ~0.6 m and the bound is ~3x it. What matters is the other
    /// end: mechanizing these NED records as ENU reaches 35 km in the same 60 s (#296), four
    /// and a half orders of magnitude outside this bound, so the test cannot pass by accident
    /// with the frame wrong.
    const MAX_STATIONARY_ALTITUDE_DRIFT_M: f64 = 2.0;

    /// Worst absolute altitude excursion from the run's own first sample, in metres.
    fn worst_altitude_excursion(results: &[NavigationResult]) -> f64 {
        let start = results[0].altitude;
        results
            .iter()
            .map(|result| (result.altitude - start).abs())
            .fold(0.0, f64::max)
    }

    #[test]
    fn test_dead_reckoning_holds_altitude_on_ned_synthetic() {
        // The regression this issue is about: `syn` emits NED, and until the frame became a
        // parameter `dead_reckoning` mechanized it as ENU, which adds the gravity model to
        // the sensed specific force instead of cancelling it and falls at 2 g.
        let records = stationary_synthetic_records(false, 60.0);
        let results = dead_reckoning(&records, false).expect("NED records must dead-reckon as NED");
        let worst = worst_altitude_excursion(&results);
        assert!(
            worst < MAX_STATIONARY_ALTITUDE_DRIFT_M,
            "stationary navigation-grade NED truth drifted {worst:.3} m of altitude over 60 s, \
             past the {MAX_STATIONARY_ALTITUDE_DRIFT_M} m budget derived in \
             MAX_STATIONARY_ALTITUDE_DRIFT_M. A figure in the tens of kilometres means the \
             frame is being double-counted again (#296); a figure a little over the bound \
             means the IMU error model or the vertical channel moved and the budget needs \
             re-deriving."
        );
    }

    #[test]
    fn test_dead_reckoning_holds_altitude_on_enu_synthetic() {
        // The other half of the same claim, and the one that stops a future default flip from
        // quietly breaking Sensor Logger recordings: matched ENU has to be just as exact as
        // matched NED, because since #321 an ENU run converts to NED internally rather than
        // approximating.
        let records = stationary_synthetic_records(true, 60.0);
        let results = dead_reckoning(&records, true).expect("ENU records must dead-reckon as ENU");
        let worst = worst_altitude_excursion(&results);
        assert!(
            worst < MAX_STATIONARY_ALTITUDE_DRIFT_M,
            "stationary navigation-grade ENU truth drifted {worst:.3} m of altitude over 60 s, \
             past the {MAX_STATIONARY_ALTITUDE_DRIFT_M} m budget derived in \
             MAX_STATIONARY_ALTITUDE_DRIFT_M"
        );
    }

    #[test]
    fn test_dead_reckoning_rejects_a_frame_the_records_contradict() {
        // Both directions, because the guard has to be a discriminator rather than a
        // one-sided preference for the new default.
        for declared_enu in [false, true] {
            let records = stationary_synthetic_records(!declared_enu, 10.0);
            let error = dead_reckoning(&records, declared_enu).expect_err(
                "records generated in one frame must not be silently mechanized in the other",
            );
            assert!(
                matches!(
                    error,
                    StrapdownError::InvalidConfiguration {
                        field: "is_enu",
                        ..
                    }
                ),
                "expected an InvalidConfiguration on is_enu, got {error:?}"
            );
            // The message has to name the way out, or the user is told only that they are
            // wrong. This is the half of #272's objection the flag alone does not answer.
            let message = error.to_string();
            assert!(
                message.contains("--enu"),
                "the rejection must name the flag to pass, got: {message}"
            );
        }
    }

    #[test]
    fn test_check_declared_frame_accepts_each_frame_at_rest() {
        let gravity = crate::earth::gravity(&0.0, &0.0);

        // Identity attitude, so the body reading is already the navigation-frame one.
        let mut enu = blank_record();
        enu.acc_z = gravity;
        assert!(check_declared_frame(std::slice::from_ref(&enu), true).is_ok());
        assert!(check_declared_frame(std::slice::from_ref(&enu), false).is_err());

        let mut ned = blank_record();
        ned.acc_z = -gravity;
        assert!(check_declared_frame(std::slice::from_ref(&ned), false).is_ok());
        assert!(check_declared_frame(std::slice::from_ref(&ned), true).is_err());
    }

    #[test]
    fn test_check_declared_frame_fires_at_the_derived_margin() {
        // Brackets FRAME_CHECK_MARGIN_G from the outside, with literal accelerations rather
        // than by restating the constant -- a straddle computed *from* the constant would
        // follow it wherever it moved and prove nothing about where it should be.
        //
        // In NED the sensed vertical specific force is $f_z = a_\text{down} - g$, so a
        // descent at $\alpha$ g reads $(\alpha - 1) g$, and the guard fires at
        // $\alpha > 1 + \text{margin}$. The four cases below are each an independent
        // constraint on the margin:
        //
        // - free fall ($\alpha = 1$, $f_z = 0$) accepted  =>  margin > 0
        // - at rest in the *other* frame rejected          =>  margin < 1
        // - a 1.45 g descent accepted                      =>  margin > 0.45
        // - a 1.55 g descent rejected                      =>  margin < 0.55
        //
        // The last two pin the constant to 0.5 +/- 0.05. The first two are the physical
        // requirements that motivate it: free fall is a real flight condition and must never
        // be mistaken for a frame error, and a stationary wrong-frame file must never be
        // mistaken for flight.
        let gravity = crate::earth::gravity(&0.0, &0.0);
        let with_downward_acceleration = |alpha: f64| {
            let mut record = blank_record();
            record.acc_z = (alpha - 1.0) * gravity;
            record
        };

        for (alpha, must_reject) in [(1.0, false), (1.45, false), (1.55, true)] {
            let record = with_downward_acceleration(alpha);
            let rejected = check_declared_frame(std::slice::from_ref(&record), false).is_err();
            assert_eq!(
                rejected, must_reject,
                "a {alpha} g descent in correctly declared NED records: expected \
                 rejected={must_reject}, got rejected={rejected}. FRAME_CHECK_MARGIN_G is \
                 {FRAME_CHECK_MARGIN_G} and these cases bracket it to 0.5 +/- 0.05."
            );
        }

        // At rest in the other frame: the case the guard exists for.
        let mut at_rest_in_enu = blank_record();
        at_rest_in_enu.acc_z = gravity;
        assert!(
            check_declared_frame(std::slice::from_ref(&at_rest_in_enu), false).is_err(),
            "a stationary ENU record declared NED must be rejected"
        );
    }

    #[test]
    fn test_check_declared_frame_tolerates_a_manoeuvring_start() {
        // The guard must not fire on a run that simply begins under acceleration: 0.4 g of
        // horizontal specific force leaves the vertical channel where it was, and a 0.9 g
        // descent -- just short of free fall -- still reads on the correct side of zero.
        let gravity = crate::earth::gravity(&0.0, &0.0);

        let mut manoeuvring = blank_record();
        manoeuvring.acc_x = 0.4 * gravity;
        manoeuvring.acc_y = -0.4 * gravity;
        manoeuvring.acc_z = -gravity;
        assert!(check_declared_frame(std::slice::from_ref(&manoeuvring), false).is_ok());

        let mut descending = blank_record();
        descending.acc_z = -0.1 * gravity; // a_down = 0.9 g
        assert!(check_declared_frame(std::slice::from_ref(&descending), false).is_ok());
    }

    #[test]
    fn test_check_declared_frame_fails_open_on_unusable_records() {
        // Deliberately permissive: an empty slice, a non-finite reading and a non-finite
        // gravity all pass. Every comparison against NaN is false anyway, so the choice is
        // between failing open explicitly and failing open by accident.
        assert!(check_declared_frame(&[], false).is_ok());
        assert!(check_declared_frame(&[], true).is_ok());

        let mut nan_accel = blank_record();
        nan_accel.acc_z = f64::NAN;
        assert!(check_declared_frame(std::slice::from_ref(&nan_accel), false).is_ok());

        let mut nan_position = blank_record();
        nan_position.acc_z = crate::earth::gravity(&0.0, &0.0);
        nan_position.latitude = f64::NAN;
        assert!(check_declared_frame(std::slice::from_ref(&nan_position), false).is_ok());
    }

    #[test]
    fn test_generate_northward_motion_records_end_latitude() {
        let records = generate_northward_motion_records();
        // The last record should have latitude close to 0.016667 (1 knot north in 1 hour)
        let last = records.last().unwrap();
        let expected_lat = 0.016667;
        let tolerance = 1e-3;
        assert!(
            (last.latitude - expected_lat).abs() < tolerance,
            "Ending latitude {} not within {} of expected {}",
            last.latitude,
            tolerance,
            expected_lat
        );
        // write to CSV
        let northward = File::create("northward_motion.csv").unwrap();
        let mut writer = csv::Writer::from_writer(northward);
        for record in &records {
            writer.serialize(record).unwrap();
        }
        writer.flush().unwrap();
        // Clean up the test file
        let _ = std::fs::remove_file("northward_motion.csv");
    }
    /// Test that reading a missing file returns an error.
    #[test]
    fn test_test_data_record_from_csv_invalid_path() {
        let path = Path::new("nonexistent.csv");
        let result = TestDataRecord::from_csv(path);
        assert!(result.is_err(), "Should error on missing file");
    }
    /// Test writing `TestDataRecord` to CSV and reading it back
    #[test]
    fn test_data_record_to_and_from_csv() {
        // Read original records
        let path = Path::new("test_file.csv");
        // Create some test data records
        let records: Vec<TestDataRecord> = vec![
            TestDataRecord {
                time: DateTime::parse_from_str("2023-01-01 00:00:00+00:00", "%Y-%m-%d %H:%M:%S%z")
                    .unwrap()
                    .with_timezone(&Utc),
                bearing_accuracy: 0.1,
                speed_accuracy: 0.1,
                vertical_accuracy: 0.1,
                horizontal_accuracy: 0.1,
                speed: 1.0,
                bearing: 90.0,
                altitude: 100.0,
                longitude: -122.0,
                latitude: 37.0,
                qz: 0.0,
                qy: 0.0,
                qx: 0.0,
                qw: 1.0,
                roll: 0.0,
                pitch: 0.0,
                yaw: 0.0,
                acc_z: 9.81,
                acc_y: 0.0,
                acc_x: 0.0,
                gyro_z: 0.01,
                gyro_y: 0.01,
                gyro_x: 0.01,
                mag_z: 50.0,
                mag_y: -30.0,
                mag_x: -20.0,
                relative_altitude: 5.0,
                pressure: 1013.25,
                grav_z: 9.81,
                grav_y: 0.0,
                grav_x: 0.0,
            },
            TestDataRecord {
                time: DateTime::parse_from_str("2023-01-01 00:01:00+00:00", "%Y-%m-%d %H:%M:%S%z")
                    .unwrap()
                    .with_timezone(&Utc),
                bearing_accuracy: 0.1,
                speed_accuracy: 0.1,
                vertical_accuracy: 0.1,
                horizontal_accuracy: 0.1,
                speed: 2.0,
                bearing: 180.0,
                altitude: 200.0,
                longitude: -121.0,
                latitude: 38.0,
                qz: 0.0,
                qy: 0.0,
                qx: 0.0,
                qw: 1.0,
                roll: 0.1,
                pitch: 0.1,
                yaw: 0.1,
                acc_z: 9.81,
                acc_y: 0.01,
                acc_x: -0.01,
                gyro_z: 0.02,
                gyro_y: -0.02,
                gyro_x: 0.02,
                mag_z: 55.0,
                mag_y: -25.0,
                mag_x: -15.0,
                relative_altitude: 10.0,
                pressure: 1012.25,
                grav_z: 9.81,
                grav_y: 0.01,
                grav_x: -0.01,
            },
        ];
        // Write to CSV
        TestDataRecord::to_csv(&records, path).expect("Failed to write test data to CSV");
        // Check to make sure the file exists
        assert!(path.exists(), "Test data CSV file should exist");
        // Read back from CSV
        let read_records =
            TestDataRecord::from_csv(path).expect("Failed to read test data from CSV");
        // Check that the read records match the original
        assert_eq!(
            read_records.len(),
            records.len(),
            "Record count should match"
        );
        for (i, record) in read_records.iter().enumerate() {
            assert_eq!(record.time, records[i].time, "Timestamps should match");
            assert!(
                (record.latitude - records[i].latitude).abs() < 1e-6,
                "Latitudes should match"
            );
            assert!(
                (record.longitude - records[i].longitude).abs() < 1e-6,
                "Longitudes should match"
            );
            assert!(
                (record.altitude - records[i].altitude).abs() < 1e-6,
                "Altitudes should match"
            );
            // Add more assertions as needed for other fields
        }
        // Clean up
        let _ = std::fs::remove_file(path);
    }
    #[test]
    fn test_navigation_result_new() {
        let nav = NavigationResult::default();
        //let expected_timestamp = chrono::Utc::now();
        //assert_eq!(nav.timestamp, expected_timestamp);
        assert_eq!(nav.latitude, 0.0);
        assert_eq!(nav.longitude, 0.0);
        assert_eq!(nav.altitude, 0.0);
        assert_eq!(nav.velocity_north, 0.0);
        assert_eq!(nav.velocity_east, 0.0);
        assert_eq!(nav.velocity_vertical, 0.0);
    }
    #[test]
    fn test_navigation_result_from_strapdown_state() {
        let state = StrapdownState {
            latitude: 1.0,
            longitude: 2.0,
            altitude: 3.0,
            velocity_north: 4.0,
            velocity_east: 5.0,
            velocity_vertical: 6.0,
            attitude: nalgebra::Rotation3::from_euler_angles(7.0, 8.0, 9.0),
            ..Default::default()
        };

        let state_vector: DVector<f64> = DVector::from_vec(vec![
            state.latitude,
            state.longitude,
            state.altitude,
            state.velocity_north,
            state.velocity_east,
            state.velocity_vertical,
            state.attitude.euler_angles().0, // roll
            state.attitude.euler_angles().1, // pitch
            state.attitude.euler_angles().2, // yaw
            0.0,                             // acc_bias_x
            0.0,                             // acc_bias_y
            0.0,                             // acc_bias_z
            0.0,                             // gyro_bias_x
            0.0,                             // gyro_bias_y
            0.0,                             // gyro_bias_z
        ]);
        let timestamp = chrono::Utc::now();
        let nav = NavigationResult::from((
            &timestamp,
            &state_vector,
            &DMatrix::from_diagonal(&DVector::from_element(15, 0.0)), // dummy covariance
        ));
        assert_eq!(nav.latitude, (1.0_f64).to_degrees());
        assert_eq!(nav.longitude, (2.0_f64).to_degrees());
        assert_eq!(nav.altitude, 3.0);
        assert_eq!(nav.velocity_north, 4.0);
        assert_eq!(nav.velocity_east, 5.0);
        assert_eq!(nav.velocity_vertical, 6.0);
    }
    #[test]
    fn test_navigation_result_to_csv_and_from_csv() {
        let mut nav = NavigationResult::new();
        nav.latitude = 1.0;
        nav.longitude = 2.0;
        nav.altitude = 3.0;
        nav.velocity_north = 4.0;
        nav.velocity_east = 5.0;
        nav.velocity_vertical = 6.0;
        let temp_file = std::env::temp_dir().join("test_nav_result.csv");
        NavigationResult::to_csv(std::slice::from_ref(&nav), &temp_file).unwrap();
        let read = NavigationResult::from_csv(&temp_file).unwrap();
        assert_eq!(read.len(), 1);
        assert_eq!(read[0].latitude, 1.0);
        assert_eq!(read[0].longitude, 2.0);
        assert_eq!(read[0].altitude, 3.0);
        assert_eq!(read[0].velocity_north, 4.0);
        assert_eq!(read[0].velocity_east, 5.0);
        assert_eq!(read[0].velocity_vertical, 6.0);
        let _ = std::fs::remove_file(&temp_file);
    }
    #[test]
    fn test_closed_loop_minimal() {
        let rec = TestDataRecord {
            time: chrono::Utc::now(),
            bearing_accuracy: 0.0,
            speed_accuracy: 0.0,
            vertical_accuracy: 0.0,
            horizontal_accuracy: 0.0,
            speed: 0.0,
            bearing: 0.0,
            altitude: 0.0,
            longitude: 0.0,
            latitude: 0.0,
            qz: 0.0,
            qy: 0.0,
            qx: 0.0,
            qw: 1.0,
            roll: 0.0,
            pitch: 0.0,
            yaw: 0.0,
            acc_z: 0.0,
            acc_y: 0.0,
            acc_x: 0.0,
            gyro_z: 0.0,
            gyro_y: 0.0,
            gyro_x: 0.0,
            mag_z: 0.0,
            mag_y: 0.0,
            mag_x: 0.0,
            relative_altitude: 0.0,
            pressure: 0.0,
            grav_z: 0.0,
            grav_y: 0.0,
            grav_x: 0.0,
        };

        // Initialize UKF
        let mut ukf = initialize_ukf(&rec, UkfConfig::default()).unwrap();

        // Create a minimal EventStream with one IMU event
        let imu_data = IMUData {
            accel: nalgebra::Vector3::new(rec.acc_x, rec.acc_y, rec.acc_z),
            gyro: nalgebra::Vector3::new(rec.gyro_x, rec.gyro_y, rec.gyro_z),
        };
        let event = Event::Imu {
            dt_s: 1.0,
            imu: imu_data,
            elapsed_s: 0.0,
        };
        let stream = EventStream {
            start_time: rec.time,
            events: vec![event],
        };

        let res = run_closed_loop(&mut ukf, stream, None, None);
        assert!(!res.unwrap().is_empty());
    }
    #[test]
    fn test_initialize_ukf_default_and_custom() {
        let rec = TestDataRecord {
            time: chrono::Utc::now(),
            bearing_accuracy: 0.0,
            speed_accuracy: 0.0,
            vertical_accuracy: 1.0,
            horizontal_accuracy: 4.0,
            speed: 1.0,
            bearing: 0.0,
            altitude: 10.0,
            longitude: 20.0,
            latitude: 30.0,
            qz: 0.0,
            qy: 0.0,
            qx: 0.0,
            qw: 1.0,
            roll: 0.0,
            pitch: 0.0,
            yaw: 0.0,
            acc_z: 0.0,
            acc_y: 0.0,
            acc_x: 0.0,
            gyro_z: 0.0,
            gyro_y: 0.0,
            gyro_x: 0.0,
            mag_z: 0.0,
            mag_y: 0.0,
            mag_x: 0.0,
            relative_altitude: 0.0,
            pressure: 0.0,
            grav_z: 0.0,
            grav_y: 0.0,
            grav_x: 0.0,
        };
        let ukf = initialize_ukf(&rec, UkfConfig::default()).unwrap();
        assert!(!ukf.get_estimate().is_empty());
        let ukf2 = initialize_ukf(
            &rec,
            UkfConfig {
                attitude_covariance: Some(vec![0.1, 0.2, 0.3]),
                imu_biases: Some(vec![0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
                ..Default::default()
            },
        )
        .unwrap();
        assert!(!ukf2.get_estimate().is_empty());
    }

    /// A record whose bearing and attitude are both non-zero and both distinctive, so that a
    /// unit error anywhere in the seeding path shows up as a wrong number rather than as a
    /// coincidentally-correct zero.
    ///
    /// `bearing` is 90 degrees -- due east -- which is the single most diagnostic value
    /// available: fed to `cos`/`sin` unconverted it gives `(-0.448, +0.894)`, so a filter that
    /// forgets the conversion seeds a due-east track as north-west at half speed. The Euler
    /// fields are deliberately set to a *different* attitude from the quaternion, the way
    /// Sensor Logger's own two conventions disagree, so a seed that reads the columns instead
    /// of the quaternion cannot pass.
    fn seeded_pose() -> (TestDataRecord, f64, f64, f64) {
        let (roll, pitch, yaw) = (0.2_f64, -0.3_f64, 1.1_f64);
        let quaternion = nalgebra::UnitQuaternion::from_euler_angles(roll, pitch, yaw);

        let mut record = blank_record();
        record.latitude = 30.0;
        record.longitude = -80.0;
        record.altitude = 100.0;
        record.speed = 5.0;
        record.bearing = 90.0;
        record.horizontal_accuracy = 4.0;
        record.vertical_accuracy = 1.0;
        record.speed_accuracy = 0.5;
        record.qw = quaternion.w;
        record.qx = quaternion.i;
        record.qy = quaternion.j;
        record.qz = quaternion.k;
        // Not the quaternion's angles, and not in its convention either.
        record.roll = 1.5;
        record.pitch = 1.5;
        record.yaw = 1.5;

        (record, roll, pitch, yaw)
    }

    /// Assert that a filter's reported estimate is the seed the record describes.
    ///
    /// The first nine entries are the navigation state in the crate's native units: latitude
    /// and longitude in radians, altitude in metres, velocities in m/s, attitude in radians.
    fn assert_seeded_estimate(
        name: &str,
        estimate: &DVector<f64>,
        roll: f64,
        pitch: f64,
        yaw: f64,
    ) {
        assert_approx_eq!(estimate[0], 30.0_f64.to_radians(), 1e-12);
        assert_approx_eq!(estimate[1], (-80.0_f64).to_radians(), 1e-12);
        assert_approx_eq!(estimate[2], 100.0, 1e-9);
        // Bearing 90 deg at 5 m/s is due east: no northward component at all. Unconverted
        // it is (-2.24, +4.47), which is what the UKF used to seed; the EKF and ESKF
        // literals already converted, and this holds all three to that one answer.
        assert_approx_eq!(estimate[3], 0.0, 1e-9);
        assert_approx_eq!(estimate[4], 5.0, 1e-9);
        assert_approx_eq!(estimate[5], 0.0, 1e-12);
        // Radians, once. Tagged `in_degrees: true` these arrived shrunk by 57.3x.
        assert_approx_eq!(estimate[6], roll, 1e-9);
        assert_approx_eq!(estimate[7], pitch, 1e-9);
        assert_approx_eq!(estimate[8], yaw, 1e-9);
        assert!(
            estimate.iter().all(|v| v.is_finite()),
            "{name} seeded a non-finite estimate: {estimate:?}"
        );
    }

    /// The property #337 was about: a record with a known bearing and a known attitude must
    /// reach every filter as that bearing and that attitude.
    ///
    /// Nothing checked this before. The workspace's integration tests all build `InitialState`
    /// as struct literals with their own values, so none of them exercised these functions'
    /// unit handling, and the two errors happened to be self-concealing -- the degrees-tagged
    /// radian attitude shrank the seed *toward* zero, and GNSS velocity aiding corrected the
    /// mis-seeded velocity within a few seconds -- so the integration metrics absorbed both.
    #[test]
    fn initialize_seeds_bearing_and_attitude_in_the_units_the_record_uses() {
        let (record, roll, pitch, yaw) = seeded_pose();

        let ukf = initialize_ukf(&record, UkfConfig::default()).unwrap();
        assert_seeded_estimate("UKF", &ukf.get_estimate(), roll, pitch, yaw);

        let ekf = initialize_ekf(&record, EkfConfig::default()).unwrap();
        assert_seeded_estimate("EKF", &ekf.get_estimate(), roll, pitch, yaw);

        let eskf = initialize_eskf(&record, EskfConfig::default()).unwrap();
        assert_seeded_estimate("ESKF", &eskf.get_estimate(), roll, pitch, yaw);
    }

    /// The seed comes from the record's quaternion, not from its Euler columns.
    ///
    /// `seeded_pose` gives the two different attitudes on purpose; this pins down which one
    /// wins, and so keeps the closed-loop seed on the same footing as `dead_reckoning`, which
    /// #302 moved to the quaternion for the reasons `TestDataRecord::attitude` documents.
    #[test]
    fn initial_state_takes_attitude_from_the_quaternion() {
        let (record, roll, pitch, yaw) = seeded_pose();
        let seed = record.initial_state(false);

        assert!(!seed.in_degrees, "the seed advertises radians");
        assert_approx_eq!(seed.roll, roll, 1e-12);
        assert_approx_eq!(seed.pitch, pitch, 1e-12);
        assert_approx_eq!(seed.yaw, yaw, 1e-12);
        // The Euler columns say 1.5 rad on all three axes; none of that reached the seed.
        assert!((seed.roll - record.roll).abs() > 1.0);
    }

    /// The declared frame is carried through untouched, and nothing else about the seed
    /// depends on it: the record reports no vertical rate in either convention, so there is
    /// no sign to get wrong here (#296).
    #[test]
    fn initial_state_carries_the_declared_frame() {
        let (record, ..) = seeded_pose();

        let ned = record.initial_state(false);
        let enu = record.initial_state(true);

        assert!(!ned.is_enu);
        assert!(enu.is_enu);
        assert_approx_eq!(ned.vertical_velocity, 0.0, 1e-12);
        assert_approx_eq!(enu.vertical_velocity, 0.0, 1e-12);
        assert_approx_eq!(ned.northward_velocity, enu.northward_velocity, 1e-12);
        assert_approx_eq!(ned.eastward_velocity, enu.eastward_velocity, 1e-12);
    }

    /// A record missing its GNSS track or its quaternion seeds a level, stationary state
    /// rather than a NaN one. `ground_track_velocity` and `attitude` each already guarantee
    /// this; the seed inherits it, and a NaN here would poison a whole run.
    #[test]
    fn initial_state_absorbs_missing_fields() {
        let mut record = blank_record();
        record.speed = f64::NAN;
        record.bearing = f64::NAN;
        record.qw = f64::NAN;
        record.qx = f64::NAN;
        record.qy = f64::NAN;
        record.qz = f64::NAN;

        let seed = record.initial_state(false);
        assert_approx_eq!(seed.northward_velocity, 0.0, 1e-12);
        assert_approx_eq!(seed.eastward_velocity, 0.0, 1e-12);
        assert_approx_eq!(seed.roll, 0.0, 1e-12);
        assert_approx_eq!(seed.pitch, 0.0, 1e-12);
        assert_approx_eq!(seed.yaw, 0.0, 1e-12);
    }

    // Helper to produce the header in the same order the struct expects
    fn test_header() -> Vec<&'static str> {
        vec![
            "time",
            "bearingAccuracy",
            "speedAccuracy",
            "verticalAccuracy",
            "horizontalAccuracy",
            "speed",
            "bearing",
            "altitude",
            "longitude",
            "latitude",
            "qz",
            "qy",
            "qx",
            "qw",
            "roll",
            "pitch",
            "yaw",
            "acc_z",
            "acc_y",
            "acc_x",
            "gyro_z",
            "gyro_y",
            "gyro_x",
            "mag_z",
            "mag_y",
            "mag_x",
            "relativeAltitude",
            "pressure",
            "grav_z",
            "grav_y",
            "grav_x",
        ]
    }
    #[test]
    fn deserialize_with_empty_fields_maps_to_nan() {
        let headers = test_header();
        let time = "2023-08-04T21:47:58Z";
        let mut row: Vec<String> = Vec::with_capacity(headers.len());
        row.push(time.to_string());
        for _ in 1..headers.len() {
            row.push(String::new());
        }
        let mut csv_data = String::new();
        csv_data.push_str(&headers.join(","));
        csv_data.push('\n');
        csv_data.push_str(&row.join(","));
        let temp_file = std::env::temp_dir().join("test_empty_fields_nan.csv");
        std::fs::write(&temp_file, csv_data).unwrap();
        let recs = TestDataRecord::from_csv(&temp_file).expect("from_csv should succeed");
        assert_eq!(recs.len(), 1);
        let r = &recs[0];
        assert_eq!(
            r.time,
            chrono::DateTime::parse_from_rfc3339(time)
                .unwrap()
                .with_timezone(&Utc)
        );
        assert!(r.speed.is_nan());
        assert!(r.latitude.is_nan());
        assert!(r.longitude.is_nan());
        assert!(r.acc_x.is_nan());
        let _ = std::fs::remove_file(&temp_file);
    }
    #[test]
    fn deserialize_with_missing_trailing_columns_returns_error() {
        let headers = test_header();
        let time = "2023-08-04T21:47:58Z";
        let row: Vec<String> = vec![time.to_string(), String::from("1.0"), String::from("2.0")];
        let mut csv_data = String::new();
        csv_data.push_str(&headers.join(","));
        csv_data.push('\n');
        csv_data.push_str(&row.join(","));
        let temp_file = std::env::temp_dir().join("test_missing_trailing.csv");
        std::fs::write(&temp_file, csv_data).unwrap();
        let recs = TestDataRecord::from_csv(&temp_file).expect("from_csv should succeed");
        assert_eq!(recs.len(), 1);
        let rec = &recs[0];
        assert_eq!(
            rec.time,
            chrono::DateTime::parse_from_rfc3339(time)
                .unwrap()
                .with_timezone(&Utc)
        );
        assert!(rec.speed.is_nan());
        assert!(rec.latitude.is_nan());
        assert!(rec.longitude.is_nan());
        let _ = std::fs::remove_file(&temp_file);
    }
    #[test]
    fn manual_padding_then_deserialize_succeeds() {
        let headers = test_header();
        let time = "2023-08-04T21:47:58Z";
        let row: Vec<String> = vec![
            time.to_string(),
            String::new(),          // bearingAccuracy
            String::new(),          // speedAccuracy
            String::new(),          // verticalAccuracy
            String::new(),          // horizontalAccuracy
            String::new(),          // speed
            String::new(),          // bearing
            String::new(),          // altitude
            String::from("-122.0"), // longitude
            String::from("37.0"),   // latitude
        ];
        let mut csv_data = String::new();
        csv_data.push_str(&headers.join(","));
        csv_data.push('\n');
        csv_data.push_str(&row.join(","));
        let temp_file = std::env::temp_dir().join("test_manual_padding.csv");
        std::fs::write(&temp_file, csv_data).unwrap();
        let got = TestDataRecord::from_csv(&temp_file).expect("from_csv should succeed");
        assert_eq!(got.len(), 1);
        let r = &got[0];
        assert_eq!(
            r.time,
            chrono::DateTime::parse_from_rfc3339(time)
                .unwrap()
                .with_timezone(&Utc)
        );
        assert_eq!(r.longitude, -122.0);
        assert_eq!(r.latitude, 37.0);
        let _ = std::fs::remove_file(&temp_file);
    }
    #[test]
    fn test_de_f64_nan_with_various_inputs() {
        // Test CSV deserialization with NaN/null/empty values
        let headers = test_header();
        let mut csv_data = String::new();
        csv_data.push_str(&headers.join(","));
        csv_data.push('\n');

        // Row with mixed NaN representations
        csv_data.push_str("2023-08-04T21:47:58Z,NaN,null,,,1.5,90.0,100.0,-122.0,37.0,");
        csv_data.push_str("0,0,0,1,0.1,0.2,0.3,9.8,0,0,0,0,0,0,0,0,0,1013.25,9.81,0,0\n");

        let temp_file = std::env::temp_dir().join("test_nan_variants.csv");
        std::fs::write(&temp_file, csv_data).unwrap();
        let recs = TestDataRecord::from_csv(&temp_file).expect("Should parse");
        assert_eq!(recs.len(), 1);
        assert!(recs[0].bearing_accuracy.is_nan());
        assert!(recs[0].speed_accuracy.is_nan());
        assert!(recs[0].vertical_accuracy.is_nan());
        assert!(recs[0].horizontal_accuracy.is_nan());
        assert_eq!(recs[0].speed, 1.5);
        let _ = std::fs::remove_file(&temp_file);
    }
    #[test]
    fn test_test_data_record_display() {
        let rec = TestDataRecord {
            time: DateTime::parse_from_str("2023-01-01 00:00:00+00:00", "%Y-%m-%d %H:%M:%S%z")
                .unwrap()
                .with_timezone(&Utc),
            latitude: 37.0,
            longitude: -122.0,
            altitude: 100.0,
            speed: 5.0,
            bearing: 90.0,
            ..Default::default()
        };
        let display_str = format!("{rec}");
        assert!(display_str.contains("37"));
        assert!(display_str.contains("-122"));
        assert!(display_str.contains("100"));
    }
    #[test]
    fn test_navigation_result_from_ukf() {
        let rec = TestDataRecord {
            time: Utc::now(),
            horizontal_accuracy: 5.0,
            vertical_accuracy: 2.0,
            speed_accuracy: 1.0,
            latitude: 37.0,
            longitude: -122.0,
            altitude: 100.0,
            speed: 10.0,
            bearing: 45.0,
            roll: 0.1,
            pitch: 0.2,
            yaw: 0.3,
            ..Default::default()
        };
        let ukf = initialize_ukf(&rec, UkfConfig::default()).unwrap();
        let timestamp = Utc::now();
        let nav_result = NavigationResult::from((&timestamp, &ukf));

        assert_eq!(nav_result.timestamp, timestamp);
        assert!(nav_result.latitude.is_finite());
        assert!(nav_result.longitude.is_finite());
        assert!(nav_result.altitude.is_finite());
    }
    #[test]
    fn test_dead_reckoning_empty_records() {
        let results = dead_reckoning(&[], false).unwrap();
        assert!(results.is_empty());
    }
    #[test]
    fn test_dead_reckoning_single_record() {
        let rec = TestDataRecord {
            time: Utc::now(),
            latitude: 37.0,
            longitude: -122.0,
            altitude: 100.0,
            speed: 1.0,
            bearing: 0.0,
            roll: 0.0,
            pitch: 0.0,
            yaw: 0.0,
            acc_x: 0.0,
            acc_y: 0.0,
            acc_z: 9.81,
            gyro_x: 0.0,
            gyro_y: 0.0,
            gyro_z: 0.0,
            ..Default::default()
        };
        // The stub's `acc_z: 9.81` with an identity attitude is ENU-convention specific
        // force, so declare ENU: `check_declared_frame` would (correctly) reject NED here.
        let results = dead_reckoning(&[rec], true).unwrap();
        assert_eq!(results.len(), 1);
    }
    #[test]
    fn test_print_ukf() {
        let rec = TestDataRecord {
            time: Utc::now(),
            horizontal_accuracy: 5.0,
            vertical_accuracy: 2.0,
            speed_accuracy: 1.0,
            latitude: 37.0,
            longitude: -122.0,
            altitude: 100.0,
            speed: 10.0,
            bearing: 45.0,
            roll: 0.1,
            pitch: 0.2,
            yaw: 0.3,
            ..Default::default()
        };
        let ukf = initialize_ukf(&rec, UkfConfig::default()).unwrap();
        // Just ensure it doesn't panic
        print_ukf(&ukf, &rec);
    }
    /// All three filter initialisers must seed the same ground track from the same record.
    ///
    /// `bearing` is degrees. `initialize_ukf` fed it to `cos`/`sin` raw while
    /// `initialize_ekf` and `initialize_eskf` converted first, so the three disagreed on
    /// the same input: at bearing 90 the UKF seeded (-4.48, 8.94) m/s north/east where the
    /// other two seeded (0.00, 10.00). All three now route through
    /// [`TestDataRecord::ground_track_velocity`], which owns the conversion.
    #[test]
    fn test_initializers_agree_on_ground_track() {
        let rec = TestDataRecord {
            time: Utc::now(),
            horizontal_accuracy: 5.0,
            vertical_accuracy: 2.0,
            speed_accuracy: 1.0,
            latitude: 37.0,
            longitude: -122.0,
            altitude: 100.0,
            speed: 10.0,
            // Chosen because degrees and radians differ most visibly here: due east should
            // put the entire 10 m/s on the east channel and nothing on the north one.
            bearing: 90.0,
            qw: 1.0,
            ..Default::default()
        };

        let ukf = initialize_ukf(&rec, UkfConfig::default()).unwrap();
        let ekf = initialize_ekf(&rec, EkfConfig::default()).unwrap();
        let eskf = initialize_eskf(&rec, EskfConfig::default()).unwrap();

        for (name, estimate) in [
            ("UKF", ukf.get_estimate()),
            ("EKF", ekf.get_estimate()),
            ("ESKF", eskf.get_estimate()),
        ] {
            assert_approx_eq!(estimate[3], 0.0, 1e-12); // northward velocity
            assert_approx_eq!(estimate[4], 10.0, 1e-12); // eastward velocity
            assert!(
                estimate[3].abs() < 1e-12,
                "{name} seeded {:.3} m/s of northward velocity from a due-east ground track, \
                 which is the un-converted-degrees signature",
                estimate[3]
            );
        }
    }

    /// All three filter initialisers must seed the attitude the record's quaternion describes.
    ///
    /// Two defects at once: `roll`/`pitch`/`yaw` are radians but were passed with
    /// `in_degrees: true`, so every filter constructor scaled them by pi/180; and those
    /// columns are not nalgebra's XYZ sequence in the first place -- see
    /// [`TestDataRecord::attitude`]. The assertion is the convention-free angle between the
    /// seeded rotation and the record's own, so it catches either failure. Under the old
    /// code this angle was 1.21 rad.
    #[test]
    fn test_initializers_seed_attitude_from_quaternion() {
        let quaternion = nalgebra::UnitQuaternion::from_euler_angles(0.4, -0.3, 1.2);
        let rec = TestDataRecord {
            time: Utc::now(),
            horizontal_accuracy: 5.0,
            vertical_accuracy: 2.0,
            speed_accuracy: 1.0,
            latitude: 37.0,
            longitude: -122.0,
            altitude: 100.0,
            speed: 10.0,
            bearing: 45.0,
            qw: quaternion.w,
            qx: quaternion.i,
            qy: quaternion.j,
            qz: quaternion.k,
            // Deliberately disagreeing with the quaternion, the way a real Sensor Logger row
            // does: these are the first sample of `core/tests/test_data.csv`.
            roll: 0.163,
            pitch: -1.340,
            yaw: 0.179,
            ..Default::default()
        };

        let expected = rec.attitude();
        let ukf = initialize_ukf(&rec, UkfConfig::default()).unwrap();
        let ekf = initialize_ekf(&rec, EkfConfig::default()).unwrap();
        let eskf = initialize_eskf(&rec, EskfConfig::default()).unwrap();

        for (name, estimate) in [
            ("UKF", ukf.get_estimate()),
            ("EKF", ekf.get_estimate()),
            ("ESKF", eskf.get_estimate()),
        ] {
            // Reconstructing rather than comparing angles elementwise: each elementary
            // rotation is 2pi-periodic, so the rotation is the quantity the three agree on
            // whatever branch a filter reports its Euler angles on.
            let seeded =
                nalgebra::Rotation3::from_euler_angles(estimate[6], estimate[7], estimate[8]);
            let error_angle = (seeded.inverse() * expected).angle();
            assert!(
                error_angle < 1e-9,
                "{name} seeded an attitude {error_angle:.6} rad away from the record's \
                 quaternion"
            );
        }
    }

    #[test]
    fn test_initialize_ukf_with_nan_angles() {
        let rec = TestDataRecord {
            time: Utc::now(),
            horizontal_accuracy: 5.0,
            vertical_accuracy: 2.0,
            speed_accuracy: 1.0,
            latitude: 37.0,
            longitude: -122.0,
            altitude: 100.0,
            speed: 10.0,
            bearing: 45.0,
            roll: f64::NAN,
            pitch: f64::NAN,
            yaw: f64::NAN,
            ..Default::default()
        };
        let ukf = initialize_ukf(&rec, UkfConfig::default()).unwrap();
        let estimate = ukf.get_estimate();
        // Should default NaN angles to 0.0
        assert!(estimate[6].abs() < 1e-6); // roll
        assert!(estimate[7].abs() < 1e-6); // pitch
        assert!(estimate[8].abs() < 1e-6); // yaw
    }

    #[test]
    fn test_initialize_ukf_with_custom_biases() {
        let rec = TestDataRecord {
            time: Utc::now(),
            horizontal_accuracy: 5.0,
            vertical_accuracy: 2.0,
            speed_accuracy: 1.0,
            latitude: 37.0,
            longitude: -122.0,
            altitude: 100.0,
            speed: 10.0,
            bearing: 45.0,
            roll: 0.0,
            pitch: 0.0,
            yaw: 0.0,
            ..Default::default()
        };
        let ukf = initialize_ukf(
            &rec,
            UkfConfig {
                attitude_covariance: Some(vec![1e-4, 2e-4, 3e-4]),
                imu_biases: Some(vec![0.01, 0.02, 0.03, 0.001, 0.002, 0.003]),
                imu_biases_covariance: Some(vec![1e-5; 6]),
                ..Default::default()
            },
        )
        .unwrap();
        let estimate = ukf.get_estimate();
        assert_eq!(estimate.len(), 15);
    }

    /// The three filter constructors seed the same IMU bias prior, and it is the grade's.
    ///
    /// This test exists because they did not, twice over, and neither disagreement was
    /// visible from any one of them.
    ///
    /// * **The estimate (#392).** `initialize_ukf` seeded `vec![1e-3; 6]` -- the *covariance*
    ///   on the line above, copied down -- where the other two seeded zero. A UKF built
    ///   without explicit biases opened by asserting a 1 mrad/s rate bias on every gyroscope
    ///   axis and subtracting it from every sample.
    /// * **The covariance (#393).** The UKF and EKF opened at `1e-3` for all six entries
    ///   while the ESKF used `1e-6`/`1e-8` -- five orders of magnitude apart on the gyro
    ///   block, and none of the three a model of any hardware. `1e-3` is a gyro-bias sigma of
    ///   1.81 deg/s against the 0.028 deg/s a `Consumer`-grade part actually has.
    ///
    /// The second one cost more than accuracy. It made "the UKF reads 72.68 deg of yaw where
    /// the ESKF reads 0.255 on the identical stream" -- the comparison #371 was diagnosed
    /// from, and which this crate's own docs asserted -- not a comparison at all: the two
    /// filters were never handed the same prior. Pinning the three against each other *and*
    /// against the grade they claim to model is what closes that class, which is why this
    /// asserts the derivation rather than a transcribed constant.
    #[test]
    fn the_three_constructors_seed_one_bias_prior_and_it_is_the_grades() {
        /// Where in a 15-state covariance diagonal the six bias entries live.
        const BIAS_STATE_INDICES: std::ops::Range<usize> = 9..15;

        let record = TestDataRecord {
            time: Utc::now(),
            horizontal_accuracy: 5.0,
            vertical_accuracy: 2.0,
            speed_accuracy: 1.0,
            latitude: 37.0,
            longitude: -122.0,
            altitude: 100.0,
            speed: 10.0,
            bearing: 45.0,
            ..Default::default()
        };

        // Every grade, not just the default: the point is that the constructors read the
        // grade, and a hard-coded constant that happened to match `Consumer` would pass a
        // single-grade test.
        for grade in [
            crate::IMUQuality::Consumer,
            crate::IMUQuality::Industrial,
            crate::IMUQuality::Tactical,
            crate::IMUQuality::Navigation,
            crate::IMUQuality::Strategic,
        ] {
            let expected = grade.initial_bias_covariance();

            let ukf = initialize_ukf(
                &record,
                UkfConfig {
                    imu_quality: grade,
                    ..Default::default()
                },
            )
            .unwrap();
            let ekf = initialize_ekf(
                &record,
                EkfConfig {
                    imu_quality: grade,
                    ..Default::default()
                },
            )
            .unwrap();
            let eskf = initialize_eskf(
                &record,
                EskfConfig {
                    imu_quality: grade,
                    ..Default::default()
                },
            )
            .unwrap();

            for (name, filter) in [
                ("UKF", ukf.get_certainty()),
                ("EKF", ekf.get_certainty()),
                ("ESKF", eskf.get_certainty()),
            ] {
                assert_eq!(
                    filter.nrows(),
                    15,
                    "{name} at {grade:?} is not carrying bias states at all"
                );
                for (offset, index) in BIAS_STATE_INDICES.enumerate() {
                    assert_approx_eq!(filter[(index, index)], expected[offset], 1e-18);
                }
            }

            // And the estimate is zero -- an uncertainty about the bias, never a claim about
            // it. `get_estimate` is the UKF's and EKF's mean state; the ESKF carries its
            // biases outside the error state, which `get_estimate` appends in the same slots.
            for (name, estimate) in [
                ("UKF", ukf.get_estimate()),
                ("EKF", ekf.get_estimate()),
                ("ESKF", eskf.get_estimate()),
            ] {
                for index in BIAS_STATE_INDICES {
                    assert_eq!(
                        estimate[index], 0.0,
                        "{name} at {grade:?} seeds a bias *estimate* at state {index}; that \
                         is a claim about the hardware, not an uncertainty about it (#392)"
                    );
                }
            }
        }

        // And the covariance is read **independently of the estimate** by all three.
        //
        // This is the case the first version of this test did not cover, because it only ever
        // built configs with `..Default::default()`. `initialize_ekf` read
        // `imu_biases_covariance` only inside its `if let Some(imu_biases)` arm, so a caller
        // who set a covariance and left the estimate at `None` -- a custom uncertainty about a
        // zero bias, the ordinary case -- had it silently replaced by the grade default. The
        // UKF and ESKF both honoured it. That is #392's defect surviving on the one
        // constructor it was not fixed on, and a test that pins the three against each other
        // has to exercise the setting independently or it pins nothing.
        let custom = vec![7e-4, 7e-4, 7e-4, 9e-8, 9e-8, 9e-8];
        let ukf = initialize_ukf(
            &record,
            UkfConfig {
                imu_biases_covariance: Some(custom.clone()),
                ..Default::default()
            },
        )
        .unwrap();
        let ekf = initialize_ekf(
            &record,
            EkfConfig {
                imu_biases_covariance: Some(custom.clone()),
                ..Default::default()
            },
        )
        .unwrap();
        let eskf = initialize_eskf(
            &record,
            EskfConfig {
                imu_biases_covariance: Some(custom.clone()),
                ..Default::default()
            },
        )
        .unwrap();
        for (name, filter) in [
            ("UKF", ukf.get_certainty()),
            ("EKF", ekf.get_certainty()),
            ("ESKF", eskf.get_certainty()),
        ] {
            for (offset, index) in BIAS_STATE_INDICES.enumerate() {
                assert_approx_eq!(filter[(index, index)], custom[offset], 1e-18);
            }
            let _ = name;
        }

        // A malformed covariance is rejected by all three, not quietly absorbed. The UKF had
        // no length check at all, so with `other_states` making the whole diagonal come out
        // the right length a short vector would have slid an extra-state variance into a bias
        // slot.
        for (name, result) in [
            (
                "UKF",
                initialize_ukf(
                    &record,
                    UkfConfig {
                        imu_biases_covariance: Some(vec![1e-4; 5]),
                        ..Default::default()
                    },
                )
                .err(),
            ),
            (
                "EKF",
                initialize_ekf(
                    &record,
                    EkfConfig {
                        imu_biases_covariance: Some(vec![1e-4; 5]),
                        ..Default::default()
                    },
                )
                .err(),
            ),
            (
                "ESKF",
                initialize_eskf(
                    &record,
                    EskfConfig {
                        imu_biases_covariance: Some(vec![1e-4; 5]),
                        ..Default::default()
                    },
                )
                .err(),
            ),
        ] {
            assert!(
                result.is_some(),
                "{name} accepted a five-element bias covariance instead of rejecting it"
            );
        }

        // The derivation itself, so the constants above cannot all drift together: a
        // consumer gyro's 100 deg/h is 0.0278 deg/s, and the variance is its square.
        let consumer = crate::IMUQuality::Consumer.initial_bias_covariance();
        let gyro_sigma_dps = consumer[3].sqrt().to_degrees();
        assert_approx_eq!(gyro_sigma_dps, 100.0 / 3600.0, 1e-9);
        assert_approx_eq!(
            consumer[0].sqrt(),
            crate::IMUQuality::Consumer.accel_bias_instability_mps2(),
            1e-12
        );
    }

    #[test]
    fn test_initialize_ukf_with_custom_process_noise() {
        let rec = TestDataRecord {
            time: Utc::now(),
            horizontal_accuracy: 5.0,
            vertical_accuracy: 2.0,
            speed_accuracy: 1.0,
            latitude: 37.0,
            longitude: -122.0,
            altitude: 100.0,
            speed: 10.0,
            bearing: 45.0,
            roll: 0.0,
            pitch: 0.0,
            yaw: 0.0,
            ..Default::default()
        };
        let custom_noise = vec![1e-5; 15];
        let ukf = initialize_ukf(
            &rec,
            UkfConfig {
                process_noise_diagonal: Some(custom_noise),
                ..Default::default()
            },
        )
        .unwrap();
        assert!(!ukf.get_estimate().is_empty());
    }
    #[test]
    fn test_health_limits_default() {
        let limits = HealthLimits::default();
        assert!(limits.lat_rad.0 < 0.0);
        assert!(limits.lat_rad.1 > 0.0);
        assert!(limits.speed_mps_max > 0.0);
        assert!(limits.cov_diag_max > 0.0);
    }

    /// Fixed reference instant plus an offset, so every timeout assertion below is
    /// exact instead of racing the scheduler. See `ExecutionMonitor::new_at` and #284.
    fn at(base: std::time::Instant, millis: u64) -> std::time::Instant {
        base + std::time::Duration::from_millis(millis)
    }

    #[test]
    fn test_execution_monitor_no_progress_timeout() {
        let t0 = std::time::Instant::now();
        let limits = ExecutionLimits {
            max_wall_clock_ratio: 0.0,
            max_wall_clock_s: 0.0,
            max_no_progress_s: 0.010,
        };
        let monitor = ExecutionMonitor::new_at(&limits, 1.0, t0);

        let result = monitor.check_at("test", at(t0, 20));
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("no progress"));
    }

    /// The no-progress timeout must fire strictly after the limit, not at or before it.
    #[test]
    fn test_execution_monitor_no_progress_boundary() {
        let t0 = std::time::Instant::now();
        let limits = ExecutionLimits {
            max_wall_clock_ratio: 0.0,
            max_wall_clock_s: 0.0,
            max_no_progress_s: 0.100,
        };
        let monitor = ExecutionMonitor::new_at(&limits, 1.0, t0);

        assert!(
            monitor.check_at("test", at(t0, 99)).is_ok(),
            "just under the limit"
        );
        assert!(
            monitor.check_at("test", at(t0, 100)).is_ok(),
            "exactly at the limit"
        );
        assert!(
            monitor.check_at("test", at(t0, 101)).is_err(),
            "just over the limit"
        );
    }

    /// `mark_progress` must restart the no-progress window.
    #[test]
    fn test_execution_monitor_progress_resets_no_progress_window() {
        let t0 = std::time::Instant::now();
        let limits = ExecutionLimits {
            max_wall_clock_ratio: 0.0,
            max_wall_clock_s: 0.0,
            max_no_progress_s: 0.100,
        };
        let mut monitor = ExecutionMonitor::new_at(&limits, 1.0, t0);

        assert!(monitor.check_at("test", at(t0, 90)).is_ok());
        monitor.mark_progress_at(at(t0, 90));
        // 150 ms since start, but only 60 ms since the last progress.
        assert!(monitor.check_at("test", at(t0, 150)).is_ok());
        // 210 ms since start and 120 ms since progress -- now it must fire.
        assert!(monitor.check_at("test", at(t0, 210)).is_err());
    }

    #[test]
    fn test_execution_monitor_wall_clock_timeout() {
        let t0 = std::time::Instant::now();
        let limits = ExecutionLimits {
            max_wall_clock_ratio: 0.0,
            max_wall_clock_s: 0.010,
            max_no_progress_s: 0.0,
        };
        let monitor = ExecutionMonitor::new_at(&limits, 1.0, t0);

        let result = monitor.check_at("test", at(t0, 20));
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("wall-clock limit"));
    }

    /// The wall-clock timeout is absolute: `mark_progress` must not defer it.
    #[test]
    fn test_execution_monitor_wall_clock_boundary_ignores_progress() {
        let t0 = std::time::Instant::now();
        let limits = ExecutionLimits {
            max_wall_clock_ratio: 0.0,
            max_wall_clock_s: 0.100,
            max_no_progress_s: 0.0,
        };
        let mut monitor = ExecutionMonitor::new_at(&limits, 1.0, t0);

        assert!(
            monitor.check_at("test", at(t0, 100)).is_ok(),
            "exactly at the limit"
        );
        monitor.mark_progress_at(at(t0, 100));
        assert!(
            monitor.check_at("test", at(t0, 101)).is_err(),
            "progress must not extend the absolute wall-clock budget"
        );
    }

    /// Steady work with regular progress marks must never trip either timeout, even
    /// once total elapsed time exceeds the no-progress budget.
    ///
    /// This is the test #284 was about. It used to sleep 10 ms per iteration and assert
    /// a 50 ms no-progress budget had not been exceeded -- an upper bound on
    /// `thread::sleep`, which the standard library never promises. It failed on loaded
    /// CI runners. Driving the clock directly tests the same property deterministically,
    /// and lets the loop run long enough to be meaningful.
    #[test]
    fn test_execution_monitor_successful_execution_with_progress() {
        let t0 = std::time::Instant::now();
        let limits = ExecutionLimits {
            max_wall_clock_ratio: 0.0,
            max_wall_clock_s: 10.0,
            max_no_progress_s: 0.050,
        };
        let mut monitor = ExecutionMonitor::new_at(&limits, 1.0, t0);

        // 200 iterations at 40 ms each: 8 s total, comfortably past the 50 ms
        // no-progress budget, but each individual gap stays under it.
        for step in 1..=200 {
            let now = at(t0, step * 40);
            assert!(
                monitor.check_at("test", now).is_ok(),
                "step {step} tripped a timeout despite regular progress"
            );
            monitor.mark_progress_at(now);
        }
    }

    #[test]
    fn test_execution_monitor_zero_timeout_disabled() {
        let t0 = std::time::Instant::now();
        let limits = ExecutionLimits {
            max_wall_clock_ratio: 0.0,
            max_wall_clock_s: 0.0,
            max_no_progress_s: 0.0,
        };
        let monitor = ExecutionMonitor::new_at(&limits, 1.0, t0);

        // All timeouts disabled: an hour of no progress is still fine.
        assert!(monitor.check_at("test", at(t0, 3_600_000)).is_ok());
    }

    #[test]
    fn test_execution_monitor_negative_timeout_disabled() {
        let t0 = std::time::Instant::now();
        let limits = ExecutionLimits {
            max_wall_clock_ratio: -1.0,
            max_wall_clock_s: -1.0,
            max_no_progress_s: -1.0,
        };
        let monitor = ExecutionMonitor::new_at(&limits, 1.0, t0);

        assert!(monitor.check_at("test", at(t0, 3_600_000)).is_ok());
    }

    /// `max_wall_clock_ratio` scales with simulated duration, and the tighter of the
    /// ratio and the absolute limit wins.
    #[test]
    fn test_execution_monitor_ratio_and_absolute_limits_combine() {
        let t0 = std::time::Instant::now();
        // ratio: 0.25 * 2 s = 500 ms; absolute: 200 ms. The absolute one is tighter.
        let limits = ExecutionLimits {
            max_wall_clock_ratio: 0.25,
            max_wall_clock_s: 0.200,
            max_no_progress_s: 0.0,
        };
        let monitor = ExecutionMonitor::new_at(&limits, 2.0, t0);
        assert!(monitor.check_at("test", at(t0, 200)).is_ok());
        assert!(monitor.check_at("test", at(t0, 201)).is_err());

        // ratio: 0.25 * 2 s = 500 ms; absolute: 5 s. Now the ratio is tighter.
        let limits = ExecutionLimits {
            max_wall_clock_ratio: 0.25,
            max_wall_clock_s: 5.0,
            max_no_progress_s: 0.0,
        };
        let monitor = ExecutionMonitor::new_at(&limits, 2.0, t0);
        assert!(monitor.check_at("test", at(t0, 500)).is_ok());
        assert!(monitor.check_at("test", at(t0, 501)).is_err());
    }

    #[test]
    fn test_health_monitor_check_valid_state() {
        let limits = HealthLimits::default();
        let mut monitor = HealthMonitor::new(limits);

        let state = vec![
            0.5, 0.5, 100.0, // lat, lon, alt (radians, radians, meters)
            10.0, 5.0, 0.0, // vn, ve, vd
            0.0, 0.0, 0.0, // roll, pitch, yaw
            0.0, 0.0, 0.0, // acc biases
            0.0, 0.0, 0.0, // gyro biases
        ];
        let cov = DMatrix::from_diagonal(&DVector::from_vec(vec![1e-6; 15]));

        let result = monitor.check(&state, &cov, None);
        assert!(result.is_ok());
    }

    /// A state too short to read returns an error rather than panicking.
    ///
    /// `check` takes `&[f64]` -- it has to, since a state is 9, 15 or 16 wide depending on the
    /// filter -- and then reads `x[0..=5]`. Nothing made that a precondition: clippy does not
    /// flag slice indexing, so the crate's deny-level `panic`/`unwrap`/`expect` policy did not
    /// reach it, and the panic sat in a `pub fn`.
    ///
    /// Five elements is the interesting length: it clears position and the first two velocity
    /// components, so it reaches `x[5]` and no earlier read.
    #[test]
    fn a_state_too_short_to_monitor_is_an_error_not_a_panic() {
        let mut monitor = HealthMonitor::new(HealthLimits::default());
        let cov = DMatrix::from_diagonal(&DVector::from_vec(vec![1e-6; 5]));

        for width in 0..HealthMonitor::MINIMUM_MONITORED_STATE {
            let state = vec![0.0; width];
            let result = monitor.check(&state, &cov, None);
            assert!(
                result.is_err(),
                "a {width}-element state was accepted; it would have panicked on x[{}]",
                HealthMonitor::MINIMUM_MONITORED_STATE - 1
            );
        }

        // ...and the narrowest acceptable state still works.
        let state = vec![0.5, 0.5, 100.0, 10.0, 5.0, 0.0];
        assert!(monitor.check(&state, &cov, None).is_ok());
    }

    #[test]
    fn test_health_monitor_check_non_finite_state() {
        let limits = HealthLimits::default();
        let mut monitor = HealthMonitor::new(limits);

        let state = vec![
            f64::NAN,
            0.5,
            100.0,
            10.0,
            5.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ];
        let cov = DMatrix::from_diagonal(&DVector::from_vec(vec![1e-6; 15]));

        let result = monitor.check(&state, &cov, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_health_monitor_check_non_finite_covariance() {
        let limits = HealthLimits::default();
        let mut monitor = HealthMonitor::new(limits);

        let state = vec![
            0.5, 0.5, 100.0, 10.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let mut cov = DMatrix::from_diagonal(&DVector::from_vec(vec![1e-6; 15]));
        cov[(0, 0)] = f64::NAN;

        let result = monitor.check(&state, &cov, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_health_monitor_check_latitude_out_of_range() {
        let limits = HealthLimits::default();
        let mut monitor = HealthMonitor::new(limits);

        let state = vec![
            5.0, 0.5, 100.0, 10.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ]; // lat > PI/2
        let cov = DMatrix::from_diagonal(&DVector::from_vec(vec![1e-6; 15]));

        let result = monitor.check(&state, &cov, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_health_monitor_check_longitude_out_of_range() {
        let limits = HealthLimits::default();
        let mut monitor = HealthMonitor::new(limits);

        let state = vec![
            0.5, 5.0, 100.0, 10.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ]; // lon > PI
        let cov = DMatrix::from_diagonal(&DVector::from_vec(vec![1e-6; 15]));

        let result = monitor.check(&state, &cov, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_health_monitor_check_altitude_out_of_range() {
        let limits = HealthLimits {
            alt_m: (-100.0, 10000.0),
            ..Default::default()
        };
        let mut monitor = HealthMonitor::new(limits);

        let state = vec![
            0.5, 0.5, 20000.0, 10.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let cov = DMatrix::from_diagonal(&DVector::from_vec(vec![1e-6; 15]));

        let result = monitor.check(&state, &cov, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_health_monitor_check_negative_variance() {
        let limits = HealthLimits::default();
        let mut monitor = HealthMonitor::new(limits);

        let state = vec![
            0.5, 0.5, 100.0, 10.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let mut cov = DMatrix::from_diagonal(&DVector::from_vec(vec![1e-6; 15]));
        cov[(2, 2)] = -1.0; // negative variance

        let result = monitor.check(&state, &cov, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_health_monitor_check_variance_too_large() {
        let limits = HealthLimits::default();
        let mut monitor = HealthMonitor::new(limits);

        let state = vec![
            0.5, 0.5, 100.0, 10.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let mut cov = DMatrix::from_diagonal(&DVector::from_vec(vec![1e-6; 15]));
        cov[(3, 3)] = 1e20; // variance too large

        let result = monitor.check(&state, &cov, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_health_monitor_check_speed_within_limit() {
        let limits = HealthLimits {
            speed_mps_max: 50.0,
            ..Default::default()
        };
        let mut monitor = HealthMonitor::new(limits);

        // vn=10, ve=5, vd=0 -> speed ~11.18 m/s, under the 50 m/s limit.
        let state = vec![
            0.5, 0.5, 100.0, 10.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let cov = DMatrix::from_diagonal(&DVector::from_vec(vec![1e-6; 15]));

        let result = monitor.check(&state, &cov, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_health_monitor_check_speed_exceeded() {
        let limits = HealthLimits {
            speed_mps_max: 50.0,
            ..Default::default()
        };
        let mut monitor = HealthMonitor::new(limits);

        // vn=100, ve=0, vd=0 -> 100 m/s, over the 50 m/s limit.
        let state = vec![
            0.5, 0.5, 100.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let cov = DMatrix::from_diagonal(&DVector::from_vec(vec![1e-6; 15]));

        let result = monitor.check(&state, &cov, None);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Speed exceeded"));
    }

    /// A naive `vn*vn + ve*ve + vd*vd` sum of squares overflows to infinity for a merely
    /// large (but finite) component; `f64::MAX.hypot(0.0)` does not, and the speed check
    /// must not let an overflowed intermediate wave a divergent-but-finite state through.
    #[test]
    fn test_health_monitor_check_speed_overflow_does_not_bypass_limit() {
        let limits = HealthLimits {
            speed_mps_max: 50.0,
            ..Default::default()
        };
        let mut monitor = HealthMonitor::new(limits);

        let state = vec![
            0.5,
            0.5,
            100.0,
            f64::MAX / 2.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ];
        let cov = DMatrix::from_diagonal(&DVector::from_vec(vec![1e-6; 15]));

        let result = monitor.check(&state, &cov, None);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Speed exceeded"));
    }

    #[test]
    fn test_health_monitor_check_nis_invalid() {
        let limits = HealthLimits::default();
        let mut monitor = HealthMonitor::new(limits);

        let state = vec![
            0.5, 0.5, 100.0, 10.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let cov = DMatrix::from_diagonal(&DVector::from_vec(vec![1e-6; 15]));

        let result = monitor.check(&state, &cov, Some(f64::NAN));
        assert!(result.is_err());
    }

    #[test]
    fn test_health_monitor_check_nis_exceeds_threshold() {
        let limits = HealthLimits {
            nis_pos_max: 10.0,
            nis_pos_consec_fail: 3,
            ..Default::default()
        };
        let mut monitor = HealthMonitor::new(limits);

        let state = vec![
            0.5, 0.5, 100.0, 10.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let cov = DMatrix::from_diagonal(&DVector::from_vec(vec![1e-6; 15]));

        // First two failures should be ok
        assert!(monitor.check(&state, &cov, Some(15.0)).is_ok());
        assert!(monitor.check(&state, &cov, Some(15.0)).is_ok());
        // Third consecutive failure should error
        let result = monitor.check(&state, &cov, Some(15.0));
        assert!(result.is_err());
    }

    #[test]
    fn test_health_monitor_check_nis_reset_on_pass() {
        let limits = HealthLimits {
            nis_pos_max: 10.0,
            nis_pos_consec_fail: 3,
            ..Default::default()
        };
        let mut monitor = HealthMonitor::new(limits);

        let state = vec![
            0.5, 0.5, 100.0, 10.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let cov = DMatrix::from_diagonal(&DVector::from_vec(vec![1e-6; 15]));

        // First failure
        assert!(monitor.check(&state, &cov, Some(15.0)).is_ok());
        // Pass resets counter
        assert!(monitor.check(&state, &cov, Some(5.0)).is_ok());
        // New failures should not accumulate with previous
        assert!(monitor.check(&state, &cov, Some(15.0)).is_ok());
        assert!(monitor.check(&state, &cov, Some(15.0)).is_ok());
    }

    #[test]
    fn test_build_scheduler_passthrough() {
        let args = SchedulerArgs {
            sched: SchedKind::Passthrough,
            interval_s: 1.0,
            phase_s: 0.0,
            on_s: 10.0,
            off_s: 10.0,
            duty_phase_s: 0.0,
        };
        let scheduler = build_scheduler(&args);
        matches!(scheduler, MeasurementScheduler::PassThrough);
    }

    #[test]
    fn test_build_scheduler_fixed() {
        let args = SchedulerArgs {
            sched: SchedKind::Fixed,
            interval_s: 2.5,
            phase_s: 0.5,
            on_s: 10.0,
            off_s: 10.0,
            duty_phase_s: 0.0,
        };
        let scheduler = build_scheduler(&args);
        if let MeasurementScheduler::FixedInterval {
            interval_s,
            phase_s,
        } = scheduler
        {
            assert_eq!(interval_s, 2.5);
            assert_eq!(phase_s, 0.5);
        } else {
            panic!("Expected FixedInterval scheduler");
        }
    }

    #[test]
    fn test_build_scheduler_duty() {
        let args = SchedulerArgs {
            sched: SchedKind::Duty,
            interval_s: 1.0,
            phase_s: 0.0,
            on_s: 15.0,
            off_s: 5.0,
            duty_phase_s: 2.0,
        };
        let scheduler = build_scheduler(&args);
        if let MeasurementScheduler::DutyCycle {
            on_s,
            off_s,
            start_phase_s,
        } = scheduler
        {
            assert_eq!(on_s, 15.0);
            assert_eq!(off_s, 5.0);
            assert_eq!(start_phase_s, 2.0);
        } else {
            panic!("Expected DutyCycle scheduler");
        }
    }

    #[test]
    fn test_build_fault_none() {
        let args = FaultArgs {
            fault: FaultKind::None,
            rho_pos: 0.99,
            sigma_pos_m: 3.0,
            rho_vel: 0.95,
            sigma_vel_mps: 0.3,
            r_scale: 5.0,
            drift_n_mps: 0.02,
            drift_e_mps: 0.0,
            q_bias: 1e-6,
            rotate_omega_rps: 0.0,
            hijack_offset_n_m: 50.0,
            hijack_offset_e_m: 0.0,
            hijack_start_s: 120.0,
            hijack_duration_s: 60.0,
        };
        let fault = build_fault(&args);
        matches!(fault, GnssFaultModel::None);
    }

    #[test]
    fn test_build_fault_degraded() {
        let args = FaultArgs {
            fault: FaultKind::Degraded,
            rho_pos: 0.98,
            sigma_pos_m: 5.0,
            rho_vel: 0.93,
            sigma_vel_mps: 0.5,
            r_scale: 10.0,
            drift_n_mps: 0.02,
            drift_e_mps: 0.0,
            q_bias: 1e-6,
            rotate_omega_rps: 0.0,
            hijack_offset_n_m: 50.0,
            hijack_offset_e_m: 0.0,
            hijack_start_s: 120.0,
            hijack_duration_s: 60.0,
        };
        let fault = build_fault(&args);
        if let GnssFaultModel::Degraded {
            rho_pos,
            sigma_pos_m,
            rho_vel,
            sigma_vel_mps,
            r_scale,
        } = fault
        {
            assert_eq!(rho_pos, 0.98);
            assert_eq!(sigma_pos_m, 5.0);
            assert_eq!(rho_vel, 0.93);
            assert_eq!(sigma_vel_mps, 0.5);
            assert_eq!(r_scale, 10.0);
        } else {
            panic!("Expected Degraded fault model");
        }
    }

    #[test]
    fn test_build_fault_slowbias() {
        let args = FaultArgs {
            fault: FaultKind::Slowbias,
            rho_pos: 0.99,
            sigma_pos_m: 3.0,
            rho_vel: 0.95,
            sigma_vel_mps: 0.3,
            r_scale: 5.0,
            drift_n_mps: 0.05,
            drift_e_mps: 0.02,
            q_bias: 1e-5,
            rotate_omega_rps: 0.01,
            hijack_offset_n_m: 50.0,
            hijack_offset_e_m: 0.0,
            hijack_start_s: 120.0,
            hijack_duration_s: 60.0,
        };
        let fault = build_fault(&args);
        if let GnssFaultModel::SlowBias {
            drift_n_mps,
            drift_e_mps,
            q_bias,
            rotate_omega_rps,
        } = fault
        {
            assert_eq!(drift_n_mps, 0.05);
            assert_eq!(drift_e_mps, 0.02);
            assert_eq!(q_bias, 1e-5);
            assert_eq!(rotate_omega_rps, 0.01);
        } else {
            panic!("Expected SlowBias fault model");
        }
    }

    #[test]
    fn test_build_fault_hijack() {
        let args = FaultArgs {
            fault: FaultKind::Hijack,
            rho_pos: 0.99,
            sigma_pos_m: 3.0,
            rho_vel: 0.95,
            sigma_vel_mps: 0.3,
            r_scale: 5.0,
            drift_n_mps: 0.02,
            drift_e_mps: 0.0,
            q_bias: 1e-6,
            rotate_omega_rps: 0.0,
            hijack_offset_n_m: 100.0,
            hijack_offset_e_m: 50.0,
            hijack_start_s: 180.0,
            hijack_duration_s: 90.0,
        };
        let fault = build_fault(&args);
        if let GnssFaultModel::Hijack {
            offset_n_m,
            offset_e_m,
            start_s,
            duration_s,
        } = fault
        {
            assert_eq!(offset_n_m, 100.0);
            assert_eq!(offset_e_m, 50.0);
            assert_eq!(start_s, 180.0);
            assert_eq!(duration_s, 90.0);
        } else {
            panic!("Expected Hijack fault model");
        }
    }

    #[test]
    fn test_ned_covariance() {
        let cov = NEDCovariance {
            latitude_cov: 1e-6,
            longitude_cov: 1e-6,
            altitude_cov: 1e-4,
            velocity_n_cov: 1e-3,
            velocity_e_cov: 1e-3,
            velocity_v_cov: 1e-3,
            roll_cov: 1e-5,
            pitch_cov: 1e-5,
            yaw_cov: 1e-5,
            acc_bias_x_cov: 1e-6,
            acc_bias_y_cov: 1e-6,
            acc_bias_z_cov: 1e-6,
            gyro_bias_x_cov: 1e-8,
            gyro_bias_y_cov: 1e-8,
            gyro_bias_z_cov: 1e-8,
        };
        assert_eq!(cov.latitude_cov, 1e-6);
        assert_eq!(cov.gyro_bias_z_cov, 1e-8);
    }

    #[test]
    fn test_navigation_result_csv_roundtrip_with_nan() {
        let nav = NavigationResult {
            latitude: 37.0,
            longitude: -122.0,
            altitude: 100.0,
            latitude_cov: f64::NAN,
            longitude_cov: f64::NAN,
            ..Default::default()
        };

        let temp_file = std::env::temp_dir().join("test_nav_nan.csv");
        NavigationResult::to_csv(std::slice::from_ref(&nav), &temp_file).unwrap();

        let read = NavigationResult::from_csv(&temp_file).unwrap();
        assert_eq!(read.len(), 1);
        assert_eq!(read[0].latitude, 37.0);
        assert!(read[0].latitude_cov.is_nan());

        let _ = std::fs::remove_file(&temp_file);
    }

    #[test]
    fn test_navigation_result_from_strapdown_state_with_rotation() {
        let timestamp = Utc::now();
        let state = StrapdownState {
            latitude: 0.5,  // radians
            longitude: 1.0, // radians
            altitude: 500.0,
            velocity_north: 10.0,
            velocity_east: 5.0,
            velocity_vertical: -1.0,
            attitude: nalgebra::Rotation3::from_euler_angles(0.1, 0.2, 0.3),
            ..Default::default()
        };

        let nav = NavigationResult::from((&timestamp, &state));
        assert_eq!(nav.timestamp, timestamp);
        assert!((nav.latitude - 0.5_f64.to_degrees()).abs() < 1e-6);
        assert!((nav.longitude - 1.0_f64.to_degrees()).abs() < 1e-6);
        assert_eq!(nav.altitude, 500.0);
        assert!(nav.latitude_cov.is_nan());
        assert_eq!(nav.acc_bias_x, 0.0);
    }

    /// The two horizontal entries must be one physical quantity, in radians.
    ///
    /// Asserting the literals back is what let #308 live: `1e-6, 1e-6, 1e-4` is a perfectly
    /// self-consistent set of numbers and a perfectly inconsistent set of *claims*, and a test
    /// that reads the array back cannot tell the difference. So this converts the horizontal
    /// pair back to metres through the inverse of the conversion that built them.
    ///
    /// # What this cannot do
    ///
    /// It cannot check [`METERS_TO_RADIANS`] itself, because it divides by the same constant
    /// the array multiplied by: redefine that constant as [`METERS_TO_DEGREES`] -- restoring
    /// exactly the 57.3x error the fix exists to remove -- and both sides move together and
    /// this still passes. The constant is checked independently, against the ellipsoid, in
    /// [`crate::earth`]'s `meters_to_radians_matches_a_wgs84_principal_radius`; without that
    /// test this one is a tautology, and the two are meant to be read as a pair.
    ///
    /// Altitude is deliberately *not* compared against [`POSITION_PROCESS_NOISE_M_PER_ROOT_S`]. The two
    /// are different quantities on purpose -- see [`VERTICAL_POSITION_PROCESS_NOISE_M2_PER_S`] --
    /// and asserting they agree would turn "the vertical channel keeps its historical tuning"
    /// into a test failure rather than the recorded decision it is.
    #[test]
    fn default_process_noise_position_entries_are_one_quantity() {
        assert_eq!(DEFAULT_PROCESS_NOISE_DENSITY.len(), 15);
        let latitude_m = DEFAULT_PROCESS_NOISE_DENSITY[0].sqrt() / METERS_TO_RADIANS;
        let longitude_m = DEFAULT_PROCESS_NOISE_DENSITY[1].sqrt() / METERS_TO_RADIANS;
        assert_approx_eq!(latitude_m, POSITION_PROCESS_NOISE_M_PER_ROOT_S, 1e-12);
        assert_approx_eq!(longitude_m, POSITION_PROCESS_NOISE_M_PER_ROOT_S, 1e-12);
        // Altitude is in metres already and keeps its own constant. Asserted as an identity
        // so that re-tying it to `POSITION_PROCESS_NOISE_M_PER_ROOT_S` -- the tidy-looking change the
        // doc comment argues against -- has to be a deliberate edit here too.
        assert_approx_eq!(
            DEFAULT_PROCESS_NOISE_DENSITY[2],
            VERTICAL_POSITION_PROCESS_NOISE_M2_PER_S,
            1e-18
        );
        // Not a re-assertion of the same arithmetic: this is the bound the doc comment derives
        // the value from, and it is what fails if someone raises the constant back towards the
        // regime where the filter stops filtering. K = q / (q + r) <= 0.1 at r = 3.81 m, the
        // reference recording's reported horizontal 1-sigma, caps q at r / 9.
        let reported_fix_accuracy_m = 3.81;
        let steady_state_gain = POSITION_PROCESS_NOISE_M_PER_ROOT_S
            / (POSITION_PROCESS_NOISE_M_PER_ROOT_S + reported_fix_accuracy_m);
        assert!(
            steady_state_gain <= 0.1,
            "position process noise implies a steady-state gain of {steady_state_gain:.3}; \
             above 0.1 the filter averages fewer than ten fixes and is on its way back to the \
             #308 regime where it lands on each one"
        );
        // The remaining entries are untouched tuning values; spot-check one so a wholesale
        // rewrite of the array does not slip past.
        assert_eq!(DEFAULT_PROCESS_NOISE_DENSITY[3], 1e-3); // velocity
    }

    #[test]
    fn test_run_closed_loop_with_health_limits() {
        let rec = TestDataRecord {
            time: Utc::now(),
            horizontal_accuracy: 5.0,
            vertical_accuracy: 2.0,
            speed_accuracy: 1.0,
            latitude: 37.0,
            longitude: -122.0,
            altitude: 100.0,
            speed: 10.0,
            bearing: 45.0,
            roll: 0.0,
            pitch: 0.0,
            yaw: 0.0,
            acc_x: 0.0,
            acc_y: 0.0,
            acc_z: 9.81,
            gyro_x: 0.0,
            gyro_y: 0.0,
            gyro_z: 0.0,
            ..Default::default()
        };

        // The stub's `acc_z: 9.81` with an all-zero (hence identity) quaternion is ENU
        // specific force, so the filter has to be told ENU: `check_declared_frame` rejects
        // NED here, which is the guard doing its job rather than collateral damage.
        let mut ukf = initialize_ukf(
            &rec,
            UkfConfig {
                is_enu: true,
                ..UkfConfig::default()
            },
        )
        .unwrap();

        let stream = EventStream {
            start_time: rec.time,
            events: vec![Event::Imu {
                dt_s: 0.1,
                imu: IMUData {
                    accel: Vector3::new(0.0, 0.0, 9.81),
                    gyro: Vector3::new(0.0, 0.0, 0.0),
                },
                elapsed_s: 0.0,
            }],
        };

        let health_limits = HealthLimits::default();
        let result = run_closed_loop(&mut ukf, stream, Some(health_limits), None);
        assert!(result.is_ok());
    }

    // ==================== Extended Kalman Filter Tests ====================

    #[test]
    fn test_initialize_ekf_default_9state() {
        let rec = TestDataRecord {
            time: Utc::now(),
            horizontal_accuracy: 5.0,
            vertical_accuracy: 2.0,
            speed_accuracy: 1.0,
            latitude: 37.0,
            longitude: -122.0,
            altitude: 100.0,
            speed: 10.0,
            bearing: 45.0, // In degrees
            roll: 0.0,
            pitch: 0.0,
            yaw: 0.0,
            ..Default::default()
        };
        let ekf = initialize_ekf(
            &rec,
            EkfConfig {
                use_biases: false,
                ..EkfConfig::default()
            },
        )
        .unwrap();
        let estimate = ekf.get_estimate();
        assert_eq!(estimate.len(), 9, "9-state EKF should have 9 states");
        // Check velocity decomposition (bearing 45° means equal north/east components)
        assert!((estimate[3] - estimate[4]).abs() < 1.0); // vn ≈ ve for 45° bearing
    }

    #[test]
    fn test_initialize_ekf_default_15state() {
        let rec = TestDataRecord {
            time: Utc::now(),
            horizontal_accuracy: 5.0,
            vertical_accuracy: 2.0,
            speed_accuracy: 1.0,
            latitude: 37.0,
            longitude: -122.0,
            altitude: 100.0,
            speed: 10.0,
            bearing: 45.0,
            roll: 0.0,
            pitch: 0.0,
            yaw: 0.0,
            ..Default::default()
        };
        let ekf = initialize_ekf(&rec, EkfConfig::default()).unwrap();
        let estimate = ekf.get_estimate();
        assert_eq!(estimate.len(), 15, "15-state EKF should have 15 states");
        // Check that biases are initialized to zero by default
        for i in 9..15 {
            assert!(
                estimate[i].abs() < 1e-6,
                "Default biases should be near zero"
            );
        }
    }

    #[test]
    fn test_initialize_ekf_with_nan_angles() {
        let rec = TestDataRecord {
            time: Utc::now(),
            horizontal_accuracy: 5.0,
            vertical_accuracy: 2.0,
            speed_accuracy: 1.0,
            latitude: 37.0,
            longitude: -122.0,
            altitude: 100.0,
            speed: 10.0,
            bearing: 45.0,
            roll: f64::NAN,
            pitch: f64::NAN,
            yaw: f64::NAN,
            ..Default::default()
        };
        let ekf = initialize_ekf(&rec, EkfConfig::default()).unwrap();
        let estimate = ekf.get_estimate();
        // Should default NaN angles to 0.0
        assert!(estimate[6].abs() < 1e-6, "NaN roll should default to 0"); // roll
        assert!(estimate[7].abs() < 1e-6, "NaN pitch should default to 0"); // pitch
        assert!(estimate[8].abs() < 1e-6, "NaN yaw should default to 0"); // yaw
    }

    #[test]
    fn test_initialize_ekf_with_custom_biases() {
        let rec = TestDataRecord {
            time: Utc::now(),
            horizontal_accuracy: 5.0,
            vertical_accuracy: 2.0,
            speed_accuracy: 1.0,
            latitude: 37.0,
            longitude: -122.0,
            altitude: 100.0,
            speed: 10.0,
            bearing: 45.0,
            roll: 0.0,
            pitch: 0.0,
            yaw: 0.0,
            ..Default::default()
        };
        let ekf = initialize_ekf(
            &rec,
            EkfConfig {
                attitude_covariance: Some(vec![1e-4, 2e-4, 3e-4]),
                imu_biases: Some(vec![0.01, 0.02, 0.03, 0.001, 0.002, 0.003]),
                imu_biases_covariance: Some(vec![1e-5; 6]),
                ..EkfConfig::default()
            },
        )
        .unwrap();
        let estimate = ekf.get_estimate();
        assert_eq!(estimate.len(), 15);
        // Check that custom biases are set
        assert!((estimate[9] - 0.01).abs() < 1e-9);
        assert!((estimate[10] - 0.02).abs() < 1e-9);
        assert!((estimate[11] - 0.03).abs() < 1e-9);
    }

    #[test]
    fn test_initialize_ekf_with_custom_process_noise() {
        let rec = TestDataRecord {
            time: Utc::now(),
            horizontal_accuracy: 5.0,
            vertical_accuracy: 2.0,
            speed_accuracy: 1.0,
            latitude: 37.0,
            longitude: -122.0,
            altitude: 100.0,
            speed: 10.0,
            bearing: 45.0,
            roll: 0.0,
            pitch: 0.0,
            yaw: 0.0,
            ..Default::default()
        };
        let custom_noise = vec![1e-7; 15];
        let ekf = initialize_ekf(
            &rec,
            EkfConfig {
                process_noise_diagonal: Some(custom_noise),
                ..EkfConfig::default()
            },
        )
        .unwrap();
        // Verify EKF was created successfully
        assert_eq!(ekf.get_estimate().len(), 15);
    }

    #[cfg(feature = "hdf5")]
    #[test]
    fn test_test_data_record_hdf5_roundtrip() {
        use tempfile::tempdir;

        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test_data.h5");

        // Create test records
        let records = vec![
            TestDataRecord {
                time: DateTime::parse_from_str("2023-01-01 00:00:00+00:00", "%Y-%m-%d %H:%M:%S%z")
                    .unwrap()
                    .with_timezone(&Utc),
                bearing_accuracy: 0.1,
                speed_accuracy: 0.1,
                vertical_accuracy: 0.1,
                horizontal_accuracy: 0.1,
                speed: 1.0,
                bearing: 90.0,
                altitude: 100.0,
                longitude: -122.0,
                latitude: 37.0,
                qz: 0.0,
                qy: 0.0,
                qx: 0.0,
                qw: 1.0,
                roll: 0.0,
                pitch: 0.0,
                yaw: 0.0,
                acc_z: 9.81,
                acc_y: 0.0,
                acc_x: 0.0,
                gyro_z: 0.01,
                gyro_y: 0.01,
                gyro_x: 0.01,
                mag_z: 50.0,
                mag_y: -30.0,
                mag_x: -20.0,
                relative_altitude: 5.0,
                pressure: 1013.25,
                grav_z: 9.81,
                grav_y: 0.0,
                grav_x: 0.0,
            },
            TestDataRecord {
                time: DateTime::parse_from_str("2023-01-01 00:01:00+00:00", "%Y-%m-%d %H:%M:%S%z")
                    .unwrap()
                    .with_timezone(&Utc),
                bearing_accuracy: 0.2,
                speed_accuracy: 0.2,
                vertical_accuracy: 0.2,
                horizontal_accuracy: 0.2,
                speed: 2.0,
                bearing: 180.0,
                altitude: 200.0,
                longitude: -121.0,
                latitude: 38.0,
                qz: 0.0,
                qy: 0.0,
                qx: 0.0,
                qw: 1.0,
                roll: 0.1,
                pitch: 0.1,
                yaw: 0.1,
                acc_z: 9.81,
                acc_y: 0.01,
                acc_x: -0.01,
                gyro_z: 0.02,
                gyro_y: -0.02,
                gyro_x: 0.02,
                mag_z: 55.0,
                mag_y: -25.0,
                mag_x: -15.0,
                relative_altitude: 10.0,
                pressure: 1012.25,
                grav_z: 9.81,
                grav_y: 0.01,
                grav_x: -0.01,
            },
        ];

        // Write to HDF5
        TestDataRecord::to_hdf5(&records, &file_path).expect("Failed to write HDF5");

        // Read back from HDF5
        let read_records = TestDataRecord::from_hdf5(&file_path).expect("Failed to read HDF5");

        // Verify
        assert_eq!(read_records.len(), records.len());
        for (i, (original, read)) in records.iter().zip(read_records.iter()).enumerate() {
            assert_eq!(original.time, read.time, "Timestamp mismatch at index {i}");
            assert!(
                (original.latitude - read.latitude).abs() < 1e-10,
                "Latitude mismatch at index {i}"
            );
            assert!(
                (original.longitude - read.longitude).abs() < 1e-10,
                "Longitude mismatch at index {i}"
            );
            assert!(
                (original.altitude - read.altitude).abs() < 1e-10,
                "Altitude mismatch at index {i}"
            );
            assert!(
                (original.speed - read.speed).abs() < 1e-10,
                "Speed mismatch at index {i}"
            );
            assert!(
                (original.bearing - read.bearing).abs() < 1e-10,
                "Bearing mismatch at index {i}"
            );
        }
    }

    /// A result file written before the geophysical columns existed must still read.
    ///
    /// The four columns are additive, so every file already on disk lacks them. Adding them as
    /// required datasets would have made this reader reject those files outright -- a
    /// backwards-incompatible change smuggled in behind a bug fix. A missing dataset reads as
    /// absent; a dataset that is present but unreadable still fails, which is what keeps this
    /// from papering over a corrupt file.
    #[cfg(feature = "hdf5")]
    #[test]
    fn test_navigation_result_hdf5_reads_a_file_without_geophysical_columns() {
        use tempfile::tempdir;

        let dir = tempdir().unwrap();
        let file_path = dir.path().join("legacy.h5");

        let mut nav = NavigationResult::new();
        nav.latitude = 37.0;
        nav.longitude = -122.0;
        nav.altitude = 100.0;
        nav.gravity_bias = Some(1.5);
        nav.gravity_bias_cov = Some(0.25);
        NavigationResult::to_hdf5(&[nav], &file_path).unwrap();

        // Delete the four datasets, reproducing a file written before they existed. Everything
        // else in the file is untouched, so this is exactly an older writer's output.
        {
            let file = hdf5::File::open_rw(&file_path).unwrap();
            let group = file.group("navigation_results").unwrap();
            for name in [
                "gravity_bias",
                "gravity_bias_cov",
                "magnetic_bias",
                "magnetic_bias_cov",
            ] {
                group.unlink(name).unwrap();
            }
        }

        let read = NavigationResult::from_hdf5(&file_path)
            .expect("a file predating the geophysical columns must still read");
        assert_eq!(read.len(), 1);
        assert_approx_eq!(read[0].latitude, 37.0, 1e-9);
        assert_eq!(
            read[0].gravity_bias, None,
            "an absent column must read as absent, not as a zero the caller would believe"
        );
        assert_eq!(read[0].gravity_bias_cov, None);
        assert_eq!(read[0].magnetic_bias, None);
        assert_eq!(read[0].magnetic_bias_cov, None);
    }

    /// The geophysical columns survive a round trip when they are present.
    ///
    /// The companion to the test above: absent must stay absent, and present must stay present
    /// with its value, or the NaN sentinel the binary writers use would be indistinguishable
    /// from a real estimate.
    #[cfg(feature = "hdf5")]
    #[test]
    fn test_navigation_result_hdf5_roundtrips_geophysical_columns() {
        use tempfile::tempdir;

        let dir = tempdir().unwrap();
        let file_path = dir.path().join("geo.h5");

        let mut nav = NavigationResult::new();
        nav.gravity_bias = Some(12.5);
        nav.gravity_bias_cov = Some(3.25);
        // Magnetic deliberately absent: this run carried only a gravity map.
        NavigationResult::to_hdf5(&[nav], &file_path).unwrap();

        let read = NavigationResult::from_hdf5(&file_path).unwrap();
        assert_eq!(read.len(), 1);
        assert_approx_eq!(read[0].gravity_bias.unwrap(), 12.5, 1e-9);
        assert_approx_eq!(read[0].gravity_bias_cov.unwrap(), 3.25, 1e-9);
        assert_eq!(
            read[0].magnetic_bias, None,
            "a run with no magnetic map must not gain a magnetic estimate on the round trip"
        );
        assert_eq!(
            read[0].magnetic_bias_cov, None,
            "the covariance column must stay absent too, not just the estimate"
        );
    }

    /// A geophysically aided particle run labels its bias states and their variances.
    ///
    /// This is the regression the layout-aware constructor exists for. The RBPF carries one
    /// extra linear state per active map after its nine navigation states, but
    /// [`NavigationResult::from_particle_filter`] is a nine-state conversion: it had nowhere
    /// to put them, so a gravity-aided run wrote rows whose `gravity_bias` column was empty
    /// while the filter had estimated one all along.
    #[test]
    fn particle_filter_conversion_labels_its_geophysical_states() {
        let timestamp = Utc::now();
        // Nine navigation states, then gravity, then magnetic -- the order
        // `ExtraStateLayout` fixes and `geonav`'s `build_event_stream` counts back from.
        let mean = DVector::from_vec(vec![
            0.7, -1.3, 100.0, 1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 12.5, -40.0,
        ]);
        let mut cov = DMatrix::<f64>::zeros(11, 11);
        for i in 0..11 {
            cov[(i, i)] = f64::from(u32::try_from(i).unwrap_or(0)) + 1.0;
        }
        // Nine navigation states wide plus the two biases, gravity at 9 and magnetic at 10 --
        // the placement `GeoBiasLayout::appended` gives an RBPF run.
        let layout = ExtraStateLayout::new(11, Some(9), Some(10));

        let result =
            NavigationResult::from_particle_filter_with_geo(&timestamp, &mean, &cov, layout);

        assert_approx_eq!(result.latitude, 0.7_f64.to_degrees(), 1e-12);
        assert_approx_eq!(result.altitude, 100.0, 1e-12);
        assert_approx_eq!(
            result.gravity_bias.expect("a gravity map was declared"),
            12.5,
            1e-12
        );
        assert_approx_eq!(
            result.magnetic_bias.expect("a magnetic map was declared"),
            -40.0,
            1e-12
        );
        // The variance has to come out alongside the mean: a bias with no uncertainty on it
        // is not one a reader can do anything with.
        assert_approx_eq!(
            result
                .gravity_bias_cov
                .expect("the gravity bias must carry its variance"),
            10.0,
            1e-12
        );
        assert_approx_eq!(
            result
                .magnetic_bias_cov
                .expect("the magnetic bias must carry its variance"),
            11.0,
            1e-12
        );
    }

    /// A magnetic-only run puts its single extra state in the magnetic column, not the first
    /// one.
    ///
    /// The whole reason the layout travels with the run: a ten-element particle estimate is
    /// gravity-only or magnetic-only depending on which maps were loaded, and reading the
    /// wrong label off it would file a nanotesla figure as milligals.
    #[test]
    fn particle_filter_conversion_reads_a_single_extra_state_by_layout() {
        let timestamp = Utc::now();
        let mean = DVector::from_vec(vec![0.7, -1.3, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -40.0]);
        let mut cov = DMatrix::<f64>::identity(10, 10);
        cov[(9, 9)] = 7.0;

        let magnetic_only = NavigationResult::from_particle_filter_with_geo(
            &timestamp,
            &mean,
            &cov,
            ExtraStateLayout::new(10, None, Some(9)),
        );
        assert_eq!(
            magnetic_only.gravity_bias, None,
            "no gravity map means no gravity column, which is not the same as a zero bias"
        );
        assert_eq!(magnetic_only.gravity_bias_cov, None);
        assert_approx_eq!(magnetic_only.magnetic_bias.unwrap(), -40.0, 1e-12);
        assert_approx_eq!(magnetic_only.magnetic_bias_cov.unwrap(), 7.0, 1e-12);

        let gravity_only = NavigationResult::from_particle_filter_with_geo(
            &timestamp,
            &mean,
            &cov,
            ExtraStateLayout::new(10, Some(9), None),
        );
        assert_approx_eq!(gravity_only.gravity_bias.unwrap(), -40.0, 1e-12);
        assert_eq!(gravity_only.magnetic_bias, None);
    }

    /// The nine-state entry point is unchanged, and still leaves the geophysical columns
    /// absent.
    #[test]
    fn particle_filter_conversion_without_geo_states_leaves_the_columns_absent() {
        let timestamp = Utc::now();
        let mean = DVector::from_vec(vec![0.7, -1.3, 100.0, 1.0, 2.0, 3.0, 0.1, 0.2, 0.3]);
        let cov = DMatrix::<f64>::identity(9, 9);

        let result = NavigationResult::from_particle_filter(&timestamp, &mean, &cov);
        assert_approx_eq!(result.longitude, (-1.3_f64).to_degrees(), 1e-12);
        assert_eq!(result.gravity_bias, None);
        assert_eq!(result.gravity_bias_cov, None);
        assert_eq!(result.magnetic_bias, None);
        assert_eq!(result.magnetic_bias_cov, None);
    }

    /// Declaring no geophysical states still rejects a filter that has them.
    ///
    /// The fix told the conversion what the extra states are; it did not loosen the
    /// invariant. Quietly dropping them is the outcome this whole change exists to remove,
    /// so the nine-state path must keep refusing a wider estimate rather than truncating it.
    #[test]
    #[should_panic(expected = "Particle filter state must have 9 elements")]
    fn particle_filter_conversion_rejects_extra_states_it_was_not_told_about() {
        let timestamp = Utc::now();
        let mean = DVector::from_vec(vec![0.7, -1.3, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
        let cov = DMatrix::<f64>::identity(10, 10);
        let _ = NavigationResult::from_particle_filter(&timestamp, &mean, &cov);
    }

    /// A particle-width layout goes through the particle constructor, not off the end.
    ///
    /// The four-tuple `From` asserts the state is `layout.state_dim()` wide -- nine, for
    /// [`ExtraStateLayout::PARTICLE_NONE`], which a particle estimate satisfies -- and then used
    /// to read `state[9]..state[14]` for the IMU-bias block a particle filter does not have.
    /// The width assertion passed and the indexing panicked, which is why every particle event
    /// loop in this workspace is hand-rolled rather than going through
    /// [`run_closed_loop_with_geo`]. Nothing passed a particle layout to it, so the panic was
    /// latent; this is the test that keeps it that way.
    #[test]
    fn a_particle_width_layout_converts_through_the_particle_constructor() {
        let timestamp = Utc::now();
        let mean = DVector::from_vec(vec![0.7, -1.3, 100.0, 1.0, 2.0, 3.0, 0.1, 0.2, 0.3]);
        let cov = DMatrix::<f64>::identity(PARTICLE_FILTER_STATES, PARTICLE_FILTER_STATES) * 0.25;

        let through_from =
            NavigationResult::from((&timestamp, &mean, &cov, ExtraStateLayout::PARTICLE_NONE));
        let through_constructor = NavigationResult::from_particle_filter(&timestamp, &mean, &cov);

        assert_eq!(through_from.latitude, through_constructor.latitude);
        assert_eq!(through_from.altitude, through_constructor.altitude);
        assert_eq!(through_from.latitude_cov, through_constructor.latitude_cov);
        // A particle filter estimates no IMU biases, so the bias block is zero and its
        // covariance is not read from a state that does not carry it.
        assert_eq!(through_from.acc_bias_x, 0.0);
        assert_eq!(through_from.gyro_bias_z, 0.0);
        // The position covariance a metric would score against must be a real number.
        assert!(through_from.latitude_cov.is_finite());
        assert!(through_from.altitude_cov.is_finite());
    }

    /// The unaided layouts differ by filter, and it is the width that differs.
    ///
    /// [`ExtraStateLayout::NONE`] describes the Kalman filters' unaided shape and is fifteen
    /// wide. Handing it to the particle conversion would fail on the first row of every
    /// ordinary particle run, which is why [`ExtraStateLayout::PARTICLE_NONE`] exists.
    #[test]
    fn unaided_layouts_carry_each_filter_s_own_width() {
        assert_eq!(ExtraStateLayout::NONE.state_dim(), NAVIGATION_STATES);
        assert_eq!(
            ExtraStateLayout::PARTICLE_NONE.state_dim(),
            PARTICLE_FILTER_STATES
        );
        assert!(ExtraStateLayout::PARTICLE_NONE.is_empty());
        assert_eq!(ExtraStateLayout::PARTICLE_NONE.gravity_index(), None);
        assert_eq!(ExtraStateLayout::PARTICLE_NONE.magnetic_index(), None);
    }

    /// A bias index inside the navigation states is refused rather than read.
    ///
    /// The particle conversion checks against its own base of nine, not the Kalman fifteen --
    /// the Kalman bound would reject every genuine particle layout. Index 8 is the yaw angle,
    /// which is exactly the state a from-the-end index resolves to when the vector is too
    /// narrow, so this is the failure the layout exists to prevent.
    #[test]
    #[should_panic(expected = "a map bias lives after the 9 navigation states")]
    fn particle_filter_conversion_rejects_a_bias_index_inside_the_navigation_states() {
        let timestamp = Utc::now();
        let mean = DVector::from_vec(vec![0.7, -1.3, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
        let cov = DMatrix::<f64>::identity(10, 10);
        let _ = NavigationResult::from_particle_filter_with_geo(
            &timestamp,
            &mean,
            &cov,
            ExtraStateLayout::new(10, Some(8), None),
        );
    }

    #[cfg(feature = "hdf5")]
    #[test]
    fn test_navigation_result_hdf5_roundtrip() {
        use tempfile::tempdir;

        let dir = tempdir().unwrap();
        let file_path = dir.path().join("nav_results.h5");

        // Create test navigation results
        let mut results = Vec::new();
        let mut nav1 = NavigationResult::new();
        nav1.timestamp =
            DateTime::parse_from_str("2023-01-01 00:00:00+00:00", "%Y-%m-%d %H:%M:%S%z")
                .unwrap()
                .with_timezone(&Utc);
        nav1.latitude = 37.0;
        nav1.longitude = -122.0;
        nav1.altitude = 100.0;
        nav1.velocity_north = 1.0;
        nav1.velocity_east = 2.0;
        nav1.velocity_vertical = 0.1;
        nav1.roll = 0.01;
        nav1.pitch = 0.02;
        nav1.yaw = 0.03;
        results.push(nav1);

        let mut nav2 = NavigationResult::new();
        nav2.timestamp =
            DateTime::parse_from_str("2023-01-01 00:00:01+00:00", "%Y-%m-%d %H:%M:%S%z")
                .unwrap()
                .with_timezone(&Utc);
        nav2.latitude = 37.0001;
        nav2.longitude = -122.0001;
        nav2.altitude = 101.0;
        nav2.velocity_north = 1.1;
        nav2.velocity_east = 2.1;
        nav2.velocity_vertical = 0.2;
        nav2.roll = 0.02;
        nav2.pitch = 0.03;
        nav2.yaw = 0.04;
        results.push(nav2);

        // Write to HDF5
        NavigationResult::to_hdf5(&results, &file_path).expect("Failed to write HDF5");

        // Read back from HDF5
        let read_results = NavigationResult::from_hdf5(&file_path).expect("Failed to read HDF5");

        // Verify
        assert_eq!(read_results.len(), results.len());
        for (i, (original, read)) in results.iter().zip(read_results.iter()).enumerate() {
            assert_eq!(
                original.timestamp, read.timestamp,
                "Timestamp mismatch at index {i}"
            );
            assert!(
                (original.latitude - read.latitude).abs() < 1e-10,
                "Latitude mismatch at index {i}"
            );
            assert!(
                (original.longitude - read.longitude).abs() < 1e-10,
                "Longitude mismatch at index {i}"
            );
            assert!(
                (original.altitude - read.altitude).abs() < 1e-10,
                "Altitude mismatch at index {i}"
            );
            assert!(
                (original.velocity_north - read.velocity_north).abs() < 1e-10,
                "Velocity north mismatch at index {i}"
            );
            assert!(
                (original.velocity_east - read.velocity_east).abs() < 1e-10,
                "Velocity east mismatch at index {i}"
            );
            assert!(
                (original.roll - read.roll).abs() < 1e-10,
                "Roll mismatch at index {i}"
            );
            assert!(
                (original.pitch - read.pitch).abs() < 1e-10,
                "Pitch mismatch at index {i}"
            );
            assert!(
                (original.yaw - read.yaw).abs() < 1e-10,
                "Yaw mismatch at index {i}"
            );
        }
    }

    #[cfg(feature = "hdf5")]
    #[test]
    fn test_test_data_record_hdf5_with_nan_values() {
        use tempfile::tempdir;

        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test_data_nan.h5");

        // Create a record with NaN values
        let record = TestDataRecord {
            time: Utc::now(),
            latitude: 37.0,
            longitude: -122.0,
            altitude: f64::NAN,
            speed: f64::NAN,
            bearing: 90.0,
            ..Default::default()
        };

        let records = vec![record.clone()];

        // Write to HDF5
        TestDataRecord::to_hdf5(&records, &file_path).expect("Failed to write HDF5");

        // Read back from HDF5
        let read_records = TestDataRecord::from_hdf5(&file_path).expect("Failed to read HDF5");

        // Verify NaN values are preserved
        assert_eq!(read_records.len(), 1);
        assert!(
            read_records[0].altitude.is_nan(),
            "NaN altitude should be preserved"
        );
        assert!(
            read_records[0].speed.is_nan(),
            "NaN speed should be preserved"
        );
        assert!((read_records[0].latitude - record.latitude).abs() < 1e-10);
        assert!((read_records[0].longitude - record.longitude).abs() < 1e-10);
    }

    #[cfg(feature = "hdf5")]
    #[test]
    fn test_navigation_result_hdf5_empty() {
        use tempfile::tempdir;

        let dir = tempdir().unwrap();
        let file_path = dir.path().join("nav_results_empty.h5");

        // Write empty results
        let results: Vec<NavigationResult> = Vec::new();
        NavigationResult::to_hdf5(&results, &file_path).expect("Failed to write empty HDF5");

        // Read back
        let read_results =
            NavigationResult::from_hdf5(&file_path).expect("Failed to read empty HDF5");

        // Verify it's empty
        assert_eq!(read_results.len(), 0);
        // Note: TestDataRecord MCAP roundtrip test is disabled due to CSV-specific deserializers
        // that conflict with binary serialization formats. TestDataRecord is optimized for CSV.
        // For MCAP usage, convert TestDataRecord to NavigationResult.
    }

    /// The netCDF codecs had no test at all until #335 changed how libnetcdf is built.
    ///
    /// `to_netcdf`/`from_netcdf` compile under `--all-features`, which CI runs, so they were
    /// type-checked but never executed -- nothing would have caught a behavioural difference
    /// between the system libnetcdf these were written against and the vendored one they now
    /// link. The hdf5 side has had round-trip, NaN and missing-column tests all along; this
    /// is the netCDF half of that.
    ///
    /// Note the whole-second timestamps. `to_netcdf` stores time as `timestamp()`, an integer
    /// number of seconds, so sub-second precision does not survive and a test using it would
    /// fail for a reason that has nothing to do with netCDF.
    #[cfg(feature = "netcdf")]
    #[test]
    fn test_test_data_record_netcdf_roundtrip() {
        use tempfile::tempdir;

        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test_data.nc");

        let record = TestDataRecord {
            time: DateTime::parse_from_str("2023-01-01 00:00:00+00:00", "%Y-%m-%d %H:%M:%S%z")
                .unwrap()
                .with_timezone(&Utc),
            latitude: 37.0,
            longitude: -122.0,
            altitude: 100.0,
            speed: 1.5,
            bearing: 90.0,
            acc_z: 9.81,
            gyro_x: 0.01,
            mag_x: -20.0,
            ..Default::default()
        };

        TestDataRecord::to_netcdf(std::slice::from_ref(&record), &file_path)
            .expect("Failed to write netCDF");
        let read = TestDataRecord::from_netcdf(&file_path).expect("Failed to read netCDF");

        assert_eq!(read.len(), 1);
        assert_eq!(read[0].time, record.time);
        assert_approx_eq!(read[0].latitude, record.latitude, 1e-10);
        assert_approx_eq!(read[0].longitude, record.longitude, 1e-10);
        assert_approx_eq!(read[0].altitude, record.altitude, 1e-10);
        assert_approx_eq!(read[0].speed, record.speed, 1e-10);
        assert_approx_eq!(read[0].bearing, record.bearing, 1e-10);
        assert_approx_eq!(read[0].acc_z, record.acc_z, 1e-10);
        assert_approx_eq!(read[0].gyro_x, record.gyro_x, 1e-10);
        assert_approx_eq!(read[0].mag_x, record.mag_x, 1e-10);
    }

    /// The `Option<f64>` geophysical columns survive a netCDF round trip as absent, not zero.
    ///
    /// netCDF has no option type, so `none_if_nan` writes `NaN` and maps it back. This is the
    /// netCDF counterpart of `test_navigation_result_hdf5_roundtrips_geophysical_columns`:
    /// a run carrying only a gravity map must not come back with a magnetic estimate.
    #[cfg(feature = "netcdf")]
    #[test]
    fn test_navigation_result_netcdf_roundtrips_geophysical_columns() {
        use tempfile::tempdir;

        let dir = tempdir().unwrap();
        let file_path = dir.path().join("geo.nc");

        let mut nav = NavigationResult::new();
        nav.gravity_bias = Some(12.5);
        nav.gravity_bias_cov = Some(3.25);
        NavigationResult::to_netcdf(std::slice::from_ref(&nav), &file_path)
            .expect("Failed to write netCDF");

        let read = NavigationResult::from_netcdf(&file_path).expect("Failed to read netCDF");
        assert_eq!(read.len(), 1);
        assert_approx_eq!(read[0].gravity_bias.unwrap(), 12.5, 1e-9);
        assert_approx_eq!(read[0].gravity_bias_cov.unwrap(), 3.25, 1e-9);
        assert_eq!(
            read[0].magnetic_bias, None,
            "a run with no magnetic map must not gain a magnetic estimate on the round trip"
        );
        assert_eq!(
            read[0].magnetic_bias_cov, None,
            "the covariance column must stay absent too, not just the estimate"
        );
    }

    /// Writing zero records to netCDF is an error, where the HDF5 writer accepts it.
    ///
    /// The asymmetry is deliberate on the writer's side (`to_netcdf` bails on an empty slice,
    /// `to_hdf5` writes a file with zero-length datasets -- see
    /// `test_navigation_result_hdf5_empty`), but it was never pinned by a test, so nothing
    /// said which half was intentional. This says it: netCDF refuses, and a caller that might
    /// hand it an empty run has to check first.
    #[cfg(feature = "netcdf")]
    #[test]
    fn test_navigation_result_netcdf_rejects_empty() {
        use tempfile::tempdir;

        let dir = tempdir().unwrap();
        let file_path = dir.path().join("nav_results_empty.nc");

        let results: Vec<NavigationResult> = Vec::new();
        let err = NavigationResult::to_netcdf(&results, &file_path)
            .expect_err("writing zero records to netCDF must fail rather than produce a file");
        assert!(
            err.to_string().contains("empty"),
            "the error should say what was wrong, got: {err}"
        );
        assert!(
            !file_path.exists(),
            "a rejected write must not leave a partial file behind"
        );
    }
    #[cfg(feature = "mcap")]
    #[test]
    fn test_navigation_result_mcap_roundtrip() {
        let temp_file = std::env::temp_dir().join("nav_results_mcap.mcap");

        // Create test navigation results
        let result1 = NavigationResult {
            timestamp: Utc::now(),
            latitude: 37.0,
            longitude: -122.0,
            altitude: 100.0,
            velocity_north: 10.0,
            velocity_east: 5.0,
            velocity_vertical: -1.0,
            roll: 0.1,
            pitch: 0.2,
            yaw: 0.3,
            ..Default::default()
        };

        let result2 = NavigationResult {
            timestamp: Utc::now() + chrono::Duration::seconds(1),
            latitude: 37.01,
            longitude: -122.01,
            altitude: 110.0,
            velocity_north: 12.0,
            velocity_east: 6.0,
            velocity_vertical: 0.5,
            roll: 0.15,
            pitch: 0.25,
            yaw: 0.35,
            ..Default::default()
        };

        let results = vec![result1, result2];

        // Write to MCAP
        NavigationResult::to_mcap(&results, &temp_file)
            .expect("Failed to write navigation results to MCAP");

        // Check file exists
        assert!(temp_file.exists(), "MCAP file should exist");

        // Read back from MCAP
        let read_results = NavigationResult::from_mcap(&temp_file)
            .expect("Failed to read navigation results from MCAP");

        // Verify count
        assert_eq!(
            read_results.len(),
            results.len(),
            "Result count should match"
        );

        // Verify content
        for (i, (original, read)) in results.iter().zip(read_results.iter()).enumerate() {
            assert_eq!(
                original.timestamp, read.timestamp,
                "Result {i} timestamp should match"
            );
            assert!(
                (original.latitude - read.latitude).abs() < 1e-6,
                "Result {i} latitude should match"
            );
            assert!(
                (original.longitude - read.longitude).abs() < 1e-6,
                "Result {i} longitude should match"
            );
            assert!(
                (original.altitude - read.altitude).abs() < 1e-6,
                "Result {i} altitude should match"
            );
            assert!(
                (original.velocity_north - read.velocity_north).abs() < 1e-6,
                "Result {i} velocity_north should match"
            );
            assert!(
                (original.roll - read.roll).abs() < 1e-6,
                "Result {i} roll should match"
            );
            assert!(
                (original.pitch - read.pitch).abs() < 1e-6,
                "Result {i} pitch should match"
            );
            assert!(
                (original.yaw - read.yaw).abs() < 1e-6,
                "Result {i} yaw should match"
            );
        }

        // Cleanup
        let _ = std::fs::remove_file(&temp_file);
    }

    /// #334: `position_rms_meters` must convert the latitude/longitude sigmas (radians) to
    /// metres before combining them with the altitude sigma (already metres) -- summing
    /// radians-squared and metres-squared as though they were the same unit made the printed
    /// RMS meaningless, in practice just the altitude sigma with noise on it.
    #[test]
    fn position_rms_converts_horizontal_sigma_to_meters() {
        let lat_deg: f64 = 30.0;
        let alt_m: f64 = 500.0;
        let pos_std_lat_rad: f64 = 2e-6;
        let pos_std_lon_rad: f64 = 3e-6;
        let pos_std_alt_m: f64 = 4.0;

        let (r_n, r_e, _) = crate::earth::principal_radii(&lat_deg, &alt_m);
        let expected_lat_m = pos_std_lat_rad * (r_n + alt_m);
        let expected_lon_m = pos_std_lon_rad * (r_e + alt_m) * lat_deg.to_radians().cos();
        let expected_rms =
            (expected_lat_m.powi(2) + expected_lon_m.powi(2) + pos_std_alt_m.powi(2)).sqrt();

        let rms = position_rms_meters(
            lat_deg,
            alt_m,
            pos_std_lat_rad,
            pos_std_lon_rad,
            pos_std_alt_m,
        );
        assert_approx_eq!(rms, expected_rms, 1e-9);

        // Pre-fix, the horizontal terms were squared *radians* (~1e-12) added to squared
        // metres, so the result was indistinguishable from the altitude sigma alone. Once
        // correctly scaled by the local radii of curvature, a few-microradian uncertainty is
        // several metres and should dominate.
        assert!(
            rms > pos_std_alt_m,
            "expected the converted horizontal uncertainty to dominate the altitude sigma, got {rms}"
        );
    }

    /// A synthetic configuration at 40N 75W, where the magnetic tests know the field.
    ///
    /// The World Magnetic Model there gives a total intensity near 50 uT, an inclination near
    /// 66 degrees down and a declination near 12 degrees west -- three independent numbers a
    /// fabricated or mis-rotated field cannot reproduce by accident.
    /// The ENU field must survive the round trip through the measurement that consumes it.
    ///
    /// This is the test that was missing when `magnetic_field_nav_ut` returned the ENU field
    /// as `(east, north, up)`. Every baseline scenario is NED and
    /// `synthetic_config_for_tests` hard-codes `is_enu: false`, so nothing exercised the ENU
    /// branch and a transposed field shipped green.
    ///
    /// Generating a heading and reading it back with anything but the real consumer would not
    /// have caught it either -- a reflected field still has a plausible magnitude and a
    /// plausible angle. So this drives `MagnetometerYawMeasurement` itself, at the record's
    /// own timestamp, with declination applied exactly as `build_event_stream` applies it.
    /// Under the old ordering it reads `pi/2 - psi` instead of `psi`, which at this trajectory
    /// is 53.13 deg against 36.87 -- off by 16.3 deg and comfortably outside the tolerance.
    #[test]
    fn the_enu_magnetic_field_round_trips_through_its_own_measurement() {
        use crate::measurements::{MagnetometerYawMeasurement, MeasurementModel};
        use chrono::Datelike;

        for is_enu in [false, true] {
            let mut config = synthetic_config_for_tests();
            config.initial_state.is_enu = is_enu;
            config.no_noise = true;
            config.duration_s = 2.0;
            let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(7);
            let (truth, records) = generate_synthetic(&config, &mut rng).expect("generation");
            let first = &records[0];

            let measurement = MagnetometerYawMeasurement {
                mag_x: first.mag_x,
                mag_y: first.mag_y,
                mag_z: first.mag_z,
                noise_std: crate::measurements::MAG_YAW_NOISE,
                apply_declination: true,
                year: first.time.year(),
                day_of_year: first.time.ordinal() as u16,
                is_enu,
            };

            // The state the measurement reads position and attitude from, in the filter's own
            // units: radians for latitude and longitude, radians for the Euler triple. The
            // truth rows already carry the attitude as Euler angles in radians.
            let (roll, pitch, yaw) = (truth[0].roll, truth[0].pitch, truth[0].yaw);
            // `NavigationResult` is mixed-unit by design: latitude and longitude in degrees,
            // the Euler triple in radians. The measurement wants the filter's units, which
            // are radians throughout, so only the two position angles convert. Getting this
            // wrong is silent rather than loud -- `get_declination` falls back to zero when
            // the model declines the position, so degrees-as-radians reads as a clean
            // 11.94 deg heading bias, which is exactly this trajectory's declination.
            let state = DVector::from_vec(vec![
                truth[0].latitude.to_radians(),
                truth[0].longitude.to_radians(),
                truth[0].altitude,
                0.0,
                0.0,
                0.0,
                roll,
                pitch,
                yaw,
            ]);

            let recovered = measurement
                .get_measurement(&state)
                .expect("heading from the synthesised field");
            let error = crate::wrap_to_pi(recovered[0] - yaw).to_degrees();
            assert!(
                error.abs() < 1.0,
                "is_enu={is_enu}: the magnetometer read {:.3} deg against a truth yaw of \
                 {:.3} deg, {error:.3} deg out. A horizontal axis swap in the generated field \
                 shows up here as roughly pi/2 - psi.",
                recovered[0].to_degrees(),
                yaw.to_degrees(),
            );
        }
    }

    /// An out-of-band altitude must not silently cost the heading its declination.
    ///
    /// `magnetic_field_nav_ut` clamps into the WMM's altitude band; `get_declination` did not,
    /// and returned **zero** when the model refused the position. So a trajectory below -1 km
    /// had the declination written into its field at the clamp and removed at zero, which is a
    /// systematic heading bias of the local declination -- about 12 degrees here -- with no
    /// error anywhere. Both now clamp by the same constants.
    ///
    /// Worth pinning beyond the synthetic path: on a real run `alt_m` is the *filter's*
    /// altitude estimate, so the input that reaches the unsupported band is produced by a
    /// filter that is already diverging.
    #[test]
    fn an_out_of_band_altitude_keeps_its_declination() {
        use crate::measurements::MagnetometerYawMeasurement;

        let measurement = |altitude_m: f64| {
            let sample = MagnetometerYawMeasurement {
                mag_x: 1.0,
                mag_y: 0.0,
                mag_z: 0.0,
                noise_std: crate::measurements::MAG_YAW_NOISE,
                apply_declination: true,
                year: 2024,
                day_of_year: 1,
                is_enu: false,
            };
            sample.get_declination(40.0, -75.0, altitude_m)
        };

        let in_band = measurement(200.0);
        assert!(
            in_band.abs() > 0.1,
            "40N 75W should have a declination of roughly -12 deg, got {} deg",
            in_band.to_degrees()
        );

        for altitude_m in [-5_000.0, 1_000_000.0] {
            let out_of_band = measurement(altitude_m);
            assert!(
                (out_of_band - in_band).abs() < 0.05,
                "at {altitude_m} m the declination came back as {} deg against {} deg in \
                 band; an unsupported altitude must clamp, not fall back to zero",
                out_of_band.to_degrees(),
                in_band.to_degrees()
            );
        }
    }

    fn synthetic_config_for_tests() -> SyntheticConfig {
        SyntheticConfig {
            output: String::new(),
            initial_state: SyntheticInitialState {
                latitude_deg: 40.0,
                longitude_deg: -75.0,
                altitude_m: 200.0,
                velocity_north_mps: 40.0,
                velocity_east_mps: 30.0,
                velocity_down_mps: 0.0,
                // Level, so the body frame is the navigation frame and the field can be read
                // off without tilt compensation.
                roll_deg: 0.0,
                pitch_deg: 0.0,
                yaw_deg: 36.869_897_645_844_02,
                angular_velocity_x_dps: 0.0,
                angular_velocity_y_dps: 0.0,
                angular_velocity_z_dps: 0.0,
                is_enu: false,
            },
            duration_s: 10.0,
            sample_rate_hz: 50.0,
            imu_quality: crate::IMUQuality::Consumer,
            seed: 42,
            no_noise: false,
            gnss_horizontal_noise_m: 3.0,
            gnss_vertical_noise_m: 5.0,
            baro_noise_std_pa: 30.0,
            mag_noise_std_ut: default_mag_noise_std_ut(),
            mag_hard_iron_std_ut: default_mag_hard_iron_std_ut(),
        }
    }

    /// The synthetic magnetometer reports the real field, not a placeholder (#369).
    ///
    /// Three independent properties of the World Magnetic Model at 40N 75W, all of which a
    /// fabricated or mis-rotated field would fail:
    ///
    /// * **total intensity** ~50 uT,
    /// * **inclination** ~66 deg downward,
    /// * **declination** ~12 deg west -- recovered here as the offset between the magnetic
    ///   heading the channels imply at a level attitude and the trajectory's true yaw.
    ///
    /// The last is the one that matters most, because it is the round trip:
    /// `MagnetometerYawMeasurement` looks the declination up again from the record's own
    /// timestamp and subtracts it. If this function wrote a field for a different date or
    /// frame, the declination would be removed at a different value than it was put in and
    /// yaw aiding would be silently biased -- which is the failure mode #305 was filed for.
    #[test]
    fn the_synthetic_magnetometer_reports_the_world_magnetic_model() {
        use rand::SeedableRng;

        let mut config = synthetic_config_for_tests();
        config.no_noise = true;
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let (_truth, records) = generate_synthetic(&config, &mut rng).expect("synthetic run");

        let first = &records[0];
        assert!(
            first.mag_x.is_finite() && first.mag_y.is_finite() && first.mag_z.is_finite(),
            "the magnetic channels must carry a field, not NaN: \
             ({}, {}, {})",
            first.mag_x,
            first.mag_y,
            first.mag_z
        );

        let total = (first.mag_x.powi(2) + first.mag_y.powi(2) + first.mag_z.powi(2)).sqrt();
        assert!(
            (40.0..60.0).contains(&total),
            "total intensity at 40N 75W should be near 50 uT, got {total:.3}"
        );

        // Level attitude, so the body frame is the navigation frame and no tilt compensation
        // is needed to read these off.
        let horizontal = first.mag_x.hypot(first.mag_y);
        let inclination_deg = first.mag_z.atan2(horizontal).to_degrees();
        assert!(
            (55.0..75.0).contains(&inclination_deg),
            "inclination at 40N 75W should be near 66 deg down, got {inclination_deg:.2}"
        );

        let magnetic_heading_deg = (-first.mag_y).atan2(first.mag_x).to_degrees();
        let true_yaw_deg = config.initial_state.yaw_deg;
        let declination_deg = true_yaw_deg - magnetic_heading_deg;
        assert!(
            (-16.0..-8.0).contains(&declination_deg),
            "declination at 40N 75W should be near 12 deg west, got {declination_deg:.2} \
             (magnetic heading {magnetic_heading_deg:.2} against true yaw {true_yaw_deg:.2})"
        );
    }

    /// Hard iron is off unless asked for, and does something when it is.
    ///
    /// The default matters: a hard-iron offset is constant in the *body* frame, so it biases
    /// heading in a way no filter can observe. Leaving it on by default would put an
    /// unobservable bias into the yaw column #371 is diagnosed from.
    #[test]
    fn hard_iron_is_off_by_default_and_shifts_the_field_when_enabled() {
        use rand::SeedableRng;

        let mut config = synthetic_config_for_tests();
        config.no_noise = true;

        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let (_, clean) = generate_synthetic(&config, &mut rng).expect("clean run");
        assert_eq!(config.mag_hard_iron_std_ut, 0.0, "the default must be zero");

        config.mag_hard_iron_std_ut = 5.0;
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let (_, ironed) = generate_synthetic(&config, &mut rng).expect("hard-iron run");

        let offset = (ironed[0].mag_x - clean[0].mag_x).hypot(ironed[0].mag_y - clean[0].mag_y);
        assert!(
            offset > 1e-6,
            "a 5 uT hard-iron offset should move the field, moved {offset:.6} uT"
        );

        // Constant in the body frame: the same offset on the last sample as on the first.
        let last = ironed.len() - 1;
        let first_dx = ironed[0].mag_x - clean[0].mag_x;
        let last_dx = ironed[last].mag_x - clean[last].mag_x;
        assert!(
            (first_dx - last_dx).abs() < 1e-9,
            "hard iron is constant per trajectory: {first_dx:.9} then {last_dx:.9}"
        );
    }

    /// With zero horizontal uncertainty the RMS collapses to the (already metric) altitude
    /// sigma, regardless of latitude or altitude.
    #[test]
    fn position_rms_is_altitude_sigma_when_horizontal_uncertainty_is_zero() {
        let rms = position_rms_meters(45.0, 1000.0, 0.0, 0.0, 7.5);
        assert_approx_eq!(rms, 7.5, 1e-12);
    }
    // ================== #367: what an output row's timestamp means ==================

    /// A filter that counts the events it has been handed instead of navigating.
    ///
    /// The point is to assert the *causal* invariant directly rather than through an RMSE: a
    /// row stamped `t_k` must hold every event at or before `t_k` and none after it. The
    /// count is carried in the altitude channel, which passes through
    /// `NavigationResult::from` untouched and sits well inside [`HealthLimits`], so each
    /// output row reports exactly how many events had been applied when it was emitted.
    #[derive(Debug, Default)]
    struct EventCountingFilter {
        /// Events applied so far, of either kind.
        applied: usize,
    }

    /// Altitude the counting filter reports with no events applied, metres.
    const COUNTING_FILTER_BASE_ALTITUDE_M: f64 = 100.0;

    impl NavigationFilter for EventCountingFilter {
        fn predict(
            &mut self,
            _control_input: &dyn crate::InputModel,
            _dt: f64,
        ) -> Result<(), StrapdownError> {
            self.applied += 1;
            Ok(())
        }

        fn update(
            &mut self,
            _measurement: &dyn crate::measurements::MeasurementModel,
        ) -> Result<crate::gating::UpdateOutcome, StrapdownError> {
            self.applied += 1;
            Ok(crate::gating::UpdateOutcome::accepted(0.0, 3))
        }

        fn get_estimate(&self) -> DVector<f64> {
            let mut state = DVector::zeros(15);
            state[2] = COUNTING_FILTER_BASE_ALTITUDE_M + self.applied as f64;
            state
        }

        fn get_certainty(&self) -> DMatrix<f64> {
            DMatrix::identity(15, 15)
        }
    }

    /// Build a stream of `epochs` epochs, each carrying `events_per_epoch` events one second
    /// apart, starting one second after `start_time` -- which mirrors `build_event_stream`,
    /// whose `windows(2)` walk leaves record 0's epoch empty.
    fn counting_stream(
        start_time: DateTime<Utc>,
        epochs: usize,
        events_per_epoch: usize,
    ) -> EventStream {
        let mut events = Vec::new();
        for epoch in 1..=epochs {
            for _ in 0..events_per_epoch {
                events.push(Event::Imu {
                    dt_s: 1.0,
                    imu: IMUData {
                        accel: Vector3::new(0.0, 0.0, 0.0),
                        gyro: Vector3::new(0.0, 0.0, 0.0),
                    },
                    elapsed_s: epoch as f64,
                });
            }
        }
        EventStream { start_time, events }
    }

    /// A row stamped `t_k` holds every event at or before `t_k`, and none after it.
    ///
    /// This is #367 stated directly. Before the fix the push sat below the `match`, so the
    /// row labelled `t_k` carried `t_{k+1}`'s first event as well -- every interior row was
    /// one propagation step ahead of its own label, worth 21.2 m of along-track error on the
    /// 1 Hz reference recording. Asserting it here rather than through a horizontal RMSE
    /// means the invariant is pinned whatever the tuning does.
    #[test]
    fn a_row_holds_every_event_at_or_before_its_own_timestamp() {
        for events_per_epoch in [1_usize, 2, 4] {
            let start_time = Utc::now();
            let epochs = 5;
            let mut filter = EventCountingFilter::default();
            let stream = counting_stream(start_time, epochs, events_per_epoch);
            let results = run_closed_loop(&mut filter, stream, None, None).unwrap();

            assert_eq!(
                results.len(),
                epochs + 1,
                "expected one seed row plus one row per epoch at {events_per_epoch} \
                 events/epoch, got {}",
                results.len()
            );

            for (index, row) in results.iter().enumerate() {
                let applied = row.altitude - COUNTING_FILTER_BASE_ALTITUDE_M;
                // Row 0 is the seed, before any event; row k covers epochs 1..=k.
                let expected = (index * events_per_epoch) as f64;
                assert_approx_eq!(applied, expected, 1e-9);

                let expected_ts =
                    start_time + Duration::milliseconds((index as f64 * 1000.0) as i64);
                assert_eq!(
                    row.timestamp, expected_ts,
                    "row {index} is stamped {} rather than {expected_ts}",
                    row.timestamp
                );
            }
        }
    }

    /// The last epoch is emitted exactly once, with all of its events applied.
    ///
    /// The `i == total - 1` push this replaces fired *in addition to* the epoch-boundary push
    /// whenever the final event was the first at its timestamp, emitting one state under two
    /// labels. One event per epoch is precisely that case; it never arose on
    /// `test_data.csv`, where each epoch carries up to four events, which is why it survived.
    #[test]
    fn the_final_epoch_is_emitted_once_with_all_its_events() {
        let start_time = Utc::now();
        let mut filter = EventCountingFilter::default();
        let results =
            run_closed_loop(&mut filter, counting_stream(start_time, 3, 1), None, None).unwrap();

        // The duplicate is one *state* under two labels, not one timestamp twice, so
        // uniqueness of the timestamps does not catch it: pre-fix this emitted
        // `[seed, t1(2 events), t2(3), t3(3)]` -- four distinct labels, but the last two
        // holding the same state, because the run had nothing left to apply between them.
        let counts: Vec<f64> = results
            .iter()
            .map(|r| r.altitude - COUNTING_FILTER_BASE_ALTITUDE_M)
            .collect();
        assert_eq!(counts, vec![0.0, 1.0, 2.0, 3.0], "rows: {counts:?}");

        for pair in counts.windows(2) {
            assert!(
                pair[1] > pair[0],
                "consecutive rows restate the same state: {counts:?}"
            );
        }
    }

    /// An empty stream yields the seed row alone, not a duplicate of it.
    #[test]
    fn an_empty_stream_yields_only_the_seed_row() {
        let start_time = Utc::now();
        let mut filter = EventCountingFilter::default();
        let stream = EventStream {
            start_time,
            events: Vec::new(),
        };
        let results = run_closed_loop(&mut filter, stream, None, None).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].timestamp, start_time);
    }
}
