//! STRAPDOWN SIM: A simulation and analysis tool for strapdown inertial navigation systems.
//!
//! This program can operate in three modes: open-loop, closed-loop, and particle-filter.
//!
//! - Open-loop mode: Relies solely on inertial measurements (IMU) and an initial position estimate
//!   for dead reckoning. Useful for high-accuracy IMUs with drift rates ≤1 nm per 24 hours.
//!
//! - Closed-loop mode: Incorporates GNSS measurements to correct IMU drift using either an
//!   Unscented Kalman Filter (UKF) or Extended Kalman Filter (EKF). Supports GNSS degradation
//!   scenarios including jamming, reduced update rates, and spoofing.
//!
//! - Particle-filter mode: Uses particle-based state estimation, supporting both standard and
//!   Rao-Blackwellized implementations.
//!
//! You can run simulations either by:
//!   1. Loading all parameters from a configuration file (TOML/JSON/YAML)
//!   2. Specifying parameters via command-line flags
//!
//! For dataset format details, see the documentation or use --help with specific subcommands.

mod common;
#[cfg(feature = "plotting")]
mod plotting;

use clap::{Args, Parser, Subcommand};
use common::{
    get_csv_files, init_logger, load_records, prompt_config_name, prompt_config_path,
    prompt_f64_with_default, prompt_input_path, prompt_output_path, read_user_input,
    resolve_output_path, validate_input_path, validate_output_path,
};
use log::{error, info};
use nalgebra::Vector3;
use rayon::prelude::*;
use std::error::Error;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use strapdown::messages::{Event, EventStream, MeasurementScheduler, build_event_stream};
use strapdown::rbpf::{RaoBlackwellizedParticleFilter, RbpfConfig};

// Geophysical navigation imports (feature-gated)
#[cfg(feature = "geonav")]
use geonav::{
    DEFAULT_GRAVITY_NOISE_MGAL, DEFAULT_MAGNETIC_NOISE_NT, GeoBiasLayout, GeoMap,
    GeophysicalAiding, GeophysicalMeasurementType, GravityResolution, MagneticResolution,
    NAVIGATION_AND_IMU_BIAS_STATE_DIM, NAVIGATION_STATE_DIM,
    build_event_stream as geo_build_event_stream,
};
use rand::SeedableRng;
use rand::rngs::StdRng;
#[cfg(feature = "geonav")]
use std::rc::Rc;
// Unconditional since #259: the RBPF event loop drives the filter through this trait, not
// through inherent methods, so it is needed with or without the geonav feature.
use strapdown::NavigationFilter;
use strapdown::gating::{
    DEFAULT_FORCED_UPDATE_AFTER, DEFAULT_REJECTION_INFLATION, GateRecovery, InnovationGate,
};
use strapdown::sim::HealthLimits;
use strapdown::sim::health::HealthMonitor;
#[cfg(feature = "geonav")]
use strapdown::sim::run_closed_loop_with_geo;
use strapdown::sim::{
    ClosedLoopConfig, EkfConfig, EskfConfig, ExecutionLimits, ExecutionMonitor, ExtraStateLayout,
    FaultArgs, FilterType, NavigationResult, ParticleFilterType, SchedulerArgs, SimulationConfig,
    SimulationMode, SyntheticConfig, TestDataRecord, UkfConfig, build_fault, build_scheduler,
    check_declared_frame, dead_reckoning, generate_synthetic, initialize_ekf, initialize_eskf,
    initialize_ukf, run_closed_loop,
};
#[cfg(feature = "geonav")]
use strapdown::sim::{DEFAULT_PROCESS_NOISE_DENSITY, GeoResolution, GeophysicalConfig};

/// The `--log-level` default.
///
/// Named rather than written twice because the `--config` path distinguishes "the user typed
/// a level" from "this is the default" by comparing against it, and a literal that drifts
/// from the flag's default silently turns that test into "always".
const DEFAULT_LOG_LEVEL: &str = "info";

const LONG_ABOUT: &str =
    "STRAPDOWN SIM: A simulation and analysis tool for strapdown inertial navigation systems.

This program can operate in three modes: open-loop, closed-loop, and particle-filter.

- Open-loop mode: Relies solely on inertial measurements (IMU) and an initial position estimate 
  for dead reckoning. Useful for high-accuracy IMUs with drift rates ≤1 nm per 24 hours.

- Closed-loop mode: Incorporates GNSS measurements to correct IMU drift using either an 
  Unscented Kalman Filter (UKF) or Extended Kalman Filter (EKF). Supports GNSS degradation 
  scenarios including jamming, reduced update rates, and spoofing.

- Particle-filter mode: Uses particle-based state estimation, supporting both standard and 
  Rao-Blackwellized implementations. CURRENTLY IN DEVELOPMENT!!!

You can run simulations either by:
  1. Loading all parameters from a configuration file (TOML/JSON/YAML)
  2. Specifying parameters via command-line flags

For dataset format details, see the documentation or use --help with specific subcommands.";

/// Command line arguments
#[derive(Parser)]
#[command(author, version, about = "A simulation and analysis tool for strapdown inertial navigation systems.", long_about = LONG_ABOUT)]
struct Cli {
    /// Run simulation from a configuration file (TOML/JSON/YAML)
    /// This option overrides any subcommand arguments
    #[arg(short, long, global = true)]
    config: Option<PathBuf>,

    /// Command to execute (ignored if --config is provided)
    #[command(subcommand)]
    command: Option<Command>,

    /// Log level (off, error, warn, info, debug, trace)
    #[arg(long, default_value = DEFAULT_LOG_LEVEL, global = true)]
    log_level: String,

    /// Log file path (if not specified, logs to stderr)
    #[arg(long, global = true)]
    log_file: Option<PathBuf>,

    /// Run simulations in parallel when processing multiple files
    #[arg(long, global = true)]
    parallel: bool,

    /// Generate performance plot comparing navigation output to GPS measurements
    #[arg(long, global = true)]
    plot: bool,
}

/// Top-level commands
#[derive(Subcommand, Clone)]
enum Command {
    #[command(
        name = "dr",
        about = "Run simulation in dead reckoning mode",
        long_about = "Run INS simulation in dead reckoning mode. In this mode, only inertial measurements (IMU) and an initial position estimate are used to propagate the navigation solution. External measurements like GNSS are not incorporated."
    )]
    DeadReckoning(SimArgs),
    #[command(
        name = "ol",
        about = "Run simulation in open-loop mode",
        long_about = "Run INS simulation in an open-loop (feed-forward) mode. In this mode, an initial position estimate and inertial measurements (IMU) are used to propagate the navigation solution. A Kalman filter (EKF or UKF) is used to estimate the errors to the navigation solution from GNSS measurements and apply the correction. Various GNSS degradation scenarios can be simulated, including jamming, reduced update rates, and spoofing."
    )]
    OpenLoop(SimArgs),
    #[command(
        name = "cl",
        about = "Run simulation in closed-loop mode",
        long_about = "Run INS simulation in a closed-loop (feedback) mode. In this mode, GNSS measurements are incorporated to correct for IMU drift and directly reset or update the navigation states using either an Unscented Kalman Filter (UKF) or Extended Kalman Filter (EKF). Various GNSS degradation scenarios can be simulated, including jamming, reduced update rates, and spoofing."
    )]
    ClosedLoop(ClosedLoopSimArgs),

    #[command(
        name = "pf",
        about = "Run simulation using particle filter.",
        long_about = "Run INS simulation using a particle filter for state estimation. This mode supports both standard and Rao-Blackwellized particle filter implementations. Various GNSS degradation scenarios can be simulated, including jamming, reduced update rates, and spoofing."
    )]
    ParticleFilter(ParticleFilterSimArgs),

    #[command(name = "config", about = "Generate a template configuration file")]
    CreateConfig,

    #[command(
        name = "syn",
        about = "Generate a synthetic INS trajectory",
        long_about = "Generate synthetic IMU, GNSS, and barometric sensor data from a defined \
initial kinematic state. The trajectory propagates at constant nav-frame velocity with constant \
body-frame angular velocity (zero linear and angular acceleration). Perfect IMU increments are \
computed via inverse mechanization then degraded per the selected IMU quality grade.\n\n\
Without --no-noise: outputs noisy TestDataRecord CSV compatible with 'cl' and 'pf' modes.\n\
With --no-noise: outputs 9-state kinematic truth in NavigationResult CSV format."
    )]
    Synthetic(SyntheticArgs),
}

/// Arguments for the `syn` (synthetic trajectory) command
#[derive(Args, Clone, Debug)]
struct SyntheticArgs {
    /// Output CSV file path
    #[arg(short, long)]
    output: PathBuf,

    /// Trajectory duration in seconds
    #[arg(long, default_value_t = 300.0)]
    duration_s: f64,

    /// IMU sample rate in Hz
    #[arg(long, default_value_t = 10.0)]
    sample_rate_hz: f64,

    /// IMU quality grade (controls noise and bias levels)
    #[arg(long, value_enum, default_value_t = strapdown::IMUQuality::Consumer)]
    imu_grade: strapdown::IMUQuality,

    /// Output 9-state kinematic truth (`NavigationResult` format) instead of noisy sensor data
    #[arg(long)]
    no_noise: bool,

    /// Random seed for reproducibility
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Initial latitude in degrees (WGS84)
    #[arg(long, default_value_t = 0.0)]
    latitude_deg: f64,

    /// Initial longitude in degrees (WGS84)
    #[arg(long, default_value_t = 0.0)]
    longitude_deg: f64,

    /// Initial altitude in meters
    #[arg(long, default_value_t = 0.0)]
    altitude_m: f64,

    /// Initial northward velocity in m/s
    #[arg(long, default_value_t = 0.0)]
    velocity_north_mps: f64,

    /// Initial eastward velocity in m/s
    #[arg(long, default_value_t = 0.0)]
    velocity_east_mps: f64,

    /// Initial vertical velocity in m/s. Positive DOWN by default (NED); with `--enu` the
    /// sign reverses and positive is UP, because the value is the state's vertical velocity
    /// and that axis points the other way. The flag keeps its NED name for compatibility.
    #[arg(long, default_value_t = 0.0)]
    velocity_down_mps: f64,

    /// Initial roll angle in degrees
    #[arg(long, default_value_t = 0.0)]
    roll_deg: f64,

    /// Initial pitch angle in degrees
    #[arg(long, default_value_t = 0.0)]
    pitch_deg: f64,

    /// Initial yaw (heading) angle in degrees
    #[arg(long, default_value_t = 0.0)]
    yaw_deg: f64,

    /// Constant body roll rate in degrees/s (angular velocity about x-axis)
    #[arg(long, default_value_t = 0.0)]
    angular_velocity_x_dps: f64,

    /// Constant body pitch rate in degrees/s (angular velocity about y-axis)
    #[arg(long, default_value_t = 0.0)]
    angular_velocity_y_dps: f64,

    /// Constant body yaw rate in degrees/s (angular velocity about z-axis)
    #[arg(long, default_value_t = 0.0)]
    angular_velocity_z_dps: f64,

    /// GNSS horizontal position noise standard deviation in meters
    #[arg(long, default_value_t = 2.5)]
    gnss_horizontal_noise_m: f64,

    /// GNSS vertical position noise standard deviation in meters
    #[arg(long, default_value_t = 5.0)]
    gnss_vertical_noise_m: f64,

    /// Barometric pressure noise standard deviation in Pascals
    #[arg(long, default_value_t = 50.0)]
    baro_noise_std_pa: f64,

    /// Magnetometer noise standard deviation in microtesla, per axis.
    ///
    /// The default is a consumer three-axis magnetometer's own noise against a field of
    /// roughly 50 uT.
    #[arg(long, default_value_t = 0.5)]
    mag_noise_std_ut: f64,

    /// Magnetometer hard-iron bias standard deviation in microtesla, per axis.
    ///
    /// Drawn once per run and held constant, the way a hard iron offset behaves. Off by
    /// default: it biases heading in a way no filter here can observe, so switching it on
    /// makes the yaw column measure the bias rather than the filter.
    #[arg(long, default_value_t = 0.0)]
    mag_hard_iron_std_ut: f64,

    /// Emit the trajectory in the ENU convention rather than NED.
    ///
    /// The mirror image of `--enu` on the simulation subcommands, so that `syn --enu` output
    /// is what `dr --enu` expects and plain `syn` output is what plain `dr` expects.
    #[arg(long)]
    enu: bool,
}

/// Common simulation arguments for input/output
#[derive(Args, Clone, Debug)]
struct SimArgs {
    /// Input CSV file path or directory containing CSV files
    /// If a directory is provided, all CSV files in it will be processed
    #[arg(short, long, value_parser)]
    input: PathBuf,

    /// Output CSV file path, or a directory to write results into
    /// A path ending in .csv is treated as a file: a single input writes straight to it,
    /// and multiple inputs write {`output_stem`}_{`input_stem}.csv` beside it.
    /// Any other path is treated as a directory, and each input writes to its own file
    /// name inside it. Writing results over an input file is refused.
    #[arg(short, long, value_parser)]
    output: PathBuf,

    /// Max wall-clock time as a ratio of simulated duration (<= 0 disables)
    #[arg(long, default_value_t = strapdown::sim::DEFAULT_MAX_WALL_CLOCK_RATIO)]
    max_wall_clock_ratio: f64,

    /// Max wall-clock time per trajectory in seconds (<= 0 disables)
    #[arg(long, default_value_t = strapdown::sim::DEFAULT_MAX_WALL_CLOCK_S)]
    max_wall_clock_s: f64,

    /// Max wall-clock time without progress in seconds (<= 0 disables)
    #[arg(long, default_value_t = strapdown::sim::DEFAULT_MAX_NO_PROGRESS_S)]
    max_no_progress_s: f64,

    /// Minimum latitude the filter estimate may reach, in degrees, before the run is failed
    #[arg(long, default_value_t = strapdown::sim::DEFAULT_HEALTH_LAT_MIN_RAD.to_degrees())]
    health_lat_min_deg: f64,

    /// Maximum latitude the filter estimate may reach, in degrees, before the run is failed
    #[arg(long, default_value_t = strapdown::sim::DEFAULT_HEALTH_LAT_MAX_RAD.to_degrees())]
    health_lat_max_deg: f64,

    /// Minimum longitude the filter estimate may reach, in degrees, before the run is failed
    #[arg(long, default_value_t = strapdown::sim::DEFAULT_HEALTH_LON_MIN_RAD.to_degrees())]
    health_lon_min_deg: f64,

    /// Maximum longitude the filter estimate may reach, in degrees, before the run is failed
    #[arg(long, default_value_t = strapdown::sim::DEFAULT_HEALTH_LON_MAX_RAD.to_degrees())]
    health_lon_max_deg: f64,

    /// Minimum altitude in metres above the ellipsoid before the run is failed
    #[arg(long, default_value_t = strapdown::sim::DEFAULT_HEALTH_ALT_MIN_M)]
    health_alt_min_m: f64,

    /// Maximum altitude in metres above the ellipsoid before the run is failed
    #[arg(long, default_value_t = strapdown::sim::DEFAULT_HEALTH_ALT_MAX_M)]
    health_alt_max_m: f64,

    /// Max velocity vector magnitude in m/s before the run is failed
    #[arg(long, default_value_t = strapdown::sim::DEFAULT_HEALTH_SPEED_MPS_MAX)]
    health_speed_mps_max: f64,

    /// Largest variance tolerated on the covariance diagonal before the run is failed
    #[arg(long, default_value_t = strapdown::sim::DEFAULT_HEALTH_COV_DIAG_MAX)]
    health_cov_diag_max: f64,

    /// NIS above which a measurement update counts as an outlier
    #[arg(long, default_value_t = strapdown::sim::DEFAULT_NIS_POS_MAX)]
    nis_pos_max: f64,

    /// Consecutive NIS exceedances that fail the run
    #[arg(long, default_value_t = strapdown::sim::DEFAULT_NIS_POS_CONSEC_FAIL)]
    nis_pos_consec_fail: usize,

    /// Interpret the input records in the ENU convention rather than NED.
    ///
    /// Sensor Logger exports are ENU: at rest their specific force lands on the device's
    /// up-axis at +9.8 m/s^2. The default is NED, which is what `syn` writes and what the
    /// library mechanizes in. A CSV carries no frame tag, so this cannot be inferred -- but
    /// declaring it wrongly is caught before propagation rather than integrated at 2 g.
    #[arg(long)]
    enu: bool,
}

/// Geophysical measurement arguments (feature-gated)
#[cfg(feature = "geonav")]
#[derive(Args, Clone, Debug)]
struct GeophysicalArgs {
    /// Enable geophysical navigation
    #[arg(long)]
    geo: bool,

    /// Gravity map resolution
    #[arg(long, value_enum, requires = "geo")]
    gravity_resolution: Option<GeoResolution>,

    /// Gravity measurement bias (mGal)
    #[arg(long, requires = "geo")]
    gravity_bias: Option<f64>,

    /// Gravity measurement noise std dev (mGal)
    #[arg(long, default_value_t = 100.0, requires = "geo")]
    gravity_noise_std: f64,

    /// Gravity map file path
    #[arg(long, requires = "geo")]
    gravity_map_file: Option<PathBuf>,

    /// Magnetic map resolution
    #[arg(long, value_enum, requires = "geo")]
    magnetic_resolution: Option<GeoResolution>,

    /// Magnetic measurement bias (nT)
    #[arg(long, requires = "geo")]
    magnetic_bias: Option<f64>,

    /// Magnetic measurement noise std dev (nT)
    #[arg(long, default_value_t = 150.0, requires = "geo")]
    magnetic_noise_std: f64,

    /// Magnetic map file path
    #[arg(long, requires = "geo")]
    magnetic_map_file: Option<PathBuf>,

    /// Seconds between geophysical measurements -- an interval, not a frequency.
    ///
    /// `alias` keeps the old `--geo-frequency-s` spelling working.
    #[arg(long, alias = "geo-frequency-s", requires = "geo")]
    geo_interval_s: Option<f64>,
}

/// Empty stub when geonav feature is disabled
#[cfg(not(feature = "geonav"))]
#[derive(Args, Clone, Debug, Default)]
struct GeophysicalArgs {}

/// Seconds over which a map bias is allowed to drift by about its own initial sigma.
///
/// Turns an initial standard deviation into a process-noise density the same way
/// [`strapdown::sim::BARO_BIAS_PROCESS_NOISE_M2_PER_S`] turns a per-hour drift into one:
/// `q = sigma^2 / tau`, so the variance the bias accumulates over `tau` seconds is back up
/// to `sigma^2`. An hour is the order of a recording, not a tuned number -- pass the
/// `--*-bias-process-noise-std` flags to set the rate directly.
#[cfg(feature = "geonav")]
const GEO_BIAS_DRIFT_TIME_CONSTANT_S: f64 = 3600.0;

/// Prior and random-walk rate for the geophysical map-bias states.
///
/// Flattened into both `cl` and `pf`, and resolved for either through [`geo_bias_setup`].
/// `pf` used to carry its own `--geo-bias-init-std`/`--geo-bias-process-noise-std` instead --
/// one pair for both channels -- and ignored these, along with `--gravity-bias` and
/// `--magnetic-bias`. Those two flags are refused now, by name; see
/// [`refuse_removed_particle_filter_flags`].
///
/// Both knobs are **standard deviations** in their channel's own units. The Kalman arms square
/// them into a prior variance and a process-noise density; the particle filter takes them as
/// they are, entry for entry, in `RbpfConfig::extra_state_init_std` and
/// `extra_state_process_noise_std`. Gravity is in mGal and magnetic in nT -- three orders of
/// magnitude apart, which is why there is a pair per channel rather than one pair for both.
#[cfg(feature = "geonav")]
#[derive(Args, Clone, Debug)]
struct GeophysicalBiasArgs {
    /// Initial standard deviation of the gravity map bias (mGal).
    ///
    /// Defaults to `--gravity-noise-std`: the map bias and the measurement noise are of the
    /// same order, and that is the prior this path always meant to carry.
    #[arg(long = "gravity-bias-init-std", requires = "geo")]
    gravity_prior_std: Option<f64>,

    /// Random-walk rate of the gravity map bias, a standard deviation in mGal per sqrt(s).
    ///
    /// Defaults to the prior spread over [`GEO_BIAS_DRIFT_TIME_CONSTANT_S`].
    #[arg(long = "gravity-bias-process-noise-std", requires = "geo")]
    gravity_drift_rate: Option<f64>,

    /// Initial standard deviation of the magnetic map bias (nT).
    ///
    /// Defaults to `--magnetic-noise-std`. A platform's own field is the thing this state
    /// exists to absorb, so a recording made inside a vehicle wants far more than the
    /// default -- thousands of nT, not hundreds.
    #[arg(long = "magnetic-bias-init-std", requires = "geo")]
    magnetic_prior_std: Option<f64>,

    /// Random-walk rate of the magnetic map bias, a standard deviation in nT per sqrt(s).
    ///
    /// Defaults to the prior spread over [`GEO_BIAS_DRIFT_TIME_CONSTANT_S`].
    #[arg(long = "magnetic-bias-process-noise-std", requires = "geo")]
    magnetic_drift_rate: Option<f64>,
}

/// Empty stub when geonav feature is disabled
#[cfg(not(feature = "geonav"))]
#[derive(Args, Clone, Debug, Default)]
struct GeophysicalBiasArgs {}

/// Closed-loop simulation arguments
#[derive(Args, Clone, Debug)]
struct ClosedLoopSimArgs {
    /// Common simulation input/output arguments
    #[command(flatten)]
    sim: SimArgs,

    /// Filter type to use for closed-loop navigation (default: the 15-state ESKF)
    #[arg(long, value_enum, default_value_t = FilterType::default())]
    filter: FilterType,

    /// UKF alpha parameter (sigma point spread)
    ///
    /// Defaults to the same `0.1` a configuration file does, not the textbook `1e-3` this
    /// flag carried. `alpha` is a numerical-conditioning parameter here as much as a tuning
    /// one -- `1e-3` costs six significant digits per step to cancellation in the weighted
    /// sigma-point mean (#399) -- and `0.1` is the value that was measured and adopted for
    /// `ClosedLoopConfig::ukf_alpha`. This flag was not moved with it, so `--filter ukf` and
    /// a config file naming the same filter ran with sigma-point spreads two orders of
    /// magnitude apart, and the two were not comparable.
    #[arg(long, default_value_t = strapdown::sim::DEFAULT_UKF_ALPHA)]
    ukf_alpha: f64,

    /// UKF beta parameter (prior distribution)
    #[arg(long, default_value_t = 2.0)]
    ukf_beta: f64,

    /// UKF kappa parameter (secondary spread control)
    #[arg(long, default_value_t = 0.0)]
    ukf_kappa: f64,

    /// RNG seed for stochastic processes
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Estimate a barometric altitude bias as an extra filter state.
    ///
    /// A barometer's reference pressure drifts and a filter that models the reading as
    /// unbiased pushes that drift into altitude. On the reference recording this takes
    /// 3-sigma vertical containment from about 0.40 to 0.84 against an ideal of 0.9973 and
    /// improves vertical RMSE by 45%, on all three filters. Adds a `baro_bias` column to the
    /// output.
    ///
    /// **On by default as of 1.0** -- pass `--no-estimate-baro-bias` to turn it off. The flag
    /// is spelled negatively because the default is now on; the old `--estimate-baro-bias`
    /// is gone rather than kept as a no-op, so a script that passes it fails loudly instead of
    /// quietly meaning nothing.
    #[arg(long = "no-estimate-baro-bias", action = clap::ArgAction::SetFalse)]
    estimate_baro_bias: bool,

    /// Reject measurements whose NIS exceeds this chi-squared confidence level.
    ///
    /// Omitted, every measurement is accepted -- the behaviour of every release so
    /// far. Given (e.g. `--gate-confidence 0.999`), each update is tested against the
    /// chi-squared quantile for that measurement's own degrees of freedom, so one
    /// number stays meaningful across 1-DOF baro, 3-DOF position and 5-DOF
    /// position+velocity aiding. Must lie strictly inside (0, 1).
    #[arg(long, value_name = "PROBABILITY")]
    gate_confidence: Option<f64>,

    /// Grow the filter's uncertainty by this factor each time the gate rejects a fix.
    ///
    /// Only used together with `--gate-confidence`. Applied in the directions the rejected
    /// measurement observed, not to the whole covariance. A gate with no recovery path is a
    /// one-way door: the filter keeps drifting while the covariance it judges the next fix
    /// against does not grow, so one rejection begets the next. Must be at least 1.0; 1.0
    /// disables inflation.
    #[arg(long, value_name = "FACTOR", default_value_t = DEFAULT_REJECTION_INFLATION)]
    gate_inflation: f64,

    /// Apply a measurement despite the gate after this many consecutive rejections.
    ///
    /// Only used together with `--gate-confidence`. A belief contradicted this many
    /// times running is likelier to be wrong than the sensor contradicting it. Zero
    /// never forces an update, which leaves `--gate-inflation` as the only way back.
    #[arg(long, value_name = "COUNT", default_value_t = DEFAULT_FORCED_UPDATE_AFTER)]
    gate_force_after: usize,

    /// GNSS scheduler settings (dropouts / reduced rate)
    #[command(flatten)]
    scheduler: SchedulerArgs,

    /// Fault model settings (corrupt measurement content)
    #[command(flatten)]
    fault: FaultArgs,

    /// Geophysical navigation options (optional, requires --features geonav)
    #[command(flatten)]
    geo: GeophysicalArgs,

    /// Map-bias prior and random-walk rate (optional, requires --features geonav)
    #[command(flatten)]
    geo_bias: GeophysicalBiasArgs,
}

/// Particle filter simulation arguments
#[derive(Args, Clone, Debug)]
struct ParticleFilterSimArgs {
    /// Common simulation input/output arguments
    #[command(flatten)]
    sim: SimArgs,

    /// Particle filter type
    #[arg(long, value_enum, default_value_t = ParticleFilterType::RaoBlackwellized)]
    filter_type: ParticleFilterType,

    /// RNG seed for stochastic processes
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Number of particles
    #[arg(long, default_value_t = 100)]
    num_particles: usize,

    /// Position uncertainty standard deviation (meters)
    #[arg(long, default_value_t = 10.0)]
    position_std: f64,

    /// Velocity uncertainty standard deviation (m/s)
    #[arg(long, default_value_t = 1.0)]
    velocity_std: f64,

    /// Attitude uncertainty standard deviation (radians)
    #[arg(long, default_value_t = 0.1)]
    attitude_std: f64,

    /// Position random-walk rate for the particle filter as `[north, east, up]` in
    /// m/sqrt(s). The filter forms the per-step standard deviation as this times
    /// `sqrt(dt)`, so the value is unchanged at a 1 s step and the spread it produces
    /// depends on elapsed time rather than on the log's sample rate.
    ///
    /// Examples:
    /// - `--process-noise-std-m 1 1 2`
    /// - `--process-noise-std-m 1,1,2`
    #[arg(long, value_delimiter = ',', num_args = 3, default_value = "1,1,1")]
    process_noise_std_m: Vec<f64>,

    /// Velocity random-walk rate (m/s per sqrt(s); unchanged at a 1 s step).
    #[arg(long, default_value_t = 1e-3)]
    velocity_process_noise_std_mps: f64,

    /// Attitude random-walk rate (rad per sqrt(s); unchanged at a 1 s step).
    #[arg(long, default_value_t = 0.01)]
    attitude_process_noise_std_rad: f64,

    /// GNSS scheduler settings (dropouts / reduced rate)
    #[command(flatten)]
    scheduler: SchedulerArgs,

    /// Fault model settings (corrupt measurement content)
    #[command(flatten)]
    fault: FaultArgs,

    /// Geophysical navigation options (optional, requires --features geonav)
    #[command(flatten)]
    geo: GeophysicalArgs,

    /// Prior and random walk of each map bias, per channel -- the same flags `cl` takes.
    #[command(flatten)]
    geo_bias: GeophysicalBiasArgs,

    /// Apply zero-vertical-velocity pseudo-measurement (RBPF only).
    #[arg(long, default_value_t = true)]
    zero_vertical_velocity: bool,

    /// Std dev for zero-vertical-velocity pseudo-measurement (m/s).
    #[arg(long, default_value_t = 0.1)]
    zero_vertical_velocity_std_mps: f64,

    /// Removed. Accepted only so it can be refused by name; see
    /// [`refuse_removed_particle_filter_flags`].
    #[arg(long, hide = true)]
    geo_bias_init_std: Option<f64>,

    /// Removed. Accepted only so it can be refused by name; see
    /// [`refuse_removed_particle_filter_flags`].
    #[arg(long, hide = true)]
    geo_bias_process_noise_std: Option<f64>,
}

/// Refuse the particle filter's two retired map-bias flags, naming what replaced them.
///
/// `--geo-bias-init-std` and `--geo-bias-process-noise-std` set one prior and one random walk
/// for every map bias, whatever its unit, while `pf` ignored the per-channel flags `cl` takes.
/// Both subcommands now read those. Dropping the old flags outright would still fail loudly --
/// clap rejects an unknown argument -- but would not say where the setting went.
///
/// # Errors
/// When either retired flag is passed.
fn refuse_removed_particle_filter_flags(
    args: &ParticleFilterSimArgs,
) -> Result<(), Box<dyn Error>> {
    if args.geo_bias_init_std.is_some() {
        return Err(
            "`--geo-bias-init-std` was removed: it was one prior for every map bias, in no \
                    particular unit. Pass `--gravity-bias-init-std` (mGal) and/or \
                    `--magnetic-bias-init-std` (nT), or omit them to default to each channel's \
                    noise standard deviation"
                .into(),
        );
    }
    if args.geo_bias_process_noise_std.is_some() {
        return Err(
            "`--geo-bias-process-noise-std` was removed: it was one random walk for every \
                    map bias, in no particular unit. Pass `--gravity-bias-process-noise-std` \
                    (mGal per sqrt(s)) and/or `--magnetic-bias-process-noise-std` (nT per \
                    sqrt(s)), or omit them to default to each prior spread over an hour"
                .into(),
        );
    }
    Ok(())
}

/// Arguments for create-config command
#[derive(Args, Clone, Debug)]
struct CreateConfigArgs {
    /// Output file path for the config file
    /// File extension determines format: .json, .yaml/.yml, or .toml (recommended)
    #[arg(short, long, value_parser)]
    output: PathBuf,

    /// Simulation mode for the template
    #[arg(short, long, value_enum, default_value_t = SimulationMode::ClosedLoop)]
    mode: SimulationMode,
}

const fn execution_limits_from_args(args: &SimArgs) -> ExecutionLimits {
    ExecutionLimits {
        max_wall_clock_ratio: args.max_wall_clock_ratio,
        max_wall_clock_s: args.max_wall_clock_s,
        max_no_progress_s: args.max_no_progress_s,
    }
}

/// The angular bands are taken in degrees on the command line and converted here, because a
/// clap flag carrying a radian literal is unreadable at the call site.
const fn health_limits_from_args(args: &SimArgs) -> HealthLimits {
    HealthLimits {
        lat_rad: (
            args.health_lat_min_deg.to_radians(),
            args.health_lat_max_deg.to_radians(),
        ),
        lon_rad: (
            args.health_lon_min_deg.to_radians(),
            args.health_lon_max_deg.to_radians(),
        ),
        alt_m: (args.health_alt_min_m, args.health_alt_max_m),
        speed_mps_max: args.health_speed_mps_max,
        cov_diag_max: args.health_cov_diag_max,
        nis_pos_max: args.nis_pos_max,
        nis_pos_consec_fail: args.nis_pos_consec_fail,
    }
}

/// The wall-clock and numerical guards a run is bounded by, carried as one value so that
/// `run_single_closed_loop_simulation` does not grow a thirteenth parameter.
#[derive(Clone, Debug)]
struct RunLimits {
    execution: ExecutionLimits,
    health: HealthLimits,
}

impl RunLimits {
    const fn from_args(args: &SimArgs) -> Self {
        Self {
            execution: execution_limits_from_args(args),
            health: health_limits_from_args(args),
        }
    }
}

/// Process a single CSV file with the given configuration
fn process_file(
    input_file: &Path,
    output: &Path,
    all_inputs: &[PathBuf],
    config: &SimulationConfig,
) -> Result<(), Box<dyn Error>> {
    info!("Processing file: {}", input_file.display());

    // Load sensor data
    let records = load_records(input_file)?;

    // Execute based on mode
    match config.mode {
        SimulationMode::DeadReckoning => {
            info!("Running dead reckoning simulation");
            let results = dead_reckoning(&records, config.is_enu)?;
            info!("Generated {} navigation results", results.len());

            let output_file = resolve_output_path(output, input_file, all_inputs)?;
            NavigationResult::to_csv(&results, &output_file)?;
            info!("Results written to {}", output_file.display());
            Ok(())
        }
        SimulationMode::OpenLoop => {
            info!("Open-loop mode is not yet fully implemented");
            Err("Open-loop mode is not yet fully implemented".into())
        }
        SimulationMode::ClosedLoop => {
            // Check the declared frame against the WHOLE leading window, not just the first
            // record. `initialize_{ukf,ekf,eskf}` run the same guard, but they are handed a
            // single `TestDataRecord`, and a one-sample window is weak in both directions: one
            // NaN or transient first sample disables it entirely (letting the 2 g double-count
            // through on the default mode), and one ordinary motion sample above 1.5 g
            // false-rejects a correctly declared file. The particle-filter paths below already
            // do this; closed loop is the default mode and needs it more, not less (#296).
            check_declared_frame(&records, config.is_enu)?;

            // A `[geophysical]` section used to be refused here. The geophysical runner was
            // reachable only through `--geo` on the command line, so a closed-loop config
            // carrying maps parsed cleanly and ran an ordinary non-geophysical simulation
            // while reporting success -- and refusing was chosen over wiring it up. The RBPF
            // arm below never had the restriction, which left the two filter families
            // configured in incompatible ways: one took a `--config`, the other a
            // twelve-flag command line that had to be kept in step with it by hand.
            //
            // Both now resolve to the same `GeoClosedLoopSettings` and the same runner.
            #[cfg(feature = "geonav")]
            if let Some(geo_config) = config.geophysical.as_ref() {
                let settings = geo_settings_from_config(config, geo_config);
                settings.validate()?;
                let output_file = resolve_output_path(output, input_file, all_inputs)?;
                return run_geo_closed_loop_file(&settings, &records, input_file, &output_file);
            }

            #[cfg(not(feature = "geonav"))]
            if config.geophysical.is_some() {
                return Err(
                    "a [geophysical] section requires the geonav feature: rebuild \
                     with `--features geonav`. Running this config without it would \
                     silently ignore the maps and produce an ordinary non-geophysical \
                     result."
                        .into(),
                );
            }

            let filter_config = config.closed_loop.clone().unwrap_or_default();

            // `ukf_alpha`/`beta`/`kappa` are read from the file rather than left at the
            // constructor's defaults. This path ignored all three, so a config file setting
            // `ukf_alpha` got the 1e-3 default silently -- the same shape of defect as #392.
            // Built through `KalmanSettings` so the geophysical arm above builds the same
            // filters and only appends its map biases.
            let kalman = KalmanSettings::from_closed_loop(&filter_config, config.is_enu);
            let ukf_config = kalman.ukf_config();
            let ekf_config = kalman.ekf_config();
            let eskf_config = kalman.eskf_config();

            // Derived from the filter, not asked for a second time; see
            // `run_single_closed_loop_simulation` for why (#372).
            let aiding = {
                let mut built = config.aiding.clone();
                built.baro_bias_index = match filter_config.filter {
                    FilterType::Ukf => ukf_config.baro_bias_index(),
                    FilterType::Ekf => ekf_config.baro_bias_index(),
                    FilterType::Eskf => eskf_config.baro_bias_index(),
                };
                built
            };

            let event_stream = build_event_stream(&records, &aiding, config.is_enu)?;
            info!(
                "Initialized event stream with {} events",
                event_stream.events.len()
            );
            let execution_limits = config.execution_limits.clone();
            let health_limits = config.health_limits.clone();

            let results = match filter_config.filter {
                FilterType::Ukf => {
                    let mut ukf = initialize_ukf(&records[0].clone(), ukf_config)?;
                    info!("Initialized UKF");
                    ukf.set_innovation_gate(filter_config.innovation_gate);
                    ukf.set_gate_recovery(filter_config.gate_recovery);
                    run_closed_loop(
                        &mut ukf,
                        event_stream,
                        Some(health_limits),
                        Some(execution_limits),
                    )
                }
                FilterType::Ekf => {
                    let mut ekf = initialize_ekf(&records[0].clone(), ekf_config)?;
                    info!("Initialized EKF");
                    ekf.set_innovation_gate(filter_config.innovation_gate);
                    ekf.set_gate_recovery(filter_config.gate_recovery);
                    run_closed_loop(
                        &mut ekf,
                        event_stream,
                        Some(health_limits),
                        Some(execution_limits),
                    )
                }
                FilterType::Eskf => {
                    let mut eskf = initialize_eskf(&records[0].clone(), eskf_config)?;
                    info!("Initialized ESKF");
                    eskf.set_innovation_gate(filter_config.innovation_gate);
                    eskf.set_gate_recovery(filter_config.gate_recovery);
                    run_closed_loop(
                        &mut eskf,
                        event_stream,
                        Some(health_limits),
                        Some(execution_limits),
                    )
                }
            };

            let output_file = resolve_output_path(output, input_file, all_inputs)?;
            match results {
                Ok(ref nav_results) => {
                    NavigationResult::to_csv(nav_results, &output_file)?;
                    info!("Results written to {}", output_file.display());

                    // Generate performance plot if requested
                    #[cfg(feature = "plotting")]
                    if config.generate_plot {
                        let plot_path = output_file.with_extension("png");
                        info!("Generating performance plot at {}", plot_path.display());

                        match plotting::plot_performance(nav_results, &records, &plot_path) {
                            Ok(()) => {
                                info!("Performance plot generated successfully");
                            }
                            Err(e) => {
                                error!("Failed to generate performance plot: {e}");
                                // Don't fail the entire process if plotting fails
                            }
                        }
                    }

                    #[cfg(not(feature = "plotting"))]
                    if config.generate_plot {
                        error!(
                            "Plotting requested but 'plotting' feature not enabled. Rebuild with --features plotting"
                        );
                    }

                    Ok(())
                }
                Err(e) => {
                    error!("Error running closed-loop simulation: {e}");
                    Err(e.into())
                }
            }
        }
        SimulationMode::ParticleFilter => {
            info!("Running particle filter simulation");

            #[cfg(feature = "geonav")]
            let (gravity_map, magnetic_map, geo_interval_s, gravity_noise_std, magnetic_noise_std) = {
                if let Some(geo_cfg) = &config.geophysical {
                    let gravity_map = if let Some(res) = geo_cfg.gravity_resolution {
                        let map_path = match &geo_cfg.gravity_map_file {
                            Some(path) => PathBuf::from(path),
                            None => find_gravity_map(input_file)?,
                        };
                        let measurement_type =
                            GeophysicalMeasurementType::Gravity(convert_resolution_gravity(res));
                        Some(Rc::new(GeoMap::load_geomap(&map_path, measurement_type)?))
                    } else {
                        None
                    };

                    let magnetic_map = if let Some(res) = geo_cfg.magnetic_resolution {
                        let map_path = match &geo_cfg.magnetic_map_file {
                            Some(path) => PathBuf::from(path),
                            None => find_magnetic_map(input_file)?,
                        };
                        let measurement_type =
                            GeophysicalMeasurementType::Magnetic(convert_resolution_magnetic(res));
                        Some(Rc::new(GeoMap::load_geomap(&map_path, measurement_type)?))
                    } else {
                        None
                    };

                    (
                        gravity_map,
                        magnetic_map,
                        geo_cfg.geo_interval_s,
                        geo_cfg
                            .gravity_noise_std
                            .unwrap_or(DEFAULT_GRAVITY_NOISE_MGAL),
                        geo_cfg
                            .magnetic_noise_std
                            .unwrap_or(DEFAULT_MAGNETIC_NOISE_NT),
                    )
                } else {
                    (
                        None,
                        None,
                        None,
                        DEFAULT_GRAVITY_NOISE_MGAL,
                        DEFAULT_MAGNETIC_NOISE_NT,
                    )
                }
            };

            // Each map bias's seed, prior and random walk, from `[geophysical]` and through the
            // helper the Kalman arms use -- so the bias `analyze geostats` measures reaches this
            // filter as it reaches those. This arm used to read none of them, and started every
            // bias at zero under `[particle_filter] geo_bias_init_std`, one prior for every
            // channel in no particular unit.
            #[cfg(feature = "geonav")]
            let geo_bias = config.geophysical.as_ref().map_or_else(
                || geo_bias_setup(None, None),
                |geo_cfg| {
                    geo_bias_setup(
                        gravity_map
                            .is_some()
                            .then_some(MapBiasPrior::gravity_from_config(geo_cfg)),
                        magnetic_map
                            .is_some()
                            .then_some(MapBiasPrior::magnetic_from_config(geo_cfg)),
                    )
                },
            );

            #[cfg(not(feature = "geonav"))]
            if config.geophysical.is_some() {
                return Err("Geophysical configuration requires the geonav feature".into());
            }

            // One layout, used both to declare the biases on the measurements and to size
            // the filter's extra states below, so the two cannot drift apart. The RBPF
            // carries no IMU bias states, so its map biases follow the navigation states.
            #[cfg(feature = "geonav")]
            let geo_bias_layout = GeoBiasLayout::appended(
                NAVIGATION_STATE_DIM,
                gravity_map.is_some(),
                magnetic_map.is_some(),
            )?;

            #[cfg(feature = "geonav")]
            let event_stream = if gravity_map.is_some() || magnetic_map.is_some() {
                geo_build_event_stream(
                    &records,
                    &config.aiding,
                    config.is_enu,
                    &GeophysicalAiding {
                        gravity_noise_std: gravity_map.as_ref().map(|_| gravity_noise_std),
                        magnetic_noise_std: magnetic_map.as_ref().map(|_| magnetic_noise_std),
                        gravity_map,
                        magnetic_map,
                        interval_s: geo_interval_s,
                        bias_layout: geo_bias_layout,
                    },
                )?
            } else {
                build_event_stream(&records, &config.aiding, config.is_enu)?
            };

            #[cfg(not(feature = "geonav"))]
            let event_stream = build_event_stream(&records, &config.aiding, config.is_enu)?;

            // The particle filter builds its nominal state here rather than through
            // `initialize_*`, so it has to run the frame guard itself.
            check_declared_frame(&records, config.is_enu)?;
            let first = &records[0];
            // Quaternion, not Euler angles: `TestDataRecord`'s roll/pitch/yaw are a
            // different convention from nalgebra's XYZ. See `TestDataRecord::attitude`.
            let attitude = first.attitude();
            let (velocity_north, velocity_east) = first.ground_track_velocity();
            let nominal = strapdown::StrapdownState {
                latitude: first.latitude.to_radians(),
                longitude: first.longitude.to_radians(),
                altitude: first.altitude,
                velocity_north,
                velocity_east,
                velocity_vertical: 0.0,
                attitude,
                // The declared frame, as everywhere else. `syn` writes NED and Sensor Logger
                // writes ENU; `strapdown::sim::check_declared_frame` is what catches the
                // wrong answer (#296).
                is_enu: config.is_enu,
            };

            let pf_cfg = config.particle_filter.clone().unwrap_or_default();
            let rbpf_defaults = RbpfConfig::default();
            let position_init_std_m = if pf_cfg.position_init_std_m.len() == 3 {
                Vector3::new(
                    pf_cfg.position_init_std_m[0],
                    pf_cfg.position_init_std_m[1],
                    pf_cfg.position_init_std_m[2],
                )
            } else {
                rbpf_defaults.position_init_std_m
            };
            let position_process_noise_std_m = if pf_cfg.position_process_noise_std_m.len() == 3 {
                Vector3::new(
                    pf_cfg.position_process_noise_std_m[0],
                    pf_cfg.position_process_noise_std_m[1],
                    pf_cfg.position_process_noise_std_m[2],
                )
            } else {
                rbpf_defaults.position_process_noise_std_m
            };
            #[cfg(feature = "geonav")]
            let geo_bias_dim = geo_bias_layout.map_or(0, |layout| layout.bias_count());
            #[cfg(not(feature = "geonav"))]
            let geo_bias_dim = 0usize;

            // The same placement, restated for `NavigationResult`, which lives in `core` and
            // so cannot name `GeoBiasLayout`. Derived from that layout rather than rebuilt
            // from the map flags, exactly as `run_geo_closed_loop_cli` derives the Kalman one,
            // so where the biases live is decided once. The unaided case is
            // `PARTICLE_NONE` and not `NONE`: this filter's estimate is nine states, not the
            // Kalman filters' fifteen.
            #[cfg(feature = "geonav")]
            let geo_layout = geo_bias_layout.map_or(ExtraStateLayout::PARTICLE_NONE, |layout| {
                ExtraStateLayout::new(
                    layout.state_dim(),
                    layout.gravity_bias().map(|bias| bias.index),
                    layout.magnetic_bias().map(|bias| bias.index),
                )
            });
            #[cfg(not(feature = "geonav"))]
            let geo_layout = ExtraStateLayout::PARTICLE_NONE;
            let mut rbpf = RaoBlackwellizedParticleFilter::new(nominal, {
                let mut built = RbpfConfig::default();
                built.num_particles = pf_cfg.num_particles;
                built.position_init_std_m = position_init_std_m;
                built.velocity_init_std_mps = pf_cfg.velocity_init_std_mps;
                built.attitude_init_std_rad = pf_cfg.attitude_init_std_rad;
                built.position_process_noise_std_m = position_process_noise_std_m;
                built.velocity_process_noise_std_mps = pf_cfg.velocity_process_noise_std_mps;
                built.attitude_process_noise_std_rad = pf_cfg.attitude_process_noise_std_rad;
                built.extra_state_dim = geo_bias_dim;
                #[cfg(feature = "geonav")]
                geo_bias.apply_to_rbpf(&mut built);
                built.seed = config.seed;
                built.zero_vertical_velocity = pf_cfg.zero_vertical_velocity;
                built.zero_vertical_velocity_std_mps = pf_cfg.zero_vertical_velocity_std_mps;
                built
            })?;

            // Geophysical measurements ride the same event stream as every other
            // measurement type, so there is no separate geo path here.
            let results = run_rbpf_event_loop(
                &mut rbpf,
                event_stream,
                &config.execution_limits,
                &config.health_limits,
                geo_layout,
            )?;
            let output_file = resolve_output_path(output, input_file, all_inputs)?;
            NavigationResult::to_csv(&results, &output_file)?;
            info!("Results written to {}", output_file.display());

            #[cfg(feature = "plotting")]
            if config.generate_plot {
                let plot_path = output_file.with_extension("png");
                info!("Generating performance plot at {}", plot_path.display());

                match plotting::plot_performance(&results, &records, &plot_path) {
                    Ok(()) => {
                        info!("Performance plot generated successfully");
                    }
                    Err(e) => {
                        error!("Failed to generate performance plot: {e}");
                    }
                }
            }

            #[cfg(not(feature = "plotting"))]
            if config.generate_plot {
                error!(
                    "Plotting requested but 'plotting' feature not enabled. Rebuild with --features plotting"
                );
            }

            Ok(())
        }
        SimulationMode::Synthetic => Err(
            "Synthetic mode does not process input files; use the 'syn' subcommand directly".into(),
        ),
    }
}

/// Execute simulation from a configuration file
fn run_from_config(
    config_path: &Path,
    cli_parallel: bool,
    cli_plot: bool,
) -> Result<(), Box<dyn Error>> {
    info!("Loading configuration from {}", config_path.display());

    let mut config = SimulationConfig::from_file(config_path)?;

    // Override parallel setting if CLI flag is set
    if cli_parallel {
        config.parallel = true;
    }

    // Override plot setting if CLI flag is set
    if cli_plot {
        config.generate_plot = true;
    }

    info!("Configuration loaded successfully");
    info!("Mode: {:?}", config.mode);

    // Synthetic mode has no input file — handle it separately before path validation
    if matches!(config.mode, SimulationMode::Synthetic) {
        let syn_config = config
            .synthetic
            .ok_or("mode is 'synthetic' but no [synthetic] section found in config file")?;
        let output = Path::new(&syn_config.output);
        if let Some(parent) = output.parent()
            && !parent.as_os_str().is_empty()
        {
            validate_output_path(parent)?;
        }
        let mut rng = StdRng::seed_from_u64(syn_config.seed);
        let (truth, sensors) = generate_synthetic(&syn_config, &mut rng)?;
        let n = truth.len();
        info!(
            "Generated {} synthetic records ({:.1} s at {:.0} Hz)",
            n, syn_config.duration_s, syn_config.sample_rate_hz
        );
        if syn_config.no_noise {
            NavigationResult::to_csv(&truth, output)?;
            info!("Truth trajectory written to {}", output.display());
        } else {
            TestDataRecord::to_csv(&sensors, output)?;
            info!("Sensor records written to {}", output.display());
        }
        return Ok(());
    }

    info!("Input: {}", config.input);
    info!("Output: {}", config.output);
    info!("Parallel: {}", config.parallel);
    info!("Generate plot: {}", config.generate_plot);
    info!(
        "Execution limits: ratio {:.2}, wall-clock {:.1}s, no-progress {:.1}s",
        config.execution_limits.max_wall_clock_ratio,
        config.execution_limits.max_wall_clock_s,
        config.execution_limits.max_no_progress_s
    );

    // Validate paths
    let input = Path::new(&config.input);
    let output = Path::new(&config.output);
    validate_input_path(input)?;
    validate_output_path(output)?;

    // Get all CSV files to process
    let csv_files = get_csv_files(input)?;
    let is_multiple = csv_files.len() > 1;

    if is_multiple {
        info!("Processing {} CSV files from directory", csv_files.len());
        if config.parallel {
            info!("Running in parallel mode");
        }
    }

    // Process files either sequentially or in parallel
    if config.parallel && is_multiple {
        // Parallel processing
        let errors = Mutex::new(Vec::new());

        csv_files.par_iter().for_each(|input_file| {
            match process_file(input_file, output, &csv_files, &config) {
                Ok(()) => {}
                Err(e) => {
                    error!("Error processing {}: {}", input_file.display(), e);
                    // Recover from a poisoned lock rather than panicking. This mutex
                    // guards the list of per-file failures; if another worker panicked
                    // while holding it, turning that into a second panic here loses the
                    // very error report this block exists to produce.
                    errors
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .push((input_file.clone(), e.to_string()));
                }
            }
        });

        let errors = errors
            .into_inner()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if !errors.is_empty() {
            error!("{} file(s) failed to process", errors.len());
            for (file, err) in &errors {
                error!("  {}: {}", file.display(), err);
            }
            return Err(format!("{} file(s) failed to process", errors.len()).into());
        }
    } else {
        // Sequential processing
        let mut failures = 0usize;
        for input_file in &csv_files {
            if let Err(e) = process_file(input_file, output, &csv_files, &config) {
                if !is_multiple {
                    return Err(e);
                }
                failures += 1;
                error!("Error processing {}: {}", input_file.display(), e);
            }
        }
        if failures > 0 {
            error!("{failures} file(s) failed to process");
            return Err(format!("{failures} file(s) failed to process").into());
        }
    }

    Ok(())
}

/// The settings every closed-loop Kalman filter is built from, besides its first record and
/// any geophysical map biases.
///
/// One value, handed to every arm that constructs a UKF, EKF or ESKF: the closed-loop branch of
/// [`process_file`], [`run_single_closed_loop_simulation`] for the command line, and the
/// geophysical runner, which takes the same configs and appends its map-bias states to them
/// (`GeoBiasSetup::apply_to_ukf` and `apply_to_ekf`). So the variants a study compares -- full
/// GNSS, degraded GNSS, and degraded GNSS with map aiding -- run one filter, barometric bias
/// included, and differ only by the GNSS they are given and the map biases an aided run carries.
///
/// The geophysical runner used to build its own. Neither of its filters estimated the
/// barometric bias, and its EKF had a hand-written P0 and Q, so a map-aided result was compared
/// against an unaided one from a different filter and the difference was reported as the map's.
#[derive(Clone, Copy, Debug, PartialEq)]
struct KalmanSettings {
    /// UKF sigma-point spread. Ignored by the EKF and ESKF, like the next two.
    ukf_alpha: f64,
    /// UKF prior-distribution parameter.
    ukf_beta: f64,
    /// UKF secondary spread parameter.
    ukf_kappa: f64,
    /// Estimate a barometric altitude bias as an extra state (#372).
    estimate_baro_bias: bool,
    /// Whether the records are ENU. Checked against the data by `check_declared_frame`.
    is_enu: bool,
}

impl KalmanSettings {
    /// The settings a scenario file's `[closed_loop]` section describes.
    const fn from_closed_loop(config: &ClosedLoopConfig, is_enu: bool) -> Self {
        Self {
            ukf_alpha: config.ukf_alpha,
            ukf_beta: config.ukf_beta,
            ukf_kappa: config.ukf_kappa,
            estimate_baro_bias: config.estimate_baro_bias,
            is_enu,
        }
    }

    /// The settings the `cl` subcommand's flags describe.
    const fn from_args(args: &ClosedLoopSimArgs) -> Self {
        Self {
            ukf_alpha: args.ukf_alpha,
            ukf_beta: args.ukf_beta,
            ukf_kappa: args.ukf_kappa,
            estimate_baro_bias: args.estimate_baro_bias,
            is_enu: args.sim.enu,
        }
    }

    /// The UKF these settings describe, before any map biases.
    fn ukf_config(self) -> UkfConfig {
        let mut built = UkfConfig::default();
        built.ukf_alpha = Some(self.ukf_alpha);
        built.ukf_beta = Some(self.ukf_beta);
        built.ukf_kappa = Some(self.ukf_kappa);
        built.estimate_baro_bias = self.estimate_baro_bias;
        built.is_enu = self.is_enu;
        built
    }

    /// The EKF these settings describe, before any map biases.
    fn ekf_config(self) -> EkfConfig {
        let mut built = EkfConfig::default();
        built.estimate_baro_bias = self.estimate_baro_bias;
        built.is_enu = self.is_enu;
        built
    }

    /// The ESKF these settings describe. It has no geophysical arm.
    fn eskf_config(self) -> EskfConfig {
        let mut built = EskfConfig::default();
        built.estimate_baro_bias = self.estimate_baro_bias;
        built.is_enu = self.is_enu;
        built
    }
}

/// Execute a single closed-loop simulation run
///
/// This is a helper function that extracts the common logic for running closed-loop simulations
/// with either UKF or EKF filters. It handles event stream creation, filter initialization,
/// simulation execution, and results writing.
#[allow(clippy::too_many_arguments)]
fn run_single_closed_loop_simulation(
    filter_type: FilterType,
    records: &[TestDataRecord],
    aiding: &strapdown::messages::AidingConfig,
    output_file: &Path,
    limits: RunLimits,
    kalman: KalmanSettings,
    innovation_gate: Option<InnovationGate>,
    gate_recovery: GateRecovery,
) -> Result<(), Box<dyn Error>> {
    // Same full-window guard as the other entry points: the `initialize_*` helpers below see
    // only one record, which is not enough evidence in either direction (#296).
    check_declared_frame(records, kalman.is_enu)?;

    let ukf_config = kalman.ukf_config();
    let ekf_config = kalman.ekf_config();
    let eskf_config = kalman.eskf_config();

    // The barometer model has to be told which state holds its bias, and the answer is the
    // filter's own. Deriving it here rather than asking for `baro_bias_index` to be set
    // beside `estimate_baro_bias` keeps the two from disagreeing -- a wrong index reads a
    // gyro bias as a barometric one (#372).
    let aiding = {
        let mut built = aiding.clone();
        built.baro_bias_index = match filter_type {
            FilterType::Ukf => ukf_config.baro_bias_index(),
            FilterType::Ekf => ekf_config.baro_bias_index(),
            FilterType::Eskf => eskf_config.baro_bias_index(),
        };
        built
    };

    // Build event stream from records and the aiding config
    let event_stream = build_event_stream(records, &aiding, kalman.is_enu)?;
    info!(
        "Initialized event stream with {} events",
        event_stream.events.len()
    );

    // Initialize and run filter based on type
    let results = match filter_type {
        FilterType::Ukf => {
            let mut ukf = initialize_ukf(&records[0].clone(), ukf_config)?;
            info!("Initialized UKF");
            ukf.set_innovation_gate(innovation_gate);
            ukf.set_gate_recovery(gate_recovery);
            run_closed_loop(
                &mut ukf,
                event_stream,
                Some(limits.health),
                Some(limits.execution),
            )
        }
        FilterType::Ekf => {
            let mut ekf = initialize_ekf(&records[0].clone(), ekf_config)?;
            info!("Initialized EKF");
            ekf.set_innovation_gate(innovation_gate);
            ekf.set_gate_recovery(gate_recovery);
            run_closed_loop(
                &mut ekf,
                event_stream,
                Some(limits.health),
                Some(limits.execution),
            )
        }
        FilterType::Eskf => {
            let mut eskf = initialize_eskf(&records[0].clone(), eskf_config)?;
            info!("Initialized ESKF");
            eskf.set_innovation_gate(innovation_gate);
            eskf.set_gate_recovery(gate_recovery);
            run_closed_loop(
                &mut eskf,
                event_stream,
                Some(limits.health),
                Some(limits.execution),
            )
        }
    };

    // Write results to CSV
    match results {
        Ok(ref nav_results) => {
            NavigationResult::to_csv(nav_results, output_file)?;
            info!("Results written to {}", output_file.display());
            Ok(())
        }
        Err(e) => {
            error!("Error running closed-loop simulation: {e}");
            Err(e.into())
        }
    }
}

/// Execute synthetic trajectory generation
fn run_synthetic(args: &SyntheticArgs) -> Result<(), Box<dyn Error>> {
    use strapdown::sim::SyntheticInitialState;

    if let Some(parent) = args.output.parent()
        && !parent.as_os_str().is_empty()
    {
        validate_output_path(parent)?;
    }

    let config = {
        let mut built = SyntheticConfig::default();
        built.output = args.output.to_string_lossy().into_owned();
        built.initial_state = SyntheticInitialState {
            latitude_deg: args.latitude_deg,
            longitude_deg: args.longitude_deg,
            altitude_m: args.altitude_m,
            velocity_north_mps: args.velocity_north_mps,
            velocity_east_mps: args.velocity_east_mps,
            velocity_down_mps: args.velocity_down_mps,
            roll_deg: args.roll_deg,
            pitch_deg: args.pitch_deg,
            yaw_deg: args.yaw_deg,
            angular_velocity_x_dps: args.angular_velocity_x_dps,
            angular_velocity_y_dps: args.angular_velocity_y_dps,
            angular_velocity_z_dps: args.angular_velocity_z_dps,
            is_enu: args.enu,
        };
        built.duration_s = args.duration_s;
        built.sample_rate_hz = args.sample_rate_hz;
        built.imu_quality = args.imu_grade;
        built.seed = args.seed;
        built.no_noise = args.no_noise;
        built.gnss_horizontal_noise_m = args.gnss_horizontal_noise_m;
        built.gnss_vertical_noise_m = args.gnss_vertical_noise_m;
        built.baro_noise_std_pa = args.baro_noise_std_pa;
        built.mag_noise_std_ut = args.mag_noise_std_ut;
        built.mag_hard_iron_std_ut = args.mag_hard_iron_std_ut;
        built
    };

    let mut rng = StdRng::seed_from_u64(args.seed);
    let (truth, sensors) = generate_synthetic(&config, &mut rng)?;

    let n = truth.len();
    info!(
        "Generated {} synthetic records ({:.1} s at {:.0} Hz)",
        n, args.duration_s, args.sample_rate_hz
    );

    if args.no_noise {
        NavigationResult::to_csv(&truth, &args.output)?;
        info!("Truth trajectory written to {}", args.output.display());
    } else {
        TestDataRecord::to_csv(&sensors, &args.output)?;
        info!("Sensor records written to {}", args.output.display());
    }

    Ok(())
}

/// Execute dead-reckoning simulation
fn run_dead_reckoning(args: &SimArgs) -> Result<(), Box<dyn Error>> {
    validate_input_path(&args.input)?;
    validate_output_path(&args.output)?;

    info!("Running in dead reckoning mode");

    // Get all CSV files to process
    let csv_files = get_csv_files(&args.input)?;
    let is_multiple = csv_files.len() > 1;

    if is_multiple {
        info!("Processing {} CSV files from directory", csv_files.len());
    }

    // Process each CSV file
    let mut failures = 0usize;
    for input_file in &csv_files {
        info!("Processing file: {}", input_file.display());

        // One unusable file must not abandon the rest of a batch. Before #311 this loop
        // could not fail here at all -- `from_csv` returned `Ok(vec![])` and the run wrote an
        // empty output file -- so aborting would trade a silent wrong answer for a loud
        // incomplete one. `run_from_config` already counts per-file failures and continues;
        // this matches it.
        let records = match load_records(input_file) {
            Ok(records) => records,
            Err(e) if is_multiple => {
                error!("Skipping {}: {e}", input_file.display());
                failures += 1;
                continue;
            }
            Err(e) => return Err(e),
        };

        // Run dead reckoning simulation
        info!(
            "Running dead reckoning simulation on {} records",
            records.len()
        );
        let results = dead_reckoning(&records, args.enu)?;
        info!("Generated {} navigation results", results.len());

        // Write results to CSV
        let output_file = resolve_output_path(&args.output, input_file, &csv_files)?;
        NavigationResult::to_csv(&results, &output_file)?;
        info!("Results written to {}", output_file.display());
    }

    if failures > 0 {
        error!("{failures} file(s) skipped because they held no usable records");
    }

    Ok(())
}

/// Execute open-loop simulation
fn run_open_loop(args: &SimArgs) -> Result<(), Box<dyn Error>> {
    validate_input_path(&args.input)?;
    validate_output_path(&args.output)?;

    let csv_files = get_csv_files(&args.input)?;
    let is_multiple = csv_files.len() > 1;

    if is_multiple {
        info!("Processing {} CSV files from directory", csv_files.len());
    }

    for input_file in &csv_files {
        info!("Processing file: {}", input_file.display());

        // TODO: Implement open-loop processing here
        // let records = TestDataRecord::from_csv(input_file)?;
        // let output_file = generate_output_path(&args.output, input_file, is_multiple);
        // ... process and write results ...
    }

    info!("Open-loop mode is not yet fully implemented");
    println!("Open-loop mode is not yet fully implemented");

    Ok(())
}

/// Build the innovation gate and its recovery policy from the closed-loop CLI arguments.
///
/// Shared by the plain and geophysical closed-loop paths so that `--gate-confidence` and its
/// two recovery flags mean the same thing in both. They did not: the geophysical path built
/// no gate at all, so a `--geo` run silently ignored every gating flag it was given.
///
/// Both are built once, before any file is processed: an out-of-range confidence or inflation
/// factor should be reported up front rather than after the first output file has been
/// written. The recovery policy is validated even when no gate is installed, because an
/// impossible factor is a mistake worth naming whether or not this run gates.
///
/// # Errors
/// [`StrapdownError::OutOfRange`](strapdown::StrapdownError::OutOfRange) or
/// [`StrapdownError::InvalidConfiguration`](strapdown::StrapdownError::InvalidConfiguration)
/// from the two constructors.
fn gating_from_args(
    args: &ClosedLoopSimArgs,
) -> Result<(Option<InnovationGate>, GateRecovery), Box<dyn Error>> {
    let innovation_gate = args
        .gate_confidence
        .map(InnovationGate::chi_squared)
        .transpose()?;
    let gate_recovery = GateRecovery::new(
        args.gate_inflation,
        (args.gate_force_after > 0).then_some(args.gate_force_after),
    )?;
    if let Some(gate) = innovation_gate {
        info!("Innovation gating enabled: {gate:?}, recovery {gate_recovery:?}");
    }
    Ok((innovation_gate, gate_recovery))
}

/// Execute closed-loop simulation
fn run_closed_loop_cli(args: &ClosedLoopSimArgs) -> Result<(), Box<dyn Error>> {
    // Check if geophysical navigation is enabled
    #[cfg(feature = "geonav")]
    if args.geo.geo {
        return run_geo_closed_loop_cli(args);
    }

    validate_input_path(&args.sim.input)?;
    validate_output_path(&args.sim.output)?;

    let filter_name = match args.filter {
        FilterType::Ukf => "Unscented Kalman Filter (UKF)",
        FilterType::Ekf => "Extended Kalman Filter (EKF)",
        FilterType::Eskf => "Error-State Kalman Filter (ESKF)",
    };
    info!("Running in closed-loop mode with {filter_name}");

    // Built before the per-file loop on purpose: an out-of-range confidence should
    // be reported once, up front, not after the first file has already been written.
    let (innovation_gate, gate_recovery) = gating_from_args(args)?;

    // Get all CSV files to process
    let csv_files = get_csv_files(&args.sim.input)?;
    let is_multiple = csv_files.len() > 1;
    let limits = RunLimits::from_args(&args.sim);

    if is_multiple {
        info!("Processing {} CSV files from directory", csv_files.len());
        //println!("Processing {} CSV files from directory", csv_files.len());
    }

    // Process each CSV file
    let mut failures = 0usize;
    for input_file in &csv_files {
        info!("Processing file: {}", input_file.display());

        // One unusable file must not abandon the rest of a batch. Before #311 this loop
        // could not fail here at all -- `from_csv` returned `Ok(vec![])` and the run wrote an
        // empty output file -- so aborting would trade a silent wrong answer for a loud
        // incomplete one. `run_from_config` already counts per-file failures and continues;
        // this matches it.
        let records = match load_records(input_file) {
            Ok(records) => records,
            Err(e) if is_multiple => {
                error!("Skipping {}: {e}", input_file.display());
                failures += 1;
                continue;
            }
            Err(e) => return Err(e),
        };

        // Build the aiding config from CLI args
        let aiding = {
            // The barometer and magnetometer schedules have no CLI flag; they take their
            // 1 Hz default, overridable from a config file through serde.
            let mut built = strapdown::messages::AidingConfig::default();
            built.scheduler = build_scheduler(&args.scheduler);
            built.fault = build_fault(&args.fault);
            built.seed = args.seed;
            built
        };

        info!("Using aiding config: {aiding:?}");
        let output_file = resolve_output_path(&args.sim.output, input_file, &csv_files)?;

        // Run simulation using the common helper function
        match run_single_closed_loop_simulation(
            args.filter,
            &records,
            &aiding,
            &output_file,
            limits.clone(),
            KalmanSettings::from_args(args),
            innovation_gate,
            gate_recovery,
        ) {
            Ok(()) => {
                // Success - result logging is handled by the helper function
            }
            Err(e) => {
                error!(
                    "Error running closed-loop simulation on {}: {}",
                    input_file.display(),
                    e
                );
                if !is_multiple {
                    return Err(e);
                }
                // For multiple files, continue processing remaining files
                error!(
                    "Error processing {}: {}. Continuing with remaining files...",
                    input_file.display(),
                    e
                );
            }
        }
    }

    if failures > 0 {
        error!("{failures} file(s) skipped because they held no usable records");
    }

    Ok(())
}

// ============================================================================
// Geophysical Navigation Functions (feature-gated)
// ============================================================================

/// Convert `GeoResolution` to `GravityResolution`
#[cfg(feature = "geonav")]
const fn convert_resolution_gravity(resolution: GeoResolution) -> GravityResolution {
    match resolution {
        GeoResolution::OneDegree => GravityResolution::OneDegree,
        GeoResolution::ThirtyMinutes => GravityResolution::ThirtyMinutes,
        GeoResolution::TwentyMinutes => GravityResolution::TwentyMinutes,
        GeoResolution::FifteenMinutes => GravityResolution::FifteenMinutes,
        GeoResolution::TenMinutes => GravityResolution::TenMinutes,
        GeoResolution::SixMinutes => GravityResolution::SixMinutes,
        GeoResolution::FiveMinutes => GravityResolution::FiveMinutes,
        GeoResolution::FourMinutes => GravityResolution::FourMinutes,
        GeoResolution::ThreeMinutes => GravityResolution::ThreeMinutes,
        GeoResolution::TwoMinutes => GravityResolution::TwoMinutes,
        _ => GravityResolution::OneMinute,
    }
}

/// Convert `GeoResolution` to `MagneticResolution`
#[cfg(feature = "geonav")]
const fn convert_resolution_magnetic(resolution: GeoResolution) -> MagneticResolution {
    match resolution {
        GeoResolution::OneDegree => MagneticResolution::OneDegree,
        GeoResolution::ThirtyMinutes => MagneticResolution::ThirtyMinutes,
        GeoResolution::TwentyMinutes => MagneticResolution::TwentyMinutes,
        GeoResolution::FifteenMinutes => MagneticResolution::FifteenMinutes,
        GeoResolution::TenMinutes => MagneticResolution::TenMinutes,
        GeoResolution::SixMinutes => MagneticResolution::SixMinutes,
        GeoResolution::FiveMinutes => MagneticResolution::FiveMinutes,
        GeoResolution::FourMinutes => MagneticResolution::FourMinutes,
        GeoResolution::ThreeMinutes => MagneticResolution::ThreeMinutes,
        _ => MagneticResolution::TwoMinutes,
    }
}

/// Auto-detect gravity map file based on input directory
#[cfg(feature = "geonav")]
fn find_gravity_map(input_path: &Path) -> Result<PathBuf, Box<dyn Error>> {
    let input_dir = input_path
        .parent()
        .ok_or("Cannot determine input directory")?;

    let input_stem = input_path
        .file_stem()
        .ok_or("Cannot determine input file stem")?
        .to_string_lossy();

    let map_file = input_dir.join(format!("{input_stem}_gravity.nc"));

    if map_file.exists() {
        Ok(map_file)
    } else {
        Err(format!("Gravity map file not found: {}", map_file.display()).into())
    }
}

/// Auto-detect magnetic map file based on input directory
#[cfg(feature = "geonav")]
fn find_magnetic_map(input_path: &Path) -> Result<PathBuf, Box<dyn Error>> {
    let input_dir = input_path
        .parent()
        .ok_or("Cannot determine input directory")?;

    let input_stem = input_path
        .file_stem()
        .ok_or("Cannot determine input file stem")?
        .to_string_lossy();

    let map_file = input_dir.join(format!("{input_stem}_magnetic.nc"));

    if map_file.exists() {
        Ok(map_file)
    } else {
        Err(format!("Magnetic map file not found: {}", map_file.display()).into())
    }
}

/// Everything the geophysical closed-loop runner needs, independent of where it came from.
///
/// The runner used to read [`ClosedLoopSimArgs`] directly, which is what made `--geo`
/// command-line-only: a configuration file has no `ClosedLoopSimArgs` to offer it, so
/// `process_file` refused a `[geophysical]` section outright rather than wire one up. The
/// RBPF arm of the same function had no such restriction and read both its maps and its
/// GNSS degradation from one file, so the two filter families were configured in
/// incompatible ways and the justfile carried a long CLI invocation for one and a
/// `--config` for the other.
///
/// This type is the seam. [`geo_settings_from_args`] builds it from the command line and
/// [`geo_settings_from_config`] from a scenario file; `run_geo_closed_loop_file` takes it
/// and cannot tell which. `builder_equivalence` asserts the two agree.
#[cfg(feature = "geonav")]
#[derive(Clone, Debug)]
struct GeoClosedLoopSettings {
    /// Which filter runs. `Eskf` is rejected before a run starts; see
    /// [`GeoClosedLoopSettings::validate`].
    filter: FilterType,
    /// The filter's own settings, exactly as the unaided closed-loop paths build them --
    /// barometric bias included. The runner appends the map biases and changes nothing else.
    kalman: KalmanSettings,
    /// Gravity map resolution. `None` disables the channel.
    gravity_resolution: Option<GeoResolution>,
    /// Explicit gravity map path, or `None` to look beside the input.
    gravity_map_file: Option<PathBuf>,
    /// Gravity measurement noise, mGal.
    gravity_noise_std: f64,
    /// Seed of the gravity map-bias state, mGal.
    gravity_bias: Option<f64>,
    /// Prior standard deviation of that bias, mGal. Defaults to `gravity_noise_std`.
    gravity_prior_std: Option<f64>,
    /// Random-walk rate of that bias, mGal per sqrt(s).
    gravity_drift_rate: Option<f64>,
    /// Magnetic map resolution. `None` disables the channel.
    magnetic_resolution: Option<GeoResolution>,
    /// Explicit magnetic map path, or `None` to look beside the input.
    magnetic_map_file: Option<PathBuf>,
    /// Magnetic measurement noise, nT.
    magnetic_noise_std: f64,
    /// Seed of the magnetic map-bias state, nT.
    magnetic_bias: Option<f64>,
    /// Prior standard deviation of that bias, nT. Defaults to `magnetic_noise_std`.
    magnetic_prior_std: Option<f64>,
    /// Random-walk rate of that bias, nT per sqrt(s).
    magnetic_drift_rate: Option<f64>,
    /// Seconds between geophysical measurements. A period, not a frequency.
    geo_interval_s: Option<f64>,
    /// GNSS scheduling and fault injection.
    aiding: strapdown::messages::AidingConfig,
    /// Run-level guards.
    health: HealthLimits,
    execution: ExecutionLimits,
    /// Per-measurement gating. `None` accepts every measurement.
    innovation_gate: Option<InnovationGate>,
    gate_recovery: GateRecovery,
    /// Whether to write a performance plot beside each result.
    generate_plot: bool,
}

#[cfg(feature = "geonav")]
impl GeoClosedLoopSettings {
    /// Reject a configuration that cannot run, before any file is read.
    ///
    /// # Errors
    ///
    /// Returns an error when no map is configured, or when the filter is the ESKF, which
    /// has no geophysical implementation. The latter matters more than it looks:
    /// [`FilterType`]'s `#[default]` is `Eskf`, so a configuration file that omits `filter`
    /// lands here rather than on a filter that works.
    fn validate(&self) -> Result<(), Box<dyn Error>> {
        if self.gravity_resolution.is_none() && self.magnetic_resolution.is_none() {
            return Err("geophysical navigation needs at least one map: set \
                 `--gravity-resolution`/`--magnetic-resolution`, or `gravity_resolution`/\
                 `magnetic_resolution` in the `[geophysical]` section of a config file"
                .into());
        }
        if matches!(self.filter, FilterType::Eskf) {
            return Err(
                "ESKF is not yet implemented for geophysical navigation. Choose \
                 `--filter ukf` or `--filter ekf` (or `filter = \"ukf\"` / `filter = \"ekf\"` \
                 in a config file's `[closed_loop]` section). Note that `eskf` is the \
                 default, so a config file that omits `filter` reaches this too."
                    .into(),
            );
        }
        Ok(())
    }

    /// The gravity channel's map-bias prior, as [`geo_bias_setup`] takes it.
    const fn gravity_bias_prior(&self) -> MapBiasPrior {
        MapBiasPrior {
            noise_std: self.gravity_noise_std,
            seed: self.gravity_bias,
            init_std: self.gravity_prior_std,
            drift_rate: self.gravity_drift_rate,
        }
    }

    /// The magnetic channel's map-bias prior, as [`geo_bias_setup`] takes it.
    const fn magnetic_bias_prior(&self) -> MapBiasPrior {
        MapBiasPrior {
            noise_std: self.magnetic_noise_std,
            seed: self.magnetic_bias,
            init_std: self.magnetic_prior_std,
            drift_rate: self.magnetic_drift_rate,
        }
    }
}

/// Build the runner's settings from the command line.
#[cfg(feature = "geonav")]
fn geo_settings_from_args(
    args: &ClosedLoopSimArgs,
) -> Result<GeoClosedLoopSettings, Box<dyn Error>> {
    let (innovation_gate, gate_recovery) = gating_from_args(args)?;
    let limits = RunLimits::from_args(&args.sim);

    // The barometer and magnetometer schedules have no CLI flag; they take their 1 Hz
    // default. A config file can set them, which is one of the things the config path now
    // gets that this one does not.
    let mut aiding = strapdown::messages::AidingConfig::default();
    aiding.scheduler = build_scheduler(&args.scheduler);
    aiding.fault = build_fault(&args.fault);
    aiding.seed = args.seed;

    Ok(GeoClosedLoopSettings {
        filter: args.filter,
        kalman: KalmanSettings::from_args(args),
        gravity_resolution: args.geo.gravity_resolution,
        gravity_map_file: args.geo.gravity_map_file.clone(),
        gravity_noise_std: args.geo.gravity_noise_std,
        gravity_bias: args.geo.gravity_bias,
        gravity_prior_std: args.geo_bias.gravity_prior_std,
        gravity_drift_rate: args.geo_bias.gravity_drift_rate,
        magnetic_resolution: args.geo.magnetic_resolution,
        magnetic_map_file: args.geo.magnetic_map_file.clone(),
        magnetic_noise_std: args.geo.magnetic_noise_std,
        magnetic_bias: args.geo.magnetic_bias,
        magnetic_prior_std: args.geo_bias.magnetic_prior_std,
        magnetic_drift_rate: args.geo_bias.magnetic_drift_rate,
        geo_interval_s: args.geo.geo_interval_s,
        aiding,
        health: limits.health,
        execution: limits.execution,
        innovation_gate,
        gate_recovery,
        generate_plot: false,
    })
}

/// Build the same settings from a scenario file.
///
/// Infallible, unlike [`geo_settings_from_args`]: a scenario file's gate is already a parsed
/// [`InnovationGate`], while the command line takes a confidence that has to be converted.
/// What a file can get wrong is caught by [`GeoClosedLoopSettings::validate`] instead.
#[cfg(feature = "geonav")]
fn geo_settings_from_config(
    config: &SimulationConfig,
    geo: &GeophysicalConfig,
) -> GeoClosedLoopSettings {
    let filter_config = config.closed_loop.clone().unwrap_or_default();

    // Cleared here and derived by the runner from the filter it builds, as the unaided arm of
    // `process_file` derives it: the index depends on how many map biases precede the
    // barometric one, which only the runner knows once the maps are loaded.
    let mut aiding = config.aiding.clone();
    aiding.baro_bias_index = None;

    GeoClosedLoopSettings {
        filter: filter_config.filter,
        kalman: KalmanSettings::from_closed_loop(&filter_config, config.is_enu),
        gravity_resolution: geo.gravity_resolution,
        gravity_map_file: geo.gravity_map_file.as_ref().map(PathBuf::from),
        gravity_noise_std: geo.gravity_noise_std.unwrap_or(DEFAULT_GRAVITY_NOISE_MGAL),
        gravity_bias: geo.gravity_bias,
        gravity_prior_std: geo.gravity_bias_init_std,
        gravity_drift_rate: geo.gravity_bias_process_noise_std,
        magnetic_resolution: geo.magnetic_resolution,
        magnetic_map_file: geo.magnetic_map_file.as_ref().map(PathBuf::from),
        magnetic_noise_std: geo.magnetic_noise_std.unwrap_or(DEFAULT_MAGNETIC_NOISE_NT),
        magnetic_bias: geo.magnetic_bias,
        magnetic_prior_std: geo.magnetic_bias_init_std,
        magnetic_drift_rate: geo.magnetic_bias_process_noise_std,
        geo_interval_s: geo.geo_interval_s,
        aiding,
        health: config.health_limits.clone(),
        execution: config.execution_limits.clone(),
        innovation_gate: filter_config.innovation_gate,
        gate_recovery: filter_config.gate_recovery,
        generate_plot: config.generate_plot,
    }
}

/// One map channel's bias prior, as configured and before its defaults are resolved.
///
/// Gathered from wherever the run was configured -- `[geophysical]` in a scenario file, `--geo`
/// and the `--*-bias-*` flags on the command line -- so that [`geo_bias_setup`] resolves the
/// defaults in one place for the Kalman arms and the particle filter alike. The particle filter
/// used to skip all of this and take one prior for every channel from `[particle_filter]`.
#[cfg(feature = "geonav")]
#[derive(Clone, Copy, Debug)]
struct MapBiasPrior {
    /// The channel's measurement noise standard deviation, which the prior defaults to.
    noise_std: f64,
    /// Seed of the bias state. `None` seeds zero.
    seed: Option<f64>,
    /// Prior standard deviation. `None` takes `noise_std`.
    init_std: Option<f64>,
    /// Random-walk rate per root-second. `None` spreads the prior over
    /// [`GEO_BIAS_DRIFT_TIME_CONSTANT_S`].
    drift_rate: Option<f64>,
}

#[cfg(feature = "geonav")]
impl MapBiasPrior {
    /// The gravity channel's prior as a `[geophysical]` section states it.
    fn gravity_from_config(geo: &GeophysicalConfig) -> Self {
        Self {
            noise_std: geo.gravity_noise_std.unwrap_or(DEFAULT_GRAVITY_NOISE_MGAL),
            seed: geo.gravity_bias,
            init_std: geo.gravity_bias_init_std,
            drift_rate: geo.gravity_bias_process_noise_std,
        }
    }

    /// The magnetic channel's prior as a `[geophysical]` section states it.
    fn magnetic_from_config(geo: &GeophysicalConfig) -> Self {
        Self {
            noise_std: geo.magnetic_noise_std.unwrap_or(DEFAULT_MAGNETIC_NOISE_NT),
            seed: geo.magnetic_bias,
            init_std: geo.magnetic_bias_init_std,
            drift_rate: geo.magnetic_bias_process_noise_std,
        }
    }

    /// The gravity channel's prior as the command line states it.
    const fn gravity_from_args(geo: &GeophysicalArgs, bias: &GeophysicalBiasArgs) -> Self {
        Self {
            noise_std: geo.gravity_noise_std,
            seed: geo.gravity_bias,
            init_std: bias.gravity_prior_std,
            drift_rate: bias.gravity_drift_rate,
        }
    }

    /// The magnetic channel's prior as the command line states it.
    const fn magnetic_from_args(geo: &GeophysicalArgs, bias: &GeophysicalBiasArgs) -> Self {
        Self {
            noise_std: geo.magnetic_noise_std,
            seed: geo.magnetic_bias,
            init_std: bias.magnetic_prior_std,
            drift_rate: bias.magnetic_drift_rate,
        }
    }
}

/// The seed, prior and random walk of each geophysical map bias, resolved.
///
/// One entry per active channel, **gravity first, then magnetic**, which is the order
/// [`GeoBiasLayout::appended`] assigns the appended slots -- a magnetic-only run takes the
/// first one. Every vector is built together for that reason: they index the same states,
/// and the UKF, the EKF and the particle filter all read them, so they cannot be allowed to
/// drift apart the way the arms' hardcoded constants once did.
///
/// The prior and the random walk are held twice, as standard deviations and squared, because
/// the two filter families take different forms: the Kalman arms a variance and a density, the
/// particle filter the standard deviations. Both come from the one loop in
/// [`geo_bias_setup`].
#[cfg(feature = "geonav")]
#[derive(Debug)]
struct GeoBiasSetup {
    /// Initial value of each bias, in its channel's own units.
    seeds: Vec<f64>,
    /// Initial standard deviation of each bias.
    init_stds: Vec<f64>,
    /// Random-walk rate of each bias, per root-second.
    drift_rates: Vec<f64>,
    /// Initial variance of each bias: the init standard deviation squared.
    variances: Vec<f64>,
    /// Process-noise **density** of each bias -- a variance per second, which every filter
    /// here turns into `Q_k = q * dt` (#374), so it is the random-walk rate squared.
    densities: Vec<f64>,
}

#[cfg(feature = "geonav")]
impl GeoBiasSetup {
    /// Give a particle filter these map biases as its extra states.
    ///
    /// Leaves `extra_state_dim` alone. The bias layout sets that, and
    /// `RaoBlackwellizedParticleFilter::new` refuses per-state vectors of any other length, so
    /// the layout and these priors cannot silently disagree about how many biases there are.
    fn apply_to_rbpf(&self, config: &mut RbpfConfig) {
        config.extra_state_initial.clone_from(&self.seeds);
        config.extra_state_init_std.clone_from(&self.init_stds);
        config
            .extra_state_process_noise_std
            .clone_from(&self.drift_rates);
    }

    /// Give a UKF these map biases as its extra states, and change nothing else about it.
    ///
    /// They land after the fifteen navigation and IMU-bias states and before the barometric
    /// bias, which `initialize_ukf` puts last; [`kalman_geo_bias_layout`] describes the same
    /// placement to the measurement models. The process noise is the config's own diagonal --
    /// the crate default when it has none, as `initialize_ukf` would take -- with the biases'
    /// densities appended, so the fifteen shared entries are exactly the unaided filter's.
    fn apply_to_ukf(&self, config: &mut UkfConfig) {
        config.other_states = Some(self.seeds.clone());
        config.other_states_covariance = Some(self.variances.clone());
        config.process_noise_diagonal =
            Some(self.extend_process_noise(config.process_noise_diagonal.take()));
    }

    /// Give an EKF these map biases as its extra states; see [`Self::apply_to_ukf`], which
    /// `EkfConfig::other_states` mirrors placement for placement.
    fn apply_to_ekf(&self, config: &mut EkfConfig) {
        config.other_states = Some(self.seeds.clone());
        config.other_states_covariance = Some(self.variances.clone());
        config.process_noise_diagonal =
            Some(self.extend_process_noise(config.process_noise_diagonal.take()));
    }

    /// A Kalman filter's base process-noise diagonal with these biases' densities appended.
    ///
    /// `base` covers the fifteen navigation and IMU-bias states and nothing after them: the
    /// constructors append the barometric bias's own entry behind the map biases'.
    fn extend_process_noise(&self, base: Option<Vec<f64>>) -> Vec<f64> {
        let mut diagonal = base.unwrap_or_else(|| DEFAULT_PROCESS_NOISE_DENSITY.to_vec());
        diagonal.extend(self.densities.iter().copied());
        diagonal
    }
}

/// Refuse a filter whose state is not the width its map and barometric biases were laid out
/// for.
///
/// The layout and the filter come from the same `KalmanSettings`, so this holds by
/// construction. It is checked anyway, once, before a run starts: a disagreement would
/// otherwise surface as a dimension error on the first geophysical fix, or as a solution whose
/// bias columns read the wrong states.
///
/// # Errors
/// When `width` is not `expected`.
#[cfg(feature = "geonav")]
fn ensure_layout_width(filter: &str, width: usize, expected: usize) -> Result<(), Box<dyn Error>> {
    if width == expected {
        Ok(())
    } else {
        Err(format!(
            "the {filter} carries {width} states, but its map and barometric biases were laid \
             out for {expected}"
        )
        .into())
    }
}

/// Where a UKF or EKF built by `initialize_ukf` / `initialize_ekf` carries its map biases.
///
/// After the fifteen navigation and IMU-bias states, gravity first -- and, when the filter
/// also estimates a barometric bias, *before* it, since both constructors put that state last.
/// The layout's width is the filter's whole state, barometric bias included, because the
/// measurement models check it: a layout ending at the map biases would be refused on the
/// first fix by a filter one state wider.
///
/// # Errors
/// Propagated from [`GeoBiasLayout`], which validates the placement.
#[cfg(feature = "geonav")]
fn kalman_geo_bias_layout(
    gravity: bool,
    magnetic: bool,
    estimate_baro_bias: bool,
) -> Result<Option<GeoBiasLayout>, strapdown::StrapdownError> {
    GeoBiasLayout::appended(NAVIGATION_AND_IMU_BIAS_STATE_DIM, gravity, magnetic)?
        .map(|layout| {
            GeoBiasLayout::new(
                layout.state_dim() + usize::from(estimate_baro_bias),
                layout.gravity_bias().map(|bias| bias.index),
                layout.magnetic_bias().map(|bias| bias.index),
            )
        })
        .transpose()
}

/// Resolve each active channel's map-bias prior.
///
/// Each channel defaults its prior to that channel's own measurement-noise standard
/// deviation, and its random-walk rate to that prior spread over
/// [`GEO_BIAS_DRIFT_TIME_CONSTANT_S`]. Both are standard deviations and are squared here for
/// the Kalman arms, which is the whole of the units fix: the previous code passed
/// `gravity_noise_std`/`magnetic_noise_std` **unsquared** into a covariance diagonal, so a
/// 150 nT measurement noise became a 150 nT^2 prior -- a 12 nT sigma -- and pinned the bias
/// next to its seed.
///
/// `None` for a channel the run does not aid.
#[cfg(feature = "geonav")]
fn geo_bias_setup(gravity: Option<MapBiasPrior>, magnetic: Option<MapBiasPrior>) -> GeoBiasSetup {
    let mut setup = GeoBiasSetup {
        seeds: Vec::new(),
        init_stds: Vec::new(),
        drift_rates: Vec::new(),
        variances: Vec::new(),
        densities: Vec::new(),
    };

    for prior in [gravity, magnetic].into_iter().flatten() {
        let init_std = prior.init_std.unwrap_or(prior.noise_std);
        let rate = prior
            .drift_rate
            .unwrap_or_else(|| init_std / GEO_BIAS_DRIFT_TIME_CONSTANT_S.sqrt());
        setup.seeds.push(prior.seed.unwrap_or(0.0));
        setup.init_stds.push(init_std);
        setup.drift_rates.push(rate);
        setup.variances.push(init_std.powi(2));
        setup.densities.push(rate.powi(2));
    }

    setup
}

/// Execute geophysical closed-loop simulation from the command line.
///
/// Resolves `--geo`'s flags into [`GeoClosedLoopSettings`] and hands each input file to
/// [`run_geo_closed_loop_file`], which is the same entry point a `--config` run uses.
#[cfg(feature = "geonav")]
fn run_geo_closed_loop_cli(args: &ClosedLoopSimArgs) -> Result<(), Box<dyn Error>> {
    validate_input_path(&args.sim.input)?;
    validate_output_path(&args.sim.output)?;

    let settings = geo_settings_from_args(args)?;
    settings.validate()?;

    // Get all CSV files to process
    let csv_files = get_csv_files(&args.sim.input)?;
    let is_multiple = csv_files.len() > 1;

    if is_multiple {
        info!("Processing {} CSV files from directory", csv_files.len());
    }

    // Process each CSV file
    let mut failures = 0usize;
    for input_file in &csv_files {
        info!("Processing file: {}", input_file.display());

        // One unusable file must not abandon the rest of a batch. Before #311 this loop
        // could not fail here at all -- `from_csv` returned `Ok(vec![])` and the run wrote an
        // empty output file -- so aborting would trade a silent wrong answer for a loud
        // incomplete one. `run_from_config` already counts per-file failures and continues;
        // this matches it.
        let records = match load_records(input_file) {
            Ok(records) => records,
            Err(e) if is_multiple => {
                error!("Skipping {}: {e}", input_file.display());
                failures += 1;
                continue;
            }
            Err(e) => return Err(e),
        };

        let output_file = resolve_output_path(&args.sim.output, input_file, &csv_files)?;
        if let Err(e) = run_geo_closed_loop_file(&settings, &records, input_file, &output_file) {
            error!(
                "Error running geophysical navigation on {}: {}",
                input_file.display(),
                e
            );
            if !is_multiple {
                return Err(e);
            }
            failures += 1;
        }
    }

    if failures > 0 {
        error!("{failures} file(s) skipped or failed");
    }

    Ok(())
}

/// Run one file's geophysical closed loop, from settings that say nothing about where they
/// came from.
///
/// This is the body the command line and a `--config` run share. Keeping it filter-agnostic
/// and source-agnostic is what lets `builder_equivalence` assert the two paths agree: there
/// is one implementation, and the only thing that varies is how its settings were built.
///
/// # Errors
///
/// Returns an error when a map is missing or unreadable, when the declared frame does not
/// match the data, or when the filter fails a health or execution limit.
#[cfg(feature = "geonav")]
fn run_geo_closed_loop_file(
    settings: &GeoClosedLoopSettings,
    records: &[TestDataRecord],
    input_file: &Path,
    output_file: &Path,
) -> Result<(), Box<dyn Error>> {
    // The full-window frame guard, for both filters. Only the EKF arm ran it, so
    // `cl --geo --filter ukf` never checked its declared frame against the data (#296); a
    // `--config` run was covered by `process_file`'s own call.
    check_declared_frame(records, settings.kalman.is_enu)?;

    // Moved into whichever filter arm runs rather than cloned per arm: only one of them
    // executes, so each may take ownership.
    let health = settings.health.clone();
    let execution = settings.execution.clone();
    let innovation_gate = settings.innovation_gate;
    let gate_recovery = settings.gate_recovery;

    let filter_name = match settings.filter {
        FilterType::Ukf => "Unscented Kalman Filter (UKF)",
        FilterType::Ekf => "Extended Kalman Filter (EKF)",
        FilterType::Eskf => "Error-State Kalman Filter (ESKF)",
    };
    info!("Running geophysical navigation in closed-loop mode with {filter_name}");

    {
        // Load gravity map if configured
        let gravity_map = if let Some(res) = settings.gravity_resolution {
            let map_path = match &settings.gravity_map_file {
                Some(path) => path.clone(),
                None => find_gravity_map(input_file)?,
            };

            info!("Loading gravity map from: {}", map_path.display());
            let measurement_type =
                GeophysicalMeasurementType::Gravity(convert_resolution_gravity(res));
            let map = Rc::new(GeoMap::load_geomap(&map_path, measurement_type)?);
            info!(
                "Loaded gravity map with {} x {} grid points",
                map.get_lats().len(),
                map.get_lons().len()
            );
            Some(map)
        } else {
            None
        };

        // Load magnetic map if configured
        let magnetic_map = if let Some(res) = settings.magnetic_resolution {
            let map_path = match &settings.magnetic_map_file {
                Some(path) => path.clone(),
                None => find_magnetic_map(input_file)?,
            };

            info!("Loading magnetic map from: {}", map_path.display());
            let measurement_type =
                GeophysicalMeasurementType::Magnetic(convert_resolution_magnetic(res));
            let map = Rc::new(GeoMap::load_geomap(&map_path, measurement_type)?);
            info!(
                "Loaded magnetic map with {} x {} grid points",
                map.get_lats().len(),
                map.get_lons().len()
            );
            Some(map)
        } else {
            None
        };

        // The filter is the one every unaided closed-loop path builds from the same
        // `KalmanSettings` -- barometric bias included -- with the map biases appended and
        // nothing else changed. Its state is `[9 navigation, 6 IMU bias, ..map biases,
        // barometric bias]`: both constructors put the barometer last, so the map biases sit
        // at fifteen onward and the layout the measurements check spans the barometer too.
        //
        // This arm used to build its own filters: neither estimated the barometric bias, and
        // the EKF was assembled by hand with a P0 and Q of its own. Its results therefore
        // measured a different filter from the degraded-GNSS runs they were scored against.
        let geo_bias = geo_bias_setup(
            gravity_map
                .is_some()
                .then_some(settings.gravity_bias_prior()),
            magnetic_map
                .is_some()
                .then_some(settings.magnetic_bias_prior()),
        );
        let kalman = settings.kalman;
        let mut ukf_config = kalman.ukf_config();
        geo_bias.apply_to_ukf(&mut ukf_config);
        let mut ekf_config = kalman.ekf_config();
        geo_bias.apply_to_ekf(&mut ekf_config);
        let geo_bias_layout = kalman_geo_bias_layout(
            gravity_map.is_some(),
            magnetic_map.is_some(),
            kalman.estimate_baro_bias,
        )?;
        let num_geo_states = geo_bias_layout.map_or(0, |layout| layout.bias_count());

        // Derived from the filter, as every other closed-loop path derives it (#372), and after
        // the map biases. It was cleared here while neither filter carried the state.
        let baro_bias_index = match settings.filter {
            FilterType::Ukf => ukf_config.baro_bias_index(),
            FilterType::Ekf => ekf_config.baro_bias_index(),
            FilterType::Eskf => None,
        };
        let mut aiding = settings.aiding.clone();
        aiding.baro_bias_index = baro_bias_index;

        // Build event stream with geophysical measurements
        let events = geo_build_event_stream(
            records,
            &aiding,
            kalman.is_enu,
            &GeophysicalAiding {
                gravity_noise_std: gravity_map.as_ref().map(|_| settings.gravity_noise_std),
                magnetic_noise_std: magnetic_map.as_ref().map(|_| settings.magnetic_noise_std),
                gravity_map,
                magnetic_map,
                interval_s: settings.geo_interval_s,
                bias_layout: geo_bias_layout,
            },
        )?;
        info!("Built event stream with {} events", events.events.len());

        // The same placement, restated for `NavigationResult`, which lives in `core` and so
        // cannot name `GeoBiasLayout`: the map biases where the layout put them, and the
        // barometric bias after them.
        let state_dim = NAVIGATION_AND_IMU_BIAS_STATE_DIM
            + num_geo_states
            + usize::from(baro_bias_index.is_some());
        let geo_layout = ExtraStateLayout::new(
            state_dim,
            geo_bias_layout
                .and_then(|layout| layout.gravity_bias())
                .map(|bias| bias.index),
            geo_bias_layout
                .and_then(|layout| layout.magnetic_bias())
                .map(|bias| bias.index),
        );
        let geo_layout = match baro_bias_index {
            Some(index) => geo_layout.with_baro_bias(index),
            None => geo_layout,
        };

        // Run simulation based on filter type
        let results = match settings.filter {
            FilterType::Ukf => {
                info!("Initializing UKF...");
                let mut ukf = initialize_ukf(&records[0].clone(), ukf_config)?;
                ensure_layout_width("UKF", ukf.get_estimate().len(), state_dim)?;
                info!(
                    "Initialized UKF with state dimension {} (base: 15, geo: {num_geo_states}, \
                     barometric bias: {})",
                    ukf.get_estimate().len(),
                    baro_bias_index.is_some()
                );
                ukf.set_innovation_gate(innovation_gate);
                ukf.set_gate_recovery(gate_recovery);

                info!("Running UKF geophysical navigation simulation...");
                run_closed_loop_with_geo(
                    &mut ukf,
                    events,
                    Some(health),
                    Some(execution),
                    geo_layout,
                )
            }
            FilterType::Ekf => {
                info!("Initializing EKF...");
                let mut ekf = initialize_ekf(&records[0].clone(), ekf_config)?;
                ensure_layout_width("EKF", ekf.get_estimate().len(), state_dim)?;
                info!(
                    "Initialized EKF with state dimension {} (base: 15, geo: {num_geo_states}, \
                     barometric bias: {})",
                    ekf.get_estimate().len(),
                    baro_bias_index.is_some()
                );
                ekf.set_innovation_gate(innovation_gate);
                ekf.set_gate_recovery(gate_recovery);

                info!("Running EKF geophysical navigation simulation...");
                run_closed_loop_with_geo(
                    &mut ekf,
                    events,
                    Some(health),
                    Some(execution),
                    geo_layout,
                )
            }
            FilterType::Eskf => {
                // Unreachable: `GeoClosedLoopSettings::validate` rejects this before any
                // file is read, so that a run fails on its configuration rather than after
                // loading maps. Kept as a arm rather than an `unreachable!()` because the
                // zero-panic policy applies here too.
                error!("ESKF is not yet implemented for geophysical navigation");
                return Err("ESKF is not yet implemented for geophysical navigation".into());
            }
        };

        let nav_results = results?;
        NavigationResult::to_csv(&nav_results, output_file)?;
        info!("Results written to {}", output_file.display());

        // Plotting reached the non-geophysical config path and never this one, so a
        // `generate_plot = true` in a geophysical config used to be accepted and ignored --
        // and every conf/*.toml sets it. The CLI path leaves it false, matching what `--geo`
        // did before.
        #[cfg(feature = "plotting")]
        if settings.generate_plot {
            let plot_path = output_file.with_extension("png");
            info!("Generating performance plot at {}", plot_path.display());
            match plotting::plot_performance(&nav_results, records, &plot_path) {
                Ok(()) => info!("Performance plot generated successfully"),
                // A missing plot is not a reason to discard a completed run.
                Err(e) => error!("Failed to generate performance plot: {e}"),
            }
        }
        #[cfg(not(feature = "plotting"))]
        if settings.generate_plot {
            error!(
                "Plotting requested but 'plotting' feature not enabled. Rebuild with --features plotting"
            );
        }

        Ok(())
    }
}

/// Abort after this many consecutive rejected measurements. Mirrors
/// `strapdown::sim::run_closed_loop_with_geo`'s own limit of the same value.
const RBPF_MAX_CONSECUTIVE_REJECTIONS: usize = 100;

/// Run the RBPF event loop.
///
/// Handles every measurement type carried by the event stream, geophysical
/// anomalies included -- the measurement models read the state the filter
/// passes them, so no separate geophysical loop is required.
///
/// `geo_layout` says which geophysical bias states the filter was configured to carry, and
/// must agree with the `extra_state_dim` its `RbpfConfig` was built with -- the caller derives
/// both from the same pair of loaded maps. The cloud is summarised with
/// `estimate_with_extra_states` rather than `estimate` so those biases and their variances
/// reach the solution: summarising a geophysically aided run as nine states drops the one
/// quantity the aiding exists to produce, and the rows then look complete with their
/// geophysical columns blank.
///
/// A recoverable measurement failure -- chiefly the estimate wandering off the loaded
/// geophysical map during a long GNSS outage -- is skipped rather than aborting the run, the
/// same contract `run_closed_loop_with_geo` gives the EKF/UKF/ESKF path. Skipping every
/// measurement would silently degrade to dead reckoning, so
/// [`RBPF_MAX_CONSECUTIVE_REJECTIONS`] consecutive rejections is still fatal.
fn run_rbpf_event_loop(
    rbpf: &mut RaoBlackwellizedParticleFilter,
    event_stream: EventStream,
    execution_limits: &ExecutionLimits,
    health_limits: &HealthLimits,
    geo_layout: ExtraStateLayout,
) -> Result<Vec<NavigationResult>, Box<dyn Error>> {
    let start_time = event_stream.start_time;
    let total = event_stream.events.len();
    let mut results = Vec::with_capacity(total);
    let mut monitor = HealthMonitor::new(health_limits.clone());
    let mut rejected_measurements: usize = 0;
    let mut consecutive_rejections: usize = 0;
    let sim_duration_s = event_stream.events.last().map_or(0.0, |event| match event {
        Event::Imu { elapsed_s, .. } | Event::Measurement { elapsed_s, .. } => *elapsed_s,
    });
    let mut execution_monitor = ExecutionMonitor::new(&execution_limits.clone(), sim_duration_s);

    let (mean, cov) = rbpf.estimate_with_extra_states();
    results.push(NavigationResult::from_particle_filter_with_geo(
        &start_time,
        &mean,
        &cov,
        geo_layout,
    ));
    let mut last_ts = start_time;

    for (i, event) in event_stream.events.into_iter().enumerate() {
        let elapsed_s = match &event {
            Event::Imu { elapsed_s, .. } | Event::Measurement { elapsed_s, .. } => *elapsed_s,
        };
        let ts = start_time + chrono::Duration::milliseconds((elapsed_s * 1000.0).round() as i64);

        // Emit the row for the epoch that just ended, before applying anything from this
        // one, so a row stamped `t_k` holds every event at or before `t_k` and none after.
        //
        // This loop used to push with the *current* `ts` on the first event of each epoch --
        // always the IMU step -- so the row labelled `t_k` was the prior at `t_k` with none
        // of that epoch's measurement updates, and the final epoch's updates were never
        // emitted at all. That is the opposite error to the one `sim::run_closed_loop` had
        // (#367), which is how the workspace came to ship three hand-rolled copies of this
        // loop under two mutually inconsistent conventions. `sim::dead_reckoning` was the
        // only one that was right; all of them now agree with it.
        if ts != last_ts {
            // The seed row pushed before this loop already covers `start_time`, whose epoch
            // is empty (`build_event_stream` walks `windows(2)`, so record 0 produces no
            // events) -- see the matching guard in `run_closed_loop_with_geo` (#367). Without
            // it, this push re-emits `start_time` a second time, now labelling the state
            // *after* the first event as if it were still the pre-event seed: every RBPF
            // output carried one extra leading row, one longer than its reference.
            if last_ts != start_time {
                let (mean, cov) = rbpf.estimate_with_extra_states();
                results.push(NavigationResult::from_particle_filter_with_geo(
                    &last_ts, &mean, &cov, geo_layout,
                ));
            }
            last_ts = ts;
        }

        // Checked before the event rather than after it, so the recoverable-measurement
        // `continue` below cannot skip progress tracking -- see the matching guard in
        // `run_closed_loop_with_geo` (#367).
        execution_monitor.check("particle-filter")?;
        execution_monitor.mark_progress();

        match event {
            Event::Imu { dt_s, imu, .. } => {
                rbpf.predict(&imu, dt_s)?;
            }
            Event::Measurement { meas, .. } => match rbpf.update(meas.as_ref()) {
                Ok(_outcome) => {
                    consecutive_rejections = 0;
                }
                // A measurement the filter cannot use -- chiefly an off-map geophysical
                // sample -- leaves the state untouched and valid. Aborting on it would make
                // geophysical aiding unusable at map edges, which is the condition it exists
                // to handle (mirrors `run_closed_loop_with_geo`, #254).
                Err(e) if e.is_recoverable() => {
                    rejected_measurements += 1;
                    consecutive_rejections += 1;
                    log::warn!("Measurement rejected at {ts} (#{i}): {e}");
                    if consecutive_rejections > RBPF_MAX_CONSECUTIVE_REJECTIONS {
                        return Err(strapdown::StrapdownError::FilterDiverged {
                            consecutive_rejections,
                            limit: RBPF_MAX_CONSECUTIVE_REJECTIONS,
                            detail: format!("most recently at {ts} (#{i}): {e}"),
                        }
                        .into());
                    }
                    // State is unchanged, so the health check below has nothing new to judge.
                    continue;
                }
                Err(e) => {
                    log::error!("Filter update failed at {ts} (#{i}): {e}");
                    return Err(e.into());
                }
            },
        }

        // The health monitor sees the geophysical states too: it reads position and velocity
        // by index from the front and then sweeps the covariance diagonal, so a diverging bias
        // variance is caught here the same way it already is on the Kalman paths, which hand
        // `run_closed_loop` the full augmented state.
        let (mean, cov) = rbpf.estimate_with_extra_states();
        if let Err(e) = monitor.check(mean.as_slice(), &cov, None) {
            return Err(e.into());
        }
    }

    // Report the total even when it is zero: a silent run and a run that rejected every
    // measurement look identical from the outside otherwise.
    if rejected_measurements > 0 {
        log::warn!(
            "particle-filter run completed with {rejected_measurements} of {total} events rejected as unusable measurements"
        );
    }

    // Flush the final epoch: the boundary push above only fires when a later timestamp
    // arrives, and there is none.
    if last_ts != start_time {
        let (mean, cov) = rbpf.estimate_with_extra_states();
        results.push(NavigationResult::from_particle_filter_with_geo(
            &last_ts, &mean, &cov, geo_layout,
        ));
    }

    Ok(results)
}

/// Execute particle filter simulation
fn run_particle_filter(args: &ParticleFilterSimArgs) -> Result<(), Box<dyn Error>> {
    refuse_removed_particle_filter_flags(args)?;
    validate_input_path(&args.sim.input)?;
    validate_output_path(&args.sim.output)?;

    let csv_files = get_csv_files(&args.sim.input)?;
    let is_multiple = csv_files.len() > 1;
    let execution_limits = execution_limits_from_args(&args.sim);
    let health_limits = health_limits_from_args(&args.sim);

    if is_multiple {
        info!("Processing {} CSV files from directory", csv_files.len());
    }

    let mut failures = 0usize;
    for input_file in &csv_files {
        info!("Processing file: {}", input_file.display());

        // One unusable file must not abandon the rest of a batch. Before #311 this loop
        // could not fail here at all -- `from_csv` returned `Ok(vec![])` and the run wrote an
        // empty output file -- so aborting would trade a silent wrong answer for a loud
        // incomplete one. `run_from_config` already counts per-file failures and continues;
        // this matches it.
        let records = match load_records(input_file) {
            Ok(records) => records,
            Err(e) if is_multiple => {
                error!("Skipping {}: {e}", input_file.display());
                failures += 1;
                continue;
            }
            Err(e) => return Err(e),
        };

        let aiding = {
            // The barometer and magnetometer schedules have no CLI flag; they take their
            // 1 Hz default, overridable from a config file through serde.
            let mut built = strapdown::messages::AidingConfig::default();
            built.scheduler = build_scheduler(&args.scheduler);
            built.fault = build_fault(&args.fault);
            built.seed = args.seed;
            built
        };

        #[cfg(feature = "geonav")]
        let (gravity_map, magnetic_map) = {
            if args.geo.geo {
                if args.geo.gravity_resolution.is_none() && args.geo.magnetic_resolution.is_none() {
                    return Err("At least one of --gravity-resolution or --magnetic-resolution must be specified when using --geo".into());
                }

                let gravity_map = if let Some(res) = args.geo.gravity_resolution {
                    let map_path = match &args.geo.gravity_map_file {
                        Some(path) => path.clone(),
                        None => find_gravity_map(input_file)?,
                    };
                    info!("Loading gravity map from: {}", map_path.display());
                    let measurement_type =
                        GeophysicalMeasurementType::Gravity(convert_resolution_gravity(res));
                    Some(Rc::new(GeoMap::load_geomap(&map_path, measurement_type)?))
                } else {
                    None
                };

                let magnetic_map = if let Some(res) = args.geo.magnetic_resolution {
                    let map_path = match &args.geo.magnetic_map_file {
                        Some(path) => path.clone(),
                        None => find_magnetic_map(input_file)?,
                    };
                    info!("Loading magnetic map from: {}", map_path.display());
                    let measurement_type =
                        GeophysicalMeasurementType::Magnetic(convert_resolution_magnetic(res));
                    Some(Rc::new(GeoMap::load_geomap(&map_path, measurement_type)?))
                } else {
                    None
                };

                (gravity_map, magnetic_map)
            } else {
                (None, None)
            }
        };

        // Each map bias's seed, prior and random walk -- `--gravity-bias`,
        // `--gravity-bias-init-std` and the rest -- resolved exactly as `cl` resolves them.
        // This path used to ignore every one of those and take a single
        // `--geo-bias-init-std` for all channels.
        #[cfg(feature = "geonav")]
        let geo_bias = geo_bias_setup(
            gravity_map
                .is_some()
                .then_some(MapBiasPrior::gravity_from_args(&args.geo, &args.geo_bias)),
            magnetic_map
                .is_some()
                .then_some(MapBiasPrior::magnetic_from_args(&args.geo, &args.geo_bias)),
        );

        #[cfg(not(feature = "geonav"))]
        let event_stream = build_event_stream(&records, &aiding, args.sim.enu)?;

        // As above: one layout drives both the measurements' declaration and the filter's
        // extra states. This path also builds an RBPF, so the base is the navigation states.
        #[cfg(feature = "geonav")]
        let geo_bias_layout = GeoBiasLayout::appended(
            NAVIGATION_STATE_DIM,
            gravity_map.is_some(),
            magnetic_map.is_some(),
        )?;

        #[cfg(feature = "geonav")]
        let event_stream = if args.geo.geo {
            geo_build_event_stream(
                &records,
                &aiding,
                args.sim.enu,
                &GeophysicalAiding {
                    gravity_noise_std: gravity_map.as_ref().map(|_| args.geo.gravity_noise_std),
                    magnetic_noise_std: magnetic_map.as_ref().map(|_| args.geo.magnetic_noise_std),
                    gravity_map: gravity_map.clone(),
                    magnetic_map: magnetic_map.clone(),
                    interval_s: args.geo.geo_interval_s,
                    bias_layout: geo_bias_layout,
                },
            )?
        } else {
            build_event_stream(&records, &aiding, args.sim.enu)?
        };

        #[cfg(feature = "geonav")]
        let geo_bias_dim = geo_bias_layout.map_or(0, |layout| layout.bias_count());
        #[cfg(not(feature = "geonav"))]
        let geo_bias_dim = 0usize;

        // The same placement, restated for `NavigationResult`, which lives in `core` and
        // so cannot name `GeoBiasLayout`. Derived from that layout rather than rebuilt
        // from the map flags, exactly as `run_geo_closed_loop_cli` derives the Kalman one,
        // so where the biases live is decided once. The unaided case is
        // `PARTICLE_NONE` and not `NONE`: this filter's estimate is nine states, not the
        // Kalman filters' fifteen.
        #[cfg(feature = "geonav")]
        let geo_layout = geo_bias_layout.map_or(ExtraStateLayout::PARTICLE_NONE, |layout| {
            ExtraStateLayout::new(
                layout.state_dim(),
                layout.gravity_bias().map(|bias| bias.index),
                layout.magnetic_bias().map(|bias| bias.index),
            )
        });
        #[cfg(not(feature = "geonav"))]
        let geo_layout = ExtraStateLayout::PARTICLE_NONE;

        check_declared_frame(&records, args.sim.enu)?;
        let first = &records[0];
        // Quaternion, not Euler angles: `TestDataRecord`'s roll/pitch/yaw are a different
        // convention from nalgebra's XYZ. See `TestDataRecord::attitude`.
        let attitude = first.attitude();
        let (velocity_north, velocity_east) = first.ground_track_velocity();
        let nominal = strapdown::StrapdownState {
            latitude: first.latitude.to_radians(),
            longitude: first.longitude.to_radians(),
            altitude: first.altitude,
            velocity_north,
            velocity_east,
            velocity_vertical: 0.0,
            attitude,
            // The declared frame, as everywhere else. This path builds its own nominal
            // state rather than going through `initialize_*`, so the guard is run
            // explicitly above (#296).
            is_enu: args.sim.enu,
        };

        let process_noise_std_m = Vector3::new(
            args.process_noise_std_m[0],
            args.process_noise_std_m[1],
            args.process_noise_std_m[2],
        );

        let config = {
            let mut built = RbpfConfig::default();
            built.num_particles = args.num_particles;
            built.position_init_std_m =
                Vector3::new(args.position_std, args.position_std, args.position_std);
            built.velocity_init_std_mps = args.velocity_std;
            built.attitude_init_std_rad = args.attitude_std;
            built.position_process_noise_std_m = process_noise_std_m;
            built.velocity_process_noise_std_mps = args.velocity_process_noise_std_mps;
            built.attitude_process_noise_std_rad = args.attitude_process_noise_std_rad;
            built.extra_state_dim = geo_bias_dim;
            #[cfg(feature = "geonav")]
            geo_bias.apply_to_rbpf(&mut built);
            built.seed = args.seed;
            built.zero_vertical_velocity = args.zero_vertical_velocity;
            built.zero_vertical_velocity_std_mps = args.zero_vertical_velocity_std_mps;
            built
        };

        // `ParticleFilterType` has a single variant today (#259 removed the two that were
        // advertised but never implemented). Dispatching on it anyway keeps adding a second
        // concrete filter a matter of extending this match rather than rediscovering that
        // the flag was never read.
        let results = match args.filter_type {
            ParticleFilterType::RaoBlackwellized => {
                let mut rbpf = RaoBlackwellizedParticleFilter::new(nominal, config)?;
                run_rbpf_event_loop(
                    &mut rbpf,
                    event_stream,
                    &execution_limits,
                    &health_limits,
                    geo_layout,
                )?
            }
        };

        let output_file = resolve_output_path(&args.sim.output, input_file, &csv_files)?;

        NavigationResult::to_csv(&results, &output_file)?;
        info!("Results written to {}", output_file.display());
    }

    info!("Particle filter simulation complete");

    if failures > 0 {
        error!("{failures} file(s) skipped because they held no usable records");
    }

    Ok(())
}

/// Prompt for simulation mode with validation
fn prompt_simulation_mode() -> SimulationMode {
    loop {
        println!(
            "Please specify the simulation mode you would like:\n\
            [1] - Dead Reckoning\n\
            [2] - Open-Loop (Feed-Forward)\n\
            [3] - Closed-Loop (Feedback)\n\
            [4] - Particle Filter\n\
            [q] - Quit\n"
        );
        if let Some(input) = read_user_input() {
            match input.as_str() {
                "1" => return SimulationMode::DeadReckoning,
                "2" => return SimulationMode::OpenLoop,
                "3" => return SimulationMode::ClosedLoop,
                "4" => return SimulationMode::ParticleFilter,
                _ => println!("Error: Invalid selection. Please enter 1, 2, 3, or q.\n"),
            }
        }
    }
}

/// Prompt for filter type with validation
fn prompt_filter_type() -> FilterType {
    loop {
        println!(
            "Please specify the filter type you would like:\n\
            [1] - Error-State Kalman Filter (ESKF, default)\n\
            [2] - Unscented Kalman Filter (UKF)\n\
            [3] - Extended Kalman Filter (EKF)\n\
            [q] - Quit\n"
        );
        if let Some(input) = read_user_input() {
            match input.as_str() {
                "1" => return FilterType::Eskf,
                "2" => return FilterType::Ukf,
                "3" => return FilterType::Ekf,
                _ => println!("Error: Invalid selection. Please enter 1, 2, 3, or q.\n"),
            }
        }
    }
}

/// Prompt for random seed (optional)
fn prompt_seed() -> u64 {
    loop {
        println!("Please specify a random seed (press Enter for default 42, or 'q' to quit):");
        match read_user_input() {
            None => return 42,
            Some(input) => match input.parse::<u64>() {
                Ok(seed) => return seed,
                Err(_) => println!("Error: Invalid seed. Please enter a positive integer.\n"),
            },
        }
    }
}

/// Prompt for GNSS scheduler configuration
fn prompt_measurement_scheduler() -> MeasurementScheduler {
    loop {
        println!(
            "Would you like to add a GNSS scheduler to simulate periodic denial or jamming?\n\
            [1] - Pass-Through (no degradation)\n\
            [2] - Fixed Interval (specific time and phase between measurements)\n\
            [3] - Duty Cycle (duration on/off)\n\
            [q] - Quit\n"
        );
        if let Some(input) = read_user_input() {
            match input.as_str() {
                "1" => return MeasurementScheduler::PassThrough,
                "2" => return prompt_fixed_interval_scheduler(),
                "3" => return prompt_duty_cycle_scheduler(),
                _ => println!("Error: Invalid selection. Please enter 1, 2, 3, or q.\n"),
            }
        }
    }
}

/// Prompt for Fixed Interval scheduler parameters
fn prompt_fixed_interval_scheduler() -> MeasurementScheduler {
    let interval_s = loop {
        println!("Enter the interval between measurements in seconds (or 'q' to quit):");
        if let Some(input) = read_user_input() {
            match input.parse::<f64>() {
                Ok(val) if val > 0.0 => break val,
                _ => println!("Error: Please enter a positive number.\n"),
            }
        }
    };

    let phase_s = loop {
        println!("Enter the initial phase offset in seconds (press Enter for 0, or 'q' to quit):");
        match read_user_input() {
            None => break 0.0,
            Some(input) => match input.parse::<f64>() {
                Ok(val) if val >= 0.0 => break val,
                _ => println!("Error: Please enter a non-negative number.\n"),
            },
        }
    };

    MeasurementScheduler::FixedInterval {
        interval_s,
        phase_s,
    }
}

/// Prompt for Duty Cycle scheduler parameters
fn prompt_duty_cycle_scheduler() -> MeasurementScheduler {
    let on_s = loop {
        println!("Enter the ON duration in seconds (or 'q' to quit):");
        if let Some(input) = read_user_input() {
            match input.parse::<f64>() {
                Ok(val) if val > 0.0 => break val,
                _ => println!("Error: Please enter a positive number.\n"),
            }
        }
    };

    let off_s = loop {
        println!("Enter the OFF duration in seconds (or 'q' to quit):");
        if let Some(input) = read_user_input() {
            match input.parse::<f64>() {
                Ok(val) if val > 0.0 => break val,
                _ => println!("Error: Please enter a positive number.\n"),
            }
        }
    };

    let start_phase_s = loop {
        println!("Enter the start phase offset in seconds (press Enter for 0, or 'q' to quit):");
        match read_user_input() {
            None => break 0.0,
            Some(input) => match input.parse::<f64>() {
                Ok(val) if val >= 0.0 => break val,
                _ => println!("Error: Please enter a non-negative number.\n"),
            },
        }
    };

    MeasurementScheduler::DutyCycle {
        on_s,
        off_s,
        start_phase_s,
    }
}

/// Prompt for GNSS fault model configuration
fn prompt_gnss_fault_model() -> strapdown::messages::GnssFaultModel {
    use strapdown::messages::GnssFaultModel;

    loop {
        println!(
            "Would you like to add a GNSS fault model to corrupt measurements?\n\
            [1] - None (no corruption)\n\
            [2] - Degraded (AR(1) random walk with increased uncertainty)\n\
            [3] - Slow Bias (slowly drifting bias)\n\
            [4] - Hijack (position spoofing)\n\
            [q] - Quit\n"
        );
        if let Some(input) = read_user_input() {
            match input.as_str() {
                "1" => return GnssFaultModel::None,
                "2" => return prompt_degraded_fault_model(),
                "3" => return prompt_slow_bias_fault_model(),
                "4" => return prompt_hijack_fault_model(),
                _ => println!("Error: Invalid selection. Please enter 1, 2, 3, 4, or q.\n"),
            }
        }
    }
}

/// Prompt for Degraded fault model parameters
fn prompt_degraded_fault_model() -> strapdown::messages::GnssFaultModel {
    use strapdown::messages::GnssFaultModel;

    println!("\nConfiguring Degraded (AR(1)) fault model...");

    let rho_pos = prompt_f64_with_default("Position autocorrelation (rho_pos)", 0.99, 0.0, 1.0);
    let sigma_pos_m =
        prompt_f64_with_default("Position noise std dev (meters)", 3.0, 0.0, f64::MAX);
    let rho_vel = prompt_f64_with_default("Velocity autocorrelation (rho_vel)", 0.95, 0.0, 1.0);
    let sigma_vel_mps = prompt_f64_with_default("Velocity noise std dev (m/s)", 0.3, 0.0, f64::MAX);
    let r_scale = prompt_f64_with_default("Measurement noise scaling factor", 5.0, 0.0, f64::MAX);

    GnssFaultModel::Degraded {
        rho_pos,
        sigma_pos_m,
        rho_vel,
        sigma_vel_mps,
        r_scale,
        // The wizard writes the per-fix form, which is what the prompts above describe.
        // A correlation time is an expert knob -- it reinterprets `sigma_pos_m` as a
        // steady-state rather than a per-step value -- so it is left to be added by hand.
        tau_pos_s: None,
        tau_vel_s: None,
    }
}

/// Prompt for Slow Bias fault model parameters
fn prompt_slow_bias_fault_model() -> strapdown::messages::GnssFaultModel {
    use strapdown::messages::GnssFaultModel;

    println!("\nConfiguring Slow Bias fault model...");

    let drift_n_mps = prompt_f64_with_default("North drift rate (m/s)", 0.02, f64::MIN, f64::MAX);
    let drift_e_mps = prompt_f64_with_default("East drift rate (m/s)", 0.0, f64::MIN, f64::MAX);
    let q_bias = prompt_f64_with_default("Bias process noise", 1e-6, 0.0, f64::MAX);
    let rotate_omega_rps =
        prompt_f64_with_default("Rotation rate (rad/s)", 0.0, f64::MIN, f64::MAX);

    GnssFaultModel::SlowBias {
        drift_n_mps,
        drift_e_mps,
        q_bias,
        rotate_omega_rps,
    }
}

/// Prompt for Hijack fault model parameters
fn prompt_hijack_fault_model() -> strapdown::messages::GnssFaultModel {
    use strapdown::messages::GnssFaultModel;

    println!("\nConfiguring Hijack (spoofing) fault model...");

    let offset_n_m = prompt_f64_with_default("North offset (meters)", 50.0, f64::MIN, f64::MAX);
    let offset_e_m = prompt_f64_with_default("East offset (meters)", 0.0, f64::MIN, f64::MAX);
    let start_s = prompt_f64_with_default("Start time (seconds)", 120.0, 0.0, f64::MAX);
    let duration_s = prompt_f64_with_default("Duration (seconds)", 60.0, 0.0, f64::MAX);

    GnssFaultModel::Hijack {
        offset_n_m,
        offset_e_m,
        start_s,
        duration_s,
    }
}

/// Prompt for parallel execution preference
fn prompt_parallel() -> bool {
    loop {
        println!(
            "Would you like to run simulations in parallel when processing multiple files?\n\
            [y] - Yes (parallel execution)\n\
            [n] - No (sequential execution, default)\n\
            [q] - Quit\n"
        );
        match read_user_input() {
            None => return false,
            Some(input) => match input.to_lowercase().as_str() {
                "y" | "yes" => return true,
                "n" | "no" => return false,
                _ => println!("Error: Please enter 'y' or 'n'.\n"),
            },
        }
    }
}

/// Prompt for the local-level frame the input records are expressed in
fn prompt_frame() -> bool {
    loop {
        println!(
            "Which local-level frame is the input data expressed in?\n\
            [n] - NED, north-east-down (default; `strapdown-sim syn` output)\n\
            [e] - ENU, east-north-up (Sensor Logger exports)\n\
            [q] - Quit\n\
            \n\
            A CSV carries no frame tag, so this cannot be inferred. Getting it wrong makes \
            the mechanization add the gravity model to the sensed specific force instead of \
            cancelling it, which is checked for and rejected before propagation.\n"
        );
        match read_user_input() {
            None => return false,
            Some(input) => match input.to_lowercase().as_str() {
                "n" | "ned" => return false,
                "e" | "enu" => return true,
                _ => println!("Error: Please enter 'n' or 'e'.\n"),
            },
        }
    }
}

/// Prompt for log level
fn prompt_log_level() -> strapdown::sim::LogLevel {
    use strapdown::sim::LogLevel;
    loop {
        println!(
            "Please select the log level:\n\
            [1] - off\n\
            [2] - error\n\
            [3] - warn\n\
            [4] - info (default)\n\
            [5] - debug\n\
            [6] - trace\n\
            [q] - Quit\n"
        );
        match read_user_input() {
            None => return LogLevel::Info,
            Some(input) => match input.as_str() {
                "1" => return LogLevel::Off,
                "2" => return LogLevel::Error,
                "3" => return LogLevel::Warn,
                "4" => return LogLevel::Info,
                "5" => return LogLevel::Debug,
                "6" => return LogLevel::Trace,
                _ => println!("Error: Invalid selection. Please enter 1-6.\n"),
            },
        }
    }
}

/// Prompt for log file path
fn prompt_log_file() -> Option<String> {
    println!("Please specify a log file path (press Enter to log to stderr, or 'q' to quit):");
    match read_user_input() {
        Some(input) if !input.trim().is_empty() => Some(input),
        _ => None,
    }
}

/// Prompt for whether to enable geophysical navigation
fn prompt_enable_geophysical() -> bool {
    use std::io::{self, Write};

    loop {
        println!("\nEnable geophysical navigation (gravity/magnetic anomaly measurements)?");
        println!("  (y)es");
        println!("  (n)o");
        print!("Choice: ");
        let _ = io::stdout().flush();

        if let Some(input) = read_user_input() {
            match input.to_lowercase().as_str() {
                "y" | "yes" => return true,
                "n" | "no" => return false,
                "q" | "quit" => {
                    println!("Configuration cancelled.");
                    std::process::exit(0);
                }
                _ => {
                    println!("Invalid input. Please enter 'y', 'n', or 'q' to quit.");
                }
            }
        } else {
            println!("Invalid input. Please enter 'y', 'n', or 'q' to quit.");
        }
    }
}

/// Type alias for geophysical measurement configuration returned by prompt functions.
///
/// Represents the configuration for a single geophysical measurement type (gravity or magnetic).
/// Contains:
/// - `GeoResolution`: The map resolution to use
/// - `Option<f64>`: Measurement bias (mGal for gravity, nT for magnetic)
/// - `Option<f64>`: Measurement noise standard deviation (mGal for gravity, nT for magnetic)
/// - `Option<String>`: Map file path (auto-detected if None)
type GeoMeasurementConfig = (
    strapdown::sim::GeoResolution,
    Option<f64>,
    Option<f64>,
    Option<String>,
);

/// Prompt for `GeoResolution` with validation
fn prompt_geo_resolution(measurement_type: &str) -> strapdown::sim::GeoResolution {
    use std::io::{self, Write};
    use strapdown::sim::GeoResolution;

    loop {
        println!("\nSelect {measurement_type} map resolution:");
        println!("  1. One Degree");
        println!("  2. Thirty Minutes");
        println!("  3. Twenty Minutes");
        println!("  4. Fifteen Minutes");
        println!("  5. Ten Minutes");
        println!("  6. Six Minutes");
        println!("  7. Five Minutes");
        println!("  8. Four Minutes");
        println!("  9. Three Minutes");
        println!(" 10. Two Minutes");
        println!(" 11. One Minute (default)");
        println!(" 12. Thirty Seconds");
        println!(" 13. Fifteen Seconds");
        println!(" 14. Three Seconds");
        println!(" 15. One Second");
        print!("Choice [11]: ");
        let _ = io::stdout().flush();

        match read_user_input() {
            Some(input) if input.is_empty() => return GeoResolution::OneMinute,
            Some(input) => match input.as_str() {
                "1" => return GeoResolution::OneDegree,
                "2" => return GeoResolution::ThirtyMinutes,
                "3" => return GeoResolution::TwentyMinutes,
                "4" => return GeoResolution::FifteenMinutes,
                "5" => return GeoResolution::TenMinutes,
                "6" => return GeoResolution::SixMinutes,
                "7" => return GeoResolution::FiveMinutes,
                "8" => return GeoResolution::FourMinutes,
                "9" => return GeoResolution::ThreeMinutes,
                "10" => return GeoResolution::TwoMinutes,
                "11" => return GeoResolution::OneMinute,
                "12" => return GeoResolution::ThirtySeconds,
                "13" => return GeoResolution::FifteenSeconds,
                "14" => return GeoResolution::ThreeSeconds,
                "15" => return GeoResolution::OneSecond,
                "q" | "quit" => {
                    println!("Configuration cancelled.");
                    std::process::exit(0);
                }
                _ => {
                    println!(
                        "Invalid choice. Please enter a number between 1 and 15, or 'q' to quit."
                    );
                }
            },
            None => return GeoResolution::OneMinute,
        }
    }
}

/// Prompt for gravity measurement configuration
fn prompt_gravity_config() -> Option<GeoMeasurementConfig> {
    use std::io::{self, Write};

    loop {
        println!("\nEnable gravity anomaly measurements?");
        println!("  (y)es");
        println!("  (n)o");
        print!("Choice: ");
        let _ = io::stdout().flush();

        if let Some(input) = read_user_input() {
            match input.to_lowercase().as_str() {
                "y" | "yes" => {
                    let resolution = prompt_geo_resolution("gravity");

                    println!("\nGravity measurement bias (mGal) [0.0]: ");
                    let bias = match read_user_input() {
                        Some(input) if !input.is_empty() => {
                            if let Ok(v) = input.parse::<f64>() {
                                Some(v)
                            } else {
                                println!("Invalid number. Using default (0.0).");
                                None
                            }
                        }
                        _ => None,
                    };

                    let noise_std = prompt_f64_with_default(
                        "Gravity measurement noise std dev (mGal)",
                        100.0,
                        0.0,
                        f64::MAX,
                    );

                    println!("\nGravity map file path (press Enter to auto-detect): ");
                    let map_file = match read_user_input() {
                        Some(input) if !input.is_empty() => Some(input),
                        _ => None,
                    };

                    return Some((resolution, bias, Some(noise_std), map_file));
                }
                "n" | "no" => return None,
                "q" | "quit" => {
                    println!("Configuration cancelled.");
                    std::process::exit(0);
                }
                _ => {
                    println!("Invalid input. Please enter 'y', 'n', or 'q' to quit.");
                }
            }
        } else {
            println!("Invalid input. Please enter 'y', 'n', or 'q' to quit.");
        }
    }
}

/// Prompt for magnetic measurement configuration
fn prompt_magnetic_config() -> Option<GeoMeasurementConfig> {
    use std::io::{self, Write};

    loop {
        println!("\nEnable magnetic anomaly measurements?");
        println!("  (y)es");
        println!("  (n)o");
        print!("Choice: ");
        let _ = io::stdout().flush();

        if let Some(input) = read_user_input() {
            match input.to_lowercase().as_str() {
                "y" | "yes" => {
                    let resolution = prompt_geo_resolution("magnetic");

                    println!("\nMagnetic measurement bias (nT) [0.0]: ");
                    let bias = match read_user_input() {
                        Some(input) if !input.is_empty() => {
                            if let Ok(v) = input.parse::<f64>() {
                                Some(v)
                            } else {
                                println!("Invalid number. Using default (0.0).");
                                None
                            }
                        }
                        _ => None,
                    };

                    let noise_std = prompt_f64_with_default(
                        "Magnetic measurement noise std dev (nT)",
                        150.0,
                        0.0,
                        f64::MAX,
                    );

                    println!("\nMagnetic map file path (press Enter to auto-detect): ");
                    let map_file = match read_user_input() {
                        Some(input) if !input.is_empty() => Some(input),
                        _ => None,
                    };

                    return Some((resolution, bias, Some(noise_std), map_file));
                }
                "n" | "no" => return None,
                "q" | "quit" => {
                    println!("Configuration cancelled.");
                    std::process::exit(0);
                }
                _ => {
                    println!("Invalid input. Please enter 'y', 'n', or 'q' to quit.");
                }
            }
        } else {
            println!("Invalid input. Please enter 'y', 'n', or 'q' to quit.");
        }
    }
}

/// Prompt for geophysical measurement frequency
fn prompt_geo_measurement_interval() -> Option<f64> {
    println!("\nSeconds between geophysical measurements [auto]: ");

    match read_user_input() {
        Some(input) if !input.is_empty() => match input.parse::<f64>() {
            Ok(interval) if interval > 0.0 => Some(interval),
            Ok(_) => {
                println!("Interval must be positive. Using auto.");
                None
            }
            Err(_) => {
                println!("Invalid number. Using auto.");
                None
            }
        },
        _ => None,
    }
}

/// Interactive configuration file creation wizard that creates a custom
/// [`SimulationConfig`] and writes it to file.
fn create_config_file() -> Result<(), Box<dyn Error>> {
    println!("\n=== Strapdown Simulation Configuration Wizard ===\n");

    // Gather all configuration parameters
    let config_name = prompt_config_name();
    let save_path = prompt_config_path();

    println!("\nCreating configuration file at: {save_path}/{config_name}\n");
    let input_path = prompt_input_path();
    let output_path = prompt_output_path();
    let mode = prompt_simulation_mode();
    let seed = prompt_seed();
    let is_enu = prompt_frame();
    let parallel = prompt_parallel();
    let execution_limits = ExecutionLimits::default();
    let health_limits = HealthLimits::default();

    // Logging configuration
    println!("\n--- Logging Configuration ---");
    let log_level = prompt_log_level();
    let log_file = prompt_log_file();
    let logging = strapdown::sim::LoggingConfig {
        level: log_level,
        file: log_file,
    };

    // Mode-specific configuration
    let closed_loop = if matches!(mode, SimulationMode::ClosedLoop) {
        let filter = prompt_filter_type();
        let closed_loop_cfg = {
            let mut built = strapdown::sim::ClosedLoopConfig::default();
            built.filter = filter;
            built
        };
        Some(closed_loop_cfg)
    } else {
        None
    };

    let particle_filter = if matches!(mode, SimulationMode::ParticleFilter) {
        println!("\nParticle filter configuration uses default values.");
        println!("Edit the generated config file to customize particle filter settings.");
        Some(strapdown::sim::ParticleFilterConfig::default())
    } else {
        None
    };

    // Aiding configuration
    let scheduler = prompt_measurement_scheduler();
    let fault = prompt_gnss_fault_model();

    let aiding = {
        let mut built = strapdown::messages::AidingConfig::default();
        built.scheduler = scheduler;
        built.fault = fault;
        built.seed = seed;
        built
    };

    // Geophysical navigation configuration
    println!("\n--- Geophysical Navigation Configuration ---");
    let geophysical = if prompt_enable_geophysical() {
        let gravity_config = prompt_gravity_config();
        let magnetic_config = prompt_magnetic_config();

        // Validate that at least one measurement type is enabled
        if gravity_config.is_none() && magnetic_config.is_none() {
            println!(
                "\nWarning: Geophysical navigation enabled but no measurement types selected."
            );
            println!("Disabling geophysical navigation.");
            None
        } else {
            let geo_interval_s = prompt_geo_measurement_interval();

            let (gravity_resolution, gravity_bias, gravity_noise_std, gravity_map_file) =
                gravity_config.map_or((None, None, None, None), |(res, bias, noise, map)| {
                    (Some(res), bias, noise, map)
                });

            let (magnetic_resolution, magnetic_bias, magnetic_noise_std, magnetic_map_file) =
                magnetic_config.map_or((None, None, None, None), |(res, bias, noise, map)| {
                    (Some(res), bias, noise, map)
                });

            Some({
                let mut built = strapdown::sim::GeophysicalConfig::default();
                built.gravity_resolution = gravity_resolution;
                built.gravity_bias = gravity_bias;
                built.gravity_noise_std = gravity_noise_std;
                built.gravity_map_file = gravity_map_file;
                built.magnetic_resolution = magnetic_resolution;
                built.magnetic_bias = magnetic_bias;
                built.magnetic_noise_std = magnetic_noise_std;
                built.magnetic_map_file = magnetic_map_file;
                built.geo_interval_s = geo_interval_s;
                built
            })
        }
    } else {
        None
    };

    // The geophysical section is offered after the filter has already been chosen, and the
    // ESKF has no geophysical implementation -- so an `eskf` answer followed by a `yes` here
    // writes a file that `GeoClosedLoopSettings::validate` refuses. The wizard exists to
    // produce a runnable config, so reconcile rather than emit one that fails. It matters
    // more than it sounds: `Eskf` is `FilterType`'s `#[default]`.
    let closed_loop = match (closed_loop, geophysical.is_some()) {
        (Some(cfg), true) if matches!(cfg.filter, FilterType::Eskf) => {
            println!(
                "\nNote: the ESKF has no geophysical implementation, so the filter has been \
                 switched to the UKF.\n      Edit `[closed_loop] filter` in the generated file \
                 to use the EKF instead."
            );
            Some({
                let mut built = cfg;
                built.filter = FilterType::Ukf;
                built
            })
        }
        (other, _) => other,
    };

    // Build the complete configuration
    let config = {
        let mut built = SimulationConfig::default();
        built.input = input_path;
        built.output = output_path;
        built.mode = mode;
        built.seed = seed;
        built.is_enu = is_enu;
        built.parallel = parallel;
        built.generate_plot = false;
        built.execution_limits = execution_limits;
        built.health_limits = health_limits;
        built.logging = logging;
        built.closed_loop = closed_loop;
        built.particle_filter = particle_filter;
        built.geophysical = geophysical;
        built.aiding = aiding;
        built.synthetic = None;
        built
    };

    // validate output location exists and write to file using appropriate format based on file extension
    let config_output_path = Path::new(&save_path).join(&config_name);
    if let Some(parent) = config_output_path.parent()
        && !parent.as_os_str().is_empty()
        && !parent.exists()
    {
        std::fs::create_dir_all(parent)?;
    }
    config.to_file(&config_output_path)?;

    println!(
        "\n✓ Configuration file successfully created: {}",
        config_output_path.display()
    );
    println!("\nYou can now run the simulation with:");
    println!("  strapdown-sim --config {}", config_output_path.display());

    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();

    // If --config is provided, load config and potentially override logger with config values
    if let Some(ref config_path) = cli.config {
        // Load config first to get logging preferences
        let config = SimulationConfig::from_file(config_path)?;

        // Determine log level: CLI flag takes precedence over config.
        //
        // `--log-level` is `global = true` with a default of "info", so it is always
        // populated and there is no `Option` to tell "the user asked for info" apart from
        // "nobody asked". Comparing against the default is the available discrimination:
        // anything else was typed. The comment here used to claim CLI precedence while the
        // line below took the config's level unconditionally, so `--log-level debug
        // --config ...` silently ran at whatever the file said.
        let log_level = if cli.log_level == DEFAULT_LOG_LEVEL {
            config.logging.level.as_str()
        } else {
            cli.log_level.as_str()
        };

        // Create PathBuf from config file string if needed
        let config_log_file = config.logging.file.as_ref().map(PathBuf::from);
        let log_file = cli.log_file.as_ref().or(config_log_file.as_ref());

        // Initialize logger with resolved settings
        init_logger(log_level, log_file)?;

        return run_from_config(config_path, cli.parallel, cli.plot);
    }

    // Initialize logger with CLI settings for command-line mode
    init_logger(&cli.log_level, cli.log_file.as_ref())?;

    // Otherwise, execute based on subcommand
    match cli.command {
        Some(Command::DeadReckoning(args)) => {
            info!(
                "Running in Dead Reckoning mode with input: {}",
                args.input.display()
            );
            run_dead_reckoning(&args)
        }
        Some(Command::OpenLoop(args)) => run_open_loop(&args),
        Some(Command::ClosedLoop(args)) => run_closed_loop_cli(&args),
        Some(Command::ParticleFilter(args)) => run_particle_filter(&args),
        Some(Command::CreateConfig) => create_config_file(),
        Some(Command::Synthetic(args)) => run_synthetic(&args),
        None => {
            eprintln!("Error: No command provided. Use -h or --help for usage information.");
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// Write a TOML fixture to a temp file and load it the way `--config` does.
    ///
    /// Through `SimulationConfig::from_file` rather than a direct `toml::from_str` so the
    /// tests exercise the extension dispatch and serde aliases the real path uses -- and so
    /// `sim` does not gain a `toml` dependency solely for its tests.
    #[cfg(feature = "geonav")]
    fn config_from_toml(body: &str) -> SimulationConfig {
        let path = std::env::temp_dir().join(format!(
            "strapdown-sim-test-{}-{:?}.toml",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::write(&path, body).expect("the fixture must be writable");
        let config = SimulationConfig::from_file(&path).expect("the fixture must parse");
        std::fs::remove_file(&path).ok();
        config
    }

    #[test]
    fn test_create_config_args_structure() {
        let args = CreateConfigArgs {
            output: PathBuf::from("test_config.toml"),
            mode: SimulationMode::ClosedLoop,
        };
        assert_eq!(args.output, PathBuf::from("test_config.toml"));
        assert!(matches!(args.mode, SimulationMode::ClosedLoop));
    }

    #[test]
    fn test_simulation_mode_variants() {
        let modes = [
            SimulationMode::OpenLoop,
            SimulationMode::ClosedLoop,
            SimulationMode::ParticleFilter,
        ];
        assert_eq!(modes.len(), 3);
    }

    #[test]
    fn test_filter_type_variants() {
        let filters = [FilterType::Ukf, FilterType::Ekf, FilterType::Eskf];
        assert_eq!(filters.len(), 3); // Updated to include ESKF
    }

    #[test]
    fn test_measurement_scheduler_variants() {
        let passthrough = MeasurementScheduler::PassThrough;
        let fixed = MeasurementScheduler::FixedInterval {
            interval_s: 1.0,
            phase_s: 0.0,
        };
        let duty = MeasurementScheduler::DutyCycle {
            on_s: 10.0,
            off_s: 10.0,
            start_phase_s: 0.0,
        };

        assert!(matches!(passthrough, MeasurementScheduler::PassThrough));
        assert!(matches!(fixed, MeasurementScheduler::FixedInterval { .. }));
        assert!(matches!(duty, MeasurementScheduler::DutyCycle { .. }));
    }

    #[test]
    fn test_config_file_formats() {
        // Test that we can detect different file extensions
        let toml_path = PathBuf::from("test.toml");
        let json_path = PathBuf::from("test.json");
        let yaml_path = PathBuf::from("test.yaml");

        assert_eq!(toml_path.extension().and_then(|s| s.to_str()), Some("toml"));
        assert_eq!(json_path.extension().and_then(|s| s.to_str()), Some("json"));
        assert_eq!(yaml_path.extension().and_then(|s| s.to_str()), Some("yaml"));
    }

    #[test]
    fn test_logging_config_default() {
        use strapdown::sim::{LogLevel, LoggingConfig};

        let logging = LoggingConfig::default();
        assert_eq!(logging.level, LogLevel::Info);
        assert!(logging.file.is_none());
    }

    #[test]
    fn test_simulation_config_with_parallel() {
        use strapdown::sim::{LogLevel, SimulationConfig};

        let config = SimulationConfig::default();
        assert!(!config.parallel); // Should default to false
        assert_eq!(config.logging.level, LogLevel::Info);
    }

    #[test]
    fn test_logging_config_creation() {
        use strapdown::sim::{LogLevel, LoggingConfig};

        let logging = LoggingConfig {
            level: LogLevel::Debug,
            file: Some("/tmp/test.log".to_string()),
        };
        assert_eq!(logging.level, LogLevel::Debug);
        assert_eq!(logging.file, Some("/tmp/test.log".to_string()));
    }

    /// The command line and a configuration file must resolve to the same settings.
    ///
    /// This is the assertion that keeps the two vocabularies from drifting. They describe the
    /// same run in different words -- `--sched fixed --interval-s 5` against
    /// `kind = "fixed_interval"` / `interval_s = 5.0`, `--gravity-resolution one-minute`
    /// against `gravity_resolution = "one_minute"` -- and before they shared a runner there
    /// was nothing to notice when one gained a default the other did not. `--ukf-alpha` was
    /// exactly that: `1e-3` on the flag against the `0.1` a config file takes, so the same
    /// named filter ran with sigma-point spreads two orders of magnitude apart.
    ///
    /// Compared through `Debug` rather than `PartialEq`: the settings carry `AidingConfig`,
    /// `HealthLimits` and `ExecutionLimits`, none of which implements it, and deriving it
    /// across `core`'s public API to serve one test is the larger change.
    #[cfg(feature = "geonav")]
    #[test]
    fn the_cli_and_config_paths_resolve_to_the_same_geophysical_settings() {
        let cli = Cli::try_parse_from([
            "strapdown-sim",
            "cl",
            "--geo",
            "-i",
            "in.csv",
            "-o",
            "out.csv",
            "--enu",
            "--seed",
            "42",
            "--filter",
            "ukf",
            "--gravity-resolution",
            "one-minute",
            "--gravity-noise-std",
            "10",
            "--magnetic-resolution",
            "two-minutes",
            "--magnetic-noise-std",
            "200",
            "--geo-interval-s",
            "1",
            "--sched",
            "fixed",
            "--interval-s",
            "5",
            "--phase-s",
            "0",
            "--fault",
            "degraded",
            "--rho-pos",
            "0.99",
            "--sigma-pos-m",
            "3",
            "--rho-vel",
            "0.95",
            "--sigma-vel-mps",
            "0.3",
            "--r-scale",
            "5",
            "--nis-pos-max",
            "1000",
            "--health-speed-mps-max",
            "5000",
        ])
        .expect("the flags above must parse");
        let Some(Command::ClosedLoop(args)) = cli.command else {
            panic!("expected the `cl` subcommand");
        };
        let from_cli = geo_settings_from_args(&args).expect("CLI settings must resolve");

        let toml = r#"
input = "in.csv"
output = "out.csv"
mode = "closed-loop"
is_enu = true
seed = 42

[closed_loop]
filter = "ukf"

[geophysical]
gravity_resolution = "one_minute"
gravity_noise_std = 10.0
magnetic_resolution = "two_minutes"
magnetic_noise_std = 200.0
geo_frequency_s = 1.0

[health_limits]
nis_pos_max = 1000.0
speed_mps_max = 5000.0

[gnss_degradation]
seed = 42

[gnss_degradation.scheduler]
kind = "fixed_interval"
interval_s = 5.0
phase_s = 0.0

[gnss_degradation.fault]
kind = "degraded"
rho_pos = 0.99
sigma_pos_m = 3.0
rho_vel = 0.95
sigma_vel_mps = 0.3
r_scale = 5.0
"#;
        let config = config_from_toml(toml);
        let geo = config
            .geophysical
            .as_ref()
            .expect("the fixture declares a [geophysical] section");
        let from_config = geo_settings_from_config(&config, geo);

        // `generate_plot` is the one field the two are meant to disagree on: a config file
        // asks for a plot and the command line has no equivalent flag on this path.
        let mut from_config_comparable = from_config;
        from_config_comparable.generate_plot = from_cli.generate_plot;

        assert_eq!(
            format!("{from_cli:?}"),
            format!("{from_config_comparable:?}"),
            "the CLI and config paths must describe the same run"
        );
    }

    /// A geophysical run with no map is refused before any file is opened.
    #[cfg(feature = "geonav")]
    #[test]
    fn geophysical_settings_require_at_least_one_map() {
        let config = config_from_toml(
            "mode = \"closed-loop\"\n\n[closed_loop]\nfilter = \"ukf\"\n\n[geophysical]\n",
        );
        let geo = config
            .geophysical
            .as_ref()
            .expect("a [geophysical] section");
        let settings = geo_settings_from_config(&config, geo);

        let error = settings
            .validate()
            .expect_err("a geophysical run with no map must be refused");
        assert!(
            error.to_string().contains("at least one map"),
            "the error should name the missing maps, got: {error}"
        );
    }

    /// The geophysical arm builds the unaided filter plus its map biases, and nothing else.
    ///
    /// The variants a study compares -- full GNSS, degraded GNSS, degraded GNSS with map aiding
    /// -- must run one filter, so that the only differences are the GNSS they see and the map
    /// biases the aided run carries. The geophysical arm used to build its own: no barometric
    /// bias on either filter, and a hand-written P0 and Q on the EKF. This builds both filters
    /// the way the unaided `process_file` arm builds them from the same `[closed_loop]` section,
    /// and the way `run_geo_closed_loop_file` builds them from the same file with maps, and
    /// requires every shared state to open identically, the map biases to sit at fifteen and
    /// sixteen, and the barometric bias -- present on both -- to follow them.
    #[cfg(feature = "geonav")]
    #[test]
    fn the_geophysical_arm_builds_the_unaided_filter_plus_map_biases() {
        let config = config_from_toml(
            r#"
mode = "closed-loop"
is_enu = true

[closed_loop]
filter = "ekf"

[geophysical]
gravity_resolution = "one_minute"
gravity_bias = 635.0
gravity_noise_std = 139.0
gravity_bias_init_std = 231.0
magnetic_resolution = "two_minutes"
magnetic_bias = 17500.0
magnetic_noise_std = 7800.0
magnetic_bias_init_std = 32000.0
"#,
        );
        let geo = config
            .geophysical
            .as_ref()
            .expect("the fixture declares a [geophysical] section");
        let settings = geo_settings_from_config(&config, geo);
        let unaided =
            KalmanSettings::from_closed_loop(&config.closed_loop.clone().unwrap_or_default(), true);
        assert_eq!(
            settings.kalman, unaided,
            "the geophysical arm reads the same filter settings as the unaided one"
        );
        assert!(
            unaided.estimate_baro_bias,
            "the barometric bias is on by default, on both arms"
        );

        let geo_bias = geo_bias_setup(
            Some(settings.gravity_bias_prior()),
            Some(settings.magnetic_bias_prior()),
        );
        let layout = kalman_geo_bias_layout(true, true, true)
            .expect("a two-map layout with a barometric bias must be valid")
            .expect("two maps yield a layout");
        assert_eq!(layout.gravity_bias().map(|bias| bias.index), Some(15));
        assert_eq!(layout.magnetic_bias().map(|bias| bias.index), Some(16));
        assert_eq!(
            layout.state_dim(),
            18,
            "the layout spans the barometric bias"
        );

        let record = TestDataRecord {
            time: chrono::Utc::now(),
            latitude: 40.0,
            longitude: -75.0,
            altitude: 100.0,
            horizontal_accuracy: 5.0,
            vertical_accuracy: 3.0,
            speed_accuracy: 0.5,
            speed: 10.0,
            bearing: 45.0,
            ..Default::default()
        };

        let mut aided_ukf_config = unaided.ukf_config();
        geo_bias.apply_to_ukf(&mut aided_ukf_config);
        assert_eq!(aided_ukf_config.baro_bias_index(), Some(17));
        let plain_ukf = initialize_ukf(&record, unaided.ukf_config()).unwrap();
        let aided_ukf = initialize_ukf(&record, aided_ukf_config).unwrap();
        assert_aided_matches_unaided(
            "UKF",
            (&plain_ukf.get_estimate(), &plain_ukf.get_certainty()),
            (&aided_ukf.get_estimate(), &aided_ukf.get_certainty()),
            &geo_bias,
        );
        assert_eq!(aided_ukf.baro_bias_index(), Some(17));

        let mut aided_ekf_config = unaided.ekf_config();
        geo_bias.apply_to_ekf(&mut aided_ekf_config);
        assert_eq!(aided_ekf_config.baro_bias_index(), Some(17));
        let plain_ekf = initialize_ekf(&record, unaided.ekf_config()).unwrap();
        let aided_ekf = initialize_ekf(&record, aided_ekf_config).unwrap();
        assert_aided_matches_unaided(
            "EKF",
            (&plain_ekf.get_estimate(), &plain_ekf.get_certainty()),
            (&aided_ekf.get_estimate(), &aided_ekf.get_certainty()),
            &geo_bias,
        );
        assert_eq!(aided_ekf.baro_bias_index(), Some(17));
    }

    /// The aided filter opens as the unaided one on every shared state, with the map biases
    /// seeded at fifteen and sixteen and the barometric bias last on both.
    #[cfg(feature = "geonav")]
    fn assert_aided_matches_unaided(
        filter: &str,
        (plain_mean, plain_cov): (&nalgebra::DVector<f64>, &nalgebra::DMatrix<f64>),
        (aided_mean, aided_cov): (&nalgebra::DVector<f64>, &nalgebra::DMatrix<f64>),
        geo_bias: &GeoBiasSetup,
    ) {
        assert_eq!(
            plain_mean.len(),
            16,
            "{filter}: fifteen states plus the barometric bias"
        );
        assert_eq!(
            aided_mean.len(),
            18,
            "{filter}: and two map biases before it"
        );
        for i in 0..15 {
            assert_eq!(aided_mean[i], plain_mean[i], "{filter}: state {i}");
            assert_eq!(
                aided_cov[(i, i)],
                plain_cov[(i, i)],
                "{filter}: variance {i}"
            );
        }
        for (k, (seed, variance)) in geo_bias.seeds.iter().zip(&geo_bias.variances).enumerate() {
            assert_eq!(aided_mean[15 + k], *seed, "{filter}: map bias {k} seed");
            assert_eq!(
                aided_cov[(15 + k, 15 + k)],
                *variance,
                "{filter}: map bias {k} prior"
            );
        }
        assert_eq!(
            aided_mean[17], plain_mean[15],
            "{filter}: barometric bias seed"
        );
        assert_eq!(
            aided_cov[(17, 17)],
            plain_cov[(15, 15)],
            "{filter}: barometric bias prior"
        );
    }

    /// The particle filter takes its map-bias prior from `[geophysical]`, as the Kalman arm does.
    ///
    /// Every RBPF recipe under `conf/` carried the bias, noise and prior `analyze geostats`
    /// measured, and the particle filter read none of it: each bias started at zero with a
    /// prior of `[particle_filter] geo_bias_init_std = 1.0`, one number for mGal and nT alike.
    /// So the gravity bias -- hundreds of mGal -- could not be absorbed, and was cancelled by
    /// selecting particles on velocity instead. This pins both halves of the fix: the RBPF's
    /// extra states come out seeded and scaled per channel, and they are the same numbers the
    /// Kalman arm resolves from the same section.
    #[cfg(feature = "geonav")]
    #[test]
    fn the_particle_filter_takes_its_map_bias_prior_from_geophysical() {
        let config = config_from_toml(
            r#"
mode = "particle-filter"

[particle_filter]
num_particles = 16

[geophysical]
gravity_resolution = "one_minute"
gravity_bias = 635.283
gravity_noise_std = 138.928
gravity_bias_init_std = 230.598
magnetic_resolution = "two_minutes"
magnetic_bias = 17535.6
magnetic_noise_std = 7846.51
magnetic_bias_init_std = 32422.5
magnetic_bias_process_noise_std = 3.0
"#,
        );
        let geo = config
            .geophysical
            .as_ref()
            .expect("the fixture declares a [geophysical] section");

        let particle = geo_bias_setup(
            Some(MapBiasPrior::gravity_from_config(geo)),
            Some(MapBiasPrior::magnetic_from_config(geo)),
        );
        let mut rbpf_config = RbpfConfig::default();
        rbpf_config.extra_state_dim = 2;
        particle.apply_to_rbpf(&mut rbpf_config);

        assert_eq!(rbpf_config.extra_state_initial, vec![635.283, 17535.6]);
        assert_eq!(rbpf_config.extra_state_init_std, vec![230.598, 32422.5]);
        // Gravity sets no rate, so it takes its prior spread over the hour; magnetic sets one.
        let gravity_rate = 230.598 / GEO_BIAS_DRIFT_TIME_CONSTANT_S.sqrt();
        assert!((rbpf_config.extra_state_process_noise_std[0] - gravity_rate).abs() < 1e-12);
        assert!((rbpf_config.extra_state_process_noise_std[1] - 3.0).abs() < 1e-12);
        RaoBlackwellizedParticleFilter::new(strapdown::StrapdownState::default(), rbpf_config)
            .expect("the resolved priors must build a filter");

        // The Kalman arm, reading the same section through its own settings, must resolve the
        // identical priors: the two families are compared against each other, so they must
        // be told the same thing about the sensor.
        let settings = geo_settings_from_config(&config, geo);
        let kalman = geo_bias_setup(
            Some(settings.gravity_bias_prior()),
            Some(settings.magnetic_bias_prior()),
        );
        assert_eq!(
            format!("{kalman:?}"),
            format!("{particle:?}"),
            "the Kalman arm and the particle filter resolved different map-bias priors from \
             the same [geophysical] section"
        );
    }

    /// `pf`'s flags and a config file resolve to the same map-bias priors.
    ///
    /// `pf` flattened `--gravity-bias` in with the other geophysical flags and then ignored it,
    /// along with every `--*-bias-init-std`; this holds it to the config path it now shares.
    #[cfg(feature = "geonav")]
    #[test]
    fn the_pf_flags_and_a_config_file_resolve_the_same_map_bias_priors() {
        let cli = Cli::try_parse_from([
            "strapdown-sim",
            "pf",
            "-i",
            "in.csv",
            "-o",
            "out.csv",
            "--geo",
            "--gravity-resolution",
            "one-minute",
            "--gravity-noise-std",
            "138.928",
            "--gravity-bias",
            "635.283",
            "--gravity-bias-init-std",
            "230.598",
            "--magnetic-resolution",
            "two-minutes",
            "--magnetic-noise-std",
            "7846.51",
            "--magnetic-bias",
            "17535.6",
            "--magnetic-bias-process-noise-std",
            "3.0",
        ])
        .expect("the flags above must parse");
        let Some(Command::ParticleFilter(args)) = cli.command else {
            panic!("expected the `pf` subcommand");
        };
        let from_cli = geo_bias_setup(
            Some(MapBiasPrior::gravity_from_args(&args.geo, &args.geo_bias)),
            Some(MapBiasPrior::magnetic_from_args(&args.geo, &args.geo_bias)),
        );

        let config = config_from_toml(
            r#"
mode = "particle-filter"

[geophysical]
gravity_resolution = "one_minute"
gravity_noise_std = 138.928
gravity_bias = 635.283
gravity_bias_init_std = 230.598
magnetic_resolution = "two_minutes"
magnetic_noise_std = 7846.51
magnetic_bias = 17535.6
magnetic_bias_process_noise_std = 3.0
"#,
        );
        let geo = config
            .geophysical
            .as_ref()
            .expect("the fixture declares a [geophysical] section");
        let from_config = geo_bias_setup(
            Some(MapBiasPrior::gravity_from_config(geo)),
            Some(MapBiasPrior::magnetic_from_config(geo)),
        );

        assert_eq!(format!("{from_cli:?}"), format!("{from_config:?}"));
        assert_eq!(from_cli.seeds, vec![635.283, 17535.6]);
    }

    /// `pf`'s two retired map-bias flags are refused, naming what replaced them.
    #[test]
    fn the_particle_filters_retired_flags_are_refused_by_name() {
        for (flag, replacement) in [
            ("--geo-bias-init-std", "--gravity-bias-init-std"),
            (
                "--geo-bias-process-noise-std",
                "--gravity-bias-process-noise-std",
            ),
        ] {
            let cli = Cli::try_parse_from([
                "strapdown-sim",
                "pf",
                "-i",
                "in.csv",
                "-o",
                "out.csv",
                flag,
                "1.0",
            ])
            .expect("the retired flag must still parse, so it can be refused by name");
            let Some(Command::ParticleFilter(args)) = cli.command else {
                panic!("expected the `pf` subcommand");
            };
            let error = refuse_removed_particle_filter_flags(&args)
                .expect_err("a retired flag must be refused")
                .to_string();
            assert!(
                error.contains(flag) && error.contains(replacement),
                "the error must name `{flag}` and point at `{replacement}`, got: {error}"
            );
        }
    }

    /// The ESKF has no geophysical implementation, and it is `FilterType`'s default.
    ///
    /// So a config file that declares `[geophysical]` and omits `filter` reaches this -- which
    /// is why the check runs before any map is loaded rather than at the filter match.
    #[cfg(feature = "geonav")]
    #[test]
    fn a_geophysical_config_that_omits_its_filter_is_refused_for_the_eskf() {
        let config = config_from_toml(
            "mode = \"closed-loop\"\n\n[geophysical]\ngravity_resolution = \"one_minute\"\n",
        );
        let geo = config
            .geophysical
            .as_ref()
            .expect("a [geophysical] section");
        let settings = geo_settings_from_config(&config, geo);

        assert!(
            matches!(settings.filter, FilterType::Eskf),
            "omitting `filter` should fall back to the ESKF default"
        );
        let error = settings
            .validate()
            .expect_err("the ESKF has no geophysical implementation");
        let message = error.to_string();
        assert!(message.contains("ESKF"), "got: {message}");
        assert!(
            message.contains("default"),
            "the error should say that omitting `filter` lands here, got: {message}"
        );
    }
}
