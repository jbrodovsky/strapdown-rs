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
use strapdown::messages::{Event, EventStream, GnssScheduler, build_event_stream};
use strapdown::rbpf::{RaoBlackwellizedParticleFilter, RbpfConfig};

// Geophysical navigation imports (feature-gated)
#[cfg(feature = "geonav")]
use geonav::{
    GeoBiasLayout, GeoMap, GeophysicalMeasurementType, GravityResolution, MagneticResolution,
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
#[cfg(feature = "geonav")]
use strapdown::kalman::ExtendedKalmanFilter;
use strapdown::sim::HealthLimits;
use strapdown::sim::health::HealthMonitor;
#[cfg(feature = "geonav")]
use strapdown::sim::run_closed_loop_with_geo;
#[cfg(feature = "geonav")]
use strapdown::sim::{
    DEFAULT_INITIAL_POSITION_UNCERTAINTY_M, DEFAULT_PROCESS_NOISE, GeoResolution,
};
use strapdown::sim::{
    EkfConfig, EskfConfig, ExecutionLimits, ExecutionMonitor, FaultArgs, FilterType,
    GeoStateLayout, NavigationResult, ParticleFilterType, SchedulerArgs, SimulationConfig,
    SimulationMode, SyntheticConfig, TestDataRecord, UkfConfig, build_fault, build_scheduler,
    check_declared_frame, dead_reckoning, generate_synthetic, initialize_ekf, initialize_eskf,
    initialize_ukf, run_closed_loop,
};

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
    #[arg(long, default_value = "info", global = true)]
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

    /// Geophysical measurement frequency (seconds)
    #[arg(long, requires = "geo")]
    geo_frequency_s: Option<f64>,
}

/// Empty stub when geonav feature is disabled
#[cfg(not(feature = "geonav"))]
#[derive(Args, Clone, Debug, Default)]
struct GeophysicalArgs {}

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
    #[arg(long, default_value_t = 1e-3)]
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

    /// Accelerometer bias uncertainty standard deviation (m/s²)
    #[arg(long, default_value_t = 0.1)]
    accel_bias_std: f64,

    /// Gyroscope bias uncertainty standard deviation (rad/s)
    #[arg(long, default_value_t = 0.01)]
    gyro_bias_std: f64,

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

    /// Velocity process noise standard deviation (m/s).
    #[arg(long, default_value_t = 1e-3)]
    velocity_process_noise_std_mps: f64,

    /// Attitude process noise standard deviation (rad).
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

    /// Apply zero-vertical-velocity pseudo-measurement (RBPF only).
    #[arg(long, default_value_t = true)]
    zero_vertical_velocity: bool,

    /// Std dev for zero-vertical-velocity pseudo-measurement (m/s).
    #[arg(long, default_value_t = 0.1)]
    zero_vertical_velocity_std_mps: f64,

    /// Initial standard deviation for geophysical bias states.
    #[arg(long, default_value_t = 1.0)]
    geo_bias_init_std: f64,

    /// Random-walk process noise standard deviation for geophysical bias states.
    #[arg(long, default_value_t = 1e-3)]
    geo_bias_process_noise_std: f64,
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

            let filter_config = config.closed_loop.clone().unwrap_or_default();

            let event_stream =
                build_event_stream(&records, &config.gnss_degradation, config.is_enu)?;
            info!(
                "Initialized event stream with {} events",
                event_stream.events.len()
            );
            let execution_limits = config.execution_limits.clone();

            let results = match filter_config.filter {
                FilterType::Ukf => {
                    let mut ukf = initialize_ukf(
                        &records[0].clone(),
                        UkfConfig {
                            is_enu: config.is_enu,
                            ..UkfConfig::default()
                        },
                    )?;
                    info!("Initialized UKF");
                    ukf.set_innovation_gate(filter_config.innovation_gate);
                    ukf.set_gate_recovery(filter_config.gate_recovery);
                    run_closed_loop(&mut ukf, event_stream, None, Some(execution_limits))
                }
                FilterType::Ekf => {
                    let mut ekf = initialize_ekf(
                        &records[0].clone(),
                        EkfConfig {
                            is_enu: config.is_enu,
                            ..EkfConfig::default()
                        },
                    )?;
                    info!("Initialized EKF");
                    ekf.set_innovation_gate(filter_config.innovation_gate);
                    ekf.set_gate_recovery(filter_config.gate_recovery);
                    run_closed_loop(&mut ekf, event_stream, None, Some(execution_limits))
                }
                FilterType::Eskf => {
                    let mut eskf = initialize_eskf(
                        &records[0].clone(),
                        EskfConfig {
                            is_enu: config.is_enu,
                            ..EskfConfig::default()
                        },
                    )?;
                    info!("Initialized ESKF");
                    eskf.set_innovation_gate(filter_config.innovation_gate);
                    eskf.set_gate_recovery(filter_config.gate_recovery);
                    run_closed_loop(&mut eskf, event_stream, None, Some(execution_limits))
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
            let (gravity_map, magnetic_map, geo_frequency_s, gravity_noise_std, magnetic_noise_std) = {
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
                        geo_cfg.geo_frequency_s,
                        geo_cfg.gravity_noise_std.unwrap_or(100.0),
                        geo_cfg.magnetic_noise_std.unwrap_or(150.0),
                    )
                } else {
                    (None, None, None, 100.0, 150.0)
                }
            };

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
                    &config.gnss_degradation,
                    gravity_map.clone(),
                    gravity_map.as_ref().map(|_| gravity_noise_std),
                    magnetic_map.clone(),
                    magnetic_map.as_ref().map(|_| magnetic_noise_std),
                    geo_frequency_s,
                    geo_bias_layout,
                )?
            } else {
                build_event_stream(&records, &config.gnss_degradation, config.is_enu)?
            };

            #[cfg(not(feature = "geonav"))]
            let event_stream =
                build_event_stream(&records, &config.gnss_degradation, config.is_enu)?;

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
            let geo_layout = geo_bias_layout.map_or(GeoStateLayout::PARTICLE_NONE, |layout| {
                GeoStateLayout::new(
                    layout.state_dim(),
                    layout.gravity_bias().map(|bias| bias.index),
                    layout.magnetic_bias().map(|bias| bias.index),
                )
            });
            #[cfg(not(feature = "geonav"))]
            let geo_layout = GeoStateLayout::PARTICLE_NONE;
            let mut rbpf = RaoBlackwellizedParticleFilter::new(
                nominal,
                RbpfConfig {
                    num_particles: pf_cfg.num_particles,
                    position_init_std_m,
                    velocity_init_std_mps: pf_cfg.velocity_init_std_mps,
                    attitude_init_std_rad: pf_cfg.attitude_init_std_rad,
                    position_process_noise_std_m,
                    velocity_process_noise_std_mps: pf_cfg.velocity_process_noise_std_mps,
                    attitude_process_noise_std_rad: pf_cfg.attitude_process_noise_std_rad,
                    extra_state_dim: geo_bias_dim,
                    extra_state_init_std: if geo_bias_dim > 0 {
                        pf_cfg.geo_bias_init_std
                    } else {
                        0.0
                    },
                    extra_state_process_noise_std: if geo_bias_dim > 0 {
                        pf_cfg.geo_bias_process_noise_std
                    } else {
                        0.0
                    },
                    seed: config.seed,
                    zero_vertical_velocity: pf_cfg.zero_vertical_velocity,
                    zero_vertical_velocity_std_mps: pf_cfg.zero_vertical_velocity_std_mps,
                    ..rbpf_defaults
                },
            )?;

            // Geophysical measurements ride the same event stream as every other
            // measurement type, so there is no separate geo path here.
            let results = run_rbpf_event_loop(
                &mut rbpf,
                event_stream,
                &config.execution_limits,
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
        }
    }

    Ok(())
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
    gnss_degradation: &strapdown::messages::GnssDegradationConfig,
    output_file: &Path,
    execution_limits: ExecutionLimits,
    ukf_alpha: f64,
    ukf_beta: f64,
    ukf_kappa: f64,
    innovation_gate: Option<InnovationGate>,
    gate_recovery: GateRecovery,
    is_enu: bool,
) -> Result<(), Box<dyn Error>> {
    // Same full-window guard as the other entry points: the `initialize_*` helpers below see
    // only one record, which is not enough evidence in either direction (#296).
    check_declared_frame(records, is_enu)?;

    // Build event stream from records and GNSS degradation config
    let event_stream = build_event_stream(records, gnss_degradation, is_enu)?;
    info!(
        "Initialized event stream with {} events",
        event_stream.events.len()
    );

    // Initialize and run filter based on type
    let results = match filter_type {
        FilterType::Ukf => {
            let mut ukf = initialize_ukf(
                &records[0].clone(),
                UkfConfig {
                    ukf_alpha: Some(ukf_alpha),
                    ukf_beta: Some(ukf_beta),
                    ukf_kappa: Some(ukf_kappa),
                    is_enu,
                    ..Default::default()
                },
            )?;
            info!("Initialized UKF");
            ukf.set_innovation_gate(innovation_gate);
            ukf.set_gate_recovery(gate_recovery);
            run_closed_loop(&mut ukf, event_stream, None, Some(execution_limits))
        }
        FilterType::Ekf => {
            let mut ekf = initialize_ekf(
                &records[0].clone(),
                EkfConfig {
                    is_enu,
                    ..EkfConfig::default()
                },
            )?;
            info!("Initialized EKF");
            ekf.set_innovation_gate(innovation_gate);
            ekf.set_gate_recovery(gate_recovery);
            run_closed_loop(&mut ekf, event_stream, None, Some(execution_limits))
        }
        FilterType::Eskf => {
            let mut eskf = initialize_eskf(
                &records[0].clone(),
                EskfConfig {
                    is_enu,
                    ..EskfConfig::default()
                },
            )?;
            info!("Initialized ESKF");
            eskf.set_innovation_gate(innovation_gate);
            eskf.set_gate_recovery(gate_recovery);
            run_closed_loop(&mut eskf, event_stream, None, Some(execution_limits))
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

    let config = SyntheticConfig {
        output: args.output.to_string_lossy().into_owned(),
        initial_state: SyntheticInitialState {
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
        },
        duration_s: args.duration_s,
        sample_rate_hz: args.sample_rate_hz,
        imu_quality: args.imu_grade,
        seed: args.seed,
        no_noise: args.no_noise,
        gnss_horizontal_noise_m: args.gnss_horizontal_noise_m,
        gnss_vertical_noise_m: args.gnss_vertical_noise_m,
        baro_noise_std_pa: args.baro_noise_std_pa,
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
    let execution_limits = execution_limits_from_args(&args.sim);

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

        // Build GNSS degradation config from CLI args
        let gnss_degradation = strapdown::messages::GnssDegradationConfig {
            scheduler: build_scheduler(&args.scheduler),
            fault: build_fault(&args.fault),
            seed: args.seed,
        };

        info!("Using GNSS degradation config: {gnss_degradation:?}");
        let output_file = resolve_output_path(&args.sim.output, input_file, &csv_files)?;

        // Run simulation using the common helper function
        match run_single_closed_loop_simulation(
            args.filter,
            &records,
            &gnss_degradation,
            &output_file,
            execution_limits.clone(),
            args.ukf_alpha,
            args.ukf_beta,
            args.ukf_kappa,
            innovation_gate,
            gate_recovery,
            args.sim.enu,
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

/// Execute geophysical closed-loop simulation
#[cfg(feature = "geonav")]
fn run_geo_closed_loop_cli(args: &ClosedLoopSimArgs) -> Result<(), Box<dyn Error>> {
    validate_input_path(&args.sim.input)?;
    validate_output_path(&args.sim.output)?;

    let filter_name = match args.filter {
        FilterType::Ukf => "Unscented Kalman Filter (UKF)",
        FilterType::Ekf => "Extended Kalman Filter (EKF)",
        FilterType::Eskf => "Error-State Kalman Filter (ESKF)",
    };
    info!("Running geophysical navigation in closed-loop mode with {filter_name}");

    // Validate that at least one geophysical map is configured
    if args.geo.gravity_resolution.is_none() && args.geo.magnetic_resolution.is_none() {
        return Err("At least one of --gravity-resolution or --magnetic-resolution must be specified when using --geo".into());
    }

    // Gating applies here exactly as it does to a non-geophysical run: geophysical anomalies
    // ride the same event stream and are scored by the same test.
    let (innovation_gate, gate_recovery) = gating_from_args(args)?;

    // Get all CSV files to process
    let csv_files = get_csv_files(&args.sim.input)?;
    let is_multiple = csv_files.len() > 1;
    // NOTE: Execution limits are not yet applied in geophysical closed-loop simulations.
    // We still parse the arguments here to validate them and keep CLI behavior consistent.
    let _ = execution_limits_from_args(&args.sim);

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

        // Load gravity map if configured
        let gravity_map = if let Some(res) = args.geo.gravity_resolution {
            let map_path = match &args.geo.gravity_map_file {
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
        let magnetic_map = if let Some(res) = args.geo.magnetic_resolution {
            let map_path = match &args.geo.magnetic_map_file {
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

        // Build GNSS degradation config from CLI args
        let gnss_degradation = strapdown::messages::GnssDegradationConfig {
            scheduler: build_scheduler(&args.scheduler),
            fault: build_fault(&args.fault),
            seed: args.seed,
        };

        // This path runs a UKF or an EKF, whose states are the nine navigation states, the
        // six IMU biases, and then the map biases -- the UKF via `other_states`, the EKF via
        // a covariance diagonal longer than its mean, which its constructor zero-pads to
        // match. So the base here is 15, not the RBPF's 9.
        let geo_bias_layout = GeoBiasLayout::appended(
            NAVIGATION_AND_IMU_BIAS_STATE_DIM,
            gravity_map.is_some(),
            magnetic_map.is_some(),
        )?;

        // Build event stream with geophysical measurements
        let events = geo_build_event_stream(
            &records,
            &gnss_degradation,
            gravity_map.clone(),
            if gravity_map.is_some() {
                Some(args.geo.gravity_noise_std)
            } else {
                None
            },
            magnetic_map.clone(),
            if magnetic_map.is_some() {
                Some(args.geo.magnetic_noise_std)
            } else {
                None
            },
            args.geo.geo_frequency_s,
            geo_bias_layout,
        )?;
        info!("Built event stream with {} events", events.events.len());

        // Determine number of geophysical states
        let num_geo_states = geo_bias_layout.map_or(0, |layout| layout.bias_count());

        // The same placement, restated for `NavigationResult`, which lives in `core` and so
        // cannot name `GeoBiasLayout`. Derived from that layout rather than rebuilt from the
        // map flags, so where the biases live is decided once: a filter that put them
        // somewhere other than the end would move both together.
        let geo_layout = geo_bias_layout.map_or(GeoStateLayout::NONE, |layout| {
            GeoStateLayout::new(
                layout.state_dim(),
                layout.gravity_bias().map(|bias| bias.index),
                layout.magnetic_bias().map(|bias| bias.index),
            )
        });

        // Run simulation based on filter type
        let results = match args.filter {
            FilterType::Ukf => {
                info!("Initializing UKF...");
                let mut process_noise: Vec<f64> = DEFAULT_PROCESS_NOISE.into();
                process_noise.extend(vec![1e-9; num_geo_states]);

                let mut geo_biases = Vec::new();
                let mut geo_noise_stds = Vec::new();

                if gravity_map.is_some() {
                    geo_biases.push(args.geo.gravity_bias.unwrap_or(0.0));
                    geo_noise_stds.push(args.geo.gravity_noise_std);
                }
                if magnetic_map.is_some() {
                    geo_biases.push(args.geo.magnetic_bias.unwrap_or(0.0));
                    geo_noise_stds.push(args.geo.magnetic_noise_std);
                }

                let mut ukf = initialize_ukf(
                    &records[0].clone(),
                    UkfConfig {
                        attitude_covariance: None,
                        imu_biases: None,
                        imu_biases_covariance: None,
                        other_states: Some(geo_biases),
                        other_states_covariance: Some(geo_noise_stds),
                        process_noise_diagonal: Some(process_noise),
                        ukf_alpha: Some(args.ukf_alpha),
                        ukf_beta: Some(args.ukf_beta),
                        ukf_kappa: Some(args.ukf_kappa),
                        is_enu: args.sim.enu,
                    },
                )?;
                info!(
                    "Initialized UKF with state dimension {} (base: 9, geo: {})",
                    ukf.get_estimate().len(),
                    num_geo_states
                );
                ukf.set_innovation_gate(innovation_gate);
                ukf.set_gate_recovery(gate_recovery);

                info!("Running UKF geophysical navigation simulation...");
                run_closed_loop_with_geo(&mut ukf, events, None, None, geo_layout)
            }
            FilterType::Ekf => {
                info!("Initializing EKF...");

                check_declared_frame(&records, args.sim.enu)?;
                // The same seed every other path builds. It was a struct literal here, and
                // carried the double conversion that gave this block its share of #337:
                // `yaw: bearing.to_radians()` beside `in_degrees: true`, converted once here
                // and a second time by the constructor, so a 270 deg bearing reached the
                // filter as 0.0822 rad (4.7 deg). It also threw away roll and pitch. The
                // guard is still run explicitly above because this path does not go through
                // `initialize_ekf` (#296).
                let initial_state = records[0].initial_state(args.sim.enu);

                let imu_biases = vec![0.0; 6];

                // Initial position uncertainty. Only the *horizontal* pair changes: latitude
                // and longitude are radians here and altitude is metres, and the `1e-6, 1e-6`
                // this used to carry was #308 in P0 -- 1e-6 rad^2 is a 6367 m claim, not the
                // 1e-3 m it reads as. The altitude entry stays at its own 1.0 m^2: it was
                // already metres-squared, it was never a units defect, and moving it to the
                // crate default's 100 m^2 would be a silent 10x retune of the vertical channel
                // folded into a units fix -- the same thing `VERTICAL_POSITION_PROCESS_NOISE_M2`
                // exists to prevent in `DEFAULT_PROCESS_NOISE`. Everything below the position
                // block is this path's own and deliberately unchanged.
                let horizontal_std_rad =
                    DEFAULT_INITIAL_POSITION_UNCERTAINTY_M * strapdown::earth::METERS_TO_RADIANS;
                let mut covariance_diagonal = vec![
                    horizontal_std_rad.powi(2),
                    horizontal_std_rad.powi(2),
                    1.0, // Position uncertainty (altitude, m^2 -- unchanged, see above)
                    0.1,
                    0.1,
                    0.1, // Velocity uncertainty
                    1e-4,
                    1e-4,
                    1e-4, // Attitude uncertainty
                    1e-6,
                    1e-6,
                    1e-6, // Accel bias uncertainty
                    1e-8,
                    1e-8,
                    1e-8, // Gyro bias uncertainty
                ];
                covariance_diagonal.extend(vec![1.0; num_geo_states]);

                // Position process noise. Again only the horizontal pair: `1e-9, 1e-9` rad^2
                // is a 201 m per-step standard deviation, the same units defect as #308 one
                // third of a magnitude smaller, so those come from the crate default. The
                // `1e-6` altitude entry was already m^2 and stays exactly where it was --
                // taking `DEFAULT_PROCESS_NOISE[0..3]` wholesale would move it to 1e-2, a
                // 10,000x variance retune of the vertical channel that no test here covers
                // (`run_geo_closed_loop_cli` has no test at all). The crate default's altitude
                // entry was retuned from 1e-4 to 1e-2 on measured evidence; this path was not
                // part of that measurement, so it keeps its own value until it has tests that
                // could see the difference. The entries below the position block are
                // deliberately tighter than the crate default.
                let mut process_noise_vec = DEFAULT_PROCESS_NOISE[0..2].to_vec();
                process_noise_vec.extend([
                    1e-6, // Altitude process noise, m^2 -- unchanged, see above
                    1e-6, 1e-6, 1e-6, // Velocity process noise
                    1e-9, 1e-9, 1e-9, // Attitude process noise
                    1e-9, 1e-9, 1e-9, // Accel bias process noise
                    1e-9, 1e-9, 1e-9, // Gyro bias process noise
                ]);
                process_noise_vec.extend(vec![1e-9; num_geo_states]);
                let process_noise = nalgebra::DMatrix::from_diagonal(&nalgebra::DVector::from_vec(
                    process_noise_vec,
                ));

                let mut ekf = ExtendedKalmanFilter::new(
                    &initial_state,
                    &imu_biases,
                    covariance_diagonal,
                    process_noise,
                    true,
                );

                info!(
                    "Initialized EKF with state dimension {} (base: 15, geo: {})",
                    ekf.get_estimate().len(),
                    num_geo_states
                );
                ekf.set_innovation_gate(innovation_gate);
                ekf.set_gate_recovery(gate_recovery);

                info!("Running EKF geophysical navigation simulation...");
                run_closed_loop_with_geo(&mut ekf, events, None, None, geo_layout)
            }
            FilterType::Eskf => {
                error!("ESKF is not yet implemented for geophysical navigation");
                return Err("ESKF is not yet implemented for geophysical navigation".into());
            }
        };

        // Write results
        let output_file = resolve_output_path(&args.sim.output, input_file, &csv_files)?;

        match results {
            Ok(ref nav_results) => {
                NavigationResult::to_csv(nav_results, &output_file)?;
                info!("Results written to {}", output_file.display());
            }
            Err(e) => {
                error!(
                    "Error running geophysical navigation on {}: {}",
                    input_file.display(),
                    e
                );
                if !is_multiple {
                    return Err(e.into());
                }
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
fn run_rbpf_event_loop(
    rbpf: &mut RaoBlackwellizedParticleFilter,
    event_stream: EventStream,
    execution_limits: &ExecutionLimits,
    geo_layout: GeoStateLayout,
) -> Result<Vec<NavigationResult>, Box<dyn Error>> {
    let start_time = event_stream.start_time;
    let mut results = Vec::with_capacity(event_stream.events.len());
    let mut monitor = HealthMonitor::new(HealthLimits::default());
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

    for event in event_stream.events {
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
            let (mean, cov) = rbpf.estimate_with_extra_states();
            results.push(NavigationResult::from_particle_filter_with_geo(
                &last_ts, &mean, &cov, geo_layout,
            ));
            last_ts = ts;
        }

        match event {
            Event::Imu { dt_s, imu, .. } => {
                rbpf.predict(&imu, dt_s)?;
            }
            Event::Measurement { meas, .. } => {
                rbpf.update(meas.as_ref())?;
            }
        }

        // The health monitor sees the geophysical states too: it reads position and velocity
        // by index from the front and then sweeps the covariance diagonal, so a diverging bias
        // variance is caught here the same way it already is on the Kalman paths, which hand
        // `run_closed_loop` the full augmented state.
        let (mean, cov) = rbpf.estimate_with_extra_states();
        if let Err(e) = monitor.check(mean.as_slice(), &cov, None) {
            return Err(e.into());
        }
        execution_monitor.check("particle-filter")?;
        execution_monitor.mark_progress();
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
    validate_input_path(&args.sim.input)?;
    validate_output_path(&args.sim.output)?;

    let csv_files = get_csv_files(&args.sim.input)?;
    let is_multiple = csv_files.len() > 1;
    let execution_limits = execution_limits_from_args(&args.sim);

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

        let gnss_degradation = strapdown::messages::GnssDegradationConfig {
            scheduler: build_scheduler(&args.scheduler),
            fault: build_fault(&args.fault),
            seed: args.seed,
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

        #[cfg(not(feature = "geonav"))]
        let event_stream = build_event_stream(&records, &gnss_degradation, args.sim.enu)?;

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
                &gnss_degradation,
                gravity_map.clone(),
                gravity_map.as_ref().map(|_| args.geo.gravity_noise_std),
                magnetic_map.clone(),
                magnetic_map.as_ref().map(|_| args.geo.magnetic_noise_std),
                args.geo.geo_frequency_s,
                geo_bias_layout,
            )?
        } else {
            build_event_stream(&records, &gnss_degradation, args.sim.enu)?
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
        let geo_layout = geo_bias_layout.map_or(GeoStateLayout::PARTICLE_NONE, |layout| {
            GeoStateLayout::new(
                layout.state_dim(),
                layout.gravity_bias().map(|bias| bias.index),
                layout.magnetic_bias().map(|bias| bias.index),
            )
        });
        #[cfg(not(feature = "geonav"))]
        let geo_layout = GeoStateLayout::PARTICLE_NONE;

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

        let config = RbpfConfig {
            num_particles: args.num_particles,
            position_init_std_m: Vector3::new(
                args.position_std,
                args.position_std,
                args.position_std,
            ),
            velocity_init_std_mps: args.velocity_std,
            attitude_init_std_rad: args.attitude_std,
            position_process_noise_std_m: process_noise_std_m,
            velocity_process_noise_std_mps: args.velocity_process_noise_std_mps,
            attitude_process_noise_std_rad: args.attitude_process_noise_std_rad,
            extra_state_dim: geo_bias_dim,
            extra_state_init_std: if geo_bias_dim > 0 {
                args.geo_bias_init_std
            } else {
                0.0
            },
            extra_state_process_noise_std: if geo_bias_dim > 0 {
                args.geo_bias_process_noise_std
            } else {
                0.0
            },
            seed: args.seed,
            zero_vertical_velocity: args.zero_vertical_velocity,
            zero_vertical_velocity_std_mps: args.zero_vertical_velocity_std_mps,
            ..RbpfConfig::default()
        };

        // `ParticleFilterType` has a single variant today (#259 removed the two that were
        // advertised but never implemented). Dispatching on it anyway keeps adding a second
        // concrete filter a matter of extending this match rather than rediscovering that
        // the flag was never read.
        let results = match args.filter_type {
            ParticleFilterType::RaoBlackwellized => {
                let mut rbpf = RaoBlackwellizedParticleFilter::new(nominal, config)?;
                run_rbpf_event_loop(&mut rbpf, event_stream, &execution_limits, geo_layout)?
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
fn prompt_gnss_scheduler() -> GnssScheduler {
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
                "1" => return GnssScheduler::PassThrough,
                "2" => return prompt_fixed_interval_scheduler(),
                "3" => return prompt_duty_cycle_scheduler(),
                _ => println!("Error: Invalid selection. Please enter 1, 2, 3, or q.\n"),
            }
        }
    }
}

/// Prompt for Fixed Interval scheduler parameters
fn prompt_fixed_interval_scheduler() -> GnssScheduler {
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

    GnssScheduler::FixedInterval {
        interval_s,
        phase_s,
    }
}

/// Prompt for Duty Cycle scheduler parameters
fn prompt_duty_cycle_scheduler() -> GnssScheduler {
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

    GnssScheduler::DutyCycle {
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
fn prompt_geo_measurement_frequency() -> Option<f64> {
    println!("\nGeophysical measurement frequency (seconds) [auto]: ");

    match read_user_input() {
        Some(input) if !input.is_empty() => match input.parse::<f64>() {
            Ok(freq) if freq > 0.0 => Some(freq),
            Ok(_) => {
                println!("Frequency must be positive. Using auto.");
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
        let closed_loop_cfg = strapdown::sim::ClosedLoopConfig {
            filter,
            ..Default::default()
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

    // GNSS degradation configuration
    let scheduler = prompt_gnss_scheduler();
    let fault = prompt_gnss_fault_model();

    let gnss_degradation = strapdown::messages::GnssDegradationConfig {
        scheduler,
        fault,
        seed,
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
            let geo_frequency_s = prompt_geo_measurement_frequency();

            let (gravity_resolution, gravity_bias, gravity_noise_std, gravity_map_file) =
                gravity_config.map_or((None, None, None, None), |(res, bias, noise, map)| {
                    (Some(res), bias, noise, map)
                });

            let (magnetic_resolution, magnetic_bias, magnetic_noise_std, magnetic_map_file) =
                magnetic_config.map_or((None, None, None, None), |(res, bias, noise, map)| {
                    (Some(res), bias, noise, map)
                });

            Some(strapdown::sim::GeophysicalConfig {
                gravity_resolution,
                gravity_bias,
                gravity_noise_std,
                gravity_map_file,
                magnetic_resolution,
                magnetic_bias,
                magnetic_noise_std,
                magnetic_map_file,
                geo_frequency_s,
            })
        }
    } else {
        None
    };

    // Build the complete configuration
    let config = SimulationConfig {
        input: input_path,
        output: output_path,
        mode,
        seed,
        is_enu,
        parallel,
        generate_plot: false,
        execution_limits,
        logging,
        closed_loop,
        particle_filter,
        geophysical,
        gnss_degradation,
        synthetic: None,
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

        // Determine log level: CLI flag takes precedence over config
        // Check if CLI log level was explicitly set (not just the default)
        let log_level = config.logging.level.as_str();

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
    fn test_gnss_scheduler_variants() {
        let passthrough = GnssScheduler::PassThrough;
        let fixed = GnssScheduler::FixedInterval {
            interval_s: 1.0,
            phase_s: 0.0,
        };
        let duty = GnssScheduler::DutyCycle {
            on_s: 10.0,
            off_s: 10.0,
            start_phase_s: 0.0,
        };

        assert!(matches!(passthrough, GnssScheduler::PassThrough));
        assert!(matches!(fixed, GnssScheduler::FixedInterval { .. }));
        assert!(matches!(duty, GnssScheduler::DutyCycle { .. }));
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
}
