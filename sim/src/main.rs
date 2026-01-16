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
    get_csv_files, init_logger, prompt_config_name, prompt_config_path, prompt_f64_with_default,
    prompt_input_path, prompt_output_path, read_user_input, validate_input_path,
    validate_output_path,
};
use log::{error, info};
use nalgebra::{Rotation3, Vector3};
use rayon::prelude::*;
use std::error::Error;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use strapdown::messages::{Event, GnssScheduler, build_event_stream};
use strapdown::rbpf::{RaoBlackwellizedParticleFilter, RbpfConfig};

// Geophysical navigation imports (feature-gated)
#[cfg(feature = "geonav")]
use geonav::{
    GeoMap, GeophysicalAnomalyMeasurementModel, GeophysicalMeasurementType, GravityMeasurement,
    GravityResolution, MagneticAnomalyMeasurement, MagneticResolution,
    build_event_stream as geo_build_event_stream, geo_closed_loop_ekf, geo_closed_loop_ukf,
};
#[cfg(feature = "geonav")]
use std::rc::Rc;
#[cfg(feature = "geonav")]
use strapdown::NavigationFilter;
#[cfg(feature = "geonav")]
use strapdown::kalman::{ExtendedKalmanFilter, InitialState};
#[cfg(feature = "geonav")]
use strapdown::sim::{DEFAULT_PROCESS_NOISE, GeoResolution};
use strapdown::sim::{
    FaultArgs, FilterType, NavigationResult, ParticleFilterType, SchedulerArgs, SimulationConfig,
    SimulationMode, TestDataRecord, build_fault, build_scheduler, dead_reckoning, initialize_ekf,
    initialize_eskf, initialize_ukf, run_closed_loop,
};
use strapdown::sim::health::HealthMonitor;
use strapdown::sim::HealthLimits;

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
    CreateConfig, //(CreateConfigArgs),
}

/// Common simulation arguments for input/output
#[derive(Args, Clone, Debug)]
struct SimArgs {
    /// Input CSV file path or directory containing CSV files
    /// If a directory is provided, all CSV files in it will be processed
    #[arg(short, long, value_parser)]
    input: PathBuf,

    /// Output CSV file path
    /// When processing multiple files, output filenames will be generated as: {output_stem}_{input_stem}.csv
    #[arg(short, long, value_parser)]
    output: PathBuf,
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

    /// Filter type to use for closed-loop navigation
    #[arg(long, value_enum, default_value_t = FilterType::Ukf)]
    filter: FilterType,

    /// RNG seed for stochastic processes
    #[arg(long, default_value_t = 42)]
    seed: u64,

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
    #[arg(long, value_enum, default_value_t = ParticleFilterType::Standard)]
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

    /// Process noise standard deviation for the velocity-based particle filter (meters)
    /// as `[lat_m, lon_m, alt_m]`.
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

/// Process a single CSV file with the given configuration
fn process_file(
    input_file: &Path,
    output: &Path,
    config: &SimulationConfig,
) -> Result<(), Box<dyn Error>> {
    info!("Processing file: {}", input_file.display());

    // Load sensor data
    let records = TestDataRecord::from_csv(input_file)?;
    info!(
        "Read {} records from {}",
        records.len(),
        input_file.display()
    );

    // Execute based on mode
    match config.mode {
        SimulationMode::DeadReckoning => {
            info!("Running dead reckoning simulation");
            let results = dead_reckoning(&records);
            info!("Generated {} navigation results", results.len());

            let output_file = output.join(input_file.file_name().ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!("Input file path '{}' has no filename", input_file.display()),
                )
            })?);
            NavigationResult::to_csv(&results, &output_file)?;
            info!("Results written to {}", output_file.display());
            Ok(())
        }
        SimulationMode::OpenLoop => {
            info!("Open-loop mode is not yet fully implemented");
            Err("Open-loop mode is not yet fully implemented".into())
        }
        SimulationMode::ClosedLoop => {
            let filter_config = config.closed_loop.clone().unwrap_or_default();
            info!(
                "Running closed-loop mode with {:?} filter",
                filter_config.filter
            );

            let event_stream = build_event_stream(&records, &config.gnss_degradation);
            info!(
                "Initialized event stream with {} events",
                event_stream.events.len()
            );

            let results = match filter_config.filter {
                FilterType::Ukf => {
                    let mut ukf =
                        initialize_ukf(records[0].clone(), None, None, None, None, None, None);
                    info!("Initialized UKF");
                    run_closed_loop(&mut ukf, event_stream, None)
                }
                FilterType::Ekf => {
                    let mut ekf = initialize_ekf(records[0].clone(), None, None, None, None, true);
                    info!("Initialized EKF");
                    run_closed_loop(&mut ekf, event_stream, None)
                }
                FilterType::Eskf => {
                    let mut eskf = initialize_eskf(records[0].clone(), None, None, None, None);
                    info!("Initialized ESKF");
                    run_closed_loop(&mut eskf, event_stream, None)
                }
            };

            let output_file = output.join(input_file.file_name().unwrap());
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
                                error!("Failed to generate performance plot: {}", e);
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
                    error!("Error running closed-loop simulation: {}", e);
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
                        Some(Rc::new(GeoMap::load_geomap(map_path, measurement_type)?))
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
                        Some(Rc::new(GeoMap::load_geomap(map_path, measurement_type)?))
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
                )
            } else {
                build_event_stream(&records, &config.gnss_degradation)
            };

            #[cfg(not(feature = "geonav"))]
            let event_stream = build_event_stream(&records, &config.gnss_degradation);

            let first = &records[0];
            let attitude = Rotation3::from_euler_angles(first.roll, first.pitch, first.yaw);
            let nominal = strapdown::StrapdownState {
                latitude: first.latitude.to_radians(),
                longitude: first.longitude.to_radians(),
                altitude: first.altitude,
                velocity_north: first.speed * first.bearing.to_radians().cos(),
                velocity_east: first.speed * first.bearing.to_radians().sin(),
                velocity_vertical: 0.0,
                attitude,
                is_enu: true,
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
            let geo_bias_dim = gravity_map.is_some() as usize + magnetic_map.is_some() as usize;
            #[cfg(not(feature = "geonav"))]
            let geo_bias_dim = 0usize;
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
            );

            let start_time = event_stream.start_time;
            let mut results = Vec::with_capacity(event_stream.events.len());
            let mut last_ts: Option<chrono::DateTime<chrono::Utc>> = None;
            let mut monitor = HealthMonitor::new(HealthLimits::default());

            for event in event_stream.events.into_iter() {
                let elapsed_s = match &event {
                    Event::Imu { elapsed_s, .. } => *elapsed_s,
                    Event::Measurement { elapsed_s, .. } => *elapsed_s,
                };
                let ts = start_time
                    + chrono::Duration::milliseconds((elapsed_s * 1000.0).round() as i64);

                match event {
                    Event::Imu { dt_s, imu, .. } => {
                        rbpf.predict(&imu, dt_s);
                    }
                    Event::Measurement { mut meas, .. } => {
                        #[cfg(feature = "geonav")]
                        if gravity_map.is_some() || magnetic_map.is_some() {
                            let (mean, _) = rbpf.estimate();
                            let strapdown: strapdown::StrapdownState =
                                mean.as_slice().try_into().unwrap();
                            if let Some(gravity) =
                                meas.as_any_mut().downcast_mut::<GravityMeasurement>()
                            {
                                gravity.set_state(&strapdown);
                            } else if let Some(magnetic) =
                                meas.as_any_mut()
                                    .downcast_mut::<MagneticAnomalyMeasurement>()
                            {
                                magnetic.set_state(&strapdown);
                            }
                        }
                        rbpf.update(meas.as_ref());
                    }
                }

                let (mean, cov) = rbpf.estimate();
                if let Err(e) = monitor.check(mean.as_slice(), &cov, None) {
                    return Err(e.into());
                }

                if Some(ts) != last_ts {
                    if let Some(prev_ts) = last_ts {
                        results.push(NavigationResult::from_particle_filter(
                            &prev_ts, &mean, &cov,
                        ));
                    }
                    last_ts = Some(ts);
                }
            }

            let output_file = output.join(input_file.file_name().unwrap());
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
                        error!("Failed to generate performance plot: {}", e);
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
    info!("Input: {}", config.input);
    info!("Output: {}", config.output);
    info!("Parallel: {}", config.parallel);
    info!("Generate plot: {}", config.generate_plot);

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
            match process_file(input_file, output, &config) {
                Ok(()) => {}
                Err(e) => {
                    error!("Error processing {}: {}", input_file.display(), e);
                    // Use expect with a descriptive message for mutex operations
                    errors
                        .lock()
                        .expect(
                            "Failed to acquire lock on error collection - another thread panicked",
                        )
                        .push((input_file.clone(), e.to_string()));
                }
            }
        });

        let errors = errors
            .into_inner()
            .expect("Failed to extract errors from mutex - another thread panicked");
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
            if let Err(e) = process_file(input_file, output, &config) {
                if !is_multiple {
                    return Err(e);
                }
                failures += 1;
                error!("Error processing {}: {}", input_file.display(), e);
            }
        }
        if failures > 0 {
            error!("{} file(s) failed to process", failures);
        }
    }

    Ok(())
}

/// Execute a single closed-loop simulation run
///
/// This is a helper function that extracts the common logic for running closed-loop simulations
/// with either UKF or EKF filters. It handles event stream creation, filter initialization,
/// simulation execution, and results writing.
fn run_single_closed_loop_simulation(
    filter_type: FilterType,
    records: &[TestDataRecord],
    gnss_degradation: &strapdown::messages::GnssDegradationConfig,
    output_file: &Path,
) -> Result<(), Box<dyn Error>> {
    // Build event stream from records and GNSS degradation config
    let event_stream = build_event_stream(records, gnss_degradation);
    info!(
        "Initialized event stream with {} events",
        event_stream.events.len()
    );

    // Initialize and run filter based on type
    let results = match filter_type {
        FilterType::Ukf => {
            let mut ukf = initialize_ukf(records[0].clone(), None, None, None, None, None, None);
            info!("Initialized UKF");
            run_closed_loop(&mut ukf, event_stream, None)
        }
        FilterType::Ekf => {
            let mut ekf = initialize_ekf(records[0].clone(), None, None, None, None, true);
            info!("Initialized EKF");
            run_closed_loop(&mut ekf, event_stream, None)
        }
        FilterType::Eskf => {
            let mut eskf = initialize_eskf(records[0].clone(), None, None, None, None);
            info!("Initialized ESKF");
            run_closed_loop(&mut eskf, event_stream, None)
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
            error!("Error running closed-loop simulation: {}", e);
            Err(e.into())
        }
    }
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
    for input_file in &csv_files {
        info!("Processing file: {}", input_file.display());

        // Load sensor data records from CSV
        let records = TestDataRecord::from_csv(input_file)?;
        info!(
            "Read {} records from {}",
            records.len(),
            input_file.display()
        );

        // Run dead reckoning simulation
        info!(
            "Running dead reckoning simulation on {} records",
            records.len()
        );
        let results = dead_reckoning(&records);
        info!("Generated {} navigation results", results.len());

        // Write results to CSV
        let output_file =
            Path::new(&args.output).join(input_file.file_name().ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!("Input file path '{}' has no filename", input_file.display()),
                )
            })?);
        NavigationResult::to_csv(&results, &output_file)?;
        info!("Results written to {}", output_file.display());
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
    info!("Running in closed-loop mode with {}", filter_name);

    // Get all CSV files to process
    let csv_files = get_csv_files(&args.sim.input)?;
    let is_multiple = csv_files.len() > 1;

    if is_multiple {
        info!("Processing {} CSV files from directory", csv_files.len());
        //println!("Processing {} CSV files from directory", csv_files.len());
    }

    // Process each CSV file
    for input_file in &csv_files {
        info!("Processing file: {}", input_file.display());

        // Load sensor data records from CSV
        let records = TestDataRecord::from_csv(input_file)?;
        info!(
            "Read {} records from {}",
            records.len(),
            input_file.display()
        );

        // Build GNSS degradation config from CLI args
        let gnss_degradation = strapdown::messages::GnssDegradationConfig {
            scheduler: build_scheduler(&args.scheduler),
            fault: build_fault(&args.fault),
            seed: args.seed,
        };

        info!("Using GNSS degradation config: {:?}", gnss_degradation);
        let output_file = Path::new(&args.sim.output).join(input_file);

        // Run simulation using the common helper function
        match run_single_closed_loop_simulation(
            args.filter,
            &records,
            &gnss_degradation,
            &output_file,
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

    Ok(())
}

// ============================================================================
// Geophysical Navigation Functions (feature-gated)
// ============================================================================

/// Convert GeoResolution to GravityResolution
#[cfg(feature = "geonav")]
fn convert_resolution_gravity(resolution: GeoResolution) -> GravityResolution {
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
        GeoResolution::OneMinute => GravityResolution::OneMinute,
        _ => GravityResolution::OneMinute,
    }
}

/// Convert GeoResolution to MagneticResolution
#[cfg(feature = "geonav")]
fn convert_resolution_magnetic(resolution: GeoResolution) -> MagneticResolution {
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
        GeoResolution::TwoMinutes => MagneticResolution::TwoMinutes,
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

    let map_file = input_dir.join(format!("{}_gravity.nc", input_stem));

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

    let map_file = input_dir.join(format!("{}_magnetic.nc", input_stem));

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
    info!(
        "Running geophysical navigation in closed-loop mode with {}",
        filter_name
    );

    // Validate that at least one geophysical map is configured
    if args.geo.gravity_resolution.is_none() && args.geo.magnetic_resolution.is_none() {
        return Err("At least one of --gravity-resolution or --magnetic-resolution must be specified when using --geo".into());
    }

    // Get all CSV files to process
    let csv_files = get_csv_files(&args.sim.input)?;
    let is_multiple = csv_files.len() > 1;

    if is_multiple {
        info!("Processing {} CSV files from directory", csv_files.len());
    }

    // Process each CSV file
    for input_file in &csv_files {
        info!("Processing file: {}", input_file.display());

        // Load sensor data records from CSV
        let records = TestDataRecord::from_csv(input_file)?;
        info!(
            "Read {} records from {}",
            records.len(),
            input_file.display()
        );

        // Load gravity map if configured
        let gravity_map = if let Some(res) = args.geo.gravity_resolution {
            let map_path = match &args.geo.gravity_map_file {
                Some(path) => path.clone(),
                None => find_gravity_map(input_file)?,
            };

            info!("Loading gravity map from: {}", map_path.display());
            let measurement_type =
                GeophysicalMeasurementType::Gravity(convert_resolution_gravity(res));
            let map = Rc::new(GeoMap::load_geomap(map_path, measurement_type)?);
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
            let map = Rc::new(GeoMap::load_geomap(map_path, measurement_type)?);
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
        );
        info!("Built event stream with {} events", events.events.len());

        // Determine number of geophysical states
        let num_geo_states = gravity_map.is_some() as usize + magnetic_map.is_some() as usize;

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
                    records[0].clone(),
                    None,
                    None,
                    None,
                    Some(geo_biases),
                    Some(geo_noise_stds),
                    Some(process_noise),
                );
                info!(
                    "Initialized UKF with state dimension {} (base: 9, geo: {})",
                    ukf.get_estimate().len(),
                    num_geo_states
                );

                info!("Running UKF geophysical navigation simulation...");
                geo_closed_loop_ukf(&mut ukf, events)
            }
            FilterType::Ekf => {
                info!("Initializing EKF...");

                let initial_state = InitialState {
                    latitude: records[0].latitude,
                    longitude: records[0].longitude,
                    altitude: records[0].altitude,
                    northward_velocity: records[0].speed * records[0].bearing.to_radians().cos(),
                    eastward_velocity: records[0].speed * records[0].bearing.to_radians().sin(),
                    vertical_velocity: 0.0,
                    roll: 0.0,
                    pitch: 0.0,
                    yaw: records[0].bearing.to_radians(),
                    in_degrees: true,
                    is_enu: true,
                };

                let imu_biases = vec![0.0; 6];

                let mut covariance_diagonal = vec![
                    1e-6, 1e-6, 1.0, // Position uncertainty
                    0.1, 0.1, 0.1, // Velocity uncertainty
                    1e-4, 1e-4, 1e-4, // Attitude uncertainty
                    1e-6, 1e-6, 1e-6, // Accel bias uncertainty
                    1e-8, 1e-8, 1e-8, // Gyro bias uncertainty
                ];
                covariance_diagonal.extend(vec![1.0; num_geo_states]);

                use nalgebra::DMatrix;
                let mut process_noise_vec = vec![
                    1e-9, 1e-9, 1e-6, // Position process noise
                    1e-6, 1e-6, 1e-6, // Velocity process noise
                    1e-9, 1e-9, 1e-9, // Attitude process noise
                    1e-9, 1e-9, 1e-9, // Accel bias process noise
                    1e-9, 1e-9, 1e-9, // Gyro bias process noise
                ];
                process_noise_vec.extend(vec![1e-9; num_geo_states]);
                let process_noise =
                    DMatrix::from_diagonal(&nalgebra::DVector::from_vec(process_noise_vec));

                let mut ekf = ExtendedKalmanFilter::new(
                    initial_state,
                    imu_biases,
                    covariance_diagonal,
                    process_noise,
                    true,
                );

                info!(
                    "Initialized EKF with state dimension {} (base: 15, geo: {})",
                    ekf.get_estimate().len(),
                    num_geo_states
                );

                info!("Running EKF geophysical navigation simulation...");
                geo_closed_loop_ekf(&mut ekf, events)
            }
            FilterType::Eskf => {
                error!("ESKF is not yet implemented for geophysical navigation");
                return Err("ESKF is not yet implemented for geophysical navigation".into());
            }
        };

        // Write results
        let output_file = args.sim.output.join(input_file.file_name().ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("Input file path '{}' has no filename", input_file.display()),
            )
        })?);

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

    Ok(())
}

/// Execute particle filter simulation
fn run_particle_filter(args: &ParticleFilterSimArgs) -> Result<(), Box<dyn Error>> {
    validate_input_path(&args.sim.input)?;
    validate_output_path(&args.sim.output)?;

    if !matches!(args.filter_type, ParticleFilterType::RaoBlackwellized) {
        return Err("Only Rao-Blackwellized particle filter is implemented in this mode".into());
    }

    let csv_files = get_csv_files(&args.sim.input)?;
    let is_multiple = csv_files.len() > 1;

    if is_multiple {
        info!("Processing {} CSV files from directory", csv_files.len());
    }

    for input_file in &csv_files {
        info!("Processing file: {}", input_file.display());

        let records = TestDataRecord::from_csv(input_file)?;
        info!(
            "Read {} records from {}",
            records.len(),
            input_file.display()
        );

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
                    Some(Rc::new(GeoMap::load_geomap(map_path, measurement_type)?))
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
                    Some(Rc::new(GeoMap::load_geomap(map_path, measurement_type)?))
                } else {
                    None
                };

                (gravity_map, magnetic_map)
            } else {
                (None, None)
            }
        };

        #[cfg(not(feature = "geonav"))]
        let event_stream = build_event_stream(&records, &gnss_degradation);

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
            )
        } else {
            build_event_stream(&records, &gnss_degradation)
        };

        #[cfg(feature = "geonav")]
        let geo_bias_dim = gravity_map.is_some() as usize + magnetic_map.is_some() as usize;
        #[cfg(not(feature = "geonav"))]
        let geo_bias_dim = 0usize;

        let first = &records[0];
        let attitude = Rotation3::from_euler_angles(first.roll, first.pitch, first.yaw);
        let nominal = strapdown::StrapdownState {
            latitude: first.latitude.to_radians(),
            longitude: first.longitude.to_radians(),
            altitude: first.altitude,
            velocity_north: first.speed * first.bearing.to_radians().cos(),
            velocity_east: first.speed * first.bearing.to_radians().sin(),
            velocity_vertical: 0.0,
            attitude,
            is_enu: true,
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

        let mut rbpf = RaoBlackwellizedParticleFilter::new(nominal, config);

        let start_time = event_stream.start_time;
        let mut results = Vec::with_capacity(event_stream.events.len());
        let mut last_ts: Option<chrono::DateTime<chrono::Utc>> = None;
        let mut monitor = HealthMonitor::new(HealthLimits::default());

        for event in event_stream.events.into_iter() {
            let elapsed_s = match &event {
                Event::Imu { elapsed_s, .. } => *elapsed_s,
                Event::Measurement { elapsed_s, .. } => *elapsed_s,
            };
            let ts =
                start_time + chrono::Duration::milliseconds((elapsed_s * 1000.0).round() as i64);

            match event {
                Event::Imu { dt_s, imu, .. } => {
                    rbpf.predict(&imu, dt_s);
                }
                Event::Measurement { mut meas, .. } => {
                    #[cfg(feature = "geonav")]
                    if args.geo.geo {
                        let (mean, _) = rbpf.estimate();
                        let strapdown: strapdown::StrapdownState =
                            mean.as_slice().try_into().unwrap();
                        if let Some(gravity) =
                            meas.as_any_mut().downcast_mut::<GravityMeasurement>()
                        {
                            gravity.set_state(&strapdown);
                        } else if let Some(magnetic) = meas
                            .as_any_mut()
                            .downcast_mut::<MagneticAnomalyMeasurement>()
                        {
                            magnetic.set_state(&strapdown);
                        }
                    }
                    rbpf.update(meas.as_ref());
                }
            }

            let (mean, cov) = rbpf.estimate();
            if let Err(e) = monitor.check(mean.as_slice(), &cov, None) {
                return Err(e.into());
            }

            if Some(ts) != last_ts {
                if let Some(prev_ts) = last_ts {
                    results.push(NavigationResult::from_particle_filter(
                        &prev_ts, &mean, &cov,
                    ));
                }
                last_ts = Some(ts);
            }
        }

        let output_file = args.sim.output.join(input_file.file_name().ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("Input file path '{}' has no filename", input_file.display()),
            )
        })?);

        NavigationResult::to_csv(&results, &output_file)?;
        info!("Results written to {}", output_file.display());
    }

    info!("Particle filter simulation complete");

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

/// Prompt for filter type (UKF or EKF) with validation
fn prompt_filter_type() -> FilterType {
    loop {
        println!(
            "Please specify the filter type you would like:\n\
            [1] - Unscented Kalman Filter (UKF)\n\
            [2] - Extended Kalman Filter (EKF)\n\
            [3] - Error-State Kalman Filter (ESKF)\n\
            [q] - Quit\n"
        );
        if let Some(input) = read_user_input() {
            match input.as_str() {
                "1" => return FilterType::Ukf,
                "2" => return FilterType::Ekf,
                "3" => return FilterType::Eskf,
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
        None => None,
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

        match read_user_input() {
            Some(input) => match input.to_lowercase().as_str() {
                "y" | "yes" => return true,
                "n" | "no" => return false,
                "q" | "quit" => {
                    println!("Configuration cancelled.");
                    std::process::exit(0);
                }
                _ => {
                    println!("Invalid input. Please enter 'y', 'n', or 'q' to quit.");
                    continue;
                }
            },
            None => {
                println!("Invalid input. Please enter 'y', 'n', or 'q' to quit.");
                continue;
            }
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

/// Prompt for GeoResolution with validation
fn prompt_geo_resolution(measurement_type: &str) -> strapdown::sim::GeoResolution {
    use std::io::{self, Write};
    use strapdown::sim::GeoResolution;

    loop {
        println!("\nSelect {} map resolution:", measurement_type);
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
                    continue;
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

        match read_user_input() {
            Some(input) => match input.to_lowercase().as_str() {
                "y" | "yes" => {
                    let resolution = prompt_geo_resolution("gravity");

                    println!("\nGravity measurement bias (mGal) [0.0]: ");
                    let bias = match read_user_input() {
                        Some(input) if !input.is_empty() => match input.parse::<f64>() {
                            Ok(v) => Some(v),
                            Err(_) => {
                                println!("Invalid number. Using default (0.0).");
                                None
                            }
                        },
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
                    continue;
                }
            },
            None => {
                println!("Invalid input. Please enter 'y', 'n', or 'q' to quit.");
                continue;
            }
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

        match read_user_input() {
            Some(input) => match input.to_lowercase().as_str() {
                "y" | "yes" => {
                    let resolution = prompt_geo_resolution("magnetic");

                    println!("\nMagnetic measurement bias (nT) [0.0]: ");
                    let bias = match read_user_input() {
                        Some(input) if !input.is_empty() => match input.parse::<f64>() {
                            Ok(v) => Some(v),
                            Err(_) => {
                                println!("Invalid number. Using default (0.0).");
                                None
                            }
                        },
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
                    continue;
                }
            },
            None => {
                println!("Invalid input. Please enter 'y', 'n', or 'q' to quit.");
                continue;
            }
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
/// [SimulationConfig] and writes it to file.
fn create_config_file() -> Result<(), Box<dyn Error>> {
    println!("\n=== Strapdown Simulation Configuration Wizard ===\n");

    // Gather all configuration parameters
    let config_name = prompt_config_name();
    let save_path = prompt_config_path();

    println!(
        "\nCreating configuration file at: {}/{}\n",
        save_path, config_name
    );
    let input_path = prompt_input_path();
    let output_path = prompt_output_path();
    let mode = prompt_simulation_mode();
    let seed = prompt_seed();
    let parallel = prompt_parallel();

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
        Some(strapdown::sim::ClosedLoopConfig { filter })
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
                gravity_config
                    .map(|(res, bias, noise, map)| (Some(res), bias, noise, map))
                    .unwrap_or((None, None, None, None));

            let (magnetic_resolution, magnetic_bias, magnetic_noise_std, magnetic_map_file) =
                magnetic_config
                    .map(|(res, bias, noise, map)| (Some(res), bias, noise, map))
                    .unwrap_or((None, None, None, None));

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
        output: output_path.clone(),
        mode,
        seed,
        parallel,
        generate_plot: false,
        logging,
        closed_loop,
        particle_filter,
        geophysical,
        gnss_degradation,
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
                &args.input.display()
            );
            run_dead_reckoning(&args)
        }
        Some(Command::OpenLoop(args)) => run_open_loop(&args),
        Some(Command::ClosedLoop(args)) => run_closed_loop_cli(&args),
        Some(Command::ParticleFilter(args)) => run_particle_filter(&args),
        Some(Command::CreateConfig) => create_config_file(),
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
