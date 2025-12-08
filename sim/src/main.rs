use clap::{Args, Parser, Subcommand};
use log::{error, info};
use std::error::Error;
use std::path::{Path, PathBuf};
use strapdown::kalman::NavigationFilter;
use strapdown::messages::{GnssDegradationConfig, build_event_stream};
use strapdown::particle::ParticleAveragingStrategy;
use strapdown::sim::{
    FaultArgs, NavigationResult, SchedulerArgs, TestDataRecord, build_fault, build_scheduler,
    initialize_particle_filter, initialize_ukf, run_closed_loop,
};

const LONG_ABOUT: &str = "STRAPDOWN: A simulation and analysis tool for strapdown inertial navigation systems.

This program can operate in two modes: open-loop and closed-loop. In open loop mode, the system relies solely on inertial measurements (IMU) and an initial position estimate and performs simple dead reckoning. This mode is only particularly useful for high-accuracy low noise IMUs such as those for aerospace or marine applications (i.e. have a drift rate <=1nm per 24 hours). In closed-loop mode, the system incorporates GNSS measurements to correct for IMU drift and improve overall navigation accuracy. The closed-loop mode can simulate various GNSS degradation scenarios, including jamming (signal dropouts, reduced update rates, and measurement corruption) and a limited form of spoofing (bias introduction, signal hijack).

This program is designed to work with tabular comma-separated value datasets that contain IMU and GNSS measurements of the form:
* time - ISO UTC timestamp of the form YYYY-MM-DD HH:mm:ss.ssss+HH:MM (where the +HH:MM is the timezone offset)
* speed - Speed measurement in meters per second
* bearing - Bearing measurement in degrees
* altitude - Altitude measurement in meters
* longitude - Longitude measurement in degrees
* latitude - Latitude measurement in degrees
* qz - Quaternion component
* qy - Quaternion component
* qx - Quaternion component
* qw - Quaternion component
* roll - Roll angle in degrees
* pitch - Pitch angle in degrees
* yaw - Yaw angle in degrees
* acc_z - Acceleration in the Z direction (meters per second squared)
* acc_y - Acceleration in the Y direction (meters per second squared)
* acc_x - Acceleration in the X direction (meters per second squared)
* gyro_z - Angular velocity around the Z axis (radians per second)
* gyro_y - Angular velocity around the Y axis (radians per second)
* gyro_x - Angular velocity around the X axis (radians per second)
* mag_z - Magnetic field strength in the Z direction (micro teslas)
* mag_y - Magnetic field strength in the Y direction (micro teslas)
* mag_x - Magnetic field strength in the X direction (micro teslas)
* relativeAltitude - Relative altitude measurement (meters)
* pressure - Atmospheric pressure measurement (milli bar)
* grav_z - Gravitational acceleration in the Z direction (meters per second squared)
* grav_y - Gravitational acceleration in the Y direction (meters per second squared)
* grav_x - Gravitational acceleration in the X direction (meters per second squared)";

/// Command line arguments
#[derive(Parser)]
#[command(author, version, about, long_about = LONG_ABOUT)]
struct Cli {
    /// Command to execute
    #[command(subcommand)]
    command: Command,
    /// Log level (off, error, warn, info, debug, trace)
    #[arg(long, default_value = "info", global = true)]
    log_level: String,
    /// Log file path (if not specified, logs to stderr)
    #[arg(long, global = true)]
    log_file: Option<PathBuf>,
}

/// Top-level commands
#[derive(Subcommand, Clone)]
enum Command {
    #[command(
        name = "open-loop",
        about = "Run the simulation in open-loop (feed-forward INS) mode"
    )]
    OpenLoop(SimArgs),
    #[command(
        name = "closed-loop",
        about = "Run the simulation in closed-loop (feedback INS) mode"
    )]
    ClosedLoop(ClosedLoopSimArgs),
    #[command(
        name = "particle-filter",
        about = "Run the simulation using a particle filter INS architecture"
    )]
    ParticleFilter(ParticleFilterSimArgs),
    #[command(
        name = "generate-config",
        about = "Generate a blank template GNSS degradation configuration file"
    )]
    GenerateConfig(GenerateConfigArgs),
}

/// Common simulation arguments for input/output
#[derive(Args, Clone, Debug)]
struct SimArgs {
    /// Input file path
    #[arg(short, long, value_parser)]
    input: PathBuf,
    /// Output file path
    #[arg(short, long, value_parser)]
    output: PathBuf,
}

/// Closed-loop simulation arguments combining SimArgs with closed-loop specific options
#[derive(Args, Clone, Debug)]
struct ClosedLoopSimArgs {
    /// Common simulation input/output arguments
    #[command(flatten)]
    sim: SimArgs,
    /// RNG seed (applies to any stochastic options)
    #[arg(long, default_value_t = 42)]
    seed: u64,
    /// Scheduler settings (dropouts / reduced rate)
    #[command(flatten)]
    scheduler: SchedulerArgs,
    /// Fault model settings (corrupt measurement content)
    #[command(flatten)]
    fault: FaultArgs,
    /// Path to a GNSS degradation config file (json|yaml|yml|toml)
    #[arg(long)]
    config: Option<PathBuf>,
}

/// Particle filter simulation arguments
#[derive(Args, Clone, Debug)]
struct ParticleFilterSimArgs {
    /// Common simulation input/output arguments
    #[command(flatten)]
    sim: SimArgs,
    /// RNG seed (applies to any stochastic options)
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
    /// Particle averaging strategy
    #[arg(long, value_enum, default_value_t = ParticleAveragingStrategy::WeightedAverage)]
    averaging_strategy: ParticleAveragingStrategy,
    /// Scheduler settings (dropouts / reduced rate)
    #[command(flatten)]
    scheduler: SchedulerArgs,
    /// Fault model settings (corrupt measurement content)
    #[command(flatten)]
    fault: FaultArgs,
    /// Path to a GNSS degradation config file (json|yaml|yml|toml)
    #[arg(long)]
    config: Option<PathBuf>,
}

/// Arguments for the generate-config command
#[derive(Args, Clone, Debug)]
struct GenerateConfigArgs {
    /// Output file path for the generated config file.
    /// The file extension determines the format: .json, .yaml/.yml, or .toml
    #[arg(short, long, value_parser)]
    output: PathBuf,
}

/// Initialize the logger with the specified configuration
fn init_logger(log_level: &str, log_file: Option<&PathBuf>) -> Result<(), Box<dyn Error>> {
    use std::io::Write;

    let level = log_level.parse::<log::LevelFilter>().unwrap_or_else(|_| {
        eprintln!("Invalid log level '{}', defaulting to 'info'", log_level);
        log::LevelFilter::Info
    });

    let mut builder = env_logger::Builder::new();
    builder.filter_level(level);
    builder.format(|buf, record| {
        writeln!(
            buf,
            "{} [{}] - {}",
            chrono::Local::now().format("%Y-%m-%d %H:%M:%S%.3f"),
            record.level(),
            record.args()
        )
    });

    if let Some(log_path) = log_file {
        let target = Box::new(
            std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(log_path)?,
        );
        builder.target(env_logger::Target::Pipe(target));
    }

    builder.try_init()?;
    Ok(())
}

/// Validate input file exists and is readable
fn validate_input_file(input: &Path) -> Result<(), Box<dyn Error>> {
    if !input.exists() {
        return Err(format!("Input file '{}' does not exist.", input.display()).into());
    }
    if !input.is_file() {
        return Err(format!("Input path '{}' is not a file.", input.display()).into());
    }
    Ok(())
}

/// Validate output path and create parent directories if needed
fn validate_output_path(output: &Path) -> Result<(), Box<dyn Error>> {
    if let Some(parent) = output.parent()
        && !parent.as_os_str().is_empty()
        && !parent.exists()
    {
        std::fs::create_dir_all(parent)?;
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();

    // Initialize logger
    init_logger(&cli.log_level, cli.log_file.as_ref())?;

    match cli.command {
        Command::GenerateConfig(ref args) => {
            // Validate output path
            validate_output_path(&args.output)?;

            // Create a default config with sensible baseline values
            let cfg = GnssDegradationConfig::default();

            // Write to file using the appropriate format based on extension
            match cfg.to_file(&args.output) {
                Ok(_) => {
                    info!("Generated config file: {}", args.output.display());
                    println!("Generated config file: {}", args.output.display());
                }
                Err(e) => {
                    error!("Failed to write config file: {}", e);
                    return Err(Box::new(e));
                }
            }
        }
        Command::OpenLoop(ref args) => {
            validate_input_file(&args.input)?;
            validate_output_path(&args.output)?;
            info!("Running in open-loop mode");
            // Note: Open-loop mode is currently not fully implemented
            // This would run dead reckoning simulation
        }
        Command::ParticleFilter(ref args) => {
            validate_input_file(&args.sim.input)?;
            validate_output_path(&args.sim.output)?;

            // Load sensor data records from CSV
            let records = TestDataRecord::from_csv(&args.sim.input)?;
            info!(
                "Read {} records from {}",
                records.len(),
                &args.sim.input.display()
            );

            let cfg = if let Some(ref cfg_path) = args.config {
                match GnssDegradationConfig::from_file(cfg_path) {
                    Ok(c) => c,
                    Err(e) => {
                        error!("Failed to read config {}: {}", cfg_path.display(), e);
                        return Err(Box::new(e));
                    }
                }
            } else {
                GnssDegradationConfig {
                    scheduler: build_scheduler(&args.scheduler),
                    fault: build_fault(&args.fault),
                    seed: args.seed,
                }
            };

            let events = build_event_stream(&records, &cfg);
            let mut pf = initialize_particle_filter(
                records[0].clone(),
                args.num_particles,
                0.0, //args.position_std,
                0.0, //args.velocity_std,
                0.0, //args.attitude_std,
                None,
            );
            info!(
                "Initialized particle filter with {} particles",
                args.num_particles
            );
            info!("Initial state mean: {:?}", pf.get_estimate());

            info!(
                "Running particle filter with {} particles, strategy: {:?}",
                args.num_particles, args.averaging_strategy
            );

            let results = run_closed_loop(&mut pf, events, None);

            match results {
                Ok(ref nav_results) => {
                    match NavigationResult::to_csv(nav_results, &args.sim.output) {
                        Ok(_) => info!("Results written to {}", args.sim.output.display()),
                        Err(e) => error!("Error writing results: {}", e),
                    }
                }
                Err(e) => error!("Error running particle filter simulation: {}", e),
            };
        }
        Command::ClosedLoop(ref args) => {
            validate_input_file(&args.sim.input)?;
            validate_output_path(&args.sim.output)?;
            info!("Running in closed-loop mode");
            // Load sensor data records from CSV, tolerant to mixed/variable-length rows and encoding issues.
            let records = TestDataRecord::from_csv(&args.sim.input)?;
            info!(
                "Read {} records from {}",
                records.len(),
                &args.sim.input.display()
            );
            let cfg = if let Some(ref cfg_path) = args.config {
                match GnssDegradationConfig::from_file(cfg_path) {
                    Ok(c) => c,
                    Err(e) => {
                        error!("Failed to read config {}: {}", cfg_path.display(), e);
                        return Err(Box::new(e));
                    }
                }
            } else {
                GnssDegradationConfig {
                    scheduler: build_scheduler(&args.scheduler),
                    fault: build_fault(&args.fault),
                    seed: args.seed,
                }
            };
            info!("Using GNSS degradation config: {:?}", cfg);
            let event_stream = build_event_stream(&records, &cfg);
            info!(
                "Initialized event stream with {} events",
                event_stream.events.len()
            );
            let mut ukf = initialize_ukf(records[0].clone(), None, None, None, None, None, None);
            info!("Initialized UKF with state: {:?}", ukf);
            let results = run_closed_loop(&mut ukf, event_stream, None);
            match results {
                Ok(ref nav_results) => {
                    match NavigationResult::to_csv(nav_results, &args.sim.output) {
                        Ok(_) => info!("Results written to {}", args.sim.output.display()),
                        Err(e) => error!("Error writing results: {}", e),
                    }
                }
                Err(e) => error!("Error running closed-loop simulation: {}", e),
            };
        }
    }
    Ok(())
}
