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

use clap::{Args, Parser, Subcommand};
use log::{error, info};
use rayon::prelude::*;
use std::error::Error;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use strapdown::messages::{GnssScheduler, build_event_stream};
use strapdown::sim::{
    FaultArgs, FilterType, NavigationResult, ParticleFilterType, SchedulerArgs, SimulationConfig,
    SimulationMode, TestDataRecord, build_fault, build_scheduler, initialize_ekf, initialize_ukf,
    run_closed_loop,
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

    #[command(
        name = "conf",
        about = "Generate a template configuration file"
    )]
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

    /// GNSS scheduler settings (dropouts / reduced rate)
    #[command(flatten)]
    scheduler: SchedulerArgs,

    /// Fault model settings (corrupt measurement content)
    #[command(flatten)]
    fault: FaultArgs,
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

/// Validate input path exists and is either a file or directory
fn validate_input_path(input: &Path) -> Result<(), Box<dyn Error>> {
    if !input.exists() {
        return Err(format!("Input path '{}' does not exist.", input.display()).into());
    }
    if !input.is_file() && !input.is_dir() {
        return Err(format!(
            "Input path '{}' is neither a file nor a directory.",
            input.display()
        )
        .into());
    }
    Ok(())
}

/// Get all CSV files from a path (either single file or all CSVs in directory)
fn get_csv_files(input: &Path) -> Result<Vec<PathBuf>, Box<dyn Error>> {
    if input.is_file() {
        if input.extension().and_then(|s| s.to_str()) != Some("csv") {
            return Err(format!("Input file '{}' is not a CSV file.", input.display()).into());
        }
        Ok(vec![input.to_path_buf()])
    } else if input.is_dir() {
        let mut csv_files: Vec<PathBuf> = std::fs::read_dir(input)?
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .filter(|path| {
                path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("csv")
            })
            .collect();

        if csv_files.is_empty() {
            return Err(format!("No CSV files found in directory '{}'.", input.display()).into());
        }

        // Sort for consistent ordering
        csv_files.sort();
        Ok(csv_files)
    } else {
        Err(format!(
            "Input path '{}' is neither a file nor a directory.",
            input.display()
        )
        .into())
    }
}

/// Generate output path for a specific input file
/// If processing multiple files, appends input filename to output path
fn generate_output_path(output: &Path, input_file: &Path, is_multiple: bool) -> PathBuf {
    if !is_multiple {
        return output.to_path_buf().join(input_file.file_name().unwrap());
    }

    let input_stem = input_file
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("output");
    let output_stem = output
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("output");
    let extension = output.extension().and_then(|s| s.to_str()).unwrap_or("csv");

    let new_filename = format!("{}_{}.{}", output_stem, input_stem, extension);

    if let Some(parent) = output.parent() {
        parent.join(new_filename)
    } else {
        PathBuf::from(new_filename)
    }
}

/// Validate output path and create parent directories if needed.
/// If output ends with `.csv`, treat as a single file output and ensure its parent exists.
/// Otherwise, treat as a directory and create it if needed.
fn validate_output_path(output: &Path) -> Result<(), Box<dyn Error>> {
    // Output is a directory, create it if it doesn't exist
    if !output.exists() {
        std::fs::create_dir_all(output)?;
    }
    Ok(())
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
            info!("Dead reckoning mode is not yet fully implemented");
            Err("Dead reckoning mode is not yet fully implemented".into())
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
            };

            let output_file = output.join(input_file.file_name().unwrap());
            match results {
                Ok(ref nav_results) => {
                    NavigationResult::to_csv(nav_results, &output_file)?;
                    info!("Results written to {}", output_file.display());
                    Ok(())
                }
                Err(e) => {
                    error!("Error running closed-loop simulation: {}", e);
                    Err(e.into())
                }
            }
        }
        SimulationMode::ParticleFilter => {
            info!("Particle filter mode is not yet fully implemented");
            Err("Particle filter mode is not yet fully implemented".into())
        }
    }
}

/// Execute simulation from a configuration file
fn run_from_config(config_path: &Path, cli_parallel: bool) -> Result<(), Box<dyn Error>> {
    info!("Loading configuration from {}", config_path.display());

    let mut config = SimulationConfig::from_file(config_path)?;
    
    // Override parallel setting if CLI flag is set
    if cli_parallel {
        config.parallel = true;
    }
    
    info!("Configuration loaded successfully");
    info!("Mode: {:?}", config.mode);
    info!("Input: {}", config.input);
    info!("Output: {}", config.output);
    info!("Parallel: {}", config.parallel);

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
                    error!(
                        "Error processing {}: {}",
                        input_file.display(),
                        e
                    );
                    // Use expect with a descriptive message for mutex operations
                    errors.lock()
                        .expect("Failed to acquire lock on error collection - another thread panicked")
                        .push((input_file.clone(), e.to_string()));
                }
            }
        });

        let errors = errors.into_inner()
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
            let mut ukf =
                initialize_ukf(records[0].clone(), None, None, None, None, None, None);
            info!("Initialized UKF");
            run_closed_loop(&mut ukf, event_stream, None)
        }
        FilterType::Ekf => {
            let mut ekf =
                initialize_ekf(records[0].clone(), None, None, None, None, true);
            info!("Initialized EKF");
            run_closed_loop(&mut ekf, event_stream, None)
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
    validate_input_path(&args.sim.input)?;
    validate_output_path(&args.sim.output)?;

    let filter_name = match args.filter {
        FilterType::Ukf => "Unscented Kalman Filter (UKF)",
        FilterType::Ekf => "Extended Kalman Filter (EKF)",
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

/// Execute particle filter simulation
fn run_particle_filter(args: &ParticleFilterSimArgs) -> Result<(), Box<dyn Error>> {
    validate_input_path(&args.sim.input)?;
    validate_output_path(&args.sim.output)?;

    let csv_files = get_csv_files(&args.sim.input)?;
    let is_multiple = csv_files.len() > 1;

    if is_multiple {
        info!("Processing {} CSV files from directory", csv_files.len());
    }

    for input_file in &csv_files {
        info!("Processing file: {}", input_file.display());

        // TODO: Implement particle filter processing here
        // let records = TestDataRecord::from_csv(input_file)?;
        // let output_file = generate_output_path(&args.sim.output, input_file, is_multiple);
        // ... process and write results ...
    }

    info!("Particle filter mode is not yet fully implemented");
    println!("Particle filter mode is not yet fully implemented");
    // Note: Implementation would go here once particle filter is ready

    Ok(())
}

/// Read a line from stdin, trimming whitespace and checking for quit command
/// Returns None if user enters 'q', otherwise returns the trimmed input
fn read_user_input() -> Option<String> {
    let mut input = String::new();
    io::stdin()
        .read_line(&mut input)
        .expect("Failed to read line");
    let input = input.trim();

    if input.eq_ignore_ascii_case("q") {
        std::process::exit(0);
    }

    if input.is_empty() {
        None
    } else {
        Some(input.to_string())
    }
}

/// Prompt for configuration name with validation
fn prompt_config_name() -> String {
    loop {
        println!("Please name your configuration file with extension (.toml, .json, .yaml) or 'q' to quit:");
        if let Some(input) = read_user_input() {
            return input;
        }
        println!("Error: Configuration path cannot be empty. Please try again.\n");
    }
}
/// Prompt for configuration file path with validation
fn prompt_config_path() -> String {
    loop {
        println!("Please specify the output configuration file path (or 'q' to quit):");
        if let Some(input) = read_user_input() {
            return input;
        }
        println!("Error: Configuration path cannot be empty. Please try again.\n");
    }
}

/// Prompt for input CSV file or directory path with validation
fn prompt_input_path() -> String {
    loop {
        println!("Please specify the input location, either a single CSV file or a directory containing them. ('q' to quit):");
        if let Some(input) = read_user_input() {
            return input;
        }
        println!("Error: Input path cannot be empty. Please try again.\n");
    }
}

/// Prompt for output CSV file path with validation
fn prompt_output_path() -> String {
    loop {
        println!("Please specify the output location to save output data. ('q' to quit):");
        if let Some(input) = read_user_input() {
            return input;
        }
        println!("Error: Output path cannot be empty. Please try again.\n");
    }
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
            [q] - Quit\n"
        );
        if let Some(input) = read_user_input() {
            match input.as_str() {
                "1" => return FilterType::Ukf,
                "2" => return FilterType::Ekf,
                _ => println!("Error: Invalid selection. Please enter 1, 2, or q.\n"),
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

/// Helper function to prompt for f64 with default value and range validation
fn prompt_f64_with_default(prompt_text: &str, default: f64, min_val: f64, max_val: f64) -> f64 {
    loop {
        println!(
            "{} (press Enter for {}, or 'q' to quit):",
            prompt_text, default
        );
        match read_user_input() {
            None => return default,
            Some(input) => match input.parse::<f64>() {
                Ok(val) if val >= min_val && val <= max_val => return val,
                Ok(_) => println!(
                    "Error: Value must be between {} and {}.\n",
                    min_val, max_val
                ),
                Err(_) => println!("Error: Please enter a valid number.\n"),
            },
        }
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
    println!(
        "Please specify a log file path (press Enter to log to stderr, or 'q' to quit):"
    );
    match read_user_input() {
        None => None,
        Some(input) if !input.trim().is_empty() => Some(input),
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
    
    
    println!("\nCreating configuration file at: {}/{}\n", save_path, config_name);
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

    // Build the complete configuration
    let config = SimulationConfig {
        input: input_path,
        output: output_path.clone(),
        mode,
        seed,
        parallel,
        logging,
        closed_loop,
        particle_filter,
        gnss_degradation,
    };

    // validate output location exists and write to file using appropriate format based on file extension
    let config_output_path = Path::new(&save_path).join(&config_name);
    if let Some(parent) = config_output_path.parent()
        && !parent.as_os_str().is_empty() && !parent.exists() {
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
        
        return run_from_config(config_path, cli.parallel);
    }

    // Initialize logger with CLI settings for command-line mode
    init_logger(&cli.log_level, cli.log_file.as_ref())?;

    // Otherwise, execute based on subcommand
    match cli.command {
        Some(Command::DeadReckoning(args)) => {
            println!("Running in Dead Reckoning mode... {}", &args.input.display());
            eprintln!("Error: Dead Reckoning mode is not yet implemented.");
            std::process::exit(1);
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
        let modes = vec![
            SimulationMode::OpenLoop,
            SimulationMode::ClosedLoop,
            SimulationMode::ParticleFilter,
        ];
        assert_eq!(modes.len(), 3);
    }

    #[test]
    fn test_filter_type_variants() {
        let filters = vec![FilterType::Ukf, FilterType::Ekf];
        assert_eq!(filters.len(), 2);
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

        assert_eq!(
            toml_path.extension().and_then(|s| s.to_str()),
            Some("toml")
        );
        assert_eq!(
            json_path.extension().and_then(|s| s.to_str()),
            Some("json")
        );
        assert_eq!(
            yaml_path.extension().and_then(|s| s.to_str()),
            Some("yaml")
        );
    }

    #[test]
    fn test_logging_config_default() {
        use strapdown::sim::{LoggingConfig, LogLevel};
        
        let logging = LoggingConfig::default();
        assert_eq!(logging.level, LogLevel::Info);
        assert!(logging.file.is_none());
    }

    #[test]
    fn test_simulation_config_with_parallel() {
        use strapdown::sim::{SimulationConfig, LogLevel};
        
        let config = SimulationConfig::default();
        assert!(!config.parallel); // Should default to false
        assert_eq!(config.logging.level, LogLevel::Info);
    }

    #[test]
    fn test_logging_config_creation() {
        use strapdown::sim::{LoggingConfig, LogLevel};
        
        let logging = LoggingConfig {
            level: LogLevel::Debug,
            file: Some("/tmp/test.log".to_string()),
        };
        assert_eq!(logging.level, LogLevel::Debug);
        assert_eq!(logging.file, Some("/tmp/test.log".to_string()));
    }
}
