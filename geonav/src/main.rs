//! GEONAV-SIM: A geophysical navigation simulation tool for strapdown inertial navigation systems.
//!
//! This program extends the basic strapdown simulation by incorporating geophysical measurements such as
//! gravity and magnetic anomalies for enhanced navigation accuracy. It loads geophysical maps (NetCDF format)
//! and simulates how these measurements can aid inertial navigation systems, particularly in GNSS-denied environments.
//!
//! The program operates in closed-loop mode, incorporating both GNSS measurements (when available) and geophysical
//! measurements from loaded maps. It can simulate various GNSS degradation scenarios while maintaining navigation
//! accuracy through geophysical aiding.
//!
//! You can run simulations either by:
//!   1. Loading all parameters from a configuration file (TOML/JSON/YAML)
//!   2. Specifying parameters via command-line flags

use clap::{Args, Parser, Subcommand};
use log::{error, info};
use std::error::Error;
use std::io;
use std::path::{Path, PathBuf};
use std::rc::Rc;

use geonav::{
    GeoMap, GeophysicalMeasurementType, GravityResolution, MagneticResolution, build_event_stream,
    geo_closed_loop, geo_closed_loop_ekf,
};
use strapdown::NavigationFilter;
use strapdown::kalman::{ExtendedKalmanFilter, InitialState};
use strapdown::sim::{
    DEFAULT_PROCESS_NOISE, FaultArgs, FilterType, GeoMeasurementType, GeoResolution,
    GeonavSimulationConfig, NavigationResult, SchedulerArgs, TestDataRecord, build_fault,
    build_scheduler, initialize_ukf,
};

const LONG_ABOUT: &str = "GEONAV-SIM: A geophysical navigation simulation tool for strapdown inertial navigation systems.

This program extends the basic strapdown simulation by incorporating geophysical measurements such as 
gravity and magnetic anomalies for enhanced navigation accuracy. It loads geophysical maps (NetCDF format) 
and simulates how these measurements can aid inertial navigation systems, particularly in GNSS-denied environments.

The program operates in closed-loop mode, incorporating both GNSS measurements (when available) and geophysical 
measurements from loaded maps. It can simulate various GNSS degradation scenarios while maintaining navigation 
accuracy through geophysical aiding.

You can run simulations either by:
  1. Loading all parameters from a configuration file (TOML/JSON/YAML)
  2. Specifying parameters via command-line flags

Input data format is identical to strapdown-sim, with additional geophysical map files:
* Input CSV: Standard IMU/GNSS data as per strapdown-sim specification
* Gravity maps: *_gravity.nc files containing gravity anomaly data
* Magnetic maps: *_magnetic.nc files containing magnetic anomaly data

The program automatically detects and loads the appropriate map file based on the measurement type specified 
and the input file directory.";

/// Command line arguments
#[derive(Parser)]
#[command(author, version, about = "A geophysical navigation simulation tool for strapdown inertial navigation systems.", long_about = LONG_ABOUT)]
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
        name = "cl",
        about = "Run geophysical navigation simulation in closed-loop mode",
        long_about = "Run geophysical INS simulation in a closed-loop (feedback) mode. In this mode, GNSS measurements and geophysical measurements are incorporated to correct for IMU drift using either an Unscented Kalman Filter (UKF) or Extended Kalman Filter (EKF). Various GNSS degradation scenarios can be simulated, including jamming, reduced update rates, and spoofing."
    )]
    ClosedLoop(Box<ClosedLoopSimArgs>),

    #[command(name = "conf", about = "Generate a template configuration file")]
    CreateConfig,
}

/// Common simulation arguments for input/output
#[derive(Args, Clone, Debug)]
struct SimArgs {
    /// Input CSV file path or directory containing CSV files
    /// If a directory is provided, all CSV files in it will be processed
    #[arg(short, long, value_parser)]
    input: PathBuf,

    /// Output CSV file path or directory
    /// When processing multiple files, output filenames will be generated as: {output_stem}_{input_stem}.csv
    #[arg(short, long, value_parser)]
    output: PathBuf,
}

/// Geophysical measurement arguments
#[derive(Args, Clone, Debug)]
struct GeophysicalArgs {
    /// Type of geophysical measurement to use
    #[arg(long, value_enum, default_value_t = GeoMeasurementType::Gravity)]
    geo_type: GeoMeasurementType,

    /// Map resolution for geophysical data
    #[arg(long, value_enum, default_value_t = GeoResolution::OneMinute)]
    geo_resolution: GeoResolution,

    /// Bias for geophysical measurement noise
    #[arg(long, default_value_t = 0.0)]
    geo_bias: f64,

    /// Standard deviation for geophysical measurement noise
    #[arg(long, default_value_t = 100.0)]
    geo_noise_std: f64,

    /// Frequency in seconds for geophysical measurements (if not specified, uses every available measurement)
    #[arg(long)]
    geo_frequency_s: Option<f64>,

    /// Custom map file path (optional - if not provided, auto-detects based on input directory)
    #[arg(long)]
    map_file: Option<PathBuf>,
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

    /// Geophysical measurement configuration
    #[command(flatten)]
    geo: GeophysicalArgs,

    /// GNSS scheduler settings (dropouts / reduced rate)
    #[command(flatten)]
    scheduler: SchedulerArgs,

    /// Fault model settings (corrupt measurement content)
    #[command(flatten)]
    fault: FaultArgs,
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

/// Validate output path and create parent directories if needed
fn validate_output_path(output: &Path) -> Result<(), Box<dyn Error>> {
    // Output is a directory, create it if it doesn't exist
    if !output.exists() {
        std::fs::create_dir_all(output)?;
    }
    Ok(())
}

/// Convert CLI geo measurement type to library type
fn build_measurement_type(
    geo_type: GeoMeasurementType,
    resolution: GeoResolution,
) -> GeophysicalMeasurementType {
    match geo_type {
        GeoMeasurementType::Gravity => {
            let gravity_res = match resolution {
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
                // For gravity, highest resolution available is 1 minute
                _ => GravityResolution::OneMinute,
            };
            GeophysicalMeasurementType::Gravity(gravity_res)
        }
        GeoMeasurementType::Magnetic => {
            let magnetic_res = match resolution {
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
                // For magnetic, highest resolution available is 2 minutes
                _ => MagneticResolution::TwoMinutes,
            };
            GeophysicalMeasurementType::Magnetic(magnetic_res)
        }
    }
}

/// Auto-detect map file based on input directory and measurement type
fn find_map_file(
    input_path: &Path,
    geo_type: GeoMeasurementType,
) -> Result<PathBuf, Box<dyn Error>> {
    let input_dir = input_path
        .parent()
        .ok_or("Cannot determine input directory")?;

    let input_stem = input_path
        .file_stem()
        .ok_or("Cannot determine input file stem")?
        .to_string_lossy();

    let suffix = match geo_type {
        GeoMeasurementType::Gravity => "_gravity.nc",
        GeoMeasurementType::Magnetic => "_magnetic.nc",
    };

    let map_file = input_dir.join(format!("{}{}", input_stem, suffix));

    if map_file.exists() {
        Ok(map_file)
    } else {
        Err(format!("Map file not found: {}", map_file.display()).into())
    }
}

/// Process a single CSV file with geophysical navigation
fn process_file(
    input_file: &Path,
    output: &Path,
    config: &GeonavSimulationConfig,
) -> Result<(), Box<dyn Error>> {
    info!("Processing file: {}", input_file.display());

    // Load sensor data
    let records = TestDataRecord::from_csv(input_file)?;
    info!(
        "Read {} records from {}",
        records.len(),
        input_file.display()
    );

    // Determine map file path
    let map_path = match &config.geophysical.map_file {
        Some(path) => PathBuf::from(path),
        None => find_map_file(input_file, config.geophysical.geo_type)?,
    };

    info!("Loading geophysical map from: {}", map_path.display());

    // Build measurement type with specified resolution
    let measurement_type = build_measurement_type(
        config.geophysical.geo_type,
        config.geophysical.geo_resolution,
    );

    // Load geophysical map
    let geomap = Rc::new(GeoMap::load_geomap(map_path, measurement_type)?);
    info!(
        "Loaded {} map with {} x {} grid points",
        geomap.get_map_type(),
        geomap.get_lats().len(),
        geomap.get_lons().len()
    );

    // Build event stream with geophysical measurements
    let events = build_event_stream(
        &records,
        &config.gnss_degradation,
        geomap,
        Some(config.geophysical.geo_noise_std),
        config.geophysical.geo_frequency_s,
    );
    info!("Built event stream with {} events", events.events.len());

    // Run simulation based on filter type
    let results = match config.filter {
        FilterType::Ukf => {
            info!("Initializing UKF...");
            let mut process_noise: Vec<f64> = DEFAULT_PROCESS_NOISE.into();
            process_noise.extend([1e-9]); // Extend for geophysical state

            let mut ukf = initialize_ukf(
                records[0].clone(),
                None,
                None,
                None,
                Some(vec![config.geophysical.geo_bias; 1]),
                Some(vec![config.geophysical.geo_noise_std; 1]),
                Some(process_noise),
            );
            info!(
                "Initialized UKF with state dimension {}",
                ukf.get_estimate().len()
            );

            info!("Running UKF geophysical navigation simulation...");
            geo_closed_loop(&mut ukf, events)
        }
        FilterType::Ekf => {
            info!("Initializing EKF...");

            // Construct initial state from first record
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

            // IMU biases (accelerometer + gyroscope)
            let imu_biases = vec![0.0; 6];

            // Initial covariance diagonal (15-state: pos, vel, att, biases)
            let covariance_diagonal = vec![
                1e-6, 1e-6, 1.0, // Position uncertainty
                0.1, 0.1, 0.1, // Velocity uncertainty
                1e-4, 1e-4, 1e-4, // Attitude uncertainty
                1e-6, 1e-6, 1e-6, // Accel bias uncertainty
                1e-8, 1e-8, 1e-8, // Gyro bias uncertainty
            ];

            // Process noise (15-state)
            use nalgebra::DMatrix;
            let process_noise = DMatrix::from_diagonal(&nalgebra::DVector::from_vec(vec![
                1e-9, 1e-9, 1e-6, // Position process noise
                1e-6, 1e-6, 1e-6, // Velocity process noise
                1e-9, 1e-9, 1e-9, // Attitude process noise
                1e-9, 1e-9, 1e-9, // Accel bias process noise
                1e-9, 1e-9, 1e-9, // Gyro bias process noise
            ]));

            let mut ekf = ExtendedKalmanFilter::new(
                initial_state,
                imu_biases,
                covariance_diagonal,
                process_noise,
                true, // Use 15-state with biases
            );

            info!(
                "Initialized EKF with state dimension {}",
                ekf.get_estimate().len()
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
    let output_file = output.join(input_file.file_name().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("Input file path '{}' has no filename", input_file.display()),
        )
    })?);

    match results {
        Ok(ref nav_results) => {
            NavigationResult::to_csv(nav_results, &output_file)?;
            info!("Results written to {}", output_file.display());
            Ok(())
        }
        Err(e) => {
            error!("Error running geophysical navigation simulation: {}", e);
            Err(e.into())
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

    let mut config = GeonavSimulationConfig::from_file(config_path)?;

    // Override parallel setting if CLI flag is set
    if cli_parallel {
        config.parallel = true;
    }

    // Override plot setting if CLI flag is set
    if cli_plot {
        config.generate_plot = true;
    }

    info!("Configuration loaded successfully");
    info!("Filter: {:?}", config.filter);
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

    // Process files sequentially (parallel not yet implemented for geonav)
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

    Ok(())
}

/// Execute closed-loop geophysical navigation simulation
fn run_closed_loop_cli(args: &ClosedLoopSimArgs) -> Result<(), Box<dyn Error>> {
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

    // Get all CSV files to process
    let csv_files = get_csv_files(&args.sim.input)?;
    let is_multiple = csv_files.len() > 1;

    if is_multiple {
        info!("Processing {} CSV files from directory", csv_files.len());
    }

    // Build configuration from CLI args
    let config = GeonavSimulationConfig {
        input: args.sim.input.to_string_lossy().to_string(),
        output: args.sim.output.to_string_lossy().to_string(),
        filter: args.filter,
        seed: args.seed,
        parallel: false,
        generate_plot: false,
        logging: Default::default(),
        geophysical: strapdown::sim::GeophysicalConfig {
            geo_type: args.geo.geo_type,
            geo_resolution: args.geo.geo_resolution,
            geo_bias: args.geo.geo_bias,
            geo_noise_std: args.geo.geo_noise_std,
            geo_frequency_s: args.geo.geo_frequency_s,
            map_file: args
                .geo
                .map_file
                .as_ref()
                .map(|p| p.to_string_lossy().to_string()),
        },
        gnss_degradation: strapdown::messages::GnssDegradationConfig {
            scheduler: build_scheduler(&args.scheduler),
            fault: build_fault(&args.fault),
            seed: args.seed,
        },
    };

    // Process each CSV file
    for input_file in &csv_files {
        match process_file(input_file, &args.sim.output, &config) {
            Ok(()) => {
                // Success
            }
            Err(e) => {
                error!(
                    "Error running geophysical navigation on {}: {}",
                    input_file.display(),
                    e
                );
                if !is_multiple {
                    return Err(e);
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

/// Read a line from stdin, trimming whitespace and checking for quit command
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
        println!(
            "Please name your configuration file with extension (.toml, .json, .yaml) or 'q' to quit:"
        );
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

/// Interactive configuration file creation wizard
fn create_config_file() -> Result<(), Box<dyn Error>> {
    println!("\n=== Geonav Simulation Configuration Wizard ===\n");
    println!(
        "This wizard will help you create a configuration file for geophysical navigation simulations."
    );
    println!(
        "For now, we'll create a basic template. You can edit it to customize your simulation.\n"
    );

    let config_name = prompt_config_name();
    let save_path = prompt_config_path();

    println!(
        "\nCreating configuration file at: {}/{}\n",
        save_path, config_name
    );

    // Create a default configuration
    let config = GeonavSimulationConfig::default();

    // Validate output location exists and write to file
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
    println!("\nYou can now edit the file to customize your simulation settings.");
    println!("Then run the simulation with:");
    println!("  geonav-sim --config {}", config_output_path.display());

    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();

    // If --config is provided, load config
    if let Some(ref config_path) = cli.config {
        // Load config first to get logging preferences
        let config = GeonavSimulationConfig::from_file(config_path)?;

        // Determine log level
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

    // Execute based on subcommand
    match cli.command {
        Some(Command::ClosedLoop(args)) => run_closed_loop_cli(&args),
        Some(Command::CreateConfig) => create_config_file(),
        None => {
            eprintln!("Error: No command provided. Use -h or --help for usage information.");
            std::process::exit(1);
        }
    }
}
