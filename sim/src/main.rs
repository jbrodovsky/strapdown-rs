use clap::{Args, Parser, Subcommand};
use log::{error, info};
use std::error::Error;
use std::path::{Path, PathBuf};
use strapdown::messages::build_event_stream;
use strapdown::sim::{
    ClosedLoopConfig, FilterType, FaultArgs, NavigationResult, ParticleFilterConfig,
    ParticleFilterType, SchedulerArgs, SimulationConfig, SimulationMode, TestDataRecord,
    build_fault, build_scheduler, initialize_ekf, initialize_ukf, run_closed_loop,
};

const LONG_ABOUT: &str = "STRAPDOWN: A simulation and analysis tool for strapdown inertial navigation systems.

This program can operate in three modes: open-loop, closed-loop, and particle-filter.

- **Open-loop mode**: Relies solely on inertial measurements (IMU) and an initial position estimate 
  for dead reckoning. Useful for high-accuracy IMUs with drift rates ≤1 nm per 24 hours.

- **Closed-loop mode**: Incorporates GNSS measurements to correct IMU drift using either an 
  Unscented Kalman Filter (UKF) or Extended Kalman Filter (EKF). Supports GNSS degradation 
  scenarios including jamming, reduced update rates, and spoofing.

- **Particle-filter mode**: Uses particle-based state estimation, supporting both standard and 
  Rao-Blackwellized implementations.

You can run simulations either by:
  1. Loading all parameters from a configuration file (TOML/JSON/YAML)
  2. Specifying parameters via command-line flags

For dataset format details, see the documentation or use --help with specific subcommands.";

/// Command line arguments
#[derive(Parser)]
#[command(author, version, about, long_about = LONG_ABOUT)]
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
}

/// Top-level commands
#[derive(Subcommand, Clone)]
enum Command {
    #[command(
        name = "open-loop",
        about = "Run simulation in open-loop (dead reckoning) mode"
    )]
    OpenLoop(SimArgs),
    
    #[command(
        name = "closed-loop",
        about = "Run simulation in closed-loop (Kalman filter) mode"
    )]
    ClosedLoop(ClosedLoopSimArgs),
    
    #[command(
        name = "particle-filter",
        about = "Run simulation using particle filter"
    )]
    ParticleFilter(ParticleFilterSimArgs),
    
    #[command(
        name = "create-config",
        about = "Generate a template configuration file"
    )]
    CreateConfig(CreateConfigArgs),
}

/// Common simulation arguments for input/output
#[derive(Args, Clone, Debug)]
struct SimArgs {
    /// Input CSV file path
    #[arg(short, long, value_parser)]
    input: PathBuf,
    
    /// Output CSV file path
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

/// Execute simulation from a configuration file
fn run_from_config(config_path: &Path) -> Result<(), Box<dyn Error>> {
    info!("Loading configuration from {}", config_path.display());
    
    let config = SimulationConfig::from_file(config_path)?;
    info!("Configuration loaded successfully");
    info!("Mode: {:?}", config.mode);
    info!("Input: {}", config.input);
    info!("Output: {}", config.output);
    
    // Validate paths
    let input = Path::new(&config.input);
    let output = Path::new(&config.output);
    validate_input_file(input)?;
    validate_output_path(output)?;
    
    // Load sensor data
    let records = TestDataRecord::from_csv(input)?;
    info!("Read {} records from {}", records.len(), config.input);
    
    // Execute based on mode
    match config.mode {
        SimulationMode::OpenLoop => {
            info!("Open-loop mode is not yet fully implemented");
            println!("Open-loop mode is not yet fully implemented");
        }
        SimulationMode::ClosedLoop => {
            let filter_config = config.closed_loop.unwrap_or_default();
            info!("Running closed-loop mode with {:?} filter", filter_config.filter);
            
            let event_stream = build_event_stream(&records, &config.gnss_degradation);
            info!("Initialized event stream with {} events", event_stream.events.len());
            
            let results = match filter_config.filter {
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
            };
            
            match results {
                Ok(ref nav_results) => {
                    NavigationResult::to_csv(nav_results, output)?;
                    info!("Results written to {}", config.output);
                    println!("Results written to {}", config.output);
                }
                Err(e) => {
                    error!("Error running closed-loop simulation: {}", e);
                    return Err(e.into());
                }
            }
        }
        SimulationMode::ParticleFilter => {
            info!("Particle filter mode is not yet fully implemented");
            println!("Particle filter mode is not yet fully implemented");
        }
    }
    
    Ok(())
}

/// Execute open-loop simulation
fn run_open_loop(args: &SimArgs) -> Result<(), Box<dyn Error>> {
    validate_input_file(&args.input)?;
    validate_output_path(&args.output)?;
    
    info!("Open-loop mode is not yet fully implemented");
    println!("Open-loop mode is not yet fully implemented");
    
    Ok(())
}

/// Execute closed-loop simulation
fn run_closed_loop_cli(args: &ClosedLoopSimArgs) -> Result<(), Box<dyn Error>> {
    validate_input_file(&args.sim.input)?;
    validate_output_path(&args.sim.output)?;
    
    let filter_name = match args.filter {
        FilterType::Ukf => "Unscented Kalman Filter (UKF)",
        FilterType::Ekf => "Extended Kalman Filter (EKF)",
    };
    info!("Running in closed-loop mode with {}", filter_name);
    
    // Load sensor data records from CSV
    let records = TestDataRecord::from_csv(&args.sim.input)?;
    info!("Read {} records from {}", records.len(), args.sim.input.display());
    
    // Build GNSS degradation config from CLI args
    let gnss_degradation = strapdown::messages::GnssDegradationConfig {
        scheduler: build_scheduler(&args.scheduler),
        fault: build_fault(&args.fault),
        seed: args.seed,
        magnetometer: Default::default(),
    };
    
    info!("Using GNSS degradation config: {:?}", gnss_degradation);
    let event_stream = build_event_stream(&records, &gnss_degradation);
    info!("Initialized event stream with {} events", event_stream.events.len());
    
    // Initialize and run filter
    let results = match args.filter {
        FilterType::Ukf => {
            let mut ukf = initialize_ukf(records[0].clone(), None, None, None, None, None, None);
            info!("Initialized UKF with state: {:?}", ukf);
            run_closed_loop(&mut ukf, event_stream, None)
        }
        FilterType::Ekf => {
            let mut ekf = initialize_ekf(records[0].clone(), None, None, None, None, true);
            info!("Initialized EKF with state: {:?}", ekf);
            run_closed_loop(&mut ekf, event_stream, None)
        }
    };
    
    match results {
        Ok(ref nav_results) => {
            NavigationResult::to_csv(nav_results, &args.sim.output)?;
            info!("Results written to {}", args.sim.output.display());
            println!("Results written to {}", args.sim.output.display());
        }
        Err(e) => {
            error!("Error running closed-loop simulation: {}", e);
            return Err(e.into());
        }
    }
    
    Ok(())
}

/// Execute particle filter simulation
fn run_particle_filter(args: &ParticleFilterSimArgs) -> Result<(), Box<dyn Error>> {
    validate_input_file(&args.sim.input)?;
    validate_output_path(&args.sim.output)?;
    
    info!("Particle filter mode is not yet fully implemented");
    println!("Particle filter mode is not yet fully implemented");
    // Note: Implementation would go here once particle filter is ready
    
    Ok(())
}

/// Generate a template configuration file
fn create_config_file(args: &CreateConfigArgs) -> Result<(), Box<dyn Error>> {
    validate_output_path(&args.output)?;
    
    // Create config with appropriate mode-specific fields
    let config = match args.mode {
        SimulationMode::OpenLoop => SimulationConfig {
            mode: SimulationMode::OpenLoop,
            closed_loop: None,
            particle_filter: None,
            ..Default::default()
        },
        SimulationMode::ClosedLoop => SimulationConfig {
            mode: SimulationMode::ClosedLoop,
            closed_loop: Some(ClosedLoopConfig::default()),
            particle_filter: None,
            ..Default::default()
        },
        SimulationMode::ParticleFilter => SimulationConfig {
            mode: SimulationMode::ParticleFilter,
            closed_loop: None,
            particle_filter: Some(ParticleFilterConfig::default()),
            ..Default::default()
        },
    };
    
    // Write to file using appropriate format
    config.to_file(&args.output)?;
    
    info!("Created config file: {}", args.output.display());
    println!("Created config file: {}", args.output.display());
    println!("Edit the file to customize your simulation, then run:");
    println!("  strapdown-sim --config {}", args.output.display());
    
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();

    // Initialize logger
    init_logger(&cli.log_level, cli.log_file.as_ref())?;

    // If --config is provided, run from config file and ignore subcommands
    if let Some(ref config_path) = cli.config {
        return run_from_config(config_path);
    }

    // Otherwise, execute based on subcommand
    match cli.command {
        Some(Command::OpenLoop(args)) => run_open_loop(&args),
        Some(Command::ClosedLoop(args)) => run_closed_loop_cli(&args),
        Some(Command::ParticleFilter(args)) => run_particle_filter(&args),
        Some(Command::CreateConfig(args)) => create_config_file(&args),
        None => {
            error!("No command specified. Use --help for usage information");
            eprintln!("No command specified. Use --help for usage information");
            eprintln!("Examples:");
            eprintln!("  strapdown-sim --config my_config.toml");
            eprintln!("  strapdown-sim create-config --output my_config.toml");
            eprintln!("  strapdown-sim closed-loop -i input.csv -o output.csv --filter ukf");
            std::process::exit(1);
        }
    }
}
