use clap::{Args, Parser, Subcommand};
use std::error::Error;
use std::path::PathBuf;
use strapdown::messages::{GnssDegradationConfig, build_event_stream};
use strapdown::sim::{
    FaultArgs, NavigationResult, SchedulerArgs, TestDataRecord, build_fault, build_scheduler,
    closed_loop, initialize_ukf,
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
    /// Mode of operation, either open-loop or closed-loop
    #[command(subcommand)]
    mode: SimMode,
    /// Input file path
    #[arg(short, long, value_parser)]
    input: PathBuf,
    /// Output file path
    #[arg(short, long, value_parser)]
    output: PathBuf,
}
#[derive(Subcommand, Clone)]
enum SimMode {
    #[command(name = "open-loop", about = "Run the simulation in open-loop mode")]
    OpenLoop,
    #[command(name = "closed-loop", about = "Run the simulation in closed-loop mode")]
    ClosedLoop(ClosedLoopArgs),
}
/* -------------------- COMPARTMENTALIZED GROUPS -------------------- */
#[derive(Args, Clone, Debug)]
struct ClosedLoopArgs {
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

fn main() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();
    // Validate the mode
    //if args.mode != "open-loop" && args.mode != "closed-loop" {
    //    return Err("Invalid mode specified. Use 'open-loop' or 'closed-loop'.".into());
    //}
    // Read the input CSV file
    // Validate that the input file exists and is readable
    if !cli.input.exists() {
        return Err(format!("Input file '{}' does not exist.", cli.input.display()).into());
    }
    if !cli.input.is_file() {
        return Err(format!("Input path '{}' is not a file.", cli.input.display()).into());
    }
    // Validate that the output file is writable
    if let Some(parent) = cli.output.parent()
        && !parent.exists()
        && parent.is_dir()
    {
        return Err(format!("Output directory '{}' does not exist.", parent.display()).into());
    }
    // Validate that all directories in the output path exist, if they don't create them
    let parents = cli.output.parent();
    if let Some(parent) = parents {
        if !parent.exists() {
            std::fs::create_dir_all(parent)?;
        }
    }
    match cli.mode {
        SimMode::OpenLoop => println!("Running in open-loop mode"),
        SimMode::ClosedLoop(ref args) => {
            // Load sensor data records from CSV, tolerant to mixed/variable-length rows and encoding issues.
            let records = TestDataRecord::from_csv(&cli.input)?;
            println!(
                "Read {} records from {}",
                records.len(),
                &cli.input.display()
            );
            let cfg = if let Some(ref cfg_path) = args.config {
                match GnssDegradationConfig::from_file(cfg_path) {
                    Ok(c) => c,
                    Err(e) => {
                        eprintln!("Failed to read config {}: {}", cfg_path.display(), e);
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
            let mut ukf = initialize_ukf(records[0].clone(), None, None, None, None, None, None);
            let results = closed_loop(&mut ukf, events);
            //sim::write_results_csv(&cli.output, &results)?;
            match results {
                Ok(ref nav_results) => match NavigationResult::to_csv(nav_results, &cli.output) {
                    Ok(_) => println!("Results written to {}", cli.output.display()),
                    Err(e) => eprintln!("Error writing results: {}", e),
                },
                Err(e) => eprintln!("Error running closed-loop simulation: {}", e),
            };
        }
    }
    Ok(())
}
