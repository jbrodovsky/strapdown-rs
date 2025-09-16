use clap::{Args, Parser, Subcommand, ValueEnum};
use std::error::Error;
use std::path::PathBuf;
use strapdown::messages::{
    GnssDegradationConfig, GnssFaultModel, GnssScheduler, build_event_stream,
};
use strapdown::sim::{NavigationResult, TestDataRecord, closed_loop, initialize_ukf};

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

// Example usage (also displayed via --help):
//
// Use a config file instead of CLI flags:
//   strapdown-sim closed-loop --config examples/configs/gnss_degradation.yaml -i data/input/2025-06-11_20-34-24.csv -o out.csv
//
// Or supply individual flags as before; the config file, if provided, takes precedence.
// Add a short note about the new --config flag in the CLI help
// (kept outside of LONG_ABOUT to avoid editing the long block literal heavily)

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

// Scheduler group
#[derive(Copy, Clone, Debug, ValueEnum)]
enum SchedKind {
    Passthrough,
    Fixed,
    Duty,
}

#[derive(Args, Clone, Debug)]
struct SchedulerArgs {
    /// Scheduler kind: passthrough | fixed | duty
    #[arg(long, value_enum, default_value_t = SchedKind::Passthrough)]
    sched: SchedKind,

    /// Fixed-interval seconds (sched=fixed)
    #[arg(long, default_value_t = 1.0)]
    interval_s: f64,

    /// Initial phase seconds (sched=fixed)
    #[arg(long, default_value_t = 0.0)]
    phase_s: f64,

    /// Duty-cycle ON seconds (sched=duty)
    #[arg(long, default_value_t = 10.0)]
    on_s: f64,

    /// Duty-cycle OFF seconds (sched=duty)
    #[arg(long, default_value_t = 10.0)]
    off_s: f64,

    /// Duty-cycle start phase seconds (sched=duty)
    #[arg(long, default_value_t = 0.0)]
    duty_phase_s: f64,
}
/// Fault group
#[derive(Args, Clone, Debug)]
struct FaultArgs {
    /// Fault kind: none | degraded | slowbias | hijack
    #[arg(long, value_enum, default_value_t = FaultKind::None)]
    fault: FaultKind,

    /// Degraded (AR(1))
    #[arg(long, default_value_t = 0.99)]
    rho_pos: f64,
    #[arg(long, default_value_t = 3.0)]
    sigma_pos_m: f64,
    #[arg(long, default_value_t = 0.95)]
    rho_vel: f64,
    #[arg(long, default_value_t = 0.3)]
    sigma_vel_mps: f64,
    #[arg(long, default_value_t = 5.0)]
    r_scale: f64,

    /// Slow bias
    #[arg(long, default_value_t = 0.02)]
    drift_n_mps: f64,
    #[arg(long, default_value_t = 0.0)]
    drift_e_mps: f64,
    #[arg(long, default_value_t = 1e-6)]
    q_bias: f64,
    #[arg(long, default_value_t = 0.0)]
    rotate_omega_rps: f64,

    /// Hijack
    #[arg(long, default_value_t = 50.0)]
    hijack_offset_n_m: f64,
    #[arg(long, default_value_t = 0.0)]
    hijack_offset_e_m: f64,
    #[arg(long, default_value_t = 120.0)]
    hijack_start_s: f64,
    #[arg(long, default_value_t = 60.0)]
    hijack_duration_s: f64,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum FaultKind {
    None,
    Degraded,
    Slowbias,
    Hijack,
}

/* -------------------- BUILDERS FROM GROUPS -------------------- */

fn build_scheduler(a: &SchedulerArgs) -> GnssScheduler {
    match a.sched {
        SchedKind::Passthrough => GnssScheduler::PassThrough,
        SchedKind::Fixed => GnssScheduler::FixedInterval {
            interval_s: a.interval_s,
            phase_s: a.phase_s,
        },
        SchedKind::Duty => GnssScheduler::DutyCycle {
            on_s: a.on_s,
            off_s: a.off_s,
            start_phase_s: a.duty_phase_s,
        },
    }
}

fn build_fault(a: &FaultArgs) -> GnssFaultModel {
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
            let mut ukf = initialize_ukf(records[0].clone(), None, None);
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
    // records = match args.gps_degradation {
    //     Some(value) => degrade_measurements(records, value),
    //     None => records,
    // };
    // let results: Vec<NavigationResult>;
    // if args.mode == "closed-loop" {
    //     println!(
    //         "Running in closed-loop mode with GPS interval: {:?}",
    //         args.gps_interval
    //     );
    //     results = closed_loop(&records, args.gps_interval);
    //     //match write_results_to_csv(&results, &args.output) {
    //     match NavigationResult::to_csv(&results, &args.output) {
    //         Ok(_) => println!("Results written to {}", args.output.display()),
    //         Err(e) => eprintln!("Error writing results: {}", e),
    //     };
    // } else if args.mode == "open-loop" {
    //     results = dead_reckoning(&records);
    //     // match write_results_to_csv(&results, &args.output) {
    //     match NavigationResult::to_csv(&results, &args.output) {
    //         Ok(_) => println!("Results written to {}", args.output.display()),
    //         Err(e) => eprintln!("Error writing results: {}", e),
    //     };
    // } else {
    //     return Err("Invalid mode specified. Use 'open-loop' or 'closed-loop'.".into());
    // }
    Ok(())
}
