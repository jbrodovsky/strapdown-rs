use clap::{Args, Parser, ValueEnum};
use std::error::Error;
use std::path::PathBuf;
use std::rc::Rc;

use geonav::{
    GeoMap, GeophysicalMeasurementType, GravityResolution, MagneticResolution, build_event_stream,
    geo_closed_loop,
};
use strapdown::messages::GnssDegradationConfig;
use strapdown::sim::{
    DEFAULT_PROCESS_NOISE, FaultArgs, NavigationResult, SchedulerArgs, TestDataRecord, build_fault,
    build_scheduler, initialize_ukf,
};

const LONG_ABOUT: &str = "GEONAV-SIM: A geophysical navigation simulation tool for strapdown inertial navigation systems.

This program extends the basic strapdown simulation by incorporating geophysical measurements such as gravity and magnetic anomalies for enhanced navigation accuracy. It loads geophysical maps (NetCDF format) and simulates how these measurements can aid inertial navigation systems, particularly in GNSS-denied environments.

The program operates in closed-loop mode, incorporating both GNSS measurements (when available) and geophysical measurements from loaded maps. It can simulate various GNSS degradation scenarios while maintaining navigation accuracy through geophysical aiding.

Input data format is identical to strapdown-sim, with additional geophysical map files:
* Input CSV: Standard IMU/GNSS data as per strapdown-sim specification
* Gravity maps: *_gravity.nc files containing gravity anomaly data
* Magnetic maps: *_magnetic.nc files containing magnetic anomaly data

The program automatically detects and loads the appropriate map file based on the measurement type specified and the input file directory.";

/// Command line arguments
#[derive(Parser)]
#[command(author, version, about, long_about = LONG_ABOUT)]
struct Cli {
    /// Input CSV file path
    #[arg(short, long, value_parser)]
    input: PathBuf,
    /// Output CSV file path
    #[arg(short, long, value_parser)]
    output: PathBuf,
    /// Geophysical measurement configuration
    #[command(flatten)]
    geo: GeophysicalArgs,
    /// GNSS degradation configuration
    #[command(flatten)]
    gnss: GnssArgs,
}

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

#[derive(Args, Clone, Debug)]
struct GnssArgs {
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

#[derive(Copy, Clone, Debug, ValueEnum)]
pub enum GeoMeasurementType {
    Gravity,
    Magnetic,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
pub enum GeoResolution {
    OneDegree,
    ThirtyMinutes,
    TwentyMinutes,
    FifteenMinutes,
    TenMinutes,
    SixMinutes,
    FiveMinutes,
    FourMinutes,
    ThreeMinutes,
    TwoMinutes,
    OneMinute,
    ThirtySeconds,
    FifteenSeconds,
    ThreeSeconds,
    OneSecond,
}

/// Convert CLI geo measurement type to library type
impl From<GeoMeasurementType> for GeophysicalMeasurementType {
    fn from(geo_type: GeoMeasurementType) -> Self {
        match geo_type {
            GeoMeasurementType::Gravity => {
                GeophysicalMeasurementType::Gravity(GravityResolution::OneMinute)
            }
            GeoMeasurementType::Magnetic => {
                GeophysicalMeasurementType::Magnetic(MagneticResolution::TwoMinutes)
            }
        }
    }
}

/// Convert CLI resolution to appropriate library resolution based on measurement type
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
    input_path: &PathBuf,
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

fn main() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();

    // Validate input file
    if !cli.input.exists() {
        return Err(format!("Input file '{}' does not exist.", cli.input.display()).into());
    }
    if !cli.input.is_file() {
        return Err(format!("Input path '{}' is not a file.", cli.input.display()).into());
    }

    // Ensure output directory exists
    if let Some(parent) = cli.output.parent() {
        if !parent.exists() {
            std::fs::create_dir_all(parent)?;
        }
    }

    // Load sensor data records from CSV
    let records = TestDataRecord::from_csv(&cli.input)?;
    println!(
        "Read {} records from {}",
        records.len(),
        cli.input.display()
    );

    // Determine map file path
    let map_path = match &cli.geo.map_file {
        Some(path) => path.clone(),
        None => find_map_file(&cli.input, cli.geo.geo_type)?,
    };

    println!("Loading geophysical map from: {}", map_path.display());

    // Build measurement type with specified resolution
    let measurement_type = build_measurement_type(cli.geo.geo_type, cli.geo.geo_resolution);

    // Load geophysical map
    let geomap = Rc::new(GeoMap::load_geomap(map_path, measurement_type)?);
    println!(
        "Loaded {} map with {} x {} grid points",
        geomap.get_map_type(),
        geomap.get_lats().len(),
        geomap.get_lons().len()
    );

    // Build GNSS degradation configuration
    let gnss_config = if let Some(ref cfg_path) = cli.gnss.config {
        match GnssDegradationConfig::from_file(cfg_path) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("Failed to read config {}: {}", cfg_path.display(), e);
                return Err(Box::new(e));
            }
        }
    } else {
        GnssDegradationConfig {
            scheduler: build_scheduler(&cli.gnss.scheduler),
            fault: build_fault(&cli.gnss.fault),
            seed: cli.gnss.seed,
        }
    };

    // Build event stream with geophysical measurements
    let events = build_event_stream(
        &records,
        &gnss_config,
        geomap,
        Some(cli.geo.geo_noise_std),
        cli.geo.geo_frequency_s,
    );
    println!("Built event stream with {} events", events.events.len());

    // Initialize UKF
    let mut process_noise: Vec<f64> = DEFAULT_PROCESS_NOISE.clone().into();
    process_noise.extend([1e-9]); // Extend for geophysical state

    let mut ukf = initialize_ukf(
        records[0].clone(),
        None,
        None,
        None,
        Some(vec![cli.geo.geo_bias; 1]),
        Some(vec![cli.geo.geo_noise_std; 1]),
        Some(process_noise),
    );
    println!(
        "Initialized UKF with state dimension {}",
        ukf.get_mean().len()
    );
    println!("Initial state: {:?}", ukf.get_mean());
    // Run closed-loop simulation
    println!("Running geophysical navigation simulation...");
    let results = geo_closed_loop(&mut ukf, events);

    // Write results
    match results {
        Ok(ref nav_results) => match NavigationResult::to_csv(nav_results, &cli.output) {
            Ok(_) => println!("Results written to {}", cli.output.display()),
            Err(e) => eprintln!("Error writing results: {}", e),
        },
        Err(e) => {
            eprintln!("Error running geophysical navigation simulation: {}", e);
            return Err(e.into());
        }
    }
    Ok(())
}
