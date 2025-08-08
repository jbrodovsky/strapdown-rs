use clap::Parser;
use csv::ReaderBuilder;
use std::error::Error;
use std::path::PathBuf;
use strapdown::sim::{NavigationResult, TestDataRecord, closed_loop, dead_reckoning, degrade_measurements};

/// Command line arguments
#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// Mode of operation, either open-loop or closed-loop
    #[clap(short, long, value_parser, default_value = "open-loop", help = "Mode of operation: 'open-loop' or 'closed-loop'")]
    mode: String,
    /// Input CSV file path
    #[clap(short, long, value_parser, help = "Path to the input CSV file containing IMU and (optionally) GPS data")]
    input: PathBuf,
    /// Output CSV file path
    #[clap(short, long, value_parser, help = "Path to the output CSV file for navigation results")]
    output: PathBuf,
    /// Optional: GPS measurement interval in seconds (for simulating intermittent GPS outages)
    #[clap(long, value_parser, help = "Interval (in seconds) between GPS measurements; used to simulate GPS outages")]
    gps_interval: Option<f64>,
    /// Optional: GPS degradation factor
    #[clap(long, value_parser, default_value = "1.0", help = "Factor by which GPS accuracy is degraded. 1.0 is no degradation. >= 1.0 is a degradation factor. This factor is applied as a scalar multiplier to the recorded GPS accuracy.")]
    gps_degradation: Option<f64>,
    /// Optional: GPS spoofing offset
    #[clap(long, value_parser, default_value = "0.0", help = "Offset to apply to GPS coordinates (in meters)")]
    gps_spoofing_offset: Option<f64>,
}
fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    // Validate the mode
    if args.mode != "open-loop" && args.mode != "closed-loop" {
        return Err("Invalid mode specified. Use 'open-loop' or 'closed-loop'.".into());
    }
    // Read the input CSV file
    let mut rdr = ReaderBuilder::new().from_path(&args.input)?;
    let mut records: Vec<TestDataRecord> = rdr.deserialize().collect::<Result<_, _>>()?;
    println!(
        "Read {} records from {}",
        records.len(),
        &args.input.display()
    );
    records = match args.gps_degradation {
        Some(value) => degrade_measurements(records, value),
        None => records,
    };
    let results: Vec<NavigationResult>;
    if args.mode == "closed-loop" {
        println!(
            "Running in closed-loop mode with GPS interval: {:?}",
            args.gps_interval
        );
        results = closed_loop(&records, args.gps_interval);
        //match write_results_to_csv(&results, &args.output) {
        match NavigationResult::to_csv(&results, &args.output) {
            Ok(_) => println!("Results written to {}", args.output.display()),
            Err(e) => eprintln!("Error writing results: {}", e),
        };
    } else if args.mode == "open-loop" {
        results = dead_reckoning(&records);
        // match write_results_to_csv(&results, &args.output) {
        match NavigationResult::to_csv(&results, &args.output) {
            Ok(_) => println!("Results written to {}", args.output.display()),
            Err(e) => eprintln!("Error writing results: {}", e),
        };
    } else {
        return Err("Invalid mode specified. Use 'open-loop' or 'closed-loop'.".into());
    }
    Ok(())
}
