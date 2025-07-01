use clap::Parser;
use csv::{ReaderBuilder, WriterBuilder};
use std::error::Error;
use std::path::PathBuf;
use strapdown::sim::{NavigationResult, TestDataRecord, closed_loop, dead_reckoning};

/// Command line arguments
#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// Mode of operation, either open-loop or closed-loop
    #[clap(short, long, value_parser, default_value = "open-loop")]
    mode: String,

    /// Input CSV file path
    #[clap(short, long, value_parser)]
    input: PathBuf,

    /// Output CSV file path
    #[clap(short, long, value_parser)]
    output: PathBuf,
    /// Optional: GPS measurement intervale in seconds (for simulating intermittent GPS outages)
    #[clap(long, value_parser)]
    gps_interval: Option<usize>,
}
fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    // Validate the mode
    if args.mode != "open-loop" && args.mode != "closed-loop" {
        return Err("Invalid mode specified. Use 'open-loop' or 'closed-loop'.".into());
    }
    // Read the input CSV file
    let mut rdr = ReaderBuilder::new().from_path(&args.input)?;
    let records: Vec<TestDataRecord> = rdr.deserialize().collect::<Result<_, _>>()?;
    println!(
        "Read {} records from {}",
        records.len(),
        &args.input.display()
    );
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
