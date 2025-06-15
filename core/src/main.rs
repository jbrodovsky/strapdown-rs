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
}

fn write_results_to_csv(
    // TODO: make this public and move to sim.rs
    results: &[NavigationResult],
    output: &PathBuf,
) -> Result<(), Box<dyn Error>> {
    let mut wtr = WriterBuilder::new().from_path(output)?;
    for result in results {
        wtr.serialize(result)?;
    }
    wtr.flush()?;
    Ok(())
}
fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    // Read the input CSV file
    let mut rdr = ReaderBuilder::new().from_path(&args.input)?;
    // Validate the mode
    if args.mode != "open-loop" && args.mode != "closed-loop" {
        return Err("Invalid mode specified. Use 'open-loop' or 'closed-loop'.".into());
    }
    let records: Vec<TestDataRecord> = rdr.deserialize().collect::<Result<_, _>>()?;
    println!(
        "Read {} records from {}",
        records.len(),
        &args.input.display()
    );
    let results: Vec<NavigationResult>;
    if args.mode == "closed-loop" {
        results = closed_loop(&records);
        match write_results_to_csv(&results, &args.output) {
            Ok(_) => println!("Results written to {}", args.output.display()),
            Err(e) => eprintln!("Error writing results: {}", e),
        };
    } else if args.mode == "open-loop" {
        results = dead_reckoning(&records);
        match write_results_to_csv(&results, &args.output) {
            Ok(_) => println!("Results written to {}", args.output.display()),
            Err(e) => eprintln!("Error writing results: {}", e),
        };
    } else {
        return Err("Invalid mode specified. Use 'open-loop' or 'closed-loop'.".into());
    }
    Ok(())
}
