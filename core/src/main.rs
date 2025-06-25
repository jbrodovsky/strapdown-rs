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
    // let mu = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    // println!("mu:\n{}", mu);
    // let p = DMatrix::from_row_slice(6, 6, &[
    //     1.0, 0.1, 0.2, 0.3, 0.4, 0.5,
    //     0.1, 1.0, 0.1, 0.2, 0.3, 0.4,
    //     0.2, 0.1, 1.0, 0.1, 0.2, 0.3,
    //     0.3, 0.2, 0.1, 1.0, 0.1, 0.2,
    //     0.4, 0.3, 0.2, 0.1, 1.0, 0.1,
    //     0.5, 0.4, 0.3, 0.2, 0.1, 1.0
    // ]);
    // println!("p:\n{}", p);
    // let alpha = 1e-3;
    // let beta = 2.0;
    // let kappa = 0.0;
    // let state_size = mu.len();
    // let lambda = alpha * alpha * (state_size as f64 + kappa) - state_size as f64;
    // let mut weights_mean = DVector::zeros(2 * state_size + 1);
    // let mut weights_cov = DVector::zeros(2 * state_size + 1);
    // weights_mean[0] = lambda / (state_size as f64 + lambda);
    // weights_cov[0] = lambda / (state_size as f64 + lambda) + (1.0 - alpha * alpha + beta);
    // for i in 1..(2 * state_size + 1) {
    //     let w = 1.0 / (2.0 * (state_size as f64 + lambda));
    //     weights_mean[i] = w;
    //     weights_cov[i] = w;
    // }
    // println!("Weights mean:\n{}", weights_mean);
    // println!("Weights cov:\n{}", weights_cov);
    // let mut sigma_points = DMatrix::<f64>::zeros(state_size, 2 * state_size + 1);
    // let sqrt_p = matrix_square_root(&p.clone());
    // println!("sqrt(p):\n{}", sqrt_p);
    // // Set the first sigma point to the mean
    // sigma_points.column_mut(0).copy_from(&mu);
    // for i in 0..state_size {
    //     sigma_points.column_mut(i + 1).copy_from(&(&mu + sqrt_p.column(i)));
    //     sigma_points.column_mut(i + 1 + state_size).copy_from(&(&mu - sqrt_p.column(i)));
    // }
    // println!("Sigma points:\n{}", &sigma_points);
    // println!("Updated mean:\n{}", &sigma_points * &weights_mean);
    // let mu_bar = sigma_points.clone() * &weights_mean;
    // //let diff = sigma_points - mu_bar;
    // // Calculate the difference from the mean
    // let mut p_bar = DMatrix::<f64>::zeros(state_size, state_size);
    // for i in 0..(2 * state_size + 1) {
    //     let diff = &(&sigma_points.column(i) - &mu_bar);
    //     p_bar += &(diff * diff.transpose()) * weights_cov[i];
    // }
    // //println!("Diff dimensions:\n nrows: {} ncols: {}", &diff.nrows(), &diff.ncols());
    // //let p_bar = (&diff * &diff.transpose());
    // println!("Covariance size: {} {}", p_bar.nrows(), p_bar.ncols());
    // println!("weights_cov size: {}", weights_cov.len());
    // println!("Updated covariance:\n{}", p_bar);

    // ---------------------------------------
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
        println!("Running in closed-loop mode with GPS interval: {:?}", args.gps_interval);
        
        results = closed_loop(&records, args.gps_interval);
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
