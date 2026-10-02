//! Score a `strapdown-sim` output CSV against a synthetic truth CSV.
//!
//! Run it with:
//!
//! ```bash
//! cargo run -p strapdown-core --example score_run -- output.csv truth.csv
//! ```
//!
//! `output.csv` is anything `strapdown-sim` writes (`dr`, `cl` or `pf`); `truth.csv` is the
//! exact trajectory `strapdown-sim syn --no-noise` writes for the same flags and seed. Both are
//! `NavigationResult` CSVs, so both load the same way.
//!
//! The scoring is [`strapdown::metrics::evaluate`], the same reduction the accuracy baseline in
//! `core/tests/perf_baseline.rs` gates on. Estimates are matched to truth rows by timestamp, so
//! the two files have to come from the same `syn` epoch; `syn` always starts at
//! 2025-01-01T00:00:00Z, which is what makes that hold.

use std::error::Error;

use strapdown::metrics::{MetricOptions, evaluate, truth_from_trajectory};
use strapdown::sim::NavigationResult;

/// Print one optional metric, or say it could not be computed for this run.
fn line(label: &str, value: Option<f64>, unit: &str) {
    let text = match value {
        Some(value) => format!("{label:<34} {value:>12.3} {unit}"),
        None => format!("{label:<34} {:>12} {unit}", "n/a"),
    };
    println!("{}", text.trim_end());
}

fn main() -> Result<(), Box<dyn Error>> {
    let arguments: Vec<String> = std::env::args().collect();
    let [_, estimate_path, truth_path] = arguments.as_slice() else {
        return Err("usage: score_run <estimate.csv> <truth.csv>".into());
    };

    // ANCHOR: score
    let estimates = NavigationResult::from_csv(estimate_path)?;
    let truth = truth_from_trajectory(&NavigationResult::from_csv(truth_path)?);

    // Skip the first 10 aligned samples (one second at 10 Hz) as initialisation transient.
    let mut options = MetricOptions::default();
    options.warmup_samples = 10;
    let metrics = evaluate(&estimates, &truth, options)?;
    // ANCHOR_END: score

    println!("{:<34} {:>12}", "aligned samples", metrics.sample_count);
    line("horizontal RMSE", metrics.horizontal_rmse_m, "m");
    line("horizontal CEP50", metrics.horizontal_cep50_m, "m");
    line("horizontal CEP95", metrics.horizontal_cep95_m, "m");
    line("horizontal max", metrics.horizontal_max_m, "m");
    line("vertical RMSE", metrics.vertical_rmse_m, "m");
    line(
        "horizontal velocity RMSE",
        metrics.velocity_horizontal_rmse_mps,
        "m/s",
    );
    line("yaw RMSE", metrics.yaw_rmse_deg, "deg");
    line("position NEES (consistent at 3)", metrics.nees_position, "");
    line(
        "3-sigma horizontal containment",
        metrics.containment_3sigma_horizontal,
        "",
    );

    Ok(())
}
