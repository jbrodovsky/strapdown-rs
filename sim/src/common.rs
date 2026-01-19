//! Common utility functions for simulation applications.
//!
//! This module contains shared utilities for CLI applications including:
//! - Logger initialization
//! - Path validation and file discovery
//! - User input prompts

use std::error::Error;
use std::io;
use std::path::{Path, PathBuf};

/// Initialize the logger with the specified configuration.
///
/// # Arguments
/// * `log_level` - Log level string (off, error, warn, info, debug, trace)
/// * `log_file` - Optional path to log file (logs to stderr if None)
///
/// # Errors
/// Returns an error if the log file cannot be opened or logger initialization fails.
pub fn init_logger(log_level: &str, log_file: Option<&PathBuf>) -> Result<(), Box<dyn Error>> {
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
        if let Some(parent) = log_path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }
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

/// Validate input path exists and is either a file or directory.
///
/// # Arguments
/// * `input` - Path to validate
///
/// # Errors
/// Returns an error if the path does not exist or is neither a file nor directory.
pub fn validate_input_path(input: &Path) -> Result<(), Box<dyn Error>> {
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

/// Get all CSV files from a path (either single file or all CSVs in directory).
///
/// # Arguments
/// * `input` - Path to a CSV file or directory containing CSV files
///
/// # Returns
/// A sorted vector of PathBuf for each CSV file found.
///
/// # Errors
/// Returns an error if:
/// - The input file is not a CSV
/// - No CSV files are found in the directory
/// - The path is neither a file nor directory
pub fn get_csv_files(input: &Path) -> Result<Vec<PathBuf>, Box<dyn Error>> {
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

/// Validate output path and create parent directories if needed.
///
/// If the output path does not exist, it will be created as a directory.
///
/// # Arguments
/// * `output` - Path to validate/create
///
/// # Errors
/// Returns an error if directory creation fails.
pub fn validate_output_path(output: &Path) -> Result<(), Box<dyn Error>> {
    if !output.exists() {
        std::fs::create_dir_all(output)?;
    }
    Ok(())
}

// ============================================================================
// User Input Utilities
// ============================================================================

/// Read a line from stdin, trimming whitespace and checking for quit command.
///
/// # Returns
/// - `None` if user enters empty input or presses Enter
/// - `Some(String)` with the trimmed input otherwise
///
/// # Panics
/// Exits the process if user enters 'q' or 'Q'.
pub fn read_user_input() -> Option<String> {
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

/// Prompt for configuration name with validation.
///
/// Continues prompting until a non-empty name is provided.
///
/// # Returns
/// The configuration filename entered by the user.
pub fn prompt_config_name() -> String {
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

/// Prompt for configuration file path with validation.
///
/// Continues prompting until a non-empty path is provided.
///
/// # Returns
/// The configuration file path entered by the user.
pub fn prompt_config_path() -> String {
    loop {
        println!("Please specify the output configuration file path (or 'q' to quit):");
        if let Some(input) = read_user_input() {
            return input;
        }
        println!("Error: Configuration path cannot be empty. Please try again.\n");
    }
}

/// Prompt for input CSV file or directory path with validation.
///
/// # Returns
/// The input path entered by the user.
pub fn prompt_input_path() -> String {
    loop {
        println!(
            "Please specify the input location, either a single CSV file or a directory containing them. ('q' to quit):"
        );
        if let Some(input) = read_user_input() {
            return input;
        }
        println!("Error: Input path cannot be empty. Please try again.\n");
    }
}

/// Prompt for output CSV file path with validation.
///
/// # Returns
/// The output path entered by the user.
pub fn prompt_output_path() -> String {
    loop {
        println!("Please specify the output location to save output data. ('q' to quit):");
        if let Some(input) = read_user_input() {
            return input;
        }
        println!("Error: Output path cannot be empty. Please try again.\n");
    }
}

/// Helper function to prompt for f64 with default value and range validation.
///
/// # Arguments
/// * `prompt_text` - Text to display to the user
/// * `default` - Default value if user presses Enter
/// * `min_val` - Minimum acceptable value
/// * `max_val` - Maximum acceptable value
///
/// # Returns
/// The validated f64 value.
pub fn prompt_f64_with_default(prompt_text: &str, default: f64, min_val: f64, max_val: f64) -> f64 {
    loop {
        println!(
            "{} (press Enter for {}, or 'q' to quit):",
            prompt_text, default
        );
        match read_user_input() {
            None => return default,
            Some(input) => match input.parse::<f64>() {
                Ok(val) if val >= min_val && val <= max_val => return val,
                Ok(_) => println!(
                    "Error: Value must be between {} and {}.\n",
                    min_val, max_val
                ),
                Err(_) => println!("Error: Please enter a valid number.\n"),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use tempfile::tempdir;

    #[test]
    fn test_validate_input_path_file() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.csv");
        File::create(&file_path).unwrap();

        assert!(validate_input_path(&file_path).is_ok());
    }

    #[test]
    fn test_validate_input_path_directory() {
        let dir = tempdir().unwrap();
        assert!(validate_input_path(dir.path()).is_ok());
    }

    #[test]
    fn test_validate_input_path_nonexistent() {
        let result = validate_input_path(Path::new("/nonexistent/path"));
        assert!(result.is_err());
    }

    #[test]
    fn test_get_csv_files_single_file() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.csv");
        File::create(&file_path).unwrap();

        let result = get_csv_files(&file_path).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], file_path);
    }

    #[test]
    fn test_get_csv_files_directory() {
        let dir = tempdir().unwrap();

        // Create multiple CSV files
        File::create(dir.path().join("a.csv")).unwrap();
        File::create(dir.path().join("b.csv")).unwrap();
        File::create(dir.path().join("c.txt")).unwrap(); // Non-CSV should be ignored

        let result = get_csv_files(dir.path()).unwrap();
        assert_eq!(result.len(), 2);
        // Should be sorted
        assert!(
            result[0].file_name().unwrap().to_str().unwrap()
                < result[1].file_name().unwrap().to_str().unwrap()
        );
    }

    #[test]
    fn test_get_csv_files_non_csv() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.txt");
        File::create(&file_path).unwrap();

        let result = get_csv_files(&file_path);
        assert!(result.is_err());
    }

    #[test]
    fn test_get_csv_files_empty_directory() {
        let dir = tempdir().unwrap();
        let result = get_csv_files(dir.path());
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_output_path_creates_directory() {
        let dir = tempdir().unwrap();
        let new_dir = dir.path().join("new_output_dir");

        assert!(!new_dir.exists());
        validate_output_path(&new_dir).unwrap();
        assert!(new_dir.exists());
    }

    #[test]
    fn test_validate_output_path_existing() {
        let dir = tempdir().unwrap();
        assert!(validate_output_path(dir.path()).is_ok());
    }
}
