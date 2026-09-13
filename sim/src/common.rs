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
pub(crate) fn init_logger(
    log_level: &str,
    log_file: Option<&PathBuf>,
) -> Result<(), Box<dyn Error>> {
    use std::io::Write;

    let level = log_level.parse::<log::LevelFilter>().unwrap_or_else(|_| {
        eprintln!("Invalid log level '{log_level}', defaulting to 'info'");
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
pub(crate) fn validate_input_path(input: &Path) -> Result<(), Box<dyn Error>> {
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
pub(crate) fn get_csv_files(input: &Path) -> Result<Vec<PathBuf>, Box<dyn Error>> {
    if input.is_file() {
        if input.extension().and_then(|s| s.to_str()) != Some("csv") {
            return Err(format!("Input file '{}' is not a CSV file.", input.display()).into());
        }
        Ok(vec![input.to_path_buf()])
    } else if input.is_dir() {
        let mut csv_files: Vec<PathBuf> = std::fs::read_dir(input)?
            .filter_map(std::result::Result::ok)
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

/// Extensions that mark an `--output` value as naming a single result file.
///
/// `--output` is overloaded: the shipped example configs use both `output = "output.csv"`
/// (a file) and `output = "data/output"` (a directory to write per-input results into),
/// so the extension is what distinguishes them.
const OUTPUT_FILE_EXTENSIONS: [&str; 2] = ["csv", "parquet"];

/// Whether `output` names a single result file rather than a directory to write into.
///
/// An existing directory always wins over the extension, so a real directory that happens
/// to be named `results.csv` is still treated as a directory.
pub(crate) fn output_names_a_file(output: &Path) -> bool {
    if output.is_dir() {
        return false;
    }
    output
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| {
            OUTPUT_FILE_EXTENSIONS
                .iter()
                .any(|known| extension.eq_ignore_ascii_case(known))
        })
}

/// Create the directory that results for `output` will be written into.
///
/// When `output` names a file (see [`output_names_a_file`]) this creates its *parent*.
/// Creating `output` itself would turn `-o results.csv` into a directory named
/// `results.csv` and send the results somewhere else entirely.
///
/// # Arguments
/// * `output` - The user's `--output` value, either a result file or a directory
///
/// # Errors
/// Returns an error if directory creation fails.
pub(crate) fn validate_output_path(output: &Path) -> Result<(), Box<dyn Error>> {
    let directory = if output_names_a_file(output) {
        output.parent()
    } else {
        Some(output)
    };

    // A bare `-o results.csv` has an empty parent, which is the current directory and so
    // needs no creating.
    if let Some(directory) = directory
        && !directory.as_os_str().is_empty()
        && !directory.exists()
    {
        std::fs::create_dir_all(directory)?;
    }
    Ok(())
}

/// Resolve where the results for `input_file` are written, given the user's `--output`.
///
/// Two shapes are supported, matching what the shipped configs already use:
///
/// - `output` names a **file** (`-o results.csv`): a single input writes straight to it;
///   multiple inputs write `{output_stem}_{input_stem}.{ext}` beside it, which is the
///   naming the `--output` help text documents.
/// - `output` names a **directory** (`-o results/`): each input writes to its own file
///   name inside that directory.
///
/// # Errors
/// Returns an error if `input_file` has no file name, or if the resolved output path is
/// the input file itself.
pub(crate) fn resolve_output_path(
    output: &Path,
    input_file: &Path,
    is_multiple: bool,
) -> Result<PathBuf, Box<dyn Error>> {
    let input_name = input_file.file_name().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("Input file path '{}' has no filename", input_file.display()),
        )
    })?;

    let resolved = if output_names_a_file(output) {
        if is_multiple {
            let output_stem = output
                .file_stem()
                .unwrap_or_else(|| std::ffi::OsStr::new("output"));
            let input_stem = Path::new(input_name)
                .file_stem()
                .unwrap_or_else(|| std::ffi::OsStr::new("input"));
            let extension = output
                .extension()
                .unwrap_or_else(|| std::ffi::OsStr::new("csv"));

            let mut file_name = output_stem.to_os_string();
            file_name.push("_");
            file_name.push(input_stem);
            file_name.push(".");
            file_name.push(extension);

            match output.parent() {
                Some(parent) if !parent.as_os_str().is_empty() => parent.join(file_name),
                _ => PathBuf::from(file_name),
            }
        } else {
            output.to_path_buf()
        }
    } else {
        // `Path::join` replaces the base when its argument is absolute, so join the file
        // name rather than the input path -- joining the path is how results ended up
        // written over the input.
        output.join(input_name)
    };

    if paths_point_to_same_file(&resolved, input_file) {
        return Err(format!(
            "Refusing to write results to '{}': that is the input file. \
             Pass a different --output path.",
            resolved.display()
        )
        .into());
    }

    Ok(resolved)
}

/// Whether two paths name the same file on disk.
///
/// The output path usually does not exist yet, so this canonicalises what it can -- the
/// parent directory -- and compares that against the canonicalised input. Symlinks, `..`
/// segments and a relative-vs-absolute spelling of the same file all compare equal.
fn paths_point_to_same_file(left: &Path, right: &Path) -> bool {
    match (
        canonical_parent_and_name(left),
        canonical_parent_and_name(right),
    ) {
        (Some(left), Some(right)) => left == right,
        // A path whose parent does not exist cannot be an existing input file.
        _ => false,
    }
}

/// Canonicalise `path` by canonicalising its parent and re-attaching its file name.
///
/// Returns `None` when the parent does not exist or the path has no file name.
fn canonical_parent_and_name(path: &Path) -> Option<PathBuf> {
    let file_name = path.file_name()?;
    let parent = path.parent().unwrap_or_else(|| Path::new(""));
    let parent = if parent.as_os_str().is_empty() {
        Path::new(".")
    } else {
        parent
    };
    parent
        .canonicalize()
        .ok()
        .map(|parent| parent.join(file_name))
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
/// Returns `None` when stdin cannot be read, which is the normal case when the tool runs
/// non-interactively — piped input, a closed stdin, or a batch invocation. That used to
/// panic, so a CLI that also supports batch mode aborted rather than falling back.
///
/// # Panics
/// Exits the process if user enters 'q' or 'Q'.
pub(crate) fn read_user_input() -> Option<String> {
    let mut input = String::new();
    if let Err(e) = io::stdin().read_line(&mut input) {
        log::warn!("could not read from stdin: {e}");
        return None;
    }
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
pub(crate) fn prompt_config_name() -> String {
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
pub(crate) fn prompt_config_path() -> String {
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
pub(crate) fn prompt_input_path() -> String {
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
pub(crate) fn prompt_output_path() -> String {
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
pub(crate) fn prompt_f64_with_default(
    prompt_text: &str,
    default: f64,
    min_val: f64,
    max_val: f64,
) -> f64 {
    loop {
        println!("{prompt_text} (press Enter for {default}, or 'q' to quit):");
        match read_user_input() {
            None => return default,
            Some(input) => match input.parse::<f64>() {
                Ok(val) if val >= min_val && val <= max_val => return val,
                Ok(_) => println!("Error: Value must be between {min_val} and {max_val}.\n"),
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

    // ------------------------------------------------------------------
    // Regression tests for #298: results were written over the input CSV
    // ------------------------------------------------------------------

    #[test]
    fn test_validate_output_path_does_not_create_a_file_output_as_a_directory() {
        // `-o results.csv` used to create a *directory* named `results.csv`, sending the
        // results somewhere else -- in #298, on top of the input file.
        let dir = tempdir().unwrap();
        let output = dir.path().join("nested").join("results.csv");

        validate_output_path(&output).unwrap();

        assert!(
            !output.exists(),
            "the output file path must not be created as a directory"
        );
        assert!(
            output.parent().unwrap().is_dir(),
            "the parent directory must be created"
        );
    }

    #[test]
    fn test_validate_output_path_accepts_a_bare_file_name() {
        // An empty parent is the current directory and needs no creating.
        assert!(validate_output_path(Path::new("results.csv")).is_ok());
        assert!(!Path::new("results.csv").exists());
    }

    #[test]
    fn test_resolve_output_path_single_file_output_writes_exactly_there() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("traj.csv");
        File::create(&input).unwrap();
        let output = dir.path().join("results.csv");

        let resolved = resolve_output_path(&output, &input, false).unwrap();

        assert_eq!(resolved, output);
    }

    #[test]
    fn test_resolve_output_path_directory_output_joins_the_file_name() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("traj.csv");
        File::create(&input).unwrap();
        let output_dir = dir.path().join("out");
        std::fs::create_dir_all(&output_dir).unwrap();

        let resolved = resolve_output_path(&output_dir, &input, false).unwrap();

        assert_eq!(resolved, output_dir.join("traj.csv"));
    }

    #[test]
    fn test_resolve_output_path_does_not_escape_the_output_directory() {
        // The #298 bug: `get_csv_files` returns absolute paths and `Path::join` replaces
        // the base when its argument is absolute, so joining the input *path* resolved to
        // the input itself. Joining the file name is what keeps the result inside `-o`.
        let dir = tempdir().unwrap();
        let input_dir = dir.path().join("in");
        std::fs::create_dir_all(&input_dir).unwrap();
        let input = input_dir.join("traj.csv");
        File::create(&input).unwrap();
        assert!(input.is_absolute());

        let output_dir = dir.path().join("out");
        std::fs::create_dir_all(&output_dir).unwrap();

        let resolved = resolve_output_path(&output_dir, &input, false).unwrap();

        assert_eq!(resolved, output_dir.join("traj.csv"));
        assert_ne!(resolved, input);
    }

    #[test]
    fn test_resolve_output_path_multiple_files_use_the_documented_stem_naming() {
        // `--output` documents: {output_stem}_{input_stem}.csv
        let dir = tempdir().unwrap();
        let input = dir.path().join("drive_a.csv");
        File::create(&input).unwrap();
        let output = dir.path().join("results.csv");

        let resolved = resolve_output_path(&output, &input, true).unwrap();

        assert_eq!(resolved, dir.path().join("results_drive_a.csv"));
    }

    #[test]
    fn test_resolve_output_path_refuses_to_overwrite_the_input() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("traj.csv");
        File::create(&input).unwrap();

        let error = resolve_output_path(&input, &input, false)
            .expect_err("writing results over the input must be refused");

        assert!(
            error.to_string().contains("that is the input file"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn test_resolve_output_path_refuses_the_input_spelled_differently() {
        // Same file reached through `.`, which a direct path comparison would miss.
        let dir = tempdir().unwrap();
        let input = dir.path().join("traj.csv");
        File::create(&input).unwrap();
        let disguised = dir.path().join(".").join("traj.csv");

        let error = resolve_output_path(&disguised, &input, false)
            .expect_err("writing results over the input must be refused");

        assert!(
            error.to_string().contains("that is the input file"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn test_output_names_a_file_prefers_an_existing_directory() {
        let dir = tempdir().unwrap();
        let oddly_named = dir.path().join("results.csv");
        std::fs::create_dir_all(&oddly_named).unwrap();

        assert!(!output_names_a_file(&oddly_named));
        assert!(output_names_a_file(
            &dir.path().join("results.csv").join("x.csv")
        ));
        assert!(!output_names_a_file(&dir.path().join("out")));
    }
}
