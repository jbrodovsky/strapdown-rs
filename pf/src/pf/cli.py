"""
Command-line interface for the geophysical particle filter.

This module provides a CLI for running the velocity-based geophysical particle
filter on INS and IMU/GNSS data files.

Usage:
    pf-sim --ins-dir <path> --imu-dir <path> --output-dir <path> --geo-type <type>

Example:
    pf-sim --ins-dir ./data/ins --imu-dir ./data/raw \
           --output-dir ./results --geo-type gravity \
           --num-particles 1000 --seed 42
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from pf.geopf import (
    AveragingStrategy,
    GeoMap,
    GeophysicalMeasurement,
    GeophysicalMeasurementType,
    GeophysicalParticleFilter,
    NavigationState,
    ParticleFilterConfig,
    ResamplingStrategy,
    build_pf_dataset,
    compute_position_error,
)

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


@dataclass
class SimulationConfig:
    """Configuration for a particle filter simulation run.

    Attributes:
        ins_dir: Directory containing INS solution CSV files.
        imu_dir: Directory containing IMU/GNSS CSV files.
        output_dir: Directory for output results.
        geo_type: Type of geophysical measurement to use.
        map_region: Region bounds for PyGMT map loading (west, east, south, north).
        gravity_resolution: Resolution for gravity maps (default: "01m" = 1 arc-minute).
        magnetic_resolution: Resolution for magnetic maps (default: "03m" = 3 arc-minutes).
        num_particles: Number of particles in the filter.
        gravity_noise_std: Measurement noise for gravity (mGal).
        magnetic_noise_std: Measurement noise for magnetic (nT).
        position_jitter_std: Process noise for position (degrees).
        altitude_jitter_std: Process noise for altitude (meters).
        resampling_strategy: Strategy for particle resampling.
        averaging_strategy: Strategy for state estimation.
        effective_sample_threshold: Threshold for resampling trigger.
        seed: Random seed for reproducibility.
        verbose: Enable verbose logging.
    """

    ins_dir: Path
    imu_dir: Path
    output_dir: Path
    geo_type: GeophysicalMeasurementType
    map_region: tuple[float, float, float, float] | None = None
    gravity_resolution: str = "01m"
    magnetic_resolution: str = "03m"
    num_particles: int = 1000
    gravity_noise_std: float = 100.0
    magnetic_noise_std: float = 150.0
    position_jitter_std: float = 0.0001
    altitude_jitter_std: float = 5.0
    resampling_strategy: ResamplingStrategy = ResamplingStrategy.SYSTEMATIC
    averaging_strategy: AveragingStrategy = AveragingStrategy.WEIGHTED_MEAN
    effective_sample_threshold: float = 0.5
    seed: int | None = None
    verbose: bool = False
    postprocess: bool = False
    generate_plots: bool = True
    generate_maps: bool = True


def load_ins_data(filepath: Path) -> pd.DataFrame:
    """Load INS navigation solution from CSV file.

    Expected columns: timestamp, latitude, longitude, altitude,
                     velocity_north, velocity_east, velocity_vertical

    Args:
        filepath: Path to INS CSV file.

    Returns:
        DataFrame with INS navigation data.
    """
    logger.info(f"Loading INS data from {filepath}")
    df = pd.read_csv(filepath)

    # Ensure required columns exist
    required_cols = [
        "timestamp",
        "latitude",
        "longitude",
        "altitude",
        "velocity_north",
        "velocity_east",
        "velocity_vertical",
    ]

    # Handle alternative column names
    column_mapping = {
        "velocity_n": "velocity_north",
        "velocity_e": "velocity_east",
        "velocity_v": "velocity_vertical",
        "velocity_d": "velocity_vertical",  # NED convention
        "v_n": "velocity_north",
        "v_e": "velocity_east",
        "v_d": "velocity_vertical",
        "lat": "latitude",
        "lon": "longitude",
        "alt": "altitude",
        "time": "timestamp",
    }

    df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        msg = f"Missing required columns in INS data: {missing}"
        raise ValueError(msg)

    # Parse timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    return df


def load_imu_data(filepath: Path) -> pd.DataFrame:
    """Load IMU/GNSS raw data from CSV file.

    Expected columns: time, latitude, longitude, altitude, speed, bearing,
                     grav_x, grav_y, grav_z, mag_x, mag_y, mag_z

    Args:
        filepath: Path to IMU CSV file.

    Returns:
        DataFrame with IMU/GNSS data.
    """
    logger.info(f"Loading IMU data from {filepath}")
    df = pd.read_csv(filepath)

    # Handle common column name variations
    # First handle timestamp column
    if "timestamp" in df.columns and "time" not in df.columns:
        df = df.rename(columns={"timestamp": "time"})

    # Only rename acc_* to grav_* if grav_* columns don't already exist
    # This avoids creating duplicate columns when both acc_* and grav_* are present
    acc_to_grav_mapping = {
        "acc_x": "grav_x",
        "acc_y": "grav_y",
        "acc_z": "grav_z",
    }
    rename_map = {
        k: v for k, v in acc_to_grav_mapping.items() if k in df.columns and v not in df.columns
    }
    if rename_map:
        df = df.rename(columns=rename_map)

    # Parse timestamp
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])

    return df


def load_maps(
    config: SimulationConfig,
    data_bounds: tuple[float, float, float, float] | None = None,
) -> tuple[GeoMap | None, GeoMap | None]:
    """Load geophysical maps from PyGMT datasets based on configuration.

    Maps are loaded from PyGMT's earth datasets:
    - Gravity: Earth free-air anomaly at specified resolution (default: 1 arc-minute)
    - Magnetic: World Digital Magnetic Anomaly Map at specified resolution (default: 3 arc-minutes)

    Args:
        config: Simulation configuration.
        data_bounds: Optional (min_lat, max_lat, min_lon, max_lon) from data.

    Returns:
        Tuple of (gravity_map, magnetic_map), either may be None.
    """
    gravity_map = None
    magnetic_map = None

    # Determine region for map loading (west, east, south, north)
    if config.map_region is not None:
        region = config.map_region
    elif data_bounds is not None:
        # Add padding around data bounds
        padding = 0.1  # degrees
        min_lat, max_lat, min_lon, max_lon = data_bounds
        region = (
            min_lon - padding,
            max_lon + padding,
            min_lat - padding,
            max_lat + padding,
        )
    else:
        region = None

    # Load gravity map if needed
    if config.geo_type in (
        GeophysicalMeasurementType.GRAVITY,
        GeophysicalMeasurementType.COMBINED,
    ):
        if region is None:
            msg = "Cannot determine region for gravity map loading. Provide --map-region or ensure data has valid bounds."
            raise ValueError(msg)
        logger.info(
            f"Loading gravity map from PyGMT (resolution={config.gravity_resolution}, region={region})"
        )
        gravity_map = GeoMap.load_gravity(region=region, resolution=config.gravity_resolution)
        logger.info(f"Loaded gravity map: {gravity_map}")

    # Load magnetic map if needed
    if config.geo_type in (
        GeophysicalMeasurementType.MAGNETIC,
        GeophysicalMeasurementType.COMBINED,
    ):
        if region is None:
            msg = "Cannot determine region for magnetic map loading. Provide --map-region or ensure data has valid bounds."
            raise ValueError(msg)
        logger.info(
            f"Loading magnetic map from PyGMT (resolution={config.magnetic_resolution}, region={region}, data_source=wdmam)"
        )
        magnetic_map = GeoMap.load_magnetic(
            region=region, resolution=config.magnetic_resolution, data_source="wdmam"
        )
        logger.info(f"Loaded magnetic map: {magnetic_map}")

    return gravity_map, magnetic_map


def get_data_bounds(df: pd.DataFrame) -> tuple[float, float, float, float]:
    """Extract geographic bounds from a DataFrame with lat/lon columns.

    Args:
        df: DataFrame with 'latitude' and 'longitude' columns.

    Returns:
        Tuple of (min_lat, max_lat, min_lon, max_lon).
    """
    return (
        df["latitude"].min(),
        df["latitude"].max(),
        df["longitude"].min(),
        df["longitude"].max(),
    )


def run_particle_filter(
    config: SimulationConfig,
    combined_df: pd.DataFrame,
    gravity_map: GeoMap | None,
    magnetic_map: GeoMap | None,
) -> pd.DataFrame:
    """Run the particle filter on the combined dataset.

    Args:
        config: Simulation configuration.
        combined_df: Combined INS and IMU data.
        gravity_map: Gravity anomaly map (optional).
        magnetic_map: Magnetic anomaly map (optional).

    Returns:
        DataFrame with filter results.
    """
    # Create filter configuration
    pf_config = ParticleFilterConfig(
        num_particles=config.num_particles,
        measurement_type=config.geo_type,
        resampling_strategy=config.resampling_strategy,
        averaging_strategy=config.averaging_strategy,
        gravity_noise_std=config.gravity_noise_std,
        magnetic_noise_std=config.magnetic_noise_std,
        position_jitter_std=config.position_jitter_std,
        altitude_jitter_std=config.altitude_jitter_std,
        effective_sample_threshold=config.effective_sample_threshold,
        seed=config.seed,
    )

    # Create filter instance
    pf = GeophysicalParticleFilter(
        config=pf_config,
        gravity_map=gravity_map,
        magnetic_map=magnetic_map,
    )

    # Initialize at first position
    first_row = combined_df.iloc[0]
    initial_position = (
        first_row["latitude"],
        first_row["longitude"],
        first_row["altitude"],
    )
    pf.initialize(initial_position)
    logger.info(f"Initialized filter at {initial_position}")

    # Prepare results storage
    results = []

    # Run filter loop
    logger.info(f"Running particle filter on {len(combined_df)} samples")
    prev_timestamp = None

    for idx in tqdm(range(len(combined_df)), desc="Processing", disable=not config.verbose):
        row = combined_df.iloc[idx]
        timestamp = row["timestamp"]

        # Compute time step
        if prev_timestamp is not None:
            dt = (timestamp - prev_timestamp) / np.timedelta64(1, "s")
        else:
            dt = 0.1  # Default for first sample
        prev_timestamp = timestamp

        # Skip if dt is too small or negative
        if dt <= 0:
            continue

        # Create navigation state
        nav_state = NavigationState(
            timestamp=float(timestamp.value) / 1e9,  # Convert to Unix timestamp
            latitude=row["latitude"],
            longitude=row["longitude"],
            altitude=row["altitude"],
            velocity_north=row["velocity_north"],
            velocity_east=row["velocity_east"],
            velocity_vertical=row["velocity_vertical"],
        )

        # Create geophysical measurement
        grav_x = row.get("grav_x", np.nan)
        grav_y = row.get("grav_y", np.nan)
        grav_z = row.get("grav_z", np.nan)
        mag_x = row.get("mag_x", np.nan)
        mag_y = row.get("mag_y", np.nan)
        mag_z = row.get("mag_z", np.nan)

        measurement = GeophysicalMeasurement(
            timestamp=float(timestamp.value) / 1e9,
            gravity_x=grav_x if not pd.isna(grav_x) else np.nan,
            gravity_y=grav_y if not pd.isna(grav_y) else np.nan,
            gravity_z=grav_z if not pd.isna(grav_z) else np.nan,
            magnetic_x=mag_x if not pd.isna(mag_x) else np.nan,
            magnetic_y=mag_y if not pd.isna(mag_y) else np.nan,
            magnetic_z=mag_z if not pd.isna(mag_z) else np.nan,
        )

        # Run filter step
        est_lat, est_lon, est_alt = pf.step(nav_state, measurement, dt)
        cov = pf.get_covariance()
        n_eff = pf.get_effective_sample_size()

        # Compute errors if ground truth available
        horiz_err = vert_err = err_3d = np.nan
        if "latitude_truth" in row and not pd.isna(row["latitude_truth"]):
            truth = (row["latitude_truth"], row["longitude_truth"], row["altitude_truth"])
            horiz_err, vert_err, err_3d = compute_position_error((est_lat, est_lon, est_alt), truth)

        # Store result
        results.append(
            {
                "timestamp": timestamp,
                "latitude_est": est_lat,
                "longitude_est": est_lon,
                "altitude_est": est_alt,
                "latitude_ins": row["latitude"],
                "longitude_ins": row["longitude"],
                "altitude_ins": row["altitude"],
                "latitude_cov": cov[0, 0],
                "longitude_cov": cov[1, 1],
                "altitude_cov": cov[2, 2],
                "effective_sample_size": n_eff,
                "horizontal_error_m": horiz_err,
                "vertical_error_m": vert_err,
                "error_3d_m": err_3d,
            }
        )

    return pd.DataFrame(results)


def find_matching_files(ins_dir: Path, imu_dir: Path) -> list[tuple[Path, Path]]:
    """Find matching INS and IMU files by name pattern.

    Args:
        ins_dir: Directory containing INS CSV files.
        imu_dir: Directory containing IMU CSV files.

    Returns:
        List of (ins_file, imu_file) tuples.
    """
    ins_files = sorted(ins_dir.glob("*.csv"))
    imu_files = sorted(imu_dir.glob("*.csv"))

    if len(ins_files) == 1 and len(imu_files) == 1:
        return [(ins_files[0], imu_files[0])]

    # Try to match by name
    matches = []
    imu_file_map = {f.stem: f for f in imu_files}

    for ins_file in ins_files:
        # Try exact match or common prefixes
        stem = ins_file.stem
        for pattern in [stem, stem.replace("_nav", ""), stem.replace("_ins", "")]:
            if pattern in imu_file_map:
                matches.append((ins_file, imu_file_map[pattern]))
                break
            # Try with _raw or _imu suffix
            for suffix in ["_raw", "_imu", ""]:
                if pattern + suffix in imu_file_map:
                    matches.append((ins_file, imu_file_map[pattern + suffix]))
                    break

    return matches


def run_simulation(config: SimulationConfig) -> None:
    """Run the full particle filter simulation pipeline.

    Args:
        config: Simulation configuration.
    """
    # Set logging level
    if config.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Ensure output directory exists
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Find matching data files
    file_pairs = find_matching_files(config.ins_dir, config.imu_dir)
    if not file_pairs:
        logger.error("No matching INS/IMU file pairs found")
        sys.exit(1)

    logger.info(f"Found {len(file_pairs)} file pair(s) to process")

    # Process each file pair
    for ins_file, imu_file in file_pairs:
        logger.info(f"Processing: {ins_file.name} + {imu_file.name}")

        try:
            # Load data
            ins_df = load_ins_data(ins_file)
            imu_df = load_imu_data(imu_file)

            # Combine datasets
            combined_df = build_pf_dataset(ins_df, imu_df)
            logger.info(f"Combined dataset has {len(combined_df)} samples")

            if len(combined_df) == 0:
                logger.warning("No overlapping data found, skipping")
                continue

            # Get data bounds for map loading
            data_bounds = get_data_bounds(combined_df)
            logger.info(
                f"Data bounds: lat [{data_bounds[0]:.4f}, {data_bounds[1]:.4f}], "
                f"lon [{data_bounds[2]:.4f}, {data_bounds[3]:.4f}]"
            )

            # Load maps
            gravity_map, magnetic_map = load_maps(config, data_bounds)

            # Run particle filter
            results_df = run_particle_filter(config, combined_df, gravity_map, magnetic_map)

            # Save results
            output_name = ins_file.stem.replace("_nav", "").replace("_ins", "") + "_pf_results.csv"
            output_path = config.output_dir / output_name
            results_df.to_csv(output_path, index=False)
            logger.info(f"Results saved to {output_path}")

            # Print summary statistics
            if "horizontal_error_m" in results_df.columns:
                valid_errors = results_df["horizontal_error_m"].dropna()
                if len(valid_errors) > 0:
                    logger.info(
                        f"Horizontal error stats: "
                        f"mean={valid_errors.mean():.2f}m, "
                        f"median={valid_errors.median():.2f}m, "
                        f"max={valid_errors.max():.2f}m"
                    )

        except Exception as e:
            logger.exception(f"Error processing {ins_file.name}: {e}")
            continue


def parse_args() -> SimulationConfig:
    """Parse command-line arguments.

    Returns:
        SimulationConfig from parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Geophysical Particle Filter Simulation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--ins-dir",
        type=Path,
        required=True,
        help="Directory containing INS solution CSV files",
    )
    parser.add_argument(
        "--imu-dir",
        type=Path,
        required=True,
        help="Directory containing IMU/GNSS CSV files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for output results",
    )
    parser.add_argument(
        "--geo-type",
        type=str,
        choices=["gravity", "magnetic", "combined"],
        required=True,
        help="Type of geophysical measurement to use",
    )

    # Map options
    map_group = parser.add_argument_group("Map Options")
    map_group.add_argument(
        "--map-region",
        type=float,
        nargs=4,
        metavar=("WEST", "EAST", "SOUTH", "NORTH"),
        default=None,
        help="Region bounds for map loading (degrees). If not specified, derived from data with padding.",
    )
    map_group.add_argument(
        "--gravity-resolution",
        type=str,
        default="01m",
        help="Resolution for gravity maps (default: 01m = 1 arc-minute)",
    )
    map_group.add_argument(
        "--magnetic-resolution",
        type=str,
        default="03m",
        help="Resolution for magnetic maps (default: 03m = 3 arc-minutes)",
    )

    # Filter options
    filter_group = parser.add_argument_group("Filter Options")
    filter_group.add_argument(
        "--num-particles",
        type=int,
        default=1000,
        help="Number of particles",
    )
    filter_group.add_argument(
        "--gravity-noise-std",
        type=float,
        default=100.0,
        help="Gravity measurement noise std (mGal)",
    )
    filter_group.add_argument(
        "--magnetic-noise-std",
        type=float,
        default=150.0,
        help="Magnetic measurement noise std (nT)",
    )
    filter_group.add_argument(
        "--position-jitter-std",
        type=float,
        default=0.0001,
        help="Position process noise std (degrees)",
    )
    filter_group.add_argument(
        "--altitude-jitter-std",
        type=float,
        default=5.0,
        help="Altitude process noise std (meters)",
    )
    filter_group.add_argument(
        "--resampling-strategy",
        type=str,
        choices=["systematic", "stratified", "residual"],
        default="systematic",
        help="Resampling strategy",
    )
    filter_group.add_argument(
        "--averaging-strategy",
        type=str,
        choices=["mean", "weighted_mean", "map"],
        default="weighted_mean",
        help="State estimation strategy",
    )
    filter_group.add_argument(
        "--effective-sample-threshold",
        type=float,
        default=0.5,
        help="Threshold ratio for resampling trigger",
    )
    filter_group.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )

    # Post-processing options
    postprocess_group = parser.add_argument_group("Post-processing Options")
    postprocess_group.add_argument(
        "--postprocess",
        action="store_true",
        help="Run post-processing (plots and summary) after simulation",
    )
    postprocess_group.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating performance plots during post-processing",
    )
    postprocess_group.add_argument(
        "--no-maps",
        action="store_true",
        help="Skip generating trajectory maps during post-processing",
    )

    # Other options
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    # Convert geo_type string to enum
    geo_type_map = {
        "gravity": GeophysicalMeasurementType.GRAVITY,
        "magnetic": GeophysicalMeasurementType.MAGNETIC,
        "combined": GeophysicalMeasurementType.COMBINED,
    }

    # Convert strategy strings to enums
    resampling_map = {
        "systematic": ResamplingStrategy.SYSTEMATIC,
        "stratified": ResamplingStrategy.STRATIFIED,
        "residual": ResamplingStrategy.RESIDUAL,
    }
    averaging_map = {
        "mean": AveragingStrategy.MEAN,
        "weighted_mean": AveragingStrategy.WEIGHTED_MEAN,
        "map": AveragingStrategy.MAP,
    }

    return SimulationConfig(
        ins_dir=args.ins_dir,
        imu_dir=args.imu_dir,
        output_dir=args.output_dir,
        geo_type=geo_type_map[args.geo_type],
        map_region=tuple(args.map_region) if args.map_region else None,
        gravity_resolution=args.gravity_resolution,
        magnetic_resolution=args.magnetic_resolution,
        num_particles=args.num_particles,
        gravity_noise_std=args.gravity_noise_std,
        magnetic_noise_std=args.magnetic_noise_std,
        position_jitter_std=args.position_jitter_std,
        altitude_jitter_std=args.altitude_jitter_std,
        resampling_strategy=resampling_map[args.resampling_strategy],
        averaging_strategy=averaging_map[args.averaging_strategy],
        effective_sample_threshold=args.effective_sample_threshold,
        seed=args.seed,
        verbose=args.verbose,
        postprocess=args.postprocess,
        generate_plots=not args.no_plots,
        generate_maps=not args.no_maps,
    )


def main() -> None:
    """Main entry point for the CLI."""
    config = parse_args()
    run_simulation(config)

    # Run post-processing if requested
    if config.postprocess:
        from pf.postprocess import run_postprocessing

        logger.info("Running post-processing...")
        run_postprocessing(
            results_dir=config.output_dir,
            output_dir=config.output_dir / "analysis",
            generate_plots=config.generate_plots,
            generate_maps=config.generate_maps,
        )


def postprocess_main() -> None:
    """Standalone entry point for post-processing."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Post-process particle filter results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing PF result CSV files (*_pf_results.csv)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for plots and summary. Defaults to input-dir/analysis",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating performance plots",
    )
    parser.add_argument(
        "--no-maps",
        action="store_true",
        help="Skip generating trajectory maps",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    from pf.postprocess import run_postprocessing

    run_postprocessing(
        results_dir=args.input_dir,
        output_dir=args.output_dir,
        generate_plots=not args.no_plots,
        generate_maps=not args.no_maps,
    )


if __name__ == "__main__":
    main()
