"""Post-processing and visualization for particle filter results.

This module provides functions for analyzing and visualizing particle filter
navigation results, including:
- Performance time-series plots comparing PF estimates to truth/INS
- Trajectory visualization on OpenStreetMap backgrounds
- Summary statistics computation and aggregation
- Comparative analysis between PF and INS solutions

The module is designed to integrate with the pf-sim CLI and can also be
used standalone for custom analysis workflows.
"""

from __future__ import annotations

import logging
import os

# Ensure non-interactive backend for matplotlib
os.environ.setdefault("MPLBACKEND", "Agg")

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from haversine import Unit, haversine_vector
from matplotlib import pyplot as plt

logger = logging.getLogger(__name__)


@dataclass
class PerformanceStats:
    """Container for performance statistics.

    Attributes:
        dataset_name: Name of the dataset.
        num_samples: Number of samples processed.
        horizontal_error_min: Minimum horizontal error (m).
        horizontal_error_max: Maximum horizontal error (m).
        horizontal_error_mean: Mean horizontal error (m).
        horizontal_error_rmse: RMSE of horizontal error (m).
        vertical_error_min: Minimum vertical error (m).
        vertical_error_max: Maximum vertical error (m).
        vertical_error_mean: Mean vertical error (m).
        vertical_error_rmse: RMSE of vertical error (m).
        error_3d_min: Minimum 3D error (m).
        error_3d_max: Maximum 3D error (m).
        error_3d_mean: Mean 3D error (m).
        error_3d_rmse: RMSE of 3D error (m).
        mean_effective_sample_size: Mean effective sample size.
    """

    dataset_name: str
    num_samples: int
    horizontal_error_min: float
    horizontal_error_max: float
    horizontal_error_mean: float
    horizontal_error_rmse: float
    vertical_error_min: float
    vertical_error_max: float
    vertical_error_mean: float
    vertical_error_rmse: float
    error_3d_min: float
    error_3d_max: float
    error_3d_mean: float
    error_3d_rmse: float
    mean_effective_sample_size: float

    def to_dict(self) -> dict:
        """Convert to dictionary for DataFrame creation."""
        return {
            "Dataset": self.dataset_name,
            "Samples": self.num_samples,
            "Horiz Min (m)": self.horizontal_error_min,
            "Horiz Max (m)": self.horizontal_error_max,
            "Horiz Mean (m)": self.horizontal_error_mean,
            "Horiz RMSE (m)": self.horizontal_error_rmse,
            "Vert Min (m)": self.vertical_error_min,
            "Vert Max (m)": self.vertical_error_max,
            "Vert Mean (m)": self.vertical_error_mean,
            "Vert RMSE (m)": self.vertical_error_rmse,
            "3D Min (m)": self.error_3d_min,
            "3D Max (m)": self.error_3d_max,
            "3D Mean (m)": self.error_3d_mean,
            "3D RMSE (m)": self.error_3d_rmse,
            "Mean N_eff": self.mean_effective_sample_size,
        }


def compute_performance_stats(
    results_df: pd.DataFrame,
    dataset_name: str,
) -> PerformanceStats:
    """Compute performance statistics from particle filter results.

    Args:
        results_df: DataFrame with PF results containing columns:
            - latitude_est, longitude_est, altitude_est (PF estimates)
            - latitude_ins, longitude_ins, altitude_ins (INS/truth)
            - horizontal_error_m, vertical_error_m, error_3d_m (pre-computed errors)
            - effective_sample_size
        dataset_name: Name identifier for this dataset.

    Returns:
        PerformanceStats object with computed metrics.
    """
    # Use pre-computed errors if available, otherwise compute from positions
    if "horizontal_error_m" in results_df.columns:
        horiz_error = results_df["horizontal_error_m"].dropna().to_numpy()
        vert_error = results_df["vertical_error_m"].dropna().to_numpy()
        error_3d = results_df["error_3d_m"].dropna().to_numpy()
    else:
        # Compute errors from positions
        est_pos = results_df[["latitude_est", "longitude_est"]].to_numpy()
        truth_pos = results_df[["latitude_ins", "longitude_ins"]].to_numpy()

        horiz_error = haversine_vector(est_pos, truth_pos, Unit.METERS)
        vert_error = np.abs(
            results_df["altitude_est"].to_numpy() - results_df["altitude_ins"].to_numpy()
        )
        error_3d = np.sqrt(horiz_error**2 + vert_error**2)

    # Get effective sample size
    n_eff = results_df.get("effective_sample_size", pd.Series([np.nan]))

    return PerformanceStats(
        dataset_name=dataset_name,
        num_samples=len(results_df),
        horizontal_error_min=float(np.nanmin(horiz_error)),
        horizontal_error_max=float(np.nanmax(horiz_error)),
        horizontal_error_mean=float(np.nanmean(horiz_error)),
        horizontal_error_rmse=float(np.sqrt(np.nanmean(horiz_error**2))),
        vertical_error_min=float(np.nanmin(vert_error)),
        vertical_error_max=float(np.nanmax(vert_error)),
        vertical_error_mean=float(np.nanmean(vert_error)),
        vertical_error_rmse=float(np.sqrt(np.nanmean(vert_error**2))),
        error_3d_min=float(np.nanmin(error_3d)),
        error_3d_max=float(np.nanmax(error_3d)),
        error_3d_mean=float(np.nanmean(error_3d)),
        error_3d_rmse=float(np.sqrt(np.nanmean(error_3d**2))),
        mean_effective_sample_size=float(np.nanmean(n_eff)),
    )


def plot_pf_performance(
    results_df: pd.DataFrame,
    output_path: Path | str,
    title: str | None = None,
) -> None:
    """Plot particle filter performance time-series.

    Creates a multi-panel figure showing:
    - Horizontal error over time
    - Vertical error over time
    - Effective sample size over time

    Args:
        results_df: DataFrame with PF results.
        output_path: Path to save the plot PNG.
        title: Optional title for the plot.
    """
    output_path = Path(output_path)

    # Parse timestamps
    if "timestamp" in results_df.columns:
        timestamps = pd.to_datetime(results_df["timestamp"])
        time_seconds = (timestamps - timestamps.iloc[0]).dt.total_seconds().to_numpy()
    else:
        time_seconds = np.arange(len(results_df))

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Panel 1: Horizontal error
    ax1 = axes[0]
    if "horizontal_error_m" in results_df.columns:
        horiz_error = results_df["horizontal_error_m"].to_numpy()
    else:
        est_pos = results_df[["latitude_est", "longitude_est"]].to_numpy()
        truth_pos = results_df[["latitude_ins", "longitude_ins"]].to_numpy()
        horiz_error = haversine_vector(est_pos, truth_pos, Unit.METERS)

    ax1.plot(time_seconds, horiz_error, "b-", linewidth=0.8, label="Horizontal Error")
    ax1.axhline(
        np.nanmean(horiz_error),
        color="r",
        linestyle="--",
        label=f"Mean: {np.nanmean(horiz_error):.1f}m",
    )
    ax1.set_ylabel("Horizontal Error (m)")
    ax1.set_ylim(bottom=0)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right")

    # Panel 2: Vertical error
    ax2 = axes[1]
    if "vertical_error_m" in results_df.columns:
        vert_error = results_df["vertical_error_m"].to_numpy()
    else:
        vert_error = np.abs(
            results_df["altitude_est"].to_numpy() - results_df["altitude_ins"].to_numpy()
        )

    ax2.plot(time_seconds, vert_error, "g-", linewidth=0.8, label="Vertical Error")
    ax2.axhline(
        np.nanmean(vert_error),
        color="r",
        linestyle="--",
        label=f"Mean: {np.nanmean(vert_error):.1f}m",
    )
    ax2.set_ylabel("Vertical Error (m)")
    ax2.set_ylim(bottom=0)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right")

    # Panel 3: Effective sample size
    ax3 = axes[2]
    if "effective_sample_size" in results_df.columns:
        n_eff = results_df["effective_sample_size"].to_numpy()
        ax3.plot(time_seconds, n_eff, "m-", linewidth=0.8, label="Effective Sample Size")
        ax3.axhline(
            np.nanmean(n_eff), color="r", linestyle="--", label=f"Mean: {np.nanmean(n_eff):.0f}"
        )
        ax3.set_ylabel("Effective Sample Size")
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc="upper right")
    else:
        ax3.text(0.5, 0.5, "N_eff not available", ha="center", va="center", transform=ax3.transAxes)

    ax3.set_xlabel("Time (s)")

    # Title
    if title is None:
        title = "Particle Filter Performance"
    fig.suptitle(title, fontsize=14)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved performance plot to {output_path}")


def plot_pf_vs_ins(
    results_df: pd.DataFrame,
    output_path: Path | str,
    title: str | None = None,
) -> None:
    """Plot PF vs INS comparison.

    Creates a figure showing the error difference between PF and INS
    (when INS errors are available), highlighting where PF outperforms INS.

    Args:
        results_df: DataFrame with PF results.
        output_path: Path to save the plot PNG.
        title: Optional title for the plot.
    """
    output_path = Path(output_path)

    # Parse timestamps
    if "timestamp" in results_df.columns:
        timestamps = pd.to_datetime(results_df["timestamp"])
        time_hours = (timestamps - timestamps.iloc[0]).dt.total_seconds().to_numpy() / 3600
    else:
        time_hours = np.arange(len(results_df)) / 3600

    # Get PF horizontal error
    if "horizontal_error_m" in results_df.columns:
        pf_error = results_df["horizontal_error_m"].to_numpy()
    else:
        est_pos = results_df[["latitude_est", "longitude_est"]].to_numpy()
        truth_pos = results_df[["latitude_ins", "longitude_ins"]].to_numpy()
        pf_error = haversine_vector(est_pos, truth_pos, Unit.METERS)

    fig, ax = plt.subplots(1, 1, figsize=(14, 5))

    # Plot PF error
    ax.plot(time_hours, pf_error, "b-", linewidth=0.8, label="PF Error", alpha=0.8)

    # Add statistics annotation
    pf_rmse = np.sqrt(np.nanmean(pf_error**2))
    pf_mean = np.nanmean(pf_error)
    pf_median = np.nanmedian(pf_error)

    stats_text = f"RMSE: {pf_rmse:.1f}m | Mean: {pf_mean:.1f}m | Median: {pf_median:.1f}m"
    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
    )

    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Horizontal Error (m)")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    if title is None:
        title = "Particle Filter Horizontal Error"
    ax.set_title(title, fontsize=14)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved comparison plot to {output_path}")


def plot_trajectory_map(
    results_df: pd.DataFrame,
    output_path: Path | str,
    title: str | None = None,
    show_ins: bool = True,
    margin: float = 0.01,
) -> None:
    """Plot trajectory on OpenStreetMap background.

    Args:
        results_df: DataFrame with PF results.
        output_path: Path to save the plot PNG.
        title: Optional title for the plot.
        show_ins: Whether to show INS trajectory for comparison.
        margin: Margin around trajectory bounds (degrees).
    """
    try:
        from cartopy import crs as ccrs
        from cartopy.io import img_tiles as cimgt
    except ImportError:
        logger.warning("Cartopy not available, skipping trajectory map plot")
        return

    output_path = Path(output_path)

    # Extract coordinates
    pf_lat = results_df["latitude_est"].to_numpy()
    pf_lon = results_df["longitude_est"].to_numpy()

    # Clean NaN values
    valid_mask = np.isfinite(pf_lat) & np.isfinite(pf_lon)
    if not np.any(valid_mask):
        logger.warning("No valid PF coordinates to plot")
        return

    pf_lat_clean = pf_lat[valid_mask]
    pf_lon_clean = pf_lon[valid_mask]

    # Compute bounds
    lat_min, lat_max = float(np.min(pf_lat_clean)), float(np.max(pf_lat_clean))
    lon_min, lon_max = float(np.min(pf_lon_clean)), float(np.max(pf_lon_clean))

    # Create figure
    osm_tiles = cimgt.OSM()
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw={"projection": ccrs.PlateCarree()})
    ax.set_extent(
        [lon_min - margin, lon_max + margin, lat_min - margin, lat_max + margin],
        crs=ccrs.PlateCarree(),
    )

    # Add OSM tiles
    ax.add_image(osm_tiles, 12)

    # Plot PF trajectory
    ax.plot(
        pf_lon_clean,
        pf_lat_clean,
        "-",
        color="blue",
        linewidth=2,
        transform=ccrs.PlateCarree(),
        label="PF Estimate",
    )

    # Plot INS trajectory if available and requested
    if show_ins and "latitude_ins" in results_df.columns:
        ins_lat = results_df["latitude_ins"].to_numpy()
        ins_lon = results_df["longitude_ins"].to_numpy()
        ins_mask = np.isfinite(ins_lat) & np.isfinite(ins_lon)
        if np.any(ins_mask):
            ax.plot(
                ins_lon[ins_mask],
                ins_lat[ins_mask],
                "--",
                color="red",
                linewidth=1.5,
                transform=ccrs.PlateCarree(),
                label="INS/Truth",
            )

    # Add gridlines
    gl = ax.gridlines(draw_labels=True, alpha=0.2)
    gl.top_labels = False
    gl.right_labels = False

    if title is None:
        title = "Particle Filter Trajectory"
    ax.set_title(title, fontsize=14)
    ax.legend(loc="upper right")

    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved trajectory map to {output_path}")


def run_postprocessing(
    results_dir: Path | str,
    output_dir: Path | str | None = None,
    generate_plots: bool = True,
    generate_maps: bool = True,
) -> pd.DataFrame:
    """Run post-processing on all PF result files in a directory.

    Args:
        results_dir: Directory containing PF result CSV files.
        output_dir: Directory for output plots and summary. If None, uses results_dir.
        generate_plots: Whether to generate performance plots.
        generate_maps: Whether to generate trajectory maps.

    Returns:
        DataFrame with summary statistics for all datasets.
    """
    results_dir = Path(results_dir)
    output_dir = Path(output_dir) if output_dir else results_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all result files
    result_files = list(results_dir.glob("*_pf_results.csv"))
    if not result_files:
        logger.warning(f"No PF result files found in {results_dir}")
        return pd.DataFrame()

    logger.info(f"Found {len(result_files)} result files to process")

    # Process each file
    all_stats = []
    for result_file in result_files:
        dataset_name = result_file.stem.replace("_pf_results", "")
        logger.info(f"Processing {dataset_name}")

        try:
            results_df = pd.read_csv(result_file, parse_dates=["timestamp"])

            # Compute statistics
            stats = compute_performance_stats(results_df, dataset_name)
            all_stats.append(stats.to_dict())

            # Generate plots
            if generate_plots:
                perf_plot_path = output_dir / f"{dataset_name}_pf_performance.png"
                plot_pf_performance(
                    results_df, perf_plot_path, title=f"PF Performance: {dataset_name}"
                )

                error_plot_path = output_dir / f"{dataset_name}_pf_error.png"
                plot_pf_vs_ins(results_df, error_plot_path, title=f"PF Error: {dataset_name}")

            # Generate trajectory map
            if generate_maps:
                map_path = output_dir / f"{dataset_name}_pf_trajectory.png"
                plot_trajectory_map(results_df, map_path, title=f"Trajectory: {dataset_name}")

        except Exception as e:
            logger.exception(f"Error processing {dataset_name}: {e}")
            continue

    # Create summary DataFrame
    if all_stats:
        summary_df = pd.DataFrame(all_stats)
        summary_file = output_dir / "pf_performance_summary.csv"
        summary_df.to_csv(summary_file, index=False)
        logger.info(f"Saved performance summary to {summary_file}")

        # Print summary to console
        print("\n" + "=" * 80)
        print("PARTICLE FILTER PERFORMANCE SUMMARY")
        print("=" * 80)
        print(f"Datasets processed: {len(summary_df)}")
        print("\nHorizontal Error Statistics:")
        print(f"  Overall Mean:   {summary_df['Horiz Mean (m)'].mean():.1f} m")
        print(f"  Overall RMSE:   {summary_df['Horiz RMSE (m)'].mean():.1f} m")
        print(
            f"  Best Dataset:   {summary_df.loc[summary_df['Horiz RMSE (m)'].idxmin(), 'Dataset']} "
            f"({summary_df['Horiz RMSE (m)'].min():.1f} m RMSE)"
        )
        print(
            f"  Worst Dataset:  {summary_df.loc[summary_df['Horiz RMSE (m)'].idxmax(), 'Dataset']} "
            f"({summary_df['Horiz RMSE (m)'].max():.1f} m RMSE)"
        )
        print("=" * 80 + "\n")

        return summary_df

    return pd.DataFrame()
