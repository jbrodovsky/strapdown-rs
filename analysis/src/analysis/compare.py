"""analysis.compare

Utility functions for comparing navigation solutions and generating statistical summaries.

This module provides utilities for:
- Computing haversine-based position errors between solutions and truth
- Calculating error statistics (RMSE, mean, median, std, min, max)
- Generating LaTeX-formatted tables for publication
- Saving detailed results to CSV

These tools are designed to analyze the performance of geophysical-aided
navigation (gravity, magnetic, combined) versus baseline degraded GNSS solutions.

Examples
--------
>>> from analysis.compare import compute_error_statistics, format_latex_table
>>> stats = compute_error_statistics(geo_errors)
>>> print(f"RMSE: {stats['rmse']:.2f} m")
"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from haversine import Unit, haversine_vector


def align_dataframes_on_common_index(*frames: pd.DataFrame) -> Tuple[pd.DataFrame, ...]:
    """Align multiple DataFrames to the intersection of their timestamp indexes."""
    if not frames:
        raise ValueError("At least one DataFrame is required for alignment.")

    sorted_frames = [frame.sort_index() for frame in frames]
    common_index = sorted_frames[0].index

    for frame in sorted_frames[1:]:
        common_index = common_index.intersection(frame.index)

    if len(common_index) == 0:
        raise ValueError("Input data do not share any timestamps.")

    aligned_frames = tuple(frame.loc[common_index].copy() for frame in sorted_frames)
    if any(frame.empty for frame in aligned_frames):
        raise ValueError("Input data are empty after timestamp alignment.")

    return aligned_frames


def align_navigation_to_reference(nav: pd.DataFrame, reference: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Align navigation and reference trajectories on their shared timestamps.

    The processed navigation outputs and reference logs should generally share a
    one-to-one timestamp index. This function makes that assumption explicit and
    gracefully trims either side to the common index when minor differences are
    present.
    """
    aligned_nav, aligned_reference = align_dataframes_on_common_index(nav, reference)
    return aligned_nav, aligned_reference


def compute_navigation_errors(
    nav: pd.DataFrame,
    reference: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Compute aligned 2D, vertical, and 3D navigation errors against reference."""
    aligned_nav, aligned_reference = align_navigation_to_reference(nav, reference)

    two_d_error = haversine_vector(
        aligned_reference[["latitude", "longitude"]].to_numpy(),
        aligned_nav[["latitude", "longitude"]].to_numpy(),
        Unit.METERS,
    )
    vertical_error = aligned_reference["altitude"].to_numpy() - aligned_nav["altitude"].to_numpy()
    three_d_error = np.sqrt(two_d_error**2 + vertical_error**2)

    return aligned_nav, aligned_reference, two_d_error, vertical_error, three_d_error


def compute_error_statistics(errors: np.ndarray) -> Dict[str, float]:
    """Compute summary statistics for an array of errors.

    Parameters
    ----------
    errors : np.ndarray
        Array of error values (typically in meters).

    Returns
    -------
    Dict[str, float]
        Dictionary containing:
        - rmse: Root mean squared error
        - mean: Arithmetic mean
        - median: Median value
        - std: Standard deviation
        - max: Maximum error
        - min: Minimum error
    """
    return {
        "rmse": float(np.sqrt(np.nanmean(errors**2))),
        "mean": float(np.nanmean(errors)),
        "median": float(np.nanmedian(errors)),
        "std": float(np.nanstd(errors)),
        "max": float(np.nanmax(errors)),
        "min": float(np.nanmin(errors)),
    }


def compute_improvement_statistics(geo_stats: Dict[str, float], baseline_stats: Dict[str, float]) -> Dict[str, float]:
    """Compute improvement statistics (geo - baseline, negative = improvement).

    Parameters
    ----------
    geo_stats : Dict[str, float]
        Statistics for geophysical-aided solution.
    baseline_stats : Dict[str, float]
        Statistics for baseline degraded solution.

    Returns
    -------
    Dict[str, float]
        Dictionary with rmse, mean, median differences.
    """
    return {
        "rmse": geo_stats["rmse"] - baseline_stats["rmse"],
        "mean": geo_stats["mean"] - baseline_stats["mean"],
        "median": geo_stats["median"] - baseline_stats["median"],
    }


def save_detailed_results_to_csv(results: List[Tuple[str, Dict, Dict, Dict]], filename: str | Path) -> None:
    """Save detailed comparison results to a CSV file.

    Parameters
    ----------
    results : List[Tuple[str, Dict, Dict, Dict]]
        List of tuples containing:
        (trajectory_name, geo_stats, degraded_stats, improvement_stats)
    filename : str | Path
        Output CSV filename or path.
    """
    rows = []
    for traj_name, geo_stats, degraded_stats, improvement_stats in results:
        row = {
            "trajectory": str(traj_name).replace(".csv", ""),
            "geo_rmse": geo_stats["rmse"],
            "geo_mean": geo_stats["mean"],
            "geo_median": geo_stats["median"],
            "geo_std": geo_stats["std"],
            "geo_max": geo_stats["max"],
            "geo_min": geo_stats["min"],
            "baseline_rmse": degraded_stats["rmse"],
            "baseline_mean": degraded_stats["mean"],
            "baseline_median": degraded_stats["median"],
            "baseline_std": degraded_stats["std"],
            "baseline_max": degraded_stats["max"],
            "baseline_min": degraded_stats["min"],
            "diff_rmse": improvement_stats["rmse"],
            "diff_mean": improvement_stats["mean"],
            "diff_median": improvement_stats["median"],
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False)


def format_latex_table(
    results: List[Tuple[str, Dict[str, float]]],
    title: str,
    label: str = "tab:comparison_results",
) -> str:
    """Format comparison results as a LaTeX table.

    Parameters
    ----------
    results : List[Tuple[str, Dict[str, float]]]
        List of (trajectory_name, improvement_stats) tuples where
        improvement_stats contains 'rmse', 'mean', and 'median' keys.
    title : str
        Caption/title for the table.
    label : str, optional
        LaTeX label for the table, by default "tab:comparison_results".

    Returns
    -------
    str
        LaTeX table as a string, ready for inclusion in a document.

    Notes
    -----
    The table includes per-trajectory rows plus summary statistics
    (mean, median, std) at the bottom.
    """
    lines = []
    lines.append("\\begin{table}[h]")
    lines.append("    \\centering")
    lines.append(f"    \\caption{{{title}}}")
    lines.append("    \\begin{tabular}{ || l || c c c || }")
    lines.append("    \\toprule")
    lines.append("    Trajectory Name & RMSE Diff (m) & Mean Diff (m) & Median Diff (m) \\\\")
    lines.append("    \\midrule")

    # Trajectory rows
    for traj_name, stats in results:
        # Clean up trajectory name (remove file extension, escape underscores)
        clean_name = str(traj_name).replace(".csv", "").replace("_", "\\_")
        lines.append(f"    {clean_name} & {stats['rmse']:.2f} & {stats['mean']:.2f} & {stats['median']:.2f} \\\\")

    # Calculate summary statistics
    rmse_diffs = [s["rmse"] for _, s in results]
    mean_diffs = [s["mean"] for _, s in results]
    median_diffs = [s["median"] for _, s in results]

    lines.append("    \\midrule")
    lines.append(f"    mean & {np.mean(rmse_diffs):.2f} & {np.mean(mean_diffs):.2f} & {np.mean(median_diffs):.2f} \\\\")
    lines.append(
        f"    median & {np.median(rmse_diffs):.2f} & {np.median(mean_diffs):.2f} & {np.median(median_diffs):.2f} \\\\"
    )
    lines.append(f"    std & {np.std(rmse_diffs):.2f} & {np.std(mean_diffs):.2f} & {np.std(median_diffs):.2f} \\\\")
    lines.append("    \\bottomrule")
    lines.append("    \\end{tabular}")
    lines.append(f"    \\label{{{label}}}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def print_summary_statistics(results: List[Tuple[str, Dict[str, float]]], label: str) -> None:
    """Print summary statistics for a set of comparison results.

    Parameters
    ----------
    results : List[Tuple[str, Dict[str, float]]]
        List of (trajectory_name, improvement_stats) tuples.
    label : str
        Descriptive label for the result set (e.g., "Gravity-aided").
    """
    if not results:
        print(f"\n{label}: No results to summarize.")
        return

    improvements = [r[1]["rmse"] for r in results]
    improved_count = sum(1 for x in improvements if x < 0)

    print(f"\n{label}:")
    print(f"  Trajectories improved (negative diff): {improved_count}/{len(improvements)}")
    print(f"  Mean RMSE difference: {np.mean(improvements):.2f} m")
    print(f"  Median RMSE difference: {np.median(improvements):.2f} m")
    print(f"  Best (most negative): {min(improvements):.2f} m")
    print(f"  Worst (most positive): {max(improvements):.2f} m")
