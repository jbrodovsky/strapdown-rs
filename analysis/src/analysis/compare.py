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

import numpy as np
import pandas as pd


def compute_error_statistics(errors: np.ndarray) -> dict[str, float]:
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


def compute_improvement_statistics(
    geo_stats: dict[str, float], baseline_stats: dict[str, float]
) -> dict[str, float]:
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


def save_detailed_results_to_csv(
    results: list[tuple[str, dict, dict, dict]], filename: str | Path
) -> None:
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


DEFAULT_LATEX_COLUMNS = [
    ("rmse", "RMSE Diff (m)"),
    ("mean", "Mean Diff (m)"),
    ("median", "Median Diff (m)"),
]


def format_latex_table(
    results: list[tuple[str, dict[str, float]]],
    title: str,
    label: str = "tab:comparison_results",
    columns: list[tuple[str, str]] | None = None,
) -> str:
    """Format comparison results as a LaTeX table.

    Parameters
    ----------
    results : List[Tuple[str, Dict[str, float]]]
        List of (trajectory_name, stats) tuples. Each ``stats`` dict must carry every key
        named in ``columns``.
    title : str
        Caption/title for the table.
    label : str, optional
        LaTeX label for the table, by default "tab:comparison_results".
    columns : List[Tuple[str, str]], optional
        (stats key, column header) pairs defining the table's data columns, in order. Defaults
        to the RMSE/Mean/Median improvement triple that :func:`compute_improvement_statistics`
        produces, matching this function's original geophysical-aiding-vs-baseline table.
        Pass a different set for a table of absolute statistics, e.g. from
        :func:`compute_error_statistics`.

    Returns
    -------
    str
        LaTeX table as a string, ready for inclusion in a document.

    Notes
    -----
    The table includes per-trajectory rows plus summary statistics
    (mean, median, std) at the bottom.
    """
    if columns is None:
        columns = DEFAULT_LATEX_COLUMNS

    lines = []
    lines.append("\\begin{table}[h]")
    lines.append("    \\centering")
    lines.append(f"    \\caption{{{title}}}")
    column_spec = " ".join("c" for _ in columns)
    lines.append(f"    \\begin{{tabular}}{{ || l || {column_spec} || }}")
    lines.append("    \\toprule")
    header = " & ".join(header for _, header in columns)
    lines.append(f"    Trajectory Name & {header} \\\\")
    lines.append("    \\midrule")

    # Trajectory rows
    for traj_name, stats in results:
        # Clean up trajectory name (remove file extension, escape underscores)
        clean_name = str(traj_name).replace(".csv", "").replace("_", "\\_")
        values = " & ".join(f"{stats[key]:.2f}" for key, _ in columns)
        lines.append(f"    {clean_name} & {values} \\\\")

    # Summary statistics, one row per aggregate, over every column
    lines.append("    \\midrule")
    for row_name, aggregate in (("mean", np.mean), ("median", np.median), ("std", np.std)):
        values = " & ".join(
            f"{aggregate([stats[key] for _, stats in results]):.2f}" for key, _ in columns
        )
        lines.append(f"    {row_name} & {values} \\\\")
    lines.append("    \\bottomrule")
    lines.append("    \\end{tabular}")
    lines.append(f"    \\label{{{label}}}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def print_summary_statistics(results: list[tuple[str, dict[str, float]]], label: str) -> None:
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
