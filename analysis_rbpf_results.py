#!/usr/bin/env python3
"""
RBPF Results Analysis Script

This script analyzes the RBPF navigation results and compares them to:
1. Baseline (degraded GNSS)
2. GPS ground truth
3. UKF results (for comparison)

Outputs LaTeX-formatted tables for the paper.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from haversine import Unit, haversine_vector

# Directories
RBPF_GRAV_DIR = Path("data/output/rbpf/grav")
RBPF_MAG_DIR = Path("data/output/rbpf/mag")
RBPF_BOTH_DIR = Path("data/output/rbpf/both")
RBPF_DEGRADED_DIR = Path("data/output/rbpf/degraded")
# UKF_GRAV_DIR = Path("data/output/ukf_original/grav")
# UKF_MAG_DIR = Path("data/output/ukf_original/mag")
# UKF_DEGRADED_DIR = Path("data/output/ukf_original/degraded")
UKF_GRAV_DIR = Path("data/output/ukf/grav")
UKF_MAG_DIR = Path("data/output/ukf/mag")
UKF_BOTH_DIR = Path("data/output/ukf/both")
UKF_DEGRADED_DIR = Path("data/output/ukf/degraded")
GPS_TRUTH_DIR = Path("data/input")


def load_trajectory(file_path: Path) -> pd.DataFrame:
    """Load a trajectory CSV file."""
    df = pd.read_csv(file_path)
    # Standardize time column name
    if "time" in df.columns:
        df = df.rename(columns={"time": "timestamp"})
    # Parse timestamps
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def calculate_haversine_errors(nav_df: pd.DataFrame, truth_df: pd.DataFrame) -> np.ndarray:
    """
    Calculate haversine distance errors between navigation solution and truth.

    Args:
        nav_df: Navigation solution with latitude, longitude columns
        truth_df: Truth data with latitude, longitude columns

    Returns:
        Array of haversine distances in meters
    """
    # Make copies to avoid modifying originals
    nav_df = nav_df.copy()
    truth_df = truth_df.copy()

    # Drop rows with NaN lat/lon in truth data
    truth_df = truth_df.dropna(subset=["latitude", "longitude"])

    # Ensure both dataframes have timestamp index
    if "timestamp" in nav_df.columns:
        nav_df = nav_df.set_index("timestamp")
    if "timestamp" in truth_df.columns:
        truth_df = truth_df.set_index("timestamp")

    # Align by index (timestamp) - use inner join to get only matching timestamps
    aligned = pd.concat(
        [nav_df[["latitude", "longitude"]], truth_df[["latitude", "longitude"]]],
        axis=1,
        keys=["nav", "truth"],
        join="inner",
    )

    # Drop any remaining NaN rows
    aligned = aligned.dropna()

    # Check if we have data after alignment
    if len(aligned) == 0:
        raise ValueError("No overlapping timestamps between navigation and truth data")

    # Create coordinate pairs
    nav_coords = list(zip(aligned["nav"]["latitude"].values, aligned["nav"]["longitude"].values))
    truth_coords = list(zip(aligned["truth"]["latitude"].values, aligned["truth"]["longitude"].values))

    # Calculate haversine distances
    distances = haversine_vector(nav_coords, truth_coords, Unit.METERS, comb=False)

    return distances


def compute_statistics(errors: np.ndarray) -> Dict[str, float]:
    """Compute error statistics."""
    return {
        "rmse": np.sqrt(np.mean(errors**2)),
        "mean": np.mean(errors),
        "median": np.median(errors),
        "std": np.std(errors),
        "max": np.max(errors),
        "min": np.min(errors),
    }


def get_trajectory_files(directory: Path) -> List[Path]:
    """Get all CSV files from a directory."""
    return sorted(directory.glob("*.csv"))


def analyze_trajectory(
    traj_name: str, geo_aided_path: Path, degraded_path: Path, truth_path: Path
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """
    Analyze a single trajectory.

    Returns:
        geo_stats: Statistics for geophysical-aided solution
        degraded_stats: Statistics for baseline degraded solution
        improvement_stats: Difference (degraded - geo_aided)
    """
    # Load data
    geo_aided = load_trajectory(geo_aided_path)
    degraded = load_trajectory(degraded_path)
    truth = load_trajectory(truth_path)

    # Calculate errors
    geo_errors = calculate_haversine_errors(geo_aided, truth)
    degraded_errors = calculate_haversine_errors(degraded, truth)

    # Compute statistics
    geo_stats = compute_statistics(geo_errors)
    degraded_stats = compute_statistics(degraded_errors)

    # Compute difference (geophysical - baseline, negative means improvement)
    improvement_stats = {
        "rmse": geo_stats["rmse"] - degraded_stats["rmse"],
        "mean": geo_stats["mean"] - degraded_stats["mean"],
        "median": geo_stats["median"] - degraded_stats["median"],
    }

    return geo_stats, degraded_stats, improvement_stats


def save_results_to_csv(results: List[Tuple[str, Dict, Dict, Dict]], filename: str):
    """
    Save detailed results to CSV for record keeping.

    Args:
        results: List of (traj_name, geo_stats, degraded_stats, improvement_stats) tuples
        filename: Output CSV filename
    """
    rows = []
    for traj_name, geo_stats, degraded_stats, improvement_stats in results:
        row = {
            "trajectory": traj_name.replace(".csv", ""),
            "geo_rmse": geo_stats["rmse"],
            "geo_mean": geo_stats["mean"],
            "geo_median": geo_stats["median"],
            "baseline_rmse": degraded_stats["rmse"],
            "baseline_mean": degraded_stats["mean"],
            "baseline_median": degraded_stats["median"],
            "diff_rmse": improvement_stats["rmse"],
            "diff_mean": improvement_stats["mean"],
            "diff_median": improvement_stats["median"],
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False)
    print(f"Saved detailed results to {filename}")


def format_latex_table(results: List[Tuple[str, Dict]], title: str) -> str:
    """
    Format results as LaTeX table.

    Args:
        results: List of (trajectory_name, improvement_stats) tuples
        title: Table title

    Returns:
        LaTeX table string
    """
    lines = []
    lines.append("\\begin{table}[h]")
    lines.append("    \\centering")
    lines.append(f"    \\caption{{{title}}}")
    lines.append("    \\begin{tabular}{ || l  ||ccc|| }")
    lines.append("    \\toprule")
    lines.append("   Trajectory Name & RMSE Diff (m) & Mean Diff (m) & Median Diff (m) \\\\")
    lines.append("        \\midrule")

    # Trajectory rows
    for traj_name, stats in results:
        # Clean up trajectory name (remove file extension)
        clean_name = traj_name.replace(".csv", "").replace("_", "\\_")
        lines.append(f"        {clean_name} & {stats['rmse']:.2f} & {stats['mean']:.2f} & {stats['median']:.2f} \\\\")

    # Calculate summary statistics
    rmse_diffs = [s["rmse"] for _, s in results]
    mean_diffs = [s["mean"] for _, s in results]
    median_diffs = [s["median"] for _, s in results]

    lines.append("        \\midrule")
    lines.append(
        f"        mean & {np.mean(rmse_diffs):.2f} & {np.mean(mean_diffs):.2f} & {np.mean(median_diffs):.2f} \\\\"
    )
    lines.append(
        f"        median & {np.median(rmse_diffs):.2f} & {np.median(mean_diffs):.2f} & {np.median(median_diffs):.2f} \\\\"
    )
    lines.append(f"        std & {np.std(rmse_diffs):.2f} & {np.std(mean_diffs):.2f} & {np.std(median_diffs):.2f} \\\\")
    lines.append("        \\bottomrule")
    lines.append("    \\end{tabular}")
    lines.append("    \\label{tab:rbpf_results}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def main():
    """Main analysis function."""
    print("=" * 80)
    print("RBPF Results Analysis")
    print("=" * 80)

    # Get all trajectory files
    grav_files = get_trajectory_files(RBPF_GRAV_DIR)
    mag_files = get_trajectory_files(RBPF_MAG_DIR)
    both_files = get_trajectory_files(RBPF_BOTH_DIR)
    degraded_files = get_trajectory_files(RBPF_DEGRADED_DIR)

    print(f"\nFound {len(grav_files)} gravity-aided trajectories")
    print(f"Found {len(mag_files)} magnetic-aided trajectories")
    print(f"Found {len(both_files)} combined-aided trajectories")
    print(f"Found {len(degraded_files)} baseline degraded trajectories")

    # Analyze gravity-aided results
    print("\n" + "=" * 80)
    print("GRAVITY-AIDED RESULTS")
    print("=" * 80)
    grav_results = []
    grav_detailed = []

    for grav_file in grav_files:
        traj_name = grav_file.name
        degraded_file = RBPF_DEGRADED_DIR / traj_name
        truth_file = GPS_TRUTH_DIR / traj_name

        if not degraded_file.exists():
            print(f"Warning: No degraded file for {traj_name}")
            continue
        if not truth_file.exists():
            print(f"Warning: No truth file for {traj_name}")
            continue

        try:
            geo_stats, degraded_stats, improvement = analyze_trajectory(traj_name, grav_file, degraded_file, truth_file)
            grav_results.append((traj_name, improvement))
            grav_detailed.append((traj_name, geo_stats, degraded_stats, improvement))

            print(f"\n{traj_name}:")
            print(f"  Gravity RMSE: {geo_stats['rmse']:.2f} m")
            print(f"  Degraded RMSE: {degraded_stats['rmse']:.2f} m")
            print(f"  Difference (Geo-Base): {improvement['rmse']:.2f} m (negative is better)")
        except Exception as e:
            print(f"Error processing {traj_name}: {e}")

    # Analyze magnetic-aided results
    print("\n" + "=" * 80)
    print("MAGNETIC-AIDED RESULTS")
    print("=" * 80)
    mag_results = []
    mag_detailed = []

    for mag_file in mag_files:
        traj_name = mag_file.name
        degraded_file = RBPF_DEGRADED_DIR / traj_name
        truth_file = GPS_TRUTH_DIR / traj_name

        if not degraded_file.exists() or not truth_file.exists():
            continue

        try:
            geo_stats, degraded_stats, improvement = analyze_trajectory(traj_name, mag_file, degraded_file, truth_file)
            mag_results.append((traj_name, improvement))
            mag_detailed.append((traj_name, geo_stats, degraded_stats, improvement))

            print(f"\n{traj_name}:")
            print(f"  Magnetic RMSE: {geo_stats['rmse']:.2f} m")
            print(f"  Degraded RMSE: {degraded_stats['rmse']:.2f} m")
            print(f"  Difference (Geo-Base): {improvement['rmse']:.2f} m (negative is better)")
        except Exception as e:
            print(f"Error processing {traj_name}: {e}")

    # Analyze combined-aided results
    print("\n" + "=" * 80)
    print("COMBINED-AIDED RESULTS")
    print("=" * 80)
    both_results = []
    both_detailed = []

    for both_file in both_files:
        traj_name = both_file.name
        degraded_file = RBPF_DEGRADED_DIR / traj_name
        truth_file = GPS_TRUTH_DIR / traj_name

        if not degraded_file.exists() or not truth_file.exists():
            continue

        try:
            geo_stats, degraded_stats, improvement = analyze_trajectory(traj_name, both_file, degraded_file, truth_file)
            both_results.append((traj_name, improvement))
            both_detailed.append((traj_name, geo_stats, degraded_stats, improvement))

            print(f"\n{traj_name}:")
            print(f"  Combined RMSE: {geo_stats['rmse']:.2f} m")
            print(f"  Degraded RMSE: {degraded_stats['rmse']:.2f} m")
            print(f"  Difference (Geo-Base): {improvement['rmse']:.2f} m (negative is better)")
        except Exception as e:
            print(f"Error processing {traj_name}: {e}")

    # Save detailed results to CSV
    print("\n" + "=" * 80)
    print("SAVING DETAILED RESULTS TO CSV")
    print("=" * 80)

    if grav_detailed:
        save_results_to_csv(grav_detailed, "rbpf_gravity_results.csv")
    if mag_detailed:
        save_results_to_csv(mag_detailed, "rbpf_magnetic_results.csv")
    if both_detailed:
        save_results_to_csv(both_detailed, "rbpf_combined_results.csv")

    # Generate LaTeX tables
    print("\n" + "=" * 80)
    print("LATEX TABLES")
    print("=" * 80)

    if grav_results:
        print("\n--- Gravity Table ---")
        grav_table = format_latex_table(
            grav_results, "RBPF Gravity-Aided Performance vs Baseline (Geo - Baseline, negative = improvement)"
        )
        print(grav_table)

        # Save to file
        with open("rbpf_gravity_table.tex", "w") as f:
            f.write(grav_table)
        print("\nSaved to rbpf_gravity_table.tex")

    if mag_results:
        print("\n--- Magnetic Table ---")
        mag_table = format_latex_table(
            mag_results, "RBPF Magnetic-Aided Performance vs Baseline (Geo - Baseline, negative = improvement)"
        )
        print(mag_table)

        with open("rbpf_magnetic_table.tex", "w") as f:
            f.write(mag_table)
        print("\nSaved to rbpf_magnetic_table.tex")

    if both_results:
        print("\n--- Combined Table ---")
        both_table = format_latex_table(
            both_results, "RBPF Combined-Aided Performance vs Baseline (Geo - Baseline, negative = improvement)"
        )
        print(both_table)

        with open("rbpf_combined_table.tex", "w") as f:
            f.write(both_table)
        print("\nSaved to rbpf_combined_table.tex")

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    if grav_results:
        grav_improvements = [r[1]["rmse"] for r in grav_results]
        print(f"\nGravity-aided:")
        print(
            f"  Number of trajectories improved (negative difference): {sum(1 for x in grav_improvements if x < 0)}/{len(grav_improvements)}"
        )
        print(f"  Mean RMSE difference: {np.mean(grav_improvements):.2f} m")
        print(f"  Median RMSE difference: {np.median(grav_improvements):.2f} m")
        print(f"  Best (most negative): {min(grav_improvements):.2f} m")
        print(f"  Worst (most positive): {max(grav_improvements):.2f} m")

    if mag_results:
        mag_improvements = [r[1]["rmse"] for r in mag_results]
        print(f"\nMagnetic-aided:")
        print(
            f"  Number of trajectories improved (negative difference): {sum(1 for x in mag_improvements if x < 0)}/{len(mag_improvements)}"
        )
        print(f"  Mean RMSE difference: {np.mean(mag_improvements):.2f} m")
        print(f"  Median RMSE difference: {np.median(mag_improvements):.2f} m")
        print(f"  Best (most negative): {min(mag_improvements):.2f} m")
        print(f"  Worst (most positive): {max(mag_improvements):.2f} m")

    if both_results:
        both_improvements = [r[1]["rmse"] for r in both_results]
        print(f"\nCombined-aided:")
        print(
            f"  Number of trajectories improved (negative difference): {sum(1 for x in both_improvements if x < 0)}/{len(both_improvements)}"
        )
        print(f"  Mean RMSE difference: {np.mean(both_improvements):.2f} m")
        print(f"  Median RMSE difference: {np.median(both_improvements):.2f} m")
        print(f"  Best (most negative): {min(both_improvements):.2f} m")
        print(f"  Worst (most positive): {max(both_improvements):.2f} m")


if __name__ == "__main__":
    main()
