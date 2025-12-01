#!/usr/bin/env python3
"""
Example analysis script for strapdown-rs navigation results.

This script demonstrates how to:
1. Load navigation results from CSV
2. Calculate position error metrics
3. Generate error plots
4. Compare multiple scenarios

Usage:
    python analyze_results.py --truth baseline.csv --test degraded.csv --output plots/

Requirements:
    pip install pandas numpy matplotlib haversine
"""

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optional: haversine for accurate distance calculation
try:
    from haversine import haversine_vector, Unit
    HAS_HAVERSINE = True
except ImportError:
    HAS_HAVERSINE = False
    print("Warning: haversine not installed. Using approximate distance calculation.")


def load_navigation_csv(path: Path) -> pd.DataFrame:
    """Load navigation result CSV with timestamp index."""
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df


def calculate_position_error(truth: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """
    Calculate 2D horizontal position error in meters.
    
    Uses haversine formula if available, otherwise falls back to
    equirectangular approximation.
    """
    truth_coords = truth[['latitude', 'longitude']].values
    test_coords = test[['latitude', 'longitude']].values
    
    if HAS_HAVERSINE:
        error_m = haversine_vector(truth_coords, test_coords, Unit.METERS)
    else:
        # Equirectangular approximation (less accurate but simple)
        R = 6371000  # Earth radius in meters
        lat1 = np.radians(truth_coords[:, 0])
        lat2 = np.radians(test_coords[:, 0])
        dlat = lat2 - lat1
        dlon = np.radians(test_coords[:, 1] - truth_coords[:, 1])
        x = dlon * np.cos((lat1 + lat2) / 2)
        y = dlat
        error_m = R * np.sqrt(x**2 + y**2)
    
    return error_m


def calculate_altitude_error(truth: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Calculate altitude error in meters."""
    return np.abs(test['altitude'].values - truth['altitude'].values)


def calculate_velocity_error(truth: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Calculate 3D velocity error magnitude in m/s."""
    vn_err = test['velocity_north'].values - truth['velocity_north'].values
    ve_err = test['velocity_east'].values - truth['velocity_east'].values
    vd_err = test['velocity_down'].values - truth['velocity_down'].values
    return np.sqrt(vn_err**2 + ve_err**2 + vd_err**2)


def compute_statistics(errors: np.ndarray) -> dict:
    """Compute error statistics."""
    return {
        'mean': np.mean(errors),
        'rms': np.sqrt(np.mean(errors**2)),
        'max': np.max(errors),
        'std': np.std(errors),
        'p50': np.percentile(errors, 50),
        'p95': np.percentile(errors, 95),
        'p99': np.percentile(errors, 99),
    }


def print_statistics(name: str, stats: dict, unit: str = 'm'):
    """Print formatted statistics."""
    print(f"\n{name} Statistics:")
    print(f"  Mean:     {stats['mean']:.3f} {unit}")
    print(f"  RMS:      {stats['rms']:.3f} {unit}")
    print(f"  Max:      {stats['max']:.3f} {unit}")
    print(f"  Std Dev:  {stats['std']:.3f} {unit}")
    print(f"  Median:   {stats['p50']:.3f} {unit}")
    print(f"  95th %:   {stats['p95']:.3f} {unit}")
    print(f"  99th %:   {stats['p99']:.3f} {unit}")


def plot_error_time_series(
    time_s: np.ndarray,
    errors: np.ndarray,
    title: str,
    ylabel: str,
    output_path: Optional[Path] = None,
):
    """Plot error vs time."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(time_s, errors, linewidth=0.8)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved: {output_path}")
    plt.show()


def plot_trajectory_comparison(
    truth: pd.DataFrame,
    test: pd.DataFrame,
    output_path: Optional[Path] = None,
):
    """Plot truth vs test trajectory on lat/lon plot."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.plot(truth['longitude'], truth['latitude'], 
            'g-', linewidth=2, label='Truth', alpha=0.8)
    ax.plot(test['longitude'], test['latitude'], 
            'r--', linewidth=1.5, label='Estimated', alpha=0.8)
    
    ax.set_xlabel('Longitude (degrees)')
    ax.set_ylabel('Latitude (degrees)')
    ax.set_title('Trajectory Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved: {output_path}")
    plt.show()


def plot_error_histogram(
    errors: np.ndarray,
    title: str,
    xlabel: str,
    output_path: Optional[Path] = None,
):
    """Plot error distribution histogram."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.hist(errors, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(errors), color='r', linestyle='--', 
               label=f'Mean: {np.mean(errors):.2f}')
    ax.axvline(np.percentile(errors, 95), color='orange', linestyle='--',
               label=f'95th %: {np.percentile(errors, 95):.2f}')
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Count')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved: {output_path}")
    plt.show()


def plot_covariance_bounds(
    time_s: np.ndarray,
    errors: np.ndarray,
    cov_values: np.ndarray,
    title: str,
    output_path: Optional[Path] = None,
):
    """Plot error with covariance bounds (±2σ)."""
    fig, ax = plt.subplots(figsize=(12, 4))
    
    sigma = np.sqrt(cov_values)
    
    ax.fill_between(time_s, 0, 2*sigma, alpha=0.3, color='blue', label='2σ bounds')
    ax.plot(time_s, errors, 'r-', linewidth=0.8, label='Error')
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Error / Uncertainty (m)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved: {output_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Analyze strapdown-rs navigation results'
    )
    parser.add_argument('--truth', type=Path, required=True,
                        help='Path to truth/baseline CSV')
    parser.add_argument('--test', type=Path, required=True,
                        help='Path to test result CSV')
    parser.add_argument('--output', type=Path, default=None,
                        help='Output directory for plots (optional)')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip generating plots')
    
    args = parser.parse_args()
    
    # Create output directory if specified
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"Loading truth: {args.truth}")
    truth = load_navigation_csv(args.truth)
    
    print(f"Loading test:  {args.test}")
    test = load_navigation_csv(args.test)
    
    # Ensure same length (truncate to shorter)
    min_len = min(len(truth), len(test))
    truth = truth.iloc[:min_len]
    test = test.iloc[:min_len]
    
    print(f"Analyzing {min_len} samples")
    
    # Calculate errors
    pos_error = calculate_position_error(truth, test)
    alt_error = calculate_altitude_error(truth, test)
    vel_error = calculate_velocity_error(truth, test)
    
    # Compute and print statistics
    pos_stats = compute_statistics(pos_error)
    alt_stats = compute_statistics(alt_error)
    vel_stats = compute_statistics(vel_error)
    
    print_statistics("2D Position Error", pos_stats, "m")
    print_statistics("Altitude Error", alt_stats, "m")
    print_statistics("Velocity Error", vel_stats, "m/s")
    
    # Generate plots
    if not args.no_plots:
        # Time vector
        time_s = (test.index - test.index[0]).total_seconds().values
        
        # Position error vs time
        plot_error_time_series(
            time_s, pos_error,
            'Horizontal Position Error Over Time',
            'Position Error (m)',
            args.output / 'position_error.png' if args.output else None
        )
        
        # Trajectory comparison
        plot_trajectory_comparison(
            truth, test,
            args.output / 'trajectory.png' if args.output else None
        )
        
        # Error histogram
        plot_error_histogram(
            pos_error,
            'Position Error Distribution',
            'Position Error (m)',
            args.output / 'error_histogram.png' if args.output else None
        )
        
        # Covariance consistency (if available)
        if 'latitude_cov' in test.columns:
            # Convert lat covariance from radians² to meters²
            R = 6371000
            lat_cov_m2 = test['latitude_cov'].values * (R * np.pi / 180)**2
            
            # Use latitude component of position error for comparison
            lat_error = np.abs(
                test['latitude'].values - truth['latitude'].values
            ) * R * np.pi / 180
            
            plot_covariance_bounds(
                time_s, lat_error, lat_cov_m2,
                'Latitude Error with Covariance Bounds',
                args.output / 'covariance_bounds.png' if args.output else None
            )
    
    print("\nAnalysis complete.")


if __name__ == '__main__':
    main()
