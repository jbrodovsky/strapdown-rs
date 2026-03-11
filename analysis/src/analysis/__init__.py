import os

# Ensure non-interactive backend for matplotlib to avoid Tkinter GUI usage
os.environ.setdefault("MPLBACKEND", "Agg")

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from haversine import Unit, haversine_vector
from pandas import DataFrame, read_csv
from tqdm import tqdm

from analysis.compare import (
    compute_error_statistics,
    compute_improvement_statistics,
    format_latex_table,
    print_summary_statistics,
    save_detailed_results_to_csv,
)
from analysis.plotting import plot_performance, plot_relative_performance
from analysis.preprocess import preprocess_data

__version__ = "0.1.0"


def main() -> None:
    parser = ArgumentParser(
        description="Data analysis and simulation orchestration tools for use with Strapdown-sim.",
        epilog="For more information, visit the Strapdown-sim documentation.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"strapdown-analysis package version {__version__}",
    )

    command = parser.add_subparsers(title="command", dest="command")

    preprocess = command.add_parser("preprocess", help="Preprocess raw trajectory data.")

    preprocess.add_argument(
        "-i",
        "--input",
        type=str,
        default="data/raw",
        help="Base directory for the sensor logger app data.",
    )
    preprocess.add_argument(
        "-o",
        "--output",
        type=str,
        default="data",
        help="Output directory for the cleaned data.",
    )
    preprocess.add_argument(
        "-b",
        "--buffer",
        type=float,
        default=0.1,
        help="Buffer amount to inflate the bounding box by (as a percentage). Default is 0.1 (10 percent).",
    )
    preprocess.add_argument(
        "--getmaps",
        action="store_true",
        help="Download geophysical maps for each trajectory.",
    )

    performance = command.add_parser("performance", help="Generate performance plots from mechanization results.")
    performance.add_argument(
        "-p",
        "--processed",
        type=str,
        help="Input directory containing the processed navigation result CSV files.",
    )
    performance.add_argument(
        "-r",
        "--reference",
        type=str,
        help="Input directory containing the reference GPS CSV files.",
        default="data/input",
    )
    performance.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output directory for the performance plots.",
        default="data/output",
    )

    geoperformance = command.add_parser("geoperformance", help="Generate geophysical performance plots.")
    geoperformance.add_argument(
        "-p",
        "--processed",
        type=str,
        help="Input directory containing the processed navigation result CSV files.",
    )
    geoperformance.add_argument(
        "-r",
        "--reference",
        type=str,
        help="Input directory containing the reference GPS CSV files.",
        default="data/input",
    )
    geoperformance.add_argument(
        "-d",
        "--degraded",
        type=str,
        help="Input directory containing the degraded navigation result CSV files.",
    )
    geoperformance.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output directory for the geophysical performance plots.",
        default="data/output",
    )
    geoperformance.add_argument(
        "-f",
        "--filter-name",
        type=str,
        default="geophysical",
        help="Name of the filter (e.g., rbpf, ukf, ekf) for labeling outputs.",
    )
    geoperformance.add_argument(
        "--geo-type",
        type=str,
        default="aided",
        help="Type of geophysical aiding (e.g., grav, mag, both) for labeling outputs.",
    )
    geoperformance.add_argument(
        "--no-latex",
        action="store_true",
        help="Disable LaTeX table generation.",
    )
    geoperformance.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable plot generation (only produce CSV and LaTeX outputs).",
    )

    # Compare filters command for cross-filter comparison
    compare_filters = command.add_parser(
        "compare-filters",
        help="Compare performance across different filter modalities (e.g., RBPF vs UKF vs EKF).",
    )
    compare_filters.add_argument(
        "-i",
        "--input-dirs",
        type=str,
        nargs="+",
        required=True,
        help="Input directories containing filter results to compare (one per filter).",
    )
    compare_filters.add_argument(
        "-l",
        "--labels",
        type=str,
        nargs="+",
        required=True,
        help="Labels for each filter (must match number of input directories).",
    )
    compare_filters.add_argument(
        "-r",
        "--reference",
        type=str,
        default="data/input",
        help="Directory containing GPS ground truth CSVs (default: data/input).",
    )
    compare_filters.add_argument(
        "-o",
        "--output",
        type=str,
        default="data/output/filter_comparison",
        help="Output directory for comparison results.",
    )
    compare_filters.add_argument(
        "--geo-type",
        type=str,
        default="comparison",
        help="Geophysical type label for outputs (e.g., grav, mag, both).",
    )

    args = parser.parse_args()

    if args.command == "preprocess":
        preprocess_data(args)
    elif args.command == "performance":
        performance_analysis(args)
    elif args.command == "geoperformance":
        geophysical_performance_analysis(args)
    else:
        parser.print_help()


def performance_analysis(args):
    """Generate performance plots from mechanization results."""
    input_dir = args.processed
    print(f"Generating performance plots from data in: {input_dir}")

    datasets = list(Path(input_dir).glob("*.csv"))
    print(f"Found {len(datasets)} datasets to process.")

    print(f"Comparing to reference data in: {args.reference}")
    references = list(Path(args.reference).glob("*.csv"))
    print(f"Found {len(references)} reference datasets.")

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Saving performance plots to: {args.output}")

    reference_path = Path(args.reference)

    summary_df = DataFrame(
        columns=[
            "Min Horizontal Error (m)",
            "Max Horizontal Error (m)",
            "Mean Horizontal Error (m)",
            "RMSE Horizontal Error (m)",
            "Min Vertical Error (m)",
            "Max Vertical Error (m)",
            "Mean Vertical Error (m)",
            "RMSE Vertical Error (m)",
            "Min 3D Error (m)",
            "Max 3D Error (m)",
            "Mean 3D Error (m)",
            "RMSE 3D Error (m)",
        ],  # ty:ignore[invalid-argument-type]
        index=[dataset.stem for dataset in datasets],  # ty:ignore[invalid-argument-type]
        # index.name = "Dataset"  # ty:ignore[unknown-argument]
    )

    for dataset in datasets:
        nav = read_csv(dataset, parse_dates=True, index_col=0)
        try:
            reference_file = reference_path / dataset.name
            gps = read_csv(reference_file, parse_dates=True, index_col=0)
        except FileNotFoundError:
            print(f"Reference file for {dataset.name} not found in {reference_path}. Skipping.")
            continue
        output_plot = output_path / f"{dataset.stem}_performance.png"
        print(f"Processing dataset {dataset} ({len(nav)}) with reference {reference_file.name} ({len(gps)})")
        try:
            plot_performance(nav, gps, output_plot)
        except Exception as e:
            print(f"Error plotting performance for {dataset.name}, possible dimension mismatch or missing data: {e}")
            continue
        two_d_error = haversine_vector(
            gps[["latitude", "longitude"]].to_numpy()[1:, :],
            nav[["latitude", "longitude"]].to_numpy(),
            Unit.METERS,
        )
        three_d_error = np.sqrt(two_d_error**2 + (gps["altitude"].to_numpy()[1:] - nav["altitude"].to_numpy()) ** 2)
        summary_df.loc[dataset.stem] = [
            np.nanmin(two_d_error),
            np.nanmax(two_d_error),
            np.nanmean(two_d_error),
            np.sqrt(np.nanmean(two_d_error**2)),
            np.nanmin(gps["altitude"].to_numpy()[1:] - nav["altitude"].to_numpy()),
            np.nanmax(gps["altitude"].to_numpy()[1:] - nav["altitude"].to_numpy()),
            np.nanmean(gps["altitude"].to_numpy()[1:] - nav["altitude"].to_numpy()),
            np.sqrt(np.nanmean((gps["altitude"].to_numpy()[1:] - nav["altitude"].to_numpy()) ** 2)),
            np.nanmin(three_d_error),
            np.nanmax(three_d_error),
            np.nanmean(three_d_error),
            np.sqrt(np.nanmean(three_d_error**2)),
        ]

    summary_file = output_path / "performance_summary.csv"
    summary_df.to_csv(summary_file)
    print("Performance analysis completed.")


def geophysical_performance_analysis(args):
    """Generate geophysical performance analysis with plots, CSV summaries, and LaTeX tables.

    This function processes geophysical-aided navigation results, comparing them against
    baseline degraded solutions and GPS ground truth. It produces:
    - Performance plots (PNG) showing error differences over time
    - CSV summary with error statistics per trajectory
    - Detailed CSV with per-trajectory statistics for geo, baseline, and differences
    - LaTeX table for publication
    - Console summary statistics
    """
    input_dir = args.processed
    filter_name = args.filter_name
    geo_type = args.geo_type
    generate_plots = not args.no_plots
    generate_latex = not args.no_latex

    print("=" * 80)
    print(f"Geophysical Performance Analysis: {filter_name.upper()} {geo_type}")
    print("=" * 80)
    print(f"Geophysical-aided data: {input_dir}")

    datasets = list(Path(input_dir).glob("*.csv"))
    print(f"Found {len(datasets)} datasets to process.")

    print(f"Reference (truth) data: {args.reference}")
    references = list(Path(args.reference).glob("*.csv"))
    print(f"Found {len(references)} reference datasets.")

    print(f"Degraded baseline data: {args.degraded}")
    degradeds = list(Path(args.degraded).glob("*.csv"))
    print(f"Found {len(degradeds)} degraded datasets.")

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {args.output}")
    print(f"Generate plots: {generate_plots}")
    print(f"Generate LaTeX: {generate_latex}")

    reference_path = Path(args.reference)
    degraded_path = Path(args.degraded)

    # Original summary DataFrame (error differences)
    summary_df = DataFrame(
        columns=[
            "Min Horizontal Error (m)",
            "Max Horizontal Error (m)",
            "Mean Horizontal Error (m)",
            "RMSE Horizontal Error (m)",
            "Min Vertical Error (m)",
            "Max Vertical Error (m)",
            "Mean Vertical Error (m)",
            "RMSE Vertical Error (m)",
            "Min 3D Error (m)",
            "Max 3D Error (m)",
            "Mean 3D Error (m)",
            "RMSE 3D Error (m)",
        ],  # ty:ignore[invalid-argument-type]
        index=[dataset.stem for dataset in datasets],  # ty:ignore[invalid-argument-type]
    )

    # For LaTeX table and detailed results
    latex_results = []  # List of (traj_name, improvement_stats)
    detailed_results = []  # List of (traj_name, geo_stats, baseline_stats, improvement_stats)

    for dataset in tqdm(datasets):
        geo = read_csv(dataset, parse_dates=True, index_col=0)
        try:
            reference_file = reference_path / dataset.name
            nav = read_csv(reference_file, parse_dates=True, index_col=0)
        except FileNotFoundError:
            print(f"Reference file for {dataset.name} not found in {reference_path}. Skipping.")
            continue
        try:
            degraded_file = degraded_path / dataset.name
            degraded_nav = read_csv(degraded_file, parse_dates=True, index_col=0)
        except FileNotFoundError:
            print(f"Degraded file for {dataset.name} not found in {degraded_path}. Skipping.")
            continue

        output_plot = output_path / f"{dataset.stem}_geophysical_performance.png"
        nav = nav.iloc[1:].copy()

        # Align datasets by index
        if not (len(nav) == len(geo)):
            if nav.index[0] not in geo.index:
                first_row = geo.iloc[[0]][["latitude", "longitude", "altitude"]].copy()
                first_row.index = [nav.index[0]]
                geo.loc[first_row.index] = first_row
                geo = geo.sort_index()
            geo = geo.reindex(nav.index)

        if not (len(nav) == len(degraded_nav)):
            if nav.index[0] not in degraded_nav.index:
                first_row = degraded_nav.iloc[[0]][["latitude", "longitude", "altitude"]].copy()
                first_row.index = [nav.index[0]]
                degraded_nav.loc[first_row.index] = first_row
                degraded_nav = degraded_nav.sort_index()
            degraded_nav = degraded_nav.reindex(nav.index)

        try:
            # Generate plot if enabled
            if generate_plots:
                plot_relative_performance(geo, degraded_nav, nav, output_plot)

            # Compute haversine errors
            geo_error = haversine_vector(
                geo[["latitude", "longitude"]].to_numpy(dtype=np.float64, copy=False),
                nav[["latitude", "longitude"]].to_numpy(),
                Unit.METERS,
            )

            deg_error = haversine_vector(
                degraded_nav[["latitude", "longitude"]].to_numpy(),
                nav[["latitude", "longitude"]].to_numpy(),
                Unit.METERS,
            )

            # Compute statistics for detailed output
            geo_stats = compute_error_statistics(geo_error)
            baseline_stats = compute_error_statistics(deg_error)
            improvement_stats = compute_improvement_statistics(geo_stats, baseline_stats)

            # Store for LaTeX and detailed CSV
            latex_results.append((dataset.stem, improvement_stats))
            detailed_results.append((dataset.stem, geo_stats, baseline_stats, improvement_stats))

            # Original summary DataFrame calculations
            err_diff = geo_error - deg_error
            geo_rmse = geo_stats["rmse"]
            deg_rmse = baseline_stats["rmse"]

            summary_df.loc[dataset.stem] = [
                np.nanmin(err_diff),
                np.nanmax(err_diff),
                np.nanmean(err_diff),
                geo_rmse - deg_rmse,
                np.nanmin(geo["altitude"].to_numpy() - nav["altitude"].to_numpy()),
                np.nanmax(geo["altitude"].to_numpy() - nav["altitude"].to_numpy()),
                np.nanmean(geo["altitude"].to_numpy() - nav["altitude"].to_numpy()),
                np.sqrt(np.nanmean((geo["altitude"].to_numpy() - nav["altitude"].to_numpy()) ** 2))
                - np.sqrt(np.nanmean((degraded_nav["altitude"].to_numpy() - nav["altitude"].to_numpy()) ** 2)),
                np.nanmin(
                    np.sqrt(geo_error**2 + (geo["altitude"].to_numpy() - nav["altitude"].to_numpy()) ** 2)
                    - np.sqrt(deg_error**2 + (degraded_nav["altitude"].to_numpy() - nav["altitude"].to_numpy()) ** 2)
                ),
                np.nanmax(
                    np.sqrt(geo_error**2 + (geo["altitude"].to_numpy() - nav["altitude"].to_numpy()) ** 2)
                    - np.sqrt(deg_error**2 + (degraded_nav["altitude"].to_numpy() - nav["altitude"].to_numpy()) ** 2)
                ),
                np.nanmean(
                    np.sqrt(geo_error**2 + (geo["altitude"].to_numpy() - nav["altitude"].to_numpy()) ** 2)
                    - np.sqrt(deg_error**2 + (degraded_nav["altitude"].to_numpy() - nav["altitude"].to_numpy()) ** 2)
                ),
                geo_rmse - deg_rmse,
            ]
        except Exception as e:
            print(f"Error processing {dataset.name}, possible dimension mismatch or missing data: {e}")
            continue

    # Add summary statistics to DataFrame
    summary_df.loc["median"] = summary_df.median()
    summary_df.loc["mean"] = summary_df.mean()
    summary_df.loc["std"] = summary_df.std()

    # Save original summary CSV
    summary_file = output_path / "geophysical_performance_summary.csv"
    summary_df.to_csv(summary_file)
    print(f"\nSaved performance summary to {summary_file}")

    # Save detailed results CSV
    if detailed_results:
        detailed_file = output_path / f"{filter_name}_{geo_type}_detailed_results.csv"
        save_detailed_results_to_csv(detailed_results, detailed_file)
        print(f"Saved detailed results to {detailed_file}")

    # Generate and save LaTeX table
    if generate_latex and latex_results:
        table_title = (
            f"{filter_name.upper()} {geo_type.capitalize()}-Aided Performance "
            "vs Baseline (Geo - Baseline, negative = improvement)"
        )
        table_label = f"tab:{filter_name}_{geo_type}_results"
        latex_table = format_latex_table(latex_results, table_title, table_label)

        tex_file = output_path / f"{filter_name}_{geo_type}_table.tex"
        with open(tex_file, "w") as f:
            f.write(latex_table)
        print(f"Saved LaTeX table to {tex_file}")

    # Print summary statistics
    if latex_results:
        print("\n" + "=" * 80)
        print("SUMMARY STATISTICS")
        print("=" * 80)
        print_summary_statistics(latex_results, f"{filter_name.upper()} {geo_type}-aided")

    print("\nGeophysical performance analysis completed.")


if __name__ == "__main__":
    main()
