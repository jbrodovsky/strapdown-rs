from numpy.char import index
import os
# Ensure non-interactive backend for matplotlib to avoid Tkinter GUI usage
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np

from argparse import ArgumentParser
from pandas import read_csv, DataFrame
from pathlib import Path

from analysis.preprocess import preprocess_data
from analysis.plotting import plot_performance, plot_relative_performance
from haversine import haversine_vector, Unit

__version__ = "0.1.0"


def main() -> None:

    parser = ArgumentParser(
        description="Data analysis and simulation orchestration tools for use with Strapdown-sim.",
        epilog="For more information, visit the Strapdown-sim documentation."
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"strapdown-analysis package version {__version__}",
    )

    command = parser.add_subparsers(title="command", dest="command")

    preprocess = command.add_parser(
        "preprocess", help="Preprocess raw trajectory data."
    )

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

    performance = command.add_parser(
        "performance", help="Generate performance plots from mechanization results."
    )
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
        default="data/input"
    )
    performance.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output directory for the performance plots.",
        default="data/output"
    )

    geoperformance = command.add_parser(
        "geoperformance", help="Generate geophysical performance plots."
    )
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
        default="data/input"
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
        default="data/output"
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
        columns=["Min Horizontal Error (m)", "Max Horizontal Error (m)", "Mean Horizontal Error (m)", "RMSE Horizontal Error (m)", "Min Vertical Error (m)", "Max Vertical Error (m)", "Mean Vertical Error (m)", "RMSE Vertical Error (m)", "Min 3D Error (m)", "Max 3D Error (m)", "Mean 3D Error (m)", "RMSE 3D Error (m)"],  # ty:ignore[invalid-argument-type]
        index=[dataset.stem for dataset in datasets],  # ty:ignore[invalid-argument-type]
        #index.name = "Dataset"  # ty:ignore[unknown-argument]
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
        three_d_error = np.sqrt(two_d_error**2 + (gps["altitude"].to_numpy()[1:] - nav["altitude"].to_numpy())**2)
        summary_df.loc[dataset.stem] = [
            np.nanmin(two_d_error),
            np.nanmax(two_d_error),
            np.nanmean(two_d_error),
            np.sqrt(np.nanmean(two_d_error**2)),
            np.nanmin(gps["altitude"].to_numpy()[1:] - nav["altitude"].to_numpy()),
            np.nanmax(gps["altitude"].to_numpy()[1:] - nav["altitude"].to_numpy()),
            np.nanmean(gps["altitude"].to_numpy()[1:] - nav["altitude"].to_numpy()),
            np.sqrt(np.nanmean((gps["altitude"].to_numpy()[1:] - nav["altitude"].to_numpy())**2)),
            np.nanmin(three_d_error),
            np.nanmax(three_d_error),
            np.nanmean(three_d_error),
            np.sqrt(np.nanmean(three_d_error**2)),
        ]


    summary_file = output_path / "performance_summary.csv"
    summary_df.to_csv(summary_file)
    print("Performance analysis completed.")

def geophysical_performance_analysis(args):
    """Generate geophysical performance plots."""
    input_dir = args.processed
    print(f"Generating geophysical performance plots from data in: {input_dir}")
    
    datasets = list(Path(input_dir).glob("*.csv"))
    print(f"Found {len(datasets)} datasets to process.")

    print(f"Comparing to reference data in: {args.reference}")
    references = list(Path(args.reference).glob("*.csv"))
    print(f"Found {len(references)} reference datasets.")

    print(f"Comparing to degraded data in: {args.degraded}")
    degradeds = list(Path(args.degraded).glob("*.csv"))
    print(f"Found {len(degradeds)} degraded datasets.")

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Saving geophysical performance plots to: {args.output}")

    reference_path = Path(args.reference)
    degraded_path = Path(args.degraded)

    for dataset in datasets:
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
        print(f"Processing dataset {dataset} ({len(nav)}) with reference {reference_file.name} ({len(nav)}) and degraded {degraded_file.name} ({len(degraded_nav)})")
        try:
            plot_relative_performance(geo, degraded_nav, nav, output_plot)
        except Exception as e:
            print(f"Error plotting geophysical performance for {dataset.name}, possible dimension mismatch or missing data: {e}")
            continue

    print("Geophysical performance analysis completed.")

if __name__ == "__main__":
    main()
