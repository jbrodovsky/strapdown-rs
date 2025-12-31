import os
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from haversine import Unit, haversine_vector


def plot_performance(nav: pd.DataFrame, gps: pd.DataFrame, output_path: Path | str):
    """
    Placeholder for performance plotting function.
    """
    output_path = Path(output_path)
    # output_path.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    two_d_error = haversine_vector(
        gps[["latitude", "longitude"]].to_numpy()[1:, :],
        nav[["latitude", "longitude"]].to_numpy(),
        Unit.METERS,
    )
    ax.plot(
        (nav.index - nav.index[0]).total_seconds(),
        two_d_error,
        label="2D Haversine Error",
    )
    ax.plot(
        (nav.index - nav.index[0]).total_seconds(),
        abs(nav["altitude"].to_numpy() - gps["altitude"].to_numpy()[1:]),
        label="Altitude Error",
    )
    ax.plot(
        (gps.index - gps.index[0]).total_seconds(),
        gps["horizontalAccuracy"],
        label="GPS Horizontal Accuracy",
        linestyle="--",
    )
    ax.plot(
        (gps.index - gps.index[0]).total_seconds(),
        gps["verticalAccuracy"],
        label="GPS Vertical Accuracy",
        linestyle="--",
    )
    ax.set_xlim(left=0)
    ax.set_ylim((0, 50))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("2D Haversine Error (m)")
    ax.set_title(
        "Strapdown INS Simulation Performance with GPS Comparison", fontsize=16
    )
    ax.grid()
    ax.legend()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def run_truth_mechanization(input_dir: str, output_dir: str):
    """
    Create truth mechanization data sets from all .csv files in input_dir.
    """
    print(f"Reading data from: {input_dir}")
    print(f"Writing results to: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".csv"):
                input_file = os.path.join(root, file)
                base = os.path.splitext(os.path.basename(input_file))[0]
                output_file = os.path.join(output_dir, f"{base}.csv")
                print(f"Processing: {input_file}")
                # try:
                #     os.system(f"strapdown --mode closed-loop --input {input_file} --output {output_file}")
                # except Exception as err:
                #     print(f"Skipping {input_file} due to error: {err}")
                #     continue

                nav = pd.read_csv(output_file, parse_dates=True, index_col=0)
                gps = pd.read_csv(input_file, parse_dates=True, index_col=0)
                output_file = Path(output_dir) / f"{base}_performance.png"
                plot_performance(nav, gps, output_file)

    print("Truth mechanization data sets created.")


def main():
    parser = ArgumentParser(description="Create truth mechanization data sets.")
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Input directory containing .csv files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for truth data sets.",
    )
    args = parser.parse_args()
    run_truth_mechanization(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
