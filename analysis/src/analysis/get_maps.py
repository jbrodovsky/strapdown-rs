"""
Helper script for downloading geophysical maps from the GMT servers.
"""

import os
from argparse import ArgumentParser
from glob import glob

from pandas import read_csv
from pygmt.datasets import (
    load_earth_free_air_anomaly,
    load_earth_magnetic_anomaly,
    load_earth_relief,
)
from tqdm import tqdm

from analysis.plotting import inflate_bounds


def get_maps(
    input_file_directory: str,
    buffer: float = 0.25,
    output_directory: str = os.path.join("data", "input"),
) -> None:
    """
    Goes through the target directory to download the geophysical maps.

    :param input_file_directory: file path to input directory containing .csv trajectory files
    :type input_file_directory: str
    :param buffer: inflation factor for bounds; defaults to .25
    :type buffer: float
    """
    input_files = glob(os.path.join(input_file_directory, "*.csv"))
    print(f"Found: {len(input_files)} input files.")
    for input_file in tqdm(input_files):
        print(f"Processing {input_file}...")
        basename = os.path.basename(input_file)
        df = read_csv(input_file, index_col=0, parse_dates=True)
        lon_min, lon_max = df["longitude"].min(), df["longitude"].max()
        lat_min, lat_max = df["latitude"].min(), df["latitude"].max()

        lon_min, lon_max, lat_min, lat_max = inflate_bounds(lon_min, lon_max, lat_min, lat_max, buffer)

        # print(f"  Longitude bounds: {lon_min} to {lon_max}")
        # print(f"  Latitude bounds: {lat_min} to {lat_max}")

        # Download the maps
        relief = load_earth_relief(resolution="15s", region=[lon_min, lon_max, lat_min, lat_max])
        relief.to_netcdf(os.path.join(output_directory, basename.replace(".csv", "_relief.nc")))
        # print(f"  Downloaded relief map: {relief.data.shape}.")

        gravity = load_earth_free_air_anomaly(resolution="01m", region=[lon_min, lon_max, lat_min, lat_max])
        gravity.to_netcdf(os.path.join(output_directory, basename.replace(".csv", "_gravity.nc")))
        # print(f"  Downloaded gravity map: {gravity.data.shape}.")

        magnetic = load_earth_magnetic_anomaly(
            resolution="03m",
            region=[lon_min, lon_max, lat_min, lat_max],
            data_source="wdmam",
        )
        magnetic.to_netcdf(os.path.join(output_directory, basename.replace(".csv", "_magnetic.nc")))
        # print(f"  Downloaded magnetic map: {magnetic.data.shape}.")


if __name__ == "__main__":
    parser = ArgumentParser(description="Download geophysical maps from the GMT servers.")
    parser.add_argument(
        "--buffer",
        type=float,
        default=0.1,
        help="Buffer amount to inflate the bounding box by (as a percentage). Default is 0.1 (10 percent).",
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        default="data/input",
        help="Input file containing the bounding box coordinates.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        default="data/output",
        help="Output directory for the downloaded maps.",
    )
    args = parser.parse_args()
    print(f"Using input directory: {args.input}")
    print(f"Using buffer: {args.buffer}")
    print(f"Using output directory: {args.output}")

    get_maps(args.input, args.buffer)
