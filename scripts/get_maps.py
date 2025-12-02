"""
Helper script for downloading geophysical maps from the GMT servers.
"""

import os
from argparse import ArgumentParser
from glob import glob
from pathlib import Path

from cartopy import crs as ccrs
from cartopy.io import img_tiles as cimgt
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from pandas import read_csv
from pygmt.datasets import (
    load_earth_free_air_anomaly,
    load_earth_magnetic_anomaly,
    load_earth_relief,
)


def inflate_bounds(
    x_min: int | float,
    x_max: int | float,
    y_min: int | float,
    y_max: int | float,
    buffer: float = 0.1,
) -> tuple[int | float, int | float, int | float, int | float]:
    """Inflate bounding box by a buffer amount. Intended to be used to provide a margin around the area of interest by
    increasing the bounds by a percentage

    :param x_min: minimum value along the x-axis
    :type x_min: int | float
    :param x_max: maximum value along the x-axis
    :type x_max: int | float
    :param y_min: minimum value along the y-axis
    :type y_min: int | float
    :param y_max: maximum value along the y-axis
    :type y_max: int | float
    :param buffer: Buffer amount. Defaults to 0.1.
    :type buffer: float

    Returns:
        tuple[int | float, int | float, int | float, int | float]: Inflated bounding box.
    """
    x_range = x_max - x_min
    y_range = y_max - y_min
    return (
        x_min - x_range * buffer,
        x_max + x_range * buffer,
        y_min - y_range * buffer,
        y_max + y_range * buffer,
    )


def plot_street_map(
    latitude: list[float], longitude: list[float], margin=0.01
) -> Figure:
    """
    Plots a street map using OpenStreetMap tiles.

    :param latitude: list of latitudes
    :type latitude: list[float]
    :param longitude: list of longitudes
    :type longitude: list[float]
    """
    # Define the map extent
    lat_min, lat_max = min(latitude), max(latitude)
    lon_min, lon_max = min(longitude), max(longitude)

    # Create a Stamen Terrain instance
    osm_tiles = cimgt.OSM()

    # Create a figure
    # Create a map using cartopy with OpenStreetMap background
    fig = plt.figure(figsize=(12, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent(
        [lon_min - margin, lon_max + margin, lat_min - margin, lat_max + margin],
        crs=ccrs.PlateCarree(),
    )  # type: ignore

    # Add the OSM tiles to the map
    ax.add_image(osm_tiles, 12)  # type: ignore

    # Plot the trajectory points
    ax.plot(longitude, latitude, "r.", transform=ccrs.PlateCarree())

    # Add gridlines with labels
    gl = ax.gridlines(draw_labels=True, alpha=0.2)  # type: ignore
    gl.top_labels = False
    gl.right_labels = False

    ax.set_title("Street Map with Trajectory Points", fontsize=16)

    return fig


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
    for input_file in input_files:
        print(f"Processing {input_file}...")
        basename = os.path.basename(input_file)
        df = read_csv(input_file, index_col=0, parse_dates=True)
        lon_min, lon_max = df["longitude"].min(), df["longitude"].max()
        lat_min, lat_max = df["latitude"].min(), df["latitude"].max()

        lon_min, lon_max, lat_min, lat_max = inflate_bounds(
            lon_min, lon_max, lat_min, lat_max, buffer
        )

        print(f"  Longitude bounds: {lon_min} to {lon_max}")
        print(f"  Latitude bounds: {lat_min} to {lat_max}")

        # Download the maps
        relief = load_earth_relief(
            resolution="15s", region=[lon_min, lon_max, lat_min, lat_max]
        )
        relief.to_netcdf(
            os.path.join(output_directory, basename.replace(".csv", "_relief.nc"))
        )
        print(f"  Downloaded relief map: {relief.data.shape}.")

        gravity = load_earth_free_air_anomaly(
            resolution="01m", region=[lon_min, lon_max, lat_min, lat_max]
        )
        gravity.to_netcdf(
            os.path.join(output_directory, basename.replace(".csv", "_gravity.nc"))
        )
        print(f"  Downloaded gravity map: {gravity.data.shape}.")

        magnetic = load_earth_magnetic_anomaly(
            resolution="03m",
            region=[lon_min, lon_max, lat_min, lat_max],
            data_source="wdmam",
        )
        magnetic.to_netcdf(
            os.path.join(output_directory, basename.replace(".csv", "_magnetic.nc"))
        )
        print(f"  Downloaded magnetic map: {magnetic.data.shape}.")

        fig = plot_street_map(df["latitude"].tolist(), df["longitude"].tolist())
        fig.savefig(
            os.path.join(output_directory, basename.replace(".csv", "_street_map.png"))
        )
        plt.close(fig)
        print(f"  Saved street map.")


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Download geophysical maps from the GMT servers."
    )
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
