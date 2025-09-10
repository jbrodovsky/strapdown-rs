import argparse
import os

import matplotlib
import pandas as pd
from cartopy import crs as ccrs
from cartopy.io import img_tiles as cimgt
from tqdm import tqdm

matplotlib.use("Agg")
from typing import Optional

from matplotlib import pyplot as plt


def inflate_bounds(
    min_x: float, min_y: float, max_x: float, max_y: float, inflation_percent: float
) -> tuple[float, float, float, float]:
    width = max_x - min_x
    height = max_y - min_y
    if width <= 1e-6:
        width = 0.1
    if height <= 1e-6:
        height = 0.1
    inflate_x = width * inflation_percent
    inflate_y = height * inflation_percent
    new_min_x = min_x - inflate_x
    new_min_y = min_y - inflate_y
    new_max_x = max_x + inflate_x
    new_max_y = max_y + inflate_y
    return new_min_x, new_min_y, new_max_x, new_max_y


def plot_route(
    cleaned_data: pd.DataFrame, output_path: str, title: Optional[str] = None
):
    """
    Plot the route from cleaned data and save to output_path.
    """
    west_lon = cleaned_data["longitude"].min()
    east_lon = cleaned_data["longitude"].max()
    south_lat = cleaned_data["latitude"].min()
    north_lat = cleaned_data["latitude"].max()
    west_lon, south_lat, east_lon, north_lat = inflate_bounds(
        west_lon, south_lat, east_lon, north_lat, 0.1
    )
    extent = [west_lon, east_lon, south_lat, north_lat]
    request = cimgt.GoogleTiles()
    ax = plt.axes(projection=request.crs)
    ax.set_extent(extent)  # type: ignore
    ax.add_image(request, 15)  # type: ignore
    ax.scatter(
        cleaned_data["longitude"],
        cleaned_data["latitude"],
        0.5,
        color="red",
        transform=ccrs.PlateCarree(),
    )
    if title:
        ax.set_title(title)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Route plot saved to {output_path}.")


def main():
    parser = argparse.ArgumentParser(
        description="Plot route(s) from cleaned data CSV file(s)."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input CSV file or directory containing CSV files with longitude/latitude columns.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for route plot images.",
    )
    parser.add_argument(
        "--title", type=str, default=None, help="Optional title for the plot(s)."
    )
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    if os.path.isfile(args.input):
        df = pd.read_csv(args.input)
        if "longitude" in df.columns and "latitude" in df.columns:
            out_name = os.path.splitext(os.path.basename(args.input))[0] + "_route.png"
            out_path = os.path.join(args.output, out_name)
            plot_route(df, out_path, title=args.title)
        else:
            print(
                f"Input file {args.input} does not contain longitude/latitude columns."
            )
    elif os.path.isdir(args.input):
        for root, _, files in os.walk(args.input):
            for file in tqdm(files):
                if file.endswith(".csv"):
                    csv_path = os.path.join(root, file)
                    try:
                        df = pd.read_csv(csv_path)
                        if "longitude" in df.columns and "latitude" in df.columns:
                            out_name = os.path.splitext(file)[0] + "_route.png"
                            out_path = os.path.join(args.output, out_name)
                            plot_route(df, out_path, title=args.title or file)
                    except Exception as e:
                        print(f"Error plotting {csv_path}: {e}")
    else:
        print(f"Input path {args.input} is not a valid file or directory.")


if __name__ == "__main__":
    main()
