"""
Module for preprocessing data from the [sensor logger](https://github.com/tszheichoi/awesome-sensor-logger) app. Simple CLI interface for pre processing the data in a given directory.
"""

import os
from argparse import ArgumentParser
from concurrent import futures
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from pygmt.datasets import (
    load_earth_free_air_anomaly,
    load_earth_magnetic_anomaly,
    load_earth_relief,
)
from tqdm import tqdm

from analysis.plotting import inflate_bounds, plot_street_map


def clean_phone_data(dataset_path: Path | str) -> pd.DataFrame:
    """
    Clean the sensor logger app data from the given dataset path.

    Parameters
    ----------
    dataset_path : str
        Path to the dataset file.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with relevant columns.
    """
    assert os.path.exists(dataset_path), f"File {dataset_path} does not exist."
    # Assert the needed .csv files exist
    # assert os.path.exists(os.path.join(dataset_path, "Accelerometer.csv")), "Accelerometer.csv does not exist."
    assert os.path.exists(os.path.join(dataset_path, "Gyroscope.csv")), (
        "Gyroscope.csv does not exist."
    )
    # Check to make sure that the trajectory is of sufficient length (>=300 seconds)
    gyro = pd.read_csv(os.path.join(dataset_path, "Gyroscope.csv"), index_col=0)
    assert gyro["seconds_elapsed"].max() >= 300, (
        f"Trajectory is too short. Minimum required is 300 seconds. Trajectory is {gyro['seconds_elapsed'].max()} seconds."
    )
    assert os.path.exists(os.path.join(dataset_path, "Magnetometer.csv")), (
        "Magnetometer.csv does not exist."
    )
    assert os.path.exists(os.path.join(dataset_path, "Barometer.csv")), (
        "Barometer.csv does not exist."
    )
    assert os.path.exists(os.path.join(dataset_path, "Gravity.csv")), (
        "Gravity.csv does not exist."
    )
    try:
        assert os.path.exists(os.path.join(dataset_path, "LocationGps.csv")), (
            "LocationGps.csv does not exist."
        )
    except AssertionError:
        assert os.path.exists(os.path.join(dataset_path, "Location.csv")), (
            "Location.csv does not exist."
        )
    assert os.path.exists(os.path.join(dataset_path, "Orientation.csv")), (
        "Orientation.csv does not exist."
    )
    # Read in raw data
    gyroscope = pd.read_csv(os.path.join(dataset_path, "Gyroscope.csv"), index_col=0)
    magnetometer = pd.read_csv(
        os.path.join(dataset_path, "Magnetometer.csv"), index_col=0
    )
    barometer = pd.read_csv(os.path.join(dataset_path, "Barometer.csv"), index_col=0)
    gravity = pd.read_csv(os.path.join(dataset_path, "Gravity.csv"), index_col=0)
    orientation = pd.read_csv(
        os.path.join(dataset_path, "Orientation.csv"), index_col=0
    )
    try:
        location = pd.read_csv(
            os.path.join(dataset_path, "LocationGps.csv"), index_col=0
        )
    except FileNotFoundError:
        location = pd.read_csv(os.path.join(dataset_path, "Location.csv"), index_col=0)
    try:
        accelerometer = pd.read_csv(
            os.path.join(dataset_path, "TotalAcceleration.csv"), index_col=0
        )
    except FileNotFoundError as e:
        print(f"TotalAcceleration.csv not found, using Accelerometer.csv instead: {e}")
        accelerometer = pd.read_csv(
            os.path.join(dataset_path, "Accelerometer.csv"), index_col=0
        )
        accelerometer["x"] += gravity["x"]
        accelerometer["y"] += gravity["y"]
        accelerometer["z"] += gravity["z"]
    # Convert index to datetime
    accelerometer.index = pd.to_datetime(accelerometer.index, utc=True)  # type: ignore
    gyroscope.index = pd.to_datetime(gyroscope.index, utc=True)  # type: ignore
    magnetometer.index = pd.to_datetime(magnetometer.index, utc=True)  # type: ignore
    barometer.index = pd.to_datetime(barometer.index, utc=True)  # type: ignore
    gravity.index = pd.to_datetime(gravity.index, utc=True)  # type: ignore
    location.index = pd.to_datetime(location.index, utc=True)  # type: ignore
    orientation.index = pd.to_datetime(orientation.index, utc=True)  # type: ignore
    # Drop "seconds_elapsed" column
    accelerometer.drop(columns=["seconds_elapsed"], inplace=True)
    gyroscope.drop(columns=["seconds_elapsed"], inplace=True)
    magnetometer.drop(columns=["seconds_elapsed"], inplace=True)
    barometer.drop(columns=["seconds_elapsed"], inplace=True)
    gravity.drop(columns=["seconds_elapsed"], inplace=True)
    location.drop(columns=["seconds_elapsed"], inplace=True)
    orientation.drop(columns=["seconds_elapsed"], inplace=True)
    # Rename columns
    magnetometer = magnetometer.rename(
        columns={"x": "mag_x", "y": "mag_y", "z": "mag_z"}
    )
    accelerometer = accelerometer.rename(
        columns={"x": "acc_x", "y": "acc_y", "z": "acc_z"}
    )
    gyroscope = gyroscope.rename(columns={"x": "gyro_x", "y": "gyro_y", "z": "gyro_z"})
    gravity = gravity.rename(columns={"x": "grav_x", "y": "grav_y", "z": "grav_z"})
    # Merge dataframes
    data = location.copy()
    data = data.merge(orientation, left_index=True, right_index=True, how="outer")
    data = data.merge(accelerometer, left_index=True, right_index=True, how="outer")
    data = data.merge(gyroscope, left_index=True, right_index=True, how="outer")
    try:
        data = data.merge(magnetometer, left_index=True, right_index=True, how="outer")
    except Exception as e:
        print(f"Error merging magnetometer data: {e}")
    try:
        data = data.merge(barometer, left_index=True, right_index=True, how="outer")
    except Exception as e:
        print(f"Error merging barometer data: {e}")
    try:
        data = data.merge(gravity, left_index=True, right_index=True, how="outer")
    except Exception as e:
        print(f"Error merging gravity data: {e}")
    # Resample to 1Hz
    # fqy_str = convert_hz_to_time_str(frequency)
    # data = data.resample(fqy_str).mean()
    # data = data.dropna()
    # Convert index to datetime
    data.index = pd.to_datetime(data.index, utc=True)  # type: ignore
    # Ensure the index is sorted
    data.sort_index(inplace=True)
    # Drop all the previous rows before the first valid timestamp
    data = data[data.index >= location.index[0]]
    data = data.resample(
        "1s",
    ).mean()
    return data


def convert_hz_to_time_str(frequency: int) -> str:
    """Convert frequency in Hz to a time string."""
    if frequency <= 0:
        raise ValueError("Frequency must be positive.")
    interval = 1 / int(frequency)
    return f"{interval}s"


def preprocess_data(args):
    """Preprocess the data based on the provided arguments."""
    # datasets = Path(args.input_dir).glob("**/*.csv")
    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    # Find all CSV files under the input directory
    all_csv = list(input_path.rglob("*.csv"))

    # Check for folders directly under input_dir that contain CSV files.
    datasets = [d for d in input_path.iterdir() if d.is_dir() and any(d.glob("*.csv"))]

    print("found the following folders with data: ")
    for folder in datasets:
        print(folder)

    # Check to see if datasets in empty, if true ask if the user would lke to download the dataset
    if not datasets:
        download = input(
            "No datasets found. Would you like to download the dataset? (y/n): "
        )
        if download.lower() == "y":
            # Code to download the dataset goes here
            print("Fetch script is currently not implemented")
            pass
        else:
            print("No datasets found. Exiting.")
            return

    # os.makedirs(os.path.join("data", "cleaned"), exist_ok=True)
    # os.makedirs(args.output_dir, exist_ok=True)
    print(
        f"Preprocessing data from {args.input_dir}. Output will be saved to {args.output_dir}."
    )
    output_path.mkdir(parents=True, exist_ok=True)

    # def process_dataset(dataset: Path):
    for dataset in tqdm(datasets):
        # dataset_path = os.path.join(args.input_dir, dataset)
        cleaned_data = pd.DataFrame()
        # print(f"Processing: {dataset}")
        try:
            cleaned_data = clean_phone_data(dataset)
        except Exception as e:
            print(f"Error processing {dataset}: {e}")
            continue
        cleaned_csv_path = output_path / f"{dataset.name}.csv"
        # print(f"Saving data to: {cleaned_csv_path}")
        cleaned_data.to_csv(cleaned_csv_path)
        # print(f"Cleaned data for {dataset} saved to {cleaned_csv_path}.")

        street_map = plot_street_map(
            cleaned_data,
            margin=0.01,
            title=dataset.name,
        )
        street_map_path = output_path / f"{dataset.name}_street_map.png"
        # os.path.join(
        #    args.output_dir, f"{dataset.name}_street_map.png"
        # )
        street_map.savefig(street_map_path, dpi=300)
        # Close the figure to avoid accumulating open figures and memory usage
        plt.close(street_map)
        # print(f"Street map for {dataset.name} saved to {street_map_path}.")

        if args.getmaps:
            lon_min = cleaned_data["longitude"].min()
            lon_max = cleaned_data["longitude"].max()
            lat_min = cleaned_data["latitude"].min()
            lat_max = cleaned_data["latitude"].max()
            lon_min, lon_max, lat_min, lat_max = inflate_bounds(
                lon_min, lon_max, lat_min, lat_max, args.buffer
            )

            # Download the maps
            relief = load_earth_relief(
                resolution="15s", region=[lon_min, lon_max, lat_min, lat_max]
            )
            relief.to_netcdf(output_path / f"{dataset.name}_relief.nc")
            # print(f"  Downloaded relief map: {relief.data.shape}.")

            gravity = load_earth_free_air_anomaly(
                resolution="01m", region=[lon_min, lon_max, lat_min, lat_max]
            )
            gravity.to_netcdf(output_path / f"{dataset.name}_gravity.nc")
            # print(f"  Downloaded gravity map: {gravity.data.shape}.")

            magnetic = load_earth_magnetic_anomaly(
                resolution="03m",
                region=[lon_min, lon_max, lat_min, lat_max],
                data_source="wdmam",
            )
            magnetic.to_netcdf(output_path / f"{dataset.name}_magnetic.nc")


def main() -> None:
    """
    Main function to clean the sensor logger app data from the given base directory or plot routes from .csv files.
    """
    parser = ArgumentParser(description="Clean sensor logger app data or plot routes.")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/raw",
        help="Base directory for the sensor logger app data.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data",
        help="Output directory for the cleaned data.",
    )
    parser.add_argument(
        "--buffer",
        type=float,
        default=0.1,
        help="Buffer amount to inflate the bounding box by (as a percentage). Default is 0.1 (10 percent).",
    )
    parser.add_argument(
        "--getmaps",
        action="store_true",
        help="Whether to download geophysical maps for each trajectory.",
    )
    args = parser.parse_args()
    assert os.path.exists(args.input_dir), (
        f"Input directory {args.input_dir} does not exist."
    )
    preprocess_data(args)


if __name__ == "__main__":
    main()
