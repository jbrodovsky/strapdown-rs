"""
Module for preprocessing data from the [sensor logger](https://github.com/tszheichoi/awesome-sensor-logger) app. Simple CLI interface for pre processing the data in a given directory.
"""

import math
import os
import shutil
from argparse import ArgumentParser
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from analysis.plotting import inflate_bounds, plot_street_map

# pygmt is imported inside `download_maps` rather than here. It dlopens the GMT C library on
# import, so a module-scope import makes *importing this module* fail on a machine without
# GMT -- and `analysis/__init__.py` imports it, so that took down every subcommand including
# ones that touch no maps at all. Only map downloading needs it.

# The columns `build_event_stream` requires before it will emit an `Event::Imu`. A row missing
# any one of them is invisible to the simulator's propagation step, so a run of such rows is a
# gap in the inertial stream no matter how many GNSS fixes land inside it.
IMU_COLUMNS = ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z")

# What the *first* row of a segment needs for the filter to initialize. `run_closed_loop` seeds
# `StrapdownState` straight from record 0: latitude/longitude/altitude go in raw, so a NaN
# there poisons the entire run. speed/bearing and the quaternion degrade more gently --
# `ground_track_velocity` returns zero velocity and `attitude` returns identity -- but a
# segment that opens with an identity attitude starts badly misaligned, so require all of them.
INIT_COLUMNS = (
    "latitude",
    "longitude",
    "altitude",
    "speed",
    "bearing",
    "qw",
    "qx",
    "qy",
    "qz",
) + IMU_COLUMNS

# Matches `DEFAULT_MAX_IMU_GAP_S` in core/src/messages.rs. Past this the simulator rejects the
# whole file with `StrapdownError::SensorStreamGap`, so it is exactly the threshold that
# decides whether a recording has to be split to be usable at all. Keep the two in step.
DEFAULT_MAX_IMU_GAP_S = 5.0

# The same 300 s floor `clean_phone_data` already asserts on a whole recording, applied per
# segment: a 28 s fragment left over between two dropouts is not a trajectory.
DEFAULT_MIN_SEGMENT_S = 300.0


class Segment(NamedTuple):
    """One contiguous stretch of usable IMU data carved out of a single recording."""

    label: str
    """Suffix distinguishing this segment, `A`/`B`/`C`...; empty when the recording did not
    need splitting. Labels count raw segments, not surviving ones, so a segment keeps its
    letter when a neighbour is dropped and `--min-segment-s` can be changed without
    renumbering everything after it."""

    data: pd.DataFrame
    """The rows themselves; empty when `status` is anything but `kept`."""

    start_s: float
    """Offset of the segment's first row from the start of the source recording."""

    end_s: float
    """Offset of the segment's last row from the start of the source recording."""

    duration_s: float
    """`end_s - start_s`, the span actually written out."""

    status: str
    """`kept`, or the reason the segment was discarded."""


def segment_label(position: int) -> str:
    """
    Spreadsheet-style label for a zero-based segment index: A, B, ... Z, AA, AB, ...

    Parameters
    ----------
    position : int
        Zero-based index of the segment within its source recording.

    Returns
    -------
    str
        The label for that position.
    """
    label = ""
    position += 1
    while position > 0:
        position, remainder = divmod(position - 1, 26)
        label = chr(ord("A") + remainder) + label
    return label


def split_on_imu_gaps(
    data: pd.DataFrame,
    max_imu_gap_s: float = DEFAULT_MAX_IMU_GAP_S,
    min_segment_s: float = DEFAULT_MIN_SEGMENT_S,
) -> list[Segment]:
    """
    Split a cleaned recording into contiguous stretches of usable IMU data.

    Three of the recordings in this dataset lose their inertial sensors partway through while
    the GNSS receiver keeps reporting -- 912 s, 1610 s, and one that simply stops at 77% of the
    GPS span. The simulator refuses such a file outright rather than dead-reckoning across the
    hole, so the usable parts have to be separated before it ever sees them.

    Segments are also trimmed at both ends: forward from the start to the first row carrying a
    full initialization set, and back from the end to the last row carrying an IMU sample. The
    trailing trim is what rescues a truncated recording, which has no interior gap at all and
    so yields a single, shorter segment rather than a split.

    Parameters
    ----------
    data : pd.DataFrame
        A cleaned, resampled recording as returned by `clean_phone_data`.
    max_imu_gap_s : float
        Longest run without a usable IMU row that stays inside one segment. Must agree with
        `AidingConfig::max_imu_gap_s` on the simulator side or this will hand back segments the
        simulator still rejects. Zero or less keeps interior gaps but still trims the ends.
    min_segment_s : float
        Segments shorter than this are reported and dropped rather than written.

    Returns
    -------
    list[Segment]
        Every segment found, including dropped ones, in recording order. Callers that only
        want files to write should filter on `status == "kept"`.
    """
    if data.empty:
        return []

    elapsed = (data.index - data.index[0]).total_seconds().to_numpy()
    usable = data[list(IMU_COLUMNS)].notna().all(axis=1).to_numpy()
    present = np.flatnonzero(usable)
    if present.size == 0:
        return [Segment("", data.iloc[:0], 0.0, 0.0, 0.0, "no usable IMU rows")]

    if max_imu_gap_s > 0:
        breaks = np.flatnonzero(np.diff(elapsed[present]) > max_imu_gap_s)
        starts = np.concatenate(([present[0]], present[breaks + 1]))
        ends = np.concatenate((present[breaks], [present[-1]]))
    else:
        starts, ends = present[[0]], present[[-1]]

    split = len(starts) > 1
    segments: list[Segment] = []
    for position, (lo, hi) in enumerate(zip(starts, ends, strict=True)):
        label = segment_label(position) if split else ""
        window = data.iloc[lo : hi + 1]
        ready = np.flatnonzero(window[list(INIT_COLUMNS)].notna().all(axis=1).to_numpy())
        if ready.size == 0:
            segments.append(
                Segment(
                    label,
                    window.iloc[:0],
                    float(elapsed[lo]),
                    float(elapsed[hi]),
                    float(elapsed[hi] - elapsed[lo]),
                    "no row carries a GNSS fix, an attitude and an IMU sample at once",
                )
            )
            continue
        lo += int(ready[0])
        duration = float(elapsed[hi] - elapsed[lo])
        if duration < min_segment_s:
            segments.append(
                Segment(
                    label,
                    data.iloc[:0],
                    float(elapsed[lo]),
                    float(elapsed[hi]),
                    duration,
                    f"shorter than the {min_segment_s:g} s minimum",
                )
            )
            continue
        segments.append(
            Segment(
                label,
                data.iloc[lo : hi + 1],
                float(elapsed[lo]),
                float(elapsed[hi]),
                duration,
                "kept",
            )
        )
    return segments


def clean_phone_data(dataset_path: Path | str, frequency: int = 1) -> pd.DataFrame:
    """
    Clean the sensor logger app data from the given dataset path.

    Parameters
    ----------
    dataset_path : str
        Path to the dataset file.
    frequency : int
        Output sample rate in Hz. The recording's inertial sensors run at 13-100 Hz while
        its GNSS receiver only ever reports at ~1 Hz, so a rate above 1 leaves the GNSS
        columns NaN in all but every Nth row. That is intentional and is what the simulator
        wants: `build_event_stream` skips epochs whose GNSS columns are NaN, giving
        high-rate inertial propagation with 1 Hz aiding. Do not interpolate the GNSS
        columns up to `frequency` -- that fabricates fixes that were never measured.

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
    assert os.path.exists(os.path.join(dataset_path, "Gravity.csv")), "Gravity.csv does not exist."
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
    magnetometer = pd.read_csv(os.path.join(dataset_path, "Magnetometer.csv"), index_col=0)
    barometer = pd.read_csv(os.path.join(dataset_path, "Barometer.csv"), index_col=0)
    gravity = pd.read_csv(os.path.join(dataset_path, "Gravity.csv"), index_col=0)
    orientation = pd.read_csv(os.path.join(dataset_path, "Orientation.csv"), index_col=0)
    try:
        location = pd.read_csv(os.path.join(dataset_path, "LocationGps.csv"), index_col=0)
    except FileNotFoundError:
        location = pd.read_csv(os.path.join(dataset_path, "Location.csv"), index_col=0)
    try:
        accelerometer = pd.read_csv(
            os.path.join(dataset_path, "TotalAcceleration.csv"), index_col=0
        )
    except FileNotFoundError as e:
        print(f"TotalAcceleration.csv not found, using Accelerometer.csv instead: {e}")
        accelerometer = pd.read_csv(os.path.join(dataset_path, "Accelerometer.csv"), index_col=0)
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
    magnetometer = magnetometer.rename(columns={"x": "mag_x", "y": "mag_y", "z": "mag_z"})
    accelerometer = accelerometer.rename(columns={"x": "acc_x", "y": "acc_y", "z": "acc_z"})
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
    # Convert index to datetime
    data.index = pd.to_datetime(data.index, utc=True)  # type: ignore
    # Ensure the index is sorted
    data.sort_index(inplace=True)
    # Drop all the previous rows before the first valid timestamp
    data = data[data.index >= location.index[0]]
    data = data.resample(convert_hz_to_time_str(frequency)).mean()
    return data


def convert_hz_to_time_str(frequency: int) -> str:
    """Convert frequency in Hz to a time string."""
    if frequency <= 0:
        raise ValueError("Frequency must be positive.")
    interval = 1 / int(frequency)
    return f"{interval}s"


def preprocess_data(args):
    """Preprocess the data based on the provided arguments."""
    input_path = Path(args.input)
    output_path = Path(args.output)
    # Check for folders directly under input that contain CSV files.
    datasets = [d for d in input_path.iterdir() if d.is_dir() and any(d.glob("*.csv"))]

    print("found the following folders with data: ")
    for folder in datasets:
        print(folder)

    # Check to see if datasets in empty, if true ask if the user would lke to download the dataset
    if not datasets:
        download = input("No datasets found. Would you like to download the dataset? (y/n): ")
        if download.lower() == "y":
            # Code to download the dataset goes here
            print("Fetch script is currently not implemented")
            pass
        else:
            print("No datasets found. Exiting.")
            return

    print(f"Preprocessing data from {args.input}. Output will be saved to {args.output}.")
    output_path.mkdir(parents=True, exist_ok=True)

    manifest: list[dict] = []

    # def process_dataset(dataset: Path):
    for dataset in tqdm(datasets):
        # print(f"Processing: {dataset}")
        try:
            cleaned_data = clean_phone_data(dataset, frequency=args.frequency)
        except Exception as e:
            print(f"Error processing {dataset}: {e}")
            manifest.append(manifest_row(dataset.name, status=f"error: {e}"))
            continue

        segments = segment_recording(cleaned_data, args)
        for segment in segments:
            manifest.append(
                manifest_row(
                    dataset.name,
                    label=segment.label,
                    output=f"{segment_stem(dataset.name, segment)}.csv"
                    if segment.status == "kept"
                    else "",
                    start_s=round(segment.start_s, 1),
                    end_s=round(segment.end_s, 1),
                    duration_s=round(segment.duration_s, 1),
                    rows=len(segment.data),
                    status=segment.status,
                )
            )

        kept = [segment for segment in segments if segment.status == "kept"]
        for segment in kept:
            if segment.label:
                print(
                    f"  {dataset.name}: segment {segment.label} covers "
                    f"t={segment.start_s:.1f}..{segment.end_s:.1f}s ({len(segment.data)} rows)"
                )
            write_segment(segment, dataset.name, output_path, args)

        for segment in segments:
            if segment.status != "kept":
                print(
                    f"  {dataset.name}: dropped segment {segment.label or '-'} "
                    f"(t={segment.start_s:.1f}..{segment.end_s:.1f}s) -- {segment.status}"
                )
        if not kept:
            print(f"  {dataset.name}: no usable segments, nothing written")

    # JSON, not CSV, and deliberately so: this file lands in the directory the simulator is
    # pointed at, and `resolve_input_files` takes *every* `*.csv` in that directory as a
    # trajectory. A `segments.csv` here would be loaded as one, silently adding a junk run to
    # every sweep and every aggregate built from it.
    manifest_path = output_path / "segments.json"
    pd.DataFrame(manifest).to_json(manifest_path, orient="records", indent=2)
    written = sum(1 for row in manifest if row.get("status") == "kept")
    print(
        f"Wrote {written} trajectory file(s) from {len(datasets)} recording(s). "
        f"Per-segment detail in {manifest_path}."
    )
    report_orphans(manifest, output_path, prune=args.prune)


def manifest_row(
    source: str,
    label: str = "",
    output: str = "",
    start_s: float = 0.0,
    end_s: float = 0.0,
    duration_s: float = 0.0,
    rows: int = 0,
    status: str = "",
) -> dict:
    """
    Build one manifest record, with every key present.

    A recording that raises has no segments to describe, but it still gets a full row: pandas
    infers a column's dtype across all the records, and one row missing `duration_s` turns
    that column into a nullable object column that then formats as `None`.

    Parameters
    ----------
    source : str
        Name of the source recording.
    label : str
        Segment label, empty for an unsplit recording or a failed one.
    output : str
        File name written, empty when nothing was.
    start_s, end_s, duration_s : float
        Segment bounds relative to the start of the source recording.
    rows : int
        Number of rows written.
    status : str
        `kept`, a drop reason, or `error: ...`.

    Returns
    -------
    dict
        The record.
    """
    return {
        "source": source,
        "label": label,
        "output": output,
        "start_s": start_s,
        "end_s": end_s,
        "duration_s": duration_s,
        "rows": rows,
        "status": status,
    }


def report_orphans(manifest: list[dict], output_path: Path, prune: bool = False) -> None:
    """
    Report CSVs in the output directory that this run did not produce.

    Splitting renames things: `2025-11-09_17-34-01.csv` becomes `_A`/`_B`/`_C`, and the
    original stays behind unless something removes it. That leftover is not inert -- the
    simulator globs every `*.csv` in this directory, so it would go on being run, go on
    failing on the gap that prompted the split, and go on being counted.

    Only CSVs and their street maps are considered. A split recording's own `.nc` maps look
    orphaned by the same test -- nothing is named after the parent any more -- but they are
    what `inherit_parent_maps` copies from, so deleting them would leave the segments with no
    maps on the next run into a fresh directory. Do not widen this to `*.nc`.

    Parameters
    ----------
    manifest : list[dict]
        The per-segment records built by `preprocess_data`.
    output_path : Path
        Directory the trajectories were written to.
    prune : bool
        Delete the orphans rather than only naming them, but only those traceable to a
        recording that preprocessed cleanly this run. A file left behind by a recording that
        *failed* is never deleted: preprocessing fails for transient reasons too, and a stale
        file is recoverable where a deleted one may not be. Off by default for the same
        reason -- this directory is the user's.
    """
    expected = {row["output"] for row in manifest if row.get("output")}
    orphans = sorted(path for path in output_path.glob("*.csv") if path.name not in expected)
    if not orphans:
        return

    healthy = {
        row["source"] for row in manifest if not str(row.get("status", "")).startswith("error")
    }
    print(
        f"Found {len(orphans)} CSV file(s) in {output_path} that this run did not write. "
        "The simulator loads every CSV in this directory, so a superseded file is still run "
        "and still counted:"
    )
    kept_back = 0
    for path in orphans:
        source = orphan_source(path.stem, healthy)
        if prune and source is not None:
            path.unlink()
            # The street map is the CSV's sibling, drawn from exactly the rows just deleted.
            # Left behind it depicts a trajectory that is no longer an input at all.
            path.with_name(f"{path.stem}_street_map.png").unlink(missing_ok=True)
            print(f"  {path.name} -- removed, superseded by {source}")
        elif source is not None:
            print(f"  {path.name} -- superseded by {source}")
        else:
            kept_back += 1
            print(f"  {path.name} -- left alone, no recording produced it this run")
    if not prune:
        print("  Re-run with --prune to delete the superseded ones.")
    elif kept_back:
        print(
            f"  {kept_back} left in place: delete them by hand once you know why their "
            "recording did not preprocess."
        )


def orphan_source(stem: str, sources: set[str]) -> str | None:
    """
    Find the recording an orphaned output file came from, if it preprocessed cleanly.

    Parameters
    ----------
    stem : str
        File stem of the orphan, e.g. `2025-11-09_17-34-01` or `2025-11-09_17-34-01_B`.
    sources : set[str]
        Names of recordings that preprocessed without raising this run.

    Returns
    -------
    str | None
        The source recording's name, or None if the orphan cannot be traced to one that
        preprocessed cleanly -- in which case it is not safe to delete on this evidence.
    """
    if stem in sources:
        return stem
    source, _, label = stem.rpartition("_")
    if source in sources and label.isalpha() and label.isupper():
        return source
    return None


def segment_recording(cleaned_data: pd.DataFrame, args) -> list[Segment]:
    """
    Apply the configured splitting policy to one cleaned recording.

    Parameters
    ----------
    cleaned_data : pd.DataFrame
        A cleaned, resampled recording as returned by `clean_phone_data`.
    args
        Parsed arguments carrying `no_split`, `max_imu_gap_s` and `min_segment_s`.

    Returns
    -------
    list[Segment]
        The segments to write, plus any that were dropped. With `--no-split` this is always
        the whole recording as a single unlabelled segment, untrimmed, which is the pre-split
        behaviour of this script.
    """
    if args.no_split:
        span = 0.0
        if not cleaned_data.empty:
            span = float((cleaned_data.index[-1] - cleaned_data.index[0]).total_seconds())
        return [Segment("", cleaned_data, 0.0, span, span, "kept")]
    return split_on_imu_gaps(
        cleaned_data,
        max_imu_gap_s=args.max_imu_gap_s,
        min_segment_s=args.min_segment_s,
    )


def segment_stem(source_name: str, segment: Segment) -> str:
    """
    File stem for a segment: the recording name, suffixed only when it was actually split.

    Parameters
    ----------
    source_name : str
        Name of the source recording directory.
    segment : Segment
        The segment being written.

    Returns
    -------
    str
        `name` for an unsplit recording, `name_A` / `name_B` / ... for a split one.
    """
    return f"{source_name}_{segment.label}" if segment.label else source_name


#: Absolute map margin, kilometres, added around every trajectory's bounding box.
#:
#: Sized for how far the filter can wander off the recorded track, which is what decides
#: whether a geophysical update finds the map. `conf/*_degraded.toml` never fully withholds
#: GNSS -- fixes still arrive every 5 s, just noisier -- so this margin is a generous, not a
#: tightly-derived, bound; 5 km is cheap too -- on a 17 km track it is about a 30% pad.
DEFAULT_MAP_MARGIN_KM = 5.0

#: Metres per degree of latitude. Matches `METRES_PER_DEGREE` in `analysis/geostats.py`, which
#: uses it in the other direction to turn grid spacing into ground distance.
METRES_PER_DEGREE = 111_320.0


def pad_bounds(
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    buffer: float,
    margin_km: float,
) -> tuple[float, float, float, float]:
    """
    Pad a trajectory's bounding box by the larger of a fractional and an absolute margin.

    `inflate_bounds` alone is not enough, for two reasons, and the second is the one that
    bites:

    1. **The thing being guarded against is an absolute distance.** A map needs to extend
       past the track by however far the filter can wander off it, and that does not scale
       with how far the vehicle drove. At a 10% fraction a 2 km track gets 200 m of margin
       and a 50 km track gets 5 km, for the same 120 s outage.
    2. **`inflate_bounds` scales each axis by its own range**, so a track that is straight in
       one axis gets almost no margin in the other. A due-north drive has a longitude range
       near zero, so its longitude pad is near zero, and any eastward excursion leaves the
       map. No choice of `buffer` fixes that -- zero times anything is zero.

    Either way the failure is silent at this end: the map is written successfully, and the
    run dies later and per-trajectory on `OutOfMapBounds`.

    Taking the larger of the two keeps the fractional behaviour for long tracks, where it is
    already generous, and puts a floor under short and straight ones.

    Parameters
    ----------
    lon_min, lon_max, lat_min, lat_max : float
        The trajectory's bounding box, degrees.
    buffer : float
        Fractional margin, as `inflate_bounds` takes it -- 0.1 is 10% of each axis's range.
    margin_km : float
        Absolute margin in kilometres, applied on every side.

    Returns
    -------
    tuple
        `(lon_min, lon_max, lat_min, lat_max)`, padded.
    """
    fractional = inflate_bounds(lon_min, lon_max, lat_min, lat_max, buffer)
    if margin_km <= 0:
        return fractional

    # Converted at the box's mean latitude: a degree of longitude shortens towards the poles,
    # so a fixed number of kilometres is more degrees of longitude the further north you are.
    # Clamped because cos() reaches zero at the pole and the conversion diverges.
    mean_lat = (lat_min + lat_max) / 2.0
    margin_m = margin_km * 1000.0
    d_lat = margin_m / METRES_PER_DEGREE
    d_lon = margin_m / (METRES_PER_DEGREE * max(math.cos(math.radians(mean_lat)), 1e-6))

    frac_lon_min, frac_lon_max, frac_lat_min, frac_lat_max = fractional
    return (
        min(frac_lon_min, lon_min - d_lon),
        max(frac_lon_max, lon_max + d_lon),
        min(frac_lat_min, lat_min - d_lat),
        max(frac_lat_max, lat_max + d_lat),
    )


def inherit_parent_maps(source_name: str, stem: str, output_path: Path) -> None:
    """
    Give a split segment the geophysical maps of the recording it came from.

    `find_gravity_map` and `find_magnetic_map` in sim/src/main.rs look for
    `<stem>_gravity.nc` and `<stem>_magnetic.nc` beside the trajectory and fail the run
    outright when they are missing, so splitting a recording without this leaves every
    segment unable to do geophysical navigation at all.

    Copying rather than re-downloading is exact, not an approximation: a segment is a subset
    of its parent's rows, so the parent's bounding box necessarily contains it. The map is
    looser than one fitted to the segment would be, which costs nothing but disk. `--getmaps`
    fits each segment its own.

    Only gravity and magnetic are copied. `_relief.nc` is used for plotting and never read by
    the simulator, and it is the large one -- 195 MB against 13 MB for gravity on the longest
    recording here.

    Parameters
    ----------
    source_name : str
        Name of the source recording, whose maps are the ones to inherit.
    stem : str
        File stem of the segment. A no-op when it equals `source_name`, since an unsplit
        recording already owns its maps.
    output_path : Path
        Directory holding the maps.
    """
    if stem == source_name:
        return
    for kind in ("gravity", "magnetic"):
        parent = output_path / f"{source_name}_{kind}.nc"
        child = output_path / f"{stem}_{kind}.nc"
        # Never clobber: a map already sitting at the segment's own name was fitted to that
        # segment by an earlier `--getmaps` run and is tighter than the parent's.
        if parent.exists() and not child.exists():
            shutil.copyfile(parent, child)


def write_segment(segment: Segment, source_name: str, output_path: Path, args) -> None:
    """
    Write one segment's CSV, street map and -- with `--getmaps` -- its geophysical maps.

    Parameters
    ----------
    segment : Segment
        The segment to write. Must have `status == "kept"`.
    source_name : str
        Name of the source recording directory.
    output_path : Path
        Directory to write into.
    args
        Parsed arguments carrying `getmaps` and `buffer`.
    """
    stem = segment_stem(source_name, segment)
    segment.data.to_csv(output_path / f"{stem}.csv")

    street_map = plot_street_map(segment.data, margin=0.01, title=stem)
    street_map.savefig(output_path / f"{stem}_street_map.png", dpi=300)
    # Close the figure to avoid accumulating open figures and memory usage
    plt.close(street_map)

    if not args.getmaps:
        inherit_parent_maps(source_name, stem, output_path)
        return

    lon_min = segment.data["longitude"].min()
    lon_max = segment.data["longitude"].max()
    lat_min = segment.data["latitude"].min()
    lat_max = segment.data["latitude"].max()
    lon_min, lon_max, lat_min, lat_max = pad_bounds(
        lon_min,
        lon_max,
        lat_min,
        lat_max,
        buffer=args.buffer,
        margin_km=getattr(args, "margin_km", DEFAULT_MAP_MARGIN_KM),
    )

    # Download the maps. See the note at the top of this module for why pygmt is imported
    # here rather than at module scope.
    from pygmt.datasets import (
        load_earth_free_air_anomaly,
        load_earth_magnetic_anomaly,
        load_earth_relief,
    )

    relief = load_earth_relief(resolution="15s", region=[lon_min, lon_max, lat_min, lat_max])
    relief.to_netcdf(output_path / f"{stem}_relief.nc")

    gravity = load_earth_free_air_anomaly(
        resolution="01m", region=[lon_min, lon_max, lat_min, lat_max]
    )
    gravity.to_netcdf(output_path / f"{stem}_gravity.nc")

    magnetic = load_earth_magnetic_anomaly(
        resolution="03m",
        region=[lon_min, lon_max, lat_min, lat_max],
        data_source="wdmam",
    )
    magnetic.to_netcdf(output_path / f"{stem}_magnetic.nc")


def add_preprocess_arguments(parser) -> None:
    """
    Register the sample-rate and IMU-gap-splitting options on a preprocessing parser.

    Shared by this module's standalone `main` and the `analyze preprocess` subcommand so the
    two cannot drift apart -- they did once already, when `--frequency` was added to only one
    of them and the other raised `AttributeError` on every run.

    Parameters
    ----------
    parser
        An `ArgumentParser` or subparser to add the options to.
    """
    parser.add_argument(
        "-f",
        "--frequency",
        type=int,
        default=1,
        help=(
            "Output sample rate in Hz (default 1). The phone's inertial sensors run at "
            "13-100 Hz but its GNSS receiver only reports at ~1 Hz, so a higher rate raises "
            "the inertial propagation rate while GNSS stays where it was actually measured."
        ),
    )
    parser.add_argument(
        "--margin-km",
        type=float,
        default=DEFAULT_MAP_MARGIN_KM,
        help=(
            f"Absolute map margin in kilometres, applied on every side in addition to "
            f"`--buffer` (default {DEFAULT_MAP_MARGIN_KM:g}). The larger of the two wins. "
            "`--buffer` alone is a fraction of the track's own extent, which under-pads a "
            "short track and gives a straight one almost no margin in the other axis -- a "
            "due-north drive has near-zero longitude range, so a fraction of it is near "
            "zero. The filter then leaves the map mid-run. Set 0 to use `--buffer` alone."
        ),
    )
    parser.add_argument(
        "--max-imu-gap-s",
        type=float,
        default=DEFAULT_MAX_IMU_GAP_S,
        help=(
            f"Split a recording wherever its IMU goes quiet for longer than this, in seconds "
            f"(default {DEFAULT_MAX_IMU_GAP_S:g}). Must match `max_imu_gap_s` on the "
            "simulator side, which rejects any file containing a longer gap."
        ),
    )
    parser.add_argument(
        "--min-segment-s",
        type=float,
        default=DEFAULT_MIN_SEGMENT_S,
        help=(
            f"Discard segments shorter than this, in seconds (default "
            f"{DEFAULT_MIN_SEGMENT_S:g}). They are reported in segments.json either way."
        ),
    )
    parser.add_argument(
        "--no-split",
        action="store_true",
        help=(
            "Write one file per recording regardless of IMU gaps, untrimmed. Recordings with "
            "a gap will then be rejected by the simulator rather than split."
        ),
    )
    parser.add_argument(
        "--prune",
        action="store_true",
        help=(
            "Delete CSVs in the output directory that this run did not write, such as the "
            "un-split original of a recording that has since been split. Without it they are "
            "only reported, and the simulator goes on loading them."
        ),
    )


def main() -> None:
    """
    Main function to clean the sensor logger app data from the given base directory or plot routes from .csv files.
    """
    parser = ArgumentParser(description="Clean sensor logger app data or plot routes.")
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw",
        help="Base directory for the sensor logger app data.",
    )
    parser.add_argument(
        "--output",
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
    add_preprocess_arguments(parser)
    args = parser.parse_args()
    assert os.path.exists(args.input), f"Input directory {args.input} does not exist."
    preprocess_data(args)


if __name__ == "__main__":
    main()
