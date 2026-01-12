"""analysis.plotting

Utilities for visualizing navigation trajectories and geophysical maps used
in the Strapdown-rs analysis workflows.

This module provides helper functions to plot performance time-series,
overlay trajectories on OpenStreetMap tiles (`plot_street_map`), and render
geophysical grids (terrain relief, free-air gravity, magnetic anomaly) with
PyGMT (`plot_geo_map`).

Conventions
- Latitude/longitude are expected in decimal degrees (WGS84).
- Trajectory `nav` and optional `gps` inputs are `pandas.DataFrame` objects
    containing `latitude` and `longitude` columns (degrees). When altitude is
    used (e.g., `plot_performance`) it is expected to be in metres.

Examples
```
from analysis.src.analysis.plotting import plot_geo_map, GeophysicalType
fig = plot_geo_map(nav=df_nav, geo_map="data/input/my_relief.nc", map_type=GeophysicalType.RELIEF)
fig.show()
```

References
- PyGMT: https://www.pygmt.org
"""

from cartopy import crs as ccrs
from cartopy.io import img_tiles as cimgt
from haversine import Unit, haversine_vector
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from pathlib import Path
from pandas import DataFrame
from pygmt.io import load_dataarray
from enum import Enum
from typing import Union
import pygmt
import xarray as xr


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


class GeophysicalType(Enum):
    RELIEF = "relief"
    GRAVITY = "gravity"
    MAGNETIC = "magnetic"
    """Enumeration of supported geophysical map types.

    RELIEF: Terrain elevation (e.g., digital elevation models).
    GRAVITY: Free-air gravity anomaly grids (units depend on source, often mGal).
    MAGNETIC: Magnetic anomaly grids (units depend on source, often nT).
    """


def plot_performance(nav: DataFrame, gps: DataFrame, output_path: Path | str):
    """Plot INS performance metrics compared with GPS/Truth.

    Produces a time-series showing 2D horizontal haversine error between the
    `nav` solution and `gps` truth, altitude error, and GPS-reported
    horizontal/vertical accuracy. Intended for quick visual checks of filter
    performance in simulation.

    Arguments
    - nav: `pandas.DataFrame` containing at least `latitude`, `longitude`, and
      `altitude` columns. Index must be a `DatetimeIndex` or convertible to one
      so that `(nav.index - nav.index[0]).total_seconds()` yields seconds.
    - gps: `pandas.DataFrame` containing GPS truth with columns
      `latitude`, `longitude`, `altitude`, `horizontalAccuracy`, and
      `verticalAccuracy`. Index should align with `nav` timing or be
      time-indexed as well.
    - output_path: path to write the PNG output.

    Notes
    - Latitude/longitude are expected in decimal degrees (WGS84).
    - Altitude and error units are metres.
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
    # ax.set_ylim((0, 50))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("2D Haversine Error (m)")
    ax.set_title(
        "Strapdown INS Simulation Performance with GPS Comparison", fontsize=16
    )
    ax.grid()
    ax.legend()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_relative_performance(
    geo: DataFrame,
    deg: DataFrame,  # degraded/baseline navigation
    nav: DataFrame,  # GPS-aided truth
    output_path: Path | str,
):
    """Plot relative performance: geophysical-aided nav w.r.t. degraded nav.

    This mirrors the geonav-sim notebook visualization by plotting the
    difference of the cumulative root-mean-squared error series:

        root_mean_geo_cum_error - root_mean_deg_cum_error

    Positive values are filled red, negative values green.
    """
    output_path = Path(output_path)

    distance_traveled = haversine_vector(
        nav[["latitude", "longitude"]].to_numpy()[:-1, :],
        nav[["latitude", "longitude"]].to_numpy()[1:, :],
        Unit.METERS,
    )
    distance_traveled = np.hstack(([0], distance_traveled))
    distance_traveled = np.nancumsum(distance_traveled)

    geo_error = haversine_vector(
        geo[["latitude", "longitude"]].to_numpy(dtype=np.float64, copy=False),
        nav[["latitude", "longitude"]].to_numpy(),
        Unit.METERS,
    )

    deg_error = haversine_vector(
        deg[["latitude", "longitude"]].to_numpy(),
        nav[["latitude", "longitude"]].to_numpy(),
        Unit.METERS,
    )

    time = (nav.index - nav.index[0]).total_seconds() / 3600

    err_diff = geo_error - deg_error
    geo_rmse = np.sqrt(np.nanmean(geo_error**2))
    deg_rmse = np.sqrt(np.nanmean(deg_error**2))
    # General errors
    fig, ax = plt.subplots(1, 1, figsize=(24, 6), layout="tight")
    ax.plot(time, err_diff, label="Error Difference", color="black", linewidth=1)
    ax.fill_between(
        time,
        err_diff,
        0,
        where=err_diff > 0,
        alpha=0.4,
        color="red",
        label="Geo INS > Degraded INS",
    )
    ax.fill_between(
        time,
        err_diff,
        0,
        where=err_diff < 0,
        alpha=0.4,
        color="green",
        label="Degraded INS > Geo INS",
    )
    ax.set_xlim(left=0)
    # ax[0].set_ylim((-0.1, 0.1))
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Distance (m)")
    ax.set_title(
        f"Geophysical Navigation Performance | RMSE difference: {geo_rmse - deg_rmse:0.2f}",
        fontsize=16,
    )
    ax.grid()
    ax.legend()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_street_map(
    nav: DataFrame,
    margin=0.01,
    title: str | None = None,
    gps: DataFrame | None = None,
) -> Figure:
    """Render a trajectory on an OpenStreetMap background.

    This function draws the trajectory contained in ``nav`` over OpenStreetMap
    tiles using Cartopy. ``nav`` must be a :class:`pandas.DataFrame` with
    ``latitude`` and ``longitude`` columns (decimal degrees, WGS84). The
    ``nav`` trajectory is plotted in a distinct colour (default ``tab:red``).
    If a ``gps`` DataFrame is supplied it is rendered in black (``GPS/Truth``)
    to make comparison straightforward.

    Returns
    - ``matplotlib.figure.Figure``: the created figure so callers can save or
      further modify it.
    """
    lat_arr = nav["latitude"].to_numpy(dtype=float)
    lon_arr = nav["longitude"].to_numpy(dtype=float)
    nav_color = "tab:red"

    valid_mask = np.isfinite(lat_arr) & np.isfinite(lon_arr)
    if not np.any(valid_mask):
        raise ValueError("No valid latitude/longitude points to plot.")

    lat_clean = lat_arr[valid_mask]
    lon_clean = lon_arr[valid_mask]

    # Define the map extent using only valid coordinates
    lat_min, lat_max = float(np.min(lat_clean)), float(np.max(lat_clean))
    lon_min, lon_max = float(np.min(lon_clean)), float(np.max(lon_clean))

    # Create an OSM tiles instance
    osm_tiles = cimgt.OSM()

    # Create a figure using Cartopy with OpenStreetMap background
    fig, ax = plt.subplots(
        figsize=(12, 10), subplot_kw={"projection": ccrs.PlateCarree()}
    )
    ax.set_extent(
        [lon_min - margin, lon_max + margin, lat_min - margin, lat_max + margin],
        crs=ccrs.PlateCarree(),
    )  # ty:ignore[unresolved-attribute]

    # Add the OSM tiles to the map
    ax.add_image(osm_tiles, 12)  # ty:ignore[invalid-argument-type, too-many-positional-arguments]

    # Plot the trajectory (nav) using a distinct color
    ax.plot(
        lon_clean,
        lat_clean,
        "-",
        color=nav_color,
        transform=ccrs.PlateCarree(),
        label="Nav",
    )

    # Optionally plot GPS/Truth in black if provided
    if gps is not None:
        try:
            gps_lat = gps["latitude"].to_numpy(dtype=float)
            gps_lon = gps["longitude"].to_numpy(dtype=float)
            gps_mask = np.isfinite(gps_lat) & np.isfinite(gps_lon)
            if np.any(gps_mask):
                ax.plot(
                    gps_lon[gps_mask],
                    gps_lat[gps_mask],
                    "k--",
                    transform=ccrs.PlateCarree(),
                    label="GPS/Truth",
                )
        except Exception:
            pass

    # Add gridlines with labels
    gl = ax.gridlines(draw_labels=True, alpha=0.2)  # ty:ignore[unresolved-attribute]
    gl.top_labels = False
    gl.right_labels = False

    if title is None:
        ax.set_title("Street Map with Trajectory Points", fontsize=16)
    else:
        ax.set_title(title, fontsize=16)

    ax.legend()
    return fig


def plot_geo_map(
    nav: DataFrame | list[float] | None,
    geo_map: Union[str, Path, xr.DataArray],
    map_type: GeophysicalType,
    gps: DataFrame | None = None,
    margin: float = 0.01,
    title: str | None = None,
) -> pygmt.Figure:
    """Render a geophysical grid with an overlaid trajectory using PyGMT.

    The `geo_map` argument may be a path to a grid file (NetCDF-style) or an
    `xarray.DataArray` already loaded into memory. The function draws the
    grid using `grdimage`, builds a continuous colormap (CPT) appropriate for
    the `map_type`, and adds contours with `grdcontour` for visual relief.
    """
    # Extract lat/lon from nav argument (support DataFrame or lists)
    if nav is None:
        raise ValueError("`nav` must be provided as a DataFrame or lat/lon lists")

    if isinstance(nav, DataFrame):
        lat_arr = nav["latitude"].to_numpy(dtype=float)
        lon_arr = nav["longitude"].to_numpy(dtype=float)
    else:
        # expect a sequence [lat_list, lon_list]
        if not isinstance(nav, (list, tuple)) or len(nav) != 2:
            raise ValueError(
                "If `nav` is not a DataFrame, provide (lat_list, lon_list)"
            )
        lat_arr = np.asarray(nav[0], dtype=float)
        lon_arr = np.asarray(nav[1], dtype=float)

    valid_mask = np.isfinite(lat_arr) & np.isfinite(lon_arr)
    if not np.any(valid_mask):
        raise ValueError("No valid latitude/longitude points to plot.")
    lat_clean = lat_arr[valid_mask]
    lon_clean = lon_arr[valid_mask]

    # Load grid
    if isinstance(geo_map, (str, Path)):
        grid = load_dataarray(str(geo_map))
    elif isinstance(geo_map, xr.DataArray):
        grid = geo_map
    else:
        raise ValueError("`geo_map` must be a path or an xarray.DataArray")

    # Determine plotting region (west,east,south,north)
    lat_min, lat_max = float(np.min(lat_clean)), float(np.max(lat_clean))
    lon_min, lon_max = float(np.min(lon_clean)), float(np.max(lon_clean))
    region = [lon_min - margin, lon_max + margin, lat_min - margin, lat_max + margin]

    # Choose colormap based on map type
    if map_type == GeophysicalType.RELIEF:
        cmap = "relief"
    elif map_type == GeophysicalType.GRAVITY:
        cmap = "vik"
    else:
        cmap = "roma"

    # Create CPT
    try:
        zmin = float(grid.min().data)
        zmax = float(grid.max().data)
    except Exception:
        zmin, zmax = float(np.nanmin(grid)), float(np.nanmax(grid))

    pygmt.makecpt(cmap=cmap, series=[zmin, zmax])

    fig = pygmt.Figure()
    fig.grdimage(grid=grid, region=region, projection="M10i", cmap=True)

    # Add contours
    try:
        interval = max((zmax - zmin) / 20.0, 0.0)
        if interval > 0:
            fig.grdcontour(grid=grid, interval=interval)
    except Exception:
        pass

    # Plot nav trajectory (distinct color)
    fig.plot(x=lon_clean, y=lat_clean, pen="2p,red")

    # Plot GPS/Truth in black if provided
    if gps is not None:
        try:
            gps_lat = gps["latitude"].to_numpy(dtype=float)
            gps_lon = gps["longitude"].to_numpy(dtype=float)
            gps_mask = np.isfinite(gps_lat) & np.isfinite(gps_lon)
            if np.any(gps_mask):
                fig.plot(x=gps_lon[gps_mask], y=gps_lat[gps_mask], pen="1p,black")
        except Exception:
            pass

    if title is not None:
        fig.text(
            x=region[0] + 0.02 * (region[1] - region[0]),
            y=region[3] - 0.02 * (region[3] - region[2]),
            text=title,
            font="14p,Helvetica-Bold",
        )

    return fig
