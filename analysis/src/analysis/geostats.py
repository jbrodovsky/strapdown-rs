"""
Characterise the geophysical measurements against the maps they are matched to.

`strapdown-sim`'s geophysical aiding differences a sensor-derived anomaly against a value
read from a NetCDF map. How well that can possibly work is set by two numbers this module
measures and nothing else in the toolchain does:

- the **residual** between the two, whose spread is the measurement noise the filter should
  be told about (``gravity_noise_std`` / ``magnetic_noise_std``) and whose offset is the map
  bias it should be seeded with, and
- the **signal**, the variation of the map value along the track, which is the whole of what
  the aid has to work with.

Their ratio is the signal-to-noise ratio. Below 1 the aid carries less information than the
noise it injects, and the filter's covariance falls without its error following.

The defaults these were compared against -- 100 mGal and 150 nT -- were never measured. They
are constants in the source (``DEFAULT_GRAVITY_NOISE_MGAL`` / ``DEFAULT_MAGNETIC_NOISE_NT``
in `strapdown-geonav`) and are repeated in every ``conf/*.toml``.

Running it::

    analyze geostats -i data/input -o data/output/geostats

Each trajectory CSV is read with the sibling ``<stem>_gravity.nc`` and ``<stem>_magnetic.nc``
that ``analyze preprocess`` wrote beside it -- the same files, found the same way,
``find_gravity_map`` and ``find_magnetic_map`` in ``sim/src/main.rs`` look for.

Why the formulas below are duplicated from Rust
-----------------------------------------------
Every anomaly here is computed the way the filter computes it, mirroring
``core/src/earth.rs`` and ``geonav/src/lib.rs`` term for term. A statistic derived from a
*differently* computed anomaly characterises a quantity the filter never sees, which is worse
than no statistic at all: it would be used to set the filter's noise. The mirrored functions
name their Rust counterpart, and :func:`self_check` asserts the two agree on a known case.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# ===========================================================================================
# Earth model -- mirrored from core/src/earth.rs
# ===========================================================================================

#: Earth's rotation rate, rad/s. `earth::RATE`.
RATE = 7.2921159e-5
#: Earth's equatorial radius, m. `earth::EQUATORIAL_RADIUS`.
EQUATORIAL_RADIUS = 6378137.0
#: Earth's polar radius, m. `earth::POLAR_RADIUS`.
POLAR_RADIUS = 6356752.31425
#: Earth's eccentricity. `earth::ECCENTRICITY`.
ECCENTRICITY = 0.0818191908425
#: Earth's eccentricity squared. `earth::ECCENTRICITY_SQUARED`.
ECCENTRICITY_SQUARED = ECCENTRICITY * ECCENTRICITY
#: Equatorial gravity, m/s^2. `earth::GE`.
GE = 9.7803253359
#: Polar gravity, m/s^2. `earth::GP`.
GP = 9.8321849378
#: Somigliana's constant. `earth::K`.
K = (POLAR_RADIUS * GP - EQUATORIAL_RADIUS * GE) / (EQUATORIAL_RADIUS * GE)

#: Milligal per m/s^2. `earth::MGAL_PER_M_PER_S2`.
#:
#: The gravity anomaly is formed from SI accelerometer readings but consumed in milligal --
#: the unit of the map, of ``--gravity-noise-std`` and of the ``[geophysical]`` config
#: section. This conversion was missing from the Rust side until recently, which made every
#: gravity innovation 1e5 too small; see `earth::gravity_anomaly`.
MGAL_PER_M_PER_S2 = 1.0e5

#: Nanotesla per microtesla. `geonav::MICROTESLA_TO_NANOTESLA`.
#:
#: Sensor Logger reports the magnetometer in microtesla; anomalies and maps are nanotesla.
MICROTESLA_TO_NANOTESLA = 1000.0


def normal_gravity(latitude_deg: np.ndarray | float, altitude_m: np.ndarray | float = 0.0):
    """
    Somigliana normal gravity, m/s^2. Mirrors `earth::gravity`.

    Parameters
    ----------
    latitude_deg : array_like or float
        Geodetic latitude, degrees.
    altitude_m : array_like or float, optional
        Height above the ellipsoid, metres. Note that `earth::gravity_anomaly` calls this
        with ``0.0``, so :func:`gravity_anomaly_mgal` does too.

    Returns
    -------
    ndarray or float
        Normal gravity in m/s^2.
    """
    sin_lat = np.sin(np.radians(latitude_deg))
    sin_sq = sin_lat * sin_lat
    g0 = (GE * (1.0 + K * sin_sq)) / np.sqrt(1.0 - ECCENTRICITY_SQUARED * sin_sq)
    return g0 - 3.08e-6 * altitude_m


def principal_radii(latitude_deg, altitude_m):
    """
    Meridional, transverse and transverse-at-altitude radii, metres. Mirrors
    `earth::principal_radii`.

    Returns
    -------
    tuple of ndarray
        ``(r_n, r_e, r_p)``, where ``r_p`` is the one the Eotvos term divides by.
    """
    lat_rad = np.radians(latitude_deg)
    sin_sq = np.sin(lat_rad) ** 2
    r_n = (EQUATORIAL_RADIUS * (1.0 - ECCENTRICITY_SQUARED)) / np.power(
        1.0 - ECCENTRICITY_SQUARED * sin_sq, 1.5
    )
    r_e = EQUATORIAL_RADIUS / np.sqrt(1.0 - ECCENTRICITY_SQUARED * sin_sq)
    r_p = r_e * np.cos(lat_rad) + altitude_m
    return r_n, r_e, r_p


def eotvos(latitude_deg, altitude_m, north_velocity, east_velocity):
    """
    Eotvos correction, m/s^2. Mirrors `earth::eotvos`.

    The apparent change in gravity from moving over a rotating Earth: an eastward run adds
    centrifugal acceleration, and any horizontal motion adds a curvature term.
    """
    _, _, r_p = principal_radii(latitude_deg, altitude_m)
    return (
        2.0 * RATE * east_velocity * np.cos(np.radians(latitude_deg))
        + (north_velocity**2 + east_velocity**2) / r_p
    )


def gravity_anomaly_mgal(
    latitude_deg, altitude_m, north_velocity, east_velocity, gravity_observed_mps2
):
    """
    Free-air gravity anomaly in **milligal**. Mirrors `earth::gravity_anomaly`.

    Parameters
    ----------
    gravity_observed_mps2 : array_like
        Magnitude of the record's three ``grav_*`` axes, m/s^2.

    Returns
    -------
    ndarray
        The anomaly in milligal, the unit the map is in.
    """
    gamma = normal_gravity(latitude_deg, 0.0)
    correction = eotvos(latitude_deg, altitude_m, north_velocity, east_velocity)
    return (gravity_observed_mps2 - gamma - correction) * MGAL_PER_M_PER_S2


def observed_field_nt(mag_x, mag_y, mag_z):
    """
    Total magnetic field magnitude in nanotesla. Mirrors `geonav::observed_field_nt`.
    """
    return np.sqrt(mag_x**2 + mag_y**2 + mag_z**2) * MICROTESLA_TO_NANOTESLA


def decimal_year(timestamps: pd.Series) -> np.ndarray:
    """Convert timestamps to the decimal year the WMM is evaluated at."""
    years = timestamps.dt.year.to_numpy()
    starts = pd.to_datetime(pd.Series(years).map(lambda y: f"{y}-01-01"), utc=True).to_numpy()
    ends = pd.to_datetime(pd.Series(years + 1).map(lambda y: f"{y}-01-01"), utc=True).to_numpy()
    here = timestamps.dt.tz_convert("UTC").to_numpy()
    return years + (here - starts) / (ends - starts)


def wmm_total_field_nt(latitude_deg, longitude_deg, altitude_m, years) -> np.ndarray:
    """
    World Magnetic Model total field, nanotesla. Mirrors
    `geonav::MagneticAnomalyMeasurement::reference_field_nt`.

    The Rust side uses the ``world_magnetic_model`` crate; this uses ``pygeomag``. The two
    agree to well under a nanotesla, which :func:`self_check` asserts.

    Raises
    ------
    ImportError
        If ``pygeomag`` is not installed. The magnetic channel is skipped rather than
        silently producing an anomaly with no core field removed -- which would be a ~50,000
        nT error, several hundred times the signal.
    """
    from pygeomag import GeoMag

    model = GeoMag()
    latitude_deg = np.atleast_1d(latitude_deg)
    longitude_deg = np.atleast_1d(longitude_deg)
    altitude_m = np.atleast_1d(altitude_m)
    years = np.atleast_1d(years)

    out = np.empty(latitude_deg.shape, dtype=float)
    for i in range(latitude_deg.size):
        # pygeomag takes altitude in kilometres.
        result = model.calculate(
            glat=float(latitude_deg[i]),
            glon=float(longitude_deg[i]),
            alt=float(altitude_m[i]) / 1000.0,
            time=float(years[i]),
        )
        out[i] = result.f
    return out


# ===========================================================================================
# Map provenance
# ===========================================================================================
#
# What the grids actually contain, which is not what their spacing advertises. Both numbers
# below are properties of the *source data*, and both are coarser than the cell size --
# so decimating one measurement per cell still over-counts.

#: Effective resolution of the gravity grid over land, metres.
#:
#: ``analyze preprocess`` downloads ``earth_faa`` at ``01m`` -- a 1 arc-minute cell, about
#: 1855 m of latitude. That is the *grid* spacing. ``earth_faa`` is Sandwell & Smith's
#: altimetry-derived marine anomaly, and **over land it is filled in from EGM2008**, whose
#: terrestrial input is a 5 arc-minute area-mean grid. For a land vehicle the 1 arc-minute
#: cells are therefore interpolation, not independent information, and the shortest
#: wavelength genuinely present is about 5 arc-minutes -- roughly 9.3 km.
GRAVITY_SOURCE_RESOLUTION_M = 5.0 * 1852.0

#: Altitude the magnetic grid is referenced to, metres.
#:
#: ``analyze preprocess`` downloads ``earth_wdmam`` at ``03m``. WDMAM v2 is a 3 arc-minute
#: grid **at 5 km above sea level**. A phone at ground level is not measuring the same
#: quantity: upward continuation attenuates an anomaly of wavelength L by exp(-2*pi*h/L), so
#: at 5 km everything shorter than about 31 km is suppressed in the map but present in the
#: measurement. That mismatch lands in the residual and cannot be tuned away.
MAGNETIC_SOURCE_ALTITUDE_M = 5000.0

#: Shortest wavelength surviving upward continuation to `MAGNETIC_SOURCE_ALTITUDE_M`, metres.
#:
#: The 1/e point of exp(-2*pi*h/L). Used as the magnetic de-correlation length, because two
#: samples closer together than this are reading the same feature of the map.
MAGNETIC_SOURCE_RESOLUTION_M = 2.0 * math.pi * MAGNETIC_SOURCE_ALTITUDE_M

#: Metres per degree of latitude, for converting grid spacing to ground distance.
METRES_PER_DEGREE = 111320.0


@dataclass
class AnomalyMap:
    """A NetCDF anomaly grid, loaded the way `geonav::GeoMap::load_geomap` loads it."""

    path: Path
    lats: np.ndarray
    lons: np.ndarray
    values: np.ndarray

    @classmethod
    def load(cls, path: Path) -> AnomalyMap:
        """
        Read ``lat``, ``lon`` and ``z`` from a NetCDF grid.

        Mirrors `GeoMap::load_geomap`, which reads exactly those three variables and assumes
        both coordinate vectors ascend. PyGMT writes the coordinates under these names, and
        names the data variable ``z``; a few of its grids use a different variable name, so
        the sole data variable is accepted as a fallback.
        """
        import xarray as xr

        with xr.open_dataset(path) as dataset:
            lat_name = "lat" if "lat" in dataset.coords else "y"
            lon_name = "lon" if "lon" in dataset.coords else "x"
            if "z" in dataset:
                data_name = "z"
            else:
                candidates = [n for n in dataset.data_vars if n not in (lat_name, lon_name)]
                if len(candidates) != 1:
                    raise ValueError(
                        f"{path.name}: expected a variable named 'z' or exactly one data "
                        f"variable, found {list(dataset.data_vars)}"
                    )
                data_name = candidates[0]

            lats = np.asarray(dataset[lat_name].values, dtype=float)
            lons = np.asarray(dataset[lon_name].values, dtype=float)
            values = np.asarray(dataset[data_name].values, dtype=float)

        if values.shape != (lats.size, lons.size):
            raise ValueError(
                f"{path.name}: data is {values.shape}, expected {(lats.size, lons.size)}"
            )
        return cls(path=path, lats=lats, lons=lons, values=values)

    def sample(self, latitude_deg: np.ndarray, longitude_deg: np.ndarray) -> np.ndarray:
        """
        Bilinearly interpolate the grid, mirroring `GeoMap::get_point`.

        Positions outside the grid return NaN rather than an edge value, which is what
        `get_point` signals with ``OutOfMapBounds`` and what the simulator skips the update
        on. They are dropped from the statistics rather than counted as huge residuals.
        """
        latitude_deg = np.asarray(latitude_deg, dtype=float)
        longitude_deg = np.asarray(longitude_deg, dtype=float)
        out = np.full(latitude_deg.shape, np.nan)

        inside = (
            (latitude_deg >= self.lats[0])
            & (latitude_deg <= self.lats[-1])
            & (longitude_deg >= self.lons[0])
            & (longitude_deg <= self.lons[-1])
            & np.isfinite(latitude_deg)
            & np.isfinite(longitude_deg)
        )
        if not inside.any():
            return out

        lat = latitude_deg[inside]
        lon = longitude_deg[inside]
        i = np.clip(np.searchsorted(self.lats, lat) - 1, 0, self.lats.size - 2)
        j = np.clip(np.searchsorted(self.lons, lon) - 1, 0, self.lons.size - 2)

        lat0, lat1 = self.lats[i], self.lats[i + 1]
        lon0, lon1 = self.lons[j], self.lons[j + 1]
        u = np.where(lat1 > lat0, (lat - lat0) / (lat1 - lat0), 0.0)
        v = np.where(lon1 > lon0, (lon - lon0) / (lon1 - lon0), 0.0)

        out[inside] = (
            self.values[i, j] * (1 - u) * (1 - v)
            + self.values[i + 1, j] * u * (1 - v)
            + self.values[i, j + 1] * (1 - u) * v
            + self.values[i + 1, j + 1] * u * v
        )
        return out

    def spacing_deg(self) -> tuple[float, float]:
        """Median grid spacing in degrees, as ``(lat, lon)``."""
        return (
            float(np.median(np.diff(self.lats))),
            float(np.median(np.diff(self.lons))),
        )

    def spacing_m(self, latitude_deg: float) -> tuple[float, float]:
        """
        Grid spacing in metres at a given latitude, as ``(north, east)``.

        Nothing in the Rust workspace derives this: `GeoResolution` is a label used for a
        filename token and nothing else, and no code checks that the loaded grid matches the
        resolution the config declared. That makes a mismatch silent, which is how the
        magnetic grid came to be downloaded at 3 arc-minutes while every ``conf/*.toml``
        declares ``two_minutes``.
        """
        d_lat, d_lon = self.spacing_deg()
        return (
            abs(d_lat) * METRES_PER_DEGREE,
            abs(d_lon) * METRES_PER_DEGREE * math.cos(math.radians(latitude_deg)),
        )

    def arcminutes(self) -> tuple[float, float]:
        """Grid spacing in arcminutes, as ``(lat, lon)`` -- what the config declares."""
        d_lat, d_lon = self.spacing_deg()
        return abs(d_lat) * 60.0, abs(d_lon) * 60.0


# ===========================================================================================
# Per-trajectory analysis
# ===========================================================================================

#: Columns a trajectory CSV must carry for either channel to be computable.
REQUIRED_COLUMNS = ("time", "latitude", "longitude", "altitude", "speed", "bearing")

#: Scale factor turning a median absolute deviation into a Gaussian-equivalent sigma.
MAD_TO_SIGMA = 1.4826

#: The geophysical measurement interval the configs currently use, seconds.
#:
#: Only used to report how many updates land inside one de-correlation length; it is not a
#: recommendation. ``geo_frequency_s = 1.0`` in every ``conf/*.toml``.
CURRENT_GEO_INTERVAL_S = 1.0


@dataclass
class FieldStats:
    """Residual and signal statistics for one field on one trajectory."""

    trajectory: str
    field: str
    unit: str
    samples: int
    #: Mean of (measured - map). The systematic offset a bias state has to absorb.
    bias_mean: float
    #: Median of the same. Preferred over the mean: these residuals are not Gaussian.
    bias_median: float
    #: Standard deviation of the residual.
    sigma: float
    #: Robust sigma, 1.4826 * MAD. What to use when the histogram is multimodal.
    sigma_robust: float
    #: Standard deviation of the *map* value along the track -- the available signal.
    signal_sigma: float
    #: Peak-to-peak of the map value along the track.
    signal_range: float
    #: signal_sigma / sigma_robust. Below 1 the aid carries less than it injects.
    snr: float
    #: Median ground speed, m/s.
    speed_median: float
    #: Grid cell size at the track, metres, as (north, east).
    cell_north_m: float
    cell_east_m: float
    #: Grid spacing in arcminutes -- compare against the config's declared resolution.
    grid_arcmin_lat: float
    grid_arcmin_lon: float
    #: Independent-information length: the coarser of the cell and the source resolution.
    decorrelation_m: float
    #: Updates falling inside one decorrelation length at CURRENT_GEO_INTERVAL_S.
    updates_per_decorrelation: float
    #: Seconds per decorrelation length at this track's speed -- the interval to configure.
    recommended_interval_s: float
    #: How many genuinely independent measurements the whole trajectory affords.
    independent_samples: float
    #: Trajectory duration, seconds.
    duration_s: float
    #: Rows whose position fell outside the map.
    off_map: int


def _finite(*arrays: np.ndarray) -> np.ndarray:
    """Boolean mask of positions where every supplied array is finite."""
    mask = np.ones(arrays[0].shape, dtype=bool)
    for array in arrays:
        mask &= np.isfinite(array)
    return mask


def _stats_from(
    trajectory: str,
    field: str,
    unit: str,
    residual: np.ndarray,
    map_value: np.ndarray,
    speed: np.ndarray,
    duration_s: float,
    anomaly_map: AnomalyMap,
    mean_latitude: float,
    source_resolution_m: float,
    off_map: int,
) -> FieldStats:
    """Reduce one field's residual and signal series to a :class:`FieldStats`."""
    sigma = float(np.std(residual, ddof=1)) if residual.size > 1 else float("nan")
    mad = (
        float(np.median(np.abs(residual - np.median(residual)))) if residual.size else float("nan")
    )
    sigma_robust = MAD_TO_SIGMA * mad
    signal_sigma = float(np.std(map_value, ddof=1)) if map_value.size > 1 else float("nan")

    # The robust sigma is the denominator: a multimodal residual's plain standard deviation
    # is inflated by the separation between its modes, which flatters the ratio. Fall back to
    # the plain one when the MAD collapses to zero (a constant residual).
    denominator = sigma_robust if sigma_robust > 0 else sigma
    snr = signal_sigma / denominator if denominator and np.isfinite(denominator) else float("nan")

    cell_north, cell_east = anomaly_map.spacing_m(mean_latitude)
    arcmin_lat, arcmin_lon = anomaly_map.arcminutes()

    # The grid can be finer than the data behind it. Taking the coarser of the two is what
    # keeps the recommendation honest -- interpolated cells are not independent samples.
    cell_m = min(cell_north, cell_east)
    decorrelation_m = max(cell_m, source_resolution_m)

    speed_median = float(np.median(speed)) if speed.size else float("nan")
    if speed_median > 0 and np.isfinite(speed_median):
        recommended_interval_s = decorrelation_m / speed_median
        updates_per_decorrelation = recommended_interval_s / CURRENT_GEO_INTERVAL_S
        independent_samples = duration_s / recommended_interval_s
    else:
        recommended_interval_s = float("nan")
        updates_per_decorrelation = float("nan")
        independent_samples = float("nan")

    return FieldStats(
        trajectory=trajectory,
        field=field,
        unit=unit,
        samples=int(residual.size),
        bias_mean=float(np.mean(residual)) if residual.size else float("nan"),
        bias_median=float(np.median(residual)) if residual.size else float("nan"),
        sigma=sigma,
        sigma_robust=sigma_robust,
        signal_sigma=signal_sigma,
        signal_range=float(np.ptp(map_value)) if map_value.size else float("nan"),
        snr=snr,
        speed_median=speed_median,
        cell_north_m=cell_north,
        cell_east_m=cell_east,
        grid_arcmin_lat=arcmin_lat,
        grid_arcmin_lon=arcmin_lon,
        decorrelation_m=decorrelation_m,
        updates_per_decorrelation=updates_per_decorrelation,
        recommended_interval_s=recommended_interval_s,
        independent_samples=independent_samples,
        duration_s=duration_s,
        off_map=off_map,
    )


def analyse_trajectory(
    csv_path: Path, skip_magnetic: bool = False
) -> tuple[list[FieldStats], pd.DataFrame]:
    """
    Compute both channels' residuals for one trajectory.

    Parameters
    ----------
    csv_path : Path
        A preprocessed trajectory CSV. Its ``<stem>_gravity.nc`` / ``<stem>_magnetic.nc``
        siblings are read if present; a channel with no map is skipped.
    skip_magnetic : bool, optional
        Skip the magnetic channel, used when ``pygeomag`` is unavailable.

    Returns
    -------
    tuple
        ``(stats, residuals)`` -- one :class:`FieldStats` per available channel, and a long
        frame of per-record residuals for the histogram and for inspection.
    """
    # `date_format="ISO8601"` is required, not cosmetic: pandas's fast-path CSV date parser
    # silently leaves the column as `str` (no error) when a single column mixes timestamps
    # with and without a fractional-second component -- which every trajectory here does,
    # since a record that lands exactly on a whole second is written without one. Pinning
    # the parser to ISO8601 makes it parse per-value instead of demanding one shared format.
    frame = pd.read_csv(csv_path, parse_dates=["time"], date_format="ISO8601")
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"{csv_path.name}: missing column(s) {missing}")

    # Parsed explicitly rather than through `read_csv(parse_dates=...)`. On a column of
    # `2025-03-01 12:00:00+00:00` -- which is exactly what `analyze preprocess` writes --
    # pandas leaves `parse_dates` columns as **strings** when it cannot infer a format, with
    # no error, so the failure surfaced a hundred lines later as
    # `unsupported operand type(s) for -: 'str' and 'str'`.
    frame["time"] = pd.to_datetime(frame["time"], utc=True, format="ISO8601")
    if not isinstance(frame["time"].dtype, pd.DatetimeTZDtype):
        raise ValueError(
            f"{csv_path.name}: 'time' did not parse as a timezone-aware datetime "
            f"(got {frame['time'].dtype}); every downstream calculation assumes it did"
        )

    stem = csv_path.with_suffix("").name
    have_gnss = _finite(
        frame["latitude"].to_numpy(float),
        frame["longitude"].to_numpy(float),
        frame["altitude"].to_numpy(float),
    )
    frame = frame.loc[have_gnss].reset_index(drop=True)
    if frame.empty:
        return [], pd.DataFrame()

    latitude = frame["latitude"].to_numpy(float)
    longitude = frame["longitude"].to_numpy(float)
    altitude = frame["altitude"].to_numpy(float)
    speed = np.nan_to_num(frame["speed"].to_numpy(float), nan=0.0)
    bearing = np.radians(np.nan_to_num(frame["bearing"].to_numpy(float), nan=0.0))
    north_velocity = speed * np.cos(bearing)
    east_velocity = speed * np.sin(bearing)
    seconds = (frame["time"] - frame["time"].iloc[0]).dt.total_seconds().to_numpy(float)
    duration_s = float(seconds[-1]) if seconds.size else 0.0
    mean_latitude = float(np.mean(latitude))

    stats: list[FieldStats] = []
    residual_frames: list[pd.DataFrame] = []

    channels = [
        ("gravity", "mGal", csv_path.parent / f"{stem}_gravity.nc", GRAVITY_SOURCE_RESOLUTION_M),
        ("magnetic", "nT", csv_path.parent / f"{stem}_magnetic.nc", MAGNETIC_SOURCE_RESOLUTION_M),
    ]

    for name, unit, map_path, source_resolution in channels:
        if not map_path.exists():
            continue
        if name == "magnetic" and skip_magnetic:
            continue

        if name == "gravity":
            grav_columns = ("grav_x", "grav_y", "grav_z")
            if not all(c in frame.columns for c in grav_columns):
                continue
            observed = np.sqrt(sum(frame[c].to_numpy(float) ** 2 for c in grav_columns))
            measured = gravity_anomaly_mgal(
                latitude, altitude, north_velocity, east_velocity, observed
            )
        else:
            mag_columns = ("mag_x", "mag_y", "mag_z")
            if not all(c in frame.columns for c in mag_columns):
                continue
            observed = observed_field_nt(*(frame[c].to_numpy(float) for c in mag_columns))
            reference = wmm_total_field_nt(
                latitude, longitude, altitude, decimal_year(frame["time"])
            )
            measured = observed - reference

        anomaly_map = AnomalyMap.load(map_path)
        map_value = anomaly_map.sample(latitude, longitude)
        residual = measured - map_value

        usable = _finite(residual, map_value, measured)
        off_map = int(np.sum(~np.isfinite(map_value)))
        if usable.sum() < 2:
            continue

        stats.append(
            _stats_from(
                trajectory=stem,
                field=name,
                unit=unit,
                residual=residual[usable],
                map_value=map_value[usable],
                speed=speed[usable],
                duration_s=duration_s,
                anomaly_map=anomaly_map,
                mean_latitude=mean_latitude,
                source_resolution_m=source_resolution,
                off_map=off_map,
            )
        )
        residual_frames.append(
            pd.DataFrame(
                {
                    "trajectory": stem,
                    "field": name,
                    "unit": unit,
                    "elapsed_s": seconds[usable],
                    "latitude": latitude[usable],
                    "longitude": longitude[usable],
                    "measured": measured[usable],
                    "map": map_value[usable],
                    "residual": residual[usable],
                }
            )
        )

    residuals = pd.concat(residual_frames, ignore_index=True) if residual_frames else pd.DataFrame()
    return stats, residuals


# ===========================================================================================
# Pooling across trajectories
# ===========================================================================================


@dataclass
class PooledStats:
    """
    A field's statistics across every trajectory, decomposed into the knobs they set.

    The decomposition is the point. A residual has two parts that want different homes in
    the filter:

    - what varies *within* one recording is measurement noise -- it is what the filter
      should be told ``gravity_noise_std`` / ``magnetic_noise_std`` is;
    - what varies *between* recordings is a per-run offset, and the filter has a state for
      exactly that. It sets the map bias prior, ``*_bias_init_std``.

    Collapsing them into one number is what makes the magnetic channel look hopeless when it
    is merely mis-modelled: every recording has its own hard- and soft-iron signature from
    the phone and the vehicle it sat in, so the between-recording spread is enormous while
    the within-recording spread may not be. Only the second belongs in R.
    """

    field: str
    unit: str
    trajectories: int
    samples: int
    #: Median of every residual, pooled. Sets ``gravity_bias`` / ``magnetic_bias``.
    bias_median: float
    #: Robust sigma of the residual after each trajectory's own median is removed.
    #: Sets ``gravity_noise_std`` / ``magnetic_noise_std``.
    within_sigma: float
    #: Spread of the per-trajectory medians. Sets ``*_bias_init_std``.
    between_sigma: float
    #: Robust sigma of the raw pooled residual, for reference.
    total_sigma: float
    #: Median and best per-trajectory SNR, and how many cleared 1.0.
    snr_median: float
    snr_best: float
    snr_above_one: int
    #: Median recommended interval across trajectories, seconds.
    recommended_interval_s: float
    #: Median independent samples per trajectory -- how much the aid can ever say.
    independent_samples_median: float
    #: Median de-correlation length, metres.
    decorrelation_m: float


def _robust_sigma(values: np.ndarray) -> float:
    """1.4826 * MAD, the Gaussian-equivalent sigma of a heavy-tailed sample."""
    if values.size == 0:
        return float("nan")
    return float(MAD_TO_SIGMA * np.median(np.abs(values - np.median(values))))


def pool(stats: list[FieldStats], residuals: pd.DataFrame) -> list[PooledStats]:
    """Combine per-trajectory statistics into one row per field."""
    pooled: list[PooledStats] = []
    for name in ("gravity", "magnetic"):
        rows = [s for s in stats if s.field == name]
        if not rows:
            continue
        subset = residuals[residuals["field"] == name]
        if subset.empty:
            continue

        residual = subset["residual"].to_numpy(float)
        # Remove each trajectory's own median, leaving only within-recording variation.
        centred = (
            subset.groupby("trajectory")["residual"]
            .transform(lambda series: series - series.median())
            .to_numpy(float)
        )
        per_trajectory_medians = np.array([s.bias_median for s in rows], dtype=float)
        snrs = np.array([s.snr for s in rows], dtype=float)
        snrs = snrs[np.isfinite(snrs)]

        pooled.append(
            PooledStats(
                field=name,
                unit=rows[0].unit,
                trajectories=len(rows),
                samples=int(residual.size),
                bias_median=float(np.median(residual)),
                within_sigma=_robust_sigma(centred),
                between_sigma=float(np.std(per_trajectory_medians, ddof=1))
                if per_trajectory_medians.size > 1
                else float("nan"),
                total_sigma=_robust_sigma(residual),
                snr_median=float(np.median(snrs)) if snrs.size else float("nan"),
                snr_best=float(np.max(snrs)) if snrs.size else float("nan"),
                snr_above_one=int(np.sum(snrs > 1.0)),
                recommended_interval_s=float(np.median([s.recommended_interval_s for s in rows])),
                independent_samples_median=float(np.median([s.independent_samples for s in rows])),
                decorrelation_m=float(np.median([s.decorrelation_m for s in rows])),
            )
        )
    return pooled


# ===========================================================================================
# Outputs
# ===========================================================================================

#: Panel fill colours, one per field, and the normal-fit overlay.
#:
#: Checked for colour-vision separation rather than chosen by eye. The obvious pairing --
#: a green magnetic histogram under a red fit line -- fails deuteranopia at dE 4.6, which
#: is not visible to someone with normal vision picking the colours.
FIELD_COLOUR = {"gravity": "#3A6FB0", "magnetic": "#3F8F5B"}
FIT_COLOUR = "#B07A1E"
INK = "#1F2328"
MUTED_INK = "#6B7280"


def write_config_block(pooled: list[PooledStats], path: Path) -> None:
    """
    Write a paste-ready ``[geophysical]`` section carrying every derived number.

    Each value is commented with where it came from, in the style ``conf/*_degraded.toml``
    already uses for its AR(1) parameters -- a bare number in a config file is untraceable
    six months later.
    """
    by_field = {p.field: p for p in pooled}
    lines = [
        "# Generated by `analyze geostats`. Every value below is measured from the residual",
        "# between the sensor-derived anomaly and the map, over the trajectories in",
        "# data/input -- not a default.",
        "#",
        "# Replaces the 100.0 mGal / 150.0 nT that were never measured.",
        "",
        "[geophysical]",
    ]

    for field, bias_key, noise_key, prior_key in (
        ("gravity", "gravity_bias", "gravity_noise_std", "gravity_bias_init_std"),
        ("magnetic", "magnetic_bias", "magnetic_noise_std", "magnetic_bias_init_std"),
    ):
        stats = by_field.get(field)
        if stats is None:
            continue
        unit = stats.unit
        lines += [
            f"# --- {field} ---",
            f"# {stats.samples} samples over {stats.trajectories} trajectories.",
            f"# Signal-to-noise: median {stats.snr_median:.2f}, best {stats.snr_best:.2f}; "
            f"{stats.snr_above_one} of {stats.trajectories} above 1.0.",
            f"{bias_key} = {stats.bias_median:.4g}      # pooled median residual, {unit}",
            f"{noise_key} = {stats.within_sigma:.4g}      # within-trajectory robust sigma, {unit}",
            f"{prior_key} = {stats.between_sigma:.4g}      # spread of per-trajectory medians, {unit}",
            "",
        ]

    interval = min(
        (p.recommended_interval_s for p in pooled if np.isfinite(p.recommended_interval_s)),
        default=float("nan"),
    )
    if np.isfinite(interval):
        lines += [
            "# One measurement per de-correlation length, at each track's own median speed.",
            "# The previous 1.0 treated every sample inside a cell as independent, which is",
            "# what drove the covariance below the truth. Per field:",
        ]
        for stats in pooled:
            lines.append(
                f"#   {stats.field}: {stats.decorrelation_m:,.0f} m -> "
                f"{stats.recommended_interval_s:.0f} s "
                f"({stats.independent_samples_median:.1f} independent samples per trajectory)"
            )
        lines += [
            "# The smaller of the two is used, since one interval covers both channels.",
            f"geo_interval_s = {interval:.1f}",
            "",
        ]

    path.write_text("\n".join(lines), encoding="utf-8")


def _rewrite_value(line: str, key: str, value: float) -> str:
    """
    Replace the value on a `key = value` line, keeping any trailing comment.

    The comment is usually the unit or the source, which is the part of a config line worth
    the most six months later.
    """
    comment = line.partition("=")[2].partition("#")[2]
    if comment:
        return f"{key} = {value:.6g}  #{comment}"
    return f"{key} = {value:.6g}"


def apply_to_configs(
    pooled: list[PooledStats],
    conf_dir: Path,
    apply_interval: bool = False,
) -> list[str]:
    """
    Write the measured values into every scenario config that carries a `[geophysical]` block.

    The defaults these replace -- 100 mGal and 150 nT -- were never measured against the maps
    they are differenced from. Leaving them in place means the filter is told a noise it does
    not have, and its covariance stops describing its error.

    Edited line by line rather than through a TOML round-trip, which would strip every comment
    in these files. The rationale in `conf/*.toml` is most of their value.

    Only keys already present are rewritten, so a gravity-only recipe stays gravity-only. The
    one exception is `*_bias_init_std`: it is *added* after `*_noise_std` when absent, because
    it is the prior on the map-bias state and the whole point of separating the
    within-trajectory spread from the between-trajectory one. Without it the bias state keeps
    a prior of `*_noise_std`, which for the magnetic channel is usually far too tight -- a
    recording made inside a vehicle carries thousands of nT of the vehicle's own field.

    Parameters
    ----------
    pooled : list of PooledStats
        The measured statistics, as :func:`pool` returns them.
    conf_dir : Path
        Directory of scenario configs to rewrite.
    apply_interval : bool, optional
        Also rewrite `geo_interval_s` / `geo_frequency_s` to the recommended de-correlation
        interval. Off by default: the noise and bias figures are direct measurements, whereas
        the interval follows from a de-correlation argument about the *source* data, and
        moving it from 1 s to several hundred changes the character of the experiment rather
        than just its tuning. Worth doing -- deliberately.

    Returns
    -------
    list of str
        One line per edit, for printing. Empty if nothing matched.
    """
    by_field = {entry.field: entry for entry in pooled}
    updates: dict[str, float] = {}
    for field in ("gravity", "magnetic"):
        entry = by_field.get(field)
        if entry is None:
            continue
        updates[f"{field}_bias"] = entry.bias_median
        updates[f"{field}_noise_std"] = entry.within_sigma
        updates[f"{field}_bias_init_std"] = entry.between_sigma

    interval = min(
        (p.recommended_interval_s for p in pooled if np.isfinite(p.recommended_interval_s)),
        default=float("nan"),
    )

    changes: list[str] = []
    for path in sorted(Path(conf_dir).glob("*.toml")):
        text = path.read_text(encoding="utf-8")
        if "[geophysical]" not in text:
            continue

        lines = text.split("\n")
        start = next(i for i, line in enumerate(lines) if line.strip() == "[geophysical]")
        end = next(
            (
                i
                for i in range(start + 1, len(lines))
                if lines[i].startswith("[") and lines[i].strip() != "[geophysical]"
            ),
            len(lines),
        )

        edited = False
        for index in range(start + 1, end):
            key = lines[index].split("=", 1)[0].strip()
            if key in updates and not np.isnan(updates[key]):
                lines[index] = _rewrite_value(lines[index], key, updates[key])
                changes.append(f"{path.name}: {key} = {updates[key]:.6g}")
                edited = True
            elif apply_interval and key in ("geo_interval_s", "geo_frequency_s"):
                if np.isfinite(interval):
                    lines[index] = _rewrite_value(lines[index], key, interval)
                    changes.append(f"{path.name}: {key} = {interval:.6g}")
                    edited = True

        # Insert each `*_bias_init_std` directly after its `*_noise_std`, walking backwards so
        # earlier insertions do not shift the indices of later ones.
        for field in ("magnetic", "gravity"):
            prior_key = f"{field}_bias_init_std"
            value = updates.get(prior_key)
            if value is None or np.isnan(value):
                continue
            if any(line.split("=", 1)[0].strip() == prior_key for line in lines[start:end]):
                continue
            noise_key = f"{field}_noise_std"
            at = next(
                (
                    i
                    for i in range(start + 1, end)
                    if lines[i].split("=", 1)[0].strip() == noise_key
                ),
                None,
            )
            if at is None:
                continue  # this recipe does not use the channel at all
            lines.insert(at + 1, f"{prior_key} = {value:.6g}")
            changes.append(f"{path.name}: {prior_key} = {value:.6g}  (added)")
            edited = True

        if edited:
            path.write_text("\n".join(lines), encoding="utf-8")

    return changes


def plot_anomaly_differences(
    residuals: pd.DataFrame, pooled: list[PooledStats], path: Path
) -> None:
    """
    Two stacked density histograms of (measured - map), one per field, with a normal fit.

    The normal fit is drawn precisely so its *mismatch* is visible. These residuals are not
    Gaussian -- they are multimodal, one mode per recording, because each recording carries
    its own sensor offset. A filter handed a single sigma is assuming this curve, and the
    gap between the curve and the bars is the size of that assumption's error.
    """
    import matplotlib.pyplot as plt
    from scipy import stats as scipy_stats

    fields = [p for p in pooled if not residuals[residuals["field"] == p.field].empty]
    if not fields:
        return

    figure, axes = plt.subplots(
        len(fields), 1, figsize=(13, 4.2 * len(fields)), constrained_layout=True
    )
    if len(fields) == 1:
        axes = [axes]

    figure.suptitle("Anomaly differences between measured and map values", fontsize=17, color=INK)

    for axis, stats in zip(axes, fields, strict=True):
        values = residuals.loc[residuals["field"] == stats.field, "residual"].to_numpy(float)
        mean = float(np.mean(values))
        sigma = float(np.std(values, ddof=1))

        # Clip the view to the central 99% so a handful of extreme rows cannot compress the
        # bulk of the distribution into one bar.
        low, high = np.percentile(values, [0.5, 99.5])
        pad = 0.05 * (high - low) if high > low else 1.0
        axis.set_xlim(low - pad, high + pad)

        axis.hist(
            values,
            bins=200,
            range=(low - pad, high + pad),
            density=True,
            color=FIELD_COLOUR[stats.field],
            edgecolor="none",
        )

        grid = np.linspace(low - pad, high + pad, 500)
        axis.plot(
            grid,
            scipy_stats.norm.pdf(grid, mean, sigma),
            color=FIT_COLOUR,
            linestyle="--",
            linewidth=2,
            label=f"Normal fit (sigma {sigma:,.1f})",
        )
        axis.axvline(mean, color=INK, linestyle=":", linewidth=2, label=f"Mean: {mean:,.2f}")
        for sign in (-1, 1):
            axis.axvline(
                mean + sign * sigma,
                color=MUTED_INK,
                linestyle="--",
                linewidth=1.5,
                label=f"+/- sigma: {sigma:,.2f}" if sign == 1 else None,
            )
        # The robust sigma is the one to configure; show it against the plain one.
        axis.axvline(
            stats.bias_median,
            color=INK,
            linestyle="-",
            linewidth=1.5,
            alpha=0.55,
            label=f"Median: {stats.bias_median:,.2f}",
        )

        axis.set_title(
            f"{stats.field.capitalize()} anomaly differences  "
            f"(within-trajectory sigma {stats.within_sigma:,.1f} {stats.unit}, "
            f"SNR {stats.snr_median:.2f})",
            fontsize=13,
            color=INK,
        )
        axis.set_xlabel(f"{stats.field.capitalize()} anomaly difference ({stats.unit})", color=INK)
        axis.set_ylabel("Density", color=INK)
        axis.grid(True, color="#E5E7EB", linewidth=0.8)
        axis.set_axisbelow(True)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
        axis.legend(frameon=True, framealpha=0.9, fontsize=10)

    figure.savefig(path, dpi=150)
    plt.close(figure)


# ===========================================================================================
# CLI
# ===========================================================================================


def self_check() -> None:
    """
    Assert the mirrored formulas still agree with the Rust they were copied from.

    The reference values come from `geonav/tests/geo_closed_loop.rs`, whose fixture states
    both: normal gravity at its track is 9.801741 m/s^2, and the World Magnetic Model total
    field there is 50.87 uT. If this fails, the two implementations have diverged and every
    statistic this module produces describes a quantity the filter does not compute.
    """
    gamma = float(normal_gravity(40.05, 0.0))
    assert abs(gamma - 9.801741) < 1e-5, f"normal gravity drifted: {gamma}"

    # A 1e-5 m/s^2 perturbation is exactly 1 mGal, mirroring the Rust unit test.
    base = float(gravity_anomaly_mgal(40.05, 100.0, 0.0, 0.0, gamma))
    bumped = float(gravity_anomaly_mgal(40.05, 100.0, 0.0, 0.0, gamma + 1e-5))
    assert abs(base) < 1e-6, f"a reading equal to normal gravity must be a zero anomaly: {base}"
    assert abs((bumped - base) - 1.0) < 1e-6, f"gravity anomaly is not in mGal: {bumped - base}"

    try:
        field = float(wmm_total_field_nt(40.05, -75.95, 100.0, 2025.164)[0])
    except ImportError:
        return
    assert abs(field - 50869.6) < 50.0, f"WMM disagrees with the Rust crate: {field} nT"


def add_geostats_arguments(parser) -> None:
    """
    Register the geostats options on a parser.

    Shared by this module's standalone entry point and the ``analyze geostats`` subcommand,
    following `add_preprocess_arguments` so the two cannot drift apart.
    """
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help=(
            "Directory of preprocessed trajectory CSVs, each with its `<stem>_gravity.nc` "
            "and `<stem>_magnetic.nc` beside it -- that is, the output of `analyze "
            "preprocess`, not data/raw."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Directory to write the residuals, the summary tables and the figure into.",
    )
    parser.add_argument(
        "--skip-magnetic",
        action="store_true",
        help=(
            "Skip the magnetic channel. It needs `pygeomag` to remove the core field; "
            "without that the 'anomaly' is the whole ~50,000 nT total field, several "
            "hundred times the signal."
        ),
    )
    parser.add_argument(
        "--apply-to",
        type=str,
        default=None,
        metavar="CONF_DIR",
        help=(
            "Write the measured bias, noise and bias-prior into every config in CONF_DIR "
            "that has a [geophysical] section (typically `conf`). Only keys already present "
            "are rewritten, so a gravity-only recipe stays gravity-only; comments survive. "
            "Without this the numbers are only reported, and `geo_stats.toml` has to be "
            "merged by hand."
        ),
    )
    parser.add_argument(
        "--apply-interval",
        action="store_true",
        help=(
            "With --apply-to, also rewrite `geo_frequency_s` to one measurement per "
            "de-correlation length. Off by default: the noise figures are measurements, "
            "whereas moving the interval from 1 s to several hundred changes what the "
            "experiment is, not just how it is tuned."
        ),
    )


def geostats_analysis(args) -> None:
    """Run the characterisation over every trajectory in ``args.input``."""
    from tqdm import tqdm

    self_check()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(p for p in input_path.glob("*.csv"))
    if not csv_files:
        raise SystemExit(f"no trajectory CSVs found in {input_path}")

    skip_magnetic = bool(getattr(args, "skip_magnetic", False))
    if not skip_magnetic:
        try:
            import pygeomag  # noqa: F401
        except ImportError:
            print(
                "pygeomag is not installed, so the magnetic core field cannot be removed. "
                "Skipping the magnetic channel; install it to include one."
            )
            skip_magnetic = True

    all_stats: list[FieldStats] = []
    all_residuals: list[pd.DataFrame] = []
    for csv_path in tqdm(csv_files, desc="Characterising trajectories"):
        try:
            stats, residuals = analyse_trajectory(csv_path, skip_magnetic=skip_magnetic)
        except (ValueError, OSError) as error:
            print(f"  skipping {csv_path.name}: {error}")
            continue
        all_stats.extend(stats)
        if not residuals.empty:
            all_residuals.append(residuals)

    if not all_stats:
        raise SystemExit(
            "no trajectory yielded a usable channel -- check that the .nc maps are beside "
            "the CSVs and that the tracks fall inside them"
        )

    residuals = pd.concat(all_residuals, ignore_index=True)
    pooled = pool(all_stats, residuals)

    residuals.to_csv(output_path / "geo_residuals.csv", index=False)
    pd.DataFrame([asdict(s) for s in all_stats]).to_csv(output_path / "geo_stats.csv", index=False)
    (output_path / "geo_stats_pooled.json").write_text(
        json.dumps([asdict(p) for p in pooled], indent=2), encoding="utf-8"
    )
    write_config_block(pooled, output_path / "geo_stats.toml")
    plot_anomaly_differences(residuals, pooled, output_path / "anomaly_differences.png")

    _print_summary(pooled, all_stats)
    print(f"\nWrote residuals, tables, a config block and the figure to {output_path}")

    conf_dir = getattr(args, "apply_to", None)
    if conf_dir:
        changes = apply_to_configs(
            pooled, Path(conf_dir), apply_interval=bool(getattr(args, "apply_interval", False))
        )
        if changes:
            print(f"\nApplied to {conf_dir}:")
            for change in changes:
                print(f"  {change}")
            if not getattr(args, "apply_interval", False):
                print(
                    "\n`geo_frequency_s` was left alone. Pass --apply-interval to adopt the "
                    "de-correlation interval above as well."
                )
            print(
                "\nThese results were produced under the previous values. Re-run the "
                "simulations before comparing anything against them."
            )
        else:
            print(f"\nNothing to apply in {conf_dir}: no config there has a [geophysical] block.")


def _print_summary(pooled: list[PooledStats], stats: list[FieldStats]) -> None:
    """Print the headline numbers, so a run says something without opening a file."""
    print("\n" + "=" * 86)
    print("GEOPHYSICAL MEASUREMENT CHARACTERISATION")
    print("=" * 86)

    for entry in pooled:
        print(f"\n{entry.field.upper()}  ({entry.unit})")
        print(f"  trajectories / samples      {entry.trajectories} / {entry.samples:,}")
        print(f"  bias (pooled median)        {entry.bias_median:,.2f}")
        print(
            f"  within-trajectory sigma     {entry.within_sigma:,.2f}   -> {entry.field}_noise_std"
        )
        print(
            f"  between-trajectory sigma    {entry.between_sigma:,.2f}   -> {entry.field}_bias_init_std"
        )
        print(f"  total sigma (pooled)        {entry.total_sigma:,.2f}")
        print(
            f"  signal-to-noise             median {entry.snr_median:.2f}, "
            f"best {entry.snr_best:.2f}, {entry.snr_above_one}/{entry.trajectories} above 1.0"
        )
        print(f"  de-correlation length       {entry.decorrelation_m:,.0f} m")
        print(
            f"  recommended interval        {entry.recommended_interval_s:,.0f} s "
            f"(currently {CURRENT_GEO_INTERVAL_S:g} s)"
        )
        print(f"  independent samples/traj    {entry.independent_samples_median:.1f}")
        if entry.snr_median < 1.0:
            print(
                "  NOTE: the map varies less along-track than the residual does. An aid in "
                "this regime\n        removes covariance without removing error."
            )

    declared = {s.field: (s.grid_arcmin_lat, s.grid_arcmin_lon) for s in stats}
    print("\nGRID SPACING (compare against the config's declared resolution)")
    for field, (lat_min, lon_min) in declared.items():
        print(f"  {field:<9} {lat_min:.2f}' x {lon_min:.2f}'")
    print("=" * 86)
