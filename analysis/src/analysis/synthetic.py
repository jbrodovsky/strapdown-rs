"""
Synthesise a vehicle-mounted gravimeter and magnetometer from the GNSS track and the maps.

The phone recordings carry no usable geophysical observation (GEO_AIDING_NOTES.md §3): the
``grav_*`` norm is pinned to a device constant, and the magnetometer sees ~56 µT of the car's
own field against a 73 nT map signal. This module replaces those two readings -- and nothing
else -- with what a low-cost dedicated sensor would have reported on the same drive:

    reading = physics at the GNSS truth + map at the GNSS truth + sensor error

The IMU, GNSS, barometer and attitude columns are untouched, so the synthetic arm runs the
same filters over the same inertial data and the same degraded GNSS as the real one, and
differs from it only in what the geophysical channel holds.

What it models, and what it does not
------------------------------------
**Sensor-intrinsic error only**: white noise, a per-drive turn-on bias and slow drift (bias
instability and residual temperature sensitivity), each taken from a datasheet or a published
characterisation of a real part under $100. Installation is ideal: the vehicle's own magnetic
field is calibrated out, its kinematic vertical acceleration is removed exactly, and the maps
are the truth. The result is therefore a *sensor-limited best case*, not a prediction of a
fielded system.

The two parts
-------------
**Magnetometer: PNI RM3100** (magneto-inductive; sensor suite $250 per 10). PNI, *RM3100 &
RM2100 Sensor Suite User Manual*, doc 1017252 r06, Table 3-1, at cycle count 200:

- noise **15 nT** per sample; maximum single-axis rate 440 Hz, so 440/3 Hz for all three axes.
  Averaged into a 10 Hz row, as preprocessing averages the phone's samples, that is
  15 nT x sqrt(10 / 146.7) = **3.92 nT**. Regoli et al. (2018, GI 7, 129) measured ~4.7 nT
  at 10 Hz and 2.2 nT at 1 s on a 40 Hz build -- the same level.
- repeatability 8 nT and hysteresis 15 nT (over +/-200 µT): the turn-on bias,
  hypot(8, 15) = **17 nT**.
- "inherently free from offset drift"; Regoli et al. saw no drift over 100 h at constant
  temperature. What does not average down is a low-frequency floor -- 2.2 nT at 1 s where
  white noise predicts 1.38 nT, so **1.7 nT**, taken with a one-hour correlation time -- and
  temperature: ~0.5 nT/°C, measured (Strabel et al. 2022, GI 11, 375, "preliminary"), and
  repeatable enough to calibrate out.

**Gravimeter: Analog Devices ADXL355** (3-axis MEMS accelerometer, $77.41 at Digi-Key), read
as a scalar gravimeter. *ADXL354/ADXL355 data sheet*, Rev. 0, Table 2 (digital output, ±2 g):

- noise density **25 µg/√Hz** (24.5 mGal/√Hz), consistent with the Z-axis root Allan variance
  of ~18 µg at 1 s (Fig. 56); later revisions quote 22.5. Per 10 Hz row: 24.5 x sqrt(5) =
  **54.8 mGal**.
- Z-axis repeatability **±9 mg** (8,826 mGal) "predicted for a 10 year life". The only turn-on
  figure the datasheet gives, and a conservative one; the filter estimates the bias in-run.
- root-Allan-variance floor ~3.5 µg at 100-300 s (Fig. 56): bias instability, modelled as a
  first-order Gauss-Markov process whose Allan deviation peaks at that floor.
- 0 g offset vs temperature ±0.02 mg/°C typical and sensitivity ±0.01 %/°C: on an axis
  reading 1 g that is ~100 mGal/°C, which is what makes a MEMS accelerometer a poor gravimeter.

The temperature terms rest on three stated assumptions, not datasheet values:
:data:`TEMPERATURE_COMPENSATION_RESIDUAL`, :data:`CABIN_TEMPERATURE_STD_C` and
:data:`CABIN_TEMPERATURE_TAU_S`.

How the readings are formed
---------------------------
The filter never sees a map value. It sees ``|grav_*|`` and ``|mag_*|`` and forms the anomaly
itself, at its own state (``geonav::build_event_stream``, mirrored in :mod:`analysis.geostats`):

    gravity   1e5 * (|g| - gamma(lat, h) + E(lat, h, v_N, v_E))            mGal
    magnetic  1000 * |m| - WMM(lat, lon, h, t)                             nT

So the synthetic reading inverts exactly that, at the GNSS truth: at the true state the filter
recovers ``map + error`` and nothing else. The mirrored functions are the ones
:func:`analysis.geostats.self_check` pins to the Rust.

Each vector keeps its **direction** and only its norm is replaced. The magnetometer's direction
feeds a 1 Hz yaw update in every run, the non-geophysical baselines included, so rescaling is
what keeps those baselines unchanged. ``grav_*`` is read by the geophysical channel alone.

Geophysical updates fire on whole elapsed seconds, and only ~28% of those land on a row that
carries a GNSS fix, so every row needs a reading: the truth is interpolated between fixes
(:func:`truth_track`). Across a long gap that interpolation cannot be trusted for gravity, whose
Eötvös term moves ~11 mGal per m/s of east velocity, so gravity is left NaN wherever the
bracketing fixes are more than :data:`MAX_TRUTH_GAP_S` apart and the filter skips that update.
The magnetic reading has no velocity term and is kept on every row: blanking it would also
blank the yaw update.
"""

from __future__ import annotations

import json
import math
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from analysis.geostats import (
    MAD_TO_SIGMA,
    MGAL_PER_M_PER_S2,
    MICROTESLA_TO_NANOTESLA,
    AnomalyMap,
    decimal_year,
    eotvos,
    normal_gravity,
    wmm_total_field_nt,
)

#: Milligal per microgravity. 1 µg = 9.80665e-6 m/s^2.
MGAL_PER_MICRO_G = 0.980665

#: Row rate the `conf/*.toml` noise figures are written for, Hz -- `just preprocess`'s `-f 10`.
#:
#: The white noise per row depends on the row rate (a longer row averages more samples), so the
#: configs are only right for data preprocessed at this rate. At 1 Hz the synthetic rows are
#: sqrt(10) quieter than the configs say.
CONFIG_ROW_RATE_HZ = 10.0

#: Longest span between two GNSS velocity fixes that gravity is interpolated across, seconds.
#:
#: The recordings report at ~1 Hz; a few fixes arrive 2-4 s apart and 76 gaps exceed 5 s (up to
#: 92 s). A straight line through the velocity across a turn in one of those is wrong by metres
#: per second, i.e. tens to hundreds of mGal of Eötvös error, presented to the filter as signal.
MAX_TRUTH_GAP_S = 2.0

#: Share of a sensor's typical temperature coefficient left after first-order compensation
#: against its own temperature sensor. ASSUMPTION, not a datasheet value.
TEMPERATURE_COMPENSATION_RESIDUAL = 0.10

#: Standard deviation of the sensor's temperature over a drive, °C. ASSUMPTION: a mount in a
#: climate-controlled cabin.
CABIN_TEMPERATURE_STD_C = 1.0

#: Correlation time of that temperature, seconds. ASSUMPTION.
CABIN_TEMPERATURE_TAU_S = 1800.0

#: Allan deviation of a first-order Gauss-Markov process peaks at this multiple of its sigma, at
#: an averaging time of 1.89 correlation times. Used to turn a datasheet's Allan floor into the
#: process that reproduces it.
GAUSS_MARKOV_ADEV_PEAK = 0.6174

#: File, beside the trajectories, recording how they were synthesised. JSON rather than CSV
#: because the simulator loads every `*.csv` in its input directory as a trajectory.
PROVENANCE_FILE = "synthetic.json"

GRAVITY_COLUMNS = ("grav_x", "grav_y", "grav_z")
MAGNETIC_COLUMNS = ("mag_x", "mag_y", "mag_z")

#: Fixed keys that separate each channel's random stream from the other's.
_CHANNEL_STREAM = {"gravity": 0, "magnetic": 1}


# ===========================================================================================
# Sensor models
# ===========================================================================================


@dataclass(frozen=True)
class DriftTerm:
    """One first-order Gauss-Markov drift component of a sensor's bias."""

    #: Steady-state standard deviation, in the sensor's unit.
    sigma: float
    #: Correlation time, seconds.
    tau_s: float
    #: Where the number comes from.
    source: str


@dataclass(frozen=True)
class SensorModel:
    """A sensor's error model, in the unit the filter's anomaly is in (mGal or nT)."""

    #: `gravity` or `magnetic`: the prefix of the `[geophysical]` keys it sets.
    channel: str
    #: The part modelled.
    part: str
    #: mGal or nT.
    unit: str
    #: White-noise density, unit per sqrt(Hz).
    noise_density: float
    #: Standard deviation of the constant bias drawn once per drive.
    turn_on_std: float
    #: Slow drift, summed.
    drift: tuple[DriftTerm, ...]
    #: Where the white noise and the turn-on bias come from.
    citation: str

    def sigma_row(self, row_rate_hz: float) -> float:
        """
        White-noise standard deviation of one row.

        A row is the mean of the sensor's samples over ``1 / row_rate_hz`` seconds, as
        `clean_phone_data` averages the phone's. A boxcar of length T passes white noise of
        one-sided density n with variance ``n^2 / (2 T)``.
        """
        return self.noise_density * math.sqrt(row_rate_hz / 2.0)

    def config_values(self, row_rate_hz: float = CONFIG_ROW_RATE_HZ) -> dict[str, float]:
        """
        The `[geophysical]` keys that describe this sensor to the filter.

        - ``*_bias`` is 0: the turn-on bias is drawn with zero mean.
        - ``*_noise_std`` is the white noise per row, since the filter reads one row per update.
        - ``*_bias_init_std`` covers the turn-on bias *and* the drift's value at the start,
          which is drawn from its stationary distribution.
        - ``*_bias_process_noise_std`` is the random-walk rate whose variance growth matches
          the Gauss-Markov drift over short intervals, sqrt(sum 2 sigma^2 / tau). Over a
          whole drive it over-states the drift, which errs on the safe side.
        """
        drift_variance = sum(term.sigma**2 for term in self.drift)
        walk_rate = math.sqrt(sum(2.0 * term.sigma**2 / term.tau_s for term in self.drift))
        return {
            f"{self.channel}_bias": 0.0,
            f"{self.channel}_noise_std": self.sigma_row(row_rate_hz),
            f"{self.channel}_bias_init_std": math.sqrt(self.turn_on_std**2 + drift_variance),
            f"{self.channel}_bias_process_noise_std": walk_rate,
        }

    def variation_values(self) -> dict[str, float]:
        """
        The `[particle_filter]` keys that describe this sensor's drift to the RBPF.

        The particle filter models a map bias as Canciani & Raquet do: a constant offset ``c``
        plus a first-order Gauss-Markov temporal variation ``V``. The turn-on bias is the
        constant; the drift terms become ``V``, collapsed into the one process with the same
        stationary variance and the same short-term drive as their sum:

        - ``sigma_V^2 = sum sigma_i^2``;
        - ``tau_V = sum sigma_i^2 / sum(sigma_i^2 / tau_i)``, so that ``2 sigma_V^2 / tau_V``
          is ``walk_rate^2`` -- the random walk `config_values` gives the Kalman filters.
        """
        drift_variance = sum(term.sigma**2 for term in self.drift)
        drive = sum(term.sigma**2 / term.tau_s for term in self.drift)
        return {
            f"{self.channel}_variation_std": math.sqrt(drift_variance),
            f"{self.channel}_variation_time_constant_s": drift_variance / drive,
        }


#: RM3100 noise per sample at cycle count 200, nT (User Manual, Table 3-1).
RM3100_SAMPLE_NOISE_NT = 15.0
#: RM3100 three-axis sample rate at cycle count 200, Hz: the 440 Hz single-axis maximum over 3.
RM3100_THREE_AXIS_RATE_HZ = 440.0 / 3.0
#: RM3100 temperature coefficient, nT/°C (Strabel et al. 2022, preliminary).
RM3100_TEMPERATURE_COEFFICIENT_NT = 0.5
#: RM3100 noise that does not average down, nT. Regoli et al. (2018, §3.1) measured 8.73 nT per
#: 40 Hz sample and 2.2 nT once 40 of them are averaged into 1 s; white noise alone would give
#: 8.73 / sqrt(40) = 1.38 nT. The difference, in quadrature, is a low-frequency floor.
RM3100_LOW_FREQUENCY_NT = math.sqrt(2.2**2 - (8.73 / math.sqrt(40.0)) ** 2)

RM3100 = SensorModel(
    channel="magnetic",
    part="PNI RM3100, cycle count 200",
    unit="nT",
    # 15 nT per sample at 146.7 samples/s is a density of 15 / sqrt(146.7 / 2) = 1.75 nT/rtHz.
    noise_density=RM3100_SAMPLE_NOISE_NT / math.sqrt(RM3100_THREE_AXIS_RATE_HZ / 2.0),
    turn_on_std=math.hypot(8.0, 15.0),
    drift=(
        DriftTerm(
            sigma=RM3100_LOW_FREQUENCY_NT,
            tau_s=3600.0,
            source="Regoli et al. 2018: 2.2 nT at 1 s against 1.38 nT of white noise",
        ),
        DriftTerm(
            sigma=RM3100_TEMPERATURE_COEFFICIENT_NT
            * TEMPERATURE_COMPENSATION_RESIDUAL
            * CABIN_TEMPERATURE_STD_C,
            tau_s=CABIN_TEMPERATURE_TAU_S,
            source="0.5 nT/°C (Strabel et al. 2022) x compensation residual x cabin sigma",
        ),
    ),
    citation=(
        "PNI RM3100 & RM2100 Sensor Suite User Manual, doc 1017252 r06, Table 3-1: 15 nT noise, "
        "8 nT repeatability, 15 nT hysteresis at cycle count 200; 440 Hz single-axis rate"
    ),
)

#: ADXL355 Z-axis root-Allan-variance floor, µg (data sheet Rev. 0, Fig. 56, 100-300 s).
ADXL355_ALLAN_FLOOR_MICRO_G = 3.5
#: ADXL355 temperature coefficient on an axis reading 1 g, mGal/°C: the 0.02 mg/°C typical
#: offset drift and the 0.01 %/°C sensitivity drift (100 µg/°C at 1 g), in quadrature.
ADXL355_TEMPERATURE_COEFFICIENT_MGAL = math.hypot(20.0, 100.0) * MGAL_PER_MICRO_G

ADXL355 = SensorModel(
    channel="gravity",
    part="Analog Devices ADXL355, ±2 g, read as a scalar gravimeter",
    unit="mGal",
    noise_density=25.0 * MGAL_PER_MICRO_G,
    turn_on_std=9000.0 * MGAL_PER_MICRO_G,
    drift=(
        DriftTerm(
            sigma=ADXL355_ALLAN_FLOOR_MICRO_G / GAUSS_MARKOV_ADEV_PEAK * MGAL_PER_MICRO_G,
            tau_s=300.0,
            source="Z-axis Allan floor 3.5 µg at 100-300 s (data sheet Rev. 0, Fig. 56)",
        ),
        DriftTerm(
            sigma=ADXL355_TEMPERATURE_COEFFICIENT_MGAL
            * TEMPERATURE_COMPENSATION_RESIDUAL
            * CABIN_TEMPERATURE_STD_C,
            tau_s=CABIN_TEMPERATURE_TAU_S,
            source="~100 mGal/°C at 1 g (Table 2) x compensation residual x cabin sigma",
        ),
    ),
    citation=(
        "ADXL354/ADXL355 data sheet Rev. 0, Table 2: 25 µg/rtHz noise density (±2 g), "
        "±9 mg Z-axis repeatability, 0.02 mg/°C offset and 0.01 %/°C sensitivity drift"
    ),
)

#: The sensors `--synthetic` models, by channel.
SENSORS: dict[str, SensorModel] = {"gravity": ADXL355, "magnetic": RM3100}


# ===========================================================================================
# The truth track
# ===========================================================================================


@dataclass
class TruthTrack:
    """The GNSS track interpolated onto every row of a recording."""

    latitude: np.ndarray
    longitude: np.ndarray
    altitude: np.ndarray
    north_velocity: np.ndarray
    east_velocity: np.ndarray
    #: Span between the two velocity fixes bracketing each row, seconds. 0 on a fix, inf
    #: before the first fix and after the last.
    velocity_gap_s: np.ndarray


def elapsed_seconds(frame: pd.DataFrame) -> np.ndarray:
    """Seconds since the first row, from the frame's datetime index."""
    return np.asarray((frame.index - frame.index[0]).total_seconds(), dtype=float)


def truth_track(frame: pd.DataFrame) -> TruthTrack:
    """
    Interpolate the recorded GNSS fixes onto every row.

    A fix is a row with finite latitude, longitude and altitude -- one row in ten at 10 Hz. The
    velocity is interpolated as north and east components, never as a bearing, which wraps at
    360°. Two recorded quirks are handled:

    - iOS reports a bearing of -1 when it has no course. All 62 such fixes in this data set are
      at a crawl (0.63 m/s at most), so they are taken as zero velocity.
    - A negative speed is not a measurement; that fix is dropped from the velocity only.

    Ends are held at the nearest fix, which matters for position only: rows outside the
    velocity fixes are marked with an infinite gap and receive no gravity reading.

    Raises
    ------
    ValueError
        With fewer than two fixes, where there is nothing to interpolate.
    """
    seconds = elapsed_seconds(frame)
    latitude = frame["latitude"].to_numpy(float)
    longitude = frame["longitude"].to_numpy(float)
    altitude = frame["altitude"].to_numpy(float)
    fix = np.isfinite(latitude) & np.isfinite(longitude) & np.isfinite(altitude)
    if fix.sum() < 2:
        raise ValueError("fewer than two GNSS fixes: no truth track to synthesise along")

    speed = frame["speed"].to_numpy(float)[fix]
    bearing = frame["bearing"].to_numpy(float)[fix]
    stopped = np.isfinite(bearing) & (bearing < 0.0)
    moving = np.isfinite(speed) & (speed >= 0.0) & np.isfinite(bearing)
    north = np.where(stopped, 0.0, speed * np.cos(np.radians(bearing)))
    east = np.where(stopped, 0.0, speed * np.sin(np.radians(bearing)))
    if moving.sum() < 2:
        raise ValueError("fewer than two GNSS velocity fixes: no truth velocity")

    fix_seconds = seconds[fix]
    velocity_seconds = fix_seconds[moving]

    right = np.searchsorted(velocity_seconds, seconds, side="left")
    count = velocity_seconds.size
    on_fix = (right < count) & (velocity_seconds[np.minimum(right, count - 1)] == seconds)
    inside = (right > 0) & (right < count)
    gap = np.full(seconds.shape, np.inf)
    gap[inside] = velocity_seconds[right[inside]] - velocity_seconds[right[inside] - 1]
    gap[on_fix] = 0.0

    return TruthTrack(
        latitude=np.interp(seconds, fix_seconds, latitude[fix]),
        longitude=np.interp(seconds, fix_seconds, longitude[fix]),
        altitude=np.interp(seconds, fix_seconds, altitude[fix]),
        north_velocity=np.interp(seconds, velocity_seconds, north[moving]),
        east_velocity=np.interp(seconds, velocity_seconds, east[moving]),
        velocity_gap_s=gap,
    )


# ===========================================================================================
# Sensor errors
# ===========================================================================================


def channel_rng(seed: int, stem: str, channel: str) -> np.random.Generator:
    """
    The random stream for one channel of one trajectory.

    Keyed by the trajectory's file stem rather than drawn from one shared generator, so a
    trajectory's errors do not depend on which recordings were processed before it, or in what
    order. `zlib.crc32` rather than `hash`, which Python salts per process.
    """
    return np.random.default_rng([seed, zlib.crc32(stem.encode("utf-8")), _CHANNEL_STREAM[channel]])


def gauss_markov(
    rows: int, sigma: float, tau_s: float, step_s: float, rng: np.random.Generator
) -> np.ndarray:
    """
    A first-order Gauss-Markov sequence on a uniform grid, started from its stationary law.

    ``x[k] = phi * x[k-1] + sigma * sqrt(1 - phi^2) * w[k]`` with ``phi = exp(-step / tau)``,
    and ``x[0] ~ N(0, sigma^2)``, so the variance is sigma^2 at every row.
    """
    phi = math.exp(-step_s / tau_s)
    gain = sigma * math.sqrt(1.0 - phi * phi)
    drive = rng.standard_normal(rows)
    if rows and gain > 0.0:
        drive[0] /= math.sqrt(1.0 - phi * phi)
    return lfilter([gain], [1.0, -phi], drive)


def channel_error(
    model: SensorModel, rows: int, row_rate_hz: float, rng: np.random.Generator
) -> tuple[np.ndarray, float]:
    """
    One drive's error series for one sensor: turn-on bias + drift + white noise.

    Every row is drawn, masked or not, and always in the same order, so a change to which rows
    are written never changes the draws on the others.

    Returns
    -------
    tuple
        ``(error, turn_on)`` -- the per-row error in the model's unit, and the turn-on bias
        drawn for this drive.
    """
    turn_on = float(model.turn_on_std * rng.standard_normal())
    error = np.full(rows, turn_on)
    for term in model.drift:
        error += gauss_markov(rows, term.sigma, term.tau_s, 1.0 / row_rate_hz, rng)
    error += model.sigma_row(row_rate_hz) * rng.standard_normal(rows)
    return error, turn_on


# ===========================================================================================
# Synthesis
# ===========================================================================================


def _row_rate_hz(seconds: np.ndarray) -> float:
    """The frame's row rate, which `clean_phone_data` makes uniform by resampling."""
    steps = np.diff(seconds)
    if steps.size == 0 or not np.all(np.isclose(steps, steps[0], rtol=1e-6, atol=1e-9)):
        raise ValueError(
            "synthesis needs uniformly spaced rows, as `clean_phone_data` resamples them"
        )
    return 1.0 / float(steps[0])


def _residual_summary(error: np.ndarray, rows: np.ndarray) -> dict[str, float]:
    """Median and robust sigma of the error on the rows `analyze geostats` scores."""
    values = error[rows]
    if values.size == 0:
        return {"error_median": float("nan"), "error_sigma_robust": float("nan")}
    median = float(np.median(values))
    return {
        "error_median": median,
        "error_sigma_robust": float(MAD_TO_SIGMA * np.median(np.abs(values - median))),
    }


def _load_map(path: Path) -> AnomalyMap:
    """Load a map, naming the flag that writes it when it is missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. --synthetic samples the maps, so it needs them beside each "
            "trajectory: pass --getmaps."
        )
    return AnomalyMap.load(path)


def _rescaled(vectors: np.ndarray, norms: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Replace the norm of ``vectors[rows]`` by ``norms[rows]``, keeping each direction."""
    out = vectors.copy()
    lengths = np.linalg.norm(vectors[rows], axis=1)
    out[rows] = vectors[rows] * (norms[rows] / lengths)[:, None]
    return out


def synthesize_gravity(
    frame: pd.DataFrame, truth: TruthTrack, anomaly_map: AnomalyMap, error: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    The synthetic gravimeter, as new ``grav_*`` columns.

    The norm is what a gravimeter carried along the truth reads -- normal gravity at its height,
    less the Eötvös term, plus the anomaly and the sensor error -- so that the filter's
    ``1e5 * (|g| - gamma + E)`` at the true state is exactly ``map + error``.

    Returns
    -------
    tuple
        ``(columns, written, masked)``: the new ``grav_*`` values, and which rows were written
        and which were blanked for lack of a trustworthy truth velocity.
    """
    vectors = frame[list(GRAVITY_COLUMNS)].to_numpy(float)
    present = np.all(np.isfinite(vectors), axis=1)
    covered = truth.velocity_gap_s <= MAX_TRUTH_GAP_S
    written = present & covered

    anomaly = anomaly_map.sample(truth.latitude, truth.longitude)
    if np.any(written & ~np.isfinite(anomaly)):
        raise ValueError(f"the truth track leaves {anomaly_map.path.name}")

    reading = (
        normal_gravity(truth.latitude, truth.altitude)
        - eotvos(truth.latitude, truth.altitude, truth.north_velocity, truth.east_velocity)
        + (anomaly + error) / MGAL_PER_M_PER_S2
    )
    columns = np.full(vectors.shape, np.nan)
    columns[written] = _rescaled(vectors, reading, written)[written]
    return columns, written, present & ~covered


def synthesize_magnetic(
    frame: pd.DataFrame, truth: TruthTrack, anomaly_map: AnomalyMap, error: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    The synthetic magnetometer, as new ``mag_*`` columns.

    The norm is the World Magnetic Model at the truth plus the anomaly and the sensor error, in
    µT, so that the filter's ``1000 * |m| - WMM`` at the true state is ``map + error``. Every
    row with a finite reading is written and none is blanked: the direction drives the yaw
    update of every run.

    Returns
    -------
    tuple
        ``(columns, written)``.

    Raises
    ------
    ValueError
        If a written row is off the map or has a zero vector, whose direction cannot be kept.
    """
    vectors = frame[list(MAGNETIC_COLUMNS)].to_numpy(float)
    written = np.all(np.isfinite(vectors), axis=1)
    if np.any(written & (np.linalg.norm(vectors, axis=1) == 0.0)):
        raise ValueError("a magnetometer row is a zero vector: there is no direction to keep")

    anomaly = anomaly_map.sample(truth.latitude, truth.longitude)
    if np.any(written & ~np.isfinite(anomaly)):
        raise ValueError(f"the truth track leaves {anomaly_map.path.name}")

    reference = np.full(len(frame), np.nan)
    years = decimal_year(pd.Series(frame.index[written]))
    reference[written] = wmm_total_field_nt(
        truth.latitude[written], truth.longitude[written], truth.altitude[written], years
    )
    reading = (reference + anomaly + error) / MICROTESLA_TO_NANOTESLA
    return _rescaled(vectors, reading, written), written


def synthesize_geophysical(
    frame: pd.DataFrame,
    stem: str,
    gravity_map: Path,
    magnetic_map: Path,
    seed: int,
    sensors: dict[str, SensorModel] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Replace a trajectory's ``grav_*`` and ``mag_*`` with synthetic sensor readings.

    Parameters
    ----------
    frame : pd.DataFrame
        One preprocessed segment, on a uniform datetime index.
    stem : str
        Its file stem; with ``seed`` it keys the random streams.
    gravity_map, magnetic_map : Path
        The ``<stem>_gravity.nc`` and ``<stem>_magnetic.nc`` the simulator will read.
    seed : int
        Base seed of the sensor errors.
    sensors : dict, optional
        The models by channel; :data:`SENSORS` by default.

    Returns
    -------
    tuple
        ``(frame, record)`` -- a copy with the two readings replaced and every other column
        untouched, and a provenance record for `synthetic.json`.
    """
    sensors = SENSORS if sensors is None else sensors
    maps = {"gravity": _load_map(gravity_map), "magnetic": _load_map(magnetic_map)}
    truth = truth_track(frame)
    row_rate_hz = _row_rate_hz(elapsed_seconds(frame))
    # The rows `analyze geostats` scores: those carrying a GNSS fix.
    on_fix = frame[["latitude", "longitude", "altitude"]].notna().all(axis=1).to_numpy()

    out = frame.copy()
    record: dict = {"trajectory": stem, "rows": len(frame), "row_rate_hz": row_rate_hz}
    for channel in ("gravity", "magnetic"):
        model = sensors[channel]
        error, turn_on = channel_error(
            model, len(frame), row_rate_hz, channel_rng(seed, stem, channel)
        )
        if channel == "gravity":
            columns, written, masked = synthesize_gravity(frame, truth, maps[channel], error)
            out[list(GRAVITY_COLUMNS)] = columns
        else:
            columns, written = synthesize_magnetic(frame, truth, maps[channel], error)
            masked = np.zeros(len(frame), dtype=bool)
            out[list(MAGNETIC_COLUMNS)] = columns
        record[channel] = {
            "turn_on": turn_on,
            "rows_written": int(written.sum()),
            "rows_masked": int(masked.sum()),
            **_residual_summary(error, written & on_fix),
        }
    return out, record


def write_provenance(records: list[dict], path: Path, seed: int) -> None:
    """
    Record how the trajectories beside ``path`` were synthesised.

    The sensor models, the assumptions, the `[geophysical]` values they imply at
    :data:`CONFIG_ROW_RATE_HZ`, and per trajectory the turn-on bias drawn and the error
    statistics on the rows `analyze geostats` scores -- so its report can be checked against
    what was planted, trajectory by trajectory.
    """
    document = {
        "generator": "analyze preprocess --synthetic",
        "seed": seed,
        "config_row_rate_hz": CONFIG_ROW_RATE_HZ,
        "max_truth_gap_s": MAX_TRUTH_GAP_S,
        "assumptions": {
            "temperature_compensation_residual": TEMPERATURE_COMPENSATION_RESIDUAL,
            "cabin_temperature_std_c": CABIN_TEMPERATURE_STD_C,
            "cabin_temperature_tau_s": CABIN_TEMPERATURE_TAU_S,
        },
        "sensors": {
            channel: {**asdict(model), "config_values": model.config_values()}
            for channel, model in SENSORS.items()
        },
        "trajectories": records,
    }
    path.write_text(json.dumps(document, indent=2), encoding="utf-8")
