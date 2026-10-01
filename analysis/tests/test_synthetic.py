"""
Tests for `analyze preprocess --synthetic`.

The central assertion is exactness: with every sensor error switched off, the anomaly the
*filter* forms from a synthetic row -- ``1e5 * (|g| - gamma + E)`` and ``1000 * |m| - WMM``,
evaluated at the true state -- must equal the map at the true position on every row, fixes and
the nine interpolated rows between them alike. Everything else here checks the error model on
top of that: that `analyze geostats` recovers what was planted, and that the configs the
simulator runs describe the same sensors.
"""

from __future__ import annotations

import dataclasses
import json
import math
import tomllib
from argparse import Namespace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from analysis.geostats import (
    METRES_PER_DEGREE,
    MGAL_PER_M_PER_S2,
    MICROTESLA_TO_NANOTESLA,
    AnomalyMap,
    analyse_trajectory,
    decimal_year,
    eotvos,
    normal_gravity,
    wmm_total_field_nt,
)
from analysis.synthetic import (
    CONFIG_ROW_RATE_HZ,
    GRAVITY_COLUMNS,
    MAGNETIC_COLUMNS,
    MAX_TRUTH_GAP_S,
    SENSORS,
    gauss_markov,
    synthesize_geophysical,
    truth_track,
    write_provenance,
)

from analysis import preprocess

CONF = Path(__file__).resolve().parents[2] / "conf"

RATE_HZ = 10
DURATION_S = 1200
LAT0, LON0 = 40.05, -75.95
SPEED = 20.0
BEARING = 60.0
ALTITUDE = 100.0
NORTH = SPEED * math.cos(math.radians(BEARING))
EAST = SPEED * math.sin(math.radians(BEARING))


def _truth(seconds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A straight track at constant velocity: linear in time, so interpolation is exact."""
    latitude = LAT0 + NORTH * seconds / METRES_PER_DEGREE
    longitude = LON0 + EAST * seconds / (METRES_PER_DEGREE * math.cos(math.radians(LAT0)))
    return latitude, longitude


def _frame(gaps: tuple[tuple[float, float], ...] = ()) -> pd.DataFrame:
    """
    10 Hz rows with a GNSS fix on every tenth, as `clean_phone_data` produces them.

    The phone's own gravity and magnetometer vectors are given a direction that turns row by
    row, so direction-keeping is tested on more than one vector.
    """
    rows = DURATION_S * RATE_HZ
    seconds = np.arange(rows) / RATE_HZ
    index = pd.date_range("2025-03-01T00:00:00Z", periods=rows, freq="100ms", name="time")
    fix = np.arange(rows) % RATE_HZ == 0
    for start, end in gaps:
        fix[(seconds > start) & (seconds < end)] = False

    latitude, longitude = _truth(seconds)
    turn = 0.01 * seconds
    frame = pd.DataFrame(
        {
            "latitude": np.where(fix, latitude, np.nan),
            "longitude": np.where(fix, longitude, np.nan),
            "altitude": np.where(fix, ALTITUDE, np.nan),
            "speed": np.where(fix, SPEED, np.nan),
            "bearing": np.where(fix, BEARING, np.nan),
            "grav_x": 0.3 * np.cos(turn),
            "grav_y": 0.3 * np.sin(turn),
            "grav_z": np.full(rows, 9.8),
            "mag_x": 20.0 * np.cos(turn),
            "mag_y": 20.0 * np.sin(turn),
            "mag_z": np.full(rows, -45.0),
        },
        index=index,
    )
    return frame


def _write_map(path: Path, amplitude: float, offset: float) -> None:
    lats = np.linspace(LAT0 - 0.2, LAT0 + 0.4, 240)
    lons = np.linspace(LON0 - 0.2, LON0 + 0.5, 240)
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
    values = offset + amplitude * (
        np.sin((lat_grid - lats[0]) * 220.0) + np.cos((lon_grid - lons[0]) * 180.0)
    )
    xr.Dataset({"z": (("lat", "lon"), values)}, coords={"lat": lats, "lon": lons}).to_netcdf(path)


@pytest.fixture(scope="module")
def maps(tmp_path_factory) -> tuple[Path, Path]:
    directory = tmp_path_factory.mktemp("maps")
    gravity, magnetic = directory / "track_gravity.nc", directory / "track_magnetic.nc"
    _write_map(gravity, 20.0, 0.0)
    _write_map(magnetic, 120.0, 50.0)
    return gravity, magnetic


def _silent(model):
    """The same sensor with every error term zeroed."""
    return dataclasses.replace(
        model,
        noise_density=0.0,
        turn_on_std=0.0,
        drift=tuple(dataclasses.replace(term, sigma=0.0) for term in model.drift),
    )


SILENT = {channel: _silent(model) for channel, model in SENSORS.items()}
WHITE_AND_TURN_ON = {
    channel: dataclasses.replace(model, drift=()) for channel, model in SENSORS.items()
}


# ===========================================================================================
# The error model
# ===========================================================================================


def test_sigma_row_is_a_boxcar_average_of_the_noise_density():
    """Averaging white noise of density n over 0.1 s leaves n * sqrt(5)."""
    model = SENSORS["gravity"]
    rng = np.random.default_rng(1)
    rate = 1000.0
    samples = model.noise_density * math.sqrt(rate / 2.0) * rng.standard_normal(400_000)
    rows = samples.reshape(-1, int(rate / RATE_HZ)).mean(axis=1)
    assert np.std(rows) == pytest.approx(model.sigma_row(RATE_HZ), rel=0.05)
    assert model.sigma_row(RATE_HZ) == pytest.approx(model.noise_density * math.sqrt(5.0))


def test_the_magnetometer_noise_is_the_datasheet_sample_noise_averaged_into_a_row():
    """15 nT per sample at 440/3 Hz, averaged over 0.1 s: 15 * sqrt(10 / 146.7) nT."""
    assert SENSORS["magnetic"].sigma_row(10.0) == pytest.approx(15.0 * math.sqrt(30.0 / 440.0))


def test_config_values_follow_from_the_model():
    for model in SENSORS.values():
        values = model.config_values()
        channel = model.channel
        drift = [(term.sigma, term.tau_s) for term in model.drift]
        assert values[f"{channel}_bias"] == 0.0
        assert values[f"{channel}_noise_std"] == pytest.approx(model.sigma_row(10.0))
        assert values[f"{channel}_bias_init_std"] == pytest.approx(
            math.sqrt(model.turn_on_std**2 + sum(s**2 for s, _ in drift))
        )
        assert values[f"{channel}_bias_process_noise_std"] == pytest.approx(
            math.sqrt(sum(2.0 * s**2 / tau for s, tau in drift))
        )


def test_gauss_markov_is_stationary_with_the_requested_correlation():
    sigma, tau, step = 3.0, 1.0, 0.1
    series = gauss_markov(400_000, sigma, tau, step, np.random.default_rng(2))
    assert np.std(series) == pytest.approx(sigma, rel=0.05)
    lag = int(tau / step)
    correlation = np.corrcoef(series[:-lag], series[lag:])[0, 1]
    assert correlation == pytest.approx(math.exp(-1.0), abs=0.03)


# ===========================================================================================
# The truth track
# ===========================================================================================


def test_truth_velocity_comes_from_components_and_ios_no_course_is_a_stop():
    frame = _frame()
    stopped, dropped = frame.index[500], frame.index[700]
    frame.loc[stopped, ["speed", "bearing"]] = [0.5, -1.0]
    frame.loc[dropped, "speed"] = -1.0

    truth = truth_track(frame)

    assert truth.north_velocity[500] == 0.0
    assert truth.east_velocity[500] == 0.0
    # A negative speed is not a measurement: that fix is interpolated past, not used.
    assert truth.north_velocity[700] == pytest.approx(NORTH)
    assert truth.east_velocity[700] == pytest.approx(EAST)


def test_the_gap_between_fixes_is_reported_per_row():
    truth = truth_track(_frame(gaps=((100.0, 120.0),)))
    seconds = np.arange(DURATION_S * RATE_HZ) / RATE_HZ
    assert truth.velocity_gap_s[seconds == 50.0] == 0.0
    assert truth.velocity_gap_s[(seconds > 50.0) & (seconds < 51.0)] == pytest.approx(1.0)
    assert np.all(truth.velocity_gap_s[(seconds > 100.0) & (seconds < 120.0)] == 20.0)
    assert np.all(np.isinf(truth.velocity_gap_s[seconds > DURATION_S - 1]))


# ===========================================================================================
# Synthesis
# ===========================================================================================


def test_error_free_readings_give_the_filter_the_map_on_every_row(maps):
    """
    At the true state the filter's own anomaly is the map, on interpolated rows too.

    Only 28% of the simulator's geophysical updates land on a row with a fix, so the nine rows
    between fixes are the ones that matter most.
    """
    gravity_map, magnetic_map = maps
    frame = _frame()
    out, _ = synthesize_geophysical(frame, "track", gravity_map, magnetic_map, 42, SILENT)

    seconds = np.arange(len(frame)) / RATE_HZ
    latitude, longitude = _truth(seconds)
    altitude = np.full(len(frame), ALTITUDE)

    g = np.linalg.norm(out[list(GRAVITY_COLUMNS)].to_numpy(float), axis=1)
    gravity = (
        g - normal_gravity(latitude, altitude) + eotvos(latitude, altitude, NORTH, EAST)
    ) * MGAL_PER_M_PER_S2
    expected = AnomalyMap.load(gravity_map).sample(latitude, longitude)
    written = np.isfinite(g)
    # Everything up to the last fix; the rows after it have no velocity to bracket them.
    assert written.sum() == len(frame) - (RATE_HZ - 1)
    np.testing.assert_allclose(gravity[written], expected[written], atol=1e-6)

    m = np.linalg.norm(out[list(MAGNETIC_COLUMNS)].to_numpy(float), axis=1)
    reference = wmm_total_field_nt(
        latitude, longitude, altitude, decimal_year(pd.Series(frame.index))
    )
    magnetic = m * MICROTESLA_TO_NANOTESLA - reference
    expected = AnomalyMap.load(magnetic_map).sample(latitude, longitude)
    # The magnetometer is never blanked, so the rows after the last fix are written too, at
    # the last fix's position: under a second of track here, and 3.7 s at most in the data.
    assert np.all(np.isfinite(m))
    np.testing.assert_allclose(magnetic[written], expected[written], atol=1e-6)


def test_only_the_geophysical_columns_change_and_keep_their_direction(maps):
    frame = _frame()
    frame.iloc[40:45, frame.columns.get_indexer(list(GRAVITY_COLUMNS))] = np.nan
    frame.iloc[60:62, frame.columns.get_indexer(list(MAGNETIC_COLUMNS))] = np.nan

    out, _ = synthesize_geophysical(frame, "track", *maps, 42)

    untouched = [c for c in frame.columns if c not in GRAVITY_COLUMNS + MAGNETIC_COLUMNS]
    pd.testing.assert_frame_equal(out[untouched], frame[untouched])
    for columns in (GRAVITY_COLUMNS, MAGNETIC_COLUMNS):
        before = frame[list(columns)].to_numpy(float)
        after = out[list(columns)].to_numpy(float)
        np.testing.assert_array_equal(np.isnan(after[:-RATE_HZ]), np.isnan(before[:-RATE_HZ]))
        rows = np.all(np.isfinite(after), axis=1)
        unit_before = before[rows] / np.linalg.norm(before[rows], axis=1)[:, None]
        unit_after = after[rows] / np.linalg.norm(after[rows], axis=1)[:, None]
        np.testing.assert_allclose(unit_after, unit_before, atol=1e-12)


def test_a_gnss_gap_blanks_gravity_but_never_the_magnetometer(maps):
    """
    Across a 20 s gap the truth velocity is a guess, and gravity's Eötvös term would carry it.

    The magnetometer is kept: its direction drives the yaw update of every run, baselines
    included, and blanking it would change those too.
    """
    frame = _frame(gaps=((100.0, 120.0),))
    out, record = synthesize_geophysical(frame, "track", *maps, 42)

    seconds = np.arange(len(frame)) / RATE_HZ
    inside = (seconds > 100.0) & (seconds < 120.0)
    assert np.all(out.loc[inside, list(GRAVITY_COLUMNS)].isna())
    assert np.all(out.loc[~inside & (seconds < 1000.0), list(GRAVITY_COLUMNS)].notna())
    assert np.all(out[list(MAGNETIC_COLUMNS)].notna())
    assert record["gravity"]["rows_masked"] == inside.sum() + RATE_HZ - 1
    assert record["magnetic"]["rows_masked"] == 0
    assert MAX_TRUTH_GAP_S < 20.0


def test_draws_are_keyed_by_seed_and_trajectory_only(maps):
    frame = _frame()

    def reading(stem: str, seed: int) -> np.ndarray:
        out, _ = synthesize_geophysical(frame, stem, *maps, seed)
        return out[list(GRAVITY_COLUMNS + MAGNETIC_COLUMNS)].to_numpy(float)

    first = reading("a", 42)
    reading("b", 42)  # processing another trajectory in between must not matter
    np.testing.assert_array_equal(reading("a", 42), first)
    assert not np.allclose(reading("b", 42), first, equal_nan=True)
    assert not np.allclose(reading("a", 43), first, equal_nan=True)


def test_geostats_recovers_what_was_planted(maps, tmp_path):
    """
    `analyze geostats` reads back the turn-on bias and the white noise, per trajectory.

    Drift is switched off so the planted spread is the white noise alone. The per-trajectory
    statistics must also match the provenance record exactly: that record is what the real
    run's `geo_stats.csv` is checked against.
    """
    gravity_map, magnetic_map = maps
    for kind, source in (("gravity", gravity_map), ("magnetic", magnetic_map)):
        (tmp_path / f"track_{kind}.nc").write_bytes(source.read_bytes())

    out, record = synthesize_geophysical(
        _frame(), "track", gravity_map, magnetic_map, 7, WHITE_AND_TURN_ON
    )
    out.to_csv(tmp_path / "track.csv")
    stats, _ = analyse_trajectory(tmp_path / "track.csv")

    fixes = DURATION_S
    for entry in stats:
        model = SENSORS[entry.field]
        planted = record[entry.field]
        standard_error = model.sigma_row(RATE_HZ) / math.sqrt(fixes)
        assert entry.bias_median == pytest.approx(planted["turn_on"], abs=5 * standard_error)
        assert entry.sigma_robust == pytest.approx(model.sigma_row(RATE_HZ), rel=0.10)
        assert entry.bias_median == pytest.approx(planted["error_median"], abs=1e-6)
        assert entry.sigma_robust == pytest.approx(planted["error_sigma_robust"], abs=1e-6)
    assert {entry.field for entry in stats} == {"gravity", "magnetic"}


def test_provenance_records_the_models_and_the_draws(maps, tmp_path):
    _, record = synthesize_geophysical(_frame(), "track", *maps, 42)
    path = tmp_path / "synthetic.json"
    write_provenance([record], path, seed=42)

    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["seed"] == 42
    assert document["config_row_rate_hz"] == CONFIG_ROW_RATE_HZ
    assert document["sensors"]["gravity"]["config_values"] == SENSORS["gravity"].config_values()
    assert document["trajectories"][0]["trajectory"] == "track"
    assert document["trajectories"][0]["row_rate_hz"] == pytest.approx(RATE_HZ)


def test_a_missing_map_fails_before_the_csv_is_written(tmp_path, monkeypatch):
    """
    Otherwise the directory would hold a CSV still carrying the phone's own readings, and
    nothing in it would say so.
    """
    monkeypatch.setattr(preprocess, "plot_street_map", lambda *args, **kwargs: plt.figure())
    segment = preprocess.Segment("", _frame(), 0.0, DURATION_S, DURATION_S, "kept")
    args = Namespace(getmaps=False, synthetic=True, synthetic_seed=42)

    with pytest.raises(FileNotFoundError, match="--getmaps"):
        preprocess.write_segment(segment, "track", tmp_path, args)
    assert not (tmp_path / "track.csv").exists()


# ===========================================================================================
# The configs the simulator runs
# ===========================================================================================

GEO_CONFIGS = [f"{f}_{g}.toml" for f in ("ukf", "ekf", "rbpf") for g in ("grav", "mag", "both")]


@pytest.mark.parametrize("name", GEO_CONFIGS)
def test_the_configs_describe_the_synthetic_sensors(name):
    """
    Every geophysical recipe tells the filter what `--synthetic` planted.

    `core/tests/example_configs.rs` checks the nine agree with each other; this checks they
    agree with the model, including the random-walk rate that test does not cover. A channel a
    recipe does not enable must stay absent.
    """
    geophysical = tomllib.loads((CONF / name).read_text(encoding="utf-8"))["geophysical"]
    for model in SENSORS.values():
        enabled = f"{model.channel}_resolution" in geophysical
        for key, value in model.config_values().items():
            if enabled:
                assert geophysical[key] == pytest.approx(value, rel=1e-5, abs=1e-12), key
            else:
                assert key not in geophysical, key


def test_the_variation_is_the_drift_as_one_gauss_markov_process():
    """
    ``V`` keeps the drift's stationary variance and its short-term drive: the random walk
    `config_values` hands the Kalman filters is ``2 sigma_V^2 / tau_V``.
    """
    for model in SENSORS.values():
        variation = model.variation_values()
        sigma = variation[f"{model.channel}_variation_std"]
        tau = variation[f"{model.channel}_variation_time_constant_s"]
        assert sigma**2 == pytest.approx(sum(term.sigma**2 for term in model.drift))
        walk_rate = model.config_values()[f"{model.channel}_bias_process_noise_std"]
        assert math.sqrt(2.0 * sigma**2 / tau) == pytest.approx(walk_rate)
        assert min(t.tau_s for t in model.drift) <= tau <= max(t.tau_s for t in model.drift)


@pytest.mark.parametrize("name", [f"rbpf_{g}.toml" for g in ("grav", "mag", "both")])
def test_the_rbpf_configs_carry_the_synthetic_sensors_drift(name):
    """
    Each RBPF recipe gives its temporal variation ``V`` the drift `--synthetic` planted, and
    leaves a channel it does not enable to the defaults.
    """
    config = tomllib.loads((CONF / name).read_text(encoding="utf-8"))
    geophysical, particle_filter = config["geophysical"], config["particle_filter"]
    for model in SENSORS.values():
        enabled = f"{model.channel}_resolution" in geophysical
        for key, value in model.variation_values().items():
            if enabled:
                assert particle_filter[key] == pytest.approx(value, rel=1e-5), key
            else:
                assert key not in particle_filter, key


@pytest.mark.parametrize("variant", ["truth", "degraded", "grav", "mag", "both"])
def test_every_rbpf_run_is_allowed_to_finish(variant):
    """
    Both budgets are off, or the ratio one still applies and completion depends on load.

    The real arm's RBPF-both lost its longest trajectory to the 1,200 s cap, so the aided and
    unaided runs were scored over different sets.
    """
    config = tomllib.loads((CONF / f"rbpf_{variant}.toml").read_text(encoding="utf-8"))
    assert config["execution_limits"]["max_wall_clock_s"] == 0.0
    assert config["execution_limits"]["max_wall_clock_ratio"] == 0.0
