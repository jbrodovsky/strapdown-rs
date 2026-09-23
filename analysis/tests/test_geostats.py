"""
Tests for `analyze geostats`.

The central assertion is a round trip: plant a known bias and sigma into a synthetic
trajectory, run the characterisation, and require it back. That is the only check that
covers the whole chain -- anomaly model, map sampling, residual, variance decomposition --
and it is the chain whose output sets the filter's measurement noise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from analysis.geostats import (
    MGAL_PER_M_PER_S2,
    MICROTESLA_TO_NANOTESLA,
    AnomalyMap,
    analyse_trajectory,
    decimal_year,
    eotvos,
    gravity_anomaly_mgal,
    normal_gravity,
    observed_field_nt,
    plot_anomaly_differences,
    pool,
    self_check,
    wmm_total_field_nt,
    write_config_block,
)

# Planted truth. The tolerances below are sampling error on these, not slack.
GRAVITY_NOISE_MGAL = 7.0
MAGNETIC_NOISE_NT = 25.0
PER_TRAJECTORY_GRAVITY_OFFSET_MGAL = 12.0
PER_TRAJECTORY_MAGNETIC_OFFSET_NT = 900.0
TRAJECTORIES = 8
ROWS = 900


def _write_map(path, lats, lons, amplitude, offset):
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
    values = offset + amplitude * (
        np.sin((lat_grid - lats[0]) * 220.0) + np.cos((lon_grid - lons[0]) * 180.0)
    )
    xr.Dataset({"z": (("lat", "lon"), values)}, coords={"lat": lats, "lon": lons}).to_netcdf(path)


@pytest.fixture(scope="module")
def planted(tmp_path_factory):
    """Synthetic trajectories whose residual statistics are known exactly."""
    directory = tmp_path_factory.mktemp("planted")
    rng = np.random.default_rng(20260923)

    for index in range(TRAJECTORIES):
        stem = f"track_{index:02d}"
        lat0, lon0 = 40.05 + 0.01 * index, -75.95 - 0.01 * index
        speed = np.full(ROWS, 24.0)
        bearing = np.full(ROWS, 45.0)
        d_lat = (speed * np.cos(np.radians(bearing))) / 111320.0
        d_lon = (speed * np.sin(np.radians(bearing))) / (111320.0 * np.cos(np.radians(lat0)))
        latitude = lat0 + np.cumsum(d_lat)
        longitude = lon0 + np.cumsum(d_lon)
        altitude = np.full(ROWS, 100.0)
        time = pd.date_range("2025-03-01T00:00:00Z", periods=ROWS, freq="1s")

        lats = np.linspace(latitude.min() - 0.2, latitude.max() + 0.2, 240)
        lons = np.linspace(longitude.min() - 0.2, longitude.max() + 0.2, 240)
        _write_map(directory / f"{stem}_gravity.nc", lats, lons, 20.0, 0.0)
        _write_map(directory / f"{stem}_magnetic.nc", lats, lons, 120.0, 50.0)

        gravity_map = AnomalyMap.load(directory / f"{stem}_gravity.nc")
        magnetic_map = AnomalyMap.load(directory / f"{stem}_magnetic.nc")

        gravity_measured = (
            gravity_map.sample(latitude, longitude)
            + PER_TRAJECTORY_GRAVITY_OFFSET_MGAL * rng.standard_normal()
            + GRAVITY_NOISE_MGAL * rng.standard_normal(ROWS)
        )
        magnetic_measured = (
            magnetic_map.sample(latitude, longitude)
            + PER_TRAJECTORY_MAGNETIC_OFFSET_NT * rng.standard_normal()
            + MAGNETIC_NOISE_NT * rng.standard_normal(ROWS)
        )

        # Invert the measurement models, so the CSV carries raw sensor values and the tool
        # has to run the forward model itself to get back what was planted.
        north = speed * np.cos(np.radians(bearing))
        east = speed * np.sin(np.radians(bearing))
        gravity_magnitude = (
            gravity_measured / MGAL_PER_M_PER_S2
            + normal_gravity(latitude, 0.0)
            + eotvos(latitude, altitude, north, east)
        )
        reference = wmm_total_field_nt(latitude, longitude, altitude, decimal_year(pd.Series(time)))
        magnetic_magnitude_ut = (magnetic_measured + reference) / MICROTESLA_TO_NANOTESLA

        pd.DataFrame(
            {
                "time": time,
                "latitude": latitude,
                "longitude": longitude,
                "altitude": altitude,
                "speed": speed,
                "bearing": bearing,
                "grav_x": 0.0,
                "grav_y": 0.0,
                "grav_z": gravity_magnitude,
                "mag_x": 0.0,
                "mag_y": 0.0,
                "mag_z": magnetic_magnitude_ut,
            }
        ).to_csv(directory / f"{stem}.csv", index=False)

    return directory


@pytest.fixture(scope="module")
def characterised(planted):
    """Run the characterisation once for the whole module."""
    stats, frames = [], []
    for csv_path in sorted(planted.glob("*.csv")):
        trajectory_stats, residuals = analyse_trajectory(csv_path)
        stats.extend(trajectory_stats)
        frames.append(residuals)
    residuals = pd.concat(frames, ignore_index=True)
    return stats, residuals, pool(stats, residuals)


def test_self_check_passes():
    """The mirrored formulas still agree with the Rust they were copied from."""
    self_check()


def test_gravity_anomaly_is_milligal():
    """Mirrors `earth::tests::test_gravity_anomaly_is_milligal`."""
    gamma = float(normal_gravity(45.0, 0.0))
    base = float(gravity_anomaly_mgal(45.0, 1000.0, 0.0, 0.0, gamma))
    bumped = float(gravity_anomaly_mgal(45.0, 1000.0, 0.0, 0.0, gamma + 1e-5))
    assert bumped - base == pytest.approx(1.0, abs=1e-6)


def test_observed_field_converts_microtesla_to_nanotesla():
    """A 49.5 uT reading is 49,500 nT, not 49.5 -- the unit bug the Rust side had."""
    assert observed_field_nt(0.0, 0.0, 49.5) == pytest.approx(49_500.0)


def test_map_sampling_is_bilinear_and_bounded(planted):
    """`AnomalyMap.sample` interpolates inside the grid and returns NaN outside it."""
    anomaly_map = AnomalyMap.load(sorted(planted.glob("*_gravity.nc"))[0])
    inside_lat = float(np.mean(anomaly_map.lats))
    inside_lon = float(np.mean(anomaly_map.lons))

    inside = anomaly_map.sample(np.array([inside_lat]), np.array([inside_lon]))
    assert np.isfinite(inside[0])

    outside = anomaly_map.sample(np.array([inside_lat + 50.0]), np.array([inside_lon]))
    assert np.isnan(outside[0]), "off-map must be NaN, not an edge value"

    # A grid node reproduces its own value exactly.
    node = anomaly_map.sample(np.array([anomaly_map.lats[10]]), np.array([anomaly_map.lons[20]]))
    assert node[0] == pytest.approx(anomaly_map.values[10, 20], rel=1e-9)


@pytest.mark.parametrize(
    ("field", "planted_sigma"),
    [("gravity", GRAVITY_NOISE_MGAL), ("magnetic", MAGNETIC_NOISE_NT)],
)
def test_within_trajectory_sigma_recovers_the_planted_noise(characterised, field, planted_sigma):
    """
    The number that sets `*_noise_std` comes back within 10%.

    This is the headline assertion: within-trajectory sigma is what belongs in R, and it has
    to survive the full forward-and-back trip through the anomaly models.
    """
    _, _, pooled = characterised
    entry = next(p for p in pooled if p.field == field)
    assert entry.within_sigma == pytest.approx(planted_sigma, rel=0.10)


@pytest.mark.parametrize(
    ("field", "planted_offset"),
    [
        ("gravity", PER_TRAJECTORY_GRAVITY_OFFSET_MGAL),
        ("magnetic", PER_TRAJECTORY_MAGNETIC_OFFSET_NT),
    ],
)
def test_between_trajectory_sigma_recovers_the_planted_offset(characterised, field, planted_offset):
    """
    The number that sets `*_bias_init_std` comes back.

    Loose tolerance on purpose: this is the sample standard deviation of only
    `TRAJECTORIES` draws, so its own relative standard error is about
    1/sqrt(2*(n-1)) -- around 27% at n=8. A tighter bound here would be a flaky test, and
    the looseness is itself the finding: with a couple of dozen recordings this estimate is
    not precise, and the bias prior should be set generously.
    """
    _, _, pooled = characterised
    entry = next(p for p in pooled if p.field == field)
    assert entry.between_sigma == pytest.approx(planted_offset, rel=0.60)


def test_within_sigma_is_far_below_total_sigma(characterised):
    """
    The decomposition separates two things a single sigma conflates.

    With a per-trajectory offset larger than the per-sample noise, the pooled spread is
    dominated by the offset. Reporting that as measurement noise is what makes the magnetic
    channel look unusable when it is mis-modelled: the offset belongs in a bias state.
    """
    _, _, pooled = characterised
    magnetic = next(p for p in pooled if p.field == "magnetic")
    assert magnetic.within_sigma < 0.25 * magnetic.total_sigma


def test_decorrelation_uses_source_resolution_not_cell_size(characterised):
    """
    The recommended interval respects the source data, not the grid spacing.

    The fixture's grids are far finer than the real source data behind `earth_faa` and
    `earth_wdmam`, so a cell-size rule would recommend a tiny interval. Taking the coarser
    of cell and source resolution is what stops the tool endorsing the over-weighting it
    exists to find.
    """
    stats, _, _ = characterised
    for entry in stats:
        assert entry.decorrelation_m > max(entry.cell_north_m, entry.cell_east_m)
        assert entry.recommended_interval_s > 1.0


def test_outputs_are_written(characterised, tmp_path):
    """The config block and the figure both render."""
    _, residuals, pooled = characterised

    toml_path = tmp_path / "geo_stats.toml"
    write_config_block(pooled, toml_path)
    text = toml_path.read_text(encoding="utf-8")
    assert "[geophysical]" in text
    assert "gravity_noise_std" in text
    assert "magnetic_bias_init_std" in text
    assert "geo_interval_s" in text

    figure_path = tmp_path / "anomaly_differences.png"
    plot_anomaly_differences(residuals, pooled, figure_path)
    assert figure_path.stat().st_size > 10_000, "the figure should not be a blank canvas"
