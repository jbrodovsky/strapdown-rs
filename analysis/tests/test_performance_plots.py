"""
Tests that the performance plots draw their series when the reference is sparse.

The reference for every error series is the recorded GNSS, which since the 10 Hz preprocessing
is NaN in nine rows out of ten. Matplotlib joins only *consecutive* finite points, so a series
plotted at the full row rate drew nothing -- the axes autoscaled to the data and the saved PNG
was an empty frame, with no error raised anywhere.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from analysis.plotting import plot_performance, plot_relative_performance

ROWS = 200
FIX_EVERY = 10


def _track(offset_deg: float = 0.0, fixes_only: bool = False) -> pd.DataFrame:
    """A 10 Hz track due north, optionally NaN everywhere but every tenth row."""
    index = pd.date_range("2025-07-11 13:33:16", periods=ROWS, freq="100ms", tz="UTC")
    frame = pd.DataFrame(
        {
            "latitude": 40.0 + 2e-5 * np.arange(ROWS) + offset_deg,
            "longitude": np.full(ROWS, -75.0),
            "altitude": np.full(ROWS, 100.0),
            "horizontalAccuracy": np.full(ROWS, 5.0),
            "verticalAccuracy": np.full(ROWS, 8.0),
        },
        index=index,
    )
    if fixes_only:
        frame.iloc[np.arange(ROWS) % FIX_EVERY != 0] = np.nan
    return frame


def _assert_drawn(line, expected_points: int) -> None:
    x, y = line.get_data()
    assert len(y) == expected_points, f"{line.get_label()}: {len(y)} points"
    assert np.all(np.isfinite(y)), f"{line.get_label()} carries NaN, so draws gaps"


def test_performance_series_are_drawn_against_a_sparse_reference(tmp_path):
    nav = _track(offset_deg=1e-4)
    gps = _track(fixes_only=True)

    fig = plot_performance(nav, gps, tmp_path / "performance.png")

    lines = fig.axes[0].get_lines()
    assert [line.get_label() for line in lines] == [
        "2D Haversine Error",
        "Altitude Error",
        "GPS Horizontal Accuracy",
        "GPS Vertical Accuracy",
    ]
    for line in lines:
        _assert_drawn(line, ROWS // FIX_EVERY)
    # The 1e-4 deg offset is ~11 m north at every fix.
    _, horizontal = lines[0].get_data()
    np.testing.assert_allclose(horizontal, 11.1, atol=0.1)


def test_relative_performance_is_drawn_against_a_sparse_reference(tmp_path):
    geo = _track(offset_deg=1e-4)
    degraded = _track(offset_deg=2e-4)
    reference = _track(fixes_only=True)

    fig = plot_relative_performance(geo, degraded, reference, tmp_path / "relative.png")

    (line,) = fig.axes[0].get_lines()
    _assert_drawn(line, ROWS // FIX_EVERY)
    _, difference = line.get_data()
    np.testing.assert_allclose(difference, -11.1, atol=0.1)


def test_performance_pairs_by_timestamp_when_the_run_skipped_a_record(tmp_path):
    """The sim writes no row for a record with neither IMU nor GNSS data.

    The output is then shorter than its input. Positional pairing either raised on the length
    mismatch -- dropping the trajectory from the plots and the summary -- or, had the lengths
    agreed, compared every later solution with the wrong epoch.
    """
    nav = _track(offset_deg=1e-4).drop(index=_track().index[35])
    gps = _track(fixes_only=True)

    fig = plot_performance(nav, gps, tmp_path / "performance.png")

    horizontal = fig.axes[0].get_lines()[0]
    _assert_drawn(horizontal, ROWS // FIX_EVERY)
    np.testing.assert_allclose(horizontal.get_data()[1], 11.1, atol=0.1)
