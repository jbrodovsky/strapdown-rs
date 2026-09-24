"""Tests for the ``compare-filters`` subcommand.

The subcommand regressed once already: its parser survived a re-check-in of the package but
its worker and its ``elif`` branch in :func:`analysis.main` did not, so ``analyze
compare-filters`` fell through to the ``else`` branch, printed the help text and exited 0
without writing anything. The tests here drive the real CLI entry point rather than calling
the worker directly, so that the dispatch wiring is covered and not just the arithmetic.
"""

import sys
from pathlib import Path

import pytest
from pandas import read_csv

from analysis import main

# One degree of latitude is very nearly this many meters on the WGS84 ellipsoid, which is what
# the haversine comparison reduces to for a purely north-south offset.
METERS_PER_DEGREE_LATITUDE = 111_194.9

TRAJECTORIES = ("traj_A", "traj_B")


def write_trajectory(
    path: Path, *, rows: int, latitude_offset_deg: float, start_hour: int = 0
) -> None:
    """Write a minimal navigation-solution CSV in the shape the analysis package reads.

    Parameters
    ----------
    path : Path
        Destination CSV path.
    rows : int
        Number of samples to write.
    latitude_offset_deg : float
        Constant latitude offset added to the baseline track, in degrees. The baseline is a
        straight northbound leg, so this offset is the horizontal error the comparison should
        recover.
    start_hour : int, optional
        Hour of the first timestamp, used to build a non-overlapping index for the
        unalignable-trajectory test.
    """
    lines = ["timestamp,latitude,longitude,altitude"]
    for i in range(rows):
        timestamp = f"2026-04-01T{start_hour:02d}:00:{i:02d}+00:00"
        latitude = 40.0 + i * 1e-4 + latitude_offset_deg
        lines.append(f"{timestamp},{latitude:.10f},-75.0,10.0")
    path.write_text("\n".join(lines) + "\n")


@pytest.fixture
def comparison_tree(tmp_path: Path) -> Path:
    """Build a reference directory plus two filter output directories of differing accuracy.

    The reference carries one more row than each solution, matching what ``strapdown-sim``
    emits: the first record seeds the filter rather than producing an estimate.
    """
    reference = tmp_path / "input"
    accurate = tmp_path / "accurate"
    coarse = tmp_path / "coarse"
    for directory in (reference, accurate, coarse):
        directory.mkdir()

    for name in TRAJECTORIES:
        write_trajectory(reference / f"{name}.csv", rows=6, latitude_offset_deg=0.0)
        # The solutions start one sample later, so their track lines up with truth[1:].
        write_solution(accurate / f"{name}.csv", offset_deg=1e-6)
        write_solution(coarse / f"{name}.csv", offset_deg=1e-5)
    return tmp_path


def write_solution(path: Path, *, offset_deg: float) -> None:
    """Write a 5-row solution offset north of the 6-row reference track's tail."""
    lines = ["timestamp,latitude,longitude,altitude"]
    for i in range(5):
        timestamp = f"2026-04-01T00:00:{i + 1:02d}+00:00"
        latitude = 40.0 + (i + 1) * 1e-4 + offset_deg
        lines.append(f"{timestamp},{latitude:.10f},-75.0,10.0")
    path.write_text("\n".join(lines) + "\n")


def run_compare_filters(monkeypatch: pytest.MonkeyPatch, *argv: str) -> None:
    """Invoke the package's console entry point with the given arguments."""
    monkeypatch.setattr(sys, "argv", ["analyze", "compare-filters", *argv])
    main()


def test_compare_filters_writes_a_comparison_csv(
    comparison_tree: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The subcommand dispatches and produces output rather than printing the help text."""
    output = comparison_tree / "out"
    run_compare_filters(
        monkeypatch,
        "-i",
        str(comparison_tree / "accurate"),
        str(comparison_tree / "coarse"),
        "-l",
        "UKF",
        "EKF",
        "-r",
        str(comparison_tree / "input"),
        "-o",
        str(output),
        "--geo-type",
        "smoke",
    )

    written = output / "filter_comparison_smoke.csv"
    assert written.exists(), "compare-filters produced no CSV -- the dispatch branch is missing"

    results = read_csv(written)
    assert list(results.columns) == [
        "filter",
        "trajectory",
        "rmse",
        "mean",
        "median",
        "std",
        "max",
        "min",
    ]
    # One row per filter per trajectory.
    assert len(results) == 2 * len(TRAJECTORIES)
    assert set(results["filter"]) == {"UKF", "EKF"}
    assert set(results["trajectory"]) == set(TRAJECTORIES)


def test_compare_filters_scores_each_filter_against_truth(
    comparison_tree: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Per-filter RMSE recovers the injected offset, so the filters can be ranked."""
    output = comparison_tree / "out"
    run_compare_filters(
        monkeypatch,
        "-i",
        str(comparison_tree / "accurate"),
        str(comparison_tree / "coarse"),
        "-l",
        "UKF",
        "EKF",
        "-r",
        str(comparison_tree / "input"),
        "-o",
        str(output),
    )

    results = read_csv(output / "filter_comparison_comparison.csv").set_index(
        ["filter", "trajectory"]
    )
    for name in TRAJECTORIES:
        accurate_rmse = results.loc[("UKF", name), "rmse"]
        coarse_rmse = results.loc[("EKF", name), "rmse"]
        assert accurate_rmse == pytest.approx(1e-6 * METERS_PER_DEGREE_LATITUDE, rel=1e-3)
        assert coarse_rmse == pytest.approx(1e-5 * METERS_PER_DEGREE_LATITUDE, rel=1e-3)
        assert accurate_rmse < coarse_rmse


def test_compare_filters_rejects_mismatched_labels(
    comparison_tree: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Fewer labels than input directories is reported, not zipped silently."""
    output = comparison_tree / "out"
    run_compare_filters(
        monkeypatch,
        "-i",
        str(comparison_tree / "accurate"),
        str(comparison_tree / "coarse"),
        "-l",
        "UKF",
        "-r",
        str(comparison_tree / "input"),
        "-o",
        str(output),
    )

    assert "must have the same count" in capsys.readouterr().out
    assert not list(output.glob("*.csv"))


def test_compare_filters_skips_unalignable_trajectory(
    comparison_tree: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A solution that shares no index with its reference is skipped, not fatal."""
    # A different length forces index alignment, and a different hour makes it fail.
    write_trajectory(
        comparison_tree / "accurate" / "traj_A.csv",
        rows=3,
        latitude_offset_deg=1e-6,
        start_hour=7,
    )
    output = comparison_tree / "out"
    run_compare_filters(
        monkeypatch,
        "-i",
        str(comparison_tree / "accurate"),
        "-l",
        "UKF",
        "-r",
        str(comparison_tree / "input"),
        "-o",
        str(output),
    )

    assert "Could not align traj_A.csv" in capsys.readouterr().out
    results = read_csv(output / "filter_comparison_comparison.csv")
    # The other trajectory is still scored.
    assert list(results["trajectory"]) == ["traj_B"]


def test_compare_filters_pairs_by_timestamp_when_the_run_skipped_a_record(
    comparison_tree: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A solution that begins at the seed record and skips one later record is scored per epoch.

    That is one row short of the reference -- the same length as ``truth[1:]`` -- which the
    scorer used to take as "already aligned" and pair by position, comparing each solution with
    the next epoch's truth up to the skipped record.
    """
    lines = ["timestamp,latitude,longitude,altitude"]
    for i in (0, 1, 2, 4, 5):
        latitude = 40.0 + i * 1e-4 + 1e-6
        lines.append(f"2026-04-01T00:00:{i:02d}+00:00,{latitude:.10f},-75.0,10.0")
    (comparison_tree / "accurate" / "traj_A.csv").write_text("\n".join(lines) + "\n")
    output = comparison_tree / "out"
    run_compare_filters(
        monkeypatch,
        "-i",
        str(comparison_tree / "accurate"),
        "-l",
        "UKF",
        "-r",
        str(comparison_tree / "input"),
        "-o",
        str(output),
    )

    results = read_csv(output / "filter_comparison_comparison.csv").set_index("trajectory")
    assert results.loc["traj_A", "max"] == pytest.approx(
        1e-6 * METERS_PER_DEGREE_LATITUDE, rel=1e-3
    )
