"""Aggregate diagnostics of the unaided RBPF under degraded GNSS (Section 6.5).

Writes ``tables/rbpf_numbers.tex``. Pooling, over the dedicated-sensor arm's
``rbpf/degraded`` runs that completed (a drive stopped by the health limit has no output and
is not counted):

- **Crossing time**: per drive, the first elapsed time at which the horizontal error against
  the recorded GNSS fix exceeds 10 km. Drives that never cross are excluded from the median
  and counted separately.
- **Second-half medians**: per drive, over GNSS-bearing records later than half the drive's
  duration, the median of the reported horizontal 1-sigma and the median horizontal error;
  then the median of each across drives, and the median of their per-drive ratio.

The reported 1-sigma combines the latitude and longitude variances (rad^2) on a sphere of
radius 6,371,008.8 m, which is how ``build_figures.figure_rbpf`` draws it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_figures as bf  # noqa: E402
import stats  # noqa: E402

CROSSING_M = 10_000.0


def drive_diagnostics(stem: str) -> dict[str, float] | None:
    """Crossing time and second-half statistics of one drive, or None if it did not run."""
    if not (stats.arm_dir("dedicated") / "rbpf" / "degraded" / f"{stem}.csv").exists():
        return None
    records = bf.read_input(stem)
    output = bf.read_output("dedicated", "rbpf", "degraded", stem)
    error = bf.horizontal_error(output, records)
    latitude = np.radians(records["latitude"].ffill().reindex(output.index).ffill().to_numpy())
    sigma = pd.Series(
        np.sqrt(
            output["latitude_cov"].clip(lower=0) * bf.EARTH_RADIUS_M**2
            + output["longitude_cov"].clip(lower=0) * (bf.EARTH_RADIUS_M * np.cos(latitude)) ** 2
        ),
        index=output.index,
    )
    joined = pd.concat([error.rename("error"), sigma.rename("sigma")], axis=1, join="inner")
    joined = joined.dropna()
    second_half = joined[joined.index > joined.index.max() / 2.0]
    crossed = joined.index[joined["error"] > CROSSING_M]
    return {
        "crossing_min": float(crossed.min() / 60.0) if crossed.size else np.nan,
        "sigma_m": float(second_half["sigma"].median()),
        "error_m": float(second_half["error"].median()),
        "ratio": float((second_half["error"] / second_half["sigma"]).median()),
    }


def main() -> None:
    """Compute the pooled diagnostics and write them as macros."""
    rows = [d for stem in bf.trajectories() if (d := drive_diagnostics(stem)) is not None]
    frame = pd.DataFrame(rows)
    crossed = frame["crossing_min"].dropna()
    values = {
        "drives": str(len(frame)),
        "crossed": str(len(crossed)),
        "crossingmin": f"{crossed.median():.0f}",
        "sigmam": f"{frame['sigma_m'].median():.0f}",
        "errorkm": f"{frame['error_m'].median() / 1000.0:,.0f}".replace(",", "{,}"),
        "ratio": f"{frame['ratio'].median():,.0f}".replace(",", "{,}"),
    }
    lines = [f"\\newcommand{{\\rbpfdiag{key}}}{{{value}}}" for key, value in sorted(values.items())]
    out = Path(__file__).resolve().parent.parent / "tables" / "rbpf_numbers.tex"
    out.write_text("\n".join(lines) + "\n")
    print(values)


if __name__ == "__main__":
    main()
