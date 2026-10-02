"""Quantify why the smartphone's gravity and magnetic readings carry no map information.

Writes ``tables/phone_numbers.tex`` (LaTeX macros quoted in Section 6.3). Statistics, each a
median over drives:

- the fraction of records whose gravity norm lies within 1 mGal of the drive's maximum, and
  the set of maxima (device constants);
- the correlation and regression slope of the gravity norm against the map, both averaged
  over 60 s blocks (a gravimeter would give a slope of one);
- the share of magnetic-intensity variance explained by vehicle course over ground, from a
  least-squares fit of ``a + b cos(course) + c sin(course)`` on records moving faster than
  5 m/s, the amplitude of that heading term, and the intensity's standard deviation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_figures as bf  # noqa: E402
import stats  # noqa: E402

BLOCK_S = 60.0
MIN_SPEED_MPS = 5.0


def gravity_statistics(stem: str, residuals: pd.DataFrame) -> dict[str, float]:
    """Pinning and map agreement of the gravity norm on one drive."""
    records = bf.read_input(stem, arm="phone")
    norm_mgal = (
        np.sqrt(records["grav_x"] ** 2 + records["grav_y"] ** 2 + records["grav_z"] ** 2) * 1e5
    )
    pinned = float(np.mean(norm_mgal.max() - norm_mgal < 1.0))
    gravity = residuals[
        (residuals["trajectory"] == stem) & (residuals["field"] == "gravity")
    ].copy()
    gravity["elapsed"] = gravity["elapsed_s"].round(1)
    joined = gravity.set_index("elapsed").join(norm_mgal.rename("norm"), how="inner")
    blocks = joined.groupby(joined["elapsed_s"] // BLOCK_S)[["norm", "map"]].mean()
    result = {"pinned": pinned, "maximum": float(norm_mgal.max() / 1e5)}
    if len(blocks) > 5 and blocks["norm"].std() > 0:
        result["corr"] = float(np.corrcoef(blocks["norm"], blocks["map"])[0, 1])
        result["slope"] = float(np.polyfit(blocks["map"], blocks["norm"], 1)[0])
    return result


def magnetic_statistics(stem: str, residuals: pd.DataFrame) -> dict[str, float]:
    """Heading dependence of the magnetic intensity on one drive."""
    records = bf.read_input(stem, arm="phone")
    magnetic = residuals[
        (residuals["trajectory"] == stem) & (residuals["field"] == "magnetic")
    ].copy()
    magnetic["elapsed"] = magnetic["elapsed_s"].round(1)
    moving = records.loc[records["speed"] > MIN_SPEED_MPS, ["bearing"]]
    joined = magnetic.set_index("elapsed").join(moving, how="inner")
    if len(joined) < 100:
        return {}
    course = np.radians(joined["bearing"].to_numpy())
    design = np.c_[np.ones(course.size), np.cos(course), np.sin(course)]
    intensity = joined["measured"].to_numpy()
    beta, *_ = np.linalg.lstsq(design, intensity, rcond=None)
    fitted_residual = intensity - design @ beta
    return {
        "r2": float(1.0 - np.var(fitted_residual) / np.var(intensity)),
        "amplitude": float(np.hypot(beta[1], beta[2])),
        "spread": float(np.std(intensity)),
        "map": float(np.std(joined["map"])),
    }


def main() -> None:
    """Compute the medians and write them as macros."""
    residuals = pd.read_csv(stats.DATA / "output_real2" / "geostats" / "geo_residuals.csv")
    gravity = pd.DataFrame([gravity_statistics(stem, residuals) for stem in bf.trajectories()])
    magnetic = pd.DataFrame(
        [m for stem in bf.trajectories() if (m := magnetic_statistics(stem, residuals))]
    )
    maxima = sorted({round(v, 5) for v in gravity["maximum"]})
    values = {
        "gravpinned": f"{100 * gravity['pinned'].median():.0f}",
        "gravcorr": f"{gravity['corr'].median():.3f}",
        "gravslope": f"{gravity['slope'].median():.2f}",
        "gravcorrn": str(int(gravity["corr"].notna().sum())),
        "magrsq": f"{100 * magnetic['r2'].median():.0f}",
        "magamplitude": f"{magnetic['amplitude'].median() / 1000:.1f}",
        "magspread": f"{magnetic['spread'].median() / 1000:.1f}",
        "magmapsd": f"{magnetic['map'].median():.0f}",
        "magn": str(len(magnetic)),
    }
    lines = [f"\\newcommand{{\\phone{name}}}{{{value}}}" for name, value in sorted(values.items())]
    out = Path(__file__).resolve().parent.parent / "tables" / "phone_numbers.tex"
    out.write_text("\n".join(lines) + "\n")
    print(values, "gravity maxima (m/s^2):", maxima)


if __name__ == "__main__":
    main()
