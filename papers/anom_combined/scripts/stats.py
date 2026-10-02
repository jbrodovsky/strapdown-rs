"""Load the strapdown-rs experiment outputs and compute the paper's summary statistics.

Every number in the manuscript is computed here from the CSVs that `just postprocess` and
`just geoperf-all` write. Nothing is copied from the dissertation, whose tables disagree with
each other in places (see NOTES.md).

Data root: the `STRAPDOWN_DATA` environment variable, defaulting to the ``data/`` directory
at the root of the repository checkout that holds this script. Two arms are read:

- ``phone``: the smartphone's own gravity and magnetometer readings (``output_real2``)
- ``dedicated``: the same drives with the anomaly norms replaced by the ADXL355 / RM3100
  datasheet sensor model (``output``)

The inertial, barometric, heading and degraded-GNSS streams are identical in the two arms,
so the unaided runs coincide.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

# papers/anom_combined/scripts/stats.py -> repository root is three levels up.
DATA = Path(os.environ.get("STRAPDOWN_DATA", Path(__file__).resolve().parents[3] / "data"))
ARMS = {"phone": "output_real2", "dedicated": "output"}
FILTERS = ("ekf", "ukf", "rbpf")
CHANNELS = ("grav", "mag", "both")
SCENARIOS = ("truth", "degraded")
FILTER_LABEL = {"ekf": "EKF", "ukf": "UKF", "rbpf": "RBPF"}
CHANNEL_LABEL = {"grav": "Gravity", "mag": "Magnetic", "both": "Combined"}
SUMMARY_ROWS = ("mean", "median", "std")
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20261002


def arm_dir(arm: str) -> Path:
    """Return the output directory of an arm."""
    return DATA / ARMS[arm]


def performance(arm: str, filt: str, scenario: str) -> pd.DataFrame:
    """Per-trajectory error statistics of one run, without the summary rows."""
    path = arm_dir(arm) / filt / scenario / "performance" / "performance_summary.csv"
    frame = pd.read_csv(path, index_col=0)
    return frame.drop(index=[r for r in SUMMARY_ROWS if r in frame.index])


def detailed(arm: str, filt: str, channel: str) -> pd.DataFrame:
    """Aided-vs-own-unaided paired results of one filter and channel.

    The RBPF's are under ``analysis/rbpf`` (``analysis/ins`` scores it against the EKF).
    Only trajectories present in both the aided and the unaided run appear, which is the
    matched-set rule: a run that failed on either side is dropped, not scored.
    """
    base = arm_dir(arm) / filt / channel / "analysis"
    if filt == "rbpf":
        base = base / "rbpf"
    frame = pd.read_csv(base / f"{filt}_{channel}_detailed_results.csv")
    return frame.set_index("trajectory").sort_index()


@dataclass(frozen=True)
class PairedSummary:
    """Aided against unaided horizontal RMSE across the matched trajectories."""

    n: int
    unaided_median: float
    unaided_mean: float
    aided_median: float
    aided_mean: float
    ratio_median: float
    ratio_low: float
    ratio_high: float
    better: int
    worse: int
    p_value: float
    diff_median: float


def bootstrap_median_ci(values: np.ndarray, seed: int = BOOTSTRAP_SEED) -> tuple[float, float]:
    """Percentile 95% bootstrap interval of the median."""
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(BOOTSTRAP_RESAMPLES, values.size), replace=True)
    medians = np.median(draws, axis=1)
    low, high = np.percentile(medians, [2.5, 97.5])
    return float(low), float(high)


def paired(arm: str, filt: str, channel: str) -> PairedSummary:
    """Summarise one filter/channel/arm cell of the aiding experiment."""
    frame = detailed(arm, filt, channel)
    aided = frame["geo_rmse"].to_numpy()
    unaided = frame["baseline_rmse"].to_numpy()
    ratio = aided / unaided
    diff = aided - unaided
    low, high = bootstrap_median_ci(ratio)
    nonzero = diff[diff != 0.0]
    p_value = float(wilcoxon(nonzero).pvalue) if nonzero.size else 1.0
    return PairedSummary(
        n=int(frame.shape[0]),
        unaided_median=float(np.median(unaided)),
        unaided_mean=float(np.mean(unaided)),
        aided_median=float(np.median(aided)),
        aided_mean=float(np.mean(aided)),
        ratio_median=float(np.median(ratio)),
        ratio_low=low,
        ratio_high=high,
        better=int(np.sum(diff < 0.0)),
        worse=int(np.sum(diff > 0.0)),
        p_value=p_value,
        diff_median=float(np.median(diff)),
    )


def baseline(arm: str, filt: str, scenario: str) -> dict[str, float]:
    """Horizontal and vertical RMSE statistics of an unaided run across trajectories."""
    frame = performance(arm, filt, scenario)
    horizontal = frame["RMSE Horizontal Error (m)"]
    vertical = frame["RMSE Vertical Error (m)"]
    return {
        "n": int(frame.shape[0]),
        "h_median": float(horizontal.median()),
        "h_mean": float(horizontal.mean()),
        "h_std": float(horizontal.std()),
        "h_min": float(horizontal.min()),
        "h_max": float(horizontal.max()),
        "v_median": float(vertical.median()),
    }


def geostats(arm: str) -> dict[str, dict]:
    """Pooled anomaly error statistics of an arm, keyed by field."""
    path = arm_dir(arm) / "geostats" / "geo_stats_pooled.json"
    return {entry["field"]: entry for entry in json.loads(path.read_text())}


def dataset_summary() -> pd.DataFrame:
    """Per-trajectory distance (km) and duration (h)."""
    frame = pd.read_csv(DATA / "output" / "dataset_summary" / "dataset_summary.csv", index_col=0)
    return frame.drop(index=[r for r in SUMMARY_ROWS if r in frame.index])


def main() -> None:
    """Print every cell, for cross-checking against the manuscript."""
    for arm in ARMS:
        print(f"== {arm}")
        for filt in FILTERS:
            for scenario in SCENARIOS:
                b = baseline(arm, filt, scenario)
                print(
                    f"  {filt:4s} {scenario:8s} n={b['n']:2d} "
                    f"med={b['h_median']:.1f} mean={b['h_mean']:.1f} std={b['h_std']:.1f}"
                )
            for channel in CHANNELS:
                s = paired(arm, filt, channel)
                print(
                    f"  {filt:4s} {channel:8s} n={s.n:2d} ratio={s.ratio_median:.3f} "
                    f"[{s.ratio_low:.3f},{s.ratio_high:.3f}] B/W={s.better}/{s.worse} "
                    f"p={s.p_value:.3g} dmed={s.diff_median:.1f} "
                    f"aided med={s.aided_median:.1f} mean={s.aided_mean:.1f}"
                )


if __name__ == "__main__":
    main()
