"""Write the manuscript's data-derived LaTeX tables into ``tables/``.

Run from anywhere: ``uv run --with scipy python papers/anom_combined/scripts/build_tables.py``.
Output is deterministic (fixed bootstrap seed), so a rerun on unchanged data is byte-identical.
Settings tables that restate configuration (degradation, filter tuning) are written by hand
in the section files, because they quote ``conf/*.toml`` rather than compute anything.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stats  # noqa: E402

TABLES = Path(__file__).resolve().parent.parent / "tables"


def thousands(value: float, digits: int = 1) -> str:
    """Format with a LaTeX thin-space-safe thousands separator."""
    text = f"{value:,.{digits}f}"
    return text.replace(",", "{,}")


def metres_or_km(value: float) -> str:
    """Metres below 10 km, kilometres above, so a diverged run reads as one."""
    if value >= 10_000.0:
        return f"{thousands(value / 1000.0)}~km"
    return thousands(value)


def p_text(p: float) -> str:
    """Two significant figures, or a bound below 10^-3."""
    if p < 1e-3:
        return r"\(<10^{-3}\)"
    return f"{p:.2g}"


def snr_text(value: float) -> str:
    """Two decimals below ten, one above."""
    return f"{value:.2f}" if value < 10.0 else f"{value:.1f}"


def write(name: str, body: str) -> None:
    """Write one table fragment."""
    TABLES.mkdir(exist_ok=True)
    (TABLES / name).write_text(body)


def dataset_table() -> None:
    """Summary statistics of the 27 MEMS-Nav trajectories."""
    frame = stats.dataset_summary()
    km = frame["Distance Traversed (km)"]
    hours = frame["Duration (h)"]
    rows = [
        ("Trajectories", f"{len(frame)}", ""),
        ("Total", f"{thousands(km.sum(), 0)}~km", f"{hours.sum():.1f}~h"),
        ("Median", f"{km.median():.1f}~km", f"{hours.median():.2f}~h"),
        ("Mean", f"{km.mean():.1f}~km", f"{hours.mean():.2f}~h"),
        ("Range", f"{km.min():.1f}--{km.max():.1f}~km", f"{hours.min():.2f}--{hours.max():.2f}~h"),
    ]
    lines = [r"\begin{tabular}{lrr}", r"\toprule", r" & Distance & Duration \\", r"\midrule"]
    lines += [f"{a} & {b} & {c} \\\\" for a, b, c in rows]
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    write("dataset.tex", "\n".join(lines))


def baseline_table() -> None:
    """Unaided horizontal RMSE with full and degraded GNSS, per filter."""
    lines = [
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Filter & GNSS & \(n\) & Median (m) & Mean (m) & Max (m) & Vert.\ median (m) \\",
        r"\midrule",
    ]
    for filt in stats.FILTERS:
        for scenario, label in (("truth", "Full"), ("degraded", "Degraded")):
            b = stats.baseline("dedicated", filt, scenario)
            name = stats.FILTER_LABEL[filt] if scenario == "truth" else ""
            lines.append(
                f"{name} & {label} & {b['n']} & {metres_or_km(b['h_median'])} & "
                f"{metres_or_km(b['h_mean'])} & {metres_or_km(b['h_max'])} & "
                f"{thousands(b['v_median'])} \\\\"
            )
        if filt != stats.FILTERS[-1]:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    write("baselines.tex", "\n".join(lines))


def signal_sigma(arm: str, field: str) -> float:
    """Median over trajectories of the along-track map standard deviation."""
    frame = pd.read_csv(stats.arm_dir(arm) / "geostats" / "geo_stats.csv")
    return float(frame.loc[frame["field"] == field, "signal_sigma"].median())


def error_model_table() -> None:
    """Measured (phone) and datasheet (dedicated) anomaly error models."""
    synthetic = json.loads((stats.DATA / "input" / "synthetic.json").read_text())["sensors"]
    lines = [
        r"\begin{tabular}{llrrrrl}",
        r"\toprule",
        r"Sensor & Field & Bias seed & \(\sqrt{R}\) & Prior \(\sigma\) & Signal \(\sigma\) & SNR \\",
        r"\midrule",
    ]
    for arm, label in (("phone", "Smartphone"), ("dedicated", "Dedicated")):
        pooled = stats.geostats(arm)
        for field, unit in (("gravity", "mGal"), ("magnetic", "nT")):
            entry = pooled[field]
            if arm == "phone":
                seed, noise, prior = (
                    entry["bias_median"],
                    entry["within_sigma"],
                    entry["between_sigma"],
                )
            else:
                values = synthetic[field]["config_values"]
                seed = values[f"{field}_bias"]
                noise = values[f"{field}_noise_std"]
                prior = values[f"{field}_bias_init_std"]
            part = (
                {"gravity": "ADXL355", "magnetic": "RM3100"}[field] if arm == "dedicated" else label
            )
            snr = f"{snr_text(entry['snr_median'])} ({snr_text(entry['snr_best'])})"
            lines.append(
                f"{part} & {field.capitalize()} ({unit}) & {thousands(seed)} & {thousands(noise)} & "
                f"{thousands(prior)} & {signal_sigma(arm, field):.1f} & {snr} \\\\"
            )
        if arm == "phone":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    write("error_model.tex", "\n".join(lines))


def aiding_cell(summary: stats.PairedSummary) -> str:
    """Ratio [CI], B/W and p for one arm."""
    return (
        f"{summary.ratio_median:.3f} [{summary.ratio_low:.3f}, {summary.ratio_high:.3f}] & "
        f"{summary.better}/{summary.worse} & {p_text(summary.p_value)}"
    )


def aiding_table() -> None:
    """Paired aided/unaided results for the Kalman filters, both arms side by side."""
    lines = [
        r"\begin{tabular}{llccc@{\hspace{1.2em}}ccc}",
        r"\toprule",
        r" & & \multicolumn{3}{c}{Smartphone observations} & \multicolumn{3}{c}{Dedicated sensors} \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
        r"Filter & Aiding & Ratio [95\% CI] & B/W & \(p\) & Ratio [95\% CI] & B/W & \(p\) \\",
        r"\midrule",
    ]
    for filt in ("ekf", "ukf"):
        for channel in stats.CHANNELS:
            phone = stats.paired("phone", filt, channel)
            dedicated = stats.paired("dedicated", filt, channel)
            name = stats.FILTER_LABEL[filt] if channel == "grav" else ""
            lines.append(
                f"{name} & {stats.CHANNEL_LABEL[channel]} & {aiding_cell(phone)} & "
                f"{aiding_cell(dedicated)} \\\\"
            )
        if filt == "ekf":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    write("aiding.tex", "\n".join(lines))


def rbpf_table() -> None:
    """RBPF paired results, reported for completeness and labelled as non-interpretable."""
    lines = [
        r"\begin{tabular}{llrrcc}",
        r"\toprule",
        r"Observations & Aiding & \(n\) & Unaided median & Aided median & B/W \\",
        r"\midrule",
    ]
    for arm, label in (("phone", "Smartphone"), ("dedicated", "Dedicated")):
        for channel in stats.CHANNELS:
            s = stats.paired(arm, "rbpf", channel)
            name = label if channel == "grav" else ""
            lines.append(
                f"{name} & {stats.CHANNEL_LABEL[channel]} & {s.n} & {metres_or_km(s.unaided_median)} & "
                f"{metres_or_km(s.aided_median)} & {s.better}/{s.worse} \\\\"
            )
        if arm == "phone":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    write("rbpf.tex", "\n".join(lines))


def macros() -> None:
    """Numbers quoted in running text, as macros, so prose cannot drift from the tables."""
    values: dict[str, str] = {}
    for filt in stats.FILTERS:
        for scenario in stats.SCENARIOS:
            b = stats.baseline("dedicated", filt, scenario)
            key = f"{filt}{scenario}".replace("truth", "full")
            values[f"{key}median"] = metres_or_km(b["h_median"])
            values[f"{key}vmedian"] = thousands(b["v_median"])
            values[f"{key}n"] = str(b["n"])
    for arm in stats.ARMS:
        for filt in stats.FILTERS:
            for channel in stats.CHANNELS:
                s = stats.paired(arm, filt, channel)
                key = f"{arm}{filt}{channel}"
                values[f"{key}ratio"] = f"{s.ratio_median:.3f}"
                values[f"{key}better"] = str(s.better)
                values[f"{key}worse"] = str(s.worse)
                values[f"{key}n"] = str(s.n)
                values[f"{key}dmed"] = thousands(abs(s.diff_median))
                values[f"{key}p"] = p_text(s.p_value)
    dataset = stats.dataset_summary()
    values["datasetkm"] = thousands(dataset["Distance Traversed (km)"].sum(), 0)
    values["datasethours"] = f"{dataset['Duration (h)'].sum():.1f}"
    lines = [f"\\newcommand{{\\num{name}}}{{{value}}}" for name, value in sorted(values.items())]
    write("numbers.tex", "\n".join(lines) + "\n")


def main() -> None:
    """Write every table."""
    np.seterr(all="raise")
    dataset_table()
    baseline_table()
    error_model_table()
    aiding_table()
    rbpf_table()
    macros()
    print(f"wrote {sorted(p.name for p in TABLES.glob('*.tex'))}")


if __name__ == "__main__":
    main()
