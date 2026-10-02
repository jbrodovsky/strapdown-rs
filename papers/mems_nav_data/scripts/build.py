"""Build every table, figure and in-text number of the MEMS-Nav data article.

Run from anywhere::

    /home/james/Code/strapdown-rs/.venv/bin/python papers/mems_nav_data/scripts/build.py

Inputs (all read-only) live under ``STRAPDOWN_DATA`` (default: the main checkout's ``data/``):

- ``raw/<recording>/``: the Sensor Logger exports, one directory per recording
- ``input_real/*.csv``: the 10 Hz trajectories with the phones' own gravity and magnetometer
  channels -- the files this article describes. (``input/`` carries simulated
  gravimeter/magnetometer norms and is *not* the released data.)
- ``input/segments.json``: which recording each trajectory was cut from
- ``output/dataset_summary/dataset_summary.csv``: per-trajectory distance and duration
- ``output_real2/{ekf,ukf,rbpf}/{truth,degraded}/performance/``: benchmark scores on the
  phone data
- ``output_real2/geostats/geo_stats_pooled.json``: anomaly signal-to-noise ratios

Outputs: ``tables/*.tex``, ``figures/*.pdf`` and ``tables/numbers.tex`` (one ``\\newcommand`` per
number quoted in the prose). Nothing is random; repeated runs give identical tables.

Helpers are reused, unmodified, from ``papers/anom_combined/scripts`` (``stats.py`` for the
data root, the performance loader and the dataset summary; ``build_figures.py`` for the
matplotlib style and the great-circle distance).
"""

from __future__ import annotations

import datetime
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import pandas as pd

sys.dont_write_bytecode = True  # leave no __pycache__ beside the borrowed anom_combined helpers

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
sys.path.insert(0, str(PAPER.parent / "anom_combined" / "scripts"))
import build_figures as bf  # noqa: E402  (sets the shared matplotlib style on import)
import stats  # noqa: E402

DATA = stats.DATA
RAW = DATA / "raw"
RELEASE = DATA / "input_real"
SEGMENTS = DATA / "input" / "segments.json"
BENCHMARK_ARM = "phone"  # output_real2: run on the phone channels that are released
TABLES = PAPER / "tables"
FIGURES = PAPER / "figures"
GNSS_COLUMNS = ("horizontalAccuracy", "verticalAccuracy", "speedAccuracy")
MGAL_PER_MPS2 = 1e5
PINNED_TOLERANCE_MGAL = 1.0
RATE_PROBE_ROWS = 200_000

# Device families, in a fixed order that also fixes their colour (validated categorical
# slots 1-5 of the reference palette; every series also carries its own marker/line style).
FAMILY_OF_MODEL = {
    "Pixel 6a": "Pixel 6a",
    "Pixel 9 Pro": "Pixel 9 Pro / Pro XL",
    "Pixel 9 Pro XL": "Pixel 9 Pro / Pro XL",
    "SM-S921U": "Samsung SM-S921U",
    "SM-A146U": "Samsung SM-A146U",
    "iPhone 12 mini": "iPhone 12/13 mini",
    "iPhone 13 mini": "iPhone 12/13 mini",
}
FAMILIES = (
    "Pixel 6a",
    "Pixel 9 Pro / Pro XL",
    "Samsung SM-S921U",
    "Samsung SM-A146U",
    "iPhone 12/13 mini",
)
FAMILY_COLOR = dict(
    zip(FAMILIES, ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"), strict=True)
)
FAMILY_STYLE = dict(zip(FAMILIES, ("-", "--", "-.", ":", (0, (5, 1, 1, 1))), strict=True))


# --------------------------------------------------------------------------------------------
# Formatting


def thousands(value: float, digits: int = 0) -> str:
    """Format with a LaTeX thousands separator."""
    return f"{value:,.{digits}f}".replace(",", "{,}")


def thousands_plain(value: float) -> str:
    """Format an integer count with a comma separator (for figure text)."""
    return f"{value:,.0f}"


def tex_escape(text: str) -> str:
    """Escape the characters that appear in recording stems and model names."""
    return text.replace("_", r"\_").replace("&", r"\&")


def write(name: str, body: str) -> None:
    """Write one generated LaTeX fragment."""
    TABLES.mkdir(exist_ok=True)
    (TABLES / name).write_text(body)
    print(f"wrote tables/{name}")


def save(fig: plt.Figure, name: str) -> None:
    """Write one vector figure into this paper's directory (not anom_combined's)."""
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / f"{name}.pdf", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    print(f"wrote figures/{name}.pdf")


# --------------------------------------------------------------------------------------------
# Loading


def segments() -> pd.DataFrame:
    """Every preprocessing outcome, one row per recording or segment."""
    frame = pd.DataFrame(json.loads(SEGMENTS.read_text()))
    frame["stem"] = frame["output"].str.removesuffix(".csv")
    return frame


def metadata(recording: str) -> dict[str, str]:
    """Device and app fields of one recording's Metadata.csv."""
    row = pd.read_csv(RAW / recording / "Metadata.csv", dtype=str).iloc[0]
    model = row["device name"]
    return {
        "recording": recording,
        "model": model,
        "family": FAMILY_OF_MODEL[model],
        "platform": {"android": "Android", "ios": "iOS"}[row["platform"]],
        "app": row["appVersion"],
        "device_id": row["device id"],
        "utc_named": datetime.datetime.fromtimestamp(
            int(row["recording epoch time"]) / 1000, datetime.UTC
        ).strftime("%Y-%m-%d_%H-%M-%S")
        == row["recording time"],
    }


def scan_file(path: Path) -> dict:
    """Row count, span and median sampling interval of one raw sensor CSV."""
    rows = 0
    with path.open("rb") as handle:
        handle.readline()
        while chunk := handle.read(1 << 24):
            rows += chunk.count(b"\n")
    record = {
        "recording": path.parent.name,
        "file": path.stem,
        "rows": rows,
        "bytes": path.stat().st_size,
        "span_s": np.nan,
        "median_hz": np.nan,
    }
    if rows > 1:
        probe = pd.read_csv(path, usecols=["seconds_elapsed"], nrows=RATE_PROBE_ROWS)[
            "seconds_elapsed"
        ]
        steps = probe.diff().dropna()
        steps = steps[steps > 0]
        record["median_hz"] = 1.0 / float(steps.median())
        last = pd.read_csv(path, usecols=["seconds_elapsed"], skiprows=range(1, max(rows - 1, 1)))
        record["span_s"] = float(last["seconds_elapsed"].iloc[-1] - probe.iloc[0])
    return record


def raw_inventory() -> pd.DataFrame:
    """One row per raw sensor file of every recording."""
    files = sorted(
        p for p in RAW.glob("*/*.csv") if p.name not in ("Metadata.csv", "Annotation.csv")
    )
    with ThreadPoolExecutor(max_workers=min(12, os.cpu_count() or 1)) as pool:
        return pd.DataFrame(list(pool.map(scan_file, files)))


def trajectory_table(kept: pd.DataFrame) -> pd.DataFrame:
    """Per-trajectory statistics of the released 10 Hz files."""
    summary = stats.dataset_summary()
    rows = []
    for _, seg in kept.sort_values("stem").iterrows():
        frame = pd.read_csv(RELEASE / f"{seg['stem']}.csv")
        gnss = frame.dropna(subset=["latitude", "longitude"])
        time = pd.to_datetime(frame["time"], utc=True, format="mixed")
        span = (time.iloc[-1] - time.iloc[0]).total_seconds()
        norm = np.sqrt(frame["grav_x"] ** 2 + frame["grav_y"] ** 2 + frame["grav_z"] ** 2).dropna()
        meta = metadata(seg["source"])
        rows.append(
            {
                "stem": seg["stem"],
                "source": seg["source"],
                **meta,
                "rows": len(frame),
                "fixes": len(gnss),
                "span_s": span,
                "fix_rate_hz": len(gnss) / span,
                "km": float(summary.loc[seg["stem"], "Distance Traversed (km)"]),
                "hours": float(summary.loc[seg["stem"], "Duration (h)"]),
                "h_acc": gnss["horizontalAccuracy"].median(),
                "h_acc_p95": gnss["horizontalAccuracy"].quantile(0.95),
                "v_acc": gnss["verticalAccuracy"].median(),
                "s_acc": gnss["speedAccuracy"].median(),
                "sentinels": int((gnss["horizontalAccuracy"] >= 999.0).sum()),
                "speed_kmh": 3.6 * gnss["speed"].median(),
                "lat_min": gnss["latitude"].min(),
                "lat_max": gnss["latitude"].max(),
                "lon_min": gnss["longitude"].min(),
                "lon_max": gnss["longitude"].max(),
                "alt_min": gnss["altitude"].min(),
                "alt_max": gnss["altitude"].max(),
                "baro_rows": int(frame["pressure"].notna().sum()),
                "grav_10hz_pinned": float(
                    np.mean((norm.max() - norm) * MGAL_PER_MPS2 < PINNED_TOLERANCE_MGAL)
                ),
            }
        )
    return pd.DataFrame(rows).set_index("stem")


def raw_gravity(recording: str) -> np.ndarray:
    """Norm of the OS gravity vector at the raw rate, in m/s^2."""
    frame = pd.read_csv(RAW / recording / "Gravity.csv", usecols=["x", "y", "z"], dtype="float64")
    return np.sqrt(frame["x"] ** 2 + frame["y"] ** 2 + frame["z"] ** 2).to_numpy()


def hard_iron_offset(recording: str) -> float | None:
    """Magnitude (uT) of the OS hard-iron correction: median uncalibrated less median calibrated."""
    uncalibrated = RAW / recording / "MagnetometerUncalibrated.csv"
    if not uncalibrated.exists() or uncalibrated.stat().st_size == 0:
        return None
    raw = pd.read_csv(uncalibrated, usecols=["x", "y", "z"]).median()
    calibrated = pd.read_csv(RAW / recording / "Magnetometer.csv", usecols=["x", "y", "z"]).median()
    return float(np.linalg.norm((raw - calibrated).to_numpy()))


def calibrated_intensity(stem: str) -> float:
    """Median OS-calibrated magnetic intensity (uT) of one released trajectory."""
    frame = pd.read_csv(RELEASE / f"{stem}.csv", usecols=["mag_x", "mag_y", "mag_z"]).dropna()
    return float(np.sqrt((frame**2).sum(axis=1)).median())


ALTITUDE_CELL_DEG = 0.002
ALTITUDE_MIN_SPEED_MPS = 15.0


def altitude_offsets(traj: pd.DataFrame) -> dict[str, tuple[int, float, float, float]]:
    """Median GNSS altitude of each family less the Pixel 9 Pro's, on shared road cells.

    Fixes above 15 m/s are binned into 0.002-degree cells; for every cell that both the
    family and the Pixel 9 Pro / Pro XL family visited, the difference of their median
    altitudes is taken. Returns (cells, median, lower quartile, upper quartile) per family.
    """
    frames = []
    for stem, row in traj.iterrows():
        frame = pd.read_csv(
            RELEASE / f"{stem}.csv", usecols=["latitude", "longitude", "altitude", "speed"]
        ).dropna()
        frame = frame[frame["speed"] > ALTITUDE_MIN_SPEED_MPS]
        frame["cell_lat"] = np.round(frame["latitude"] / ALTITUDE_CELL_DEG).astype(int)
        frame["cell_lon"] = np.round(frame["longitude"] / ALTITUDE_CELL_DEG).astype(int)
        frame["family"] = row["family"]
        frames.append(frame)
    cells = (
        pd.concat(frames).groupby(["cell_lat", "cell_lon", "family"])["altitude"].median().unstack()
    )
    reference = "Pixel 9 Pro / Pro XL"
    result = {}
    for family in FAMILIES:
        if family == reference or family not in cells:
            continue
        diff = (cells[family] - cells[reference]).dropna()
        if diff.size:
            result[family] = (
                int(diff.size),
                float(diff.median()),
                float(diff.quantile(0.25)),
                float(diff.quantile(0.75)),
            )
    return result


# --------------------------------------------------------------------------------------------
# Tables


def table_devices(traj: pd.DataFrame, recordings: pd.DataFrame) -> None:
    """Device inventory: recordings, released trajectories, hours and distance."""
    lines = [
        r"\begin{tabular}{llrrrrl}",
        r"\toprule",
        r"Device (as logged) & OS & Rec. & Traj. & Hours & km & GNSS file \\",
        r"\midrule",
    ]
    for model, group in recordings.groupby("model", sort=False):
        sub = traj[traj["model"] == model]
        gnss_file = (
            "LocationGps"
            if group["has_location_gps"].all()
            else ("Location" if not group["has_location_gps"].any() else "mixed")
        )
        lines.append(
            f"{tex_escape(model)} & {group['platform'].iloc[0]} & {len(group)} & {len(sub)} & "
            f"{sub['hours'].sum():.1f} & {thousands(sub['km'].sum())} & \\texttt{{{gnss_file}}} \\\\"
        )
    lines += [
        r"\midrule",
        f"Total & & {len(recordings)} & {len(traj)} & {traj['hours'].sum():.1f} & "
        f"{thousands(traj['km'].sum())} & \\\\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    write("devices.tex", "\n".join(lines) + "\n")


def rate_text(
    inventory: pd.DataFrame,
    used: list[str],
    files: tuple[str, ...],
    family: str,
    families: dict[str, str],
) -> str:
    """Median of the per-recording median sampling rates of a channel, for one family."""
    sub = inventory[inventory["recording"].isin([r for r in used if families[r] == family])]
    for name in files:
        values = sub[(sub["file"] == name) & (sub["rows"] > 1)]["median_hz"]
        if not values.empty:
            return f"{values.median():.0f}"
    return "--"


def table_channels(inventory: pd.DataFrame, used: list[str], families: dict[str, str]) -> None:
    """Sensor channels: raw file, released columns, unit and median raw rate per family."""
    channels = (
        (
            "Specific force",
            ("TotalAcceleration", "Accelerometer"),
            r"\texttt{acc\_*}",
            r"m\,s\(^{-2}\)",
        ),
        ("Angular rate", ("Gyroscope",), r"\texttt{gyro\_*}", r"rad\,s\(^{-1}\)"),
        ("Magnetic field", ("Magnetometer",), r"\texttt{mag\_*}", r"\(\mu\)T"),
        ("Gravity (fused)", ("Gravity",), r"\texttt{grav\_*}", r"m\,s\(^{-2}\)"),
        ("Attitude (fused)", ("Orientation",), r"\texttt{q*}, \texttt{roll}, \ldots", "--, rad"),
        ("Pressure", ("Barometer",), r"\texttt{pressure}, \ldots", "hPa, m"),
        ("GNSS fix", ("LocationGps", "Location"), r"\texttt{latitude}, \ldots", "see text"),
    )
    header = " & ".join(
        ["Channel", "Raw file", "Columns", "Unit"]
        + [f"\\rotatebox{{90}}{{{tex_escape(f)}}}" for f in FAMILIES]
    )
    lines = [
        r"\begin{tabular}{llll" + "r" * len(FAMILIES) + "}",
        r"\toprule",
        header + r" \\",
        r"\midrule",
    ]
    for label, files, columns, unit in channels:
        rates = [rate_text(inventory, used, files, f, families) for f in FAMILIES]
        marker = {"TotalAcceleration": "a", "LocationGps": "b"}.get(files[0])
        raw_file = (
            r"\texttt{" + files[0] + "}" + (f"\\textsuperscript{{{marker}}}" if marker else "")
        )
        lines.append(" & ".join([label, raw_file, columns, unit] + rates) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("channels.tex", "\n".join(lines) + "\n")


def table_trajectories(traj: pd.DataFrame) -> None:
    """One row per released trajectory, with summary rows."""
    lines = [
        r"\begin{tabular}{llrrrrrrrr}",
        r"\toprule",
        r"Trajectory & Device & h & km & \makecell{Speed\\(km/h)} & Fixes & \makecell{Fix\\rate (Hz)} "
        r"& \makecell{\(\sigma_h\)\\(m)} & \makecell{\(\sigma_v\)\\(m)} & \makecell{\(\sigma_s\)\\(m/s)} \\",
        r"\midrule",
    ]
    for stem, row in traj.iterrows():
        lines.append(
            f"{tex_escape(stem)} & {tex_escape(row['model'])} & {row['hours']:.2f} & {row['km']:.1f} & "
            f"{row['speed_kmh']:.0f} & {thousands(row['fixes'])} & {row['fix_rate_hz']:.2f} & "
            f"{row['h_acc']:.1f} & {row['v_acc']:.1f} & {row['s_acc']:.2f} \\\\"
        )
    lines += [
        r"\midrule",
        f"Total & & {traj['hours'].sum():.1f} & {thousands(traj['km'].sum())} & & "
        f"{thousands(traj['fixes'].sum())} & & & & \\\\",
        f"Median & & {traj['hours'].median():.2f} & {traj['km'].median():.1f} & "
        f"{traj['speed_kmh'].median():.0f} & {thousands(traj['fixes'].median())} & "
        f"{traj['fix_rate_hz'].median():.2f} & {traj['h_acc'].median():.1f} & "
        f"{traj['v_acc'].median():.1f} & {traj['s_acc'].median():.2f} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    write("trajectories.tex", "\n".join(lines) + "\n")


def fmt_error(metres: float) -> str:
    """Metres below 10 km, kilometres above."""
    if metres >= 10_000:
        return f"{thousands(metres / 1000.0, 1)}~km"
    return thousands(metres, 1)


def table_benchmark() -> dict[str, dict[str, float]]:
    """Unaided EKF/UKF/RBPF horizontal RMSE with full and degraded GNSS."""
    lines = [
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Filter & GNSS & \(n\) & Median (m) & Mean (m) & Max (m) & Vert.\ median (m) \\",
        r"\midrule",
    ]
    cells: dict[str, dict[str, float]] = {}
    for index, filt in enumerate(stats.FILTERS):
        if index:
            lines.append(r"\midrule")
        for scenario, label in (("truth", "Full"), ("degraded", "Degraded")):
            b = stats.baseline(BENCHMARK_ARM, filt, scenario)
            cells[f"{filt}{scenario}"] = b
            name = stats.FILTER_LABEL[filt] if scenario == "truth" else ""
            lines.append(
                f"{name} & {label} & {b['n']} & {fmt_error(b['h_median'])} & {fmt_error(b['h_mean'])} & "
                f"{fmt_error(b['h_max'])} & {fmt_error(b['v_median'])} \\\\"
            )
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("benchmark.tex", "\n".join(lines) + "\n")
    return cells


# --------------------------------------------------------------------------------------------
# Figures


def figure_tracks(traj: pd.DataFrame) -> None:
    """All 27 GNSS tracks over the shaded relief tiles shipped with them, by device family."""
    fig, ax = plt.subplots(figsize=(bf.WIDTH_IN, 3.3), constrained_layout=True)
    for stem in traj.index:
        with netCDF4.Dataset(RELEASE / f"{stem}_relief.nc") as data:
            lat = data["lat"][::4].filled(np.nan)
            lon = data["lon"][::4].filled(np.nan)
            z = data["z"][::4, ::4].filled(np.nan)
        ax.pcolormesh(
            lon, lat, z, cmap="Greys", vmin=-400, vmax=1200, shading="nearest", rasterized=True
        )
    for family in FAMILIES:
        first = True
        for stem in traj.index[traj["family"] == family]:
            track = (
                pd.read_csv(RELEASE / f"{stem}.csv", usecols=["latitude", "longitude"])
                .dropna()
                .iloc[::10]
            )
            ax.plot(
                track["longitude"],
                track["latitude"],
                color=FAMILY_COLOR[family],
                linewidth=1.6,
                label=family if first else None,
            )
            first = False
    ax.set_xlim(traj["lon_min"].min() - 0.3, traj["lon_max"].max() + 0.3)
    ax.set_ylim(traj["lat_min"].min() - 0.3, traj["lat_max"].max() + 0.3)
    ax.set_aspect(1.0 / np.cos(np.radians(40.3)))
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    ax.grid(False)
    ax.legend(loc="lower right", frameon=True, framealpha=0.9, edgecolor=bf.GRID, title="Device")
    save(fig, "fig_tracks")


def ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Empirical CDF."""
    values = np.sort(values)
    return values, np.arange(1, values.size + 1) / values.size


def figure_gnss(traj: pd.DataFrame) -> None:
    """ECDFs of the receiver's advertised horizontal and vertical accuracy, by device family."""
    fig, axes = plt.subplots(1, 2, figsize=(bf.WIDTH_IN, 2.4), constrained_layout=True, sharey=True)
    for family in FAMILIES:
        stems = traj.index[traj["family"] == family]
        frames = [
            pd.read_csv(
                RELEASE / f"{s}.csv", usecols=["latitude", "horizontalAccuracy", "verticalAccuracy"]
            ).dropna(subset=["latitude"])
            for s in stems
        ]
        pooled = pd.concat(frames)
        for ax, column in zip(axes, ("horizontalAccuracy", "verticalAccuracy"), strict=True):
            x, y = ecdf(pooled[column].dropna().to_numpy())
            ax.step(
                x,
                y,
                where="post",
                color=FAMILY_COLOR[family],
                linestyle=FAMILY_STYLE[family],
                linewidth=1.4,
                label=f"{family} ({len(pooled):,} fixes)",
            )
    for ax, title in zip(
        axes,
        ("(a) Advertised horizontal accuracy", "(b) Advertised vertical accuracy"),
        strict=True,
    ):
        ax.set_xscale("log")
        ax.set_xlim(0.5, 1200)
        ax.set_xlabel("Reported accuracy (m)")
        ax.set_title(title, loc="left")
    axes[0].set_ylabel("Fraction of fixes")
    axes[1].legend(loc="lower right", frameon=False, fontsize=6.5)
    save(fig, "fig_gnss_accuracy")


def figure_gravity(norms: dict[str, np.ndarray], families: dict[str, str]) -> None:
    """Raw-rate OS gravity norm less each recording's maximum, pooled by device family."""
    fig, ax = plt.subplots(figsize=(bf.WIDTH_IN, 2.4), constrained_layout=True)
    bins = np.concatenate([[0.0], np.logspace(-2, 3, 51)])
    for family in FAMILIES:
        deficits = np.concatenate(
            [(n.max() - n) * MGAL_PER_MPS2 for r, n in norms.items() if families[r] == family]
        )
        weights = np.full(deficits.size, 1.0 / deficits.size)
        ax.hist(
            np.clip(deficits, 1e-2, 999.0),
            bins=bins[1:],
            weights=weights,
            histtype="step",
            color=FAMILY_COLOR[family],
            linestyle=FAMILY_STYLE[family],
            linewidth=1.4,
            label=f"{family} ({thousands_plain(deficits.size)} samples)",
        )
    ax.axvline(PINNED_TOLERANCE_MGAL, color=bf.MUTED, linewidth=0.8)
    ax.text(
        PINNED_TOLERANCE_MGAL * 1.1,
        0.9,
        "1 mGal",
        transform=ax.get_xaxis_transform(),
        color=bf.MUTED,
        fontsize=7,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(
        r"Recording maximum of $\|\mathbf{g}_\mathrm{OS}\|$ less $\|\mathbf{g}_\mathrm{OS}\|$ (mGal; clipped to [0.01, 1000])"
    )
    ax.set_ylabel("Fraction of samples")
    fig.legend(
        *ax.get_legend_handles_labels(),
        loc="outside lower center",
        ncol=3,
        frameon=False,
        fontsize=6.5,
    )
    save(fig, "fig_gravity_norm")


# --------------------------------------------------------------------------------------------
# Numbers


def numbers(macros: dict[str, str]) -> None:
    """Write every in-text number as a macro."""
    body = "% Generated by scripts/build.py -- do not edit.\n"
    body += "".join(
        f"\\newcommand{{\\num{key}}}{{{value}}}\n" for key, value in sorted(macros.items())
    )
    write("numbers.tex", body)


def main() -> None:
    """Build everything."""
    seg = segments()
    kept = seg[seg["status"] == "kept"]
    recordings = pd.DataFrame([metadata(r.name) for r in sorted(RAW.iterdir()) if r.is_dir()])
    recordings["has_location_gps"] = [
        (RAW / r / "LocationGps.csv").exists() for r in recordings["recording"]
    ]
    used = sorted(kept["source"].unique())
    families = dict(zip(recordings["recording"], recordings["family"], strict=True))

    inventory = raw_inventory()
    traj = trajectory_table(kept)
    norms = {r: raw_gravity(r) for r in used}
    offsets = {r: hard_iron_offset(r) for r in used}
    offsets = {r: v for r, v in offsets.items() if v is not None}
    intensity = {s: calibrated_intensity(s) for s in traj.index}

    table_devices(traj, recordings)
    table_channels(inventory, used, families)
    table_trajectories(traj)
    cells = table_benchmark()

    figure_tracks(traj)
    figure_gnss(traj)
    figure_gravity(norms, families)

    # ---- numbers quoted in the prose
    excluded = seg[seg["status"] != "kept"]
    span = inventory.groupby("recording")["span_s"].max()
    recorded = recordings[recordings["recording"].isin(span.dropna().index)]
    bad = traj.loc["2025-06-14_21-17-02"]
    bad_raw = pd.read_csv(RAW / "2025-06-14_21-17-02" / "Location.csv")
    bad_dt = bad_raw["seconds_elapsed"].diff().dropna()
    pinned_raw = {
        r: float(np.mean((n.max() - n) * MGAL_PER_MPS2 < PINNED_TOLERANCE_MGAL))
        for r, n in norms.items()
    }
    pinned_raw_pixel_apple_s24 = [
        v for r, v in pinned_raw.items() if families[r] != "Samsung SM-A146U"
    ]
    maxima = {f: [norms[r].max() for r in used if families[r] == f] for f in FAMILIES}
    geo = stats.geostats(BENCHMARK_ARM)
    no_baro = [
        r
        for r in recordings["recording"]
        if (RAW / r / "Gyroscope.csv").exists()
        and (
            not (RAW / r / "Barometer.csv").exists()
            or (RAW / r / "Barometer.csv").stat().st_size == 0
        )
    ]
    no_baro_hours = sum(span[r] for r in no_baro) / 3600.0
    ios = traj[traj["platform"] == "iOS"]
    android = traj[traj["platform"] == "Android"]
    macros = {
        "RawRecordings": str(len(recordings)),
        "UtcNamed": str(int(recordings["utc_named"].sum())),
        "RawNonEmpty": str(len(recorded)),
        "RawGB": f"{inventory['bytes'].sum() / 1e9:.1f}",
        "RawRowsMillion": f"{inventory['rows'].sum() / 1e6:.0f}",
        "RawHours": f"{span.dropna().sum() / 3600.0:.1f}",
        "RawFirst": recordings["recording"].min()[:7],
        "RawLast": recordings["recording"].max()[:7],
        "UsedRecordings": str(len(used)),
        "ExcludedRecordings": str(len(recordings) - len(used)),
        "NoBaroRecordings": str(len(no_baro)),
        "NoBaroHours": f"{no_baro_hours:.1f}",
        "Trajectories": str(len(traj)),
        "SplitRecordings": str(int((kept.groupby("source").size() > 1).sum())),
        "DroppedSegments": str(int(excluded["status"].str.startswith("shorter").sum())),
        "Devices": str(recordings["model"].nunique()),
        "DeviceIds": str(recordings["device_id"].nunique()),
        "TotalKm": thousands(traj["km"].sum()),
        "TotalHours": f"{traj['hours'].sum():.1f}",
        "MedianKm": f"{traj['km'].median():.1f}",
        "MedianHours": f"{traj['hours'].median():.2f}",
        "MaxKm": f"{traj['km'].max():.1f}",
        "MaxHours": f"{traj['hours'].max():.2f}",
        "MinKm": f"{traj['km'].min():.1f}",
        "MinHours": f"{traj['hours'].min():.2f}",
        "Rows": thousands(traj["rows"].sum()),
        "Fixes": thousands(traj["fixes"].sum()),
        "MedianFixRate": f"{traj['fix_rate_hz'].median():.2f}",
        "MedianSpeed": f"{traj['speed_kmh'].median():.0f}",
        "MedianHacc": f"{traj['h_acc'].median():.1f}",
        "MedianVacc": f"{traj['v_acc'].median():.1f}",
        "MedianSacc": f"{traj['s_acc'].median():.2f}",
        "HaccMin": f"{traj['h_acc'].min():.1f}",
        "HaccMax": f"{traj['h_acc'].max():.1f}",
        "LatMin": f"{traj['lat_min'].min():.2f}",
        "LatMax": f"{traj['lat_max'].max():.2f}",
        "LonMin": f"{-traj['lon_min'].min():.2f}",
        "LonMax": f"{-traj['lon_max'].max():.2f}",
        "AltMin": f"{traj['alt_min'].min():.0f}",
        "AltMax": f"{traj['alt_max'].max():.0f}",
        "IosBaroFraction": f"{100 * (ios['baro_rows'] / ios['rows']).median():.0f}",
        "AndroidBaroFraction": f"{100 * (android['baro_rows'] / android['rows']).median():.0f}",
        "BadFixes": str(int(bad["fixes"])),
        "BadRawFixes": str(len(bad_raw)),
        "BadSpan": f"{bad['span_s']:.0f}",
        "BadFixRate": f"{bad['fix_rate_hz']:.2f}",
        "BadHacc": f"{bad['h_acc']:.1f}",
        "BadSentinels": str(int(bad["sentinels"])),
        "BadHaccMax": thousands(bad_raw["horizontalAccuracy"].max()),
        "BadMaxGap": f"{bad_dt.max():.1f}",
        "SentinelTrajectories": str(int((traj["sentinels"] > 0).sum())),
        "SentinelTotal": str(int(traj["sentinels"].sum())),
        "GravPixel": f"{np.median(maxima['Pixel 6a'] + maxima['Pixel 9 Pro / Pro XL']):.6f}",
        "GravSamsungApple": f"{np.median(maxima['Samsung SM-S921U'] + maxima['iPhone 12/13 mini']):.6f}",
        "GravAfourteen": f"{np.median(maxima['Samsung SM-A146U']):.6f}",
        "GravRawPinnedMin": f"{100 * min(pinned_raw_pixel_apple_s24):.1f}",
        "GravRawPinnedRecordings": str(sum(v == 1.0 for v in pinned_raw.values())),
        "GravAfourteenPinned": f"{100 * pinned_raw[[r for r in used if families[r] == 'Samsung SM-A146U'][0]]:.1f}",
        "GravTenHzPinnedMedian": f"{100 * traj['grav_10hz_pinned'].median():.0f}",
        "HardIronCount": str(len(offsets)),
        "HardIronMin": f"{min(offsets.values()):.0f}",
        "HardIronMax": f"{max(offsets.values()):.0f}",
        "MagIntensityMin": f"{min(intensity.values()):.0f}",
        "MagIntensityMax": f"{max(intensity.values()):.0f}",
        "SnrGravity": f"{geo['gravity']['snr_median']:.2f}",
        "SnrMagnetic": f"{geo['magnetic']['snr_median']:.2f}",
        "SnrGravityBest": f"{geo['gravity']['snr_best']:.2f}",
        "SnrMagneticBest": f"{geo['magnetic']['snr_best']:.2f}",
    }
    for key, cell in cells.items():
        tag = (
            key.replace("truth", "Full")
            .replace("degraded", "Degraded")
            .replace("ekf", "Ekf")
            .replace("ukf", "Ukf")
            .replace("rbpf", "Rbpf")
        )
        macros[f"{tag}Median"] = fmt_error(cell["h_median"]).replace("~km", "~km")
        macros[f"{tag}N"] = str(cell["n"])
        macros[f"{tag}VMedian"] = fmt_error(cell["v_median"])
    bad_stem = "2025-06-14_21-17-02"
    for filt in ("ekf", "ukf"):
        perf = stats.performance(BENCHMARK_ARM, filt, "truth")["RMSE Horizontal Error (m)"]
        macros[f"{filt.capitalize()}FullBad"] = f"{perf[bad_stem]:.0f}"
        macros[f"{filt.capitalize()}FullMaxOthers"] = f"{perf.drop(bad_stem).max():.1f}"
    missing = sorted(
        set(traj.index) - set(stats.performance(BENCHMARK_ARM, "rbpf", "degraded").index)
    )
    macros["RbpfDegradedMissing"] = ", ".join(tex_escape(m) for m in missing)
    a14 = [r for r in used if families[r] == "Samsung SM-A146U"][0]
    a14_deficit = (norms[a14].max() - norms[a14]) * MGAL_PER_MPS2
    macros["GravAfourteenMedianDeficit"] = f"{np.median(a14_deficit):.0f}"
    macros["GravAfourteenPfiveDeficit"] = f"{np.percentile(a14_deficit, 5):.0f}"
    macros["GravAfourteenPninetyfiveDeficit"] = f"{np.percentile(a14_deficit, 95):.0f}"
    altitude = altitude_offsets(traj)
    ios_cells, ios_median, ios_low, ios_high = altitude["iPhone 12/13 mini"]
    macros["AltIosCells"] = thousands(ios_cells)
    macros["AltIosMedian"] = f"{ios_median:.0f}"
    macros["AltIosLow"] = f"{ios_low:.0f}"
    macros["AltIosHigh"] = f"{ios_high:.0f}"
    android_offsets = [abs(v[1]) for f, v in altitude.items() if f != "iPhone 12/13 mini"]
    macros["AltAndroidMaxOffset"] = f"{max(android_offsets):.0f}"
    numbers(macros)
    inventory.to_csv(HERE / "raw_inventory.csv", index=False)
    traj.drop(columns=["device_id"]).to_csv(HERE / "trajectory_stats.csv")
    print("wrote scripts/raw_inventory.csv, scripts/trajectory_stats.csv")


if __name__ == "__main__":
    main()
