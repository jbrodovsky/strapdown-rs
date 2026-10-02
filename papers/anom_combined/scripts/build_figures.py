"""Draw the manuscript's figures into ``figures/``.

Run from anywhere: ``uv run --with scipy python papers/anom_combined/scripts/build_figures.py``.
Every figure is computed from the experiment outputs and inputs under ``STRAPDOWN_DATA``
(see ``stats.py``); nothing is copied from an earlier rendering. Errors are horizontal
great-circle distances to the recorded GNSS fix, evaluated on the rows that carry one, which
is how ``analyze performance`` scores a run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stats  # noqa: E402

FIGURES = Path(__file__).resolve().parent.parent / "figures"
EXAMPLE = "2025-03-01_16-46-39"
EARTH_RADIUS_M = 6_371_008.8
WIDTH_IN = 6.5

# Categorical slots 1-3 of the validated reference palette, which pass the all-pairs CVD
# check; each filter also carries its own marker so identity is never colour alone.
COLOR = {"ekf": "#2a78d6", "ukf": "#eb6834", "rbpf": "#1baf7a"}
MARKER = {"ekf": "o", "ukf": "s", "rbpf": "^"}
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#d9d8d4"

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8.5,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8.5,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.0,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
    }
)


# --------------------------------------------------------------------------------------------
# Loading


def trajectories() -> list[str]:
    """The 27 trajectory stems, sorted."""
    return sorted(stats.dataset_summary().index)


def read_input(stem: str, arm: str = "dedicated") -> pd.DataFrame:
    """Preprocessed input records, indexed by elapsed seconds (rounded to 0.1 s)."""
    directory = stats.DATA / ("input" if arm == "dedicated" else "input_real")
    frame = pd.read_csv(directory / f"{stem}.csv")
    time = pd.to_datetime(frame["time"], utc=True, format="mixed")
    frame["elapsed"] = ((time - time.iloc[0]).dt.total_seconds()).round(1)
    return frame.set_index("elapsed")


def read_output(arm: str, filt: str, scenario: str, stem: str) -> pd.DataFrame:
    """Navigation output of one run, indexed by elapsed seconds (rounded to 0.1 s)."""
    path = stats.arm_dir(arm) / filt / scenario / f"{stem}.csv"
    columns = ["timestamp", "latitude", "longitude", "latitude_cov", "longitude_cov"]
    frame = pd.read_csv(path, usecols=columns)
    time = pd.to_datetime(frame["timestamp"], utc=True, format="mixed")
    frame["elapsed"] = ((time - time.iloc[0]).dt.total_seconds()).round(1)
    return frame.set_index("elapsed")


def haversine_m(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance in metres between degree coordinates."""
    lat1, lon1, lat2, lon2 = (
        np.radians(np.asarray(x, dtype=float)) for x in (lat1, lon1, lat2, lon2)
    )
    a = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def horizontal_error(output: pd.DataFrame, records: pd.DataFrame) -> pd.Series:
    """Horizontal error of a run against the recorded GNSS fixes, by elapsed time.

    Output latitude and longitude are in degrees (the CSV writer converts); the result is
    indexed by elapsed time and has one value per GNSS-bearing record.
    """
    truth = records[["latitude", "longitude"]].dropna()
    joined = truth.join(output[["latitude", "longitude"]], rsuffix="_est", how="inner")
    error = haversine_m(
        joined["latitude"], joined["longitude"], joined["latitude_est"], joined["longitude_est"]
    )
    return pd.Series(error, index=joined.index)


def save(fig: plt.Figure, name: str) -> None:
    """Write a vector PDF."""
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / f"{name}.pdf")
    plt.close(fig)
    print(f"wrote figures/{name}.pdf")


# --------------------------------------------------------------------------------------------
# Figure: maps and tracks


def read_tile(stem: str, field: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Latitude, longitude and anomaly grid of one cropped map tile."""
    with netCDF4.Dataset(stats.DATA / "input" / f"{stem}_{field}.nc") as data:
        return (
            data["lat"][:].filled(np.nan),
            data["lon"][:].filled(np.nan),
            data["z"][:].filled(np.nan),
        )


def figure_maps() -> None:
    """WDMAM and IGPP tiles with the 27 tracks drawn over them."""
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_IN, 2.6), constrained_layout=True)
    panels = (
        ("magnetic", "nT", 250.0, "(a) WDMAM magnetic anomaly, 3 arc-min"),
        ("gravity", "mGal", 50.0, "(b) IGPP free-air gravity anomaly, 1 arc-min"),
    )
    tracks = {
        stem: read_input(stem)[["latitude", "longitude"]].dropna().iloc[::20]
        for stem in trajectories()
    }
    for ax, (field, unit, limit, title) in zip(axes, panels, strict=True):
        mesh = None
        for stem in trajectories():
            lat, lon, grid = read_tile(stem, field)
            mesh = ax.pcolormesh(
                lon,
                lat,
                grid,
                cmap="RdBu_r",
                vmin=-limit,
                vmax=limit,
                shading="nearest",
                rasterized=True,
            )
        for track in tracks.values():
            ax.plot(track["longitude"], track["latitude"], color=INK, linewidth=0.6)
        ax.set_aspect(1.0 / np.cos(np.radians(40.3)))
        ax.set_title(title, loc="left")
        ax.set_xlabel("Longitude (deg)")
        ax.grid(False)
        fig.colorbar(mesh, ax=ax, shrink=0.85, label=unit, extend="both")
    axes[0].set_ylabel("Latitude (deg)")
    save(fig, "fig_maps")


# --------------------------------------------------------------------------------------------
# Figure: residuals against the map


def residuals(arm: str) -> pd.DataFrame:
    """Per-sample anomaly residuals with each trajectory's own median removed."""
    frame = pd.read_csv(
        stats.arm_dir(arm) / "geostats" / "geo_residuals.csv",
        usecols=["trajectory", "field", "residual", "map"],
    )
    grouped = frame.groupby(["trajectory", "field"])
    frame["centred"] = frame["residual"] - grouped["residual"].transform("median")
    frame["map_centred"] = frame["map"] - grouped["map"].transform("mean")
    return frame


def figure_residuals() -> None:
    """Residual spread against the map signal, phone vs dedicated sensors, per field."""
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH_IN, 3.6), constrained_layout=True)
    data = {arm: residuals(arm) for arm in ("phone", "dedicated")}
    layout = (
        ("gravity", "mGal", ("Smartphone", "ADXL355")),
        ("magnetic", "nT", ("Smartphone", "RM3100")),
    )
    for row, (field, unit, names) in enumerate(layout):
        for col, arm in enumerate(("phone", "dedicated")):
            ax = axes[row, col]
            frame = data[arm][data[arm]["field"] == field]
            resid = frame["centred"].to_numpy()
            signal = frame["map_centred"].to_numpy()
            spread = 1.4826 * np.median(np.abs(resid))
            span = max(4.0 * spread, 4.0 * np.std(signal))
            bins = np.linspace(-span, span, 81)
            ax.hist(
                resid,
                bins=bins,
                density=True,
                color=COLOR["ekf"],
                alpha=0.85,
                label="Residual about its median",
            )
            ax.hist(
                signal,
                bins=bins,
                density=True,
                histtype="step",
                color=INK,
                linewidth=1.0,
                label="Map anomaly along track",
            )
            ax.set_title(f"({'abcd'[2 * row + col]}) {names[col]} {field}", loc="left")
            ax.set_xlabel(f"{unit}")
            ax.set_yticks([])
            ax.text(
                0.98,
                0.95,
                f"robust $\\sigma$ = {spread:,.1f} {unit}\npooled map $\\sigma$ = {np.std(signal):,.1f} {unit}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=7,
                color=INK,
            )
    axes[0, 0].legend(loc="upper left", frameon=False, fontsize=6.5)
    save(fig, "fig_residuals")


# --------------------------------------------------------------------------------------------
# Figure: why the smartphone channels carry no map information


def figure_phone_sensors() -> None:
    """Pinned gravity norm and heading-dependent magnetic intensity on the example drive."""
    records = read_input(EXAMPLE, arm="phone")
    resid = pd.read_csv(stats.DATA / "output_real2" / "geostats" / "geo_residuals.csv")
    resid = resid[resid["trajectory"] == EXAMPLE]

    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_IN, 2.4), constrained_layout=True)
    norm = np.sqrt(records["grav_x"] ** 2 + records["grav_y"] ** 2 + records["grav_z"] ** 2)
    ceiling = norm.max()
    minutes = records.index.to_numpy() / 60.0
    gravity_map = resid[resid["field"] == "gravity"]
    ax = axes[0]
    ax.plot(
        minutes,
        (norm - ceiling) * 1e5,
        color=COLOR["ekf"],
        linewidth=0.4,
        label=r"$\|\mathbf{g}_\mathrm{obs}\|$ less its maximum",
    )
    ax.plot(
        gravity_map["elapsed_s"] / 60.0,
        gravity_map["map"] - gravity_map["map"].mean(),
        color=INK,
        linewidth=1.0,
        label="Map anomaly along track (mean removed)",
    )
    ax.set_ylim(-150, 60)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("mGal")
    pinned = np.mean((ceiling - norm) * 1e5 < 1.0)
    ax.set_title(
        f"(a) Gravity: norm pinned at {ceiling:.6f} m/s$^2$ ({100 * pinned:.0f}% within 1 mGal)",
        loc="left",
    )
    ax.legend(loc="lower left", frameon=False, fontsize=6.5)

    magnetic = resid[resid["field"] == "magnetic"]
    ax = axes[1]
    ax.plot(
        magnetic["elapsed_s"] / 60.0,
        (magnetic["measured"] - magnetic["measured"].median()) / 1000.0,
        color=COLOR["ekf"],
        linewidth=0.4,
        label=r"$\|\mathbf{B}_\mathrm{obs}\| - F_\mathrm{WMM}$ (median removed)",
    )
    ax.plot(
        magnetic["elapsed_s"] / 60.0,
        (magnetic["map"] - magnetic["map"].mean()) / 1000.0,
        color=INK,
        linewidth=1.0,
        label="Map anomaly along track (mean removed)",
    )
    ax.set_xlabel("Time (min)")
    ax.set_ylabel(r"$\mu$T")
    ax.set_title("(b) Magnetic: observed intensity varies by microtesla", loc="left")
    ax.legend(loc="upper left", frameon=False, fontsize=6.5)
    save(fig, "fig_phone_sensors")


# --------------------------------------------------------------------------------------------
# Figure: baselines


def figure_baselines() -> None:
    """Per-trajectory horizontal RMSE of each unaided filter with full and degraded GNSS."""
    fig, ax = plt.subplots(figsize=(WIDTH_IN, 2.3), constrained_layout=True)
    rows = []
    for scenario, label in (("truth", "full GNSS"), ("degraded", "GNSS every 60 s")):
        for filt in stats.FILTERS:
            rows.append((filt, scenario, f"{stats.FILTER_LABEL[filt]}, {label}"))
    rng = np.random.default_rng(1)
    for y, (filt, scenario, _) in enumerate(rows):
        values = stats.performance("dedicated", filt, scenario)[
            "RMSE Horizontal Error (m)"
        ].to_numpy()
        jitter = rng.uniform(-0.18, 0.18, values.size)
        ax.scatter(
            values,
            y + jitter,
            s=12,
            marker=MARKER[filt],
            facecolor=COLOR[filt],
            edgecolor="white",
            linewidth=0.4,
            zorder=3,
        )
        median = np.median(values)
        ax.plot([median, median], [y - 0.32, y + 0.32], color=INK, linewidth=1.4, zorder=4)
    ax.set_xscale("log")
    ax.set_yticks(range(len(rows)), [r[2] for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("Horizontal RMSE against the recorded GNSS track (m); bar = median")
    ax.grid(axis="y", visible=False)
    save(fig, "fig_baselines")


# --------------------------------------------------------------------------------------------
# Figure: paired ratios


def figure_ratios() -> None:
    """Per-trajectory aided/unaided RMSE ratio for the Kalman filters, both observation sets."""
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_IN, 2.5), constrained_layout=True)
    rng = np.random.default_rng(2)
    for ax, arm, title in (
        (axes[0], "phone", "(a) Smartphone observations"),
        (axes[1], "dedicated", "(b) Dedicated sensors (ADXL355, RM3100)"),
    ):
        ticks, labels = [], []
        for i, filt in enumerate(("ekf", "ukf")):
            for j, channel in enumerate(stats.CHANNELS):
                x = i * 3.6 + j
                frame = stats.detailed(arm, filt, channel)
                ratio = (frame["geo_rmse"] / frame["baseline_rmse"]).to_numpy()
                summary = stats.paired(arm, filt, channel)
                ax.scatter(
                    x + rng.uniform(-0.22, 0.22, ratio.size),
                    ratio,
                    s=9,
                    marker=MARKER[filt],
                    facecolor=COLOR[filt],
                    edgecolor="white",
                    linewidth=0.3,
                    alpha=0.9,
                    zorder=3,
                )
                ax.errorbar(
                    x + 0.36,
                    summary.ratio_median,
                    yerr=[
                        [summary.ratio_median - summary.ratio_low],
                        [summary.ratio_high - summary.ratio_median],
                    ],
                    fmt="_",
                    color=INK,
                    markersize=8,
                    capsize=2,
                    linewidth=1.0,
                    zorder=4,
                )
                ticks.append(x)
                labels.append(
                    f"{stats.FILTER_LABEL[filt]}\n{ {'grav': 'Grav.', 'mag': 'Mag.', 'both': 'Comb.'}[channel] }"
                )
        ax.axhline(1.0, color=MUTED, linewidth=0.8, zorder=2)
        ax.set_xticks(ticks, labels)
        ax.set_title(title, loc="left")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Aided / unaided horizontal RMSE")
    axes[0].ticklabel_format(axis="y", useOffset=False)
    save(fig, "fig_ratios")


# --------------------------------------------------------------------------------------------
# Figure: example drive


def figure_example() -> None:
    """Aided minus unaided EKF error on the example drive, phone vs RM3100, on one scale."""
    records = read_input(EXAMPLE)
    fig, axes = plt.subplots(
        2, 1, figsize=(WIDTH_IN, 3.0), sharex=True, sharey=True, constrained_layout=True
    )
    for ax, arm, title in (
        (axes[0], "phone", "(a) Smartphone magnetometer"),
        (axes[1], "dedicated", "(b) Simulated RM3100 magnetometer"),
    ):
        unaided = horizontal_error(read_output(arm, "ekf", "degraded", EXAMPLE), records)
        aided = horizontal_error(read_output(arm, "ekf", "mag", EXAMPLE), records)
        diff = (aided - unaided).dropna()
        minutes = diff.index.to_numpy() / 60.0
        ax.fill_between(
            minutes,
            0,
            diff.to_numpy(),
            where=diff.to_numpy() < 0,
            color=COLOR["rbpf"],
            linewidth=0,
            label="Aiding reduced the error",
        )
        ax.fill_between(
            minutes,
            0,
            diff.to_numpy(),
            where=diff.to_numpy() >= 0,
            color=COLOR["ukf"],
            linewidth=0,
            label="Aiding increased the error",
        )
        rmse_change = np.sqrt(np.mean(aided**2)) - np.sqrt(np.mean(unaided**2))
        ax.set_title(f"{title}: RMSE change {rmse_change:+.1f} m", loc="left")
        ax.set_ylabel("Error change (m)")
        ax.axhline(0, color=MUTED, linewidth=0.6)
    axes[1].set_xlabel("Time (min)")
    axes[0].legend(loc="upper left", frameon=False, ncol=2)
    save(fig, "fig_example")


# --------------------------------------------------------------------------------------------
# Figure: RBPF divergence


def figure_rbpf() -> None:
    """RBPF error and reported uncertainty under 60 s GNSS against the EKF, on the example."""
    records = read_input(EXAMPLE)
    output = read_output("dedicated", "rbpf", "degraded", EXAMPLE)
    error = horizontal_error(output, records)
    ekf = horizontal_error(read_output("dedicated", "ekf", "degraded", EXAMPLE), records)
    full = horizontal_error(read_output("dedicated", "rbpf", "truth", EXAMPLE), records)
    lat = np.radians(records["latitude"].ffill().reindex(output.index).ffill().to_numpy())
    sigma = np.sqrt(
        output["latitude_cov"].clip(lower=0) * EARTH_RADIUS_M**2
        + output["longitude_cov"].clip(lower=0) * (EARTH_RADIUS_M * np.cos(lat)) ** 2
    )
    fig, ax = plt.subplots(figsize=(WIDTH_IN, 2.4), constrained_layout=True)
    ax.plot(error.index / 60.0, error, color=COLOR["rbpf"], label="RBPF error, GNSS every 60 s")
    ax.plot(
        output.index / 60.0,
        sigma,
        color=COLOR["rbpf"],
        linestyle="--",
        linewidth=0.8,
        label=r"RBPF reported horizontal 1$\sigma$",
    )
    ax.plot(ekf.index / 60.0, ekf, color=COLOR["ekf"], label="EKF error, GNSS every 60 s")
    ax.plot(full.index / 60.0, full, color=MUTED, linewidth=0.6, label="RBPF error, full GNSS")
    ax.set_yscale("log")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Horizontal (m)")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), frameon=False, ncol=4, fontsize=7)
    save(fig, "fig_rbpf")


# --------------------------------------------------------------------------------------------
# Figure: localization bound


def bilinear_gradient(
    lat: np.ndarray, lon: np.ndarray, grid: np.ndarray, plat: np.ndarray, plon: np.ndarray
) -> np.ndarray:
    """Gradient magnitude, per km, of the bilinear interpolant at each track point.

    This is the derivative the filters' map model has: each point is located in its grid
    cell and the bilinear surface of that cell is differentiated at the point's fractional
    position. Points outside the tile are NaN.
    """
    i = np.searchsorted(lat, plat, side="right") - 1
    j = np.searchsorted(lon, plon, side="right") - 1
    inside = (i >= 0) & (i < lat.size - 1) & (j >= 0) & (j < lon.size - 1)
    i = np.clip(i, 0, lat.size - 2)
    j = np.clip(j, 0, lon.size - 2)
    cell_lat = lat[i + 1] - lat[i]
    cell_lon = lon[j + 1] - lon[j]
    t_lat = (plat - lat[i]) / cell_lat
    t_lon = (plon - lon[j]) / cell_lon
    f00, f01 = grid[i, j], grid[i, j + 1]
    f10, f11 = grid[i + 1, j], grid[i + 1, j + 1]
    per_deg_north = ((1 - t_lon) * (f10 - f00) + t_lon * (f11 - f01)) / cell_lat
    per_deg_east = ((1 - t_lat) * (f01 - f00) + t_lat * (f11 - f10)) / cell_lon
    km_per_deg = np.radians(1.0) * EARTH_RADIUS_M / 1000.0
    north = per_deg_north / km_per_deg
    east = per_deg_east / (km_per_deg * np.cos(np.radians(plat)))
    return np.where(inside, np.hypot(north, east), np.nan)


def median_gradients() -> dict[str, float]:
    """Pooled median map-gradient magnitude along the 27 tracks, per field."""
    result = {}
    for field in ("gravity", "magnetic"):
        values = []
        for stem in trajectories():
            track = read_input(stem)[["latitude", "longitude"]].dropna()
            lat, lon, grid = read_tile(stem, field)
            values.append(
                bilinear_gradient(
                    lat, lon, grid, track["latitude"].to_numpy(), track["longitude"].to_numpy()
                )
            )
        result[field] = float(np.nanmedian(np.concatenate(values)))
    return result


def write_bound_macros(
    gradients: dict[str, float], bounds: list[float], degraded_km: float
) -> None:
    """Gradient, scale and noise-requirement numbers quoted in the text, as LaTeX macros."""
    names = ("phonemag", "phonegrav", "adxl", "rmthree")
    values = {
        "gradgrav": f"{gradients['gravity']:.2f}",
        "gradmag": f"{gradients['magnetic']:.2f}",
        "degradedkm": f"{degraded_km:.2f}",
        "reqgrav": f"{degraded_km * gradients['gravity']:.1f}",
        "reqmag": f"{degraded_km * gradients['magnetic']:.1f}",
    }
    for name, bound in zip(names, bounds, strict=True):
        values[f"scale{name}"] = f"{bound:,.1f}" if bound < 10 else f"{bound:,.0f}"
    lines = [
        f"\\newcommand{{\\bound{key}}}{{{value.replace(',', '{,}')}}}"
        for key, value in sorted(values.items())
    ]
    (FIGURES.parent / "tables" / "bound_numbers.tex").write_text("\n".join(lines) + "\n")


def figure_bound(gradients: dict[str, float]) -> None:
    """Noise over map gradient for each observation, against the degraded-GNSS error."""
    phone = stats.geostats("phone")
    entries = [
        ("Smartphone magnetic", phone["magnetic"]["within_sigma"], "magnetic"),
        ("Smartphone gravity", phone["gravity"]["within_sigma"], "gravity"),
        ("ADXL355 gravity (simulated)", 54.8208, "gravity"),
        ("RM3100 magnetic (simulated)", 3.91675, "magnetic"),
    ]
    bounds = [noise / gradients[field] for _, noise, field in entries]
    degraded_km = stats.baseline("dedicated", "ekf", "degraded")["h_median"] / 1000.0
    write_bound_macros(gradients, bounds, degraded_km)
    fig, ax = plt.subplots(figsize=(WIDTH_IN, 1.8), constrained_layout=True)
    ax.barh(range(len(entries)), bounds, color=[MUTED, MUTED, MUTED, COLOR["ekf"]], height=0.55)
    for y, value in enumerate(bounds):
        ax.text(
            value * 1.12,
            y,
            f"{value:,.1f} km" if value < 10 else f"{value:,.0f} km",
            va="center",
            fontsize=7,
            color=INK,
        )
    ax.axvline(
        degraded_km,
        color=COLOR["ukf"],
        linewidth=1.2,
        label=f"EKF degraded-GNSS median error, {degraded_km:.2f} km",
    )
    ax.legend(loc="upper right", frameon=False, fontsize=7)
    ax.set_xscale("log")
    ax.set_xlim(0.1, 1e5)
    ax.set_yticks(range(len(entries)), [e[0] for e in entries])
    ax.set_xlabel(
        r"Single-update position scale along the gradient, $\sqrt{R}\,/\,|\nabla\mathcal{M}|$ (km)"
    )
    ax.grid(axis="y", visible=False)
    save(fig, "fig_bound")
    print(
        f"median gradients: gravity {gradients['gravity']:.2f} mGal/km, magnetic {gradients['magnetic']:.2f} nT/km"
    )
    print("bounds (km):", {e[0]: round(b, 2) for e, b in zip(entries, bounds, strict=True)})


def main() -> None:
    """Draw every figure."""
    figure_maps()
    figure_residuals()
    figure_phone_sensors()
    figure_baselines()
    figure_ratios()
    figure_example()
    figure_rbpf()
    figure_bound(median_gradients())


if __name__ == "__main__":
    main()
