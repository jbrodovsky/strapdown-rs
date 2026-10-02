# Anomaly aiding: EKF, UKF and RBPF (ION *NAVIGATION* manuscript)

Journal manuscript that combines and **corrects** the two ION conference papers
(ITM 2026, UKF; Pacific PNT 2026, RBPF) and adds the EKF. Template: ION's NAVIGATION
LaTeX template (`IONconf-v2.cls`, from `https://www.ion.org/navi/upload/LaTexTemplate.zip`),
APA 7 via biblatex/biber.

## Layout

| Path | What |
|---|---|
| `main.tex` | Preamble, front matter, section includes |
| `sections/*.tex` | One file per section |
| `tables/*.tex` | **Generated** tables and `numbers.tex` / `phone_numbers.tex` macros quoted in the text |
| `figures/*.pdf` | **Generated** figures (`ion_logo.pdf` comes from the template) |
| `scripts/stats.py` | Loads run outputs; paired ratio, bootstrap CI, Wilcoxon |
| `scripts/build_tables.py` | Writes `tables/` |
| `scripts/build_figures.py` | Writes `figures/` |
| `scripts/phone_mechanism.py` | Writes `tables/phone_numbers.tex` (why the phone channels are uninformative) |
| `scripts/rbpf_diagnostics.py` | Writes `tables/rbpf_numbers.tex` (RBPF crossing time, reported σ vs. error) |
| `build_figures.py` also writes | `tables/bound_numbers.tex` (map gradients, single-update scales, noise targets) |
| `NOTES.md` | Open items for the authors |

## Rebuild

The run outputs are not in git. The scripts read them from `STRAPDOWN_DATA`, which defaults to
`data/` at the root of the checkout that holds the scripts (set it when the data live
elsewhere, e.g. when working in a git worktree):

- `data/output` — dedicated-sensor (synthetic ADXL355/RM3100) arm, plus the unaided runs
- `data/output_real2` — smartphone arm
- `data/input`, `data/input_real` — the preprocessed 10 Hz inputs of each arm

```bash
export STRAPDOWN_DATA=/home/james/Code/strapdown-rs/data
uv run --with scipy python papers/anom_combined/scripts/build_tables.py
uv run --with scipy python papers/anom_combined/scripts/phone_mechanism.py
uv run --with scipy python papers/anom_combined/scripts/rbpf_diagnostics.py
uv run --with scipy python papers/anom_combined/scripts/build_figures.py
cd papers/anom_combined && latexmk -pdf main.tex
```

`build_tables.py` is deterministic (seeded bootstrap): rerunning it on unchanged data gives
byte-identical `tables/`.

The dedicated-sensor and unaided runs read the `conf/*.toml` recipes as committed at `2a40f1e`
(2,000 RBPF particles with full GNSS, 1,000 otherwise); the binary's revision was not recorded. The provenance of the smartphone-arm runs in
`data/output_real2` (which configs, which commit) is an open item: see `NOTES.md`.
