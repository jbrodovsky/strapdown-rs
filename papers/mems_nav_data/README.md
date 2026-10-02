# MEMS-Nav data article (Data in Brief)

The revised MEMS-Nav data paper, retargeted from IJRR to Elsevier *Data in Brief* (fallback: MDPI *Data*).

## Layout

- `main.tex`: `elsarticle` preprint (12pt, author-year natbib), which `\input`s `sections/*.tex` in Data in Brief order
- `sections/`: abstract, specifications table, value of the data, background, data description, methods, limitations, statements
- `tables/`: **generated**, do not edit. `numbers.tex` holds one `\num...` macro for every number quoted in the prose
- `figures/`: **generated**, do not edit
- `scripts/build.py`: builds `tables/`, `figures/`, `scripts/raw_inventory.csv` and `scripts/trajectory_stats.csv`
- `references.bib`, `NOTES.md` (open items for the authors)

## Rebuild

```bash
# 1. tables, figures and in-text numbers (about a minute; reads ~23 GB of raw CSV once)
/home/james/Code/strapdown-rs/.venv/bin/python papers/mems_nav_data/scripts/build.py

# 2. the PDF
cd papers/mems_nav_data && latexmk -pdf main.tex
```

`build.py` needs pandas, numpy, matplotlib and netCDF4, plus scipy (imported by the shared helper). It reads, and never writes:

- `$STRAPDOWN_DATA/raw/*/`: the Sensor Logger recordings (read-only)
- `$STRAPDOWN_DATA/input_real/`: the released 10 Hz trajectories and map tiles. `input/` is **not** used for sensor statistics because its `grav_*`/`mag_*` columns are simulated
- `$STRAPDOWN_DATA/input/segments.json`
- `$STRAPDOWN_DATA/output/dataset_summary/dataset_summary.csv`
- `$STRAPDOWN_DATA/output_real2/{ekf,ukf,rbpf}/{truth,degraded}/performance/performance_summary.csv` and `output_real2/geostats/geo_stats_pooled.json`

`STRAPDOWN_DATA` defaults to `/home/james/Code/strapdown-rs/data`. The loaders, the matplotlib style and the data root are imported unmodified from `papers/anom_combined/scripts/{stats,build_figures}.py`. The build is deterministic: two runs give byte-identical tables and figures.
