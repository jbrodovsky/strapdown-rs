# Install both halves of the monorepo: the Rust toolchain fetches itself on the first cargo
# command, and `uv sync` builds the `analysis` workspace member from the root pyproject.toml.
# Install both halves of the monorepo: the Rust toolchain and the Python workspace.
setup:
    cargo fetch
    uv sync

# Build the project in release mode
build:
    cargo build --release --workspace --all-features

# Lint and test the Python half, the same four commands .github/workflows/python.yml runs.
check-python:
    uv run ruff check
    uv run ruff format --check
    uv run pytest -q

# Rebuild data/input from the Sensor Logger exports in data/raw. `data/raw` is read-only --
# nothing in this repo may write to it.
#
# Recordings whose IMU drops out mid-flight are split at the gap into `<name>_A.csv`,
# `<name>_B.csv`, ... and ones whose IMU stops before the GPS does are trimmed back to their
# last inertial sample, because `strapdown-sim` rejects any file carrying a gap longer than
# `max_imu_gap_s` (5 s) rather than dead-reckoning across it. `--max-imu-gap-s` here must stay
# equal to that limit. See data/input/segments.json for what each run produced.
#
# `--prune` deletes CSVs in data/input that the run did not write, so the directory stays a
# faithful function of data/raw. It matters most right after a recording starts being split:
# the un-split original would otherwise stay behind, and since the simulator loads every CSV
# in the directory it would go on being run and go on failing. Everything here is derived
# from data/raw, which is read-only and is never written by any recipe.
#
# 10 Hz, into `data/input`. GNSS is only ever recorded at ~1 Hz, so the extra rows carry
# inertial data only and the GNSS columns stay NaN in 9 bins out of 10 -- which is what gives
# 10 Hz propagation against 1 Hz aiding. Do not interpolate GNSS up to match.
#
# The output directory is what every consumer reads: `input = "data/input"` in all 15
# conf/*.toml, `-i data/input` in `geo-stats`, and `-r data/input` in `postprocess` and the
# `geoperf-*` recipes. It wrote to `data/input_10hz` for a while, which nothing downstream
# read, so `just pipeline` preprocessed into one directory and simulated from whatever stale
# data happened to be in the other. Change the rate here and everything follows; change the
# directory and 15 files have to follow it.
#
# `--getmaps` is required, not optional. Without it `preprocess` calls `inherit_parent_maps`,
# which copies `data/input/<source>_{gravity,magnetic}.nc` onto each split segment -- but
# `clean` has just deleted those, so there is nothing to inherit and the directory ends up
# with trajectories and no maps at all. Every geophysical run then dies on "Gravity map file
# not found", three steps downstream of the cause. It needs the GMT C library (`libgmt`) and
# network access to the GMT data server.
#
# `-b` is a *fraction*, not a percentage: `inflate_bounds` computes `x_min - x_range * buffer`
# and defaults to 0.1 for a 10% margin. It was `-b 10` for a while, which padded each map's
# bounding box by ten times the track's own extent per side -- a box about 21x wider and 21x
# taller than the track. That was harmless while `--getmaps` was absent and nothing was
# downloaded; with it, the relief grid for the longest recording here is already 195 MB at a
# 10% margin, so 21x the area is gigabytes per trajectory.
#
# `--margin-km` is the absolute floor `pad_bounds` adds on top of `-b`, and matters more than
# `-b` for a short or straight track (see `pad_bounds`'s own docstring). `DEFAULT_MAP_MARGIN_KM`
# (5 km) was sized for the 5 s GNSS scheduler; widening `interval_s` to 60 s in
# `conf/rbpf_*.toml` lets the RBPF estimate wander further before the next fix pulls it back,
# and 22 of 27 trajectories left the 5 km tile at points a few tens of metres past the
# boundary. 10 km keeps comfortable headroom over that without material extra download size.
#
# `--getmaps` downloads each trajectory's `_gravity.nc`/`_magnetic.nc` siblings via pygmt.
# Without it `data/input` never gets them, and `geo-stats` silently has nothing to measure --
# it reports "no trajectory yielded a usable channel" rather than failing loudly. `write_segment`
# re-fetches and overwrites both files on every run regardless (pygmt caches the underlying
# grids locally, so a re-run costs no network, but it still rewrites the `.nc` files).
#
# `--synthetic` makes data/input the SYNTHETIC arm. The phone's gravity and magnetometer
# readings carry no map information (GEO_AIDING_NOTES.md §3), so they are replaced -- and only
# they: every IMU, GNSS, barometer and attitude column is untouched -- by what a low-cost
# ADXL355 gravimeter and RM3100 magnetometer would have read: each map sampled at the GNSS
# track, plus the part's datasheet error model (analysis/src/analysis/synthetic.py). The seed
# fixes the sensor errors, and data/input/synthetic.json records the models and every draw.
# conf/*.toml describe these sensors, per 10 Hz row. The real arm is frozen: its inputs in
# data/input_real, its results in data/output_real, its configs in conf/real/.
#
# Rebuild data/input from data/raw at 10 Hz with synthetic gravimeter and magnetometer readings.
preprocess:
    uv run analyze preprocess -i data/raw -o data/input -f 10 \
        -b 1.5 --margin-km 10.0 --getmaps --max-imu-gap-s 5.0 --min-segment-s 300.0 --prune \
        --synthetic --synthetic-seed 42

# conf/*.toml give the synthetic sensors' white noise per 10 Hz row. A 1 Hz row averages ten
# times as many samples, so its noise is sqrt(10) smaller than those configs tell the filter.
#
# Rebuild data/input at 1 Hz instead, matching the rate every result before this branch used.
preprocess-1hz:
    uv run analyze preprocess -i data/raw -o data/input -f 1 \
        --margin-km 10.0 --max-imu-gap-s 5.0 --min-segment-s 300.0 --prune --getmaps \
        --synthetic --synthetic-seed 42

# Reports only -- it writes `data/output/geostats/geo_stats.toml` and leaves `conf/` alone,
# which is why `pipeline` can run it without changing the experiment underneath itself.
#
# On the synthetic data/input it is the check that the planted sensors came out as specified:
# within-trajectory sigma near each `*_noise_std` in conf/*.toml, and per-trajectory medians
# equal to the turn-on biases drawn in data/input/synthetic.json.
#
# Measure the geophysical residual against the maps: bias, noise, SNR and figure.
geo-stats:
    uv run analyze geostats -i data/input -o data/output/geostats

# The 100 mGal / 150 nT the configs shipped with were never measured against these maps. A
# filter told a noise it does not have produces a covariance that stops describing its error,
# which is the whole failure mode this branch exists to fix -- so adopting the measurement is
# the point of `geo-stats`, not an optional extra.
#
# Deliberately separate from `geo-stats` and absent from `pipeline`: it rewrites tracked files
# and invalidates every result already in data/output. Run it, read `git diff conf/`, then
# re-run the simulations.
#
# `geo_frequency_s` is left at 1.0 unless you add `--apply-interval`. One measurement per
# de-correlation length is several hundred seconds, which changes what the experiment asks of
# the aid rather than how it is tuned. That one is a decision, not a measurement.
#
# That was the real arm's method, and its adopted values are frozen in conf/real/. conf/*.toml
# now carry the synthetic sensors' datasheet values, which analysis/tests/test_synthetic.py
# checks against the model; this would replace them with numbers re-measured from data/input,
# and that test would fail. `pipeline` no longer runs it.
#
# Write the measured bias, noise and bias prior into all 9 geophysical configs.
geo-adopt *ARGS:
    uv run analyze geostats -i data/input -o data/output/geostats --apply-to conf {{ARGS}}

# Run the truth configurations.
truth:
    -./target/release/strapdown-sim --config conf/ukf_truth.toml
    -./target/release/strapdown-sim --config conf/ekf_truth.toml

# Run the degraded configurations.
degraded:
    -./target/release/strapdown-sim --config conf/ukf_degraded.toml
    -./target/release/strapdown-sim --config conf/ekf_degraded.toml

# Geophysical aiding runs. These now point at conf/*.toml like `truth` and `degraded` do:
# a closed-loop config carrying a [geophysical] section used to be refused, so these recipes
# carried the equivalent command line by hand and the comment here warned that the
# --sched/--fault flags had to be kept in step with conf/*_degraded.toml by hand. They no
# longer can drift: the GNSS profile lives in the config file beside the maps.
#
# `geoperf-all` scores each geo-aided run against that filter's degraded run, so the
# [gnss_degradation] and [health_limits] sections of conf/{ukf,ekf}_{grav,mag,both}.toml must
# stay identical to conf/{ukf,ekf}_degraded.toml. Otherwise the "improvement" it reports is
# the difference between two degradations.

# Run the UKF geophysical aiding configurations (both, gravity-only, magnetic-only).
ukf-geo:
    -./target/release/strapdown-sim --config conf/ukf_both.toml
    -./target/release/strapdown-sim --config conf/ukf_grav.toml
    -./target/release/strapdown-sim --config conf/ukf_mag.toml

# Run the EKF geophysical aiding configurations (both, gravity-only, magnetic-only).
ekf-geo:
    -./target/release/strapdown-sim --config conf/ekf_both.toml
    -./target/release/strapdown-sim --config conf/ekf_grav.toml
    -./target/release/strapdown-sim --config conf/ekf_mag.toml

# RBPF sims (truth, degraded, and each geophysical aiding combination). Like ukf-geo/ekf-geo
# these run entirely from conf/rbpf_*.toml; RBPF was simply the only filter that could,
# before closed-loop mode learned to carry a [geophysical] section.

# Run the RBPF configurations.
rbpf-sim:
    -./target/release/strapdown-sim --config conf/rbpf_truth.toml
    -./target/release/strapdown-sim --config conf/rbpf_degraded.toml
    -./target/release/strapdown-sim --config conf/rbpf_grav.toml
    -./target/release/strapdown-sim --config conf/rbpf_mag.toml
    -./target/release/strapdown-sim --config conf/rbpf_both.toml

# Performance postprocessing for all scenarios. `analyze` is the analysis/ package's CLI. It
# is a checked-in member of the uv workspace declared in the root pyproject.toml, so `uv run`
# from the repository root resolves it -- no `--project analysis`, and no assumption that it
# is on PATH.
#
# Each invocation writes `performance_summary.csv` (per-trajectory min/max/mean/RMSE plus
# mean/median/std rows) and `performance_table.tex` -- a pasteable LaTeX table of horizontal,
# vertical and 3D RMSE against GPS truth, the `truth`/`degraded` counterpart to the geo-aided
# tables `geoperf-all` writes. Pass `--no-latex` to skip it.
#
# `dataset-summary` runs once against `data/input` itself, not per filter/scenario: distance
# traveled and duration describe the recordings, not any navigation solution, so they do not
# vary across ukf/ekf/rbpf or truth/degraded/geo-aided. This used to be computed ad hoc in an
# untracked notebook (`data.ipynb`, gitignored) and was never reproducible from the tracked
# pipeline; see `dataset_summary_analysis` in analysis/src/analysis/__init__.py.

# Postprocess the performance of all scenarios.
postprocess:
    -uv run analyze dataset-summary --input data/input --output data/output/dataset_summary
    -uv run analyze performance --processed data/output/ukf/truth --reference data/input/ --output data/output/ukf/truth/performance
    -uv run analyze performance --processed data/output/ukf/degraded --reference data/input/ --output data/output/ukf/degraded/performance
    -uv run analyze performance --processed data/output/ukf/both --reference data/input/ --output data/output/ukf/both/performance
    -uv run analyze performance --processed data/output/ukf/grav --reference data/input/ --output data/output/ukf/grav/performance
    -uv run analyze performance --processed data/output/ukf/mag --reference data/input/ --output data/output/ukf/mag/performance
    -uv run analyze performance --processed data/output/ekf/truth --reference data/input/ --output data/output/ekf/truth/performance
    -uv run analyze performance --processed data/output/ekf/degraded --reference data/input/ --output data/output/ekf/degraded/performance
    -uv run analyze performance --processed data/output/ekf/both --reference data/input/ --output data/output/ekf/both/performance
    -uv run analyze performance --processed data/output/ekf/grav --reference data/input/ --output data/output/ekf/grav/performance
    -uv run analyze performance --processed data/output/ekf/mag --reference data/input/ --output data/output/ekf/mag/performance
    -uv run analyze performance --processed data/output/rbpf/truth --reference data/input/ --output data/output/rbpf/truth/performance
    -uv run analyze performance --processed data/output/rbpf/degraded --reference data/input/ --output data/output/rbpf/degraded/performance
    -uv run analyze performance --processed data/output/rbpf/both --reference data/input/ --output data/output/rbpf/both/performance
    -uv run analyze performance --processed data/output/rbpf/grav --reference data/input/ --output data/output/rbpf/grav/performance
    -uv run analyze performance --processed data/output/rbpf/mag --reference data/input/ --output data/output/rbpf/mag/performance

# Geophysical performance analysis for all filter types. Each geo-aided run is scored twice:
#
#   analysis/       (or analysis/rbpf) against its OWN filter's degraded run -- what the
#                   geophysical measurement contributes to that filter.
#   analysis/ins    against conf/ekf_degraded.toml's run -- what the whole geo-aided filter
#                   is worth against the canonical degraded INS.
#
# The INS baseline is the **EKF**, not the UKF. It is the conventional integration filter and
# the one to beat; scoring the RBPF against a UKF makes the comparison read as RBPF-vs-UKF,
# which is not the claim being tested. The two degraded runs are near-identical on this
# dataset anyway (median RMSE 45.53 m UKF vs 45.44 m EKF), so this changes the framing far
# more than the numbers -- which is exactly why it is worth getting right.

# Run the geophysical performance analysis for all filter types, against the `degraded`
# baseline. This is the established comparison: every result recorded so far used it.
# Score every geo-aided run against its filter's `degraded` run.
geoperf-all:
    -uv run analyze geoperformance -p data/output/ukf/grav -d data/output/ukf/degraded -r data/input -o data/output/ukf/grav/analysis -f ukf --geo-type grav
    -uv run analyze geoperformance -p data/output/ukf/mag -d data/output/ukf/degraded -r data/input -o data/output/ukf/mag/analysis -f ukf --geo-type mag
    -uv run analyze geoperformance -p data/output/ukf/both -d data/output/ukf/degraded -r data/input -o data/output/ukf/both/analysis -f ukf --geo-type both
    -uv run analyze geoperformance -p data/output/ekf/grav -d data/output/ekf/degraded -r data/input -o data/output/ekf/grav/analysis -f ekf --geo-type grav
    -uv run analyze geoperformance -p data/output/ekf/mag -d data/output/ekf/degraded -r data/input -o data/output/ekf/mag/analysis -f ekf --geo-type mag
    -uv run analyze geoperformance -p data/output/ekf/both -d data/output/ekf/degraded -r data/input -o data/output/ekf/both/analysis -f ekf --geo-type both
    -uv run analyze geoperformance -p data/output/rbpf/grav -d data/output/rbpf/degraded -r data/input -o data/output/rbpf/grav/analysis/rbpf -f rbpf --geo-type grav
    -uv run analyze geoperformance -p data/output/rbpf/mag -d data/output/rbpf/degraded -r data/input -o data/output/rbpf/mag/analysis/rbpf -f rbpf --geo-type mag
    -uv run analyze geoperformance -p data/output/rbpf/both -d data/output/rbpf/degraded -r data/input -o data/output/rbpf/both/analysis/rbpf -f rbpf --geo-type both
    -uv run analyze geoperformance -p data/output/rbpf/grav -d data/output/ekf/degraded -r data/input -o data/output/rbpf/grav/analysis/ins -f rbpf --geo-type grav
    -uv run analyze geoperformance -p data/output/rbpf/mag -d data/output/ekf/degraded -r data/input -o data/output/rbpf/mag/analysis/ins -f rbpf --geo-type mag
    -uv run analyze geoperformance -p data/output/rbpf/both -d data/output/ekf/degraded -r data/input -o data/output/rbpf/both/analysis/ins -f rbpf --geo-type both

# The RBPF half of `geoperf-all`, for iterating on the particle filter without re-scoring the
# UKF and EKF. These six lines must stay byte-identical to their counterparts above, output
# paths included -- the point of the recipe is that it writes the same artifacts to the same
# places, so it can be run instead of `geoperf-all` rather than as well as it.

# Run the geophysical performance analysis for the RBPF only.
geoperf-rbpf:
    -uv run analyze geoperformance -p data/output/rbpf/grav -d data/output/rbpf/degraded -r data/input -o data/output/rbpf/grav/analysis/rbpf -f rbpf --geo-type grav
    -uv run analyze geoperformance -p data/output/rbpf/mag -d data/output/rbpf/degraded -r data/input -o data/output/rbpf/mag/analysis/rbpf -f rbpf --geo-type mag
    -uv run analyze geoperformance -p data/output/rbpf/both -d data/output/rbpf/degraded -r data/input -o data/output/rbpf/both/analysis/rbpf -f rbpf --geo-type both
    -uv run analyze geoperformance -p data/output/rbpf/grav -d data/output/ekf/degraded -r data/input -o data/output/rbpf/grav/analysis/ins -f rbpf --geo-type grav
    -uv run analyze geoperformance -p data/output/rbpf/mag -d data/output/ekf/degraded -r data/input -o data/output/rbpf/mag/analysis/ins -f rbpf --geo-type mag
    -uv run analyze geoperformance -p data/output/rbpf/both -d data/output/ekf/degraded -r data/input -o data/output/rbpf/both/analysis/ins -f rbpf --geo-type both

# Run the full analysis pipeline for all configurations.
#
# The steps are recipe **dependencies**, listed after the colon. They were body lines, which
# `just` runs as shell commands -- so this recipe tried to execute `build`, `preprocess` and
# the rest as programs and died on the first one with "command not found".
#
# `just` runs each dependency once and in order, so the cleanup below still happens first:
# a recipe's own body runs after its dependencies, which is why the `rm -rf` lines moved into
# their own `clean` recipe rather than staying here.
#
# This is the synthetic arm: `preprocess` synthesises the geophysical readings and conf/*.toml
# describe those sensors, so `geo-stats` runs as a report on them and `geo-adopt` does not run.
# `clean` never touches data/input_real or data/output_real, where the real arm is frozen.
#
# Run the whole experiment end to end, from data/raw to the scored analyses.
pipeline: clean build preprocess geo-stats truth degraded ukf-geo ekf-geo rbpf-sim postprocess geoperf-all

# `data/input_10hz` is included because `preprocess` used to write there. Nothing does now,
# but a checkout that ran the old recipe still has it, and a stale directory full of
# trajectories is the kind of thing that gets simulated by accident.
#
# Remove everything the pipeline regenerates. `data/raw` is read-only and is never touched.
clean:
    rm -rf data/input data/input_10hz data/output log
