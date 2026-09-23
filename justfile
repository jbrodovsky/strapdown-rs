# Install both halves of the monorepo: the Rust toolchain fetches itself on the first cargo
# command, and `uv sync` builds the `analysis` workspace member from the root pyproject.toml.
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

# Rebuild data/input from the Sensor Logger exports in data/raw.
# preprocess:
#    uv run analyze preprocess -i data/raw -o data/input -f 1 \
#        --max-imu-gap-s 5.0 --min-segment-s 300.0 --prune

# The 10 Hz variant. GNSS is only ever recorded at ~1 Hz, so the extra rows carry inertial
# data only and the GNSS columns stay NaN in 9 bins out of 10 -- which is what gives 10 Hz
# propagation against 1 Hz aiding. Do not interpolate GNSS up to match.

# Rebuild data/input_10hz from the Sensor Logger exports in data/raw.
preprocess:
    uv run analyze preprocess -i data/raw -o data/input_10hz -f 10 \
        -b 10 --max-imu-gap-s 5.0 --min-segment-s 300.0 --prune

# Characterise the geophysical measurements against the maps they are matched against.
#
# Reads data/input -- the *output* of `preprocess`, not data/raw -- because it needs the
# split, gap-trimmed trajectories and the `_gravity.nc` / `_magnetic.nc` that preprocessing
# writes beside each one. It is a separate recipe rather than a step inside `preprocess` for
# that reason: it consumes what preprocessing produces.
#
# Writes the per-record residuals, a per-trajectory summary, a paste-ready `[geophysical]`
# config block and the two-panel histogram figure. The noise standard deviations in
# conf/*.toml should come from its geo_stats.toml, not from the 100 mGal / 150 nT defaults,
# which were never measured against anything.
geo-stats:
    uv run analyze geostats -i data/input -o data/output/geostats

# Run the truth configurations.
truth:
    -./target/release/strapdown-sim --config conf/ukf_truth.toml
    -./target/release/strapdown-sim --config conf/ekf_truth.toml

# Run the degraded configurations.
degraded:
    -./target/release/strapdown-sim --config conf/ukf_degraded.toml
    -./target/release/strapdown-sim --config conf/ekf_degraded.toml

# Intermittent GNSS denial: 30 s of fixes in every 150, no fault on the fixes that arrive.
# The tier that resembles jamming -- a receiver inside a jammed area reports nothing, it does
# not report a noisier position. Sweep `off_s` in the configs to trace how an aid's
# contribution grows with outage length. See book/src/gnss/fault-simulation.md.
denied:
    -./target/release/strapdown-sim --config conf/ukf_denied.toml
    -./target/release/strapdown-sim --config conf/ekf_denied.toml
    -./target/release/strapdown-sim --config conf/rbpf_denied.toml

# The recalibrated accuracy tier: 35 m steady-state wander with the correlation time pinned
# in seconds, so the fix interval can be swept without moving the error model. Kept separate
# from `degraded`, which is the baseline every result so far was measured against.
jammed:
    -./target/release/strapdown-sim --config conf/ukf_jammed.toml
    -./target/release/strapdown-sim --config conf/ekf_jammed.toml
    -./target/release/strapdown-sim --config conf/rbpf_jammed.toml

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
ukf-geo:
    -./target/release/strapdown-sim --config conf/ukf_both.toml
    -./target/release/strapdown-sim --config conf/ukf_grav.toml
    -./target/release/strapdown-sim --config conf/ukf_mag.toml

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

# Postprocess the performance of all scenarios.
postprocess:
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

# Run the geophysical performance analysis for all filter types.
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
pipeline:
    rm -rf data/input
    rm -rf data/output/
    rm -rf log
    build
    preprocess
    truth
    degraded
    ukf-geo
    ekf-geo
    rbpf-sim
    postprocess
    geoperf-all
    geoperf-rbpf