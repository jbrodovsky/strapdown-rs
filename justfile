# Build the project in release mode
build:
    cargo build --release --workspace --all-features

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
#    uv run --project analysis analyze preprocess -i data/raw -o data/input -f 1 \
#        --max-imu-gap-s 5.0 --min-segment-s 300.0 --prune

# The 10 Hz variant. GNSS is only ever recorded at ~1 Hz, so the extra rows carry inertial
# data only and the GNSS columns stay NaN in 9 bins out of 10 -- which is what gives 10 Hz
# propagation against 1 Hz aiding. Do not interpolate GNSS up to match.

# Rebuild data/input_10hz from the Sensor Logger exports in data/raw.
preprocess:
    uv run --project analysis analyze preprocess -i data/raw -o data/input_10hz -f 10 \
        -b 10 --max-imu-gap-s 5.0 --min-segment-s 300.0 --prune

# Run the truth configurations.
truth:
    -./target/release/strapdown-sim --config conf/ukf_truth.toml
    -./target/release/strapdown-sim --config conf/ekf_truth.toml

# Run the degraded configurations.
degraded:
    -./target/release/strapdown-sim --config conf/ukf_degraded.toml
    -./target/release/strapdown-sim --config conf/ekf_degraded.toml

# `--geo` is only reachable from the `cl` subcommand's own flags, never from a `--config`
# file: a closed-loop config with a [geophysical] section is refused on purpose (#296 /
# see main.rs process_file), so these recipes can't just point at conf/*.toml like `truth`
# and `degraded` do. The flags below are the CLI equivalent of the retired
# conf/{ukf,ekf}_{both,grav,mag}.toml files.
#
# The --sched/--fault flags below MUST stay in step with conf/*_degraded.toml. `geoperf-all`
# scores each geo-aided run against that filter's degraded run, so if the two carry different
# GNSS profiles the "improvement" it reports is just the difference between the two
# degradations, not the contribution of the geophysical measurement.
#
# So must --nis-pos-max/--health-speed-mps-max, for a less obvious reason. These flags do not
# change the trajectory, only whether a run is allowed to finish -- but `geoperformance` skips
# any trajectory missing from either side, so a guard that fails here and not in the degraded
# run silently drops that trajectory from the comparison. Left at their defaults these six
# runs scored 12-27 trajectories each while the degraded baselines scored all 27, which means
# every config's "improvement" was an average over a different subset.

# Run the UKF geophysical configurations.
ukf-geo:
    -./target/release/strapdown-sim cl --geo \
        -i data/input -o data/output/ukf/both --enu --seed 42 \
        --filter ukf \
        --gravity-resolution one-minute --gravity-noise-std 100.0 \
        --magnetic-resolution two-minutes --magnetic-noise-std 150.0 \
        --geo-interval-s 1.0 \
        --sched fixed --interval-s 5.0 --phase-s 0.0 \
        --fault degraded --rho-pos 0.99 --sigma-pos-m 3.0 \
        --rho-vel 0.95 --sigma-vel-mps 0.3 --r-scale 5.0 \
        --nis-pos-max 1000.0 --health-speed-mps-max 5000.0 \
        --log-level info --log-file log/ukf_both.log
    -./target/release/strapdown-sim cl --geo \
        -i data/input -o data/output/ukf/grav --enu --seed 42 \
        --filter ukf \
        --gravity-resolution one-minute --gravity-noise-std 100.0 \
        --geo-interval-s 1.0 \
        --sched fixed --interval-s 5.0 --phase-s 0.0 \
        --fault degraded --rho-pos 0.99 --sigma-pos-m 3.0 \
        --rho-vel 0.95 --sigma-vel-mps 0.3 --r-scale 5.0 \
        --nis-pos-max 1000.0 --health-speed-mps-max 5000.0 \
        --log-level info --log-file log/ukf_grav.log
    -./target/release/strapdown-sim cl --geo \
        -i data/input -o data/output/ukf/mag --enu --seed 42 \
        --filter ukf \
        --magnetic-resolution two-minutes --magnetic-noise-std 150.0 \
        --geo-interval-s 1.0 \
        --sched fixed --interval-s 5.0 --phase-s 0.0 \
        --fault degraded --rho-pos 0.99 --sigma-pos-m 3.0 \
        --rho-vel 0.95 --sigma-vel-mps 0.3 --r-scale 5.0 \
        --nis-pos-max 1000.0 --health-speed-mps-max 5000.0 \
        --log-level info --log-file log/ukf_mag.log

# Run the EKF geophysical configurations.
ekf-geo:
    -./target/release/strapdown-sim cl --geo \
        -i data/input -o data/output/ekf/both --enu --seed 42 \
        --filter ekf \
        --gravity-resolution one-minute --gravity-noise-std 100.0 \
        --magnetic-resolution two-minutes --magnetic-noise-std 150.0 \
        --geo-interval-s 1.0 \
        --sched fixed --interval-s 5.0 --phase-s 0.0 \
        --fault degraded --rho-pos 0.99 --sigma-pos-m 3.0 \
        --rho-vel 0.95 --sigma-vel-mps 0.3 --r-scale 5.0 \
        --nis-pos-max 1000.0 --health-speed-mps-max 5000.0 \
        --log-level info --log-file log/ekf_both.log
    -./target/release/strapdown-sim cl --geo \
        -i data/input -o data/output/ekf/grav --enu --seed 42 \
        --filter ekf \
        --gravity-resolution one-minute --gravity-noise-std 100.0 \
        --geo-interval-s 1.0 \
        --sched fixed --interval-s 5.0 --phase-s 0.0 \
        --fault degraded --rho-pos 0.99 --sigma-pos-m 3.0 \
        --rho-vel 0.95 --sigma-vel-mps 0.3 --r-scale 5.0 \
        --nis-pos-max 1000.0 --health-speed-mps-max 5000.0 \
        --log-level info --log-file log/ekf_grav.log
    -./target/release/strapdown-sim cl --geo \
        -i data/input -o data/output/ekf/mag --enu --seed 42 \
        --filter ekf \
        --magnetic-resolution two-minutes --magnetic-noise-std 150.0 \
        --geo-interval-s 1.0 \
        --sched fixed --interval-s 5.0 --phase-s 0.0 \
        --fault degraded --rho-pos 0.99 --sigma-pos-m 3.0 \
        --rho-vel 0.95 --sigma-vel-mps 0.3 --r-scale 5.0 \
        --nis-pos-max 1000.0 --health-speed-mps-max 5000.0 \
        --log-level info --log-file log/ekf_mag.log

# RBPF sims (truth, degraded, and each geophysical aiding combination). Unlike ukf-geo/ekf-geo,
# these run entirely from conf/rbpf_*.toml: RBPF is the one filter whose config-driven
# `mode = "particle-filter"` path is allowed to carry a [geophysical] section (main.rs only
# refuses it for `mode = "closed-loop"`), so `--geo` CLI flags aren't needed here.

# Run the RBPF configurations.
rbpf-sim:
    -./target/release/strapdown-sim --config conf/rbpf_truth.toml
    -./target/release/strapdown-sim --config conf/rbpf_degraded.toml
    -./target/release/strapdown-sim --config conf/rbpf_grav.toml
    -./target/release/strapdown-sim --config conf/rbpf_mag.toml
    -./target/release/strapdown-sim --config conf/rbpf_both.toml

# Performance postprocessing for all scenarios. `analyze` is the analysis/ package's CLI --
# gitignored local tooling, not part of this Rust-only repo (see CLAUDE.md) -- so it's run via
# `uv run --project analysis` rather than assumed to be on PATH.

# Postprocess the performance of all scenarios.
postprocess:
    -uv run --project analysis analyze performance --processed data/output/ukf/truth --reference data/input/ --output data/output/ukf/truth/performance
    -uv run --project analysis analyze performance --processed data/output/ukf/degraded --reference data/input/ --output data/output/ukf/degraded/performance
    -uv run --project analysis analyze performance --processed data/output/ukf/both --reference data/input/ --output data/output/ukf/both/performance
    -uv run --project analysis analyze performance --processed data/output/ukf/grav --reference data/input/ --output data/output/ukf/grav/performance
    -uv run --project analysis analyze performance --processed data/output/ukf/mag --reference data/input/ --output data/output/ukf/mag/performance
    -uv run --project analysis analyze performance --processed data/output/ekf/truth --reference data/input/ --output data/output/ekf/truth/performance
    -uv run --project analysis analyze performance --processed data/output/ekf/degraded --reference data/input/ --output data/output/ekf/degraded/performance
    -uv run --project analysis analyze performance --processed data/output/ekf/both --reference data/input/ --output data/output/ekf/both/performance
    -uv run --project analysis analyze performance --processed data/output/ekf/grav --reference data/input/ --output data/output/ekf/grav/performance
    -uv run --project analysis analyze performance --processed data/output/ekf/mag --reference data/input/ --output data/output/ekf/mag/performance
    -uv run --project analysis analyze performance --processed data/output/rbpf/truth --reference data/input/ --output data/output/rbpf/truth/performance
    -uv run --project analysis analyze performance --processed data/output/rbpf/degraded --reference data/input/ --output data/output/rbpf/degraded/performance
    -uv run --project analysis analyze performance --processed data/output/rbpf/both --reference data/input/ --output data/output/rbpf/both/performance
    -uv run --project analysis analyze performance --processed data/output/rbpf/grav --reference data/input/ --output data/output/rbpf/grav/performance
    -uv run --project analysis analyze performance --processed data/output/rbpf/mag --reference data/input/ --output data/output/rbpf/mag/performance

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
    -uv run --project analysis analyze geoperformance -p data/output/ukf/grav -d data/output/ukf/degraded -r data/input -o data/output/ukf/grav/analysis -f ukf --geo-type grav
    -uv run --project analysis analyze geoperformance -p data/output/ukf/mag -d data/output/ukf/degraded -r data/input -o data/output/ukf/mag/analysis -f ukf --geo-type mag
    -uv run --project analysis analyze geoperformance -p data/output/ukf/both -d data/output/ukf/degraded -r data/input -o data/output/ukf/both/analysis -f ukf --geo-type both
    -uv run --project analysis analyze geoperformance -p data/output/ekf/grav -d data/output/ekf/degraded -r data/input -o data/output/ekf/grav/analysis -f ekf --geo-type grav
    -uv run --project analysis analyze geoperformance -p data/output/ekf/mag -d data/output/ekf/degraded -r data/input -o data/output/ekf/mag/analysis -f ekf --geo-type mag
    -uv run --project analysis analyze geoperformance -p data/output/ekf/both -d data/output/ekf/degraded -r data/input -o data/output/ekf/both/analysis -f ekf --geo-type both
    -uv run --project analysis analyze geoperformance -p data/output/rbpf/grav -d data/output/rbpf/degraded -r data/input -o data/output/rbpf/grav/analysis/rbpf -f rbpf --geo-type grav
    -uv run --project analysis analyze geoperformance -p data/output/rbpf/mag -d data/output/rbpf/degraded -r data/input -o data/output/rbpf/mag/analysis/rbpf -f rbpf --geo-type mag
    -uv run --project analysis analyze geoperformance -p data/output/rbpf/both -d data/output/rbpf/degraded -r data/input -o data/output/rbpf/both/analysis/rbpf -f rbpf --geo-type both
    -uv run --project analysis analyze geoperformance -p data/output/rbpf/grav -d data/output/ekf/degraded -r data/input -o data/output/rbpf/grav/analysis/ins -f rbpf --geo-type grav
    -uv run --project analysis analyze geoperformance -p data/output/rbpf/mag -d data/output/ekf/degraded -r data/input -o data/output/rbpf/mag/analysis/ins -f rbpf --geo-type mag
    -uv run --project analysis analyze geoperformance -p data/output/rbpf/both -d data/output/ekf/degraded -r data/input -o data/output/rbpf/both/analysis/ins -f rbpf --geo-type both

# The RBPF half of `geoperf-all`, for iterating on the particle filter without re-scoring the
# UKF and EKF. These six lines must stay byte-identical to their counterparts above, output
# paths included -- the point of the recipe is that it writes the same artifacts to the same
# places, so it can be run instead of `geoperf-all` rather than as well as it.

# Run the geophysical performance analysis for the RBPF only.
geoperf-rbpf:
    -uv run --project analysis analyze geoperformance -p data/output/rbpf/grav -d data/output/rbpf/degraded -r data/input -o data/output/rbpf/grav/analysis/rbpf -f rbpf --geo-type grav
    -uv run --project analysis analyze geoperformance -p data/output/rbpf/mag -d data/output/rbpf/degraded -r data/input -o data/output/rbpf/mag/analysis/rbpf -f rbpf --geo-type mag
    -uv run --project analysis analyze geoperformance -p data/output/rbpf/both -d data/output/rbpf/degraded -r data/input -o data/output/rbpf/both/analysis/rbpf -f rbpf --geo-type both
    -uv run --project analysis analyze geoperformance -p data/output/rbpf/grav -d data/output/ekf/degraded -r data/input -o data/output/rbpf/grav/analysis/ins -f rbpf --geo-type grav
    -uv run --project analysis analyze geoperformance -p data/output/rbpf/mag -d data/output/ekf/degraded -r data/input -o data/output/rbpf/mag/analysis/ins -f rbpf --geo-type mag
    -uv run --project analysis analyze geoperformance -p data/output/rbpf/both -d data/output/ekf/degraded -r data/input -o data/output/rbpf/both/analysis/ins -f rbpf --geo-type both


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