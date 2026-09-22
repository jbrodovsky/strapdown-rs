build:
    cargo build --release --workspace --all-features

truth:
    -./target/release/strapdown-sim --config conf/ukf_truth.toml
    -./target/release/strapdown-sim --config conf/ekf_truth.toml
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
        --log-level info --log-file log/ukf_both.log
    -./target/release/strapdown-sim cl --geo \
        -i data/input -o data/output/ukf/grav --enu --seed 42 \
        --filter ukf \
        --gravity-resolution one-minute --gravity-noise-std 100.0 \
        --geo-interval-s 1.0 \
        --sched fixed --interval-s 5.0 --phase-s 0.0 \
        --fault degraded --rho-pos 0.99 --sigma-pos-m 3.0 \
        --rho-vel 0.95 --sigma-vel-mps 0.3 --r-scale 5.0 \
        --log-level info --log-file log/ukf_grav.log
    -./target/release/strapdown-sim cl --geo \
        -i data/input -o data/output/ukf/mag --enu --seed 42 \
        --filter ukf \
        --magnetic-resolution two-minutes --magnetic-noise-std 150.0 \
        --geo-interval-s 1.0 \
        --sched fixed --interval-s 5.0 --phase-s 0.0 \
        --fault degraded --rho-pos 0.99 --sigma-pos-m 3.0 \
        --rho-vel 0.95 --sigma-vel-mps 0.3 --r-scale 5.0 \
        --log-level info --log-file log/ukf_mag.log

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
        --log-level info --log-file log/ekf_both.log
    -./target/release/strapdown-sim cl --geo \
        -i data/input -o data/output/ekf/grav --enu --seed 42 \
        --filter ekf \
        --gravity-resolution one-minute --gravity-noise-std 100.0 \
        --geo-interval-s 1.0 \
        --sched fixed --interval-s 5.0 --phase-s 0.0 \
        --fault degraded --rho-pos 0.99 --sigma-pos-m 3.0 \
        --rho-vel 0.95 --sigma-vel-mps 0.3 --r-scale 5.0 \
        --log-level info --log-file log/ekf_grav.log
    -./target/release/strapdown-sim cl --geo \
        -i data/input -o data/output/ekf/mag --enu --seed 42 \
        --filter ekf \
        --magnetic-resolution two-minutes --magnetic-noise-std 150.0 \
        --geo-interval-s 1.0 \
        --sched fixed --interval-s 5.0 --phase-s 0.0 \
        --fault degraded --rho-pos 0.99 --sigma-pos-m 3.0 \
        --rho-vel 0.95 --sigma-vel-mps 0.3 --r-scale 5.0 \
        --log-level info --log-file log/ekf_mag.log

# RBPF sims (truth, degraded, and each geophysical aiding combination). Unlike ukf-geo/ekf-geo,
# these run entirely from conf/rbpf_*.toml: RBPF is the one filter whose config-driven
# `mode = "particle-filter"` path is allowed to carry a [geophysical] section (main.rs only
# refuses it for `mode = "closed-loop"`), so `--geo` CLI flags aren't needed here.
rbpf-sim:
    -./target/release/strapdown-sim --config conf/rbpf_truth.toml
    -./target/release/strapdown-sim --config conf/rbpf_degraded.toml
    -./target/release/strapdown-sim --config conf/rbpf_grav.toml
    -./target/release/strapdown-sim --config conf/rbpf_mag.toml
    -./target/release/strapdown-sim --config conf/rbpf_both.toml

# Performance postprocessing for all scenarios. `analyze` is the analysis/ package's CLI --
# gitignored local tooling, not part of this Rust-only repo (see CLAUDE.md) -- so it's run via
# `uv run --project analysis` rather than assumed to be on PATH.
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

# Geophysical performance analysis for all filter types: each filter compared against its own
# degraded run, plus RBPF compared against the UKF degraded run as an RBPF-vs-INS baseline.
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
    -uv run --project analysis analyze geoperformance -p data/output/rbpf/grav -d data/output/ukf/degraded -r data/input -o data/output/rbpf/grav/analysis/ins -f rbpf --geo-type grav
    -uv run --project analysis analyze geoperformance -p data/output/rbpf/mag -d data/output/ukf/degraded -r data/input -o data/output/rbpf/mag/analysis/ins -f rbpf --geo-type mag
    -uv run --project analysis analyze geoperformance -p data/output/rbpf/both -d data/output/ukf/degraded -r data/input -o data/output/rbpf/both/analysis/ins -f rbpf --geo-type both

# Geophysical performance analysis for RBPF only.
geoperf-rbpf:
    -uv run --project analysis analyze geoperformance -p data/output/rbpf/grav -d data/output/rbpf/degraded -r data/input -o data/output/rbpf/grav/analysis -f rbpf --geo-type grav
    -uv run --project analysis analyze geoperformance -p data/output/rbpf/mag -d data/output/rbpf/degraded -r data/input -o data/output/rbpf/mag/analysis -f rbpf --geo-type mag
    -uv run --project analysis analyze geoperformance -p data/output/rbpf/both -d data/output/rbpf/degraded -r data/input -o data/output/rbpf/both/analysis -f rbpf --geo-type both


