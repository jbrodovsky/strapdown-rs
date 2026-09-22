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
ukf-geo:
    -./target/release/strapdown-sim cl --geo \
        -i data/input -o data/output/ukf/both --enu --seed 42 \
        --filter ukf \
        --gravity-resolution one-minute --gravity-noise-std 100.0 \
        --magnetic-resolution two-minutes --magnetic-noise-std 150.0 \
        --geo-interval-s 1.0 \
        --sched fixed --interval-s 60.0 --phase-s 0.0 \
        --fault degraded --rho-pos 0.99 --sigma-pos-m 15.0 \
        --rho-vel 0.95 --sigma-vel-mps 5.0 --r-scale 15.0 \
        --log-level info --log-file log/ukf_both.log
    -./target/release/strapdown-sim cl --geo \
        -i data/input -o data/output/ukf/grav --enu --seed 42 \
        --filter ukf \
        --gravity-resolution one-minute --gravity-noise-std 100.0 \
        --geo-interval-s 1.0 \
        --sched fixed --interval-s 60.0 --phase-s 0.0 \
        --fault degraded --rho-pos 0.99 --sigma-pos-m 15.0 \
        --rho-vel 0.95 --sigma-vel-mps 5.0 --r-scale 15.0 \
        --log-level info --log-file log/ukf_grav.log
    -./target/release/strapdown-sim cl --geo \
        -i data/input -o data/output/ukf/mag --enu --seed 42 \
        --filter ukf \
        --magnetic-resolution two-minutes --magnetic-noise-std 150.0 \
        --geo-interval-s 1.0 \
        --sched fixed --interval-s 60.0 --phase-s 0.0 \
        --fault degraded --rho-pos 0.99 --sigma-pos-m 15.0 \
        --rho-vel 0.95 --sigma-vel-mps 5.0 --r-scale 15.0 \
        --log-level info --log-file log/ukf_mag.log

ekf-geo:
    -./target/release/strapdown-sim cl --geo \
        -i data/input -o data/output/ekf/both --enu --seed 42 \
        --filter ekf \
        --gravity-resolution one-minute --gravity-noise-std 100.0 \
        --magnetic-resolution two-minutes --magnetic-noise-std 150.0 \
        --geo-interval-s 1.0 \
        --sched fixed --interval-s 60.0 --phase-s 0.0 \
        --fault degraded --rho-pos 0.99 --sigma-pos-m 15.0 \
        --rho-vel 0.95 --sigma-vel-mps 5.0 --r-scale 15.0 \
        --log-level info --log-file log/ekf_both.log
    -./target/release/strapdown-sim cl --geo \
        -i data/input -o data/output/ekf/grav --enu --seed 42 \
        --filter ekf \
        --gravity-resolution one-minute --gravity-noise-std 100.0 \
        --geo-interval-s 1.0 \
        --log-level info --log-file log/ekf_grav.log
    -./target/release/strapdown-sim cl --geo \
        -i data/input -o data/output/ekf/mag --enu --seed 42 \
        --filter ekf \
        --magnetic-resolution two-minutes --magnetic-noise-std 150.0 \
        --geo-interval-s 1.0 \
        --log-level info --log-file log/ekf_mag.log


