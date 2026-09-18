build:
    cargo build --release --workspace --all-features

truth:
    ./target/release/strapdown-sim --config conf/ukf_truth.toml;
    ./target/release/strapdown-sim --config conf/ekf_truth.toml;
degraded:
    ./target/release/strapdown-sim --config conf/ukf_degraded.toml;
    ./target/release/strapdown-sim --config conf/ekf_degraded.toml;

ukf-geo:
    ./target/release/strapdown-sim --config conf/ukf_both.toml;
    ./target/release/strapdown-sim --config conf/ukf_grav.toml;
    ./target/release/strapdown-sim --config conf/ukf_mag.toml;

ekf_geo:
    ./target/release/strapdown-sim --config conf/ekf_both.toml;
    ./target/release/strapdown-sim --config conf/ekf_grav.toml;
    ./target/release/strapdown-sim --config conf/ekf_mag.toml;


