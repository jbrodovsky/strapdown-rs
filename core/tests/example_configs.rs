#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    reason = "integration tests are a separate crate target, so `clippy.toml`'s \
              allow-unwrap-in-tests -- which only covers `#[cfg(test)]` items -- does not \
              reach them; unwrapping is how these assert"
)]
//! Every scenario config shipped under `examples/configs/` must deserialize into the scenario
//! it claims to describe.
//!
//! These files are the worked examples the README and the mdBook point users at, and they are
//! the first thing anyone runs. Nothing else in the test suite loaded them, so the names drifted
//! out of sync with [`MeasurementScheduler`] and [`GnssFaultModel`] unnoticed: at the time this test
//! was written eight of them named variants that do not exist (`duty`, `passthrough`,
//! `slowbias`), and failed on the user's machine rather than in CI.
//!
//! Parsing alone is not enough of a check. Every field of [`AidingConfig`] has a
//! default, so a file whose `scheduler` section is misspelled deserializes happily into a
//! pass-through config and silently simulates nothing. The variant assertions below are what
//! make a misspelling fail.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use strapdown::messages::{AidingConfig, GnssFaultModel, MeasurementScheduler};
use strapdown::sim::SimulationConfig;

/// Scenario configs live at the top level of `examples/configs/` and in `conf/`.
///
/// The `json/` subdirectory of `examples/configs/` is deliberately excluded: those files are
/// `{name, args}` CLI invocation presets, not [`AidingConfig`] documents, and they use the
/// *CLI's* vocabulary (`--sched duty`) rather than the config schema's (`kind: duty_cycle`).
fn example_config_paths() -> Vec<PathBuf> {
    let mut found = Vec::new();
    // `conf/` covers the experiment recipes the justfile runs. They were not covered until
    // the geophysical config path landed, and six of them had drifted to a GNSS profile that
    // did not match the degraded run they are scored against -- exactly the class of defect
    // the variant assertions below exist to catch, in the directory that is actually run.
    for directory in ["examples/configs", "conf"] {
        found.extend(config_paths_in(directory));
    }
    found.sort();
    assert!(!found.is_empty(), "no configs found");
    found
}

/// Every parseable config file directly inside one workspace-relative directory.
fn config_paths_in(relative: &str) -> Vec<PathBuf> {
    // `CARGO_MANIFEST_DIR` is the `core/` crate; both directories live at the workspace root.
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("core/ has a parent")
        .join(relative);

    let mut found: Vec<PathBuf> = std::fs::read_dir(&root)
        .unwrap_or_else(|e| panic!("reading {}: {e}", root.display()))
        .map(|entry| entry.expect("directory entry should be readable").path())
        .filter(|path| {
            path.is_file()
                && matches!(
                    path.extension().and_then(|e| e.to_str()),
                    Some("yaml" | "yml" | "json" | "toml")
                )
        })
        .collect();

    found.sort();
    assert!(
        !found.is_empty(),
        "no configs found under {}",
        root.display()
    );
    found
}

/// The scheduler and fault a file's text says it configures.
///
/// Derived from the file rather than from a hardcoded table, so a config added later is
/// covered without editing this test.
fn declared_variants(text: &str) -> (&'static str, &'static str) {
    let text = &strip_comments(text);
    let scheduler = if text.contains("duty_cycle") {
        "DutyCycle"
    } else if text.contains("fixed_interval") {
        "FixedInterval"
    } else {
        "PassThrough"
    };
    let fault = if text.contains("slow_bias") {
        "SlowBias"
    } else if text.contains("hijack") {
        "Hijack"
    } else if text.contains("degraded") {
        "Degraded"
    } else {
        "None"
    };
    (scheduler, fault)
}

/// Drop `#` comments, so the heuristic above reads a file's configuration rather than its prose.
///
/// The `conf/*_truth.toml` recipes carry `kind = "none"` and a comment explaining that their
/// health limits are "kept identical across truth and degraded" -- which made a naive
/// `contains("degraded")` declare a fault the file does not configure. TOML and YAML both
/// comment to end of line with `#`, and JSON has no comments, so one rule covers all three.
/// Quoted `#` is respected: a Windows path or a colour literal in a string is not a comment.
fn strip_comments(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for line in text.lines() {
        let mut in_string = false;
        let mut cut = line.len();
        for (index, character) in line.char_indices() {
            match character {
                '"' => in_string = !in_string,
                '#' if !in_string => {
                    cut = index;
                    break;
                }
                _ => {}
            }
        }
        out.push_str(&line[..cut]);
        out.push('\n');
    }
    out
}

const fn scheduler_variant(scheduler: &MeasurementScheduler) -> &'static str {
    match scheduler {
        MeasurementScheduler::PassThrough => "PassThrough",
        MeasurementScheduler::FixedInterval { .. } => "FixedInterval",
        MeasurementScheduler::DutyCycle { .. } => "DutyCycle",
    }
}

const fn fault_variant(fault: &GnssFaultModel) -> &'static str {
    match fault {
        GnssFaultModel::None => "None",
        GnssFaultModel::Degraded { .. } => "Degraded",
        GnssFaultModel::SlowBias { .. } => "SlowBias",
        GnssFaultModel::Hijack { .. } => "Hijack",
        GnssFaultModel::Combo(_) => "Combo",
    }
}

#[test]
fn every_example_config_deserializes_into_what_it_describes() {
    let mut failures = Vec::new();

    for path in example_config_paths() {
        let name = path.file_name().unwrap_or_default().to_string_lossy();
        let text = std::fs::read_to_string(&path).expect("config should be readable");

        // Parsed as a `SimulationConfig`, which is what `--config` actually feeds them to
        // (`sim/src/main.rs`), NOT as a bare `AidingConfig`. That distinction is the whole
        // point: until the v1.0 freeze this test parsed every file as an `AidingConfig`, so a
        // file that was really a full simulation config -- `geonav_example.toml` -- parsed to
        // all-defaults, declared no scheduler or fault, and passed trivially while being
        // broken six ways (`mode = "ClosedLoop"`, `filter = "Ukf"`, two `"OneMinute"`
        // resolutions, and `type =` where the tag key is `kind`). Running it produced
        // `unknown variant `ClosedLoop``. Parsing as the type the CLI uses is what catches it.
        let config = match SimulationConfig::from_file(&path) {
            Ok(config) => config,
            Err(error) => {
                failures.push(format!("{name}: failed to parse: {error}"));
                continue;
            }
        };

        let (expected_scheduler, expected_fault) = declared_variants(&text);
        let actual_scheduler = scheduler_variant(&config.aiding.scheduler);
        let actual_fault = fault_variant(&config.aiding.fault);

        // `Combo` wraps other faults, so its file mentions their names too; accept it for any
        // declared fault rather than trying to guess which of them is the outer one.
        if actual_scheduler != expected_scheduler {
            failures.push(format!(
                "{name}: names scheduler {expected_scheduler} but parsed as {actual_scheduler} \
                 -- a misspelled variant silently falls back to the default"
            ));
        }
        if actual_fault != expected_fault && actual_fault != "Combo" {
            failures.push(format!(
                "{name}: names fault {expected_fault} but parsed as {actual_fault} \
                 -- a misspelled variant silently falls back to the default"
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "{} example config problem(s):\n  {}",
        failures.len(),
        failures.join("\n  ")
    );
}

/// The `gnss_degradation` alias is the compatibility guarantee, so something must exercise it.
///
/// `SimulationConfig::aiding` was `gnss_degradation` until the v1.0 freeze and carries
/// `#[serde(alias = "gnss_degradation")]` so that configuration files written under the old
/// name keep parsing -- the fifteen recipes under `conf/` among them, which no test loads.
///
/// Until this test, nothing checked it. The two shipped examples that still used the old
/// spelling were migrated to `[aiding]` in the same commit as this test, which would have left
/// the alias entirely uncovered: a future rename could have dropped it and every test would
/// still have passed while every existing user config silently lost its whole aiding section.
///
/// Both spellings must produce the same scenario, and the old one must not be *silently*
/// dropped -- which is the failure mode, since `SimulationConfig` has no
/// `deny_unknown_fields` and `aiding` has a default.
#[test]
fn the_old_gnss_degradation_spelling_still_parses() {
    let old_spelling = r"
mode: closed-loop
gnss_degradation:
  scheduler:
    kind: duty_cycle
    on_s: 30.0
    off_s: 60.0
    start_phase_s: 0.0
  fault:
    kind: slow_bias
    drift_n_mps: 0.5
    drift_e_mps: 0.25
    q_bias: 0.0
    rotate_omega_rps: 0.0
  seed: 7
";

    let config: SimulationConfig =
        serde_yaml::from_str(old_spelling).expect("the `gnss_degradation` alias must still parse");

    assert_eq!(
        scheduler_variant(&config.aiding.scheduler),
        "DutyCycle",
        "the aliased section deserialized into a default pass-through scheduler, which means \
         the alias was dropped and the section silently ignored"
    );
    assert_eq!(
        fault_variant(&config.aiding.fault),
        "SlowBias",
        "the aliased section's fault model was silently dropped"
    );
    assert_eq!(
        config.aiding.seed, 7,
        "the aliased section's seed was silently dropped"
    );

    // And the new spelling gives the same thing, so the alias is not a second code path.
    let new_spelling = old_spelling.replace("gnss_degradation:", "aiding:");
    let renamed: SimulationConfig =
        serde_yaml::from_str(&new_spelling).expect("the `aiding` spelling must parse");
    assert_eq!(
        scheduler_variant(&renamed.aiding.scheduler),
        scheduler_variant(&config.aiding.scheduler),
        "the two spellings must produce the same scheduler"
    );
}

/// A plain `AidingConfig` document, so the type's own defaults are pinned independently of
/// whichever `SimulationConfig` happens to embed it.
#[test]
fn an_empty_aiding_section_schedules_the_other_two_channels() {
    let aiding: AidingConfig =
        serde_yaml::from_str("{}").expect("an empty aiding section must deserialize");

    assert_eq!(
        scheduler_variant(&aiding.baro_scheduler),
        "FixedInterval",
        "the barometer defaults to a 1 Hz fixed interval, not to pass-through"
    );
    assert_eq!(
        scheduler_variant(&aiding.magnetometer_scheduler),
        "FixedInterval",
        "the magnetometer defaults to a 1 Hz fixed interval, not to pass-through"
    );
}

/// Every geophysically-aided recipe must carry the GNSS profile of the run it is scored
/// against.
///
/// `analyze geoperformance` measures a geo-aided run against a non-geo baseline and reports
/// the difference as the contribution of the geophysical measurement. That is only true when
/// the two runs differ in *nothing else*. If their GNSS degradation or their health limits
/// disagree, the "improvement" is the difference between two GNSS profiles wearing the
/// geophysical measurement's name.
///
/// This is not hypothetical. `conf/{ukf,ekf}_{grav,mag,both}.toml` sat in the tree carrying
/// `interval_s = 1.0` with `sigma_pos_m = 15.0`, `sigma_vel_mps = 5.0` and `r_scale = 15.0`,
/// against a `conf/*_degraded.toml` at `5.0 / 3.0 / 0.3 / 5.0` -- six files, every one of
/// them scored against a baseline it did not match, with nothing to say so.
///
/// The pairing is taken from the filename: `<filter>_<geo>.toml` is scored against
/// `<filter>_degraded.toml`, and `<filter>_denied_<geo>.toml` against `<filter>_denied.toml`.
/// Adding a geo recipe therefore enrolls it in this check automatically.
#[test]
fn every_geophysical_config_matches_the_baseline_it_is_scored_against() {
    const FILTERS: [&str; 3] = ["ukf", "ekf", "rbpf"];
    const GEO_TYPES: [&str; 3] = ["grav", "mag", "both"];

    let mut failures = Vec::new();
    let mut checked = 0usize;

    for filter in FILTERS {
        for (suffix, baseline) in [("", "degraded"), ("denied_", "denied")] {
            let baseline_name = format!("{filter}_{baseline}.toml");
            let Some(baseline_config) = load_conf(&baseline_name) else {
                failures.push(format!("{baseline_name}: missing, but geo recipes name it"));
                continue;
            };

            for geo in GEO_TYPES {
                let name = format!("{filter}_{suffix}{geo}.toml");
                let Some(config) = load_conf(&name) else {
                    failures.push(format!("{name}: missing"));
                    continue;
                };
                checked += 1;

                // Compared through `Debug` because neither `AidingConfig` nor `HealthLimits`
                // implements `PartialEq`, and deriving it across `core`'s public API to serve
                // one test is the larger change. The rendering is total, so a difference in
                // any field of either fails this.
                if format!("{:?}", config.aiding) != format!("{:?}", baseline_config.aiding) {
                    failures.push(format!(
                        "{name}: [gnss_degradation] differs from {baseline_name}\n  \
                         geo:      {:?}\n  baseline: {:?}",
                        config.aiding, baseline_config.aiding
                    ));
                }
                if format!("{:?}", config.health_limits)
                    != format!("{:?}", baseline_config.health_limits)
                {
                    failures.push(format!(
                        "{name}: [health_limits] differs from {baseline_name}\n  \
                         geo:      {:?}\n  baseline: {:?}",
                        config.health_limits, baseline_config.health_limits
                    ));
                }
            }
        }
    }

    assert_eq!(
        checked, 18,
        "expected 3 filters x 3 geo types x 2 baselines; found {checked}"
    );
    assert!(
        failures.is_empty(),
        "{} geophysical config(s) do not match their baseline:\n  {}",
        failures.len(),
        failures.join("\n  ")
    );
}

/// Load one `conf/` recipe by file name, or `None` if it is absent.
fn load_conf(name: &str) -> Option<SimulationConfig> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("core/ has a parent")
        .join("conf")
        .join(name);
    path.is_file().then(|| {
        SimulationConfig::from_file(&path).unwrap_or_else(|e| panic!("{name} must parse: {e}"))
    })
}

/// Every geophysical recipe describes the *same sensor*, so it must carry the same numbers.
///
/// `analyze geostats --apply-to conf` measures the bias, the measurement noise and the bias
/// prior of each channel once, from the residual against the maps, and writes them into all
/// eighteen recipes. Those figures characterise the phone's gravimeter and magnetometer
/// against the maps -- not the scenario -- so a UKF run and an RBPF run under a denial
/// profile are looking at an instrument with identical statistics.
///
/// The failure this guards against is a *partial* adoption: one config edited by hand, or
/// `--apply-to` pointed at a directory during a re-run that left a few files behind. The
/// result still parses, still runs, and produces a comparison whose difference is attributed
/// to the filter when it actually came from a different R. That is the same class of silent,
/// research-invalidating drift as the GNSS-block divergence the test above catches.
///
/// Only keys that are *present* are compared: a `*_grav.toml` declares no magnetic channel
/// and must not be forced to.
#[test]
fn every_geophysical_config_describes_the_same_sensor() {
    const FILTERS: [&str; 3] = ["ukf", "ekf", "rbpf"];
    const GEO_TYPES: [&str; 3] = ["grav", "mag", "both"];

    // Field name -> the value seen first, and the config it came from.
    let mut seen: BTreeMap<&'static str, (f64, String)> = BTreeMap::new();
    let mut failures = Vec::new();
    let mut checked = 0usize;

    for filter in FILTERS {
        for prefix in ["", "denied_"] {
            for geo in GEO_TYPES {
                let name = format!("{filter}_{prefix}{geo}.toml");
                let Some(config) = load_conf(&name) else {
                    failures.push(format!("{name}: missing"));
                    continue;
                };
                let Some(geophysical) = config.geophysical else {
                    failures.push(format!("{name}: is a geo recipe with no [geophysical]"));
                    continue;
                };
                checked += 1;

                for (field, value) in [
                    ("gravity_bias", geophysical.gravity_bias),
                    ("gravity_noise_std", geophysical.gravity_noise_std),
                    ("gravity_bias_init_std", geophysical.gravity_bias_init_std),
                    ("magnetic_bias", geophysical.magnetic_bias),
                    ("magnetic_noise_std", geophysical.magnetic_noise_std),
                    ("magnetic_bias_init_std", geophysical.magnetic_bias_init_std),
                    ("geo_interval_s", geophysical.geo_interval_s),
                ] {
                    let Some(value) = value else { continue };
                    match seen.get(field) {
                        None => {
                            seen.insert(field, (value, name.clone()));
                        }
                        Some((first, first_name)) if (first - value).abs() > f64::EPSILON => {
                            failures.push(format!(
                                "{name}: {field} = {value} but {first_name} has {first}. \
                                 Re-run `analyze geostats --apply-to conf` so every recipe \
                                 adopts the measured value, or none does."
                            ));
                        }
                        Some(_) => {}
                    }
                }
            }
        }
    }

    assert_eq!(
        checked, 18,
        "expected 3 filters x 3 geo types x 2 baselines; found {checked}"
    );
    assert!(
        failures.is_empty(),
        "{} geophysical setting(s) disagree across the recipes:\n  {}",
        failures.len(),
        failures.join("\n  ")
    );
}
