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

use std::path::{Path, PathBuf};

use strapdown::messages::{AidingConfig, GnssFaultModel, MeasurementScheduler};
use strapdown::sim::SimulationConfig;

/// Scenario configs live at the top level of `examples/configs/`.
///
/// The `json/` subdirectory is deliberately excluded: those files are `{name, args}` CLI
/// invocation presets, not [`AidingConfig`] documents, and they use the *CLI's*
/// vocabulary (`--sched duty`) rather than the config schema's (`kind: duty_cycle`).
fn example_config_paths() -> Vec<PathBuf> {
    // `CARGO_MANIFEST_DIR` is the `core/` crate; the examples live at the workspace root.
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("core/ has a parent")
        .join("examples/configs");

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
        "no example configs found under {}",
        root.display()
    );
    found
}

/// The scheduler and fault a file's text says it configures.
///
/// Derived from the file rather than from a hardcoded table, so a config added later is
/// covered without editing this test.
fn declared_variants(text: &str) -> (&'static str, &'static str) {
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
