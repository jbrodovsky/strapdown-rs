//! Every config type's `Default` impl and its serde defaults must agree.
//!
//! `#[serde(default)]` on a *field* fills it from `<FieldType as Default>::default()`, which
//! has nothing to do with the struct's own `impl Default`. So a hand-written `Default` that
//! sets a field to anything other than its type's default -- `true` for a `bool`, a non-zero
//! number -- silently disagrees with what a config file omitting that field produces.
//!
//! That is not a theoretical hazard. `ClosedLoopConfig::estimate_baro_bias` shipped exactly
//! this way: `Default::default()` gave `true` while a `[closed_loop]` section that omitted the
//! field gave `false`, so the Rust API and the config file -- the primary interface for this
//! tool -- disagreed about whether the barometric bias state was estimated at all. The gated
//! accuracy baseline measures the Rust path, so the published tables described behaviour that
//! config-file users were not getting.
//!
//! The rule this file enforces: **deserializing a document that sets nothing must produce
//! `Default::default()`.** The usual way to satisfy it is `#[serde(default)]` on the
//! *container*, which makes the hand-written `Default` the single source of truth for every
//! missing field at once.
//!
//! Compared through `serde_json::Value` rather than `PartialEq`, because these types do not
//! derive it and the value diff names the offending field.

use strapdown::NavigationFilter;
use strapdown::engine::InsEngineConfig;
use strapdown::messages::AidingConfig;
use strapdown::sim::{
    ClosedLoopConfig, ExecutionLimits, HealthLimits, LoggingConfig, ParticleFilterConfig,
    SimulationConfig, SyntheticConfig, SyntheticInitialState,
};

/// Deserialize `$t` from a document that sets nothing beyond what it must, and require the
/// result to serialize identically to `<$t>::default()`.
///
/// A macro rather than a generic function so that the failure paths expand *inside* each
/// `#[test]` function: `clippy.toml` sets `allow-panic-in-tests`, but clippy only applies that
/// to code it can see is a test, and a generic helper called by a test is not.
macro_rules! assert_document_matches_default {
    ($t:ty, $document:expr) => {{
        let document: &str = $document;
        let name = std::any::type_name::<$t>();

        let from_document: $t = serde_json::from_str(document)
            .unwrap_or_else(|e| panic!("{name} could not be deserialized from `{document}`: {e}"));

        let document_value = serde_json::to_value(&from_document)
            .unwrap_or_else(|e| panic!("{name} is not serializable: {e}"));
        let default_value = serde_json::to_value(<$t>::default())
            .unwrap_or_else(|e| panic!("{name} is not serializable: {e}"));

        assert_eq!(
            document_value, default_value,
            "{name}: a config document that sets nothing does not produce `Default::default()`.\n\
             A field-level `#[serde(default)]` fills from the FIELD TYPE's default, not this \
             struct's `Default` impl. Put `#[serde(default)]` on the container so the \
             hand-written `Default` is the single source of truth.\n\
             left = from `{document}`, right = `Default::default()`"
        );
    }};
}

#[test]
fn closed_loop_config_serde_defaults_match_its_default_impl() {
    assert_document_matches_default!(ClosedLoopConfig, "{}");
}

#[test]
fn particle_filter_config_serde_defaults_match_its_default_impl() {
    assert_document_matches_default!(ParticleFilterConfig, "{}");
}

#[test]
fn logging_config_serde_defaults_match_its_default_impl() {
    assert_document_matches_default!(LoggingConfig, "{}");
}

#[test]
fn execution_limits_serde_defaults_match_its_default_impl() {
    assert_document_matches_default!(ExecutionLimits, "{}");
}

#[test]
fn health_limits_serde_defaults_match_its_default_impl() {
    assert_document_matches_default!(HealthLimits, "{}");
}

/// The way this knob is actually reached for: loosen one guard, inherit the rest.
///
/// Each field carries its own `#[serde(default = "..."]`, so a section naming only
/// `nis_pos_max` must leave the others at their documented defaults. Without those per-field
/// defaults the untouched tuple bands would deserialize to `(0.0, 0.0)` and fail every
/// estimate on the first update -- a far more confusing failure than the one being relaxed.
#[test]
fn a_partial_health_limits_section_keeps_every_other_guard_at_its_default() {
    let loosened: HealthLimits = serde_json::from_str(r#"{"nis_pos_max": 100000.0}"#)
        .expect("a health_limits section naming one field should deserialize");
    let default = HealthLimits::default();

    assert!(
        (loosened.nis_pos_max - 100_000.0).abs() < f64::EPSILON,
        "the field that was set should take the document's value, got {}",
        loosened.nis_pos_max
    );
    assert_eq!(
        loosened.lat_rad, default.lat_rad,
        "lat_rad was not inherited"
    );
    assert_eq!(
        loosened.lon_rad, default.lon_rad,
        "lon_rad was not inherited"
    );
    assert_eq!(loosened.alt_m, default.alt_m, "alt_m was not inherited");
    assert!(
        (loosened.speed_mps_max - default.speed_mps_max).abs() < f64::EPSILON,
        "speed_mps_max was not inherited"
    );
    assert!(
        (loosened.cov_diag_max - default.cov_diag_max).abs() < f64::EPSILON,
        "cov_diag_max was not inherited"
    );
    assert_eq!(
        loosened.nis_pos_consec_fail, default.nis_pos_consec_fail,
        "nis_pos_consec_fail was not inherited"
    );
}

/// `latitude_deg`/`longitude_deg`/`altitude_m` carry no serde default on purpose: a synthetic
/// trajectory silently starting at the Gulf of Guinea is a worse failure than a missing-field
/// error. The document therefore sets exactly those three, to the values `Default` uses, and
/// every remaining field must still agree.
#[test]
fn synthetic_initial_state_serde_defaults_match_its_default_impl() {
    assert_document_matches_default!(
        SyntheticInitialState,
        r#"{"latitude_deg":0.0,"longitude_deg":0.0,"altitude_m":0.0}"#
    );
}

/// `output` and `duration_s` are the two fields serde treats as required, as
/// `SyntheticConfig`'s `Default` doc records. The document sets them to the values that
/// `Default` supplies, so the comparison still covers every other field.
#[test]
fn synthetic_config_serde_defaults_match_its_default_impl() {
    assert_document_matches_default!(SyntheticConfig, r#"{"output":"","duration_s":300.0}"#);
}

#[test]
fn aiding_config_serde_defaults_match_its_default_impl() {
    assert_document_matches_default!(AidingConfig, "{}");
}

#[test]
fn ins_engine_config_serde_defaults_match_its_default_impl() {
    assert_document_matches_default!(InsEngineConfig, "{}");
}

/// `SimulationConfig::mode` is deliberately the one non-defaulted field in the whole config
/// tree -- it is what makes a bare aiding fragment fail loudly instead of deserializing into a
/// pass-through config that simulates nothing. So the minimal document sets it, and
/// `SimulationConfig::default()` must agree with that document on everything else.
///
/// The optional sub-configs are compared as the one consumer reads them. `sim/src/main.rs`
/// takes `config.closed_loop.clone().unwrap_or_default()`, so an absent `[closed_loop]`
/// section and one holding every default are the same run; `Default` keeps `Some(..)` because
/// the `config` subcommand serialises it into the template, where a visible section is what
/// tells a user the knobs exist.
#[test]
fn simulation_config_serde_defaults_match_its_default_impl() {
    let from_document: SimulationConfig = serde_json::from_str(r#"{"mode":"closed-loop"}"#)
        .expect("a document setting only `mode` must deserialize");

    let mut document_value =
        serde_json::to_value(&from_document).expect("SimulationConfig is serializable");
    let default_value = serde_json::to_value(SimulationConfig::default())
        .expect("SimulationConfig is serializable");

    // `None` and `Some(ClosedLoopConfig::default())` are the same run; normalise the absent
    // section to the defaults the consumer would substitute, then require exact agreement.
    if document_value
        .get("closed_loop")
        .is_none_or(serde_json::Value::is_null)
    {
        let defaults = serde_json::to_value(ClosedLoopConfig::default())
            .expect("ClosedLoopConfig is serializable");
        document_value["closed_loop"] = defaults;
    }

    assert_eq!(
        document_value, default_value,
        "SimulationConfig: a document setting only `mode` does not produce `Default::default()`"
    );
}

/// The particle filter's two retired map-bias keys are refused by name, not dropped.
///
/// `[particle_filter] geo_bias_init_std` and `geo_bias_process_noise_std` set one prior and one
/// random walk for every map bias, in no particular unit, while the particle filter ignored the
/// per-channel `[geophysical]` keys the Kalman filters read. It reads those now. Serde drops a
/// key it does not recognise without a word, so without an explicit refusal a config written
/// for the old keys would run on the new defaults and never say so. The error must name the
/// key and where its replacement lives, through the file format a user actually writes.
#[test]
fn the_particle_filters_retired_map_bias_keys_are_refused_by_name() {
    for (key, replacement) in [
        ("geo_bias_init_std", "gravity_bias_init_std"),
        (
            "geo_bias_process_noise_std",
            "gravity_bias_process_noise_std",
        ),
    ] {
        let section = format!(r#"{{"num_particles": 10, "{key}": 1.0}}"#);
        let error = serde_json::from_str::<ParticleFilterConfig>(&section)
            .expect_err("a retired key must be refused, not dropped")
            .to_string();
        assert!(
            error.contains(key) && error.contains(replacement) && error.contains("[geophysical]"),
            "the JSON error must name `{key}` and point at `{replacement}` in [geophysical], \
             got: {error}"
        );

        let document = format!(
            "mode = \"particle-filter\"\n\n[particle_filter]\nnum_particles = 10\n{key} = 1.0\n"
        );
        let error = toml::from_str::<SimulationConfig>(&document)
            .expect_err("a retired key must be refused in a TOML scenario file too")
            .to_string();
        assert!(
            error.contains(key) && error.contains(replacement),
            "the TOML error must name `{key}` and point at `{replacement}`, got: {error}"
        );
    }

    // A section that does not mention them is unaffected.
    let parsed: ParticleFilterConfig = serde_json::from_str(r#"{"num_particles": 10}"#)
        .expect("a section without the retired keys must parse");
    assert_eq!(parsed.num_particles, 10);
}

/// The keys the Canciani & Raquet restructure retired are refused by name as well.
///
/// Position no longer has process noise of its own (eq. 19) and the barometer loop in the
/// mechanization replaced the zero-vertical-velocity constraint. A config still setting them
/// would otherwise run a different filter than it describes without a word.
#[test]
fn the_particle_filters_restructure_retired_keys_are_refused_by_name() {
    for (setting, key, replacement) in [
        (
            "position_process_noise_std_m = [1.0, 1.0, 1.0]",
            "position_process_noise_std_m",
            "horizontal_process_noise_std_m",
        ),
        (
            "zero_vertical_velocity = true",
            "zero_vertical_velocity",
            "baro_loop_time_constant_s",
        ),
        (
            "zero_vertical_velocity_std_mps = 0.1",
            "zero_vertical_velocity_std_mps",
            "baro_loop_time_constant_s",
        ),
    ] {
        let document = format!(
            "mode = \"particle-filter\"\n\n[particle_filter]\nnum_particles = 10\n{setting}\n"
        );
        let error = toml::from_str::<SimulationConfig>(&document)
            .expect_err("a retired key must be refused, not dropped")
            .to_string();
        assert!(
            error.contains(key) && error.contains(replacement),
            "the TOML error must name `{key}` and point at `{replacement}`, got: {error}"
        );
    }
}

/// The shipped default itself, through both construction paths.
///
/// The tests above hold the two paths to the *same* answer; this one pins what that answer is.
/// Turning the barometric bias state on by default is the change #372 was left open for, and
/// the gated accuracy baseline in `perf_baseline.rs` is blessed with it on -- so if this flips
/// back, the published tables in `book/src/development/baseline-tables.md` stop describing what
/// the crate actually does, which is the failure this whole file exists to prevent.
#[test]
fn the_barometric_bias_state_is_on_by_default() {
    assert!(
        ClosedLoopConfig::default().estimate_baro_bias,
        "`ClosedLoopConfig::default()` must estimate the barometric bias: it is the v1.0 \
         default and what the gated accuracy baseline measures"
    );

    let from_empty_section: ClosedLoopConfig =
        serde_json::from_str("{}").expect("an empty `closed_loop` section must deserialize");
    assert!(
        from_empty_section.estimate_baro_bias,
        "a `[closed_loop]` section that does not mention `estimate_baro_bias` must get the \
         same `true` the Rust API gets -- this is the exact divergence that shipped once"
    );
}

/// A nine-state filter must survive the documented runner.
///
/// `EkfConfig::use_biases` is a `pub` field and `initialize_ekf` honours it, so a caller can
/// build an EKF with no IMU-bias block -- `test_initialize_ekf_default_9state` in `sim.rs`
/// asserts exactly that shape. `run_closed_loop` then chose `ExtraStateLayout::NONE`, whose
/// width is a hardcoded fifteen, and the conversion to `NavigationResult` aborted on the very
/// first row with `State vector must have 15 elements; got 9` -- before a single event was
/// processed, by panic, in a crate whose lint policy denies panics in library code.
///
/// The branch that closed the other two reachable panics missed this one because it is reached
/// only through a configuration nothing in `strapdown-sim` sets: `use_biases` has no CLI flag
/// and appears in no shipped config, so only a library caller finds it. A published crate has
/// library callers.
#[test]
fn the_documented_runner_accepts_a_nine_state_filter() {
    let record = strapdown::sim::TestDataRecord::default();

    let mut config = strapdown::sim::EkfConfig::default();
    config.use_biases = false;
    let mut ekf = strapdown::sim::initialize_ekf(&record, config)
        .expect("a nine-state EKF is a supported configuration");

    assert_eq!(
        ekf.get_estimate().len(),
        9,
        "this test is only meaningful against a nine-state filter"
    );

    let stream = strapdown::messages::build_event_stream(
        std::slice::from_ref(&record),
        &strapdown::messages::AidingConfig::default(),
        false,
    )
    .expect("a single record is enough to build a stream");

    let results = strapdown::sim::run_closed_loop(&mut ekf, stream, None, None)
        .expect("the documented runner must accept a filter the library can build");

    assert!(
        !results.is_empty(),
        "the runner returned no rows for a nine-state filter"
    );
}

/// ...and the opposite case must still be refused.
///
/// The first attempt at the fix above derived the width from the filter unconditionally, which
/// also accepted a *wider* filter -- one carrying geophysical bias states. The plain runner
/// cannot label those: the filter cannot say which extra state is a gravity anomaly and which
/// a magnetic one, so an unlabelled solution would put a milligal figure in a nanotesla column.
/// `run_closed_loop_with_geo` is the entry point that takes the layout from the caller.
///
/// `geonav/tests/geo_closed_loop.rs::plain_closed_loop_still_rejects_a_geophysical_filter`
/// caught that, but it lives in a crate behind a feature flag. This keeps the two halves of the
/// invariant next to each other: narrower than fifteen is describable, wider is not.
#[test]
fn a_wider_filter_without_a_baro_index_is_still_refused() {
    let record = strapdown::sim::TestDataRecord::default();

    let mut config = strapdown::sim::UkfConfig::default();
    // A sixteenth state with no `baro_bias_index` set: exactly the shape a geophysical filter
    // has when handed to the plain runner by a caller who forgot the layout.
    config.other_states = Some(vec![0.0]);
    config.other_states_covariance = Some(vec![1.0]);
    // The extra state widens the filter, so the process-noise diagonal has to match it.
    let mut process_noise = strapdown::sim::DEFAULT_PROCESS_NOISE_DENSITY.to_vec();
    process_noise.push(1e-6);
    config.process_noise_diagonal = Some(process_noise);
    let mut ukf = strapdown::sim::initialize_ukf(&record, config)
        .expect("a sixteen-state UKF is constructible");

    assert_eq!(ukf.get_estimate().len(), 16);
    assert!(
        ukf.baro_bias_index().is_none(),
        "this test is only meaningful when the extra state is unlabelled"
    );

    let stream = strapdown::messages::build_event_stream(
        std::slice::from_ref(&record),
        &strapdown::messages::AidingConfig::default(),
        false,
    )
    .expect("a single record is enough to build a stream");

    let refused = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = strapdown::sim::run_closed_loop(&mut ukf, stream, None, None);
    }))
    .is_err();

    assert!(
        refused,
        "the plain runner accepted a filter with unlabelled extra states; it would have \
         written the gravity bias into the wrong column rather than refusing"
    );
}
