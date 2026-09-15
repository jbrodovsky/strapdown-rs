//! Navigation-accuracy regression gate (#328).
#![allow(
    clippy::unwrap_used,
    clippy::panic,
    clippy::expect_used,
    reason = "integration tests are a separate crate target, so `clippy.toml`'s \
              allow-unwrap-in-tests -- which only covers `#[cfg(test)]` items -- does not \
              reach them; unwrapping is how these assert"
)]
//!
//! This file scores a fixed matrix of scenarios with [`strapdown::metrics`] and compares the
//! result against `core/tests/perf_baseline.json`, a checked-in record of what the navigation
//! solution measured when it was last blessed. It fails in **both** directions:
//!
//! * a metric more than `regress_fraction` worse than its baseline is a regression;
//! * a metric more than `improve_fraction` better is *also* a failure, whose message asks for
//!   the baseline to be re-blessed. An improvement that is not recorded is one the next change
//!   can silently undo, which is the hole this file exists to close.
//!
//! Re-bless with:
//!
//! ```text
//! UPDATE_PERF_BASELINE=1 cargo test -p strapdown-core --test perf_baseline
//! # PowerShell:
//! $env:UPDATE_PERF_BASELINE=1; cargo test -p strapdown-core --test perf_baseline
//! ```
//!
//! then read the diff before committing it: every changed number is a claim about the
//! navigation solution, and the pull request should say which change produced it.
//!
//! # How this differs from `integration_tests.rs`
//!
//! That file asserts *ceilings*, several of which are derived physical bounds -- yaw against
//! the magnetometer's own measured error, roll and pitch against gravity observability. Those
//! are claims about physics and must hold whatever any baseline says. This file asserts
//! *drift*: relative, ratcheted, with no physical content of its own. Both are wanted and
//! neither subsumes the other, which is why the thresholds over there were left alone.
//!
//! It also drives the **public** entry points -- `sim::initialize_ukf`/`_ekf`/`_eskf` plus
//! `sim::run_closed_loop` -- rather than the test-local filter builders `integration_tests.rs`
//! uses. The numbers here therefore do not match that file's exactly: `initialize_*` derives
//! the initial covariance from the record's own accuracy columns where the test helpers use a
//! fixed default. Measuring what ships is the point.
//!
//! # Reading the numbers
//!
//! Three caveats apply to every figure in the baseline, and none of them is a defect in this
//! harness:
//!
//! 1. **Real-data metrics are scored against the GNSS fix, which is also the aiding source.**
//!    They measure agreement with the aid, and cannot fall below the receiver's own 3.81 m
//!    horizontal noise however good the filter is.
//! 2. **Every horizontal number on a real-data scenario carries a one-step propagation
//!    offset.** `sim::run_closed_loop` pushes a row *after* applying the event, so the row
//!    labelled `t_k` holds a state already propagated through the first event of `t_{k+1}`. At
//!    1 Hz and 21.19 m/s that is 21.2 m of along-track error on its own. Tracked in #367; the
//!    synthetic scenarios run at 50 Hz specifically so the same offset is ~1 m there. When
//!    #367 is fixed every horizontal metric here trips the improvement side at once.
//! 3. **The synthetic scenarios carry no magnetometer, and the yaw column says which filters
//!    need one.** `generate_synthetic` models no magnetic field, so the only thing aiding
//!    heading there is the GNSS velocity fix on a moving trajectory. That turns out to be
//!    enough: the EKF holds 0.97 deg of yaw and the ESKF 2.17 deg on `syn_cruise_1hz`, while
//!    the UKF sits at 48.7 deg. The UKF figure is not an observability limit -- both of the
//!    other filters see the same measurements -- it is #336, which averages sigma-point Euler
//!    angles linearly. It is recorded and gated like everything else, so fixing #336 will trip
//!    the improvement side and ask for a re-bless.
//! 4. **The consistency metrics mean something different on the two sources.** On the synthetic
//!    scenarios `npes_position` lands at 3.5 to 5.7 against an ideal of 3.0, which is a real
//!    measurement of whether the filters believe the right thing. On the real-data scenarios it
//!    reaches the hundreds, because caveat 2's 21 m offset enters the numerator while the
//!    covariance in the denominator knows nothing about it. Those values are recorded as a
//!    drift detector, not read as a consistency verdict.
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use nalgebra::Rotation3;
use serde::{Deserialize, Serialize};

use strapdown::messages::{
    GnssDegradationConfig, GnssFaultModel, GnssScheduler, build_event_stream,
};
use strapdown::metrics::{AccuracyMetrics, MetricDirection, MetricId, MetricOptions, TruthSample};
use strapdown::rbpf::{RaoBlackwellizedParticleFilter, RbpfConfig};
use strapdown::sim::{
    EkfConfig, EskfConfig, GeoStateLayout, NavigationResult, SyntheticConfig,
    SyntheticInitialState, TestDataRecord, UkfConfig, dead_reckoning, generate_synthetic,
    initialize_ekf, initialize_eskf, initialize_ukf, run_closed_loop, run_closed_loop_with_geo,
};
use strapdown::{IMUQuality, StrapdownState};

/// Environment variable that switches this test from gating to recording.
const UPDATE_ENV: &str = "UPDATE_PERF_BASELINE";

/// Schema version of `perf_baseline.json`. Bump it when the file's shape changes.
const SCHEMA_VERSION: u32 = 1;

/// Fraction a metric may worsen before it is called a regression.
const DEFAULT_REGRESS_FRACTION: f64 = 0.10;

/// Fraction a metric may improve before the baseline is called stale.
const DEFAULT_IMPROVE_FRACTION: f64 = 0.25;

/// Horizontal RMSE above which a scenario is reported as diverged rather than as regressed.
///
/// A baseline is a contract saying "this is acceptable", and a filter thousands of kilometres
/// from truth is not. This guard refuses to bless such a value even when someone asks, and
/// keeps a genuine divergence from being reported as a 400,000-percent regression, which is
/// unreadable and hides which scenario actually broke.
const DIVERGENCE_CEILING_M: f64 = 1000.0;

/// `core/tests/test_data.csv` is an ENU Sensor Logger export.
const REAL_DATA_IS_ENU: bool = true;

/// Records taken from the front of `test_data.csv` for the particle-filter scenario.
///
/// The particle filter is the only estimator here whose cost is measured in seconds rather
/// than tenths, so it runs over a slice. 1,200 samples is 20 minutes of the recording, which
/// is long enough for the cloud to settle and for a bias to show.
const RBPF_SLICE_SAMPLES: usize = 1200;

/// Particle count for the particle-filter scenario, matching `RbpfConfig::default()`.
const RBPF_PARTICLES: usize = 500;

/// Seed for every stochastic component of the suite.
const SEED: u64 = 42;

// ---------------------------------------------------------------------------------------
// Scenario definitions
// ---------------------------------------------------------------------------------------

/// Which trajectory a scenario runs over.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Source {
    /// The whole of `core/tests/test_data.csv`.
    Real,
    /// The first [`RBPF_SLICE_SAMPLES`] records of `core/tests/test_data.csv`.
    RealSlice,
    /// The 300 s synthetic trajectory, scored against its own exact truth.
    Synthetic,
    /// The first 120 s of the synthetic trajectory.
    SyntheticShort,
}

/// Which estimator a scenario drives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Estimator {
    /// Unaided mechanization, via `sim::dead_reckoning`.
    DeadReckoning,
    /// Unscented Kalman filter, via `sim::initialize_ukf`.
    Ukf,
    /// Extended Kalman filter, via `sim::initialize_ekf`.
    Ekf,
    /// Error-state Kalman filter, via `sim::initialize_eskf`.
    Eskf,
    /// Rao-Blackwellized particle filter.
    Rbpf,
}

impl Estimator {
    /// Suffix used in the scenario id.
    const fn key(self) -> &'static str {
        match self {
            Self::DeadReckoning => "dead_reckoning",
            Self::Ukf => "ukf",
            Self::Ekf => "ekf",
            Self::Eskf => "eskf",
            Self::Rbpf => "rbpf",
        }
    }
}

/// One row of the matrix: a trajectory, a GNSS condition and an estimator.
#[derive(Debug)]
struct Scenario {
    /// Stable id, `condition__estimator`, and the baseline file's key.
    id: String,
    /// One line describing the run, rendered into the baseline for a human reader.
    description: String,
    /// Which trajectory to run over.
    source: Source,
    /// GNSS scheduling and fault model.
    gnss: GnssDegradationConfig,
    /// Which estimator to drive.
    estimator: Estimator,
}

/// Every scenario in the suite, in report order.
///
/// Kept as a function rather than a `const` because `GnssFaultModel` owns a `Vec` in one of
/// its variants and so is not a constant-constructible type.
fn scenarios() -> Vec<Scenario> {
    let mut out = Vec::new();

    let clean = || GnssDegradationConfig {
        scheduler: GnssScheduler::PassThrough,
        fault: GnssFaultModel::None,
        seed: SEED,
    };

    // Canonical reference: the configuration every other real-data row is read against.
    for estimator in [Estimator::Ukf, Estimator::Ekf, Estimator::Eskf] {
        out.push(Scenario {
            id: format!("real_clean__{}", estimator.key()),
            description: "test_data.csv, GNSS every epoch, no fault".to_string(),
            source: Source::Real,
            gnss: clean(),
            estimator,
        });
    }

    // Aiding at 1/5 the rate. Exercises propagation between fixes, which pass-through aiding
    // at 1 Hz never does.
    for estimator in [Estimator::Ukf, Estimator::Eskf] {
        out.push(Scenario {
            id: format!("real_sparse_5s__{}", estimator.key()),
            description: "test_data.csv, GNSS every 5 s, no fault".to_string(),
            source: Source::Real,
            gnss: GnssDegradationConfig {
                scheduler: GnssScheduler::FixedInterval {
                    interval_s: 5.0,
                    phase_s: 0.0,
                },
                fault: GnssFaultModel::None,
                seed: SEED,
            },
            estimator,
        });
    }

    // The headline v1.0 case. Errors here reach tens of metres, so the one-step labelling
    // offset described in the module docs is a small share of the number rather than most of
    // it -- which is what makes this row a usable regression detector on real data.
    out.push(Scenario {
        id: "real_outage_60s__eskf".to_string(),
        description: "test_data.csv, 120 s of GNSS then a 60 s outage, repeating".to_string(),
        source: Source::Real,
        gnss: GnssDegradationConfig {
            scheduler: GnssScheduler::DutyCycle {
                on_s: 120.0,
                off_s: 60.0,
                start_phase_s: 0.0,
            },
            fault: GnssFaultModel::None,
            seed: SEED,
        },
        estimator: Estimator::Eskf,
    });

    // A correlated position/velocity fault rather than an outage: this is the row that moves
    // when innovation gating or the measurement noise model changes.
    for estimator in [Estimator::Ukf, Estimator::Eskf] {
        out.push(Scenario {
            id: format!("real_degraded__{}", estimator.key()),
            description: "test_data.csv, GNSS every epoch, AR(1) position and velocity fault"
                .to_string(),
            source: Source::Real,
            gnss: GnssDegradationConfig {
                scheduler: GnssScheduler::PassThrough,
                fault: GnssFaultModel::Degraded {
                    rho_pos: 0.98,
                    sigma_pos_m: 3.0,
                    rho_vel: 0.98,
                    sigma_vel_mps: 0.3,
                    r_scale: 1.0,
                },
                seed: SEED,
            },
            estimator,
        });
    }

    // Exact truth, so these are the only horizontal numbers in the file that measure the
    // filter rather than the reference.
    for estimator in [Estimator::Ukf, Estimator::Ekf, Estimator::Eskf] {
        out.push(Scenario {
            id: format!("syn_cruise_1hz__{}", estimator.key()),
            description: "synthetic 300 s at 50 Hz, GNSS every 1 s, scored against exact truth"
                .to_string(),
            source: Source::Synthetic,
            gnss: GnssDegradationConfig {
                scheduler: GnssScheduler::FixedInterval {
                    interval_s: 1.0,
                    phase_s: 0.0,
                },
                fault: GnssFaultModel::None,
                seed: SEED,
            },
            estimator,
        });
    }

    // Exact truth plus 60 s of free inertial: the most informative rows in the suite.
    for estimator in [Estimator::Ukf, Estimator::Eskf] {
        out.push(Scenario {
            id: format!("syn_outage_60s__{}", estimator.key()),
            description: "synthetic 300 s at 50 Hz, 60 s of GNSS then 60 s of outage".to_string(),
            source: Source::Synthetic,
            gnss: GnssDegradationConfig {
                scheduler: GnssScheduler::DutyCycle {
                    on_s: 60.0,
                    off_s: 60.0,
                    start_phase_s: 0.0,
                },
                fault: GnssFaultModel::None,
                seed: SEED,
            },
            estimator,
        });
    }

    // The unaided anchor, and the scenario that exercises the all-NaN-covariance path with a
    // real run rather than only a unit test. Deliberately short: unaided dead reckoning over
    // the full recording ends millions of metres out by way of an altitude at which
    // `earth::transport_rate`'s denominator is within a few percent of zero, and whose
    // finiteness is a floating-point accident that differs across platforms (#299).
    out.push(Scenario {
        id: "syn_dead_reckoning".to_string(),
        description: "synthetic first 120 s, unaided mechanization".to_string(),
        source: Source::SyntheticShort,
        gnss: clean(),
        estimator: Estimator::DeadReckoning,
    });

    // The one non-Kalman estimator.
    out.push(Scenario {
        id: "real_rbpf_slice__rbpf".to_string(),
        description: format!(
            "first {RBPF_SLICE_SAMPLES} records of test_data.csv, GNSS every epoch, \
             {RBPF_PARTICLES} particles"
        ),
        source: Source::RealSlice,
        gnss: clean(),
        estimator: Estimator::Rbpf,
    });

    out
}

// ---------------------------------------------------------------------------------------
// Running a scenario
// ---------------------------------------------------------------------------------------

/// Path to `core/tests/test_data.csv`, resolved from the manifest rather than the cwd.
fn real_data_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/test_data.csv")
}

/// Path to the checked-in baseline.
fn baseline_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/perf_baseline.json")
}

/// The synthetic trajectory every `syn_*` scenario runs over.
///
/// 50 Hz, not the 10 Hz default, on purpose: `run_closed_loop` labels each row with the
/// timestamp of the event *before* the one it has already applied, so a solution carries one
/// propagation step of along-track offset. At 50 m/s that is 1 m at 50 Hz and 5 m at 10 Hz,
/// and the point of these rows is to resolve filter error of a few metres.
const fn synthetic_config(duration_s: f64) -> SyntheticConfig {
    SyntheticConfig {
        output: String::new(),
        initial_state: SyntheticInitialState {
            latitude_deg: 40.0,
            longitude_deg: -75.0,
            altitude_m: 200.0,
            // Off-axis, so both horizontal channels carry signal rather than one being zero.
            velocity_north_mps: 40.0,
            velocity_east_mps: 30.0,
            velocity_down_mps: 0.0,
            roll_deg: 0.0,
            pitch_deg: 0.0,
            yaw_deg: 36.869_897_645_844_02,
            angular_velocity_x_dps: 0.0,
            angular_velocity_y_dps: 0.0,
            angular_velocity_z_dps: 0.0,
            is_enu: false,
        },
        duration_s,
        sample_rate_hz: 50.0,
        imu_quality: IMUQuality::Consumer,
        seed: SEED,
        no_noise: false,
        gnss_horizontal_noise_m: 3.0,
        gnss_vertical_noise_m: 5.0,
        baro_noise_std_pa: 30.0,
    }
}

/// The sensor records and reference truth a scenario is run over and scored against.
struct Trajectory {
    /// Sensor log driving the filter.
    records: Vec<TestDataRecord>,
    /// Reference series the solution is scored against.
    truth: Vec<TruthSample>,
    /// Frame the records are in.
    is_enu: bool,
}

/// Load or generate the trajectory for one source. Cached by the caller, not here.
fn trajectory(source: Source) -> Trajectory {
    match source {
        Source::Real | Source::RealSlice => {
            let mut records = TestDataRecord::from_csv(real_data_path())
                .expect("core/tests/test_data.csv must load");
            assert!(
                records.len() > RBPF_SLICE_SAMPLES,
                "test_data.csv parsed to only {} records; `from_csv` drops unparseable rows \
                 and returns Ok, so a short read is silent",
                records.len()
            );
            if source == Source::RealSlice {
                records.truncate(RBPF_SLICE_SAMPLES);
            }
            let truth = strapdown::metrics::truth_from_records(&records);
            Trajectory {
                records,
                truth,
                is_enu: REAL_DATA_IS_ENU,
            }
        }
        Source::Synthetic | Source::SyntheticShort => {
            let duration_s = if source == Source::SyntheticShort {
                120.0
            } else {
                300.0
            };
            let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::seed_from_u64(SEED);
            let (truth_rows, records) = generate_synthetic(&synthetic_config(duration_s), &mut rng)
                .expect("synthetic generation must succeed");
            let truth = strapdown::metrics::truth_from_trajectory(&truth_rows);
            Trajectory {
                records,
                truth,
                is_enu: false,
            }
        }
    }
}

/// Seed a [`StrapdownState`] from the first record, for the particle filter.
fn nominal_state(first: &TestDataRecord, is_enu: bool) -> StrapdownState {
    let (roll, pitch, yaw) = first.attitude().euler_angles();
    let (velocity_north, velocity_east) = first.ground_track_velocity();
    StrapdownState {
        latitude: first.latitude.to_radians(),
        longitude: first.longitude.to_radians(),
        altitude: first.altitude,
        velocity_north,
        velocity_east,
        velocity_vertical: 0.0,
        attitude: Rotation3::from_euler_angles(roll, pitch, yaw),
        is_enu,
    }
}

/// Drive one scenario end to end and return its navigation solution.
fn solve(scenario: &Scenario, trajectory: &Trajectory) -> Vec<NavigationResult> {
    let records = &trajectory.records;
    let is_enu = trajectory.is_enu;

    if scenario.estimator == Estimator::DeadReckoning {
        return dead_reckoning(records, is_enu).expect("dead reckoning must run");
    }

    let stream = build_event_stream(records, &scenario.gnss, is_enu)
        .expect("event stream construction must succeed");
    let first = &records[0];

    match scenario.estimator {
        Estimator::Ukf => {
            let mut filter = initialize_ukf(
                first,
                UkfConfig {
                    is_enu,
                    ..UkfConfig::default()
                },
            )
            .expect("UKF initialization");
            run_closed_loop(&mut filter, stream, None, None).expect("UKF closed loop")
        }
        Estimator::Ekf => {
            let mut filter = initialize_ekf(
                first,
                EkfConfig {
                    is_enu,
                    ..EkfConfig::default()
                },
            )
            .expect("EKF initialization");
            run_closed_loop(&mut filter, stream, None, None).expect("EKF closed loop")
        }
        Estimator::Eskf => {
            let mut filter = initialize_eskf(
                first,
                EskfConfig {
                    is_enu,
                    ..EskfConfig::default()
                },
            )
            .expect("ESKF initialization");
            run_closed_loop(&mut filter, stream, None, None).expect("ESKF closed loop")
        }
        Estimator::Rbpf => {
            // Through the shared runner, not a hand-rolled loop: `GeoStateLayout::PARTICLE_NONE`
            // now dispatches to the particle constructor inside `NavigationResult`'s four-tuple
            // `From`, which is what makes `run_closed_loop_with_geo` usable here.
            let mut filter = RaoBlackwellizedParticleFilter::new(
                nominal_state(first, is_enu),
                RbpfConfig {
                    num_particles: RBPF_PARTICLES,
                    seed: SEED,
                    ..RbpfConfig::default()
                },
            )
            .expect("RBPF construction");
            run_closed_loop_with_geo(
                &mut filter,
                stream,
                None,
                None,
                GeoStateLayout::PARTICLE_NONE,
            )
            .expect("RBPF closed loop")
        }
        Estimator::DeadReckoning => unreachable!("handled above"),
    }
}

/// Run every scenario and score it.
fn measure_all() -> Vec<(Scenario, AccuracyMetrics)> {
    let mut cache: BTreeMap<&'static str, Trajectory> = BTreeMap::new();
    let mut measured = Vec::new();

    for scenario in scenarios() {
        let key: &'static str = match scenario.source {
            Source::Real => "real",
            Source::RealSlice => "real_slice",
            Source::Synthetic => "synthetic",
            Source::SyntheticShort => "synthetic_short",
        };
        let trajectory = cache
            .entry(key)
            .or_insert_with(|| self::trajectory(scenario.source));

        let solution = solve(&scenario, trajectory);
        let metrics =
            strapdown::metrics::evaluate(&solution, &trajectory.truth, MetricOptions::default())
                .unwrap_or_else(|e| panic!("scoring `{}` failed: {e}", scenario.id));
        measured.push((scenario, metrics));
    }
    measured
}

// ---------------------------------------------------------------------------------------
// The baseline file
// ---------------------------------------------------------------------------------------

/// The checked-in record of what the navigation solution measured when it was last blessed.
#[derive(Debug, Serialize, Deserialize)]
struct BaselineFile {
    /// Shape of this file; see [`SCHEMA_VERSION`].
    schema_version: u32,
    /// Free text carried through re-blessing, for a human opening the file first.
    note: String,
    /// Bands applied to any metric that does not override them.
    default_tolerance: Tolerance,
    /// One entry per scenario id, sorted so a diff is stable.
    scenarios: BTreeMap<String, BaselineScenario>,
}

/// Two-sided tolerance band.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
struct Tolerance {
    /// Fraction a metric may worsen before it is a regression.
    regress_fraction: f64,
    /// Fraction a metric may improve before the baseline is stale.
    improve_fraction: f64,
}

/// Recorded metrics for one scenario.
#[derive(Debug, Serialize, Deserialize)]
struct BaselineScenario {
    /// What this scenario runs, copied from the scenario table on every bless.
    description: String,
    /// Aligned sample count, compared for exact equality before any metric is.
    sample_count: usize,
    /// One entry per metric key.
    metrics: BTreeMap<String, BaselineMetric>,
}

/// One recorded metric, plus any per-metric policy a maintainer has attached to it.
#[derive(Debug, Serialize, Deserialize)]
struct BaselineMetric {
    /// The measured value, or `null` when the metric is not computable for this scenario.
    ///
    /// `Option<f64>` rather than `f64`: `serde_json` writes `f64::NAN` as `null` and then
    /// refuses to read `null` back into an `f64`, so a NaN here would produce a file that
    /// cannot be loaded.
    value: Option<f64>,
    /// Override for [`Tolerance::regress_fraction`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    regress_fraction: Option<f64>,
    /// Override for [`Tolerance::improve_fraction`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    improve_fraction: Option<f64>,
    /// Whether this metric is asserted. `Some(false)` keeps it measured, recorded and
    /// printed, but not gated -- for a quantity that is real but unobservable in this
    /// scenario. Absent means gated, which is why this is an `Option` rather than a `bool`
    /// with a default: the common case then writes nothing at all into the file.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    gated: Option<bool>,
    /// Why this entry carries an override, for whoever reads the diff next.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    note: Option<String>,
}

impl BaselineMetric {
    /// Whether this metric is asserted; absent means yes.
    const fn is_gated(&self) -> bool {
        match self.gated {
            Some(gated) => gated,
            None => true,
        }
    }

    /// The band this metric is judged against, falling back to the file default.
    fn tolerance(&self, default: Tolerance) -> Tolerance {
        Tolerance {
            regress_fraction: self.regress_fraction.unwrap_or(default.regress_fraction),
            improve_fraction: self.improve_fraction.unwrap_or(default.improve_fraction),
        }
    }
}

/// Round to `digits` significant figures.
///
/// Full-precision output makes an unreadable diff and turns a one-ulp platform difference into
/// a changed line. Six figures is far finer than any band here, so the comparison is unaffected.
fn round_significant(value: f64, digits: i32) -> f64 {
    if !value.is_finite() || value == 0.0 {
        return value;
    }
    let magnitude = value.abs().log10().floor() as i32;
    let factor = 10f64.powi(digits - 1 - magnitude);
    (value * factor).round() / factor
}

/// Build the file that records `measured`, preserving any annotations already on disk.
///
/// Values are replaced; `gated`, the per-metric overrides and the notes are carried over, so
/// blessing never silently discards a maintainer's decision about a metric.
fn render_baseline(
    measured: &[(Scenario, AccuracyMetrics)],
    previous: Option<&BaselineFile>,
) -> BaselineFile {
    let mut scenarios = BTreeMap::new();
    for (scenario, metrics) in measured {
        let old = previous.and_then(|p| p.scenarios.get(&scenario.id));
        let mut entries = BTreeMap::new();
        for (id, value) in metrics.iter() {
            let old_metric = old.and_then(|o| o.metrics.get(id.key()));
            entries.insert(
                id.key().to_string(),
                BaselineMetric {
                    value: value.map(|v| round_significant(v, 6)),
                    regress_fraction: old_metric.and_then(|m| m.regress_fraction),
                    improve_fraction: old_metric.and_then(|m| m.improve_fraction),
                    gated: old_metric.and_then(|m| m.gated),
                    note: old_metric.and_then(|m| m.note.clone()),
                },
            );
        }
        scenarios.insert(
            scenario.id.clone(),
            BaselineScenario {
                description: scenario.description.clone(),
                sample_count: metrics.sample_count,
                metrics: entries,
            },
        );
    }

    BaselineFile {
        schema_version: SCHEMA_VERSION,
        note: previous.map_or_else(default_note, |p| p.note.clone()),
        default_tolerance: previous.map_or(
            Tolerance {
                regress_fraction: DEFAULT_REGRESS_FRACTION,
                improve_fraction: DEFAULT_IMPROVE_FRACTION,
            },
            |p| p.default_tolerance,
        ),
        scenarios,
    }
}

/// The note written into a baseline created from nothing.
fn default_note() -> String {
    format!(
        "Navigation-accuracy baseline (#328). Regenerate with \
         `{UPDATE_ENV}=1 cargo test -p strapdown-core --test perf_baseline`, then review the \
         diff: every changed number is a claim about the navigation solution. Values are \
         rounded to 6 significant figures. `null` means the metric is not computable for that \
         scenario -- dead reckoning reports no covariance -- and is not gated. `\"gated\": \
         false` keeps a metric recorded but unasserted. See the module docs in \
         core/tests/perf_baseline.rs for what the numbers do and do not measure."
    )
}

// ---------------------------------------------------------------------------------------
// The gate
// ---------------------------------------------------------------------------------------

/// Verdict for one measured metric against its baseline.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Verdict {
    /// Inside the band, or not gated.
    Ok,
    /// Further from the metric's ideal than the band allows.
    Regressed,
    /// Closer to the ideal than the band allows: the baseline is stale.
    Improved,
}

/// Judge one measured value against one recorded value.
///
/// Both are reduced to a distance from the metric's ideal -- zero for a
/// [`MetricDirection::LowerIsBetter`] metric, the stated target otherwise -- so the same
/// arithmetic covers "smaller is better" and "three sigma should contain 99.73 percent".
///
/// The absolute [`MetricId::noise_floor`] is added to the regression limit and subtracted from
/// the improvement limit, which widens the "unchanged" region in both directions. Near a
/// target that matters: a metric already within a noise floor of its ideal cannot improve
/// detectably, and reporting that as a stale baseline every run would train everyone to
/// re-bless without reading.
fn judge(id: MetricId, baseline: f64, measured: f64, tolerance: Tolerance) -> Verdict {
    let target = match id.direction() {
        MetricDirection::LowerIsBetter => 0.0,
        MetricDirection::TowardTarget { target } => target,
    };
    let floor = id.noise_floor();
    let baseline_distance = (baseline - target).abs();
    let measured_distance = (measured - target).abs();

    let regress_limit = baseline_distance * (1.0 + tolerance.regress_fraction) + floor;
    let improve_limit = baseline_distance * (1.0 - tolerance.improve_fraction) - floor;

    if measured_distance > regress_limit {
        Verdict::Regressed
    } else if measured_distance < improve_limit {
        Verdict::Improved
    } else {
        Verdict::Ok
    }
}

/// Compare every measured scenario against the baseline, collecting all violations.
///
/// Everything is collected rather than bailing on the first problem: one CI log listing every
/// metric that moved is worth far more than the first one in `MetricId::ALL` order.
fn compare(measured: &[(Scenario, AccuracyMetrics)], baseline: &BaselineFile) -> Vec<String> {
    let mut problems = Vec::new();
    let known_keys: BTreeSet<&str> = MetricId::ALL.iter().map(|m| m.key()).collect();
    let mut seen = BTreeSet::new();

    for (scenario, metrics) in measured {
        seen.insert(scenario.id.clone());

        // A diverged run is reported as diverged, whatever the baseline says. Comparing a
        // filter thousands of kilometres from truth against a 23 m baseline produces a
        // percentage nobody can read and buries which scenario actually broke.
        match metrics.horizontal_rmse_m {
            Some(rmse) if rmse.is_finite() && rmse < DIVERGENCE_CEILING_M => {}
            other => {
                problems.push(format!(
                    "DIVERGED  `{}`: horizontal RMSE is {other:?}, at or beyond the \
                     {DIVERGENCE_CEILING_M} m ceiling. This is not a regression to be \
                     re-blessed -- the scenario no longer produces a navigation solution.",
                    scenario.id
                ));
                continue;
            }
        }

        let Some(recorded) = baseline.scenarios.get(&scenario.id) else {
            problems.push(format!(
                "MISSING   scenario `{}` produced {} metrics but has no baseline entry.",
                scenario.id,
                MetricId::ALL.len()
            ));
            continue;
        };

        if recorded.sample_count != metrics.sample_count {
            problems.push(format!(
                "COUNT     `{}`: scored {} aligned samples, baseline records {}. Every metric \
                 beneath a changed sample count is incomparable, so they are not checked.",
                scenario.id, metrics.sample_count, recorded.sample_count
            ));
            continue;
        }

        for stale in recorded
            .metrics
            .keys()
            .filter(|k| !known_keys.contains(k.as_str()))
        {
            problems.push(format!(
                "STALE     `{}` carries metric `{stale}`, which this suite no longer computes.",
                scenario.id
            ));
        }

        for (id, value) in metrics.iter() {
            let Some(entry) = recorded.metrics.get(id.key()) else {
                problems.push(format!(
                    "MISSING   `{}` is missing metric `{}`.",
                    scenario.id,
                    id.key()
                ));
                continue;
            };
            problems.extend(check_metric(
                &scenario.id,
                id,
                value,
                entry,
                baseline.default_tolerance,
            ));
        }
    }

    for orphan in baseline.scenarios.keys().filter(|k| !seen.contains(*k)) {
        problems.push(format!(
            "STALE     scenario `{orphan}` is in the baseline but was not produced by this run."
        ));
    }

    problems
}

/// Compare one metric, returning zero or one problem lines.
fn check_metric(
    scenario_id: &str,
    id: MetricId,
    measured: Option<f64>,
    entry: &BaselineMetric,
    default: Tolerance,
) -> Vec<String> {
    match (entry.value, measured) {
        (None, None) => Vec::new(),
        (None, Some(now)) => vec![format!(
            "CHANGED   `{scenario_id}` / `{}`: baseline records this as not computable, but \
             the run produced {now:.6}. Re-bless to record it.",
            id.key()
        )],
        (Some(was), None) => vec![format!(
            "CHANGED   `{scenario_id}` / `{}`: baseline records {was:.6}, but the run could \
             not compute it -- usually a covariance that has become non-finite.",
            id.key()
        )],
        (Some(was), Some(now)) => {
            if !entry.is_gated() {
                return Vec::new();
            }
            let tolerance = entry.tolerance(default);
            match judge(id, was, now, tolerance) {
                Verdict::Ok => Vec::new(),
                Verdict::Regressed => vec![format!(
                    "REGRESSED `{scenario_id}` / `{}`: {now:.6} {unit} against a baseline of \
                     {was:.6} {unit} ({dir}, band +{band:.0}%, noise floor {floor}).",
                    id.key(),
                    unit = unit_suffix(id),
                    dir = describe(id),
                    band = tolerance.regress_fraction * 100.0,
                    floor = id.noise_floor(),
                )],
                Verdict::Improved => vec![format!(
                    "IMPROVED  `{scenario_id}` / `{}`: {now:.6} {unit} against a baseline of \
                     {was:.6} {unit} ({dir}). This is better than the band allows, so the \
                     baseline is stale -- re-bless it.",
                    id.key(),
                    unit = unit_suffix(id),
                    dir = describe(id),
                )],
            }
        }
    }
}

/// Human-readable statement of which way a metric improves.
fn describe(id: MetricId) -> String {
    match id.direction() {
        MetricDirection::LowerIsBetter => "lower is better".to_string(),
        MetricDirection::TowardTarget { target } => format!("target {target:.4}"),
    }
}

/// Unit symbol for a message, blank for a dimensionless metric.
///
/// `"1"` is the correct symbol for a dimensionless quantity and is what the printed table
/// column uses, but "3.56 1 against a baseline of 4.38 1" is not a readable sentence.
fn unit_suffix(id: MetricId) -> &'static str {
    if id.unit() == "1" { "" } else { id.unit() }
}

// ---------------------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------------------

/// Render the measured suite as a Markdown table, for pasting into the book page.
fn markdown_table(measured: &[(Scenario, AccuracyMetrics)]) -> String {
    let mut out = String::new();
    out.push_str("\n| scenario | samples |");
    for id in MetricId::ALL {
        let _ = write!(out, " {} |", id.key());
    }
    out.push_str("\n|---|---:|");
    for _ in MetricId::ALL {
        out.push_str("---:|");
    }
    out.push('\n');

    for (scenario, metrics) in measured {
        let _ = write!(out, "| `{}` | {} |", scenario.id, metrics.sample_count);
        for (_, value) in metrics.iter() {
            match value {
                Some(v) => {
                    let _ = write!(out, " {v:.4} |");
                }
                None => out.push_str(" -- |"),
            }
        }
        out.push('\n');
    }
    out
}

// ---------------------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------------------

/// Score the suite and gate on it, or record it when [`UPDATE_ENV`] is set.
///
/// One test function, not one per scenario, and that is a requirement rather than a
/// preference: `cargo test` runs test functions on parallel threads, and several of them each
/// rewriting one JSON file is a data race. It also means a failing run produces a single log
/// listing every metric that moved.
#[test]
fn navigation_accuracy_matches_the_recorded_baseline() {
    let measured = measure_all();
    println!("{}", markdown_table(&measured));

    let path = baseline_path();
    let existing: Option<BaselineFile> = if path.exists() {
        let text = std::fs::read_to_string(&path).expect("baseline file must be readable");
        Some(serde_json::from_str(&text).unwrap_or_else(|e| {
            panic!(
                "{} is not a valid baseline: {e}. Fix it by hand or regenerate it with \
                 `{UPDATE_ENV}=1 cargo test -p strapdown-core --test perf_baseline`.",
                path.display()
            )
        }))
    } else {
        None
    };

    if std::env::var_os(UPDATE_ENV).is_some() {
        let diverged: Vec<&str> = measured
            .iter()
            .filter(|(_, m)| {
                !m.horizontal_rmse_m
                    .is_some_and(|v| v.is_finite() && v < DIVERGENCE_CEILING_M)
            })
            .map(|(s, _)| s.id.as_str())
            .collect();
        assert!(
            diverged.is_empty(),
            "refusing to bless a diverged run: {diverged:?} exceeded the \
             {DIVERGENCE_CEILING_M} m horizontal ceiling. A baseline is a contract saying \
             this is acceptable, and a diverged solution is not."
        );

        let rendered = render_baseline(&measured, existing.as_ref());
        let mut text = serde_json::to_string_pretty(&rendered).expect("baseline must serialize");
        text.push('\n');
        std::fs::write(&path, text).expect("baseline file must be writable");
        println!("Wrote {}", path.display());
        return;
    }

    let baseline = existing.unwrap_or_else(|| {
        panic!(
            "{} does not exist. Create it with \
             `{UPDATE_ENV}=1 cargo test -p strapdown-core --test perf_baseline`.",
            path.display()
        )
    });
    assert_eq!(
        baseline.schema_version, SCHEMA_VERSION,
        "baseline schema version {} does not match this harness's {SCHEMA_VERSION}",
        baseline.schema_version
    );

    let problems = compare(&measured, &baseline);
    assert!(
        problems.is_empty(),
        "navigation accuracy no longer matches {}:\n\n{}\n\n\
         If the change is intended, re-bless and commit the diff:\n\n    \
         {UPDATE_ENV}=1 cargo test -p strapdown-core --test perf_baseline\n    \
         # PowerShell: $env:{UPDATE_ENV}=1; cargo test -p strapdown-core --test perf_baseline\n",
        path.display(),
        problems.join("\n")
    );
}

#[cfg(test)]
mod gate_tests {
    use super::*;

    const BAND: Tolerance = Tolerance {
        regress_fraction: DEFAULT_REGRESS_FRACTION,
        improve_fraction: DEFAULT_IMPROVE_FRACTION,
    };

    #[test]
    fn a_lower_is_better_metric_gates_in_both_directions() {
        let id = MetricId::HorizontalRmse;
        assert_eq!(judge(id, 20.0, 20.5, BAND), Verdict::Ok);
        assert_eq!(judge(id, 20.0, 21.9, BAND), Verdict::Ok);
        assert_eq!(judge(id, 20.0, 25.0, BAND), Verdict::Regressed);
        assert_eq!(judge(id, 20.0, 19.0, BAND), Verdict::Ok);
        assert_eq!(judge(id, 20.0, 10.0, BAND), Verdict::Improved);
    }

    /// The noise floor is what keeps a near-zero baseline from gating on floating-point dust.
    #[test]
    fn the_noise_floor_widens_the_band_near_zero() {
        let id = MetricId::VerticalBias;
        // A 1 mm bias against a 1 mm baseline: a relative band alone would call 2 mm a
        // regression, which is meaningless in metres.
        assert_eq!(judge(id, 0.001, 0.002, BAND), Verdict::Ok);
        assert_eq!(judge(id, 0.001, 0.1, BAND), Verdict::Regressed);
    }

    /// A metric whose ideal is not zero is judged on its distance from that ideal.
    #[test]
    fn a_toward_target_metric_improves_by_approaching_its_target() {
        let id = MetricId::NpesPosition;
        // An over-conservative filter, two orders of magnitude below the consistent value.
        assert_eq!(judge(id, 0.0002, 0.0003, BAND), Verdict::Ok);
        // Fixing the covariance moves it to the target, which is an improvement, not a
        // regression -- and the baseline recording 0.0002 is now stale.
        assert_eq!(judge(id, 0.0002, 3.0, BAND), Verdict::Improved);
        // Overshooting into over-confidence is a regression on the far side.
        assert_eq!(judge(id, 3.0, 12.0, BAND), Verdict::Regressed);
    }

    /// Three-sigma containment measured at 1.000 sits 0.0027 from its ideal, which is inside
    /// the fractional noise floor -- so it gates on a real collapse and not on drift.
    #[test]
    fn containment_gates_on_a_collapse_not_on_drift() {
        let id = MetricId::Containment3SigmaHorizontal;
        assert_eq!(judge(id, 1.0, 0.995, BAND), Verdict::Ok);
        assert_eq!(judge(id, 1.0, 0.99, BAND), Verdict::Ok);
        assert_eq!(judge(id, 1.0, 0.90, BAND), Verdict::Regressed);
    }

    #[test]
    fn rounding_keeps_six_significant_figures() {
        assert!((round_significant(23.482_137_9, 6) - 23.4821).abs() < 1e-9);
        assert!((round_significant(0.000_123_456_789, 6) - 0.000_123_457).abs() < 1e-12);
        assert_eq!(round_significant(0.0, 6), 0.0);
        assert!(round_significant(f64::NAN, 6).is_nan());
    }

    /// Every scenario id must be unique, or one silently overwrites another in the file.
    #[test]
    fn scenario_ids_are_unique() {
        let ids: BTreeSet<String> = scenarios().into_iter().map(|s| s.id).collect();
        assert_eq!(ids.len(), scenarios().len());
    }
}
