//! `geonav`'s event builder must agree with `core`'s on everything that is not geophysical.
//!
//! #411. `geonav::build_event_stream` used to reimplement the whole of
//! `strapdown::messages::build_event_stream` -- the elapsed clock, the GNSS schedule, the
//! fault model, the barometer, the IMU events -- and the two had diverged in three ways, each
//! of which changed what a `--geo` run simulated:
//!
//! * `DutyCycle` emitted **one** GNSS fix per ON window rather than every fix during it,
//!   because the copy toggled a flag on each emit instead of deriving the window from elapsed
//!   time. `--on-s 100 --off-s 50` delivered a fix every 150 s instead of for 100 s in every
//!   150.
//! * `start_phase_s` was destructured away and never applied.
//! * The barometer was emitted on every record, ignoring `baro_scheduler` -- 50 updates a
//!   second on a 50 Hz log against the 1 Hz everything else uses -- and no magnetometer
//!   update was emitted at all.
//!
//! Every geonav test passed throughout. They exercised the geophysical measurements, which
//! were correct, and never compared the scheduling against the implementation it shadowed.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    reason = "integration tests are a separate crate target, so `clippy.toml`'s \
              allow-unwrap-in-tests does not reach them; unwrapping is how these assert"
)]

use chrono::{TimeZone, Utc};
use strapdown::messages::{
    AidingConfig, Event, GnssFaultModel, MeasurementScheduler, build_event_stream,
};
use strapdown::sim::TestDataRecord;

use geonav::{GeophysicalAiding, build_event_stream as geo_build_event_stream};

/// One record per second, carrying everything the builders read.
fn track(samples: usize) -> Vec<TestDataRecord> {
    let start = Utc.with_ymd_and_hms(2025, 3, 1, 0, 0, 0).unwrap();
    (0..samples)
        .map(|i| TestDataRecord {
            time: start + chrono::Duration::seconds(i64::try_from(i).unwrap_or(i64::MAX)),
            latitude: 40.05 + f64::from(u32::try_from(i).unwrap_or(u32::MAX)) * 1e-5,
            longitude: -75.95,
            altitude: 100.0,
            speed: 1.0,
            bearing: 0.0,
            horizontal_accuracy: 5.0,
            vertical_accuracy: 3.0,
            speed_accuracy: 0.5,
            bearing_accuracy: 1.0,
            acc_z: -9.81,
            relative_altitude: 1.0,
            mag_x: 20.0,
            mag_y: 0.0,
            mag_z: 45.0,
            qw: 1.0,
            ..Default::default()
        })
        .collect()
}

/// A comparable shape for one event: what kind it is and when it happened.
///
/// The measurement itself is a `Box<dyn MeasurementModel>` and cannot be compared, so the
/// sequence is compared by kind and time -- which is exactly what "the same schedule" means,
/// and what all three defects changed.
fn shape(events: &[Event]) -> Vec<(&'static str, i64)> {
    events
        .iter()
        .map(|event| match event {
            Event::Imu { elapsed_s, .. } => ("imu", (elapsed_s * 1000.0).round() as i64),
            Event::Measurement { elapsed_s, .. } => {
                ("measurement", (elapsed_s * 1000.0).round() as i64)
            }
        })
        .collect()
}

/// With no maps loaded, the geophysical builder must produce precisely the core stream.
///
/// #411 acceptance criterion 1. This is the assertion that makes the delegation load-bearing:
/// if anyone reimplements the scheduling here, this fails.
#[test]
fn the_two_builders_agree_on_every_non_geophysical_event() {
    let records = track(400);

    for (name, scheduler) in [
        ("pass_through", MeasurementScheduler::PassThrough),
        (
            "fixed_interval",
            MeasurementScheduler::FixedInterval {
                interval_s: 5.0,
                phase_s: 2.0,
            },
        ),
        (
            "duty_cycle",
            MeasurementScheduler::DutyCycle {
                on_s: 100.0,
                off_s: 50.0,
                start_phase_s: 30.0,
            },
        ),
    ] {
        let mut config = AidingConfig::default();
        config.scheduler = scheduler;
        config.fault = GnssFaultModel::None;

        let core = build_event_stream(&records, &config, false).unwrap();
        let geo = geo_build_event_stream(&records, &config, false, &GeophysicalAiding::default())
            .unwrap();

        assert_eq!(
            geo.start_time, core.start_time,
            "`{name}`: the two builders disagree on the stream's epoch"
        );
        assert_eq!(
            shape(&geo.events),
            shape(&core.events),
            "`{name}`: the geophysical builder's non-geophysical events differ from the core \
             builder's. With no maps loaded the two must be identical -- anything else is a \
             second copy of the scheduler drifting from the first (#411)."
        );
    }
}

/// A duty cycle must deliver every fix inside its ON window, not one at the boundary.
///
/// #411 acceptance criterion 2. The old implementation returned `duty_on` only on the step
/// that toggled it, so 100 s ON / 50 s OFF produced a single fix every 150 s.
#[test]
fn a_duty_cycle_delivers_every_fix_inside_the_on_window() {
    let records = track(450);
    let mut config = AidingConfig::default();
    config.scheduler = MeasurementScheduler::DutyCycle {
        on_s: 100.0,
        off_s: 50.0,
        start_phase_s: 0.0,
    };
    config.fault = GnssFaultModel::None;
    // Silence the other two channels so the count below is GNSS alone.
    config.baro_scheduler = MeasurementScheduler::FixedInterval {
        interval_s: 1.0e9,
        phase_s: 1.0e9,
    };
    config.magnetometer_scheduler = MeasurementScheduler::FixedInterval {
        interval_s: 1.0e9,
        phase_s: 1.0e9,
    };

    let stream =
        geo_build_event_stream(&records, &config, false, &GeophysicalAiding::default()).unwrap();
    let fixes = stream
        .events
        .iter()
        .filter(|e| matches!(e, Event::Measurement { .. }))
        .count();

    // 449 one-second windows, of which roughly two thirds fall in an ON phase. The old
    // behaviour produced 3 -- one per 150 s cycle.
    // Two-sided on purpose. A lower bound alone passes both for the correct 300 *and* for a
    // scheduler that has stopped filtering altogether and emits all 449 -- which is the other
    // way this can break. The measured value is exactly 300: 100 s of every 150 over 449
    // one-second windows.
    assert!(
        (250..350).contains(&fixes),
        "a 100 s ON / 50 s OFF duty cycle over 449 one-second windows delivered {fixes} \
         fixes. It should be near 300 -- two thirds of them. A handful means the scheduler \
         emits only on the transition into ON (#411's defect); close to 449 means it has \
         stopped gating on the window at all."
    );
}

/// `start_phase_s` must shift the window on the geophysical path too.
///
/// #411 acceptance criterion 3. The old implementation destructured it away with `..`.
#[test]
fn the_duty_cycle_start_phase_is_honoured() {
    let records = track(200);

    let with_phase = |start_phase_s: f64| {
        let mut config = AidingConfig::default();
        config.scheduler = MeasurementScheduler::DutyCycle {
            on_s: 20.0,
            off_s: 20.0,
            start_phase_s,
        };
        config.fault = GnssFaultModel::None;
        config.baro_scheduler = MeasurementScheduler::FixedInterval {
            interval_s: 1.0e9,
            phase_s: 1.0e9,
        };
        config.magnetometer_scheduler = MeasurementScheduler::FixedInterval {
            interval_s: 1.0e9,
            phase_s: 1.0e9,
        };
        let stream =
            geo_build_event_stream(&records, &config, false, &GeophysicalAiding::default())
                .unwrap();
        shape(&stream.events)
            .into_iter()
            .filter(|(kind, _)| *kind == "measurement")
            .map(|(_, ms)| ms)
            .collect::<Vec<_>>()
    };

    let unshifted = with_phase(0.0);
    let shifted = with_phase(20.0);

    assert_ne!(
        unshifted, shifted,
        "shifting `start_phase_s` by a full ON window produced an identical fix schedule, so \
         the phase is being ignored on the geophysical path (#411)."
    );
}
