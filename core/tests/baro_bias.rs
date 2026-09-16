//! The barometric bias state is *estimated*, in all three Kalman filters (#372).
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    reason = "integration tests are a separate crate target, so `clippy.toml`'s \
              allow-unwrap-in-tests -- which only covers `#[cfg(test)]` items -- does not \
              reach them; unwrapping is how these assert"
)]
//!
//! # Why this is an integration test and not three unit tests
//!
//! #372's failure mode is not a wrong number, it is a state that is *carried but never
//! observed*: seeded, propagated, written out, and with a Kalman gain of identically zero.
//! That is #394's defect -- six EKF bias states inert for the whole of the project's history --
//! and it was invisible to a suite that read only RMSEs, because a state nothing corrects
//! changes no RMSE. It reaches the barometer by a second route: a nine-column Jacobian padded
//! out to the filter's width puts a **zero** in the bias column, which
//! `expand_measurement_jacobian` accepts without complaint.
//!
//! So the assertions here are about the covariance, not the trajectory:
//!
//! * the variance of the bias must fall **below its prior** -- a state only loses variance by
//!   being measured;
//! * the estimate must move **off its zero seed**;
//! * and the measurement's own Jacobian must carry a non-zero bias column at full width.
//!
//! An RMSE bound would pass on a filter that never touched the state. These cannot.

use nalgebra::DVector;
use strapdown::NavigationFilter;
use strapdown::measurements::{MeasurementModel, RelativeAltitudeMeasurement};
use strapdown::messages::{
    GnssDegradationConfig, GnssFaultModel, GnssScheduler, build_event_stream,
};
use strapdown::sim::{
    EkfConfig, EskfConfig, INITIAL_BARO_BIAS_VARIANCE_M2, TestDataRecord, UkfConfig,
    initialize_ekf, initialize_eskf, initialize_ukf,
};

/// `core/tests/test_data.csv` is a Sensor Logger export, which is ENU.
const REAL_DATA_IS_ENU: bool = true;

/// Records taken from the front of the recording. Enough for the bias to converge -- it
/// settles within the first minutes -- without paying for the whole log three times.
const SAMPLES: usize = 400;

/// Where all three constructors put the bias. Asserted against each config's own
/// `baro_bias_index` below rather than assumed.
const BARO_INDEX: usize = 15;

fn records() -> Vec<TestDataRecord> {
    let mut records = TestDataRecord::from_csv(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/test_data.csv"),
    )
    .expect("core/tests/test_data.csv must load");
    assert!(
        records.len() > SAMPLES,
        "test_data.csv parsed to only {} records",
        records.len()
    );
    records.truncate(SAMPLES);
    records
}

/// Drive one filter over the recording with the barometer aiding it, and assert that the
/// bias state was estimated rather than merely carried.
///
/// Generic rather than `&mut dyn NavigationFilter` because `run_closed_loop_with_geo` is
/// generic and its type parameter is `Sized`; the three call sites below are the loop.
fn check<F: NavigationFilter>(name: &str, filter: &mut F, records: &[TestDataRecord]) {
    assert_eq!(
        filter.get_estimate().len(),
        16,
        "{name} did not open with a sixteenth state"
    );
    let stream = build_event_stream(
        records,
        &GnssDegradationConfig {
            scheduler: GnssScheduler::PassThrough,
            fault: GnssFaultModel::None,
            baro_bias_index: Some(BARO_INDEX),
            ..Default::default()
        },
        REAL_DATA_IS_ENU,
    )
    .expect("event stream");
    // Deliberately the plain runner, not `run_closed_loop_with_geo`. `GeoStateLayout::NONE`
    // is fifteen wide and the conversion into `NavigationResult` asserts the state matches,
    // so before #372 wired `NavigationFilter::baro_bias_index` through, this call panicked on
    // every filter carrying the state -- a `pub` config flag that broke the documented runner.
    let solution = strapdown::sim::run_closed_loop(filter, stream, None, None)
        .expect("the plain runner must handle a filter that estimates a barometric bias");
    let reported = solution
        .last()
        .and_then(|row| row.baro_bias)
        .expect("the solution must carry the barometric bias column");

    let bias = filter.get_estimate()[BARO_INDEX];
    let variance = filter.get_certainty()[(BARO_INDEX, BARO_INDEX)];
    // The runner labelled the right state: a layout off by one would report a gyro bias here.
    assert!(
        (reported - bias).abs() < 1e-12,
        "{name} wrote {reported} into the barometric column but holds {bias} in the state"
    );
    assert_eq!(
        filter.baro_bias_index(),
        Some(BARO_INDEX),
        "{name} does not report the index the runner needs"
    );

    // A state that is never corrected keeps its seed exactly. This is the assertion #394
    // did not have.
    assert!(
        bias.abs() > 1e-6,
        "{name} left the barometric bias at its zero seed ({bias:e} m), which is what an \
         unobservable state looks like"
    );
    // And it only loses variance by being measured; process noise can only add.
    assert!(
        variance < INITIAL_BARO_BIAS_VARIANCE_M2,
        "{name} ended with the bias variance at {variance:.4} m^2, no better than its \
         {INITIAL_BARO_BIAS_VARIANCE_M2:.4} m^2 prior -- the state is being propagated but \
         not observed"
    );
    // Physically: a hectopascal of reference drift is 8.3 m, so a converged bias on this
    // recording belongs in single-digit metres. A hundred would mean the state had absorbed
    // something that is not a barometric offset.
    assert!(
        bias.abs() < 20.0,
        "{name} put {bias:.3} m into the barometric bias, far outside the ~8.3 m a \
         hectopascal of reference-pressure drift can explain"
    );
}

#[test]
fn every_filter_actually_estimates_the_barometric_bias() {
    let records = records();

    let ukf_config = UkfConfig {
        is_enu: REAL_DATA_IS_ENU,
        estimate_baro_bias: true,
        ..UkfConfig::default()
    };
    let ekf_config = EkfConfig {
        is_enu: REAL_DATA_IS_ENU,
        estimate_baro_bias: true,
        ..EkfConfig::default()
    };
    let eskf_config = EskfConfig {
        is_enu: REAL_DATA_IS_ENU,
        estimate_baro_bias: true,
        ..EskfConfig::default()
    };
    // The index this file drives the measurement at has to be the one each constructor
    // actually used, or the assertions below would be reading a gyro bias.
    assert_eq!(ukf_config.baro_bias_index(), Some(BARO_INDEX));
    assert_eq!(ekf_config.baro_bias_index(), Some(BARO_INDEX));
    assert_eq!(eskf_config.baro_bias_index(), Some(BARO_INDEX));

    let mut ukf = initialize_ukf(&records[0], ukf_config).expect("UKF");
    let mut ekf = initialize_ekf(&records[0], ekf_config).expect("EKF");
    let mut eskf = initialize_eskf(&records[0], eskf_config).expect("ESKF");

    check("UKF", &mut ukf, &records);
    check("EKF", &mut ekf, &records);
    check("ESKF", &mut eskf, &records);
}

#[test]
fn the_barometers_jacobian_carries_a_non_zero_bias_column() {
    // The whole reason `relative_altitude_bias_jacobian` returns the filter's full width.
    // Nine columns padded to sixteen would put a zero here, and nothing would report it.
    let baro = RelativeAltitudeMeasurement {
        relative_altitude: 3.0,
        reference_altitude: 100.0,
        noise_std: strapdown::measurements::BAROMETRIC_ALTITUDE_NOISE_M,
        bias_index: Some(BARO_INDEX),
    };
    let state = DVector::from_element(16, 0.0);
    let h = baro.get_jacobian(&state).expect("jacobian");
    assert_eq!(h.ncols(), 16, "the row must span the whole state");
    assert!(
        (h[(0, BARO_INDEX)] - 1.0).abs() < f64::EPSILON,
        "the bias column is {}, not 1 -- the state would be unobservable",
        h[(0, BARO_INDEX)]
    );
    assert!(
        (h[(0, 2)] - 1.0).abs() < f64::EPSILON,
        "the altitude column is {}, not 1",
        h[(0, 2)]
    );

    // And a state too short to hold the declared index is rejected rather than read off the
    // end -- `ZaruMeasurement::require_bias_states`' idiom.
    let short = DVector::from_element(9, 0.0);
    assert!(
        baro.get_measurement(&short).is_err(),
        "a 9-element state cannot hold a bias at index {BARO_INDEX}"
    );
}

#[test]
fn the_state_is_absent_unless_asked_for() {
    let records = records();
    let ukf = initialize_ukf(
        &records[0],
        UkfConfig {
            is_enu: REAL_DATA_IS_ENU,
            ..UkfConfig::default()
        },
    )
    .expect("UKF");
    let eskf = initialize_eskf(
        &records[0],
        EskfConfig {
            is_enu: REAL_DATA_IS_ENU,
            ..EskfConfig::default()
        },
    )
    .expect("ESKF");
    assert_eq!(ukf.get_estimate().len(), 15);
    assert_eq!(eskf.get_estimate().len(), 15);
    assert_eq!(UkfConfig::default().baro_bias_index(), None);
    assert_eq!(EskfConfig::default().baro_bias_index(), None);
}
