//! Variance-based stationary detection for zero-velocity and zero-angular-rate aiding.
//!
//! An unaided strapdown solution drifts whether or not the vehicle is moving, and a
//! prolonged stop is the worst case: position error grows cubically in time with
//! accelerometer bias while the true position does not move at all. It is also the
//! easiest case to fix, because standing still is itself information. Detecting the
//! stop turns "the vehicle is stationary" into two pseudo-measurements --
//! [`ZuptMeasurement`] on velocity and [`ZaruMeasurement`] on gyro bias -- that the
//! filter can consume like any other aiding source.
//!
//! # The detector
//!
//! [`StationaryDetector`] keeps a sliding window of the most recent IMU samples and
//! declares the platform stationary when all four of these hold across the window:
//!
//! 1. **Specific-force variance** below [`StationaryConfig::accel_variance_threshold`].
//!    A moving vehicle shakes; a parked one mostly does not.
//! 2. **Specific-force magnitude** within [`StationaryConfig::gravity_tolerance_mps2`]
//!    of local gravity. A platform in free fall or under sustained acceleration is not
//!    stationary even when the variance is briefly small.
//! 3. **Angular-rate variance** below [`StationaryConfig::gyro_variance_threshold`].
//! 4. **Mean angular-rate magnitude** below [`StationaryConfig::gyro_mean_threshold`].
//!    This is what separates a stop from a steady turn: a constant-rate turn has
//!    near-zero gyro *variance* and a large mean, and applying ZARU to it would drive
//!    the gyro bias estimate straight into the turn rate.
//!
//! Conditions 2 and 4 are the reason this is not a bare variance test. The remaining
//! false positive is genuine constant-velocity straight-line cruise on a smooth
//! surface, which satisfies all four; in vehicle applications that is ruled out by an
//! odometer or by wheel speed, neither of which this crate models. Choose window and
//! thresholds accordingly, and prefer a conservative (small) variance threshold: a
//! missed stop costs an aiding opportunity, whereas a false stop injects a hard
//! zero-velocity constraint into a moving solution.
//!
//! # Hysteresis
//!
//! The window must be full before any verdict is issued, and
//! [`StationaryConfig::min_stationary_samples`] consecutive stationary windows are
//! required before the detector latches on. Release is immediate: one window that
//! fails any condition drops the verdict on the same sample. Slow to trust, quick to
//! let go, because the asymmetry of the two error cases above is itself asymmetric.
//!
//! # Example
//!
//! ```rust
//! use nalgebra::Vector3;
//! use strapdown::IMUData;
//! use strapdown::stationary::{StationaryConfig, StationaryDetector};
//!
//! let mut detector = StationaryDetector::new(StationaryConfig::default());
//! let at_rest = IMUData {
//!     accel: Vector3::new(0.0, 0.0, 9.81),
//!     gyro: Vector3::zeros(),
//! };
//!
//! // The window has to fill, and then the dwell has to elapse, before it latches.
//! let mut latched = false;
//! for _ in 0..200 {
//!     latched = detector.push(&at_rest);
//! }
//! assert!(latched);
//!
//! // One sample of real motion releases it immediately.
//! let driving = IMUData {
//!     accel: Vector3::new(3.0, 0.0, 9.81),
//!     gyro: Vector3::new(0.0, 0.0, 0.4),
//! };
//! assert!(!detector.push(&driving));
//! ```
//!
//! # References
//!
//! - Groves, P. D., *Principles of GNSS, Inertial, and Multisensor Integrated
//!   Navigation Systems*, 2nd ed., Section 15.2 (pedestrian dead reckoning; ZUPT).
//! - Skog, I., Handel, P., Nilsson, J.-O., Rantakokko, J., "Zero-Velocity Detection --
//!   An Algorithm Evaluation", *IEEE Trans. Biomed. Eng.* 57(11), 2010.
//!
//! [`ZuptMeasurement`]: crate::measurements::ZuptMeasurement
//! [`ZaruMeasurement`]: crate::measurements::ZaruMeasurement

use std::collections::VecDeque;

use nalgebra::Vector3;
use serde::{Deserialize, Serialize};

use crate::{IMUData, ImuSample, StrapdownError};

/// Nominal gravity magnitude used by the specific-force magnitude check, m/s^2.
///
/// The check has a tolerance of order 0.5 m/s^2, which is two orders of magnitude
/// wider than the ~0.05 m/s^2 that gravity varies by over the WGS84 ellipsoid, so a
/// constant is enough here and the detector does not need a position to run.
pub const NOMINAL_GRAVITY_MPS2: f64 = 9.807;

/// Thresholds and window length for [`StationaryDetector`].
///
/// The defaults are tuned for a consumer-grade MEMS IMU at ~100 Hz, the regime the
/// Sensor Logger datasets in this repository come from. Tactical-grade hardware
/// supports thresholds one to two orders of magnitude tighter; a noisier unit needs
/// them loosened, or it will never declare a stop.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct StationaryConfig {
    /// Number of samples in the sliding window.
    ///
    /// At 100 Hz the default is a 1 s window: long enough to average out sensor
    /// noise, short enough to catch the end of a stop before the vehicle has gone
    /// anywhere.
    pub window: usize,
    /// Maximum trace of the specific-force sample covariance, (m/s^2)^2.
    pub accel_variance_threshold: f64,
    /// Maximum deviation of mean specific-force magnitude from
    /// [`NOMINAL_GRAVITY_MPS2`], m/s^2.
    pub gravity_tolerance_mps2: f64,
    /// Maximum trace of the angular-rate sample covariance, (rad/s)^2.
    pub gyro_variance_threshold: f64,
    /// Maximum mean angular-rate magnitude, rad/s.
    ///
    /// Must stay above Earth rate (7.29e-5 rad/s), which a perfectly stationary
    /// gyro genuinely senses, or a good IMU will never be declared stationary.
    pub gyro_mean_threshold: f64,
    /// Consecutive stationary windows required before the verdict latches.
    pub min_stationary_samples: usize,
}

impl Default for StationaryConfig {
    fn default() -> Self {
        Self {
            window: 100,
            accel_variance_threshold: 0.05,
            gravity_tolerance_mps2: 0.5,
            gyro_variance_threshold: 1.0e-3,
            gyro_mean_threshold: 5.0e-3,
            min_stationary_samples: 50,
        }
    }
}

impl StationaryConfig {
    /// Validate the configuration.
    ///
    /// # Errors
    /// [`StrapdownError::InvalidConfiguration`] if the window is empty or any
    /// threshold is not finite and positive. A zero window would make the variance
    /// undefined and a negative threshold would reject every sample, both of which
    /// are silent failures once the detector is running.
    pub fn validate(&self) -> Result<(), StrapdownError> {
        if self.window == 0 {
            return Err(StrapdownError::InvalidConfiguration {
                field: "stationary detector window",
                reason: "window must contain at least one sample".to_owned(),
            });
        }
        let thresholds = [
            self.accel_variance_threshold,
            self.gravity_tolerance_mps2,
            self.gyro_variance_threshold,
            self.gyro_mean_threshold,
        ];
        if !thresholds.iter().all(|t| t.is_finite() && *t > 0.0) {
            return Err(StrapdownError::InvalidConfiguration {
                field: "stationary detector thresholds",
                reason: "every threshold must be finite and strictly positive".to_owned(),
            });
        }
        Ok(())
    }
}

/// Sliding-window variance detector for stationary intervals.
///
/// Feed it every IMU sample with [`push`](Self::push); it returns the current
/// verdict. See the [module documentation](self) for what the verdict means and
/// what it cannot see.
#[derive(Clone, Debug)]
pub struct StationaryDetector {
    config: StationaryConfig,
    accel_window: VecDeque<Vector3<f64>>,
    gyro_window: VecDeque<Vector3<f64>>,
    consecutive_stationary: usize,
}

impl StationaryDetector {
    /// Build a detector with the given configuration.
    ///
    /// Takes the config unvalidated so this stays infallible for the common
    /// `StationaryConfig::default()` path; call [`StationaryConfig::validate`] first
    /// when the values came from a config file. An invalid window of 0 makes the
    /// detector permanently report "moving" rather than panicking.
    #[must_use]
    pub fn new(config: StationaryConfig) -> Self {
        Self {
            accel_window: VecDeque::with_capacity(config.window),
            gyro_window: VecDeque::with_capacity(config.window),
            config,
            consecutive_stationary: 0,
        }
    }

    /// The configuration this detector was built with.
    #[must_use]
    pub const fn config(&self) -> &StationaryConfig {
        &self.config
    }

    /// Add one IMU sample and return the current verdict.
    ///
    /// `true` means the platform has been stationary for at least
    /// [`StationaryConfig::min_stationary_samples`] consecutive full windows.
    pub fn push(&mut self, imu: &IMUData) -> bool {
        if self.config.window == 0 {
            return false;
        }
        if !imu
            .accel
            .iter()
            .chain(imu.gyro.iter())
            .all(|v| v.is_finite())
        {
            // A non-finite sample poisons the running variance for a whole window.
            // Drop the history and start over rather than latch on a NaN comparison,
            // which is false for every threshold test and so would read as "moving"
            // for the window anyway -- this just makes that explicit and recoverable.
            self.reset();
            return false;
        }

        if self.accel_window.len() == self.config.window {
            self.accel_window.pop_front();
            self.gyro_window.pop_front();
        }
        self.accel_window.push_back(imu.accel);
        self.gyro_window.push_back(imu.gyro);

        if self.accel_window.len() < self.config.window {
            // A partial window has an optimistically small variance: the first two
            // samples of hard braking look stationary. Withhold the verdict instead.
            self.consecutive_stationary = 0;
            return false;
        }

        if self.window_is_stationary() {
            self.consecutive_stationary = self.consecutive_stationary.saturating_add(1);
        } else {
            self.consecutive_stationary = 0;
        }
        self.is_stationary()
    }

    /// Add one delta-v/delta-theta sample and return the current verdict.
    ///
    /// Converts to average rates over the interval and defers to [`push`](Self::push);
    /// the thresholds are expressed in rate units so they do not move with the sample
    /// interval.
    ///
    /// # Errors
    /// [`StrapdownError::OutOfRange`] if `sample.dt` is not strictly positive, since
    /// the rates are undefined then. The detector's history is left untouched.
    pub fn push_sample(&mut self, sample: &ImuSample) -> Result<bool, StrapdownError> {
        Ok(self.push(&sample.to_rates()?))
    }

    /// The current verdict, without adding a sample.
    #[must_use]
    pub const fn is_stationary(&self) -> bool {
        self.consecutive_stationary >= self.config.min_stationary_samples
    }

    /// How many consecutive stationary windows have been seen.
    ///
    /// Exposed for logging and for tests that need to see the dwell filling up
    /// rather than only its latched result.
    #[must_use]
    pub const fn consecutive_stationary(&self) -> usize {
        self.consecutive_stationary
    }

    /// Discard the window and the dwell counter.
    ///
    /// Call after a discontinuity in the sample stream -- a gap in the data, a
    /// re-initialization -- so the variance is not computed across samples that were
    /// never adjacent in time.
    pub fn reset(&mut self) {
        self.accel_window.clear();
        self.gyro_window.clear();
        self.consecutive_stationary = 0;
    }

    /// Whether the current full window satisfies all four stationary conditions.
    fn window_is_stationary(&self) -> bool {
        let (accel_mean, accel_variance) = window_statistics(&self.accel_window);
        let (gyro_mean, gyro_variance) = window_statistics(&self.gyro_window);

        let gravity_error = (accel_mean.norm() - NOMINAL_GRAVITY_MPS2).abs();

        accel_variance <= self.config.accel_variance_threshold
            && gravity_error <= self.config.gravity_tolerance_mps2
            && gyro_variance <= self.config.gyro_variance_threshold
            && gyro_mean.norm() <= self.config.gyro_mean_threshold
    }
}

/// Sample mean and total variance (trace of the sample covariance) of a window.
///
/// Returns a zero mean and zero variance for an empty window; callers only reach
/// this with a full one.
fn window_statistics(window: &VecDeque<Vector3<f64>>) -> (Vector3<f64>, f64) {
    let count = window.len();
    if count == 0 {
        return (Vector3::zeros(), 0.0);
    }
    let mean = window.iter().sum::<Vector3<f64>>() / count as f64;
    // Trace of the sample covariance: sum over axes of the per-axis variance, which
    // is the same as the mean squared distance from the mean vector. Normalized by
    // N rather than N-1 -- the window is fixed-length and the thresholds are
    // calibrated against this definition, so the distinction is a constant factor
    // of at most 1% at the default window of 100.
    let variance = window
        .iter()
        .map(|sample| (sample - mean).norm_squared())
        .sum::<f64>()
        / count as f64;
    (mean, variance)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use rand_distr::{Distribution, Normal};

    /// Samples needed before [`test_config`] can latch: the window has to fill, and
    /// the sample that fills it is also the first of the dwell.
    const LATCH_SAMPLES: usize = 20 + 10 - 1;

    /// A short window keeps the tests fast; the dwell is kept proportionally short.
    fn test_config() -> StationaryConfig {
        StationaryConfig {
            window: 20,
            min_stationary_samples: 10,
            ..StationaryConfig::default()
        }
    }

    /// Feed `count` samples drawn from `sample` and return the final verdict.
    fn feed(
        detector: &mut StationaryDetector,
        count: usize,
        mut sample: impl FnMut() -> IMUData,
    ) -> bool {
        let mut verdict = false;
        for _ in 0..count {
            verdict = detector.push(&sample());
        }
        verdict
    }

    /// Noisy but genuinely stationary MEMS output: gravity on z, noise on everything.
    fn resting_imu(rng: &mut StdRng) -> IMUData {
        let accel_noise = Normal::new(0.0, 0.02).unwrap();
        let gyro_noise = Normal::new(0.0, 0.002).unwrap();
        IMUData {
            accel: Vector3::new(
                accel_noise.sample(rng),
                accel_noise.sample(rng),
                NOMINAL_GRAVITY_MPS2 + accel_noise.sample(rng),
            ),
            gyro: Vector3::new(
                gyro_noise.sample(rng),
                gyro_noise.sample(rng),
                gyro_noise.sample(rng),
            ),
        }
    }

    #[test]
    fn detects_a_noisy_but_stationary_platform() {
        let mut rng = StdRng::seed_from_u64(20_260_912);
        let mut detector = StationaryDetector::new(test_config());
        assert!(feed(&mut detector, 200, || resting_imu(&mut rng)));
    }

    #[test]
    fn withholds_a_verdict_until_the_window_is_full() {
        // A partial window has an optimistically small variance -- the first two
        // samples of hard braking look stationary -- so there must be no verdict
        // before it fills, and none before the dwell elapses either.
        let mut rng = StdRng::seed_from_u64(1);
        let mut detector = StationaryDetector::new(test_config());
        // The sample that fills the window also produces the first complete window,
        // so it counts as the first of the dwell: the verdict turns true on sample
        // `window + min_stationary_samples - 1` = 20 + 10 - 1 = 29, not 30.
        for step in 0..LATCH_SAMPLES - 1 {
            assert!(
                !detector.push(&resting_imu(&mut rng)),
                "latched after only {} samples",
                step + 1
            );
        }
        assert!(detector.push(&resting_imu(&mut rng)));
    }

    #[test]
    fn rejects_a_vibrating_platform() {
        // Engine idle: gravity is still there on average, but the variance is not.
        let mut rng = StdRng::seed_from_u64(2);
        let shake = Normal::new(0.0, 1.5).unwrap();
        let mut detector = StationaryDetector::new(test_config());
        let latched = feed(&mut detector, 200, || IMUData {
            accel: Vector3::new(
                shake.sample(&mut rng),
                shake.sample(&mut rng),
                NOMINAL_GRAVITY_MPS2 + shake.sample(&mut rng),
            ),
            gyro: Vector3::zeros(),
        });
        assert!(!latched);
    }

    #[test]
    fn rejects_a_steady_turn() {
        // The case bare variance cannot see: a constant-rate turn has near-zero gyro
        // variance and a large mean. Without the mean check, ZARU would be applied
        // here and would drive the gyro bias estimate straight into the turn rate.
        let mut detector = StationaryDetector::new(test_config());
        let latched = feed(&mut detector, 200, || IMUData {
            accel: Vector3::new(0.0, 0.0, NOMINAL_GRAVITY_MPS2),
            gyro: Vector3::new(0.0, 0.0, 0.35),
        });
        assert!(!latched, "a steady 0.35 rad/s turn was declared stationary");
    }

    #[test]
    fn rejects_sustained_acceleration() {
        // Constant 4 m/s^2 forward acceleration: zero variance on both sensors, zero
        // gyro mean, but the specific-force magnitude is wrong for a platform at
        // rest. Only the gravity check separates this from a genuine stop.
        let mut detector = StationaryDetector::new(test_config());
        let latched = feed(&mut detector, 200, || IMUData {
            accel: Vector3::new(4.0, 0.0, NOMINAL_GRAVITY_MPS2),
            gyro: Vector3::zeros(),
        });
        assert!(!latched, "sustained acceleration was declared stationary");
    }

    #[test]
    fn rejects_free_fall() {
        let mut detector = StationaryDetector::new(test_config());
        let latched = feed(&mut detector, 200, || IMUData {
            accel: Vector3::zeros(),
            gyro: Vector3::zeros(),
        });
        assert!(!latched, "free fall was declared stationary");
    }

    #[test]
    fn releases_immediately_when_motion_resumes() {
        // Slow to trust, quick to let go: one bad window drops the verdict on the
        // same sample, because a false stop injects a hard zero-velocity constraint
        // into a moving solution.
        let mut rng = StdRng::seed_from_u64(3);
        let mut detector = StationaryDetector::new(test_config());
        assert!(feed(&mut detector, 200, || resting_imu(&mut rng)));

        let moving = IMUData {
            accel: Vector3::new(5.0, 2.0, NOMINAL_GRAVITY_MPS2),
            gyro: Vector3::new(0.0, 0.0, 0.5),
        };
        assert!(!detector.push(&moving), "verdict survived a moving sample");
        assert_eq!(detector.consecutive_stationary(), 0);
    }

    #[test]
    fn relatches_after_motion_stops_again() {
        let mut rng = StdRng::seed_from_u64(4);
        let mut detector = StationaryDetector::new(test_config());
        assert!(feed(&mut detector, 200, || resting_imu(&mut rng)));
        assert!(!feed(&mut detector, 30, || IMUData {
            accel: Vector3::new(5.0, 2.0, NOMINAL_GRAVITY_MPS2),
            gyro: Vector3::new(0.0, 0.0, 0.5),
        }));
        assert!(feed(&mut detector, 200, || resting_imu(&mut rng)));
    }

    #[test]
    fn tracks_a_stop_and_go_profile() {
        // End to end: drive, stop, drive. The detector should be quiet through both
        // driving phases and latched through the stop.
        let mut rng = StdRng::seed_from_u64(5);
        let mut detector = StationaryDetector::new(test_config());
        let driving = Normal::new(0.0, 0.8).unwrap();

        let mut verdicts = Vec::new();
        for phase in 0..3 {
            let stationary_phase = phase == 1;
            for _ in 0..120 {
                let sample = if stationary_phase {
                    resting_imu(&mut rng)
                } else {
                    IMUData {
                        accel: Vector3::new(
                            1.5 + driving.sample(&mut rng),
                            driving.sample(&mut rng),
                            NOMINAL_GRAVITY_MPS2 + driving.sample(&mut rng),
                        ),
                        gyro: Vector3::new(0.0, 0.0, 0.1 + driving.sample(&mut rng) * 0.05),
                    }
                };
                verdicts.push((stationary_phase, detector.push(&sample)));
            }
        }

        let false_positives = verdicts.iter().filter(|(truth, v)| !truth && *v).count();
        let detected = verdicts.iter().filter(|(truth, v)| *truth && *v).count();
        assert_eq!(false_positives, 0, "declared stationary while driving");
        // 120 stationary samples minus the LATCH_SAMPLES spent filling the window
        // and the dwell, less a little slack for the first windows after the
        // transition, which still contain driving samples.
        assert!(
            detected >= 120 - LATCH_SAMPLES - 5,
            "only {detected} of 120 stationary samples were detected"
        );
    }

    #[test]
    fn a_non_finite_sample_clears_the_window() {
        let mut rng = StdRng::seed_from_u64(6);
        let mut detector = StationaryDetector::new(test_config());
        assert!(feed(&mut detector, 200, || resting_imu(&mut rng)));

        let poisoned = IMUData {
            accel: Vector3::new(f64::NAN, 0.0, NOMINAL_GRAVITY_MPS2),
            gyro: Vector3::zeros(),
        };
        assert!(!detector.push(&poisoned));
        assert_eq!(detector.consecutive_stationary(), 0);
        // The history is gone, so the window has to refill before any new verdict.
        assert!(!feed(&mut detector, LATCH_SAMPLES - 1, || resting_imu(
            &mut rng
        )));
        assert!(detector.push(&resting_imu(&mut rng)));
    }

    #[test]
    fn reset_discards_history() {
        let mut rng = StdRng::seed_from_u64(7);
        let mut detector = StationaryDetector::new(test_config());
        assert!(feed(&mut detector, 200, || resting_imu(&mut rng)));
        detector.reset();
        assert!(!detector.is_stationary());
        assert!(!feed(&mut detector, LATCH_SAMPLES - 1, || resting_imu(
            &mut rng
        )));
        assert!(detector.push(&resting_imu(&mut rng)));
    }

    #[test]
    fn thresholds_are_expressed_in_rate_units() {
        // `push_sample` divides out dt, so the same physical motion gives the same
        // verdict whether the driver reports rates at 100 Hz or increments at 50 Hz.
        let mut rng = StdRng::seed_from_u64(8);
        let mut from_rates = StationaryDetector::new(test_config());
        let mut from_increments = StationaryDetector::new(test_config());

        for _ in 0..200 {
            let imu = resting_imu(&mut rng);
            let rate_verdict = from_rates.push(&imu);
            let sample = ImuSample::from_rates(&imu, 0.02);
            let increment_verdict = from_increments.push_sample(&sample).unwrap();
            assert_eq!(rate_verdict, increment_verdict);
        }
        assert!(from_increments.is_stationary());
    }

    #[test]
    fn push_sample_rejects_a_non_positive_interval() {
        let mut detector = StationaryDetector::new(test_config());
        let bad = ImuSample {
            delta_v: Vector3::zeros(),
            delta_theta: Vector3::zeros(),
            dt: 0.0,
        };
        let error = detector.push_sample(&bad).unwrap_err();
        assert!(matches!(error, StrapdownError::OutOfRange { .. }));
    }

    #[test]
    fn a_zero_window_reports_moving_rather_than_panicking() {
        // `new` takes the config unvalidated so the common default path stays
        // infallible; a nonsense window must then degrade, not divide by zero.
        let mut detector = StationaryDetector::new(StationaryConfig {
            window: 0,
            ..StationaryConfig::default()
        });
        for _ in 0..50 {
            assert!(!detector.push(&IMUData {
                accel: Vector3::new(0.0, 0.0, NOMINAL_GRAVITY_MPS2),
                gyro: Vector3::zeros(),
            }));
        }
    }

    #[test]
    fn validate_rejects_unusable_configurations() {
        assert!(StationaryConfig::default().validate().is_ok());
        assert!(
            StationaryConfig {
                window: 0,
                ..StationaryConfig::default()
            }
            .validate()
            .is_err()
        );
        for bad in [0.0, -1.0, f64::NAN] {
            assert!(
                StationaryConfig {
                    accel_variance_threshold: bad,
                    ..StationaryConfig::default()
                }
                .validate()
                .is_err(),
                "accepted accel threshold {bad}"
            );
            assert!(
                StationaryConfig {
                    gyro_mean_threshold: bad,
                    ..StationaryConfig::default()
                }
                .validate()
                .is_err(),
                "accepted gyro mean threshold {bad}"
            );
        }
    }

    #[test]
    fn default_gyro_mean_threshold_admits_earth_rate() {
        // A good gyro genuinely senses Earth rate while perfectly stationary. A
        // threshold below it would mean such a unit is never declared stationary --
        // the better the hardware, the worse the detector.
        let earth_rate = 7.292_115_9e-5;
        assert!(
            StationaryConfig::default().gyro_mean_threshold > earth_rate,
            "default threshold would reject a stationary tactical-grade gyro"
        );
    }

    #[test]
    fn config_round_trips_through_yaml() {
        let config = test_config();
        let yaml = serde_yaml::to_string(&config).unwrap();
        assert_eq!(
            serde_yaml::from_str::<StationaryConfig>(&yaml).unwrap(),
            config
        );
    }

    #[test]
    fn window_statistics_match_a_hand_computation() {
        let window: VecDeque<Vector3<f64>> = VecDeque::from(vec![
            Vector3::new(1.0, 0.0, 0.0),
            Vector3::new(3.0, 0.0, 0.0),
            Vector3::new(5.0, 0.0, 0.0),
        ]);
        let (mean, variance) = window_statistics(&window);
        assert!((mean[0] - 3.0).abs() < 1e-15);
        // Deviations are -2, 0, +2; mean squared distance is (4 + 0 + 4) / 3.
        assert!((variance - 8.0 / 3.0).abs() < 1e-15);
    }

    #[test]
    fn window_statistics_of_an_empty_window_are_zero() {
        let (mean, variance) = window_statistics(&VecDeque::new());
        assert!(mean.norm() < 1e-15);
        assert!(variance.abs() < 1e-15);
    }
}
