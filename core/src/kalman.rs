//! Kalman-style navigation filters (UKF/EKF/ESKF)
//!
//! This module contains the traditional Kalman filter style implementation of strapdown
//! inertial navigation systems. These filter build on the dead-reckoning functions
//! provided in the [crate] root module.

use crate::StrapdownError;
use crate::gating::{GatePolicy, GateRecovery, InnovationGate, UpdateOutcome};
use crate::linalg::{matrix_square_root, robust_spd_solve, symmetrize};
use crate::measurements::MeasurementModel;
use crate::{
    IMUData, ImuSample, NavigationFilter, StrapdownState, mechanize, wrap_to_180, wrap_to_pi,
};

use std::fmt::{self, Debug, Display};

use nalgebra::{DMatrix, DVector, Rotation3, UnitQuaternion, Vector3};

/// Initial navigation state used to seed filters.
///
/// This struct contains the minimal navigation state required to initialize
/// either the UKF or EKF implementations in this module. Fields represent
/// a local-level navigation solution (latitude, longitude, altitude, NED/ENU
/// velocity components, and Euler attitude angles). The `in_degrees` flag
/// indicates whether the provided angles/lat/lon are in degrees; see the note
/// on unit-tagged storage below for what that flag means downstream.
/// The `is_enu` flag determines whether the navigation frame is ENU (true)
/// or NED (false) for internal mechanization. It defaults to NED, matching
/// [`StrapdownState`] and the rest of the crate.
///
/// Field units and conventions:
/// - `latitude`, `longitude`: degrees if `in_degrees==true`, otherwise radians
/// - `altitude`: meters
/// - velocities: m/s (north, east, vertical)
/// - `roll`, `pitch`, `yaw`: degrees if `in_degrees==true`, otherwise radians --
///   the same rule as the position fields, which is what the filter constructors assume
///
/// Storage is *unit-tagged* rather than normalized: every angular field is held in the
/// units it was supplied in and `in_degrees` says which, so the flag has to travel with
/// the values. Converting to radians is the filter constructors' job, and they do it
/// exactly when the flag is set.
///
/// # Example
///
/// ```rust
/// use strapdown::kalman::InitialState;
/// // `None` selects the crate default, NED. Pass `Some(true)` for ENU.
/// let init = InitialState::new(45.0, -122.0, 100.0, 0.0, 0.0, 0.0,
///                              0.0, 0.0, 0.0, true, None);
/// ```
#[derive(Clone, Debug, Default)]
pub struct InitialState {
    /// Geodetic latitude, in degrees when `in_degrees` is set and in radians otherwise.
    ///
    /// [`InitialState::new`] passes this through unwrapped; only longitude is wrapped there.
    pub latitude: f64,
    /// Geodetic longitude, in degrees when `in_degrees` is set and in radians otherwise.
    ///
    /// [`InitialState::new`] wraps it to the range -180 to 180 degrees, or $-\pi$ to $\pi$
    /// radians, to match.
    pub longitude: f64,
    /// Height above the WGS84 ellipsoid in meters, positive up in both NED and ENU.
    pub altitude: f64,
    /// Northward velocity in m/s, resolved in the local-level frame.
    pub northward_velocity: f64,
    /// Eastward velocity in m/s, resolved in the local-level frame.
    pub eastward_velocity: f64,
    /// Vertical velocity in m/s: positive *down* in NED (the default), positive *up* in ENU.
    pub vertical_velocity: f64,
    /// Roll, the first angle of the XYZ Euler sequence that gives the body-to-navigation
    /// rotation, in degrees when `in_degrees` is set and in radians otherwise.
    ///
    /// [`InitialState::new`] wraps it to the range -180 to 180 degrees, or $-\pi$ to $\pi$
    /// radians -- the branch [`Rotation3::euler_angles`] returns -- which leaves the rotation
    /// it represents unchanged.
    ///
    /// [`Rotation3::euler_angles`]: nalgebra::Rotation3::euler_angles
    pub roll: f64,
    /// Pitch, the second angle of the XYZ Euler sequence, in degrees when `in_degrees` is set
    /// and in radians otherwise; wrapped by [`InitialState::new`] like `roll`.
    pub pitch: f64,
    /// Yaw, the third angle of the XYZ Euler sequence, in degrees when `in_degrees` is set and
    /// in radians otherwise; wrapped by [`InitialState::new`] like `roll`.
    pub yaw: f64,
    /// Unit tag for every angular field: `true` if latitude, longitude, roll, pitch and yaw are
    /// stored in degrees, `false` if they are already radians.
    ///
    /// The filter constructors convert those five fields to radians exactly when this is set,
    /// so the flag must travel with the values rather than being reset independently.
    pub in_degrees: bool,
    /// Local-level frame convention: `true` for ENU, `false` for NED (the crate default).
    ///
    /// Copied straight into the filter it seeds, where it selects the mechanization's vertical
    /// sign conventions; see the crate-level "Frame convention" section.
    pub is_enu: bool,
}
impl InitialState {
    /// Create a new `InitialState`, wrapping angles into range without changing their units.
    ///
    /// The constructor accepts latitude/longitude and Euler angles either in
    /// degrees (when `in_degrees==true`) or already in radians. It wraps each
    /// value into range *in the units it was given* and stores it that way,
    /// tagged by `in_degrees`; it does not normalize the stored position to
    /// radians. Converting to radians is the filter constructors' job, which
    /// they do exactly when `in_degrees` is set. The optional `is_enu`
    /// parameter selects the local-frame convention (defaults to NED when
    /// omitted).
    ///
    /// The Euler angles follow the same contract as longitude: `roll`, `pitch` and
    /// `yaw` are wrapped -- to -180..180 on the degrees path, $-\pi$..$\pi$ on the radian
    /// path -- and stored in the unit `in_degrees` names, never converted here.
    /// Latitude is stored exactly as supplied. So every angular field leaves this
    /// constructor in the unit the flag advertises, which is the unit the filter
    /// constructors read it back in.
    ///
    /// # Arguments
    ///
    /// * `latitude` - Latitude (degrees if `in_degrees==true`, otherwise radians)
    /// * `longitude` - Longitude (degrees if `in_degrees==true`, otherwise radians)
    /// * `altitude` - Altitude in meters
    /// * `northward_velocity` - Northward velocity in m/s
    /// * `eastward_velocity` - Eastward velocity in m/s
    /// * `vertical_velocity` - Vertical velocity in m/s
    /// * `roll` - Roll angle (degrees if `in_degrees==true`)
    /// * `pitch` - Pitch angle (degrees if `in_degrees==true`)
    /// * `yaw` - Yaw angle (degrees if `in_degrees==true`)
    /// * `in_degrees` - If true the latitude/longitude/angles are provided in degrees
    /// * `is_enu` - Optional: use ENU frame if true, NED if false (defaults to NED)
    ///
    /// # Returns
    ///
    /// An `InitialState` whose longitude and Euler angles are wrapped into range, whose
    /// latitude is stored as given, and whose angular fields are all held in the units
    /// they were supplied in, alongside the `in_degrees` flag, so that the filter
    /// constructors -- which convert only when the flag is set -- read them back
    /// consistently.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        latitude: f64,
        longitude: f64,
        altitude: f64,
        northward_velocity: f64,
        eastward_velocity: f64,
        vertical_velocity: f64,
        mut roll: f64,
        mut pitch: f64,
        mut yaw: f64,
        in_degrees: bool,
        is_enu: Option<bool>,
    ) -> Self {
        // The filter constructors read `in_degrees` back off the stored struct and convert
        // only when it is `true`, so the radian path must store *radians*. It previously
        // converted latitude to degrees while leaving `in_degrees == false`, so a caller
        // passing 40 deg N (0.698 rad) had it stored as 40.0 and then consumed as 40
        // radians -- and longitude, left alone, disagreed with it. Each branch now wraps
        // in its own units and stores what the constructors expect.
        let (latitude, longitude) = if in_degrees {
            (latitude, wrap_to_180(longitude))
        } else {
            (latitude, wrap_to_pi(longitude))
        };
        let is_enu = is_enu.unwrap_or(false);
        if in_degrees {
            // Wrapped in degrees and *stored* in degrees, matching latitude and longitude
            // above. Converting to radians here while leaving `in_degrees == true` made
            // every filter constructor convert a second time -- `if initial_state.in_degrees
            // { roll.to_radians() }` -- so a 45 degree seed was stored as 0.785 and reached
            // the filter as 0.0137 rad. Only a zero attitude survived the round trip, which
            // is why it went unnoticed: the workspace's other seeds are struct literals, and
            // the one caller that used this constructor with a non-zero heading was #262's
            // `InsEngine`, whose lever-arm compensation rotates by the estimate and so was
            // quietly resolving the antenna offset along the wrong axis.
            //
            // Wrapped symmetrically about zero rather than onto 0..360, so that the seed a
            // filter reports back before its first `predict` is on the same branch as the
            // one every `predict` afterwards writes; a -5 degree roll seed stays -5 instead
            // of becoming 355 (#314).
            roll = wrap_to_180(roll);
            pitch = wrap_to_180(pitch);
            yaw = wrap_to_180(yaw);
        } else {
            roll = wrap_to_pi(roll);
            pitch = wrap_to_pi(pitch);
            yaw = wrap_to_pi(yaw);
        }
        Self {
            latitude,
            longitude,
            altitude,
            northward_velocity,
            eastward_velocity,
            vertical_velocity,
            roll,
            pitch,
            yaw,
            in_degrees,
            is_enu,
        }
    }
}
/// Widen a measurement Jacobian to the filter's state dimension.
///
/// Most `MeasurementModel` implementations return a 9-column Jacobian: they observe
/// navigation states and say nothing about IMU biases, so the bias columns are zero
/// and the model has no reason to know how many of them the filter carries. A few --
/// [`ZaruMeasurement`](crate::measurements::ZaruMeasurement) is the motivating case --
/// observe a bias directly and must return the full width. Padding on the right is
/// correct for the first kind and a no-op for the second, so one helper covers both
/// and the filters stop caring which they were handed.
///
/// # Errors
/// [`StrapdownError::DimensionMismatch`] if the Jacobian is *wider* than the state.
/// That is a measurement built for a bigger filter than the one running it -- ZARU
/// against a 9-state EKF, say -- and padding cannot rescue it.
pub(crate) fn expand_measurement_jacobian(
    jacobian: DMatrix<f64>,
    state_size: usize,
) -> Result<DMatrix<f64>, StrapdownError> {
    let (rows, cols) = (jacobian.nrows(), jacobian.ncols());
    if cols == state_size {
        return Ok(jacobian);
    }
    if cols > state_size {
        return Err(StrapdownError::DimensionMismatch {
            what: "measurement Jacobian columns exceed filter state size",
            expected: state_size,
            got: cols,
        });
    }
    let mut expanded = DMatrix::<f64>::zeros(rows, state_size);
    expanded.view_mut((0, 0), (rows, cols)).copy_from(&jacobian);
    Ok(expanded)
}

/// Wrap the Euler-angle block of a filter state onto -pi..pi, in place.
///
/// The EKF and UKF carry roll/pitch/yaw as plain state elements, so a correction can push
/// them off the principal branch. Which branch they are put back on is a presentation
/// choice -- every consumer rebuilds the rotation with `Rotation3::from_euler_angles`,
/// which is 2*pi-periodic -- so the only thing it decides is what a reader of
/// `get_estimate` sees.
///
/// -pi..pi is the branch `Rotation3::euler_angles` returns, which is what both filters'
/// `predict` already writes straight back into the state, what the RBPF and the
/// dead-reckoning CSV writer already report, and the convention `wrap_to_180` gives
/// longitude. Wrapping onto 0..2*pi instead put the branch cut at zero roll and zero
/// pitch -- the attitude of a level vehicle -- so a hair of negative roll was reported as
/// 359.99 degrees and `estimate - truth` came out a full turn wrong (#314).
///
/// Pitch is deliberately *not* clamped to -pi/2..pi/2. That is the range the Euler
/// decomposition produces, but these two filters can hold a larger pitch between a seed and
/// the first `predict`, and clamping would change the rotation rather than rename it; the
/// next `predict` re-derives the triple through `euler_angles` and canonicalises it.
fn wrap_attitude_onto_principal_branch(state: &mut DVector<f64>) {
    for index in ATTITUDE_STATE_INDICES {
        state[index] = wrap_to_pi(state[index]);
    }
}

/// Indices of the three Euler angles within a filter state vector.
///
/// These are the channels that live on the circle rather than the line: the ones
/// [`wrap_attitude_onto_principal_branch`] wraps and
/// [`unwrap_attitude_onto_reference_branch`] re-expresses. Every other channel is an
/// ordinary linear quantity. Named for the same reason
/// [`rbpf::ATTITUDE_STATE_INDICES`](crate::rbpf) is: a bare `6..9` beside a position
/// block and a velocity block says nothing about why those three are special.
const ATTITUDE_STATE_INDICES: std::ops::Range<usize> = 6..9;

/// The roll row of a filter state, by name.
///
/// Derived from [`ATTITUDE_STATE_INDICES`] rather than written as `6`, so the manifold
/// helpers below cannot drift away from the range the rest of the module uses.
const ATTITUDE_ROLL_INDEX: usize = ATTITUDE_STATE_INDICES.start;
/// The pitch row of a filter state, by name. See [`ATTITUDE_ROLL_INDEX`].
const ATTITUDE_PITCH_INDEX: usize = ATTITUDE_STATE_INDICES.start + 1;
/// The yaw row of a filter state, by name. See [`ATTITUDE_ROLL_INDEX`].
const ATTITUDE_YAW_INDEX: usize = ATTITUDE_STATE_INDICES.start + 2;

/// Re-express a state's Euler angles on the same branch as `reference`, in place.
///
/// [`Rotation3::euler_angles`] derives roll and yaw with `atan2`, so it hands every
/// rotation back on `[-pi, pi]` with a branch cut at `+/-pi`. That is the right thing for
/// *reporting* an attitude and the wrong thing for combining several: two attitudes 2 deg
/// apart either side of the cut come back as `+179` and `-179` deg, and any linear
/// combination of those two numbers describes neither.
///
/// This puts `state` on whatever branch `reference` is on, by adding whole turns, so the
/// numbers can be combined arithmetically again. It is exact as long as the two attitudes
/// are less than a half turn apart in each angle, which for a sigma-point spread is the
/// same condition the linearisation already needs.
///
/// It is deliberately *not* a circular mean. The UKF's sigma-point weights are not convex
/// -- with `alpha = 1e-3` and `n = 15`, `w_0` is about -1e6 against `w_i` of about +3e4 --
/// and `atan2(sum w sin, sum w cos)` is not meaningful for weights that can be large and
/// negative. Unwrapping onto a reference and then taking the ordinary weighted sum keeps
/// the unscented transform's own arithmetic intact; the RBPF, whose weights *are* convex,
/// uses [`rbpf::circular_mean`](crate::rbpf) instead.
///
/// # Why this adds whole turns rather than rebuilding the angle
///
/// The obvious spelling is `state = reference + wrap_to_pi(state - reference)`, and it is
/// wrong in a way that only shows up over a long run: when no wrap is needed that round trip
/// through a subtraction and an addition still costs an ulp or two, and the UKF's mean is not
/// a place where ulps stay small. `mu_bar` is `w_0 * x_0 + sum w_i * x_i` with `w_0` about
/// -1e6, so a 1-ulp nudge to a sigma point moves the mean by ~1e-10 -- which then seeds the
/// next sigma set, and compounds. Rewriting every angle that way moved this suite's *northbound*
/// UKF yaw, where nothing straddles the cut and the fix should do nothing at all, by 0.09 rad
/// over 1500 steps.
///
/// Adding the whole turns instead makes the no-wrap case exactly identity --
/// [`wrap_to_pi`] returns an in-range input untouched, so `turns` is a literal `0.0` and the
/// element is never written -- and leaves every baseline that does not straddle the cut
/// bit-for-bit where it was. Only the case this exists to fix changes.
fn unwrap_attitude_onto_reference_branch(state: &mut DVector<f64>, reference: &[f64; 3]) {
    for (index, reference_angle) in ATTITUDE_STATE_INDICES.zip(reference) {
        let offset = state[index] - reference_angle;
        // Whole turns, or exactly zero when the angle is already on the reference's branch.
        let turns = wrap_to_pi(offset) - offset;
        if turns != 0.0 {
            state[index] += turns;
        }
    }
}

/// Read a state vector's attitude triple as the rotation it denotes.
///
/// Rows 6..9 of every filter state in this crate are an intrinsic XYZ Euler triple. This is
/// where the UKF turns them back into a rotation, and it is the reason the manifold
/// arithmetic below is insensitive to branch: two triples a whole turn apart in any angle
/// build the *identical* matrix, so nothing downstream of this call can see which branch a
/// `euler_angles()` call happened to canonicalise onto.
fn attitude_of<S>(state: &S) -> Rotation3<f64>
where
    S: std::ops::Index<usize, Output = f64> + ?Sized,
{
    Rotation3::from_euler_angles(
        state[ATTITUDE_ROLL_INDEX],
        state[ATTITUDE_PITCH_INDEX],
        state[ATTITUDE_YAW_INDEX],
    )
}

/// Write a rotation back into a state vector's attitude triple.
fn set_attitude_of(state: &mut DVector<f64>, attitude: &Rotation3<f64>) {
    let (roll, pitch, yaw) = attitude.euler_angles();
    state[ATTITUDE_ROLL_INDEX] = roll;
    state[ATTITUDE_PITCH_INDEX] = pitch;
    state[ATTITUDE_YAW_INDEX] = yaw;
}

/// The rotation vector taking `reference` to `point`: $\log(R_{\text{ref}}^\top R)$.
///
/// This is the attitude difference the unscented transform actually wants -- an element of
/// the tangent space at `reference`, which is a genuine vector and may therefore be scaled,
/// summed and squared like every other state. The difference of two Euler triples is not.
///
/// # Why the quaternion route, and not `Rotation3::scaled_axis`
///
/// They compute the same quantity and only one of them is usable at this filter's weights.
/// `Rotation3::scaled_axis` recovers the angle from the matrix trace -- an `acos` evaluated
/// within an ulp of 1 for a small rotation, where its derivative is unbounded. The measured
/// sigma-point attitude spread here is about `1.2e-7` rad (`alpha = 1e-3` places the points
/// at `0.0039` sigma), and at that magnitude it returns roughly 1.5% relative error. The
/// scaled transform's weights are large and cancelling -- `w_i` is about `+3.3e4` against a
/// `w_0` of about `-1.0e6` -- so that error does not stay small: it becomes about `6e-5` rad
/// of fabricated rotation per step, which is 0.17 deg/s at 50 Hz, the same order as the gyro
/// bias the filter is trying to estimate.
///
/// `UnitQuaternion::from_rotation_matrix` goes through the quaternion's vector part instead,
/// which is linear in the angle near identity. On the identical sum that lands at `3.7e-12`
/// rad -- sixteen million times smaller, and the zero it should be.
fn attitude_tangent(reference: &Rotation3<f64>, point: &Rotation3<f64>) -> Vector3<f64> {
    UnitQuaternion::from_rotation_matrix(&(reference.transpose() * point)).scaled_axis()
}

/// Overwrite a state difference's attitude rows with the tangent-space residual.
///
/// The other rows of `difference` are ordinary vector subtraction and stay as they are.
fn set_attitude_residual(difference: &mut DVector<f64>, residual: &Vector3<f64>) {
    difference[ATTITUDE_ROLL_INDEX] = residual[0];
    difference[ATTITUDE_PITCH_INDEX] = residual[1];
    difference[ATTITUDE_YAW_INDEX] = residual[2];
}

/// Unscented Kalman Filter (UKF) implementation for strapdown navigation.
///
/// The UKF approximates the posterior distribution using a deterministic set
/// of sigma points which are propagated through the nonlinear strapdown
/// mechanization. This implementation stores the mean state and covariance
/// in `nalgebra` `DVector`/`DMatrix` types and supports optional IMU bias
/// states and additional user states appended to the navigation state.
///
/// # State layout
/// The base navigation state ordering matches the rest of the crate:
/// `[lat, lon, alt, v_n, v_e, v_d, roll, pitch, yaw, ...]` with any IMU
/// biases or extra states appended after the ninth element.
///
/// # References
/// - Julier, S. & Uhlmann, J. "Unscented Filtering and Nonlinear Estimation".
#[derive(Clone)]
pub struct UnscentedKalmanFilter {
    mean_state: DVector<f64>,
    covariance: DMatrix<f64>,
    process_noise: DMatrix<f64>,
    lambda: f64,
    state_size: usize,
    weights_mean: DVector<f64>,
    weights_cov: DVector<f64>,
    is_enu: bool,
    /// Innovation gate applied by `update` together with the recovery policy that keeps
    /// a rejection from being permanent; an empty gate accepts every measurement.
    gate_policy: GatePolicy,
}
impl Debug for UnscentedKalmanFilter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UKF")
            .field("mean_state", &self.mean_state)
            .field("covariance", &self.covariance)
            .field("process_noise", &self.process_noise)
            .field("lambda", &self.lambda)
            .field("state_size", &self.state_size)
            .finish_non_exhaustive()
    }
}
impl Display for UnscentedKalmanFilter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("UnscentedKalmanFilter")
            .field("mean_state", &self.mean_state)
            .field("covariance", &self.covariance)
            .field("process_noise", &self.process_noise)
            .field("lambda", &self.lambda)
            .field("state_size", &self.state_size)
            .finish()
    }
}
impl UnscentedKalmanFilter {
    #[allow(clippy::too_many_arguments)]
    /// Create a new UKF instance.
    ///
    /// # Arguments
    ///
    /// * `initial_state` - Navigation initial state (`InitialState`).
    /// * `imu_biases` - Initial IMU bias estimates appended to the state.
    /// * `other_states` - Optional additional state vector to append.
    /// * `covariance_diagonal` - Initial diagonal elements for the covariance matrix.
    /// * `process_noise` - Process noise covariance matrix (state-space Q).
    /// * `alpha`, `beta`, `kappa` - UKF tuning parameters (see Julier & Uhlmann).
    ///
    /// # Returns
    ///
    /// A configured `UnscentedKalmanFilter` with computed sigma weights.
    ///
    /// # Example
    ///
    /// ```rust
    /// use strapdown::kalman::{UnscentedKalmanFilter, InitialState};
    /// use nalgebra::DMatrix;
    /// let init = InitialState::default();
    /// // Position entries are rad^2, not m^2: convert metres once (#308). An identity
    /// // process noise would be 1 rad^2 per step, i.e. ~6367 km of horizontal drift.
    /// let horizontal_std_rad = 10.0 * strapdown::earth::METERS_TO_RADIANS;
    /// let mut covariance = vec![horizontal_std_rad.powi(2), horizontal_std_rad.powi(2), 100.0];
    /// covariance.extend([0.25; 3]); // velocity, (m/s)^2
    /// covariance.extend([1e-4; 3]); // attitude, rad^2
    /// let process_noise = DMatrix::from_diagonal(
    ///     &nalgebra::DVector::from_vec(strapdown::sim::DEFAULT_PROCESS_NOISE_DENSITY[0..9].to_vec()),
    /// );
    /// let ukf = UnscentedKalmanFilter::new(&init, &[0.0;6], None, covariance, process_noise, 1e-3, 2.0, 0.0);
    /// ```
    pub fn new(
        initial_state: &InitialState,
        imu_biases: &[f64],
        other_states: Option<&[f64]>,
        covariance_diagonal: Vec<f64>,
        process_noise: DMatrix<f64>,
        alpha: f64,
        beta: f64,
        kappa: f64,
    ) -> Self {
        let mut mean = if initial_state.in_degrees {
            vec![
                initial_state.latitude.to_radians(),
                initial_state.longitude.to_radians(),
                initial_state.altitude,
                initial_state.northward_velocity,
                initial_state.eastward_velocity,
                initial_state.vertical_velocity,
                initial_state.roll.to_radians(),
                initial_state.pitch.to_radians(),
                initial_state.yaw.to_radians(),
            ]
        } else {
            vec![
                initial_state.latitude,
                initial_state.longitude,
                initial_state.altitude,
                initial_state.northward_velocity,
                initial_state.eastward_velocity,
                initial_state.vertical_velocity,
                initial_state.roll,
                initial_state.pitch,
                initial_state.yaw,
            ]
        };
        mean.extend(imu_biases);
        if let Some(other_states) = other_states {
            mean.extend(other_states.iter().copied());
        }
        let state_size = mean.len();
        let mean_state = DVector::from_vec(mean);
        let covariance = DMatrix::<f64>::from_diagonal(&DVector::from_vec(covariance_diagonal));
        let lambda = alpha * alpha * (state_size as f64 + kappa) - state_size as f64;
        let mut weights_mean = DVector::zeros(2 * state_size + 1);
        let mut weights_cov = DVector::zeros(2 * state_size + 1);
        weights_mean[0] = lambda / (state_size as f64 + lambda);
        weights_cov[0] = lambda / (state_size as f64 + lambda) + (1.0 - alpha * alpha + beta);
        for i in 1..=(2 * state_size) {
            let w = 1.0 / (2.0 * (state_size as f64 + lambda));
            weights_mean[i] = w;
            weights_cov[i] = w;
        }
        Self {
            mean_state,
            covariance,
            process_noise,
            lambda,
            state_size,
            weights_mean,
            weights_cov,
            is_enu: initial_state.is_enu,
            gate_policy: GatePolicy::default(),
        }
    }
    /// # Errors
    /// [`StrapdownError::NotSquare`] if the covariance is not square, propagated from
    /// [`matrix_square_root`].
    pub fn get_sigma_points(&self) -> Result<DMatrix<f64>, StrapdownError> {
        // Generate the augmented sigma points matrix for the current mean and covariance.
        //
        // The returned matrix has dimensions `(state_size) x (2*state_size + 1)` where each
        // column is a sigma point. Sigma point generation follows the scaled unscented
        // transform: sqrt((n+lambda) P) columns added/subtracted from the mean.
        let p = (self.state_size as f64 + self.lambda) * self.covariance.clone();
        let sqrt_p = matrix_square_root(&p)?;
        let mu = self.mean_state.clone();
        let mut pts = DMatrix::<f64>::zeros(self.state_size, 2 * self.state_size + 1);
        // Sigma point 0 is canonicalised through the same `euler_angles()` round trip as
        // every other column, and that is load-bearing rather than tidiness. `Rotation3`
        // canonicalises pitch onto `[-pi/2, pi/2]`, so a mean carrying an equivalent
        // non-principal triple -- which `InitialState` accepts, and which the constructor
        // stores verbatim -- would leave column 0 in one representation and the other `2n`
        // in the other. The two denote the *same rotation*, so the manifold arithmetic is
        // unaffected, but a measurement model reads the yaw row directly, and
        // `unwrap_attitude_onto_reference_branch` adds whole turns and cannot repair a
        // half-turn-plus-reflection.
        //
        // Measured with an initial pitch of -3.0 rad and a covariance of 1e-10 (a true
        // spread of 4e-8 rad): the sigma set came out with a **3.1416 rad** yaw spread and a
        // 2.8584 rad pitch spread -- a fabricated half turn, handed to the heading update as
        // if it were uncertainty. Reachable only on the first `get_sigma_points` after
        // construction, because `predict` and `update` both write the mean back through
        // `set_attitude_of` and so leave it canonical; that is one call with a wildly wrong
        // covariance, at the one moment the filter has no history to absorb it.
        let mut canonical_mean = mu.clone();
        set_attitude_of(&mut canonical_mean, &attitude_of(&mu));
        pts.column_mut(0).copy_from(&canonical_mean);
        let mu = canonical_mean;
        // Attitude is perturbed on the manifold, every other state in the chart. The linear
        // `mu +/- column` is exactly right for a state whose difference is a vector and
        // wrong for the three that parameterise a rotation: adding a rotation vector to an
        // Euler triple is not the same rotation as composing it, and the discrepancy is
        // second order in the perturbation -- which the scaled transform's own weights then
        // amplify by the factor the spread shrank by. See `attitude_tangent` (#371).
        let mean_attitude = attitude_of(&mu);
        let reference_branch = [
            mu[ATTITUDE_ROLL_INDEX],
            mu[ATTITUDE_PITCH_INDEX],
            mu[ATTITUDE_YAW_INDEX],
        ];
        for i in 0..sqrt_p.ncols() {
            let column = sqrt_p.column(i);
            let tangent = Vector3::new(
                column[ATTITUDE_ROLL_INDEX],
                column[ATTITUDE_PITCH_INDEX],
                column[ATTITUDE_YAW_INDEX],
            );
            let mut plus = &mu + column;
            let mut minus = &mu - column;
            set_attitude_of(
                &mut plus,
                &(mean_attitude * Rotation3::from_scaled_axis(tangent)),
            );
            set_attitude_of(
                &mut minus,
                &(mean_attitude * Rotation3::from_scaled_axis(-tangent)),
            );
            // `euler_angles` canonicalises each triple onto `[-pi, pi]` independently, which
            // the linear form never had to care about because it never left the chart. The
            // attitude arithmetic downstream is branch-free (see `attitude_of`), but a
            // measurement model reading the yaw row of a sigma point directly is not, so put
            // the points back on the mean's branch the way `predict` already does. Exactly
            // identity when nothing straddles the cut.
            unwrap_attitude_onto_reference_branch(&mut plus, &reference_branch);
            unwrap_attitude_onto_reference_branch(&mut minus, &reference_branch);
            pts.set_column(i + 1, &plus);
            pts.set_column(i + 1 + self.state_size, &minus);
        }
        Ok(pts)
    }
    fn robust_kalman_gain(
        cross_covariance: &DMatrix<f64>,
        s: &DMatrix<f64>,
    ) -> Result<DMatrix<f64>, StrapdownError> {
        // Compute a numerically robust Kalman gain K = P_xz * S^{-1} using a
        // symmetric positive-definite solver. This helps avoid instability when
        // the innovation covariance `s` is poorly conditioned.
        let kt = robust_spd_solve(&symmetrize(s), &cross_covariance.transpose())?;
        Ok(kt.transpose())
    }
}
/// Relative tolerance for reconciling an [`ImuSample`]'s own `dt` with the `dt` argument
/// [`NavigationFilter::predict`](crate::NavigationFilter::predict) is called with.
///
/// Not exact equality: a caller that derives both from the same pair of timestamps can
/// legitimately land a few ulps apart. Anything looser would let a genuinely wrong rate
/// through, which is the defect this check exists to prevent.
const TIMESTEP_AGREEMENT_RELATIVE_TOLERANCE: f64 = 1e-9;

/// Resolve an [`InputModel`](crate::InputModel) trait object into the [`ImuSample`] the
/// increment-domain filters mechanize with.
///
/// Accepts both forms of inertial input. An [`ImuSample`] is taken as given -- it is what a
/// real IMU emits and what [`mechanize`] consumes. An [`IMUData`] is rectangular-integrated
/// over `dt` by [`ImuSample::from_rates`], which is exactly the conversion the deprecated
/// [`forward`](crate::forward) performed, so callers still holding rates are unaffected.
///
/// # Errors
/// * [`StrapdownError::InconsistentTimestep`] if `input` is an [`ImuSample`] whose `dt`
///   disagrees with `dt`. Preferring one silently would make the integration rate quietly
///   wrong rather than loudly absent.
/// * [`StrapdownError::UnsupportedInput`] if `input` is neither inertial form -- `VelocityData`
///   also implements [`InputModel`](crate::InputModel), so the type system permits it here.
pub(crate) fn imu_sample_from_input(
    input: &dyn crate::InputModel,
    filter: &'static str,
    dt: f64,
) -> Result<ImuSample, StrapdownError> {
    if let Some(sample) = input.as_any().downcast_ref::<ImuSample>() {
        let scale = sample.dt.abs().max(dt.abs()).max(f64::MIN_POSITIVE);
        if (sample.dt - dt).abs() > TIMESTEP_AGREEMENT_RELATIVE_TOLERANCE * scale {
            return Err(StrapdownError::InconsistentTimestep {
                sample_dt: sample.dt,
                arg_dt: dt,
            });
        }
        return Ok(*sample);
    }
    input
        .as_any()
        .downcast_ref::<IMUData>()
        .map(|imu| ImuSample::from_rates(imu, dt))
        .ok_or(StrapdownError::UnsupportedInput {
            filter,
            expected: "ImuSample or IMUData",
        })
}

impl NavigationFilter for UnscentedKalmanFilter {
    /// Predict step for the UKF: propagate sigma points through the mechanization.
    ///
    /// # Arguments
    ///
    /// * `control_input` - an [`ImuSample`] (integrated $\Delta v$ / $\Delta\theta$) or,
    ///   for callers still holding instantaneous rates, an [`IMUData`].
    /// * `dt` - Time step in seconds. When `control_input` is an [`ImuSample`] this must
    ///   agree with the sample's own `dt`; see the Errors section.
    ///
    /// # Errors
    /// * [`StrapdownError::UnsupportedInput`] if `control_input` is neither inertial form.
    /// * [`StrapdownError::InconsistentTimestep`] if an [`ImuSample`]'s `dt` disagrees with
    ///   the `dt` argument.
    /// * [`StrapdownError::OutOfRange`] or [`StrapdownError::NonFinite`] propagated from
    ///   [`mechanize`], or [`StrapdownError::NotSquare`] from the sigma-point square root.
    ///
    /// # Bias compensation
    ///
    /// Each sigma point carries its own bias hypothesis in states 9..15, so the correction
    /// is applied per sigma point rather than once to the shared input. Biases are rates and
    /// the sensed quantities are their integrals, so the correction is each bias integrated
    /// over the interval -- the same increment-domain form the ESKF uses:
    /// $$
    /// \begin{aligned}
    /// \Delta v^b &= \Delta v^b_{\text{measured}} - b_a \Delta t \\\\
    /// \Delta\theta^b &= \Delta\theta^b_{\text{measured}} - b_g \Delta t
    /// \end{aligned}
    /// $$
    ///
    /// # Attitude is averaged on one branch
    ///
    /// The sigma points enter [`mechanize`] on a common branch -- they are the mean plus and
    /// minus the columns of a matrix square root -- and leave it canonicalised onto
    /// `[-pi, pi]` *per point*, because that is what `Rotation3::euler_angles` returns. At a
    /// southerly heading the set straddles the cut, so a linear mean of the numbers is not
    /// the mean attitude. They are put back on sigma point 0's branch by
    /// [`unwrap_attitude_onto_reference_branch`] before the weighted sum, which is both
    /// where the mean and the covariance become meaningful again (#336).
    fn predict(
        &mut self,
        control_input: &dyn crate::InputModel,
        dt: f64,
    ) -> Result<(), StrapdownError> {
        let sample = imu_sample_from_input(control_input, "UnscentedKalmanFilter", dt)?;

        let mut sigma_points = self.get_sigma_points()?;
        // Branch the propagated attitudes are re-expressed on before they are averaged; set
        // from sigma point 0, which is the propagated mean and therefore the most
        // representative member of the set. See `unwrap_attitude_onto_reference_branch`.
        let mut attitude_branch_reference = [0.0_f64; 3];
        for i in 0..sigma_points.ncols() {
            let mut sigma_point_vec = sigma_points.column(i).clone_owned();
            let mut state = StrapdownState {
                latitude: sigma_point_vec[0],
                longitude: sigma_point_vec[1],
                altitude: sigma_point_vec[2],
                velocity_north: sigma_point_vec[3],
                velocity_east: sigma_point_vec[4],
                velocity_vertical: sigma_point_vec[5],
                attitude: Rotation3::from_euler_angles(
                    sigma_point_vec[6],
                    sigma_point_vec[7],
                    sigma_point_vec[8],
                ),
                is_enu: self.is_enu,
            };
            let (accel_biases, gyro_biases) = if self.state_size >= 15 {
                (
                    Vector3::new(sigma_point_vec[9], sigma_point_vec[10], sigma_point_vec[11]),
                    Vector3::new(
                        sigma_point_vec[12],
                        sigma_point_vec[13],
                        sigma_point_vec[14],
                    ),
                )
            } else {
                (Vector3::zeros(), Vector3::zeros())
            };
            // Correct in the increment domain rather than dividing back out to rates: a
            // sample that arrived as genuine delta-v / delta-theta then never makes a lossy
            // round trip through `dt`.
            let corrected_sample = ImuSample {
                delta_v: sample.delta_v - accel_biases * sample.dt,
                delta_theta: sample.delta_theta - gyro_biases * sample.dt,
                dt: sample.dt,
            };
            mechanize(&mut state, &corrected_sample)?;
            sigma_point_vec[0] = state.latitude;
            sigma_point_vec[1] = state.longitude;
            sigma_point_vec[2] = state.altitude;
            sigma_point_vec[3] = state.velocity_north;
            sigma_point_vec[4] = state.velocity_east;
            sigma_point_vec[5] = state.velocity_vertical;
            let (roll, pitch, yaw) = state.attitude.euler_angles();
            sigma_point_vec[6] = roll;
            sigma_point_vec[7] = pitch;
            sigma_point_vec[8] = yaw;
            // The sigma points went into `mechanize` on a common branch -- they are the mean
            // plus and minus the columns of a matrix square root -- but come back out of
            // `euler_angles` each canonicalised onto `[-pi, pi]` independently, which is
            // what breaks that. Put them back on one branch before anything averages them.
            if i == 0 {
                attitude_branch_reference =
                    [sigma_point_vec[6], sigma_point_vec[7], sigma_point_vec[8]];
            } else {
                unwrap_attitude_onto_reference_branch(
                    &mut sigma_point_vec,
                    &attitude_branch_reference,
                );
            }
            sigma_points.set_column(i, &sigma_point_vec);
        }
        let mut mu_bar = DVector::<f64>::zeros(self.state_size);
        for (i, sigma_point) in sigma_points.column_iter().enumerate() {
            mu_bar += self.weights_mean[i] * sigma_point;
        }
        // The attitude rows of that sum are not the mean attitude, and the error is not
        // small. A weighted arithmetic mean of Euler triples is a linear operation on a
        // nonlinear chart: 31 rotations all lying inside a 0.24 deg cone average, that way,
        // to a point 14.3 deg *outside* it -- sixty times the spread of the inputs. Across a
        // 10^4 sweep of the sigma-point spread the discrepancy moves 0.4%, where a genuinely
        // second-order error would have fallen by 10^8; the scaled transform's weights are
        // what hold it up (#371).
        //
        // So average on the group instead: anchor at sigma point 0 -- the propagated mean,
        // and the member the rest are closest to -- take each residual into the tangent
        // space there, sum them with the transform's own weights, and map the result back.
        // `sum_i w_i == 1` holds even though the weights are not convex, so this is well
        // defined, and one anchored pass is the whole of it. A Karcher *iteration* is not
        // available here: re-anchoring on the running mean makes sigma point 0's residual
        // non-zero, and `w_0` (about -1e6) multiplies it, which NaNs the filter within a few
        // samples.
        let reference_attitude = attitude_of(&sigma_points.column(0));
        let mut tangent_mean = Vector3::<f64>::zeros();
        for (i, sigma_point) in sigma_points.column_iter().enumerate() {
            tangent_mean += self.weights_mean[i]
                * attitude_tangent(&reference_attitude, &attitude_of(&sigma_point));
        }
        let mean_attitude = reference_attitude * Rotation3::from_scaled_axis(tangent_mean);
        set_attitude_of(&mut mu_bar, &mean_attitude);
        let mut p_bar = DMatrix::<f64>::zeros(self.state_size, self.state_size);
        for (i, sigma_point) in sigma_points.column_iter().enumerate() {
            // No angular wrap here: the attitude rows are replaced below by a tangent-space
            // residual, which has no branch cut under a half turn at all, and the remaining
            // rows are ordinary vector differences.
            let mut diff = sigma_point - &mu_bar;
            set_attitude_residual(
                &mut diff,
                &attitude_tangent(&mean_attitude, &attitude_of(&sigma_point)),
            );
            p_bar += self.weights_cov[i] * &diff * &diff.transpose();
        }
        p_bar += process_noise_for_step(&self.process_noise, sample.dt)?;
        // Report attitude on the same branch `update` writes (#314). The reference branch is
        // whichever one sigma point 0 landed on, so without this a state that sat near the
        // cut would drift a turn away from the principal branch over successive steps.
        // Applied after `p_bar` for the reason the comment above gives.
        wrap_attitude_onto_principal_branch(&mut mu_bar);
        self.mean_state = mu_bar;
        self.covariance = symmetrize(&p_bar);
        Ok(())
    }
    /// Update step for the UKF: map sigma points into measurement space and
    /// compute cross-covariances to form the Kalman gain.
    ///
    /// # Arguments
    ///
    /// * `measurement` - A measurement model implementing `MeasurementModel`.
    fn update(
        &mut self,
        measurement: &dyn MeasurementModel,
    ) -> Result<UpdateOutcome, StrapdownError> {
        //let measurement_sigma_points = measurement.get_sigma_points(&self.get_sigma_points());
        let mut measurement_sigma_points =
            DMatrix::<f64>::zeros(measurement.get_dimension(), 2 * self.state_size + 1);
        let mut z_hat = DVector::<f64>::zeros(measurement.get_dimension());
        let sigma_points = self.get_sigma_points()?;
        for (i, sigma_point) in sigma_points.column_iter().enumerate() {
            //let sigma_point_vec = sigma_point.clone_owned();
            let sigma_point = measurement.get_expected_measurement(&sigma_point.clone_owned());
            measurement_sigma_points.set_column(i, &sigma_point);
            z_hat += self.weights_mean[i] * sigma_point;
        }
        let mut s = DMatrix::<f64>::zeros(measurement.get_dimension(), measurement.get_dimension());
        for (i, sigma_point) in measurement_sigma_points.column_iter().enumerate() {
            let diff = sigma_point - &z_hat;
            s += self.weights_cov[i] * &diff * &diff.transpose();
        }
        s += measurement.get_noise();

        let mut innovation = measurement.get_measurement(&self.mean_state)? - &z_hat;
        // Keep angular innovations on the circle (see `wrap_residual`, #286).
        // (The sigma-point spread above is left linearised: with a sane yaw
        // uncertainty the points do not straddle the cut.)
        measurement.wrap_residual(&mut innovation);

        // Gate before the gain: a rejected measurement must leave the state untouched, and
        // there is no point paying for a gain that will not be applied.
        let decision = self.gate_policy.evaluate(&innovation, &s, "UKF")?;
        let outcome = decision.outcome;

        // The cross covariance is `P H^T` in sigma-point form, so it is what the gain needs
        // on the accepted path *and* what the observed-subspace inflation projects through
        // on the rejected one. It is therefore computed before the branch rather than after
        // it, which costs one accumulation on a rejection and buys a recovery that inflates
        // only the states this measurement observed (#340).
        //
        // The sigma points above are reused rather than regenerated: neither the mean nor
        // the covariance has changed since they were drawn, and `get_sigma_points` pays for
        // a matrix square root every time it is called.
        let mut cross_covariance =
            DMatrix::<f64>::zeros(self.state_size, measurement.get_dimension());
        let mean_attitude = attitude_of(&self.mean_state);
        for (i, measurement_sigma_point) in measurement_sigma_points.column_iter().enumerate() {
            let measurement_diff = measurement_sigma_point - &z_hat;
            // Same chart arithmetic as `predict`'s `p_bar`, and the same fix: the attitude
            // rows are the tangent-space residual at the mean, not a difference of Euler
            // triples (#371).
            let mut state_diff = sigma_points.column(i) - &self.mean_state;
            set_attitude_residual(
                &mut state_diff,
                &attitude_tangent(&mean_attitude, &attitude_of(&sigma_points.column(i))),
            );
            cross_covariance += self.weights_cov[i] * state_diff * measurement_diff.transpose();
        }

        if !outcome.accepted {
            // The covariance, unlike the state, does not come through a rejection
            // unchanged: inflating it is what stops the next fix being rejected harder
            // than this one (#340). `s` is `H P H^T + R`, so the measurement noise comes
            // back off to leave the filter's own contribution.
            let observed = &s - measurement.get_noise();
            decision.inflate_observed(&mut self.covariance, &cross_covariance, &observed);
            return Ok(outcome);
        }

        let k = Self::robust_kalman_gain(&cross_covariance, &s)?;
        let correction = &k * innovation;
        // The gain was formed against tangent-space attitude residuals, so its attitude rows
        // are a rotation vector and have to be *composed* onto the mean rather than added to
        // its Euler triple. Adding them is the same chart error as averaging them (#371).
        let corrected_attitude = mean_attitude
            * Rotation3::from_scaled_axis(Vector3::new(
                correction[ATTITUDE_ROLL_INDEX],
                correction[ATTITUDE_PITCH_INDEX],
                correction[ATTITUDE_YAW_INDEX],
            ));
        self.mean_state += correction;
        set_attitude_of(&mut self.mean_state, &corrected_attitude);
        // Report attitude on the same branch `predict` writes (#314).
        wrap_attitude_onto_principal_branch(&mut self.mean_state);
        self.covariance -= &k * &s * &k.transpose();
        // Symmetrise, then jitter each diagonal entry in proportion to its own scale.
        // Adding an absolute `1e-9` here -- which this did until #373 -- put ~(201 m)^2 of
        // horizontal variance into a state whose position entries are radians, and the
        // resulting Kalman gain of 0.999 made this filter copy its fixes rather than
        // filter them.
        regularize_covariance_in_place(&mut self.covariance, &self.process_noise);
        Ok(outcome)
    }

    fn set_innovation_gate(&mut self, gate: Option<InnovationGate>) -> bool {
        self.gate_policy.set_gate(gate);
        true
    }

    fn set_gate_recovery(&mut self, recovery: GateRecovery) -> bool {
        self.gate_policy.set_recovery(recovery);
        true
    }
    /// Return the current mean state estimate.
    ///
    /// Roll, pitch and yaw come back on `[-pi, pi]` -- the branch `Rotation3::euler_angles`
    /// returns -- after both [`Self::predict`] and [`Self::update`], each of which ends by
    /// calling `wrap_attitude_onto_principal_branch`.
    ///
    /// `predict` did not always do so, and the value it left was worse than off-branch. It
    /// took the *linear* weighted mean of the sigma points' Euler angles, which
    /// `euler_angles` had each canonicalised onto `[-pi, pi]` independently; sigma points
    /// straddling the cut at `+/-pi` -- which is what a southerly heading gives -- were
    /// therefore averaged across it. With the UKF's non-convex mean weights (`alpha = 1e-3`,
    /// `n = 15`, so `w_0` is about -1e6 against `w_i` of about +3e4) that average does not
    /// merely land between the points, it extrapolates away from them: seeded due south,
    /// this filter reached a reported pitch of 5.7 rad and a yaw of 8e5 rad within three
    /// samples, and wrapping afterwards would only have renamed it (#336).
    ///
    /// `predict` now puts every propagated sigma point back on one branch before averaging,
    /// so the mean is an attitude again and the wrap is the presentation step it looks like.
    fn get_estimate(&self) -> DVector<f64> {
        self.mean_state.clone()
    }

    /// Return the current state covariance (certainty) matrix.
    fn get_certainty(&self) -> DMatrix<f64> {
        self.covariance.clone()
    }
}

/// Extended Kalman Filter (EKF) implementation for strapdown INS
///
/// The Extended Kalman Filter provides a linearized Gaussian approximation to the
/// Bayesian filtering problem for nonlinear systems. Unlike the UKF which uses
/// sigma points to propagate uncertainty, the EKF linearizes the system dynamics
/// and measurement models using first-order Taylor series approximations (Jacobians).
///
/// # Mathematical Background
///
/// The EKF operates in two stages:
///
/// ## Predict Step
///
/// The predict step propagates the state estimate and covariance forward in time
/// using the nonlinear dynamics and linearized uncertainty propagation:
///
/// $$
/// \begin{aligned}
/// \bar{x}_{k+1} &= f(x_k, u_k) \\\\
/// \bar{P}_{k+1} &= F_k P_k F_k^T + G_k Q_k G_k^T
/// \end{aligned}
/// $$
///
/// where:
/// - $\bar{x}_{k+1}$ is the predicted state estimate
/// - $f(\cdot)$ is the nonlinear state transition function (strapdown mechanization)
/// - $F_k = \frac{\partial f}{\partial x}\big|_{x_k}$ is the state transition Jacobian
/// - $G_k$ is the process noise Jacobian
/// - $Q_k$ is the process noise covariance
///
/// ## Update Step
///
/// The update step incorporates a new measurement to correct the predicted estimate:
///
/// $$
/// \begin{aligned}
/// K_k &= \bar{P}_k H_k^T (H_k \bar{P}_k H_k^T + R_k)^{-1} \\\\
/// x_k &= \bar{x}_k + K_k (z_k - h(\bar{x}_k)) \\\\
/// P_k &= (I - K_k H_k) \bar{P}_k
/// \end{aligned}
/// $$
///
/// where:
/// - $K_k$ is the Kalman gain
/// - $H_k = \frac{\partial h}{\partial x}\big|_{\bar{x}_k}$ is the measurement Jacobian
/// - $h(\cdot)$ is the nonlinear measurement function
/// - $R_k$ is the measurement noise covariance
/// - $z_k$ is the measurement
///
/// # State Configuration
///
/// The EKF supports two state configurations:
///
/// - **9-state**: Navigation states only (position, velocity, attitude)
/// - **15-state**: Navigation states + IMU biases (accelerometer and gyroscope biases)
///
/// The state vector ordering follows:
/// ```text
/// x = [lat, lon, alt, v_n, v_e, v_d, roll, pitch, yaw, b_ax, b_ay, b_az, b_gx, b_gy, b_gz]
/// ```
///
/// # Advantages and Limitations
///
/// **Advantages:**
/// - Computationally efficient (linear algebra only, no sigma point propagation)
/// - Well-understood theory with decades of applications
/// - Deterministic (no random sampling)
/// - Lower memory footprint than UKF
///
/// **Limitations:**
/// - First-order linearization can introduce errors for highly nonlinear systems
/// - May diverge if linearization is poor or process noise is underestimated
/// - Assumes Gaussian distributions (like UKF)
///
/// # References
///
/// - Groves, P. D. "Principles of GNSS, Inertial, and Multisensor Integrated
///   Navigation Systems, 2nd Edition", Chapter 14.2
/// - Bar-Shalom, Y., et al. "Estimation with Applications to Tracking and Navigation",
///   Chapter 5
///
/// # Example
///
/// ```rust
/// use strapdown::NavigationFilter;
/// use strapdown::kalman::{ExtendedKalmanFilter, InitialState};
/// use strapdown::measurements::GPSPositionMeasurement;
/// use strapdown::IMUData;
/// use nalgebra::{DMatrix, Vector3};
///
/// // Initialize with 15-state configuration (with biases)
/// let initial_state = InitialState {
///     latitude: 45.0,
///     longitude: -122.0,
///     altitude: 100.0,
///     northward_velocity: 0.0,
///     eastward_velocity: 0.0,
///     vertical_velocity: 0.0,
///     roll: 0.0,
///     pitch: 0.0,
///     yaw: 0.0,
///     in_degrees: true,
///     is_enu: true,
/// };
///
/// let mut ekf = ExtendedKalmanFilter::new(
///     &initial_state,
///     &[0.0; 6], // IMU biases (3 accel + 3 gyro)
///     vec![1e-6; 15], // Initial covariance diagonal
///     DMatrix::from_diagonal(&nalgebra::DVector::from_vec(vec![1e-9; 15])), // Process noise
///     true, // use_biases
/// );
///
/// // Predict with IMU data
/// let imu_data = IMUData {
///     accel: Vector3::new(0.0, 0.0, 9.81),
///     gyro: Vector3::zeros(),
/// };
/// ekf.predict(&imu_data, 0.01);
///
/// // Update with GPS measurement
/// let gps_meas = GPSPositionMeasurement {
///     latitude: 45.0,
///     longitude: -122.0,
///     altitude: 100.0,
///     horizontal_noise_std: 5.0,
///     vertical_noise_std: 2.0,
/// };
/// ekf.update(&gps_meas);
/// ```
#[derive(Clone)]
pub struct ExtendedKalmanFilter {
    /// State estimate vector (9 or 15 elements)
    mean_state: DVector<f64>,
    /// State covariance matrix (9x9 or 15x15)
    covariance: DMatrix<f64>,
    /// Process noise covariance matrix
    process_noise: DMatrix<f64>,
    /// Size of the state vector
    state_size: usize,
    /// Whether to use 15-state (with biases) or 9-state configuration
    use_biases: bool,
    /// Coordinate frame flag (true for ENU, false for NED)
    is_enu: bool,
    /// Innovation gate applied by `update` together with the recovery policy that keeps
    /// a rejection from being permanent; an empty gate accepts every measurement.
    gate_policy: GatePolicy,
}

impl Debug for ExtendedKalmanFilter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EKF")
            .field("mean_state", &self.mean_state)
            .field("covariance", &self.covariance)
            .field("process_noise", &self.process_noise)
            .field("state_size", &self.state_size)
            .field("use_biases", &self.use_biases)
            .field("is_enu", &self.is_enu)
            .field("gate_policy", &self.gate_policy)
            .finish()
    }
}

impl Display for ExtendedKalmanFilter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExtendedKalmanFilter")
            .field("mean_state", &self.mean_state)
            .field("covariance", &self.covariance)
            .field("process_noise", &self.process_noise)
            .field("state_size", &self.state_size)
            .field("use_biases", &self.use_biases)
            .field("is_enu", &self.is_enu)
            .finish()
    }
}

impl ExtendedKalmanFilter {
    /// Create a new Extended Kalman Filter
    ///
    /// # Arguments
    ///
    /// * `initial_state` - Initial navigation state (position, velocity, attitude)
    /// * `imu_biases` - Initial IMU bias estimates [`b_ax`, `b_ay`, `b_az`, `b_gx`, `b_gy`, `b_gz`]
    /// * `covariance_diagonal` - Initial state uncertainty (diagonal covariance elements)
    /// * `process_noise` - Process noise covariance matrix Q
    /// * `use_biases` - If true, uses 15-state (with biases), otherwise 9-state
    ///
    /// # Returns
    ///
    /// A new `ExtendedKalmanFilter` instance
    ///
    /// # Example
    ///
    /// ```rust
    /// use strapdown::kalman::{ExtendedKalmanFilter, InitialState};
    /// use nalgebra::DMatrix;
    ///
    /// let initial_state = InitialState::default();
    /// let ekf = ExtendedKalmanFilter::new(
    ///     &initial_state,
    ///     &[0.0; 6],
    ///     vec![1e-6; 15],
    ///     DMatrix::from_diagonal(&nalgebra::DVector::from_vec(vec![1e-9; 15])),
    ///     true,
    /// );
    /// ```
    pub fn new(
        initial_state: &InitialState,
        imu_biases: &[f64],
        covariance_diagonal: Vec<f64>,
        process_noise: DMatrix<f64>,
        use_biases: bool,
    ) -> Self {
        // Construct initial state vector
        let mut mean = if initial_state.in_degrees {
            vec![
                initial_state.latitude.to_radians(),
                initial_state.longitude.to_radians(),
                initial_state.altitude,
                initial_state.northward_velocity,
                initial_state.eastward_velocity,
                initial_state.vertical_velocity,
                initial_state.roll.to_radians(),
                initial_state.pitch.to_radians(),
                initial_state.yaw.to_radians(),
            ]
        } else {
            vec![
                initial_state.latitude,
                initial_state.longitude,
                initial_state.altitude,
                initial_state.northward_velocity,
                initial_state.eastward_velocity,
                initial_state.vertical_velocity,
                initial_state.roll,
                initial_state.pitch,
                initial_state.yaw,
            ]
        };

        // Add biases if requested
        if use_biases {
            mean.extend(imu_biases);
        }

        // Check if we have augmented states (covariance_diagonal longer than current mean)
        // If so, initialize augmented states to zero
        let expected_state_size = covariance_diagonal.len();
        while mean.len() < expected_state_size {
            mean.push(0.0);
        }

        let state_size = mean.len();
        let mean_state = DVector::from_vec(mean);
        let covariance = DMatrix::<f64>::from_diagonal(&DVector::from_vec(covariance_diagonal));

        Self {
            mean_state,
            covariance,
            process_noise,
            state_size,
            use_biases,
            is_enu: initial_state.is_enu,
            gate_policy: GatePolicy::default(),
        }
    }
}

impl NavigationFilter for ExtendedKalmanFilter {
    /// Predict step: propagate state and covariance using IMU measurements
    ///
    /// The predict step consists of:
    /// 1. Nonlinear state propagation: $\bar{x} = f(x, u)$
    /// 2. Covariance propagation: $\bar{P} = F P F^T + G Q G^T$
    ///
    /// where $F$ is the state transition Jacobian and $G$ is the process noise Jacobian.
    ///
    /// # Arguments
    ///
    /// * `control_input` - an [`ImuSample`] (integrated $\Delta v$ / $\Delta\theta$) or,
    ///   for callers still holding instantaneous rates, an [`IMUData`].
    /// * `dt` - Time step in seconds. When `control_input` is an [`ImuSample`] this must
    ///   agree with the sample's own `dt`; see the Errors section.
    ///
    /// # Errors
    /// * [`StrapdownError::UnsupportedInput`] if `control_input` is neither inertial form.
    /// * [`StrapdownError::InconsistentTimestep`] if an [`ImuSample`]'s `dt` disagrees with
    ///   the `dt` argument.
    /// * [`StrapdownError::OutOfRange`] or [`StrapdownError::NonFinite`] propagated from
    ///   [`mechanize`].
    ///
    /// # Mathematical Details
    ///
    /// The state transition Jacobian $F$ captures how perturbations in the current state
    /// affect the next state. For the 9-state navigation filter:
    ///
    /// $$
    /// F = \frac{\partial f}{\partial x} = I + \begin{bmatrix}
    /// \frac{\partial \dot{p}}{\partial p} & \frac{\partial \dot{p}}{\partial v} & \frac{\partial \dot{p}}{\partial \epsilon} \\\\
    /// \frac{\partial \dot{v}}{\partial p} & \frac{\partial \dot{v}}{\partial v} & \frac{\partial \dot{v}}{\partial \epsilon} \\\\
    /// \frac{\partial \dot{\epsilon}}{\partial p} & \frac{\partial \dot{\epsilon}}{\partial v} & \frac{\partial \dot{\epsilon}}{\partial \epsilon}
    /// \end{bmatrix} dt
    /// $$
    ///
    /// # Process Noise
    ///
    /// The covariance propagation uses $P_{k+1} = F_k P_k F_k^T + Q_k$, where $Q_k$ is
    /// the process noise covariance matrix. In the full formulation, $Q_k = G Q_w G^T$,
    /// where $G$ is the process noise Jacobian mapping IMU noise to state uncertainty,
    /// and $Q_w$ is the IMU noise covariance. In this implementation, the `process_noise`
    /// parameter is assumed to already incorporate $G Q_w G^T$, i.e., it represents
    /// the final process noise covariance in state space.
    fn predict(
        &mut self,
        control_input: &dyn crate::InputModel,
        dt: f64,
    ) -> Result<(), StrapdownError> {
        let sample = imu_sample_from_input(control_input, "ExtendedKalmanFilter", dt)?;

        // Extract current state
        let mut state = StrapdownState {
            latitude: self.mean_state[0],
            longitude: self.mean_state[1],
            altitude: self.mean_state[2],
            velocity_north: self.mean_state[3],
            velocity_east: self.mean_state[4],
            velocity_vertical: self.mean_state[5],
            attitude: Rotation3::from_euler_angles(
                self.mean_state[6],
                self.mean_state[7],
                self.mean_state[8],
            ),
            is_enu: self.is_enu,
        };

        // Extract biases if present
        let (accel_biases, gyro_biases) = if self.use_biases && self.state_size >= 15 {
            (
                Vector3::new(self.mean_state[9], self.mean_state[10], self.mean_state[11]),
                Vector3::new(
                    self.mean_state[12],
                    self.mean_state[13],
                    self.mean_state[14],
                ),
            )
        } else {
            (Vector3::zeros(), Vector3::zeros())
        };

        // Compensate the sensed increments for the estimated biases. Biases are rates and
        // the increments are their integrals, so the correction is each bias integrated
        // across the interval -- the same increment-domain form the ESKF uses, which keeps
        // a genuine delta-v / delta-theta sample off a lossy round trip through `dt`.
        let corrected_sample = ImuSample {
            delta_v: sample.delta_v - accel_biases * sample.dt,
            delta_theta: sample.delta_theta - gyro_biases * sample.dt,
            dt: sample.dt,
        };
        // The state-transition Jacobian is derived in the rate domain, so it needs the
        // average rates over the interval rather than the increments themselves.
        let corrected_rates = corrected_sample.to_rates()?;

        // Compute state transition Jacobian F (before propagation)
        let f_matrix = crate::linearize::euler_state_transition_jacobian(
            &state,
            &corrected_rates.accel,
            &corrected_rates.gyro,
            corrected_sample.dt,
        );

        // Extend F to full state size if using biases or augmented states
        let f_full = if self.use_biases && self.state_size >= 15 {
            let mut f_ext = DMatrix::<f64>::identity(self.state_size, self.state_size);
            f_ext.view_mut((0, 0), (9, 9)).copy_from(&f_matrix);
            // Bias states and any augmented states have identity dynamics (random walk)
            f_ext
        } else if self.state_size > 9 {
            // Handle augmented states without biases (should not happen, but be defensive)
            let mut f_ext = DMatrix::<f64>::identity(self.state_size, self.state_size);
            f_ext.view_mut((0, 0), (9, 9)).copy_from(&f_matrix);
            f_ext
        } else {
            f_matrix
        };

        // Nonlinear state propagation
        mechanize(&mut state, &corrected_sample)?;

        // Update state vector with propagated values
        self.mean_state[0] = state.latitude;
        self.mean_state[1] = state.longitude;
        self.mean_state[2] = state.altitude;
        self.mean_state[3] = state.velocity_north;
        self.mean_state[4] = state.velocity_east;
        self.mean_state[5] = state.velocity_vertical;
        self.mean_state[6] = state.attitude.euler_angles().0;
        self.mean_state[7] = state.attitude.euler_angles().1;
        self.mean_state[8] = state.attitude.euler_angles().2;
        // Biases remain unchanged (random walk model)

        // Covariance propagation: P_bar = F * P * F^T + Q
        self.covariance = &f_full * &self.covariance * f_full.transpose()
            + process_noise_for_step(&self.process_noise, sample.dt)?;

        // Relative, not absolute: see `regularize_covariance_in_place` and #373. This filter
        // applied the absolute form here *and* in `update`, so it accumulated ~(493 m)^2 of
        // fabricated horizontal variance between fixes.
        regularize_covariance_in_place(&mut self.covariance, &self.process_noise);
        Ok(())
    }

    /// Update step: correct state estimate using a measurement
    ///
    /// The update step incorporates a new measurement to refine the state estimate:
    ///
    /// $$
    /// \begin{aligned}
    /// K &= P H^T (H P H^T + R)^{-1} \\\\
    /// x &= x + K (z - h(x)) \\\\
    /// P &= (I - K H) P
    /// \end{aligned}
    /// $$
    ///
    /// # Arguments
    ///
    /// * `measurement` - Measurement model implementing the `MeasurementModel` trait
    ///
    /// # Supported Measurements
    ///
    /// The EKF supports the following measurement types:
    /// - GPS position (latitude, longitude, altitude)
    /// - GPS velocity (north, east, vertical)
    /// - GPS position + velocity (combined)
    /// - Relative altitude (barometric)
    ///
    /// # Mathematical Details
    ///
    /// The measurement Jacobian $H$ linearizes the measurement model around the
    /// current state estimate:
    ///
    /// $$
    /// H = \frac{\partial h}{\partial x}\bigg|_{\bar{x}}
    /// $$
    ///
    /// For GPS position measurements, $H$ is simply an identity matrix selecting
    /// the position states. For more complex measurements, $H$ captures the
    /// relationship between the measurement and all state components.
    ///
    /// The innovation (measurement residual) is:
    ///
    /// $$
    /// \nu = z - h(\bar{x})
    /// $$
    ///
    /// where $z$ is the actual measurement and $h(\bar{x})$ is the expected
    /// measurement given the predicted state.
    fn update(
        &mut self,
        measurement: &dyn MeasurementModel,
    ) -> Result<UpdateOutcome, StrapdownError> {
        // Jacobian FIRST, deliberately. A geophysical model whose estimate has left the
        // loaded map reports that as an error here, whereas `get_expected_measurement`
        // returns NaN for the same condition. Evaluating the expected measurement first
        // would poison `z_hat` before the error was ever seen.
        let h_9state = measurement.get_jacobian(&self.mean_state)?;

        // Get expected measurement from current state
        let z_hat = measurement.get_expected_measurement(&self.mean_state);

        // Widen H to the filter's state size. Most models observe navigation states
        // only and return 9 columns; a bias-observing model (ZARU) returns the full
        // width and passes through unchanged.
        let h_matrix = expand_measurement_jacobian(h_9state, self.state_size)?;

        // Innovation covariance: S = H * P * H^T + R
        let s = &h_matrix * &self.covariance * h_matrix.transpose() + measurement.get_noise();

        // Innovation (measurement residual): nu = z - z_hat
        let mut innovation = measurement.get_measurement(&self.mean_state)? - &z_hat;
        // Keep angular innovations on the circle (see `wrap_residual`, #286).
        measurement.wrap_residual(&mut innovation);

        // Gate before the gain, so a rejected measurement costs one solve and leaves the
        // state exactly as it was. The covariance is the deliberate exception: a
        // rejection inflates it, or the next fix is judged against the same covariance
        // by a filter that has drifted further, and the rejection is self-reinforcing
        // (#340).
        let decision = self.gate_policy.evaluate(&innovation, &s, "EKF")?;
        let outcome = decision.outcome;
        if !outcome.accepted {
            // Only the directions `h_matrix` observes: a rejected barometric altitude is
            // evidence about the altitude channel and about nothing else, and inflating
            // the whole covariance on it lets the *next* accepted measurement of another
            // kind apply a wildly over-weighted correction.
            let projected = &self.covariance * h_matrix.transpose();
            let observed = &h_matrix * &projected;
            decision.inflate_observed(&mut self.covariance, &projected, &observed);
            return Ok(outcome);
        }

        // Kalman gain: K = P * H^T * S^(-1)
        let k = self.covariance.clone()
            * h_matrix.transpose()
            * robust_spd_solve(&symmetrize(&s), &DMatrix::identity(s.nrows(), s.ncols()))?
                .transpose();

        // State update: x = x + K * nu
        self.mean_state += &k * innovation;

        // Report attitude on the same branch `predict` writes (#314).
        wrap_attitude_onto_principal_branch(&mut self.mean_state);

        // Covariance update (Joseph form for numerical stability):
        // P = (I - K*H)*P*(I - K*H)^T + K*R*K^T
        let i_kh = DMatrix::identity(self.state_size, self.state_size) - &k * &h_matrix;
        let r = measurement.get_noise();
        self.covariance = &i_kh * &self.covariance * i_kh.transpose() + &k * r * k.transpose();

        // Relative, not absolute: see `regularize_covariance_in_place` and #373.
        regularize_covariance_in_place(&mut self.covariance, &self.process_noise);
        Ok(outcome)
    }

    fn set_innovation_gate(&mut self, gate: Option<InnovationGate>) -> bool {
        self.gate_policy.set_gate(gate);
        true
    }

    fn set_gate_recovery(&mut self, recovery: GateRecovery) -> bool {
        self.gate_policy.set_recovery(recovery);
        true
    }

    /// Get the current state estimate
    ///
    /// Returns the mean state vector containing position, velocity, attitude,
    /// and optionally IMU biases (if configured with 15-state mode).
    ///
    /// # Returns
    ///
    /// State vector: [lat (rad), lon (rad), alt (m), `v_n` (m/s), `v_e` (m/s), `v_d` (m/s),
    ///                roll (rad), pitch (rad), yaw (rad), `b_ax`, `b_ay`, `b_az`, `b_gx`, `b_gy`, `b_gz`]
    ///
    /// The three Euler angles come back on -pi..pi, the branch `Rotation3::euler_angles`
    /// returns; pitch is additionally inside -pi/2..pi/2 once the estimate has been through
    /// a `predict`, since that is where the triple is re-derived from the rotation.
    fn get_estimate(&self) -> DVector<f64> {
        self.mean_state.clone()
    }

    /// Get the current state uncertainty (covariance matrix)
    ///
    /// Returns the covariance matrix representing uncertainty in the state estimate.
    ///
    /// # Returns
    ///
    /// Covariance matrix P (9x9 or 15x15 depending on configuration)
    fn get_certainty(&self) -> DMatrix<f64> {
        self.covariance.clone()
    }
}

/// Error-State Kalman Filter (ESKF) implementation for strapdown INS
///
/// The Error-State Kalman Filter is the standard approach for strapdown inertial navigation
/// systems. Unlike the full-state EKF which directly estimates position, velocity, and
/// attitude, the ESKF maintains:
///
/// 1. **Nominal state**: A high-fidelity nonlinear propagation of the full navigation state
///    using quaternion or DCM attitude representation
/// 2. **Error state**: A small-perturbation linear estimate of errors in position, velocity,
///    attitude, and IMU biases
///
/// # Key Advantages Over Full-State EKF
///
/// - **Avoids attitude singularities**: Uses quaternions for nominal state, small angles for errors
/// - **Better linearization**: Errors remain small, making linear approximations more accurate
/// - **Maintains quaternion normalization**: Nominal quaternion is always unit-length
/// - **More accurate**: Error dynamics are simpler and better behaved
///
/// # Mathematical Background
///
/// The ESKF operates in two stages:
///
/// ## Predict Step
///
/// 1. **Nominal state propagation** (nonlinear, high-fidelity):
///    $$
///    \dot{x}_{\text{nom}} = f(x_{\text{nom}}, u - b)
///    $$
///    where $b$ are the IMU biases and $u$ are the IMU measurements
///
/// 2. **Error state propagation** (linear):
///    $$
///    \begin{aligned}
///    \dot{\delta x} &= F_{\delta x} \delta x + G w \\\\
///    \delta P &= F_{\delta x} P F_{\delta x}^T + G Q G^T
///    \end{aligned}
///    $$
///
/// ## Update Step
///
/// 1. **Measurement residual**:
///    $$
///    \nu = z - h(x_{\text{nom}})
///    $$
///
/// 2. **Kalman gain and error state update**:
///    $$
///    \begin{aligned}
///    K &= P H^T (H P H^T + R)^{-1} \\\\
///    \delta x &= K \nu
///    \end{aligned}
///    $$
///
/// 3. **Error injection** (reset nominal state):
///    $$
///    \begin{aligned}
///    x_{\text{nom}} &\leftarrow x_{\text{nom}} \oplus \delta x \\\\
///    \delta x &\leftarrow 0 \\\\
///    P &\leftarrow (I - K H) P (I - K H)^T + K R K^T
///    \end{aligned}
///    $$
///    where $\oplus$ represents the error injection operation (different for each state component)
///
/// # State Representation
///
/// ## Nominal State (9 components, stored as specific types):
/// - **Position**: Geodetic coordinates (lat, lon, alt)
/// - **Velocity**: NED/ENU frame (`v_n`, `v_e`, `v_d`)  
/// - **Attitude**: Unit quaternion q or DCM
///
/// ## Error State (15 components, always small):
/// ```text
/// δx = [δp_n, δp_e, δp_d,           // position error (m)
///       δv_n, δv_e, δv_d,           // velocity error (m/s)
///       δθ_x, δθ_y, δθ_z,           // attitude error (small angles, rad)
///       δb_ax, δb_ay, δb_az,        // accelerometer bias error (m/s²)
///       δb_gx, δb_gy, δb_gz]        // gyroscope bias error (rad/s)
/// ```
///
/// ## IMU Biases (6 components, part of nominal state):
/// - Accelerometer biases: `b_a` ∈ ℝ³ (m/s²)
/// - Gyroscope biases: `b_g` ∈ ℝ³ (rad/s)
/// - Modeled as random walk: $\dot{b} = w_b$ where $w_b ~ N(0, Q_b)$
///
/// # Error Injection (Reset)
///
/// After each measurement update, errors are injected into the nominal state:
///
/// - **Position**: $p \leftarrow p + \delta p$ (simple addition in local frame)
/// - **Velocity**: $v \leftarrow v + \delta v$ (simple addition)
/// - **Attitude**: $q \leftarrow q \otimes q(\delta\theta)$ (quaternion multiplication)
///   where $q(\delta\theta) \approx [1, \frac{1}{2}\delta\theta_x, \frac{1}{2}\delta\theta_y, \frac{1}{2}\delta\theta_z]^T$
/// - **Biases**: $b \leftarrow b + \delta b$ (simple addition)
///
/// Then error state is reset: $\delta x \leftarrow 0$ and covariance is updated.
///
/// # References
///
/// - Sola, J. "Quaternion kinematics for the error-state Kalman filter" (2017)
/// - Groves, P. D. "Principles of GNSS, Inertial, and Multisensor Integrated\
///   Navigation Systems, 2nd Edition", Chapter 14
/// - Trawny, N. & Roumeliotis, S. "Indirect Kalman Filter for 3D Attitude Estimation" (2005)
///
/// # Example
///
/// ```rust
/// use strapdown::NavigationFilter;
/// use strapdown::kalman::{ErrorStateKalmanFilter, InitialState};
/// use strapdown::measurements::GPSPositionMeasurement;
/// use strapdown::IMUData;
/// use nalgebra::{DMatrix, Vector3};
///
/// // Initialize ESKF
/// let initial_state = InitialState {
///     latitude: 45.0,
///     longitude: -122.0,
///     altitude: 100.0,
///     northward_velocity: 0.0,
///     eastward_velocity: 0.0,
///     vertical_velocity: 0.0,
///     roll: 0.0,
///     pitch: 0.0,
///     yaw: 0.0,
///     in_degrees: true,
///     is_enu: true,
/// };
///
/// let mut eskf = ErrorStateKalmanFilter::new(
///     &initial_state,
///     &[0.0; 6], // Initial IMU biases (3 accel + 3 gyro)
///     vec![1e-6; 15], // Initial error covariance diagonal
///     DMatrix::from_diagonal(&nalgebra::DVector::from_vec(vec![1e-9; 15])), // Process noise
/// );
///
/// // Predict with IMU data
/// let imu_data = IMUData {
///     accel: Vector3::new(0.0, 0.0, 9.81),
///     gyro: Vector3::zeros(),
/// };
/// eskf.predict(&imu_data, 0.01);
///
/// // Update with GPS measurement
/// let gps_meas = GPSPositionMeasurement {
///     latitude: 45.0,
///     longitude: -122.0,
///     altitude: 100.0,
///     horizontal_noise_std: 5.0,
///     vertical_noise_std: 2.0,
/// };
/// eskf.update(&gps_meas);
/// ```
#[derive(Clone)]
pub struct ErrorStateKalmanFilter {
    /// Nominal position state (latitude, longitude, altitude)
    nominal_latitude: f64, // radians
    nominal_longitude: f64, // radians
    nominal_altitude: f64,  // meters

    /// Nominal velocity state (NED/ENU frame)
    nominal_velocity_north: f64, // m/s
    nominal_velocity_east: f64,     // m/s
    nominal_velocity_vertical: f64, // m/s

    /// Nominal attitude as unit quaternion [w, x, y, z]
    /// Represents rotation from body frame to local-level frame (NED/ENU)
    nominal_quaternion: nalgebra::Vector4<f64>,

    /// IMU biases (part of nominal state, augmented)
    nominal_accel_bias: Vector3<f64>, // m/s²
    nominal_gyro_bias: Vector3<f64>, // rad/s

    /// Error state vector (15 elements: 3 pos + 3 vel + 3 att + 3 `acc_bias` + 3 `gyro_bias`)
    /// Initialized to zero and reset to zero after each update
    error_state: DVector<f64>,

    /// Error state covariance matrix (15x15)
    error_covariance: DMatrix<f64>,

    /// Process noise covariance matrix (15x15)
    process_noise: DMatrix<f64>,

    /// Coordinate frame flag (true for ENU, false for NED)
    is_enu: bool,

    /// Innovation gate applied by `update` together with the recovery policy that keeps
    /// a rejection from being permanent; an empty gate accepts every measurement.
    gate_policy: GatePolicy,
}

impl Debug for ErrorStateKalmanFilter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ESKF")
            .field(
                "nominal_position",
                &[
                    self.nominal_latitude,
                    self.nominal_longitude,
                    self.nominal_altitude,
                ],
            )
            .field(
                "nominal_velocity",
                &[
                    self.nominal_velocity_north,
                    self.nominal_velocity_east,
                    self.nominal_velocity_vertical,
                ],
            )
            .field("nominal_quaternion", &self.nominal_quaternion)
            .field("error_state", &self.error_state)
            .field("error_covariance", &self.error_covariance)
            .field("process_noise", &self.process_noise)
            .field("is_enu", &self.is_enu)
            .finish_non_exhaustive()
    }
}

impl Display for ErrorStateKalmanFilter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ErrorStateKalmanFilter")
            .field(
                "nominal_position",
                &[
                    self.nominal_latitude,
                    self.nominal_longitude,
                    self.nominal_altitude,
                ],
            )
            .field(
                "nominal_velocity",
                &[
                    self.nominal_velocity_north,
                    self.nominal_velocity_east,
                    self.nominal_velocity_vertical,
                ],
            )
            .field("nominal_quaternion", &self.nominal_quaternion)
            .field("error_state", &self.error_state)
            .field("error_covariance", &self.error_covariance)
            .field("process_noise", &self.process_noise)
            .field("is_enu", &self.is_enu)
            .finish()
    }
}

/// Relative floor used to keep a filter covariance numerically conditioned.
///
/// A navigation state spans twelve orders of magnitude in units: latitude and
/// longitude are radians (a 1 m error is ~1.6e-7 rad, so ~2.5e-14 rad² of variance),
/// while accelerometer biases are m/s². A single absolute value added to every
/// diagonal entry cannot serve both -- the `1e-9` this used to add was simultaneously
/// ~(200 m)² of bogus horizontal position variance and a rounding error for velocity,
/// and it silently overwrote any deliberately small covariance (freezing a state by
/// giving it a 1e-12 variance did nothing). See #266.
///
/// #266 fixed that for the ESKF and stopped there. The UKF and EKF kept the absolute
/// form until #373, which is why they reported a horizontal sigma they could not
/// possibly have earned -- see [`regularize_covariance_in_place`].
const COVARIANCE_JITTER_RELATIVE: f64 = 1e-9;

/// Build the per-step process-noise matrix from a spectral density and a step interval.
///
/// `density` carries a variance **per second** ([`crate::sim::DEFAULT_PROCESS_NOISE_DENSITY`]),
/// so the noise a step contributes is `q * dt`. Before #374 the filters added the array
/// itself, once per IMU sample, with no `dt` anywhere -- which made the process noise a
/// trajectory saw proportional to its sample rate rather than to elapsed time. The data this
/// repository ships spans 1 Hz to 50 Hz, so one constant meant a 50x spread in the modelled
/// random walk.
///
/// A negative or non-finite `dt` cannot produce a valid covariance, so it is refused rather
/// than propagated into one.
///
/// # Errors
///
/// [`StrapdownError::OutOfRange`] when `dt` is negative or not finite.
fn process_noise_for_step(density: &DMatrix<f64>, dt: f64) -> Result<DMatrix<f64>, StrapdownError> {
    if !dt.is_finite() || dt < 0.0 {
        return Err(StrapdownError::OutOfRange {
            what: "process noise step interval",
            value: dt,
            min: 0.0,
            max: f64::INFINITY,
        });
    }
    Ok(density * dt)
}

/// Symmetrise a covariance and add a jitter proportional to each state's own scale.
///
/// The alternative -- one absolute value added to every diagonal entry -- cannot work on
/// a state vector whose units differ by twelve orders of magnitude, and the arithmetic of
/// getting it wrong is worth writing down because it went unnoticed for so long.
///
/// A latitude variance floored at `1e-9 rad²` is a standard deviation of
/// `sqrt(1e-9) = 3.16e-5 rad`, which at the Earth's radius is **201.5 m**. Against a GNSS
/// fix specified at 5 m (`R = 6.16e-13 rad²`), that prior is 1,624 times the measurement
/// noise, so the Kalman gain `K = P/(P+R)` is 0.9994: the filter discards its own
/// prediction and lands on each fix. The EKF applied the same floor in `predict` *and*
/// `update`, so between fixes it accumulated ~`6e-9 rad²` -- **493.5 m**, a prior share of
/// 9,742, and a gain of 0.9999.
///
/// Those are not estimates. `core/tests/aiding.rs` measured 201 m and 493 m with prior
/// shares of 1,622 and 9,729 before this was understood, and recorded them beside the
/// ESKF's 0.063 without being able to say why the ESKF differed. The ESKF differed because
/// #266 had already given it this function's behaviour.
///
/// `process_noise` supplies the per-state floor, so a state whose variance has collapsed
/// to zero still receives a jitter in its own units rather than nothing at all.
fn regularize_covariance_in_place(covariance: &mut DMatrix<f64>, process_noise: &DMatrix<f64>) {
    *covariance = symmetrize(covariance);
    for i in 0..covariance.nrows() {
        let scale = covariance[(i, i)].abs().max(process_noise[(i, i)].abs());
        covariance[(i, i)] += COVARIANCE_JITTER_RELATIVE * scale;
    }
}

/// Anti-windup caps for ESKF bias estimates (see `inject_error_state`, #286).
///
/// Orders of magnitude above legitimate consumer-MEMS turn-on biases
/// (~0.1 m/s², ~0.01 rad/s) and far below the runaway values a persistently
/// faulty aiding sensor can otherwise produce (9 m/s², 6 rad/s).
const MAX_ACCEL_BIAS_MPS2: f64 = 2.0;
const MAX_GYRO_BIAS_RPS: f64 = 0.05;

impl ErrorStateKalmanFilter {
    /// Create a new Error-State Kalman Filter
    ///
    /// # Arguments
    ///
    /// * `initial_state` - Initial navigation state (position, velocity, attitude)
    /// * `imu_biases` - Initial IMU bias estimates [`b_ax`, `b_ay`, `b_az`, `b_gx`, `b_gy`, `b_gz`]
    /// * `error_covariance_diagonal` - Initial error state uncertainty (15 diagonal elements)
    /// * `process_noise` - Process noise covariance matrix Q (15x15)
    ///
    /// # Returns
    ///
    /// A new `ErrorStateKalmanFilter` instance with error state initialized to zero
    ///
    /// # Example
    ///
    /// ```rust
    /// use strapdown::kalman::{ErrorStateKalmanFilter, InitialState};
    /// use nalgebra::DMatrix;
    ///
    /// let initial_state = InitialState::default();
    /// let eskf = ErrorStateKalmanFilter::new(
    ///     &initial_state,
    ///     &[0.0; 6],
    ///     vec![1e-6; 15],
    ///     DMatrix::from_diagonal(&nalgebra::DVector::from_vec(vec![1e-9; 15])),
    /// );
    /// ```
    pub fn new(
        initial_state: &InitialState,
        imu_biases: &[f64],
        error_covariance_diagonal: Vec<f64>,
        process_noise: DMatrix<f64>,
    ) -> Self {
        // Convert initial Euler angles to quaternion for nominal state.
        // `from_euler_angles` takes radians unconditionally, so degree inputs
        // must be converted here. (This was previously inverted -- radian
        // inputs were converted *to* degrees -- which built a wildly wrong
        // initial DCM whenever the initial attitude was non-zero and drove
        // the vertical channel unstable within seconds. See #286.)
        let (roll, pitch, yaw) = if initial_state.in_degrees {
            (
                initial_state.roll.to_radians(),
                initial_state.pitch.to_radians(),
                initial_state.yaw.to_radians(),
            )
        } else {
            (initial_state.roll, initial_state.pitch, initial_state.yaw)
        };

        // Convert Euler angles to quaternion (XYZ sequence: roll, pitch, yaw)
        let rotation = Rotation3::from_euler_angles(roll, pitch, yaw);
        let unit_quat = UnitQuaternion::from_rotation_matrix(&rotation);
        let nominal_quaternion =
            nalgebra::Vector4::new(unit_quat.w, unit_quat.i, unit_quat.j, unit_quat.k);

        // Initialize nominal state
        let (nominal_latitude, nominal_longitude) = if initial_state.in_degrees {
            (
                initial_state.latitude.to_radians(),
                initial_state.longitude.to_radians(),
            )
        } else {
            (initial_state.latitude, initial_state.longitude)
        };

        let nominal_accel_bias = Vector3::new(imu_biases[0], imu_biases[1], imu_biases[2]);
        let nominal_gyro_bias = Vector3::new(imu_biases[3], imu_biases[4], imu_biases[5]);

        // Initialize error state to zero (15 elements)
        let error_state = DVector::zeros(15);

        // Initialize error covariance
        let error_covariance =
            DMatrix::from_diagonal(&DVector::from_vec(error_covariance_diagonal));

        Self {
            nominal_latitude,
            nominal_longitude,
            nominal_altitude: initial_state.altitude,
            nominal_velocity_north: initial_state.northward_velocity,
            nominal_velocity_east: initial_state.eastward_velocity,
            nominal_velocity_vertical: initial_state.vertical_velocity,
            nominal_quaternion,
            nominal_accel_bias,
            nominal_gyro_bias,
            error_state,
            error_covariance,
            process_noise,
            is_enu: initial_state.is_enu,
            gate_policy: GatePolicy::default(),
        }
    }

    /// Regularise the error covariance after a predict or update.
    ///
    /// Symmetrises, then adds a jitter proportional to each state's own variance
    /// rather than a single absolute value shared across states whose units differ
    /// by twelve orders of magnitude. The process-noise diagonal supplies a
    /// per-state floor so a state whose variance has collapsed to zero still gets a
    /// jitter in its own units.
    fn regularize_covariance(&mut self) {
        regularize_covariance_in_place(&mut self.error_covariance, &self.process_noise);
    }

    /// Inject error state into nominal state and reset error state to zero
    ///
    /// This is the key operation that distinguishes ESKF from full-state EKF.
    /// After computing the error state correction from measurements, we:
    /// 1. Add position/velocity/bias errors directly to nominal state
    /// 2. Apply attitude error using quaternion multiplication (small angle approximation)
    /// 3. Reset error state to zero
    /// 4. Update error covariance to account for the reset
    ///
    /// # Mathematical Details
    ///
    /// For position, velocity, and biases:
    /// $$
    /// x_{\text{nom}} \leftarrow x_{\text{nom}} + \delta x
    /// $$
    ///
    /// For attitude (using small angle approximation):
    /// $$
    /// q_{\text{nom}} \leftarrow q_{\text{nom}} \otimes \begin{bmatrix} 1 \\\\ \frac{1}{2}\delta\theta \end{bmatrix}
    /// $$
    /// where $\delta\theta$ is the attitude error (small angles)
    ///
    /// The error state is then reset: $\delta x \leftarrow 0$
    fn inject_error_state(&mut self) {
        // Position error injection.
        //
        // delta_p is carried in the SAME units the error-state transition Jacobian and
        // the measurement Jacobians use: radians for latitude/longitude, metres for
        // altitude. `error_state_transition_jacobian` sets f[(0,3)] = dt / r_n, which
        // converts a velocity error in m/s into a *radian* position-error rate, and the
        // GPS position Jacobian is the identity against a radian-valued measurement
        // (see `measurements::GPSPositionMeasurement::get_measurement`). So the
        // correction is applied directly, with no metres-to-radians conversion.
        //
        // Dividing by the principal radii here (as this did before) rescaled the
        // horizontal correction by ~1/6.4e6, leaving the horizontal channel effectively
        // open loop: the lat/lon innovation never nulled, and the filter drove the
        // residual into velocity, tilt and accelerometer bias instead. See #266.
        self.nominal_latitude += self.error_state[0];
        self.nominal_longitude += self.error_state[1];
        self.nominal_altitude += self.error_state[2];

        // Velocity error injection
        self.nominal_velocity_north += self.error_state[3];
        self.nominal_velocity_east += self.error_state[4];
        self.nominal_velocity_vertical += self.error_state[5];

        // Attitude error injection using quaternion multiplication
        // Small angle approximation: q(δθ) ≈ [1, δθ/2]^T
        let delta_theta = Vector3::new(
            self.error_state[6],
            self.error_state[7],
            self.error_state[8],
        );

        // Create error quaternion from small angle vector
        let delta_q = nalgebra::Vector4::new(
            1.0,
            delta_theta[0] * 0.5,
            delta_theta[1] * 0.5,
            delta_theta[2] * 0.5,
        );

        // Quaternion multiplication: q_new = q_nominal ⊗ q_error
        let w = self.nominal_quaternion[0];
        let x = self.nominal_quaternion[1];
        let y = self.nominal_quaternion[2];
        let z = self.nominal_quaternion[3];

        let dw = delta_q[0];
        let dx = delta_q[1];
        let dy = delta_q[2];
        let dz = delta_q[3];

        self.nominal_quaternion[0] = w * dw - x * dx - y * dy - z * dz;
        self.nominal_quaternion[1] = w * dx + x * dw + y * dz - z * dy;
        self.nominal_quaternion[2] = w * dy - x * dz + y * dw + z * dx;
        self.nominal_quaternion[3] = w * dz + x * dy - y * dx + z * dw;

        // Normalize quaternion to maintain unit length
        let norm = self.nominal_quaternion.norm();
        self.nominal_quaternion /= norm;

        // Bias error injection
        self.nominal_accel_bias[0] += self.error_state[9];
        self.nominal_accel_bias[1] += self.error_state[10];
        self.nominal_accel_bias[2] += self.error_state[11];

        self.nominal_gyro_bias[0] += self.error_state[12];
        self.nominal_gyro_bias[1] += self.error_state[13];
        self.nominal_gyro_bias[2] += self.error_state[14];

        // Anti-windup: keep bias estimates within physically plausible bounds.
        //
        // Biases are observed only indirectly (bias -> tilt -> velocity), so a
        // persistently faulty aiding sensor (e.g. an uncalibrated phone
        // magnetometer disagreeing with yaw by ~80°, see #286) can drag them to
        // physically impossible values (9 m/s², 6 rad/s), after which the
        // bias-corrupted nominal propagation defeats every other update. The
        // EKF never hits this because its bias states never move at all; the
        // ESKF must actively defend the linear regime its error model assumes.
        // Caps are set orders of magnitude above legitimate consumer-MEMS
        // turn-on biases, so they never bind in normal operation.
        for b in self.nominal_accel_bias.iter_mut() {
            *b = b.clamp(-MAX_ACCEL_BIAS_MPS2, MAX_ACCEL_BIAS_MPS2);
        }
        for b in self.nominal_gyro_bias.iter_mut() {
            *b = b.clamp(-MAX_GYRO_BIAS_RPS, MAX_GYRO_BIAS_RPS);
        }

        // Reset error state to zero
        self.error_state.fill(0.0);
    }

    /// Finite-difference Jacobian of the expected measurement with respect
    /// to the body-frame attitude error vector.
    ///
    /// The analytic Jacobians supplied by measurement models differentiate
    /// with respect to Euler angles, but the ESKF error state carries a
    /// body-frame rotation vector. The two parameterisations are related by a
    /// nontrivial map that degenerates at high pitch, so copying the analytic
    /// attitude columns into the error-state `H` rotates corrections onto the
    /// wrong axes (see #286). Differentiating the expected measurement
    /// numerically with the *same* perturbation convention the injection uses
    /// (right-multiplied body-frame small rotation) is exact by construction
    /// and generic across measurement types, including future
    /// attitude-dependent ones.
    ///
    /// Only the attitude columns are produced here; position/velocity columns
    /// share units between the 9-state and error-state forms and are copied
    /// from the analytic Jacobian by the caller.
    ///
    /// Note the differentiation target: it must be the expected measurement
    /// `h(x)`, not the innovation `z - h(x)`, because the update always forms
    /// the residual as `z - h(x)` while taking `H = dh/dx`. Differentiating
    /// the innovation would flip the sign of every column that `z` does not
    /// share (e.g. yaw), destabilising the very channel being corrected.
    pub(crate) fn attitude_error_jacobian<M: MeasurementModel + ?Sized>(
        measurement: &M,
        nominal_quaternion_wxyz: &nalgebra::Vector4<f64>,
        nominal_state_vec: &DVector<f64>,
    ) -> DMatrix<f64> {
        const EPS: f64 = 1e-8;
        let meas_dim = measurement.get_dimension();
        let mut h_att = DMatrix::<f64>::zeros(meas_dim, 3);
        let q_nom = UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(
            nominal_quaternion_wxyz[0],
            nominal_quaternion_wxyz[1],
            nominal_quaternion_wxyz[2],
            nominal_quaternion_wxyz[3],
        ));
        let h_base = measurement.get_expected_measurement(nominal_state_vec);
        for i in 0..3 {
            let axis = if i == 0 {
                Vector3::x_axis()
            } else if i == 1 {
                Vector3::y_axis()
            } else {
                Vector3::z_axis()
            };
            let q_pert = q_nom * UnitQuaternion::from_axis_angle(&axis, EPS);
            let euler_pert = q_pert.euler_angles();
            let mut vec_pert = nominal_state_vec.clone();
            vec_pert[6] = euler_pert.0;
            vec_pert[7] = euler_pert.1;
            vec_pert[8] = euler_pert.2;
            let h_pert = measurement.get_expected_measurement(&vec_pert);
            for r in 0..meas_dim {
                h_att[(r, i)] = (h_pert[r] - h_base[r]) / EPS;
            }
        }
        h_att
    }
}

impl NavigationFilter for ErrorStateKalmanFilter {
    /// Predict step: propagate nominal state and error covariance
    ///
    /// The ESKF predict consists of two parts:
    /// 1. Nonlinear nominal state propagation using full strapdown equations
    /// 2. Linear error state covariance propagation
    ///
    /// # Arguments
    ///
    /// * `control_input` - an [`ImuSample`] (integrated $\Delta v$ / $\Delta\theta$) or,
    ///   for callers still holding instantaneous rates, an [`IMUData`].
    /// * `dt` - Time step in seconds. When `control_input` is an [`ImuSample`] this must
    ///   agree with the sample's own `dt`; see the Errors section.
    ///
    /// # Errors
    /// * [`StrapdownError::UnsupportedInput`] if `control_input` is neither inertial form.
    /// * [`StrapdownError::InconsistentTimestep`] if an [`ImuSample`]'s `dt` disagrees with
    ///   the `dt` argument.
    /// * [`StrapdownError::OutOfRange`] or [`StrapdownError::NonFinite`] propagated from
    ///   [`mechanize`].
    ///
    /// # Mathematical Details
    ///
    /// Nominal state propagation uses the bias-corrected increments. The biases are rates,
    /// so the correction is each bias integrated across the same interval:
    /// $$
    /// \begin{aligned}
    /// \Delta v^b &= \Delta v^b_{\text{measured}} - b_a \Delta t \\\\
    /// \Delta\theta^b &= \Delta\theta^b_{\text{measured}} - b_g \Delta t
    /// \end{aligned}
    /// $$
    ///
    /// Error covariance propagation:
    /// $$
    /// P_{k+1} = F_k P_k F_k^T + G_k Q_k G_k^T
    /// $$
    ///
    /// where $F_k$ is the error-state transition Jacobian and $G_k$ maps
    /// process noise to error states.
    fn predict(
        &mut self,
        control_input: &dyn crate::InputModel,
        dt: f64,
    ) -> Result<(), StrapdownError> {
        let sample = imu_sample_from_input(control_input, "ErrorStateKalmanFilter", dt)?;

        // Compensate the sensed increments for the estimated biases. Biases are rates
        // (m/s^2, rad/s) and the increments are their integrals, so the correction is the
        // bias integrated over the same interval. Doing this in the increment domain rather
        // than converting back to rates keeps a sample that arrived as genuine delta-v /
        // delta-theta from making a lossy round trip through a division by `dt`.
        let corrected_sample = ImuSample {
            delta_v: sample.delta_v - self.nominal_accel_bias * sample.dt,
            delta_theta: sample.delta_theta - self.nominal_gyro_bias * sample.dt,
            dt: sample.dt,
        };
        // The error-state Jacobian is derived in the rate domain (Groves 14.2), so it needs
        // the average rates over the interval rather than the increments themselves.
        let corrected_rates = corrected_sample.to_rates()?;
        let corrected_accel = corrected_rates.accel;
        let corrected_gyro = corrected_rates.gyro;

        // ===== Nominal State Propagation (Nonlinear) =====

        // Convert quaternion to rotation matrix for propagation
        let qw = self.nominal_quaternion[0];
        let qx = self.nominal_quaternion[1];
        let qy = self.nominal_quaternion[2];
        let qz = self.nominal_quaternion[3];

        let unit_quat = UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(qw, qx, qy, qz));
        let rotation =
            Rotation3::from_matrix_unchecked(unit_quat.to_rotation_matrix().into_inner());

        // Create StrapdownState for nominal propagation
        let mut nominal_state = StrapdownState {
            latitude: self.nominal_latitude,
            longitude: self.nominal_longitude,
            altitude: self.nominal_altitude,
            velocity_north: self.nominal_velocity_north,
            velocity_east: self.nominal_velocity_east,
            velocity_vertical: self.nominal_velocity_vertical,
            attitude: rotation,
            is_enu: self.is_enu,
        };

        // Propagate nominal state using full strapdown mechanization
        mechanize(&mut nominal_state, &corrected_sample)?;

        // Update nominal state from propagation
        self.nominal_latitude = nominal_state.latitude;
        self.nominal_longitude = nominal_state.longitude;
        self.nominal_altitude = nominal_state.altitude;
        self.nominal_velocity_north = nominal_state.velocity_north;
        self.nominal_velocity_east = nominal_state.velocity_east;
        self.nominal_velocity_vertical = nominal_state.velocity_vertical;

        // Convert rotation back to quaternion and normalize
        let unit_quat = UnitQuaternion::from_rotation_matrix(&nominal_state.attitude);
        self.nominal_quaternion =
            nalgebra::Vector4::new(unit_quat.w, unit_quat.i, unit_quat.j, unit_quat.k);
        let norm = self.nominal_quaternion.norm();
        self.nominal_quaternion /= norm;

        // ===== Error State Covariance Propagation (Linear) =====

        // Compute error-state transition Jacobian F_δx
        // Note: This is different from full-state Jacobian because we're linearizing
        // around the nominal trajectory, and attitude errors use small angles
        let f_error = crate::linearize::error_state_transition_jacobian(
            &nominal_state,
            &corrected_accel,
            &corrected_gyro,
            corrected_sample.dt,
        );

        // Propagate error covariance: P = F * P * F^T + Q
        self.error_covariance = &f_error * &self.error_covariance * f_error.transpose()
            + process_noise_for_step(&self.process_noise, sample.dt)?;

        self.regularize_covariance();
        Ok(())
    }

    /// Update step: compute error state correction and inject into nominal state
    ///
    /// The ESKF update:
    /// 1. Computes innovation using nominal state
    /// 2. Updates error state using standard Kalman equations
    /// 3. Injects error into nominal state
    /// 4. Resets error state to zero
    /// 5. Updates error covariance
    ///
    /// # Arguments
    ///
    /// * `measurement` - Measurement model implementing the `MeasurementModel` trait
    ///
    /// # Mathematical Details
    ///
    /// Innovation (residual):
    /// $$
    /// \nu = z - h(x_{\text{nom}})
    /// $$
    ///
    /// Kalman gain:
    /// $$
    /// K = P H^T (H P H^T + R)^{-1}
    /// $$
    ///
    /// Error state update:
    /// $$
    /// \delta x = K \nu
    /// $$
    ///
    /// Error injection and reset (see `inject_error_state` for details)
    fn update(
        &mut self,
        measurement: &dyn MeasurementModel,
    ) -> Result<UpdateOutcome, StrapdownError> {
        // Nominal state vector for measurement prediction, in the 15-state layout the
        // error state uses. The bias entries matter: ZARU observes the gyro bias
        // directly, and a 9-element nominal vector would have left it nothing to
        // predict from.
        let mut nominal_state_vec = DVector::zeros(15);
        nominal_state_vec[0] = self.nominal_latitude;
        nominal_state_vec[1] = self.nominal_longitude;
        nominal_state_vec[2] = self.nominal_altitude;
        nominal_state_vec[3] = self.nominal_velocity_north;
        nominal_state_vec[4] = self.nominal_velocity_east;
        nominal_state_vec[5] = self.nominal_velocity_vertical;

        // Convert quaternion to Euler angles for measurement model
        let quat = nalgebra::UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(
            self.nominal_quaternion[0],
            self.nominal_quaternion[1],
            self.nominal_quaternion[2],
            self.nominal_quaternion[3],
        ));
        let euler = quat.euler_angles();
        nominal_state_vec[6] = euler.0; // roll
        nominal_state_vec[7] = euler.1; // pitch
        nominal_state_vec[8] = euler.2; // yaw

        // Biases, in `get_estimate` order: 3 accelerometer then 3 gyroscope.
        // Left unwrapped and unrounded -- this vector is a linearization point, not
        // an output.
        for axis in 0..3 {
            nominal_state_vec[9 + axis] = self.nominal_accel_bias[axis];
            nominal_state_vec[12 + axis] = self.nominal_gyro_bias[axis];
        }

        // Get expected measurement from nominal state
        let z_hat = measurement.get_expected_measurement(&nominal_state_vec);

        // Measurement Jacobian, supplied by the measurement model itself.
        // Every `MeasurementModel` implementor is required to provide this, so new
        // measurement types (ZUPT/ZARU, geophysical anomalies) work here without
        // the filter needing to know about them. Nine-column Jacobians are padded
        // into the bias block; a ZARU Jacobian already spans all fifteen.
        let meas_dim = measurement.get_dimension();
        let mut h_error =
            expand_measurement_jacobian(measurement.get_jacobian(&nominal_state_vec)?, 15)?;

        // Innovation (measurement residual): nu = z - z_hat. Computed here
        // (rather than below) because the attitude-column correction needs it.
        // Angular components are wrapped onto the circle first so a z/z_hat
        // pair straddling the branch cut cannot inject a phantom ±2π kick.
        let mut innovation = measurement.get_measurement(&nominal_state_vec)? - &z_hat;
        measurement.wrap_residual(&mut innovation);

        // The analytic attitude columns differentiate w.r.t. Euler angles;
        // the error state needs derivatives w.r.t. the body-frame rotation
        // vector, so overwrite columns 6..8 with the finite-difference form
        // (see `attitude_error_jacobian`, #286). Measurements whose analytic
        // attitude block is identically zero (e.g. GPS, baro) are
        // attitude-independent, so the finite differences would be zero too
        // and are skipped to keep them out of the hot loop.
        let analytic_attitude_block_zero = h_error
            .view((0, 6), (meas_dim, 3))
            .iter()
            .all(|v| *v == 0.0);
        if !analytic_attitude_block_zero {
            let h_att = Self::attitude_error_jacobian(
                measurement,
                &self.nominal_quaternion,
                &nominal_state_vec,
            );
            h_error.view_mut((0, 6), (meas_dim, 3)).copy_from(&h_att);
        }

        // Innovation covariance: S = H * P * H^T + R
        let s = &h_error * &self.error_covariance * h_error.transpose() + measurement.get_noise();

        // Gate before injecting anything. The ESKF makes this ordering load-bearing
        // rather than merely tidy: `inject_error_state` mutates the nominal state and
        // zeroes the error state, so there is no "undo" once the correction starts.
        let decision = self.gate_policy.evaluate(&innovation, &s, "ESKF")?;
        let outcome = decision.outcome;
        if !outcome.accepted {
            // Inflate the error covariance on the way out, in the directions this
            // measurement observed. The nominal state is untouched, so this is the filter
            // recording that it is further from that nominal than it thought -- which is
            // what re-opens the gate (#340) -- and confining it to `h_error`'s row space
            // keeps a rejection on one sensor out of every other sensor's gain.
            let projected = &self.error_covariance * h_error.transpose();
            let observed = &h_error * &projected;
            decision.inflate_observed(&mut self.error_covariance, &projected, &observed);
            return Ok(outcome);
        }

        // Kalman gain: K = P * H^T * S^(-1)
        let k = self.error_covariance.clone()
            * h_error.transpose()
            * robust_spd_solve(&symmetrize(&s), &DMatrix::identity(s.nrows(), s.ncols()))?
                .transpose();

        // Error state update: δx = K * nu
        self.error_state = &k * &innovation;

        // Inject error state into nominal state and reset
        self.inject_error_state();

        // Error covariance update (Joseph form for numerical stability):
        // P = (I - K*H)*P*(I - K*H)^T + K*R*K^T
        let i_kh = DMatrix::identity(15, 15) - &k * &h_error;
        let r = measurement.get_noise();
        self.error_covariance =
            &i_kh * &self.error_covariance * i_kh.transpose() + &k * r * k.transpose();

        self.regularize_covariance();
        Ok(outcome)
    }

    fn set_innovation_gate(&mut self, gate: Option<InnovationGate>) -> bool {
        self.gate_policy.set_gate(gate);
        true
    }

    fn set_gate_recovery(&mut self, recovery: GateRecovery) -> bool {
        self.gate_policy.set_recovery(recovery);
        true
    }

    /// Get the current nominal state estimate
    ///
    /// Returns the nominal state vector in the same format as EKF/UKF for compatibility:
    /// [lat (rad), lon (rad), alt (m), `v_n`, `v_e`, `v_d`, roll, pitch, yaw, `b_ax`, `b_ay`, `b_az`, `b_gx`, `b_gy`, `b_gz`]
    ///
    /// Note: The internal representation uses quaternions, but this converts to Euler angles
    /// on `Rotation3::euler_angles`'s principal branch: roll and yaw on -pi..pi, pitch on
    /// -pi/2..pi/2.
    fn get_estimate(&self) -> DVector<f64> {
        let mut state = DVector::zeros(15);

        // Position
        state[0] = self.nominal_latitude;
        state[1] = self.nominal_longitude;
        state[2] = self.nominal_altitude;

        // Velocity
        state[3] = self.nominal_velocity_north;
        state[4] = self.nominal_velocity_east;
        state[5] = self.nominal_velocity_vertical;

        // Attitude (convert quaternion to Euler angles)
        let quat = nalgebra::UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(
            self.nominal_quaternion[0],
            self.nominal_quaternion[1],
            self.nominal_quaternion[2],
            self.nominal_quaternion[3],
        ));
        // `euler_angles` already returns the principal branch -- roll and yaw from `atan2`
        // on -pi..pi, pitch from `asin` on -pi/2..pi/2 -- so there is nothing to wrap. The
        // `wrap_to_2pi` that used to sit here only moved a correct answer onto 0..2*pi,
        // reporting a level vehicle's roll as 359.99 degrees (#314), and disagreed with the
        // deliberately unwrapped linearization point this same filter builds in `update`.
        let euler = quat.euler_angles();
        state[6] = euler.0; // roll
        state[7] = euler.1; // pitch
        state[8] = euler.2; // yaw

        // Biases
        state[9] = self.nominal_accel_bias[0];
        state[10] = self.nominal_accel_bias[1];
        state[11] = self.nominal_accel_bias[2];
        state[12] = self.nominal_gyro_bias[0];
        state[13] = self.nominal_gyro_bias[1];
        state[14] = self.nominal_gyro_bias[2];

        state
    }

    /// Get the current error state uncertainty (covariance matrix)
    ///
    /// Returns the error covariance matrix P (15x15) representing uncertainty
    /// in the error states (not the nominal states)
    fn get_certainty(&self) -> DMatrix<f64> {
        self.error_covariance.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::earth;
    use crate::measurements::{
        GPSPositionAndVelocityMeasurement, GPSPositionMeasurement, GPSVelocityMeasurement,
        RelativeAltitudeMeasurement,
    };
    use assert_approx_eq::assert_approx_eq;
    use nalgebra::Vector3;

    const IMU_BIASES: [f64; 6] = [0.0; 6];
    const N: usize = 15;
    const COVARIANCE_DIAGONAL: [f64; N] = [1e-9; N];
    const PROCESS_NOISE_DIAGONAL: [f64; N] = [1e-9; N];

    const ALPHA: f64 = 1e-3;
    const BETA: f64 = 2.0;
    const KAPPA: f64 = 0.0;
    const UKF_PARAMS: InitialState = InitialState {
        latitude: 0.0,
        longitude: 0.0,
        altitude: 0.0,
        northward_velocity: 0.0,
        eastward_velocity: 0.0,
        vertical_velocity: 0.0,
        roll: 0.0,
        pitch: 0.0,
        yaw: 0.0,
        in_degrees: false,
        is_enu: true,
    };

    #[test]
    fn ukf_construction() {
        let measurement_bias = vec![0.0; 3]; // Example measurement bias
        let ukf = UnscentedKalmanFilter::new(
            &UKF_PARAMS,
            &IMU_BIASES,
            Some(&measurement_bias),
            vec![1e-3; 18],
            DMatrix::from_diagonal(&DVector::from_vec(vec![1e-3; 18])),
            ALPHA,
            BETA,
            KAPPA,
        );
        assert_eq!(ukf.mean_state.len(), 18);
        let wms = ukf.weights_mean;
        let wcs = ukf.weights_cov;
        assert_eq!(wms.len(), (2 * ukf.state_size) + 1);
        assert_eq!(wcs.len(), (2 * ukf.state_size) + 1);
        // Check that the weights are correct
        let lambda = ALPHA.powi(2) * (18.0 + KAPPA) - 18.0;
        assert_eq!(lambda, ukf.lambda);
        let wm_0 = lambda / (18.0 + lambda);
        let wc_0 = wm_0 + (1.0 - ALPHA.powi(2)) + BETA;
        let w_i = 1.0 / (2.0 * (18.0 + lambda));
        assert_approx_eq!(wms[0], wm_0, 1e-6);
        assert_approx_eq!(wcs[0], wc_0, 1e-6);
        for i in 1..wms.len() {
            assert_approx_eq!(wms[i], w_i, 1e-6);
            assert_approx_eq!(wcs[i], w_i, 1e-6);
        }
    }

    #[test]
    fn ukf_get_sigma_points() {
        let ukf = UnscentedKalmanFilter::new(
            &UKF_PARAMS,
            &IMU_BIASES,
            None,
            COVARIANCE_DIAGONAL.to_vec(),
            DMatrix::from_diagonal(&DVector::from_vec(PROCESS_NOISE_DIAGONAL.to_vec())),
            ALPHA,
            BETA,
            KAPPA,
        );
        let sigma_points = ukf.get_sigma_points().unwrap();
        assert_eq!(sigma_points.ncols(), (2 * ukf.state_size) + 1);

        let mu = ukf.get_sigma_points().unwrap() * ukf.weights_mean;
        assert_eq!(mu.nrows(), ukf.state_size);
        assert_eq!(mu.ncols(), 1);
        assert_approx_eq!(mu[0], 0.0, 1e-6);
        assert_approx_eq!(mu[1], 0.0, 1e-6);
        assert_approx_eq!(mu[2], 0.0, 1e-6);
        assert_approx_eq!(mu[3], 0.0, 1e-6);
        assert_approx_eq!(mu[4], 0.0, 1e-6);
        assert_approx_eq!(mu[5], 0.0, 1e-6);
        assert_approx_eq!(mu[6], 0.0, 1e-6);
        assert_approx_eq!(mu[7], 0.0, 1e-6);
        assert_approx_eq!(mu[8], 0.0, 1e-6);
    }

    #[test]
    fn ukf_propagate() {
        let mut ukf = UnscentedKalmanFilter::new(
            &UKF_PARAMS,
            &[0.0; 6],
            None,         //Some(measurement_bias.clone()),
            vec![0.0; N], // Absolute certainty use for testing the process
            DMatrix::from_diagonal(&DVector::from_vec(PROCESS_NOISE_DIAGONAL.to_vec())),
            1e-3,
            2.0,
            0.0,
        );
        let dt = 1.0;
        let imu_data = IMUData {
            accel: Vector3::new(0.0, 0.0, earth::gravity(&0.0, &0.0)),
            gyro: Vector3::new(0.0, 0.0, 0.0), // No rotation
        };
        ukf.predict(&imu_data, dt).unwrap();
        assert_eq!(ukf.mean_state.len(), 15);
        let measurement = GPSPositionMeasurement {
            latitude: 0.0,
            longitude: 0.0,
            altitude: 0.0,
            horizontal_noise_std: 1e-3,
            vertical_noise_std: 1e-3,
        };
        ukf.update(&measurement).unwrap();
        // Check that the state has not changed
        assert_approx_eq!(ukf.mean_state[0], 0.0, 1e-3);
        assert_approx_eq!(ukf.mean_state[1], 0.0, 1e-3);
        assert_approx_eq!(ukf.mean_state[2], 0.0, 0.1);
        assert_approx_eq!(ukf.mean_state[3], 0.0, 0.1);
        assert_approx_eq!(ukf.mean_state[4], 0.0, 0.1);
        assert_approx_eq!(ukf.mean_state[5], 0.0, 0.1);
    }

    #[test]
    fn ukf_debug_display() {
        // Test Debug and Display implementations for UKF
        let ukf = UnscentedKalmanFilter::new(
            &UKF_PARAMS,
            &IMU_BIASES,
            None,
            COVARIANCE_DIAGONAL.to_vec(),
            DMatrix::from_diagonal(&DVector::from_vec(PROCESS_NOISE_DIAGONAL.to_vec())),
            ALPHA,
            BETA,
            KAPPA,
        );

        // Test Debug
        let debug_str = format!("{ukf:?}");
        assert!(debug_str.contains("UKF"));
        assert!(debug_str.contains("mean_state"));

        // Test Display
        let display_str = format!("{ukf}");
        assert!(display_str.contains("UnscentedKalmanFilter"));
        assert!(display_str.contains("covariance"));
    }

    #[test]
    fn ukf_predict_with_biases() {
        // Test UKF predict with non-zero biases
        let mut ukf = UnscentedKalmanFilter::new(
            &UKF_PARAMS,
            &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], // non-zero biases
            None,
            COVARIANCE_DIAGONAL.to_vec(),
            DMatrix::from_diagonal(&DVector::from_vec(PROCESS_NOISE_DIAGONAL.to_vec())),
            ALPHA,
            BETA,
            KAPPA,
        );

        let imu_data = IMUData {
            accel: Vector3::new(0.0, 0.0, -9.81),
            gyro: Vector3::new(0.0, 0.0, 0.0),
        };

        ukf.predict(&imu_data, 0.1).unwrap();

        // Just verify prediction completed without panic
        assert_eq!(ukf.mean_state.len(), 15);
    }

    #[test]
    fn ukf_update_with_cross_covariance() {
        // Test UKF update to cover cross-covariance calculation
        let mut ukf = UnscentedKalmanFilter::new(
            &UKF_PARAMS,
            &IMU_BIASES,
            None,
            COVARIANCE_DIAGONAL.to_vec(),
            DMatrix::from_diagonal(&DVector::from_vec(PROCESS_NOISE_DIAGONAL.to_vec())),
            ALPHA,
            BETA,
            KAPPA,
        );

        // First predict to move state
        let imu_data = IMUData {
            accel: Vector3::new(0.0, 0.0, -9.81),
            gyro: Vector3::new(0.0, 0.0, 0.0),
        };
        ukf.predict(&imu_data, 0.1).unwrap();

        // Update with GPS position measurement
        let measurement = GPSPositionMeasurement {
            latitude: 0.001,
            longitude: 0.001,
            altitude: 10.0,
            horizontal_noise_std: 5.0,
            vertical_noise_std: 2.0,
        };

        ukf.update(&measurement).unwrap();

        // Verify update completed
        assert!(!ukf.mean_state.is_empty());
    }

    #[test]
    fn ukf_with_additional_states() {
        // Test UKF construction with additional states beyond 15
        let measurement_bias = vec![1.0, 2.0, 3.0];
        let total_states = 15 + measurement_bias.len();

        let ukf = UnscentedKalmanFilter::new(
            &UKF_PARAMS,
            &IMU_BIASES,
            Some(&measurement_bias),
            vec![1e-6; total_states],
            DMatrix::from_diagonal(&DVector::from_vec(vec![1e-9; total_states])),
            ALPHA,
            BETA,
            KAPPA,
        );

        assert_eq!(ukf.state_size, total_states);
        assert_eq!(ukf.mean_state.len(), total_states);
    }

    #[test]
    fn ukf_with_velocity_measurement() {
        // Test UKF with velocity measurement
        let mut ukf = UnscentedKalmanFilter::new(
            &UKF_PARAMS,
            &IMU_BIASES,
            None,
            COVARIANCE_DIAGONAL.to_vec(),
            DMatrix::from_diagonal(&DVector::from_vec(PROCESS_NOISE_DIAGONAL.to_vec())),
            ALPHA,
            BETA,
            KAPPA,
        );

        let vel_meas = GPSVelocityMeasurement {
            northward_velocity: 0.0,
            eastward_velocity: 0.0,
            vertical_velocity: 0.0,
            horizontal_noise_std: 0.5,
            vertical_noise_std: 0.5,
        };

        ukf.update(&vel_meas).unwrap();

        // Verify update completed
        assert_eq!(ukf.mean_state.len(), 15);
    }

    #[test]
    fn ukf_with_position_velocity_measurement() {
        // Test UKF with combined position and velocity measurement
        let mut ukf = UnscentedKalmanFilter::new(
            &UKF_PARAMS,
            &IMU_BIASES,
            None,
            COVARIANCE_DIAGONAL.to_vec(),
            DMatrix::from_diagonal(&DVector::from_vec(PROCESS_NOISE_DIAGONAL.to_vec())),
            ALPHA,
            BETA,
            KAPPA,
        );

        let meas = GPSPositionAndVelocityMeasurement {
            latitude: 0.0,
            longitude: 0.0,
            altitude: 0.0,
            northward_velocity: 0.0,
            eastward_velocity: 0.0,
            horizontal_noise_std: 5.0,
            vertical_noise_std: 2.0,
            velocity_noise_std: 0.5,
        };

        ukf.update(&meas).unwrap();

        // Verify update completed
        assert_eq!(ukf.mean_state.len(), 15);
    }

    #[test]
    fn ukf_with_altitude_measurement() {
        // Test UKF with relative altitude measurement
        let initial_state = InitialState {
            latitude: 0.0,
            longitude: 0.0,
            altitude: 100.0,
            northward_velocity: 0.0,
            eastward_velocity: 0.0,
            vertical_velocity: 0.0,
            roll: 0.0,
            pitch: 0.0,
            yaw: 0.0,
            in_degrees: false,
            is_enu: true,
        };

        let mut ukf = UnscentedKalmanFilter::new(
            &initial_state,
            &IMU_BIASES,
            None,
            COVARIANCE_DIAGONAL.to_vec(),
            DMatrix::from_diagonal(&DVector::from_vec(PROCESS_NOISE_DIAGONAL.to_vec())),
            ALPHA,
            BETA,
            KAPPA,
        );

        let alt_meas = RelativeAltitudeMeasurement {
            relative_altitude: 5.0,
            reference_altitude: 95.0,
        };

        ukf.update(&alt_meas).unwrap();

        // Should pull altitude toward 100m
        assert!(ukf.mean_state[2] > 90.0 && ukf.mean_state[2] < 110.0);
    }

    #[test]
    fn ukf_free_fall_motion() {
        // Test UKF with free fall motion profile
        let initial_state = InitialState {
            latitude: 0.0,
            longitude: 0.0,
            altitude: 100.0,
            northward_velocity: 0.0,
            eastward_velocity: 0.0,
            vertical_velocity: 0.0,
            roll: 0.0,
            pitch: 0.0,
            yaw: 0.0,
            in_degrees: false,
            is_enu: true,
        };

        let mut ukf = UnscentedKalmanFilter::new(
            &initial_state,
            &IMU_BIASES,
            None,
            vec![
                1e-6, 1e-6, 1.0, 0.1, 0.1, 0.1, 1e-4, 1e-4, 1e-4, 1e-6, 1e-6, 1e-6, 1e-8, 1e-8,
                1e-8,
            ],
            DMatrix::from_diagonal(&DVector::from_vec(vec![
                1e-9, 1e-9, 1e-6, 1e-6, 1e-6, 1e-6, 1e-9, 1e-9, 1e-9, 1e-9, 1e-9, 1e-9, 1e-9, 1e-9,
                1e-9,
            ])),
            ALPHA,
            BETA,
            KAPPA,
        );

        let dt = 0.1;
        let num_steps = 10;

        // Simulate free fall with only gravity (no vertical acceleration resistance)
        for _ in 0..num_steps {
            let imu_data = IMUData {
                accel: Vector3::new(0.0, 0.0, 0.0), // Free fall - no measured acceleration
                gyro: Vector3::new(0.0, 0.0, 0.0),
            };
            ukf.predict(&imu_data, dt).unwrap();
        }

        // After 1 second of free fall, should have accumulated vertical velocity
        // v = g*t = 9.81 * 1.0 = 9.81 m/s (downward is negative in ENU)
        let final_vd = ukf.mean_state[5];
        assert!(
            final_vd < -5.0,
            "Expected significant vertical velocity, got {final_vd}"
        );

        // Altitude should have decreased
        let final_altitude = ukf.mean_state[2];
        assert!(
            final_altitude < 100.0,
            "Expected altitude decrease, got {final_altitude}"
        );

        // Apply measurement update with GPS position
        let measurement = GPSPositionMeasurement {
            latitude: 0.0,
            longitude: 0.0,
            altitude: final_altitude,
            horizontal_noise_std: 5.0,
            vertical_noise_std: 2.0,
        };
        ukf.update(&measurement).unwrap();

        // After measurement update, estimate should remain close to measurement
        assert_approx_eq!(ukf.mean_state[2], final_altitude, 5.0);
    }

    #[test]
    fn ukf_hover_motion() {
        // Test UKF with hover (stationary vertical) motion profile
        let initial_state = InitialState {
            latitude: 0.0,
            longitude: 0.0,
            altitude: 100.0,
            northward_velocity: 0.0,
            eastward_velocity: 0.0,
            vertical_velocity: 0.0,
            roll: 0.0,
            pitch: 0.0,
            yaw: 0.0,
            in_degrees: false,
            is_enu: true,
        };

        let mut ukf = UnscentedKalmanFilter::new(
            &initial_state,
            &IMU_BIASES,
            None,
            vec![
                1e-6, 1e-6, 1.0, 0.1, 0.1, 0.1, 1e-4, 1e-4, 1e-4, 1e-6, 1e-6, 1e-6, 1e-8, 1e-8,
                1e-8,
            ],
            DMatrix::from_diagonal(&DVector::from_vec(vec![
                1e-9, 1e-9, 1e-6, 1e-6, 1e-6, 1e-6, 1e-9, 1e-9, 1e-9, 1e-9, 1e-9, 1e-9, 1e-9, 1e-9,
                1e-9,
            ])),
            ALPHA,
            BETA,
            KAPPA,
        );

        let dt = 0.1;
        let num_steps = 10;

        // Simulate hover with upward acceleration exactly canceling gravity
        for _ in 0..num_steps {
            let imu_data = IMUData {
                accel: Vector3::new(0.0, 0.0, earth::gravity(&0.0, &0.0)),
                gyro: Vector3::new(0.0, 0.0, 0.0),
            };
            ukf.predict(&imu_data, dt).unwrap();
        }

        // Velocity should remain near zero
        let final_vn = ukf.mean_state[3];
        let final_ve = ukf.mean_state[4];
        let final_vd = ukf.mean_state[5];
        assert_approx_eq!(final_vn, 0.0, 0.5);
        assert_approx_eq!(final_ve, 0.0, 0.5);
        assert_approx_eq!(final_vd, 0.0, 0.5);

        // Altitude should remain approximately constant
        let final_altitude = ukf.mean_state[2];
        assert_approx_eq!(final_altitude, 100.0, 1.0);

        // Apply GPS velocity measurement to verify zero velocity state
        let vel_measurement = GPSVelocityMeasurement {
            northward_velocity: 0.0,
            eastward_velocity: 0.0,
            vertical_velocity: 0.0,
            horizontal_noise_std: 0.5,
            vertical_noise_std: 0.5,
        };
        ukf.update(&vel_measurement).unwrap();

        // After update, velocities should remain near zero
        assert_approx_eq!(ukf.mean_state[3], 0.0, 0.5);
        assert_approx_eq!(ukf.mean_state[4], 0.0, 0.5);
        assert_approx_eq!(ukf.mean_state[5], 0.0, 0.5);
    }

    #[test]
    fn ukf_northward_motion() {
        // Test UKF with constant northward velocity motion profile
        let initial_state = InitialState {
            latitude: 0.0,
            longitude: 0.0,
            altitude: 100.0,
            northward_velocity: 10.0, // 10 m/s northward
            eastward_velocity: 0.0,
            vertical_velocity: 0.0,
            roll: 0.0,
            pitch: 0.0,
            yaw: 0.0,
            in_degrees: false,
            is_enu: true,
        };

        let mut ukf = UnscentedKalmanFilter::new(
            &initial_state,
            &IMU_BIASES,
            None,
            vec![
                1e-6, 1e-6, 1.0, 0.1, 0.1, 0.1, 1e-4, 1e-4, 1e-4, 1e-6, 1e-6, 1e-6, 1e-8, 1e-8,
                1e-8,
            ],
            DMatrix::from_diagonal(&DVector::from_vec(vec![
                1e-9, 1e-9, 1e-6, 1e-6, 1e-6, 1e-6, 1e-9, 1e-9, 1e-9, 1e-9, 1e-9, 1e-9, 1e-9, 1e-9,
                1e-9,
            ])),
            ALPHA,
            BETA,
            KAPPA,
        );

        let dt = 0.1;
        let num_steps = 10;
        let initial_lat = ukf.mean_state[0];

        // Simulate constant northward motion with gravity compensation
        for _ in 0..num_steps {
            let imu_data = IMUData {
                accel: Vector3::new(0.0, 0.0, earth::gravity(&0.0, &0.0)),
                gyro: Vector3::new(0.0, 0.0, 0.0),
            };
            ukf.predict(&imu_data, dt).unwrap();
        }

        // Latitude should have increased (moving north)
        let final_lat = ukf.mean_state[0];
        assert!(
            final_lat > initial_lat,
            "Expected latitude increase, got initial: {initial_lat} final: {final_lat}"
        );

        // Northward velocity should remain approximately constant
        let final_vn = ukf.mean_state[3];
        assert_approx_eq!(final_vn, 10.0, 2.0);

        // Eastward velocity should remain near zero
        let final_ve = ukf.mean_state[4];
        assert_approx_eq!(final_ve, 0.0, 0.5);

        // Apply GPS position and velocity measurement
        let meas = GPSPositionAndVelocityMeasurement {
            latitude: final_lat.to_degrees(),
            longitude: 0.0,
            altitude: 100.0,
            northward_velocity: 10.0,
            eastward_velocity: 0.0,
            horizontal_noise_std: 5.0,
            vertical_noise_std: 2.0,
            velocity_noise_std: 0.5,
        };
        ukf.update(&meas).unwrap();

        // After measurement, velocities should be close to measured values
        assert_approx_eq!(ukf.mean_state[3], 10.0, 1.0);
        assert_approx_eq!(ukf.mean_state[4], 0.0, 0.5);
    }

    #[test]
    fn ukf_eastward_motion() {
        // Test UKF with constant eastward velocity motion profile
        let initial_state = InitialState {
            latitude: 0.0,
            longitude: 0.0,
            altitude: 100.0,
            northward_velocity: 0.0,
            eastward_velocity: 15.0, // 15 m/s eastward
            vertical_velocity: 0.0,
            roll: 0.0,
            pitch: 0.0,
            yaw: 0.0,
            in_degrees: false,
            is_enu: true,
        };

        let mut ukf = UnscentedKalmanFilter::new(
            &initial_state,
            &IMU_BIASES,
            None,
            vec![
                1e-6, 1e-6, 1.0, 0.1, 0.1, 0.1, 1e-4, 1e-4, 1e-4, 1e-6, 1e-6, 1e-6, 1e-8, 1e-8,
                1e-8,
            ],
            DMatrix::from_diagonal(&DVector::from_vec(vec![
                1e-9, 1e-9, 1e-6, 1e-6, 1e-6, 1e-6, 1e-9, 1e-9, 1e-9, 1e-9, 1e-9, 1e-9, 1e-9, 1e-9,
                1e-9,
            ])),
            ALPHA,
            BETA,
            KAPPA,
        );

        let dt = 0.1;
        let num_steps = 10;
        let initial_lon = ukf.mean_state[1];

        // Simulate constant eastward motion with gravity compensation
        for _ in 0..num_steps {
            let imu_data = IMUData {
                accel: Vector3::new(0.0, 0.0, earth::gravity(&0.0, &0.0)),
                gyro: Vector3::new(0.0, 0.0, 0.0),
            };
            ukf.predict(&imu_data, dt).unwrap();
        }

        // Longitude should have increased (moving east)
        let final_lon = ukf.mean_state[1];
        assert!(
            final_lon > initial_lon,
            "Expected longitude increase, got initial: {initial_lon} final: {final_lon}"
        );

        // Eastward velocity should remain approximately constant
        let final_ve = ukf.mean_state[4];
        assert_approx_eq!(final_ve, 15.0, 2.0);

        // Northward velocity should remain near zero
        let final_vn = ukf.mean_state[3];
        assert_approx_eq!(final_vn, 0.0, 0.5);

        // Vertical velocity should remain near zero
        let final_vd = ukf.mean_state[5];
        assert_approx_eq!(final_vd, 0.0, 0.5);

        // Apply GPS position measurement
        let pos_meas = GPSPositionMeasurement {
            latitude: 0.0,
            longitude: final_lon.to_degrees(),
            altitude: 100.0,
            horizontal_noise_std: 5.0,
            vertical_noise_std: 2.0,
        };
        ukf.update(&pos_meas).unwrap();

        // Position should remain close to measurement
        assert_approx_eq!(ukf.mean_state[1], final_lon, 0.01);
        assert_approx_eq!(ukf.mean_state[2], 100.0, 5.0);

        // Apply velocity measurement
        let vel_meas = GPSVelocityMeasurement {
            northward_velocity: 0.0,
            eastward_velocity: 15.0,
            vertical_velocity: 0.0,
            horizontal_noise_std: 0.5,
            vertical_noise_std: 0.5,
        };
        ukf.update(&vel_meas).unwrap();

        // After measurement, velocities should be close to measured values
        assert_approx_eq!(ukf.mean_state[3], 0.0, 0.5);
        assert_approx_eq!(ukf.mean_state[4], 15.0, 1.0);
        assert_approx_eq!(ukf.mean_state[5], 0.0, 0.5);
    }

    #[test]
    fn ukf_combined_horizontal_motion() {
        // Test UKF with combined northward and eastward motion
        let initial_state = InitialState {
            latitude: 0.0,
            longitude: 0.0,
            altitude: 100.0,
            northward_velocity: 10.0,
            eastward_velocity: 10.0,
            vertical_velocity: 0.0,
            roll: 0.0,
            pitch: 0.0,
            yaw: 0.0,
            in_degrees: false,
            is_enu: true,
        };

        let mut ukf = UnscentedKalmanFilter::new(
            &initial_state,
            &IMU_BIASES,
            None,
            vec![
                1e-6, 1e-6, 1.0, 0.1, 0.1, 0.1, 1e-4, 1e-4, 1e-4, 1e-6, 1e-6, 1e-6, 1e-8, 1e-8,
                1e-8,
            ],
            DMatrix::from_diagonal(&DVector::from_vec(vec![
                1e-9, 1e-9, 1e-6, 1e-6, 1e-6, 1e-6, 1e-9, 1e-9, 1e-9, 1e-9, 1e-9, 1e-9, 1e-9, 1e-9,
                1e-9,
            ])),
            ALPHA,
            BETA,
            KAPPA,
        );

        let dt = 0.1;
        let num_steps = 10;
        let initial_lat = ukf.mean_state[0];
        let initial_lon = ukf.mean_state[1];

        // Simulate combined motion
        for _ in 0..num_steps {
            let imu_data = IMUData {
                accel: Vector3::new(0.0, 0.0, earth::gravity(&0.0, &0.0)),
                gyro: Vector3::new(0.0, 0.0, 0.0),
            };
            ukf.predict(&imu_data, dt).unwrap();
        }

        // Both latitude and longitude should have increased
        let final_lat = ukf.mean_state[0];
        let final_lon = ukf.mean_state[1];
        assert!(final_lat > initial_lat, "Expected latitude increase");
        assert!(final_lon > initial_lon, "Expected longitude increase");

        // Both velocities should remain approximately constant
        assert_approx_eq!(ukf.mean_state[3], 10.0, 2.0);
        assert_approx_eq!(ukf.mean_state[4], 10.0, 2.0);

        // Apply combined position and velocity measurement
        let meas = GPSPositionAndVelocityMeasurement {
            latitude: final_lat.to_degrees(),
            longitude: final_lon.to_degrees(),
            altitude: 100.0,
            northward_velocity: 10.0,
            eastward_velocity: 10.0,
            horizontal_noise_std: 5.0,
            vertical_noise_std: 2.0,
            velocity_noise_std: 0.5,
        };
        ukf.update(&meas).unwrap();

        // After measurement, state should be well-constrained
        assert_approx_eq!(ukf.mean_state[3], 10.0, 1.0);
        assert_approx_eq!(ukf.mean_state[4], 10.0, 1.0);
        assert_approx_eq!(ukf.mean_state[2], 100.0, 5.0);
    }

    // ==================== Extended Kalman Filter Tests ====================

    #[test]
    fn ekf_construction_9state() {
        // Test EKF construction with 9-state configuration (no biases)
        let ekf = ExtendedKalmanFilter::new(
            &UKF_PARAMS,
            &[0.0; 6], // Biases provided but won't be used
            vec![1e-3; 9],
            DMatrix::from_diagonal(&DVector::from_vec(vec![1e-3; 9])),
            false, // Don't use biases
        );
        assert_eq!(ekf.mean_state.len(), 9);
        assert_eq!(ekf.state_size, 9);
        assert!(!ekf.use_biases);
    }

    #[test]
    fn ekf_construction_15state() {
        // Test EKF construction with 15-state configuration (with biases)
        let ekf = ExtendedKalmanFilter::new(
            &UKF_PARAMS,
            &IMU_BIASES,
            COVARIANCE_DIAGONAL.to_vec(),
            DMatrix::from_diagonal(&DVector::from_vec(PROCESS_NOISE_DIAGONAL.to_vec())),
            true, // Use biases
        );
        assert_eq!(ekf.mean_state.len(), 15);
        assert_eq!(ekf.state_size, 15);
        assert!(ekf.use_biases);
    }

    #[test]
    fn ekf_debug_display() {
        // Test Debug and Display implementations for EKF
        let ekf = ExtendedKalmanFilter::new(
            &UKF_PARAMS,
            &IMU_BIASES,
            COVARIANCE_DIAGONAL.to_vec(),
            DMatrix::from_diagonal(&DVector::from_vec(PROCESS_NOISE_DIAGONAL.to_vec())),
            true,
        );

        // Test Debug
        let debug_str = format!("{ekf:?}");
        assert!(debug_str.contains("EKF"));
        assert!(debug_str.contains("mean_state"));

        // Test Display
        let display_str = format!("{ekf}");
        assert!(display_str.contains("ExtendedKalmanFilter"));
        assert!(display_str.contains("covariance"));
    }

    #[test]
    fn ekf_propagate_9state() {
        // Test EKF predict without biases
        let mut ekf = ExtendedKalmanFilter::new(
            &UKF_PARAMS,
            &[0.0; 6],
            vec![0.0; 9], // Absolute certainty for testing
            DMatrix::from_diagonal(&DVector::from_vec(vec![1e-9; 9])),
            false, // 9-state
        );
        let dt = 1.0;
        let imu_data = IMUData {
            accel: Vector3::new(0.0, 0.0, earth::gravity(&0.0, &0.0)),
            gyro: Vector3::new(0.0, 0.0, 0.0),
        };
        ekf.predict(&imu_data, dt).unwrap();
        assert_eq!(ekf.mean_state.len(), 9);

        // Test GPS position measurement update
        let measurement = GPSPositionMeasurement {
            latitude: 0.0,
            longitude: 0.0,
            altitude: 0.0,
            horizontal_noise_std: 1e-3,
            vertical_noise_std: 1e-3,
        };
        ekf.update(&measurement).unwrap();

        // Check that the state has not changed significantly
        assert_approx_eq!(ekf.mean_state[0], 0.0, 1e-3);
        assert_approx_eq!(ekf.mean_state[1], 0.0, 1e-3);
        assert_approx_eq!(ekf.mean_state[2], 0.0, 0.1);
        assert_approx_eq!(ekf.mean_state[3], 0.0, 0.1);
        assert_approx_eq!(ekf.mean_state[4], 0.0, 0.1);
        assert_approx_eq!(ekf.mean_state[5], 0.0, 0.1);
    }

    #[test]
    fn ekf_propagate_15state() {
        // Test EKF predict with biases
        let mut ekf = ExtendedKalmanFilter::new(
            &UKF_PARAMS,
            &[0.0; 6],
            vec![0.0; 15], // Absolute certainty for testing
            DMatrix::from_diagonal(&DVector::from_vec(vec![1e-9; 15])),
            true, // 15-state with biases
        );
        let dt = 1.0;
        let imu_data = IMUData {
            accel: Vector3::new(0.0, 0.0, earth::gravity(&0.0, &0.0)),
            gyro: Vector3::new(0.0, 0.0, 0.0),
        };
        ekf.predict(&imu_data, dt).unwrap();
        assert_eq!(ekf.mean_state.len(), 15);

        // Test GPS position measurement update
        let measurement = GPSPositionMeasurement {
            latitude: 0.0,
            longitude: 0.0,
            altitude: 0.0,
            horizontal_noise_std: 1e-3,
            vertical_noise_std: 1e-3,
        };
        ekf.update(&measurement).unwrap();

        // Check that the state has not changed significantly
        assert_approx_eq!(ekf.mean_state[0], 0.0, 1e-3);
        assert_approx_eq!(ekf.mean_state[1], 0.0, 1e-3);
        assert_approx_eq!(ekf.mean_state[2], 0.0, 0.1);
    }

    #[test]
    fn ekf_predict_with_nonzero_biases() {
        // Test EKF predict with non-zero biases
        let mut ekf = ExtendedKalmanFilter::new(
            &UKF_PARAMS,
            &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], // non-zero biases
            COVARIANCE_DIAGONAL.to_vec(),
            DMatrix::from_diagonal(&DVector::from_vec(PROCESS_NOISE_DIAGONAL.to_vec())),
            true,
        );

        let imu_data = IMUData {
            accel: Vector3::new(0.0, 0.0, -9.81),
            gyro: Vector3::new(0.0, 0.0, 0.0),
        };

        ekf.predict(&imu_data, 0.1).unwrap();

        // Just verify prediction completed without panic
        assert_eq!(ekf.mean_state.len(), 15);
        // Biases should remain unchanged (random walk model)
        assert_approx_eq!(ekf.mean_state[9], 0.1, 1e-6);
        assert_approx_eq!(ekf.mean_state[12], 0.4, 1e-6);
    }

    #[test]
    fn ekf_with_velocity_measurement() {
        // Test EKF with velocity measurement
        let mut ekf = ExtendedKalmanFilter::new(
            &UKF_PARAMS,
            &IMU_BIASES,
            COVARIANCE_DIAGONAL.to_vec(),
            DMatrix::from_diagonal(&DVector::from_vec(PROCESS_NOISE_DIAGONAL.to_vec())),
            true,
        );

        let vel_meas = GPSVelocityMeasurement {
            northward_velocity: 0.0,
            eastward_velocity: 0.0,
            vertical_velocity: 0.0,
            horizontal_noise_std: 0.5,
            vertical_noise_std: 0.5,
        };

        ekf.update(&vel_meas).unwrap();

        // Verify update completed
        assert_eq!(ekf.mean_state.len(), 15);
    }

    #[test]
    fn ekf_with_position_velocity_measurement() {
        // Test EKF with combined position and velocity measurement
        let mut ekf = ExtendedKalmanFilter::new(
            &UKF_PARAMS,
            &IMU_BIASES,
            COVARIANCE_DIAGONAL.to_vec(),
            DMatrix::from_diagonal(&DVector::from_vec(PROCESS_NOISE_DIAGONAL.to_vec())),
            true,
        );

        let meas = GPSPositionAndVelocityMeasurement {
            latitude: 0.0,
            longitude: 0.0,
            altitude: 0.0,
            northward_velocity: 0.0,
            eastward_velocity: 0.0,
            horizontal_noise_std: 5.0,
            vertical_noise_std: 2.0,
            velocity_noise_std: 0.5,
        };

        ekf.update(&meas).unwrap();

        // Verify update completed
        assert_eq!(ekf.mean_state.len(), 15);
    }

    #[test]
    fn ekf_with_altitude_measurement() {
        // Test EKF with relative altitude measurement
        let initial_state = InitialState {
            latitude: 0.0,
            longitude: 0.0,
            altitude: 100.0,
            northward_velocity: 0.0,
            eastward_velocity: 0.0,
            vertical_velocity: 0.0,
            roll: 0.0,
            pitch: 0.0,
            yaw: 0.0,
            in_degrees: false,
            is_enu: true,
        };

        let mut ekf = ExtendedKalmanFilter::new(
            &initial_state,
            &IMU_BIASES,
            COVARIANCE_DIAGONAL.to_vec(),
            DMatrix::from_diagonal(&DVector::from_vec(PROCESS_NOISE_DIAGONAL.to_vec())),
            true,
        );

        let alt_meas = RelativeAltitudeMeasurement {
            relative_altitude: 5.0,
            reference_altitude: 95.0,
        };

        ekf.update(&alt_meas).unwrap();

        // Should pull altitude toward 100m
        assert!(ekf.mean_state[2] > 90.0 && ekf.mean_state[2] < 110.0);
    }

    #[test]
    fn ekf_free_fall_motion() {
        // Test EKF with free fall motion profile
        let initial_state = InitialState {
            latitude: 0.0,
            longitude: 0.0,
            altitude: 100.0,
            northward_velocity: 0.0,
            eastward_velocity: 0.0,
            vertical_velocity: 0.0,
            roll: 0.0,
            pitch: 0.0,
            yaw: 0.0,
            in_degrees: false,
            is_enu: true,
        };

        let mut ekf = ExtendedKalmanFilter::new(
            &initial_state,
            &IMU_BIASES,
            vec![
                1e-6, 1e-6, 1.0, 0.1, 0.1, 0.1, 1e-4, 1e-4, 1e-4, 1e-6, 1e-6, 1e-6, 1e-8, 1e-8,
                1e-8,
            ],
            DMatrix::from_diagonal(&DVector::from_vec(vec![
                1e-9, 1e-9, 1e-6, 1e-6, 1e-6, 1e-6, 1e-9, 1e-9, 1e-9, 1e-9, 1e-9, 1e-9, 1e-9, 1e-9,
                1e-9,
            ])),
            true,
        );

        let dt = 0.1;
        let num_steps = 10;

        // Simulate free fall with only gravity (no vertical acceleration resistance)
        for _ in 0..num_steps {
            let imu_data = IMUData {
                accel: Vector3::new(0.0, 0.0, 0.0), // Free fall - no measured acceleration
                gyro: Vector3::new(0.0, 0.0, 0.0),
            };
            ekf.predict(&imu_data, dt).unwrap();
        }

        // After 1 second of free fall, should have accumulated vertical velocity
        let final_vd = ekf.mean_state[5];
        assert!(
            final_vd < -5.0,
            "Expected significant vertical velocity, got {final_vd}"
        );

        // Altitude should have decreased
        let final_altitude = ekf.mean_state[2];
        assert!(
            final_altitude < 100.0,
            "Expected altitude decrease, got {final_altitude}"
        );

        // Apply measurement update with GPS position
        let measurement = GPSPositionMeasurement {
            latitude: 0.0,
            longitude: 0.0,
            altitude: final_altitude,
            horizontal_noise_std: 5.0,
            vertical_noise_std: 2.0,
        };
        ekf.update(&measurement).unwrap();

        // After measurement update, estimate should remain close to measurement
        assert_approx_eq!(ekf.mean_state[2], final_altitude, 5.0);
    }

    #[test]
    fn ekf_hover_motion() {
        // Test EKF with hover (stationary vertical) motion profile
        let initial_state = InitialState {
            latitude: 0.0,
            longitude: 0.0,
            altitude: 100.0,
            northward_velocity: 0.0,
            eastward_velocity: 0.0,
            vertical_velocity: 0.0,
            roll: 0.0,
            pitch: 0.0,
            yaw: 0.0,
            in_degrees: false,
            is_enu: true,
        };

        let mut ekf = ExtendedKalmanFilter::new(
            &initial_state,
            &IMU_BIASES,
            vec![
                1e-6, 1e-6, 1.0, 0.1, 0.1, 0.1, 1e-4, 1e-4, 1e-4, 1e-6, 1e-6, 1e-6, 1e-8, 1e-8,
                1e-8,
            ],
            DMatrix::from_diagonal(&DVector::from_vec(vec![
                1e-9, 1e-9, 1e-6, 1e-6, 1e-6, 1e-6, 1e-9, 1e-9, 1e-9, 1e-9, 1e-9, 1e-9, 1e-9, 1e-9,
                1e-9,
            ])),
            true,
        );

        let dt = 0.1;
        let num_steps = 10;

        // Simulate hover with upward acceleration exactly canceling gravity
        for _ in 0..num_steps {
            let imu_data = IMUData {
                accel: Vector3::new(0.0, 0.0, earth::gravity(&0.0, &0.0)),
                gyro: Vector3::new(0.0, 0.0, 0.0),
            };
            ekf.predict(&imu_data, dt).unwrap();
        }

        // Velocity should remain near zero
        let final_vn = ekf.mean_state[3];
        let final_ve = ekf.mean_state[4];
        let final_vd = ekf.mean_state[5];
        assert_approx_eq!(final_vn, 0.0, 0.5);
        assert_approx_eq!(final_ve, 0.0, 0.5);
        assert_approx_eq!(final_vd, 0.0, 0.5);

        // Altitude should remain approximately constant
        let final_altitude = ekf.mean_state[2];
        assert_approx_eq!(final_altitude, 100.0, 1.0);

        // Apply GPS velocity measurement to verify zero velocity state
        let vel_measurement = GPSVelocityMeasurement {
            northward_velocity: 0.0,
            eastward_velocity: 0.0,
            vertical_velocity: 0.0,
            horizontal_noise_std: 0.5,
            vertical_noise_std: 0.5,
        };
        ekf.update(&vel_measurement).unwrap();

        // After update, velocities should remain near zero
        assert_approx_eq!(ekf.mean_state[3], 0.0, 0.5);
        assert_approx_eq!(ekf.mean_state[4], 0.0, 0.5);
        assert_approx_eq!(ekf.mean_state[5], 0.0, 0.5);
    }

    #[test]
    fn ekf_northward_motion() {
        // Test EKF with constant northward velocity motion profile
        let initial_state = InitialState {
            latitude: 0.0,
            longitude: 0.0,
            altitude: 100.0,
            northward_velocity: 10.0, // 10 m/s northward
            eastward_velocity: 0.0,
            vertical_velocity: 0.0,
            roll: 0.0,
            pitch: 0.0,
            yaw: 0.0,
            in_degrees: false,
            is_enu: true,
        };

        let mut ekf = ExtendedKalmanFilter::new(
            &initial_state,
            &IMU_BIASES,
            vec![
                1e-6, 1e-6, 1.0, 0.1, 0.1, 0.1, 1e-4, 1e-4, 1e-4, 1e-6, 1e-6, 1e-6, 1e-8, 1e-8,
                1e-8,
            ],
            DMatrix::from_diagonal(&DVector::from_vec(vec![
                1e-9, 1e-9, 1e-6, 1e-6, 1e-6, 1e-6, 1e-9, 1e-9, 1e-9, 1e-9, 1e-9, 1e-9, 1e-9, 1e-9,
                1e-9,
            ])),
            true,
        );

        let dt = 0.1;
        let num_steps = 10;
        let initial_lat = ekf.mean_state[0];

        // Simulate constant northward motion with gravity compensation
        for _ in 0..num_steps {
            let imu_data = IMUData {
                accel: Vector3::new(0.0, 0.0, earth::gravity(&0.0, &0.0)),
                gyro: Vector3::new(0.0, 0.0, 0.0),
            };
            ekf.predict(&imu_data, dt).unwrap();
        }

        // Latitude should have increased (moving north)
        let final_lat = ekf.mean_state[0];
        assert!(
            final_lat > initial_lat,
            "Expected latitude increase, got initial: {initial_lat} final: {final_lat}"
        );

        // Northward velocity should remain approximately constant
        let final_vn = ekf.mean_state[3];
        assert_approx_eq!(final_vn, 10.0, 2.0);

        // Eastward velocity should remain near zero
        let final_ve = ekf.mean_state[4];
        assert_approx_eq!(final_ve, 0.0, 0.5);

        // Apply GPS position and velocity measurement
        let meas = GPSPositionAndVelocityMeasurement {
            latitude: final_lat.to_degrees(),
            longitude: 0.0,
            altitude: 100.0,
            northward_velocity: 10.0,
            eastward_velocity: 0.0,
            horizontal_noise_std: 5.0,
            vertical_noise_std: 2.0,
            velocity_noise_std: 0.5,
        };
        ekf.update(&meas).unwrap();

        // After measurement, velocities should be close to measured values
        assert_approx_eq!(ekf.mean_state[3], 10.0, 1.0);
        assert_approx_eq!(ekf.mean_state[4], 0.0, 0.5);
    }

    #[test]
    fn ekf_eastward_motion() {
        // Test EKF with constant eastward velocity motion profile
        let initial_state = InitialState {
            latitude: 0.0,
            longitude: 0.0,
            altitude: 100.0,
            northward_velocity: 0.0,
            eastward_velocity: 15.0, // 15 m/s eastward
            vertical_velocity: 0.0,
            roll: 0.0,
            pitch: 0.0,
            yaw: 0.0,
            in_degrees: false,
            is_enu: true,
        };

        let mut ekf = ExtendedKalmanFilter::new(
            &initial_state,
            &IMU_BIASES,
            vec![
                1e-6, 1e-6, 1.0, 0.1, 0.1, 0.1, 1e-4, 1e-4, 1e-4, 1e-6, 1e-6, 1e-6, 1e-8, 1e-8,
                1e-8,
            ],
            DMatrix::from_diagonal(&DVector::from_vec(vec![
                1e-9, 1e-9, 1e-6, 1e-6, 1e-6, 1e-6, 1e-9, 1e-9, 1e-9, 1e-9, 1e-9, 1e-9, 1e-9, 1e-9,
                1e-9,
            ])),
            true,
        );

        let dt = 0.1;
        let num_steps = 10;
        let initial_lon = ekf.mean_state[1];

        // Simulate constant eastward motion with gravity compensation
        for _ in 0..num_steps {
            let imu_data = IMUData {
                accel: Vector3::new(0.0, 0.0, earth::gravity(&0.0, &0.0)),
                gyro: Vector3::new(0.0, 0.0, 0.0),
            };
            ekf.predict(&imu_data, dt).unwrap();
        }

        // Longitude should have increased (moving east)
        let final_lon = ekf.mean_state[1];
        assert!(
            final_lon > initial_lon,
            "Expected longitude increase, got initial: {initial_lon} final: {final_lon}"
        );

        // Eastward velocity should remain approximately constant
        let final_ve = ekf.mean_state[4];
        assert_approx_eq!(final_ve, 15.0, 2.0);

        // Northward velocity should remain near zero
        let final_vn = ekf.mean_state[3];
        assert_approx_eq!(final_vn, 0.0, 0.5);

        // Vertical velocity should remain near zero
        let final_vd = ekf.mean_state[5];
        assert_approx_eq!(final_vd, 0.0, 0.5);

        // Apply GPS position measurement
        let pos_meas = GPSPositionMeasurement {
            latitude: 0.0,
            longitude: final_lon.to_degrees(),
            altitude: 100.0,
            horizontal_noise_std: 5.0,
            vertical_noise_std: 2.0,
        };
        ekf.update(&pos_meas).unwrap();

        // Position should remain close to measurement
        assert_approx_eq!(ekf.mean_state[1], final_lon, 0.01);
        assert_approx_eq!(ekf.mean_state[2], 100.0, 5.0);

        // Apply velocity measurement
        let vel_meas = GPSVelocityMeasurement {
            northward_velocity: 0.0,
            eastward_velocity: 15.0,
            vertical_velocity: 0.0,
            horizontal_noise_std: 0.5,
            vertical_noise_std: 0.5,
        };
        ekf.update(&vel_meas).unwrap();

        // After measurement, velocities should be close to measured values
        assert_approx_eq!(ekf.mean_state[3], 0.0, 0.5);
        assert_approx_eq!(ekf.mean_state[4], 15.0, 1.0);
        assert_approx_eq!(ekf.mean_state[5], 0.0, 0.5);
    }

    #[test]
    fn ekf_combined_horizontal_motion() {
        // Test EKF with combined northward and eastward motion
        let initial_state = InitialState {
            latitude: 0.0,
            longitude: 0.0,
            altitude: 100.0,
            northward_velocity: 10.0,
            eastward_velocity: 10.0,
            vertical_velocity: 0.0,
            roll: 0.0,
            pitch: 0.0,
            yaw: 0.0,
            in_degrees: false,
            is_enu: true,
        };

        let mut ekf = ExtendedKalmanFilter::new(
            &initial_state,
            &IMU_BIASES,
            vec![
                1e-6, 1e-6, 1.0, 0.1, 0.1, 0.1, 1e-4, 1e-4, 1e-4, 1e-6, 1e-6, 1e-6, 1e-8, 1e-8,
                1e-8,
            ],
            DMatrix::from_diagonal(&DVector::from_vec(vec![
                1e-9, 1e-9, 1e-6, 1e-6, 1e-6, 1e-6, 1e-9, 1e-9, 1e-9, 1e-9, 1e-9, 1e-9, 1e-9, 1e-9,
                1e-9,
            ])),
            true,
        );

        let dt = 0.1;
        let num_steps = 10;
        let initial_lat = ekf.mean_state[0];
        let initial_lon = ekf.mean_state[1];

        // Simulate combined motion
        for _ in 0..num_steps {
            let imu_data = IMUData {
                accel: Vector3::new(0.0, 0.0, earth::gravity(&0.0, &0.0)),
                gyro: Vector3::new(0.0, 0.0, 0.0),
            };
            ekf.predict(&imu_data, dt).unwrap();
        }

        // Both latitude and longitude should have increased
        let final_lat = ekf.mean_state[0];
        let final_lon = ekf.mean_state[1];
        assert!(final_lat > initial_lat, "Expected latitude increase");
        assert!(final_lon > initial_lon, "Expected longitude increase");

        // Both velocities should remain approximately constant
        assert_approx_eq!(ekf.mean_state[3], 10.0, 2.0);
        assert_approx_eq!(ekf.mean_state[4], 10.0, 2.0);

        // Apply combined position and velocity measurement
        let meas = GPSPositionAndVelocityMeasurement {
            latitude: final_lat.to_degrees(),
            longitude: final_lon.to_degrees(),
            altitude: 100.0,
            northward_velocity: 10.0,
            eastward_velocity: 10.0,
            horizontal_noise_std: 5.0,
            vertical_noise_std: 2.0,
            velocity_noise_std: 0.5,
        };
        ekf.update(&meas).unwrap();

        // After measurement, state should be well-constrained
        assert_approx_eq!(ekf.mean_state[3], 10.0, 1.0);
        assert_approx_eq!(ekf.mean_state[4], 10.0, 1.0);
        assert_approx_eq!(ekf.mean_state[2], 100.0, 5.0);
    }

    #[test]
    fn ekf_covariance_reduction() {
        // Test that measurement updates reduce covariance
        let mut ekf = ExtendedKalmanFilter::new(
            &UKF_PARAMS,
            &IMU_BIASES,
            vec![1.0; 15], // Start with high uncertainty
            DMatrix::from_diagonal(&DVector::from_vec(PROCESS_NOISE_DIAGONAL.to_vec())),
            true,
        );

        // Get initial covariance trace (sum of diagonal elements)
        let initial_trace: f64 = (0..15).map(|i| ekf.covariance[(i, i)]).sum();

        // Apply measurement update
        let measurement = GPSPositionMeasurement {
            latitude: 0.0,
            longitude: 0.0,
            altitude: 0.0,
            horizontal_noise_std: 1.0,
            vertical_noise_std: 1.0,
        };
        ekf.update(&measurement).unwrap();

        // Get final covariance trace
        let final_trace: f64 = (0..15).map(|i| ekf.covariance[(i, i)]).sum();

        // Covariance should decrease after measurement update
        assert!(
            final_trace < initial_trace,
            "Covariance should decrease after measurement update: {final_trace} >= {initial_trace}"
        );
    }

    #[test]
    fn ekf_angle_wrapping() {
        // Test that angles are properly wrapped to [-pi, pi]
        let initial_state = InitialState {
            latitude: 0.0,
            longitude: 0.0,
            altitude: 100.0,
            northward_velocity: 0.0,
            eastward_velocity: 0.0,
            vertical_velocity: 0.0,
            // Negative, and close to -pi, deliberately. A *positive* 3.0 would make this
            // test degenerate: the measurement below has zero innovation, so the state is
            // unchanged, and `wrap_to_2pi(3.0)` and `wrap_to_pi(3.0)` are both 3.0 -- the
            // assertion would hold against the pre-#314 code and against no wrap at all.
            // At -3.0 the old `wrap_to_2pi` reported 3.283, outside `[-pi, pi]`.
            roll: -3.0,
            pitch: -3.0,
            yaw: -3.0,
            in_degrees: false,
            is_enu: true,
        };

        let mut ekf = ExtendedKalmanFilter::new(
            &initial_state,
            &IMU_BIASES,
            COVARIANCE_DIAGONAL.to_vec(),
            DMatrix::from_diagonal(&DVector::from_vec(PROCESS_NOISE_DIAGONAL.to_vec())),
            true,
        );

        // Apply a measurement update (which triggers angle wrapping)
        let measurement = GPSPositionMeasurement {
            latitude: 0.0,
            longitude: 0.0,
            altitude: 100.0,
            horizontal_noise_std: 5.0,
            vertical_noise_std: 2.0,
        };
        ekf.update(&measurement).unwrap();

        // Angles should land on `Rotation3::euler_angles`'s principal branch, `[-pi, pi]`.
        // Note this test never calls `predict`, so pitch is still ~-3.0 rad -- inside
        // `[-pi, pi]` but outside `[-pi/2, pi/2]`. That is why
        // `wrap_attitude_onto_principal_branch` must not clamp pitch to the half range:
        // doing so here would change the rotation the filter holds rather than rename it.
        for (name, angle) in [
            ("roll", ekf.mean_state[6]),
            ("pitch", ekf.mean_state[7]),
            ("yaw", ekf.mean_state[8]),
        ] {
            assert!(
                (-std::f64::consts::PI..=std::f64::consts::PI).contains(&angle),
                "{name} should lie on [-pi, pi] after an update, got {angle}"
            );
            // The range check alone would pass for a filter that had silently flipped the
            // sign; pin that the seed's own sign survives, which a 0..2*pi wrap destroys.
            assert!(
                angle < 0.0,
                "{name} was seeded at -3.0 rad and should still be negative, got {angle}"
            );
        }
    }

    /// #336: sigma points either side of the `+/-pi` cut are put back on one branch.
    ///
    /// The unit-level statement of the defect. `euler_angles` canonicalises each propagated
    /// sigma point onto `[-pi, pi]` independently, so a set spread across a southerly
    /// heading comes back as a mix of `+179` and `-179` deg; averaging those numbers
    /// linearly gives ~0 deg -- due *north* -- and with the UKF's non-convex weights it does
    /// not even stay between them.
    #[test]
    fn unwrap_attitude_puts_sigma_points_back_on_one_branch() {
        let reference = [0.0, 0.0, std::f64::consts::PI - 0.01];
        let mut straddling = DVector::<f64>::zeros(9);
        // The same attitude a hundredth of a radian the *other* side of the cut, as
        // `euler_angles` would report it.
        straddling[8] = -std::f64::consts::PI + 0.01;

        unwrap_attitude_onto_reference_branch(&mut straddling, &reference);

        // 0.02 rad from the reference, not the 6.26 rad the raw values differ by.
        assert_approx_eq!(straddling[8] - reference[2], 0.02, 1e-12);
        // And still the same rotation it was handed.
        assert_approx_eq!(
            wrap_to_pi(straddling[8]),
            -std::f64::consts::PI + 0.01,
            1e-12
        );
    }

    /// An angle already on the reference's branch is left *bit-for-bit* alone.
    ///
    /// Not a nicety. The UKF mean weights run to about -1e6, so an ulp of drift per sigma
    /// point per step becomes a visible change in the estimate over a long run: the first
    /// spelling of this helper, which rebuilt every angle as `reference + wrap_to_pi(angle -
    /// reference)`, moved `filter_comparison`'s *northbound* UKF yaw by 0.09 rad, on a
    /// heading where nothing straddles the cut and the fix is supposed to do nothing.
    #[test]
    fn unwrap_attitude_is_exactly_identity_away_from_the_cut() {
        let reference = [0.1, -0.2, 0.3];
        let original = DVector::from_vec(vec![
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            0.100_000_1,
            -0.199_999_9,
            0.300_000_2,
        ]);
        let mut unwrapped = original.clone();

        unwrap_attitude_onto_reference_branch(&mut unwrapped, &reference);

        assert_eq!(
            unwrapped, original,
            "an angle on the reference's own branch must survive untouched, not merely \
             approximately"
        );
    }

    /// #371: averaging sigma-point attitudes in the Euler chart puts the mean nowhere near
    /// the points, and shrinking the spread does not help.
    ///
    /// The numbers are the whole argument. With this filter's own weights -- `n = 15`,
    /// `alpha = 1e-3`, so `w_0` is about `-1.0e6` against `w_i` of about `+3.3e4`, summing to
    /// exactly 1 -- 31 rotations all lying inside a **0.24 degree** cone average, in the
    /// chart, to a point **34 degrees outside** it. That is 140 times the spread of the
    /// inputs, from a set whose members are indistinguishable by eye.
    ///
    /// A second-order error would fall as the square of the spread. This one does not fall at
    /// all: the spread shrinking is exactly what makes the weights large, and the two cancel.
    /// That is why no value of `alpha` ever reached this defect and why the original `alpha`
    /// sweep (89.8, 90.5, 90.1, 96.5, 108.3 degrees over `1e-3 .. 1.0`) got *worse* as the
    /// spread grew rather than better.
    ///
    /// **The level-north case is the control.** There the chart is locally linear, the two
    /// means agree to 1e-9 degrees, and the defect is simply absent -- which is what makes
    /// this a chart nonlinearity rather than an arithmetic slip. It is also why this went
    /// unnoticed: the attitude has to be genuinely away from level before the chart bites.
    #[test]
    fn the_euler_chart_mean_leaves_the_cone_its_inputs_sit_in() {
        /// Half-angle of the cone every sigma point is placed inside, degrees.
        const CONE_HALF_ANGLE_DEG: f64 = 0.24;
        /// How far outside that cone the chart mean must land before this test is satisfied.
        /// Measured at 33.96 degrees; the bound is loose because the point is the order of
        /// magnitude, not the digit.
        const MIN_CHART_ERROR_DEG: f64 = 10.0;

        fn rotation_vector(rotation: &Rotation3<f64>) -> Vector3<f64> {
            UnitQuaternion::from_rotation_matrix(rotation).scaled_axis()
        }

        /// Place `2n + 1` rotations in the sigma-point pattern around `base`, then report how
        /// far the chart mean and the manifold mean each land from it, in degrees.
        fn two_means_about(base: Rotation3<f64>, weights: &DVector<f64>) -> (f64, f64) {
            let state_size = (weights.len() - 1) / 2;
            let half_angle = CONE_HALF_ANGLE_DEG.to_radians();
            let mut points = vec![base];
            for sign in [1.0, -1.0] {
                for i in 0..state_size {
                    let mut axis = Vector3::zeros();
                    axis[i % 3] = sign * half_angle;
                    points.push(base * Rotation3::from_scaled_axis(axis));
                }
            }

            let mut chart = Vector3::<f64>::zeros();
            let mut tangent = Vector3::<f64>::zeros();
            for (i, point) in points.iter().enumerate() {
                let (roll, pitch, yaw) = point.euler_angles();
                chart += weights[i] * Vector3::new(roll, pitch, yaw);
                tangent += weights[i] * rotation_vector(&(base.transpose() * point));
            }
            let chart_mean = Rotation3::from_euler_angles(chart[0], chart[1], chart[2]);
            let manifold_mean = base * Rotation3::from_scaled_axis(tangent);

            (
                rotation_vector(&(base.transpose() * chart_mean))
                    .norm()
                    .to_degrees(),
                rotation_vector(&(base.transpose() * manifold_mean))
                    .norm()
                    .to_degrees(),
            )
        }

        // The filter's real weights, read off a real filter rather than re-derived here.
        let filter = UnscentedKalmanFilter::new(
            &InitialState::default(),
            &[0.0; 6],
            None,
            vec![1e-6; 15],
            DMatrix::from_diagonal(&DVector::from_vec(vec![1e-9; 15])),
            1e-3,
            2.0,
            0.0,
        );
        assert_approx_eq!(filter.weights_mean.sum(), 1.0, 1e-9);
        assert!(
            filter.weights_mean[0] < -1e5 && filter.weights_mean[1] > 1e4,
            "the defect is about non-convex weights; if these are convex the test proves \
             nothing. w_0 = {}, w_1 = {}",
            filter.weights_mean[0],
            filter.weights_mean[1]
        );

        // Control: level and pointing north, where the chart is locally linear.
        let (level_chart, level_manifold) = two_means_about(
            Rotation3::from_euler_angles(0.0, 0.0, 0.0),
            &filter.weights_mean,
        );
        assert!(
            level_chart < 1e-9 && level_manifold < 1e-9,
            "at level north the two means must agree -- if they do not, this test is \
             measuring something other than the chart. chart {level_chart} deg, manifold \
             {level_manifold} deg"
        );

        // The case that matters: a banked, non-level attitude of the kind any real
        // trajectory spends its time in.
        let (banked_chart, banked_manifold) = two_means_about(
            Rotation3::from_euler_angles(0.3, -0.2, 1.1),
            &filter.weights_mean,
        );
        assert!(
            banked_manifold < 1e-6,
            "the manifold mean of a symmetric point set is its anchor, exactly; got \
             {banked_manifold} deg"
        );
        assert!(
            banked_chart > MIN_CHART_ERROR_DEG,
            "the chart mean is supposed to be badly wrong here -- that is the defect this \
             filter's arithmetic avoids. Inputs inside a {CONE_HALF_ANGLE_DEG} deg cone, \
             chart mean {banked_chart} deg out"
        );
    }

    /// Every sigma point is in one Euler representation, including sigma point 0.
    ///
    /// `InitialState` accepts any roll/pitch/yaw and the constructor stores them verbatim,
    /// but `Rotation3::euler_angles` canonicalises pitch onto `[-pi/2, pi/2]`. A mean carrying
    /// an equivalent *non-principal* triple -- pitch -3.0 rad is the same rotation as pitch
    /// -0.1416 with roll and yaw shifted by a half turn -- therefore has two spellings, and
    /// `get_sigma_points` writes the perturbed columns through `euler_angles` while sigma
    /// point 0 is a copy of the mean.
    ///
    /// When those spellings differ the manifold arithmetic is still correct, because it reads
    /// the rows back as a rotation and a rotation does not care. A **measurement model** does:
    /// `MagnetometerYawMeasurement` reads the yaw row straight out of the sigma point. Before
    /// sigma point 0 was canonicalised too, the set below came out with a **3.1416 rad** yaw
    /// spread and a 2.8584 rad pitch spread against a covariance whose true spread is 4e-8 --
    /// a fabricated half turn, handed to the heading update as if it were uncertainty.
    /// `unwrap_attitude_onto_reference_branch` cannot repair it: it adds whole turns, and this
    /// is a half turn plus a reflection.
    ///
    /// Reachable only on the first call after construction -- `predict` and `update` both
    /// write the mean back through `euler_angles` and so leave it canonical -- which is the
    /// one moment the filter has no history to absorb a wildly wrong covariance.
    #[test]
    fn every_sigma_point_shares_one_euler_spelling_of_the_mean() {
        /// A tight covariance, so any spread worth seeing is the defect and not the prior.
        const TIGHT_VARIANCE: f64 = 1e-10;
        /// `TIGHT_VARIANCE` scaled by the transform gives about 4e-8 rad; anything past this
        /// is a fabricated branch, not a sigma point.
        const MAX_HONEST_SPREAD_RAD: f64 = 1e-6;

        for (label, pitch) in [("principal", 0.2_f64), ("non-principal", -3.0)] {
            let initial_state = InitialState {
                latitude: 40.0,
                longitude: -75.0,
                altitude: 100.0,
                roll: 0.1,
                pitch,
                yaw: 0.3,
                in_degrees: false,
                ..Default::default()
            };
            let filter = UnscentedKalmanFilter::new(
                &initial_state,
                &[0.0; 6],
                None,
                vec![TIGHT_VARIANCE; 15],
                DMatrix::from_diagonal(&DVector::from_vec(vec![1e-12; 15])),
                1e-3,
                2.0,
                0.0,
            );

            let points = filter.get_sigma_points().expect("sigma points");
            for row in ATTITUDE_STATE_INDICES {
                let reference = points[(row, 0)];
                for column in 1..points.ncols() {
                    let spread = (points[(row, column)] - reference).abs();
                    assert!(
                        spread < MAX_HONEST_SPREAD_RAD,
                        "{label} pitch: attitude row {row} of sigma point {column} sits \
                         {spread} rad from sigma point 0, against a prior whose spread is \
                         ~4e-8. That is a second Euler spelling of the same rotation, not \
                         uncertainty."
                    );
                }
            }
        }
    }

    /// #336: a UKF seeded due south holds its heading instead of running away.
    ///
    /// The filter-level statement. Before the fix this reached a reported pitch of 5.7 rad
    /// and a yaw of 8e5 rad within three samples -- values that are not attitudes at all,
    /// since `euler_angles` cannot produce a pitch outside `[-pi/2, pi/2]` and 8e5 rad is
    /// 133,000 turns. Three `predict`s is enough because the mean re-seeds the next sigma
    /// set, so the error compounds immediately rather than accumulating slowly.
    #[test]
    fn ukf_holds_a_southerly_heading_across_predicts() {
        // 1e-3 rad rather than something tighter because a zero gyro does not hold the
        // platform level: with nothing cancelling Earth rate it tilts at ~7.3e-5 rad/s, so
        // 0.6 s of the stream below legitimately moves roll by 4.4e-5 rad. That is physics,
        // not the defect, and the defect is eight orders of magnitude the other side of this
        // bound -- the pre-fix run reached 5.7 rad of pitch and 8e5 rad of yaw.
        const MAX_ATTITUDE_DRIFT_RAD: f64 = 1e-3;

        let due_south = InitialState {
            yaw: std::f64::consts::PI,
            ..UKF_PARAMS
        };
        let mut ukf = UnscentedKalmanFilter::new(
            &due_south,
            &IMU_BIASES,
            None,
            COVARIANCE_DIAGONAL.to_vec(),
            DMatrix::from_diagonal(&DVector::from_vec(PROCESS_NOISE_DIAGONAL.to_vec())),
            1e-3,
            2.0,
            0.0,
        );

        // A still, level vehicle: no sensed rotation, and specific force opposing gravity.
        // Whatever this stream does to the attitude, it is the same thing at every heading.
        let sample = ImuSample {
            delta_v: Vector3::new(0.0, 0.0, -9.81 * 0.2),
            delta_theta: Vector3::zeros(),
            dt: 0.2,
        };
        for _ in 0..3 {
            ukf.predict(&sample, 0.2).unwrap();
        }

        let estimate = ukf.get_estimate();
        for (name, index) in [("roll", 6), ("pitch", 7), ("yaw", 8)] {
            assert!(
                (-std::f64::consts::PI..=std::f64::consts::PI).contains(&estimate[index]),
                "{name} = {} rad is off the -pi..pi branch `predict` reports on",
                estimate[index]
            );
        }
        // Differenced with `wrap_to_pi` so that a yaw reported as `-pi` counts as holding a
        // `+pi` seed rather than as a full turn of error.
        assert_approx_eq!(
            wrap_to_pi(estimate[8] - std::f64::consts::PI).abs(),
            0.0,
            MAX_ATTITUDE_DRIFT_RAD
        );
        assert_approx_eq!(estimate[6], 0.0, MAX_ATTITUDE_DRIFT_RAD);
        assert_approx_eq!(estimate[7], 0.0, MAX_ATTITUDE_DRIFT_RAD);
    }

    // ==================== Error-State Kalman Filter Tests ====================

    /// #266: the position rows of the ESKF error state are radians, not metres.
    ///
    /// `error_state_transition_jacobian` produces a radian position-error rate
    /// (`f[(0,3)] = dt / r_n`) and the GPS position Jacobian is the identity against a
    /// radian-valued measurement, so the injection must add the correction directly.
    /// Dividing by the principal radii here rescaled every horizontal correction by
    /// ~1/6.4e6 and left the horizontal channel effectively open loop.
    #[test]
    fn eskf_position_error_injection_is_in_radians() {
        let mut eskf = ErrorStateKalmanFilter::new(
            &UKF_PARAMS,
            &IMU_BIASES,
            COVARIANCE_DIAGONAL.to_vec(),
            DMatrix::from_diagonal(&DVector::from_vec(PROCESS_NOISE_DIAGONAL.to_vec())),
        );
        // Put the filter somewhere with a non-trivial cos(latitude) so a stray
        // 1/(r_e cos(lat)) factor cannot coincidentally cancel.
        eskf.nominal_latitude = 45.0_f64.to_radians();
        eskf.nominal_longitude = (-122.0_f64).to_radians();
        eskf.nominal_altitude = 100.0;

        let lat0 = eskf.nominal_latitude;
        let lon0 = eskf.nominal_longitude;
        let alt0 = eskf.nominal_altitude;

        let d_lat = 1.0e-6; // radians (~6.4 m)
        let d_lon = 2.0e-6; // radians
        let d_alt = 3.0; // metres
        eskf.error_state[0] = d_lat;
        eskf.error_state[1] = d_lon;
        eskf.error_state[2] = d_alt;

        eskf.inject_error_state();

        // Tolerance is bounded by double-precision cancellation at ~0.8 rad
        // (eps * 0.8 ~ 1.7e-16), not by the arithmetic under test.
        assert_approx_eq!(eskf.nominal_latitude - lat0, d_lat, 1e-15);
        assert_approx_eq!(eskf.nominal_longitude - lon0, d_lon, 1e-15);
        assert_approx_eq!(eskf.nominal_altitude - alt0, d_alt, 1e-12);

        // The specific regression: treating delta_p as metres would divide by the
        // principal radii, making the applied correction ~6.4e6 times too small.
        assert!(
            (eskf.nominal_latitude - lat0) > d_lat * 0.5,
            "latitude correction was rescaled -- delta_p is being treated as metres (#266)"
        );
    }

    /// #266: a filter whose tests pass by exact floating-point bit pattern is not
    /// tested. Perturbing an input by ~1e-12 relative must not change the answer by a
    /// physically meaningful amount.
    ///
    /// Before the fix, a 1e-14 perturbation grew ~1% per sample and reached 1.5e8 m of
    /// altitude over a 5,366-sample run.
    #[test]
    fn eskf_is_insensitive_to_tiny_input_perturbation() {
        fn run(perturb: f64) -> DVector<f64> {
            let mut eskf = ErrorStateKalmanFilter::new(
                &UKF_PARAMS,
                &IMU_BIASES,
                vec![1e-12; 15],
                DMatrix::from_diagonal(&DVector::from_vec(vec![1e-12; 15])),
            );
            let dt = 0.01;
            for step in 0..2000 {
                // Gentle, non-degenerate motion so the attitude is never identity.
                let t = f64::from(step) * dt;
                let imu = IMUData {
                    accel: Vector3::new(0.05 * t.cos() * (1.0 + perturb), -0.03 * t.sin(), 9.81),
                    gyro: Vector3::new(0.001, -0.002, 0.01),
                };
                eskf.predict(&imu, dt).unwrap();

                if step % 25 == 0 {
                    let meas = crate::measurements::GPSPositionMeasurement {
                        latitude: 0.0,
                        longitude: 0.0,
                        altitude: 0.0,
                        horizontal_noise_std: 3.0,
                        vertical_noise_std: 5.0,
                    };
                    eskf.update(&meas).unwrap();
                }
            }
            eskf.get_estimate()
        }

        let baseline = run(0.0);
        let perturbed = run(1e-12);

        for i in 0..15 {
            assert!(
                baseline[i].is_finite() && perturbed[i].is_finite(),
                "state {i} is not finite"
            );
        }

        // Altitude is the channel that ran away. A 1e-12 relative input change must
        // not move it by even a millimetre.
        let d_alt = (baseline[2] - perturbed[2]).abs();
        assert!(
            d_alt < 1e-3,
            "altitude moved {d_alt:e} m from a 1e-12 relative input perturbation -- \
             the filter is amplifying rounding noise (#266)"
        );

        let d_vel = (baseline[5] - perturbed[5]).abs();
        assert!(
            d_vel < 1e-4,
            "vertical velocity moved {d_vel:e} m/s from a 1e-12 relative perturbation (#266)"
        );
    }

    /// #258: the ESKF's propagation interface is the increment domain.
    ///
    /// Rates remain accepted, and the two must agree exactly: `ImuSample::from_rates` is the
    /// same rectangular integration the filter used to perform internally, so routing an
    /// `IMUData` through it may not perturb a single bit of the resulting state.
    /// Process noise must depend on elapsed time, not on how often the filter was stepped.
    ///
    /// This is #374's acceptance criterion, and the defect it guards is the reason it exists:
    /// the three Kalman filters added `DEFAULT_PROCESS_NOISE_DENSITY` once per IMU sample with
    /// no `dt` anywhere, so the process noise a trajectory accumulated was proportional to its
    /// sample rate. Across the data this repository ships -- 1 Hz on `test_data.csv`, 10 Hz
    /// from `generate_synthetic`'s default, 50 Hz on the `syn_*` baseline scenarios -- that is
    /// a **50x spread** from a single constant, and resampling a log silently retuned the
    /// filter.
    ///
    /// One second of stationary propagation at 100 Hz and at 10 Hz. `F P F^T` still differs a
    /// little between the two, because a coarser step is a coarser discretisation of the same
    /// continuous dynamics, so the bound is a few percent rather than exact. Before the fix
    /// the two differed by the rate ratio itself -- a factor of ten, which no tolerance of
    /// this kind would admit.
    #[test]
    fn process_noise_accumulates_with_elapsed_time_not_with_step_count() {
        const ELAPSED_S: f64 = 1.0;
        const TOLERANCE: f64 = 0.05;

        let level = IMUData {
            accel: Vector3::new(0.0, 0.0, earth::gravity(&0.0, &0.0)),
            gyro: Vector3::zeros(),
        };
        let density = DMatrix::from_diagonal(&DVector::from_vec(
            crate::sim::DEFAULT_PROCESS_NOISE_DENSITY.to_vec(),
        ));

        // `(steps, dt)` pairs that both cover exactly one second.
        let schedules = [(100_usize, 0.01_f64), (10_usize, 0.1_f64)];

        for filter_name in ["UKF", "EKF", "ESKF"] {
            let diagonals: Vec<DVector<f64>> = schedules
                .iter()
                .map(|&(steps, dt)| {
                    let mut filter: Box<dyn NavigationFilter> = match filter_name {
                        "UKF" => Box::new(UnscentedKalmanFilter::new(
                            &UKF_PARAMS,
                            &IMU_BIASES,
                            None,
                            COVARIANCE_DIAGONAL.to_vec(),
                            density.clone(),
                            ALPHA,
                            BETA,
                            KAPPA,
                        )),
                        "EKF" => Box::new(ExtendedKalmanFilter::new(
                            &UKF_PARAMS,
                            &IMU_BIASES,
                            COVARIANCE_DIAGONAL.to_vec(),
                            density.clone(),
                            true,
                        )),
                        _ => Box::new(ErrorStateKalmanFilter::new(
                            &UKF_PARAMS,
                            &IMU_BIASES,
                            COVARIANCE_DIAGONAL.to_vec(),
                            density.clone(),
                        )),
                    };
                    for _ in 0..steps {
                        filter.predict(&level, dt).expect("stationary predict");
                    }
                    filter.get_certainty().diagonal()
                })
                .collect();

            let (fast, slow) = (&diagonals[0], &diagonals[1]);
            for index in 0..fast.len() {
                let (a, b) = (fast[index], slow[index]);
                let scale = a.abs().max(b.abs());
                if scale == 0.0 {
                    continue;
                }
                assert!(
                    (a - b).abs() / scale < TOLERANCE,
                    "{filter_name} state {index} covariance after {ELAPSED_S} s is {a:.6e} at \
                     100 Hz but {b:.6e} at 10 Hz -- a ratio of {:.2}. Process noise must \
                     accumulate with elapsed time, not with step count (#374).",
                    a / b
                );
            }
        }
    }

    #[test]
    fn eskf_predicts_identically_from_a_sample_and_from_rates() {
        let dt = 0.02;
        let imu = IMUData {
            accel: Vector3::new(0.35, -0.12, earth::gravity(&0.0, &0.0)),
            gyro: Vector3::new(0.004, -0.011, 0.007),
        };

        let build = || {
            ErrorStateKalmanFilter::new(
                &UKF_PARAMS,
                // Non-zero biases: the increment-domain correction subtracts `bias * dt`
                // where the rate-domain one subtracted `bias`, so a zero bias would let a
                // wrong correction pass unnoticed.
                &[0.05, -0.03, 0.02, 1e-3, -2e-3, 5e-4],
                COVARIANCE_DIAGONAL.to_vec(),
                DMatrix::from_diagonal(&DVector::from_vec(PROCESS_NOISE_DIAGONAL.to_vec())),
            )
        };

        let mut from_rates = build();
        let mut from_sample = build();
        let sample = ImuSample::from_rates(&imu, dt);

        for _ in 0..25 {
            from_rates.predict(&imu, dt).unwrap();
            from_sample.predict(&sample, dt).unwrap();
        }

        let expected = from_rates.get_estimate();
        let actual = from_sample.get_estimate();
        for i in 0..15 {
            assert_eq!(
                expected[i], actual[i],
                "state {i} differs between the rate and increment inputs: \
                 {} vs {}",
                expected[i], actual[i]
            );
        }
        assert_eq!(
            from_rates.error_covariance, from_sample.error_covariance,
            "error covariance differs between the rate and increment inputs"
        );
    }

    /// A sample carries the interval its increments were accumulated over. If that
    /// disagrees with the `dt` the filter is stepped by, one of the two is wrong and
    /// neither can be preferred silently -- that is how #292 happened.
    #[test]
    fn eskf_predict_rejects_a_sample_whose_dt_disagrees() {
        let mut eskf = ErrorStateKalmanFilter::new(
            &UKF_PARAMS,
            &IMU_BIASES,
            COVARIANCE_DIAGONAL.to_vec(),
            DMatrix::from_diagonal(&DVector::from_vec(PROCESS_NOISE_DIAGONAL.to_vec())),
        );
        let sample = ImuSample {
            delta_v: Vector3::new(0.0, 0.0, earth::gravity(&0.0, &0.0) * 0.02),
            delta_theta: Vector3::zeros(),
            dt: 0.02,
        };

        let err = eskf
            .predict(&sample, 0.01)
            .expect_err("a 2x timestep disagreement must be rejected");
        assert!(
            matches!(
                err,
                StrapdownError::InconsistentTimestep {
                    sample_dt,
                    arg_dt
                } if (sample_dt - 0.02).abs() < 1e-12 && (arg_dt - 0.01).abs() < 1e-12
            ),
            "expected InconsistentTimestep, got {err:?}"
        );

        // Rounding-level disagreement is not a defect: a caller deriving both from the same
        // timestamps can legitimately land a few ulps apart.
        eskf.predict(&sample, 0.02 * (1.0 + 1e-15))
            .expect("a few ulps of disagreement must still propagate");
    }

    /// `InputModel` admits `VelocityData` too, so the type system permits an input the ESKF
    /// cannot mechanize. It must be reported, not aborted on.
    #[test]
    fn eskf_predict_rejects_a_non_inertial_input() {
        let mut eskf = ErrorStateKalmanFilter::new(
            &UKF_PARAMS,
            &IMU_BIASES,
            COVARIANCE_DIAGONAL.to_vec(),
            DMatrix::from_diagonal(&DVector::from_vec(PROCESS_NOISE_DIAGONAL.to_vec())),
        );
        let velocity = crate::VelocityData {
            linear: Vector3::new(1.0, 0.0, 0.0),
            angular: Vector3::zeros(),
        };

        let err = eskf
            .predict(&velocity, 0.02)
            .expect_err("VelocityData is not an inertial input");
        assert!(
            matches!(
                err,
                StrapdownError::UnsupportedInput {
                    filter: "ErrorStateKalmanFilter",
                    expected: "ImuSample or IMUData",
                }
            ),
            "expected UnsupportedInput, got {err:?}"
        );
    }

    #[test]
    fn eskf_construction() {
        // Test ESKF construction
        let eskf = ErrorStateKalmanFilter::new(
            &UKF_PARAMS,
            &IMU_BIASES,
            COVARIANCE_DIAGONAL.to_vec(),
            DMatrix::from_diagonal(&DVector::from_vec(PROCESS_NOISE_DIAGONAL.to_vec())),
        );

        // Verify state size
        let state = eskf.get_estimate();
        assert_eq!(state.len(), 15);

        // Verify error state is initialized to zero
        assert_eq!(eskf.error_state.len(), 15);
        for i in 0..15 {
            assert_approx_eq!(eskf.error_state[i], 0.0, 1e-10);
        }

        // Verify quaternion is normalized
        let quat_norm = eskf.nominal_quaternion.norm();
        assert_approx_eq!(quat_norm, 1.0, 1e-6);
    }

    #[test]
    fn eskf_debug_display() {
        // Test Debug and Display implementations for ESKF
        let eskf = ErrorStateKalmanFilter::new(
            &UKF_PARAMS,
            &IMU_BIASES,
            COVARIANCE_DIAGONAL.to_vec(),
            DMatrix::from_diagonal(&DVector::from_vec(PROCESS_NOISE_DIAGONAL.to_vec())),
        );

        // Test Debug
        let debug_str = format!("{eskf:?}");
        assert!(debug_str.contains("ESKF"));
        assert!(debug_str.contains("nominal_position"));

        // Test Display
        let display_str = format!("{eskf}");
        assert!(display_str.contains("ErrorStateKalmanFilter"));
        assert!(display_str.contains("nominal_quaternion"));
    }

    #[test]
    fn eskf_quaternion_normalization() {
        // Test that quaternion remains normalized after predict/update cycles
        let mut eskf = ErrorStateKalmanFilter::new(
            &UKF_PARAMS,
            &IMU_BIASES,
            COVARIANCE_DIAGONAL.to_vec(),
            DMatrix::from_diagonal(&DVector::from_vec(PROCESS_NOISE_DIAGONAL.to_vec())),
        );

        // Run several predict/update cycles
        for _ in 0..10 {
            let imu_data = IMUData {
                accel: Vector3::new(0.0, 0.0, earth::gravity(&0.0, &0.0)),
                gyro: Vector3::new(0.01, 0.01, 0.01), // Small rotation
            };
            eskf.predict(&imu_data, 0.01).unwrap();

            let measurement = GPSPositionMeasurement {
                latitude: 0.0,
                longitude: 0.0,
                altitude: 0.0,
                horizontal_noise_std: 5.0,
                vertical_noise_std: 2.0,
            };
            eskf.update(&measurement).unwrap();
        }

        // Verify quaternion is still normalized
        let quat_norm = eskf.nominal_quaternion.norm();
        assert_approx_eq!(quat_norm, 1.0, 1e-6);
    }

    #[test]
    fn eskf_error_reset_after_update() {
        // Test that error state is reset to zero after measurement update
        let mut eskf = ErrorStateKalmanFilter::new(
            &UKF_PARAMS,
            &IMU_BIASES,
            vec![1e-3; 15], // Higher initial uncertainty
            DMatrix::from_diagonal(&DVector::from_vec(PROCESS_NOISE_DIAGONAL.to_vec())),
        );

        // Predict to build up error covariance
        let imu_data = IMUData {
            accel: Vector3::new(0.0, 0.0, earth::gravity(&0.0, &0.0)),
            gyro: Vector3::zeros(),
        };
        eskf.predict(&imu_data, 1.0).unwrap();

        // Update with measurement
        let measurement = GPSPositionMeasurement {
            latitude: 0.001, // Small offset from initial
            longitude: 0.001,
            altitude: 1.0,
            horizontal_noise_std: 5.0,
            vertical_noise_std: 2.0,
        };
        let before = eskf.get_estimate();
        eskf.update(&measurement).unwrap();
        let after = eskf.get_estimate();

        // The reset assertion below is only meaningful if a correction was actually
        // computed and injected. Without this, a no-op update would satisfy it trivially:
        // the error state starts at zero. The measurement sits north-east of and above the
        // nominal state, so the injection must move all three position components towards
        // it (#258).
        assert!(
            after[0] > before[0] && after[1] > before[1] && after[2] > before[2],
            "update injected no correction: position went from \
             [{:e}, {:e}, {:.3}] to [{:e}, {:e}, {:.3}]",
            before[0],
            before[1],
            before[2],
            after[0],
            after[1],
            after[2]
        );

        // Verify error state is reset to zero after update
        for i in 0..15 {
            assert_approx_eq!(eskf.error_state[i], 0.0, 1e-10);
        }
    }

    #[test]
    fn eskf_bias_estimation() {
        // Test that ESKF estimates and corrects biases
        let initial_state = InitialState {
            latitude: 0.0,
            longitude: 0.0,
            altitude: 100.0,
            northward_velocity: 0.0,
            eastward_velocity: 0.0,
            vertical_velocity: 0.0,
            roll: 0.0,
            pitch: 0.0,
            yaw: 0.0,
            in_degrees: false,
            is_enu: true,
        };

        // Initialize with non-zero biases
        let true_accel_bias = Vector3::new(0.1, 0.05, 0.08);
        let true_gyro_bias = Vector3::new(0.01, 0.015, 0.02);

        let mut eskf = ErrorStateKalmanFilter::new(
            &initial_state,
            &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], // Start with zero bias estimate
            vec![1e-6; 15],
            DMatrix::from_diagonal(&DVector::from_vec(vec![
                1e-9, 1e-9, 1e-6, 1e-6, 1e-6, 1e-6, 1e-9, 1e-9, 1e-9, 1e-6, 1e-6,
                1e-6, // Allow bias to change
                1e-8, 1e-8, 1e-8,
            ])),
        );

        // Run predict/update cycles with biased IMU data
        for _ in 0..20 {
            let imu_data = IMUData {
                accel: Vector3::new(0.0, 0.0, earth::gravity(&0.0, &0.0)) + true_accel_bias,
                gyro: true_gyro_bias,
            };
            eskf.predict(&imu_data, 0.1).unwrap();

            // Perfect measurements to help converge
            let measurement = GPSPositionMeasurement {
                latitude: 0.0,
                longitude: 0.0,
                altitude: 100.0,
                horizontal_noise_std: 1.0,
                vertical_noise_std: 0.5,
            };
            eskf.update(&measurement).unwrap();
        }

        // Verify biases remain bounded (not diverging)
        let state = eskf.get_estimate();
        assert!(state[9].abs() < 1.0); // accel bias x
        assert!(state[10].abs() < 1.0); // accel bias y
        assert!(state[11].abs() < 1.0); // accel bias z
        assert!(state[12].abs() < 0.5); // gyro bias x
        assert!(state[13].abs() < 0.5); // gyro bias y
        assert!(state[14].abs() < 0.5); // gyro bias z
    }

    #[test]
    fn eskf_hover_motion() {
        // Test ESKF with stationary hover motion
        let initial_state = InitialState {
            latitude: 0.0,
            longitude: 0.0,
            altitude: 100.0,
            northward_velocity: 0.0,
            eastward_velocity: 0.0,
            vertical_velocity: 0.0,
            roll: 0.0,
            pitch: 0.0,
            yaw: 0.0,
            in_degrees: false,
            is_enu: true,
        };

        let mut eskf = ErrorStateKalmanFilter::new(
            &initial_state,
            &IMU_BIASES,
            vec![
                1e-6, 1e-6, 1.0, 0.1, 0.1, 0.1, 1e-4, 1e-4, 1e-4, 1e-6, 1e-6, 1e-6, 1e-8, 1e-8,
                1e-8,
            ],
            DMatrix::from_diagonal(&DVector::from_vec(vec![
                1e-9, 1e-9, 1e-6, 1e-6, 1e-6, 1e-6, 1e-9, 1e-9, 1e-9, 1e-9, 1e-9, 1e-9, 1e-9, 1e-9,
                1e-9,
            ])),
        );

        let dt = 0.1;
        let num_steps = 10;

        // Simulate hover with gravity compensation
        for _ in 0..num_steps {
            let imu_data = IMUData {
                accel: Vector3::new(0.0, 0.0, earth::gravity(&0.0, &0.0)),
                gyro: Vector3::zeros(),
            };
            eskf.predict(&imu_data, dt).unwrap();
        }

        // Verify state remained approximately constant
        let state = eskf.get_estimate();
        assert_approx_eq!(state[0], 0.0, 0.001); // latitude
        assert_approx_eq!(state[1], 0.0, 0.001); // longitude
        assert_approx_eq!(state[2], 100.0, 1.0); // altitude
        assert_approx_eq!(state[3], 0.0, 0.5); // velocity north
        assert_approx_eq!(state[4], 0.0, 0.5); // velocity east
        assert_approx_eq!(state[5], 0.0, 0.5); // velocity vertical
    }

    #[test]
    fn eskf_with_velocity_measurement() {
        // Test ESKF with velocity measurement
        let mut eskf = ErrorStateKalmanFilter::new(
            &UKF_PARAMS,
            &IMU_BIASES,
            COVARIANCE_DIAGONAL.to_vec(),
            DMatrix::from_diagonal(&DVector::from_vec(PROCESS_NOISE_DIAGONAL.to_vec())),
        );

        let vel_meas = GPSVelocityMeasurement {
            northward_velocity: 0.0,
            eastward_velocity: 0.0,
            vertical_velocity: 0.0,
            horizontal_noise_std: 0.5,
            vertical_noise_std: 0.5,
        };

        eskf.update(&vel_meas).unwrap();

        // Verify update completed and error state is reset
        let state = eskf.get_estimate();
        assert_eq!(state.len(), 15);
        for i in 0..15 {
            assert_approx_eq!(eskf.error_state[i], 0.0, 1e-10);
        }
    }

    #[test]
    fn eskf_covariance_reduction() {
        // Test that measurement updates reduce covariance
        let mut eskf = ErrorStateKalmanFilter::new(
            &UKF_PARAMS,
            &IMU_BIASES,
            vec![1.0; 15], // Start with high uncertainty
            DMatrix::from_diagonal(&DVector::from_vec(PROCESS_NOISE_DIAGONAL.to_vec())),
        );

        // Get initial covariance trace
        let initial_trace: f64 = (0..15).map(|i| eskf.error_covariance[(i, i)]).sum();

        // Apply measurement update
        let measurement = GPSPositionMeasurement {
            latitude: 0.0,
            longitude: 0.0,
            altitude: 0.0,
            horizontal_noise_std: 1.0,
            vertical_noise_std: 1.0,
        };
        eskf.update(&measurement).unwrap();

        // Get final covariance trace
        let final_trace: f64 = (0..15).map(|i| eskf.error_covariance[(i, i)]).sum();

        // Covariance should decrease after measurement update
        assert!(
            final_trace < initial_trace,
            "Covariance should decrease after measurement update: {final_trace} >= {initial_trace}"
        );
    }

    #[test]
    fn eskf_angle_wrapping() {
        // Test that angles are properly wrapped
        let initial_state = InitialState {
            latitude: 0.0,
            longitude: 0.0,
            altitude: 100.0,
            northward_velocity: 0.0,
            eastward_velocity: 0.0,
            vertical_velocity: 0.0,
            roll: 3.0, // Close to pi
            pitch: 3.0,
            yaw: 3.0,
            in_degrees: false,
            is_enu: true,
        };

        let mut eskf = ErrorStateKalmanFilter::new(
            &initial_state,
            &IMU_BIASES,
            COVARIANCE_DIAGONAL.to_vec(),
            DMatrix::from_diagonal(&DVector::from_vec(PROCESS_NOISE_DIAGONAL.to_vec())),
        );

        // Apply a measurement update (which triggers angle wrapping in get_estimate)
        let measurement = GPSPositionMeasurement {
            latitude: 0.0,
            longitude: 0.0,
            altitude: 100.0,
            horizontal_noise_std: 5.0,
            vertical_noise_std: 2.0,
        };
        eskf.update(&measurement).unwrap();

        // Get state (which decomposes the nominal quaternion into Euler angles)
        let state = eskf.get_estimate();

        // The ESKF's triple always comes back through `UnitQuaternion::euler_angles`, so
        // unlike the EKF the tighter pitch bound is a derived fact here rather than a
        // convention: roll and yaw from `atan2` on -pi..pi, pitch from `asin` on
        // -pi/2..pi/2. The 3/3/3 rad seed canonicalises through the quaternion to roughly
        // (-0.14, 0.14, -0.14), so it is well inside both.
        for (name, angle, bound) in [
            ("roll", state[6], std::f64::consts::PI),
            ("pitch", state[7], std::f64::consts::FRAC_PI_2),
            ("yaw", state[8], std::f64::consts::PI),
        ] {
            assert!(
                (-bound..=bound).contains(&angle),
                "{name} should lie on the principal branch, got {angle}"
            );
        }
    }

    #[test]
    fn eskf_no_singularities() {
        // Test that ESKF avoids singularities even with large rotations
        let initial_state = InitialState {
            latitude: 45.0,
            longitude: -122.0,
            altitude: 100.0,
            northward_velocity: 10.0,
            eastward_velocity: 5.0,
            vertical_velocity: 0.0,
            roll: std::f64::consts::FRAC_PI_2 - 0.1, // Close to gimbal lock
            pitch: 0.0,
            yaw: 0.0,
            in_degrees: false,
            is_enu: true,
        };

        let mut eskf = ErrorStateKalmanFilter::new(
            &initial_state,
            &IMU_BIASES,
            COVARIANCE_DIAGONAL.to_vec(),
            DMatrix::from_diagonal(&DVector::from_vec(PROCESS_NOISE_DIAGONAL.to_vec())),
        );

        // Apply large rotations
        for _ in 0..10 {
            let imu_data = IMUData {
                accel: Vector3::new(0.0, 0.0, earth::gravity(&0.785, &100.0)),
                gyro: Vector3::new(0.5, 0.5, 0.5), // Large rotation rates
            };
            eskf.predict(&imu_data, 0.01).unwrap();
        }

        // Verify quaternion is still normalized (no singularities)
        let quat_norm = eskf.nominal_quaternion.norm();
        assert_approx_eq!(quat_norm, 1.0, 1e-6);

        // Verify state is still valid
        let state = eskf.get_estimate();
        assert!(state[0].is_finite()); // latitude
        assert!(state[1].is_finite()); // longitude
        assert!(state[2].is_finite()); // altitude
        for i in 6..9 {
            assert!(state[i].is_finite()); // angles
        }
    }

    /// #286: filter constructors must honour `in_degrees` for attitude.
    ///
    /// The ESKF built its initial quaternion from degree values when radians
    /// were supplied (and vice versa), producing a wildly wrong initial DCM
    /// for any non-zero attitude; the EKF/UKF dropped the attitude conversion
    /// in the degrees branch. All three must agree across unit modes and match
    /// a direct radians construction.
    #[test]
    fn filter_init_attitude_respects_angle_units() {
        let roll_rad: f64 = 0.1626;
        let pitch_rad: f64 = -1.3395;
        let yaw_rad: f64 = 0.1792;
        let mk = |in_degrees: bool| InitialState {
            latitude: 0.7,
            longitude: -1.3,
            altitude: 100.0,
            northward_velocity: 0.0,
            eastward_velocity: 0.0,
            vertical_velocity: 0.0,
            roll: if in_degrees {
                roll_rad.to_degrees()
            } else {
                roll_rad
            },
            pitch: if in_degrees {
                pitch_rad.to_degrees()
            } else {
                pitch_rad
            },
            yaw: if in_degrees {
                yaw_rad.to_degrees()
            } else {
                yaw_rad
            },
            in_degrees,
            is_enu: true,
        };
        let q15 = DMatrix::identity(15, 15);

        // ESKF: both unit modes must yield the same nominal quaternion...
        let eskf_rad = ErrorStateKalmanFilter::new(
            &mk(false),
            &IMU_BIASES,
            COVARIANCE_DIAGONAL.to_vec(),
            DMatrix::from_diagonal(&DVector::from_vec(PROCESS_NOISE_DIAGONAL.to_vec())),
        );
        let eskf_deg = ErrorStateKalmanFilter::new(
            &mk(true),
            &IMU_BIASES,
            COVARIANCE_DIAGONAL.to_vec(),
            DMatrix::from_diagonal(&DVector::from_vec(PROCESS_NOISE_DIAGONAL.to_vec())),
        );
        for i in 0..4 {
            assert_approx_eq!(
                eskf_rad.nominal_quaternion[i],
                eskf_deg.nominal_quaternion[i],
                1e-12
            );
        }
        // ...and it must equal the direct radians construction (catches an
        // inverted conversion, which also agrees across modes).
        let expected = UnitQuaternion::from_rotation_matrix(&Rotation3::from_euler_angles(
            roll_rad, pitch_rad, yaw_rad,
        ));
        assert_approx_eq!(eskf_rad.nominal_quaternion[0], expected.w, 1e-12);
        assert_approx_eq!(eskf_rad.nominal_quaternion[1], expected.i, 1e-12);
        assert_approx_eq!(eskf_rad.nominal_quaternion[2], expected.j, 1e-12);
        assert_approx_eq!(eskf_rad.nominal_quaternion[3], expected.k, 1e-12);

        // EKF/UKF: degree inputs must land in the mean state as radians.
        let ekf =
            ExtendedKalmanFilter::new(&mk(true), &IMU_BIASES, vec![1e-6; 15], q15.clone(), true);
        assert_approx_eq!(ekf.mean_state[6], roll_rad, 1e-12);
        assert_approx_eq!(ekf.mean_state[7], pitch_rad, 1e-12);
        assert_approx_eq!(ekf.mean_state[8], yaw_rad, 1e-12);
        let ukf = UnscentedKalmanFilter::new(
            &mk(true),
            &IMU_BIASES,
            None,
            vec![1e-6; 15],
            q15,
            1e-3,
            2.0,
            0.0,
        );
        assert_approx_eq!(ukf.mean_state[6], roll_rad, 1e-12);
        assert_approx_eq!(ukf.mean_state[7], pitch_rad, 1e-12);
        assert_approx_eq!(ukf.mean_state[8], yaw_rad, 1e-12);
    }

    /// `InitialState::new` in radians must agree with the equivalent struct literal.
    ///
    /// The radian branch used to store `latitude.to_degrees()` while leaving
    /// `in_degrees == false`, so the filter constructors -- which convert only
    /// when `in_degrees` is true -- consumed 40 deg N (0.698 rad) as the number
    /// 40 *radians*. Longitude was passed through untouched, so the two did not
    /// even agree with each other.
    #[test]
    fn initial_state_new_in_radians_matches_struct_literal() {
        let latitude_rad: f64 = 40.0_f64.to_radians();
        let longitude_rad: f64 = (-75.0_f64).to_radians();

        let constructed = InitialState::new(
            latitude_rad,
            longitude_rad,
            100.0,
            1.0,
            2.0,
            3.0,
            0.1626,
            -0.9395,
            0.1792,
            false,
            None,
        );
        let literal = InitialState {
            latitude: latitude_rad,
            longitude: longitude_rad,
            altitude: 100.0,
            northward_velocity: 1.0,
            eastward_velocity: 2.0,
            vertical_velocity: 3.0,
            roll: 0.1626,
            pitch: -0.9395,
            yaw: 0.1792,
            in_degrees: false,
            is_enu: false,
        };

        // The stored fields are already radians, so the two agree before any filter sees them.
        assert_approx_eq!(constructed.latitude, literal.latitude, 1e-15);
        assert_approx_eq!(constructed.longitude, literal.longitude, 1e-15);

        let eskf = |initial_state: &InitialState| {
            ErrorStateKalmanFilter::new(
                initial_state,
                &IMU_BIASES,
                COVARIANCE_DIAGONAL.to_vec(),
                DMatrix::from_diagonal(&DVector::from_vec(PROCESS_NOISE_DIAGONAL.to_vec())),
            )
        };
        let from_constructor = eskf(&constructed);
        let from_literal = eskf(&literal);
        assert_approx_eq!(
            from_constructor.nominal_latitude,
            from_literal.nominal_latitude,
            1e-15
        );
        assert_approx_eq!(
            from_constructor.nominal_longitude,
            from_literal.nominal_longitude,
            1e-15
        );
        // ...and both are the radians that went in, not degrees read as radians.
        assert_approx_eq!(from_constructor.nominal_latitude, latitude_rad, 1e-15);
        assert_approx_eq!(from_constructor.nominal_longitude, longitude_rad, 1e-15);

        // Same for the EKF and UKF, which read lat/lon into the mean state.
        let q15 = DMatrix::identity(15, 15);
        let ekf_constructed =
            ExtendedKalmanFilter::new(&constructed, &IMU_BIASES, vec![1e-6; 15], q15.clone(), true);
        let ekf_literal =
            ExtendedKalmanFilter::new(&literal, &IMU_BIASES, vec![1e-6; 15], q15.clone(), true);
        assert_approx_eq!(
            ekf_constructed.mean_state[0],
            ekf_literal.mean_state[0],
            1e-15
        );
        assert_approx_eq!(
            ekf_constructed.mean_state[1],
            ekf_literal.mean_state[1],
            1e-15
        );
        assert_approx_eq!(ekf_constructed.mean_state[0], latitude_rad, 1e-15);
        assert_approx_eq!(ekf_constructed.mean_state[1], longitude_rad, 1e-15);

        let ukf = |initial_state: &InitialState| {
            UnscentedKalmanFilter::new(
                initial_state,
                &IMU_BIASES,
                None,
                vec![1e-6; 15],
                q15.clone(),
                1e-3,
                2.0,
                0.0,
            )
        };
        let ukf_constructed = ukf(&constructed);
        let ukf_literal = ukf(&literal);
        assert_approx_eq!(
            ukf_constructed.mean_state[0],
            ukf_literal.mean_state[0],
            1e-15
        );
        assert_approx_eq!(
            ukf_constructed.mean_state[1],
            ukf_literal.mean_state[1],
            1e-15
        );
        assert_approx_eq!(ukf_constructed.mean_state[0], latitude_rad, 1e-15);
        assert_approx_eq!(ukf_constructed.mean_state[1], longitude_rad, 1e-15);
    }

    /// #286: bias injection clamps to physically plausible bounds.
    #[test]
    fn eskf_inject_clamps_biases_to_physical_bounds() {
        let mut eskf = ErrorStateKalmanFilter::new(
            &InitialState::default(),
            &IMU_BIASES,
            COVARIANCE_DIAGONAL.to_vec(),
            DMatrix::from_diagonal(&DVector::from_vec(PROCESS_NOISE_DIAGONAL.to_vec())),
        );
        eskf.error_state[9] = 5.0;
        eskf.error_state[11] = -5.0;
        eskf.error_state[12] = 1.0;
        eskf.error_state[14] = -1.0;
        eskf.inject_error_state();
        assert_approx_eq!(eskf.nominal_accel_bias[0], MAX_ACCEL_BIAS_MPS2, 1e-12);
        assert_approx_eq!(eskf.nominal_accel_bias[1], 0.0, 1e-12);
        assert_approx_eq!(eskf.nominal_accel_bias[2], -MAX_ACCEL_BIAS_MPS2, 1e-12);
        assert_approx_eq!(eskf.nominal_gyro_bias[0], MAX_GYRO_BIAS_RPS, 1e-12);
        assert_approx_eq!(eskf.nominal_gyro_bias[1], 0.0, 1e-12);
        assert_approx_eq!(eskf.nominal_gyro_bias[2], -MAX_GYRO_BIAS_RPS, 1e-12);
        // Error state still resets to zero after a clamped injection.
        assert!(eskf.error_state.iter().all(|v| *v == 0.0));
    }

    /// #286: finite-difference attitude columns carry the expected-measurement
    /// sensitivity (yaw-only for the mag model), not the tilt sensitivity of
    /// the raw sensor reading.
    ///
    /// The update always forms the residual as `z - h(x)` with `H = dh/dx`,
    /// so the attitude block must differentiate `h` (here: state yaw), whose
    /// rotation-vector derivative at level attitude is `[0, 0, 1]`. In
    /// particular the FD yaw column must equal the analytic `+1.0`, while the
    /// FD tilt columns are ~0.
    ///
    /// The analytic Jacobian used to return the raw sensor's tilt sensitivity,
    /// `dz/d(roll, pitch)`, in those columns -- which is why this filter
    /// overrides them at all: copying them into an error-state `H` lets the
    /// update feed tilt sensitivity back with the wrong sign whenever the tilt
    /// derivatives exceed 1 (e.g. at high pitch), amplifying the residual
    /// instead of nulling it. `magnetometer_yaw_jacobian` no longer does that
    /// (#305), so the two forms now agree here for a second reason as well.
    /// The override stays regardless: the Euler/rotation-vector mismatch it
    /// was written for is independent of that, as the high-pitch test below
    /// shows.
    #[test]
    fn fd_attitude_columns_match_analytic_at_level_attitude() {
        use crate::measurements::{MagnetometerYawMeasurement, MeasurementModel};

        let m = MagnetometerYawMeasurement {
            mag_x: 20.0,
            mag_y: 5.0,
            mag_z: -45.0,
            noise_std: 0.2,
            apply_declination: false,
            year: 2025,
            day_of_year: 1,
            is_enu: false, // NED fixture
        };
        // Level attitude with a non-zero yaw (exercises the yaw column).
        let nominal = DVector::from_vec(vec![0.7, -1.3, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3]);
        let q = UnitQuaternion::from_rotation_matrix(&Rotation3::from_euler_angles(0.0, 0.0, 0.3));
        let q_wxyz = nalgebra::Vector4::new(q.w, q.i, q.j, q.k);
        let analytic = m.get_jacobian(&nominal).unwrap();
        let fd = ErrorStateKalmanFilter::attitude_error_jacobian(&m, &q_wxyz, &nominal);
        // Yaw column agrees with the analytic +1.0 ...
        assert_approx_eq!(fd[(0, 2)], analytic[(0, 8)], 1e-6);
        assert_approx_eq!(fd[(0, 2)], 1.0, 1e-6);
        // ... while the tilt columns are ~0 (h is yaw-only).
        assert_approx_eq!(fd[(0, 0)], 0.0, 1e-6);
        assert_approx_eq!(fd[(0, 1)], 0.0, 1e-6);
    }

    /// #286: at high pitch the Euler and rotation-vector parameterisations
    /// genuinely differ, so the FD columns must differ from the analytic copy
    /// there. Locks in *why* the FD form exists (fails if someone reverts to
    /// copying the analytic attitude columns into the error-state H).
    ///
    /// Since #305 the analytic attitude block is the exact Euler-frame
    /// `dh/dx = [0, 0, 1]`, so what this now measures is the parameterisation
    /// gap alone rather than that plus the old tilt-sensitivity error: the FD
    /// tilt columns are non-zero at this attitude precisely because a
    /// body-frame rotation about x or y moves the *Euler* yaw when the vehicle
    /// is pitched 77 degrees up. That is the whole content of #286, and it is
    /// why the override cannot be dropped now that the analytic form is
    /// otherwise correct.
    #[test]
    fn fd_attitude_columns_differ_from_analytic_at_high_pitch() {
        use crate::measurements::{MagnetometerYawMeasurement, MeasurementModel};

        let m = MagnetometerYawMeasurement {
            mag_x: 20.0,
            mag_y: 5.0,
            mag_z: -45.0,
            noise_std: 0.2,
            apply_declination: false,
            year: 2025,
            day_of_year: 1,
            is_enu: false, // NED fixture
        };
        // Representative of the test dataset's mount: pitched up steeply.
        let nominal = DVector::from_vec(vec![0.7, -1.3, 100.0, 0.0, 0.0, 0.0, 0.16, -1.34, 0.18]);
        let q =
            UnitQuaternion::from_rotation_matrix(&Rotation3::from_euler_angles(0.16, -1.34, 0.18));
        let q_wxyz = nalgebra::Vector4::new(q.w, q.i, q.j, q.k);
        let analytic = m.get_jacobian(&nominal).unwrap();
        let fd = ErrorStateKalmanFilter::attitude_error_jacobian(&m, &q_wxyz, &nominal);
        let max_diff = (0..3)
            .map(|i| (fd[(0, i)] - analytic[(0, 6 + i)]).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_diff > 0.05,
            "FD and analytic attitude columns should differ at high pitch, max diff {max_diff}"
        );
    }
}
