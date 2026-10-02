# State Representation

Three layers of state appear in this library: the nine-element navigation state that the
mechanization propagates, the fifteen-element layout every filter reports, and the extra bias
states a filter may append. This page defines each one, with units, and shows how they reach
the output CSV.

## The navigation state: `StrapdownState`

`StrapdownState` is what [the mechanization](./concepts.md) propagates:

| Field | Quantity | Units |
| --- | --- | --- |
| `latitude` | Geodetic latitude | **radians** |
| `longitude` | Geodetic longitude | **radians** |
| `altitude` | Height above the WGS84 ellipsoid, positive up in both frames | m |
| `velocity_north` | Earth-relative velocity, north component | m/s |
| `velocity_east` | Earth-relative velocity, east component | m/s |
| `velocity_vertical` | Earth-relative velocity, vertical component: positive down in NED, positive up in ENU | m/s |
| `attitude` | Body-to-navigation rotation $\mathbf C_b^n$ (`nalgebra::Rotation3`) | -- |
| `is_enu` | Frame flag: `false` for NED (the default), `true` for ENU | -- |

Latitude and longitude are radians inside the library and degrees in every CSV file, input and
output. As a flat vector (`Vec<f64>` or `DVector<f64>` via `From`) the state is the nine
elements

$$
[L,\ \lambda,\ h,\ v_N,\ v_E,\ v_{\text{vert}},\ \phi,\ \theta,\ \psi]
$$

with the angles in radians and the attitude expressed as roll, pitch and yaw
([Coordinate Frames](./coordinate-frames.md) gives the Euler convention). The vector carries no
frame flag: converting nine numbers back with `TryFrom` produces an NED state.

## The fifteen-element layout

Every filter reports its estimate in one common layout, the navigation state followed by the
two IMU bias vectors:

| Index | State | Units |
| --- | --- | --- |
| 0, 1 | Latitude, longitude | rad |
| 2 | Altitude | m |
| 3, 4, 5 | Velocity north, east, vertical | m/s |
| 6, 7, 8 | Roll, pitch, yaw | rad |
| 9, 10, 11 | Accelerometer bias $\mathbf b_a$, body $x, y, z$ | m/s² |
| 12, 13, 14 | Gyroscope bias $\mathbf b_g$, body $x, y, z$ | rad/s |

The width is `sim::NAVIGATION_STATES` (15), and the crate root's `ERROR_STATE_DIMENSION` (15)
is the same count for the error-state form. `sim::DEFAULT_PROCESS_NOISE_DENSITY` has one entry
per row; see [The Navigation Model](./concepts.md#process-noise-is-a-spectral-density).

The IMU biases are modelled as random walks. The Kalman filters subtract their bias estimates
from each IMU sample before mechanizing it.

Extra states, when a filter carries them, are appended after index 14.

## What each filter carries

### ESKF (the `cl` default)

The error-state Kalman filter keeps two things apart:

- a **nominal state**: position (radians and metres), velocity, attitude as a unit
  quaternion, the two IMU biases, and the barometric bias when it is carried;
- an **error state** $\delta\mathbf x$ and its covariance: 15 elements, or 16 with the
  barometric bias.

| Index | Error state | Units |
| --- | --- | --- |
| 0, 1, 2 | $\delta L$, $\delta\lambda$, $\delta h$ | rad, rad, m |
| 3, 4, 5 | $\delta v_N$, $\delta v_E$, $\delta v_{\text{vert}}$ | m/s |
| 6, 7, 8 | $\delta\boldsymbol\theta$: attitude error as a body-frame rotation vector | rad |
| 9, 10, 11 | $\delta\mathbf b_a$ | m/s² |
| 12, 13, 14 | $\delta\mathbf b_g$ | rad/s |
| 15 | $\delta b_{\text{baro}}$ (only when the barometric bias is carried) | m |

Only the error state is estimated. After each measurement update it is injected into the
nominal state and reset to zero. The attitude correction is composed on the right,
$\mathbf q \leftarrow \mathbf q \otimes \mathrm{Exp}(\delta\boldsymbol\theta)$, which is what
makes $\delta\boldsymbol\theta$ a rotation of the *body* frame, not a Euler-angle difference.
Position, velocity and the biases are corrected by addition. When reporting, the ESKF
converts its nominal state into the fifteen-element layout above, plus the barometric bias.

**How many states, exactly.** The library's `EskfConfig` defaults to `estimate_baro_bias =
false`, which gives a 15-element error state. `strapdown-sim cl` and a `[closed_loop]`
configuration section default to `estimate_baro_bias = true`, so **the ESKF that `cl` runs has
16 error states**. `--no-estimate-baro-bias` (or `estimate_baro_bias = false`) takes it back
to 15. The ESKF has no geophysical arm, so it never carries map biases.

### EKF and UKF (`cl --filter ekf|ukf`)

These carry the **full state**, not an error state: a mean vector in the fifteen-element
layout and its covariance, with extra states appended in this order:

1. map biases from `strapdown-geonav`, gravity first, then magnetic, one state per loaded map
   (indices 15 and 16 when both maps are loaded);
2. the barometric bias, last, when it is carried.

The barometric bias goes after the map biases because geonav numbers its map-bias states from
15; putting the barometer at 15 would collide with a gravity bias. Its index is therefore not
a constant, and `UkfConfig::baro_bias_index` and `EkfConfig::baro_bias_index` compute it.
Without maps, `cl --filter ekf` and `cl --filter ukf` carry 16 states. With both maps and the
barometric bias, they carry 18, the barometric bias at index 17.

The library EKF can also run with `use_biases = false`, which makes it a nine-state filter
with no IMU-bias states; the simulator does not use that form.

### RBPF (`pf`)

The Rao-Blackwellized particle filter is structured differently inside. Its particles sample
the horizontal position error only; one Kalman filter shared by all particles holds the
altitude error, velocity error, navigation-frame tilt, the barometer-aiding loop's two states,
and for each map a time-varying part $V$ and a constant part $c$ of the map bias. It carries
**no IMU-bias states**. [Rao-Blackwellized Particle Filter](../filters/rbpf.md) gives the full
partition.

It reports the same layout as the Kalman filters, `[9 navigation, b_a(3), b_g(3)]`, followed
by one total map bias $V + c$ per map. The six IMU-bias rows, and their variances, are zero.
It has no barometric bias state in the sense of the Kalman filters: its barometer error lives
inside the aiding loop, and its output leaves the `baro_bias` column empty.

## The extra bias states

| State | Units | Carried by | Default |
| --- | --- | --- | --- |
| Barometric bias | m | ESKF, EKF, UKF | **On** in `strapdown-sim cl` and `[closed_loop]`; off in the library's `EskfConfig`, `EkfConfig`, `UkfConfig` |
| Gravity anomaly map bias | mGal | EKF, UKF, RBPF, when a gravity map is loaded (`--geo`, experimental) | -- |
| Magnetic anomaly map bias | nT | EKF, UKF, RBPF, when a magnetic map is loaded (`--geo`, experimental) | -- |

**Barometric bias.** A barometer measures height through pressure, and its reference pressure
drifts with the weather. A filter that treats the reading as unbiased pushes that drift into
altitude. The bias state is a random walk that starts at zero with variance
`sim::INITIAL_BARO_BIAS_VARIANCE_M2` $= (8.3\ \text{m})^2 = 68.89$ m² and grows at
`sim::BARO_BIAS_PROCESS_NOISE_M2_PER_S` $= (8.3\ \text{m})^2$ per hour, both derived from about
one hectopascal of reference drift per hour. The barometer measurement model reads it through
the index the filter declares (`AidingConfig::baro_bias_index`), which `strapdown-sim` sets
from the filter configuration. A library caller who builds a 16-state filter by hand must set
that index too, or nothing observes the state.

**Map biases** are experimental and documented with [Geophysical
Navigation](../geonav/overview.md). Their layout is described by `strapdown-geonav`'s
`GeoBiasLayout`; `sim::ExtraStateLayout` is its counterpart on the `strapdown-core` side, which
the simulator uses to label the output columns.

## How the state reaches the output

Every mode writes one `NavigationResult` row per IMU record. The state appears as follows
([Output Format](./output-format.md) lists every column):

| Columns | Content | Units |
| --- | --- | --- |
| `latitude`, `longitude` | Estimate | **degrees** |
| `altitude`, `velocity_north`, `velocity_east`, `velocity_vertical` | Estimate | m, m/s |
| `roll`, `pitch`, `yaw` | Estimate | rad |
| `acc_bias_x/y/z`, `gyro_bias_x/y/z` | IMU bias estimates; zero for `dr` and `pf` | m/s², rad/s |
| `latitude_cov`, `longitude_cov` | Variance | **rad²** |
| `altitude_cov` | Variance | m² |
| `latitude_longitude_cov`, `latitude_altitude_cov`, `longitude_altitude_cov` | The off-diagonal terms of the position block | rad², rad·m, rad·m |
| `velocity_n_cov`, `velocity_e_cov`, `velocity_v_cov`, `roll_cov`, `pitch_cov`, `yaw_cov`, `acc_bias_*_cov`, `gyro_bias_*_cov` | Variances (the covariance diagonal) | the state's units, squared |
| `gravity_bias`, `magnetic_bias`, `baro_bias` and their `_cov` | Extra bias states and their variances | mGal, nT, m |

Points to watch when reading these columns:

- **Latitude and longitude are degrees, their variances radians squared.** Convert the
  position error to radians before comparing it with the variance, not the variance to degrees.
  With $P_{LL}$ = `latitude_cov` and $P_{\lambda\lambda}$ = `longitude_cov`, the north and
  east standard deviations in metres are approximately $\sigma_N = \sqrt{P_{LL}} (R_N + h)$
  and $\sigma_E = \sqrt{P_{\lambda\lambda}} (R_E + h) \cos L$, with the radii of
  [The Navigation Model](./concepts.md#the-earth-model).
- **Only the position block is written in full.** The other states carry their variances
  only. The three position off-diagonals are there so that a position NEES,
  $\mathbf e^\top \mathbf P^{-1} \mathbf e$, can be computed from the output.
- **The ESKF's attitude variances are of its body-frame attitude error** $\delta\boldsymbol\theta$.
  They equal Euler-angle variances only at level attitude.
- **An empty cell means "not carried", not zero.** The three extra-bias pairs are optional:
  a run without a gravity map leaves `gravity_bias` empty, and a run that estimated a bias of
  exactly zero writes `0`. The columns are always present, so every result file has the same
  40 columns.
- **Dead reckoning has no covariance.** `dr` runs no filter, so it writes `NaN` in all
  eighteen covariance columns, zeros in the IMU-bias columns, and leaves the extra-bias
  columns empty.

On the [Quick Start](../quick-start.md)'s ESKF run, for example, the first row has
`baro_bias_cov` = 68.89, the initial variance above, and `latitude_cov` = 2.467 × 10⁻¹² rad²:
the default 10 m initial horizontal standard deviation
(`sim::DEFAULT_INITIAL_POSITION_UNCERTAINTY_M`), converted to radians and squared.
