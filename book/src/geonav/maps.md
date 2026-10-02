# Maps and Measurement Models

> **Experimental.** This page documents `strapdown-geonav` 0.1 (imported as `geonav`). See
> [Geophysical Navigation Overview](./overview.md) for how to run it.

A geophysical measurement compares two numbers: an **anomaly computed from a sensor reading**,
and the **anomaly read off a map** at the filter's estimated position. Everything below is
about how each of those is formed, and how the map's own error is absorbed by a bias state.

## Maps: `GeoMap`

`GeoMap` holds one regular latitude/longitude grid: a vector of latitudes, a vector of
longitudes (both in degrees, ascending), a matrix of values with one row per latitude and one
column per longitude, and a `GeophysicalMeasurementType` saying what the values are.

`GeoMap::load_geomap(path, map_type)` reads a NetCDF file with three variables:

| variable | shape | contents |
|---|---|---|
| `lat` | 1-D | latitudes, degrees |
| `lon` | 1-D | longitudes, degrees |
| `z` | `lat` × `lon` | anomaly: milligal for gravity, nanotesla for magnetic |

This is the layout GMT writes for its gridded datasets. The file does not say what kind of map
it is, so the caller supplies the type. A file that cannot be opened, lacks one of the three
variables, or whose `z` does not have `len(lat) × len(lon)` values is rejected with
`StrapdownError::MapLoad`.

### Map types and resolutions

`GeophysicalMeasurementType` is `Relief`, `Gravity` or `Magnetic`, each carrying a resolution
enum. The resolutions are the grid spacings GMT publishes its remote datasets at, and each
formats as GMT's code for it (`01d`, `30m`, ... `01m`):

| type | resolutions |
|---|---|
| `GravityResolution` | one degree; 30, 20, 15, 10, 6, 5, 4, 3, 2 and 1 arc-minutes |
| `MagneticResolution` | one degree; 30, 20, 15, 10, 6, 5, 4, 3 and 2 arc-minutes |
| `ReliefResolution` | also includes arc-second spacings; no measurement model uses relief |

The resolution is metadata. Interpolation uses the grid actually stored in the file, so a label
that disagrees with the file changes nothing numerically.

### Reading a value: bilinear interpolation

`GeoMap::get_point(lat, lon)` interpolates bilinearly between the four surrounding grid nodes
$(\varphi_1, \lambda_1)$, $(\varphi_2, \lambda_2)$:

$$
m(\varphi, \lambda) = \frac{
  q_{11}(\lambda_2-\lambda)(\varphi_2-\varphi) + q_{12}(\lambda_2-\lambda)(\varphi-\varphi_1) +
  q_{21}(\lambda-\lambda_1)(\varphi_2-\varphi) + q_{22}(\lambda-\lambda_1)(\varphi-\varphi_1)
}{(\lambda_2-\lambda_1)(\varphi_2-\varphi_1)}
$$

with $q_{ij}$ the value at $(\varphi_i, \lambda_j)$, and linear interpolation along an edge. A
non-finite coordinate is `StrapdownError::NonFinite`; a point outside the grid is
`StrapdownError::OutOfMapBounds`.

What happens when the estimate leaves the grid depends on the filter:

- **EKF and particle filter:** `OutOfMapBounds` is classed as recoverable, so the measurement is
  skipped with a warning naming the map bounds, and the run continues unaided by the map.
- **UKF:** the predicted measurement is evaluated at each sigma point through a method that
  cannot return an error, and an off-map point yields `NaN` there. The UKF checks for that before
  it touches the state: if any sigma point has no finite prediction, the update returns the
  recoverable `MeasurementUnavailable`, and the measurement is skipped with a warning as on the
  other filters. This happens a little sooner than on the EKF, since a sigma point can leave the
  map while the estimate itself is still on it.

Either way, size the map with room to spare around the track.

### The map gradient

The EKF and the particle filter's Kalman part need $\partial m / \partial \varphi$ and
$\partial m / \partial \lambda$. `GeoMap::get_gradient` takes central differences with a step of
$\epsilon = 10^{-6}$ degrees, falling back to a one-sided difference at the edge of the grid:

$$
\frac{\partial m}{\partial \varphi} \approx \frac{m(\varphi+\epsilon, \lambda) - m(\varphi-\epsilon, \lambda)}{2\epsilon},
\qquad
\frac{\partial m}{\partial \lambda} \approx \frac{m(\varphi, \lambda+\epsilon) - m(\varphi, \lambda-\epsilon)}{2\epsilon}.
$$

The result is per degree and is converted to per radian, the unit of the state. Because the
interpolant is piecewise bilinear, this is the slope of the cell the estimate is in.

## Measurement models

Both models implement core's `MeasurementModel` trait and are one-dimensional. Each predicts the
map value plus a bias, $h(\mathbf{x}) = m(\varphi, \lambda) + b$, and has
$R = \sigma^2$ from its `noise_std`. The Jacobian row is

$$
H = \begin{bmatrix} \dfrac{\partial m}{\partial \varphi} & \dfrac{\partial m}{\partial \lambda} & 0 & \cdots & 0 & 1 & 0 & \cdots \end{bmatrix}
$$

with the 1 in the column of that channel's bias state, if the filter carries one.

What makes them unusual is the *observation*. The sensor does not measure an anomaly; it
measures a total field. Turning that into an anomaly needs the vehicle's position, altitude,
velocity and the date, and those are only known once a filter is running. So the observation
is computed from the state the filter hands the model, at update time, not when the event
stream is built.

### Gravity: `GravityMeasurement`

The observed specific-force magnitude $g_\text{obs}$ is the norm of the record's
`grav_x`, `grav_y`, `grav_z` columns (m/s²). The anomaly, in milligal, is

$$
\Delta g = \left[\  g_\text{obs} - \gamma(\varphi, h) + E \ \right] \times 10^{5}
$$

where:

- $\gamma(\varphi, h) = \gamma_e \dfrac{1 + k \sin^2\varphi}{\sqrt{1 - e^2 \sin^2\varphi}} - 3.08\times10^{-6}\ h$
  is Somigliana normal gravity on the WGS84 ellipsoid with the free-air gradient
  (`earth::gravity`), evaluated at the observation height;
- $E = 2\Omega v_E \cos\varphi + \dfrac{v_N^2}{R_N + h} + \dfrac{v_E^2}{R_E + h}$ is the Eötvös
  correction (`earth::eotvos`): the down component of the Coriolis and transport-rate term of
  Groves eq. 5.54, with the transport rate of eq. 5.44. A gravimeter on a moving platform reads
  low by $E$, so it is added back;
- $10^5$ converts m/s² to milligal (`earth::MGAL_PER_M_PER_S2`).

Because $h$ is height above the ellipsoid, this is strictly the gravity disturbance rather than
the classical free-air anomaly, which is referred to the geoid. The two differ by about
$0.3086\ N$ mGal for a geoid undulation of $N$ metres: near-constant over a trajectory, and
absorbed by the bias state. This is `earth::gravity_anomaly`.

### Magnetic: `MagneticAnomalyMeasurement`

The observed total field $|\mathbf{B}_\text{obs}|$ is the norm of `mag_x`, `mag_y`, `mag_z`,
which are microtesla, scaled by 1000 into nanotesla. The anomaly is

$$
\Delta F = |\mathbf{B}\_\text{obs}| - F\_\text{WMM}(\varphi, \lambda, h, t)
$$

where $F_\text{WMM}$ is the World Magnetic Model's total intensity at the estimated position and
the record's date, from the `world_magnetic_model` crate, in nanotesla. Altitude is clamped to
the model's valid band of −1 km to 850 km. A position or date the model cannot serve is
`StrapdownError::ExternalModel`, which is recoverable: the measurement is skipped.

The total-field magnitude is used because it does not depend on the sensor's orientation, so
attitude error does not enter the observation.

### Both: `CombinedGeophysicalMeasurement`

When a run loads both maps and a record carries both readings, the two are emitted as a single
two-dimensional measurement, $[\Delta g, \Delta F]$, with a diagonal $R$: the two channels'
noises are taken as independent. For the filters that use the Jacobian, either half falling
off its map fails the whole measurement rather than applying the other half alone.

## Bias states

The map is not the truth. Grids are smoothed, referenced to a different datum or epoch, and a
magnetometer in a vehicle also sees the vehicle. Each channel therefore has a bias state $b$
in the map's unit, and $h(\mathbf{x}) = m + b$.

**UKF and EKF.** The biases are appended after the 15 navigation and IMU-bias states, gravity
first, then magnetic, then the barometric bias if it is estimated (the default; see
`--no-estimate-baro-bias`). Each is a random walk: initial value `gravity_bias` /
`magnetic_bias`, initial variance `*_bias_init_std`², and process-noise density
`*_bias_process_noise_std`² (a variance per second, so $Q = q\ \Delta t$). The
initial standard deviation defaults to the measurement noise; the random-walk rate defaults to
that prior spread over an hour, $\sigma_0/\sqrt{3600\ \text{s}}$.

`GeoBiasLayout` records where these states sit and how wide the filter's state is. Each
measurement carries a `BiasState` with both numbers and checks them against the state vector
it is handed, so a measurement built for one filter layout and given to another fails with
`StrapdownError::DimensionMismatch` instead of reading an IMU bias as a map bias.

**Particle filter.** Following Canciani & Raquet, each channel's bias is split into a
first-order Gauss-Markov temporal variation $V$ and a constant $c$, both linear states of the
filter's shared Kalman part, and the measurement selects $V + c$. See
[Rao-Blackwellized Particle Filter](../filters/rbpf.md).

## When measurements are emitted

`geonav::build_event_stream` builds the ordinary event stream with
`strapdown::messages::build_event_stream` (IMU, GNSS, barometer and magnetometer heading, with
their schedules and GNSS faults) and merges the geophysical events into it. Its geophysical
settings come in a `GeophysicalAiding` struct: the maps, the two noise values, the interval
between measurements, and the bias layout.

- With `geo_interval_s` unset, every record that carries the reading produces a measurement;
  with it set, one is produced at most every `geo_interval_s` seconds.
- A record with any `NaN` among `grav_*` (or `mag_*`) produces no measurement on that channel.
- At equal timestamps the geophysical measurement comes after the others.
- Geophysical measurements continue through a GNSS outage; that is the point of them.

## Where the maps come from

The Python tooling downloads them. `analyze preprocess --getmaps` (see
[Python Analysis Tooling](../development/analysis.md)) calls PyGMT for each trajectory, over its
bounding box padded by a fraction (`--buffer`, default 0.1) and a fixed margin
(`--margin-km`, default 5 km):

| file | PyGMT call | grid |
|---|---|---|
| `<stem>_gravity.nc` | `load_earth_free_air_anomaly(resolution="01m")` | 1 arc-minute free-air anomaly, mGal |
| `<stem>_magnetic.nc` | `load_earth_magnetic_anomaly(resolution="03m", data_source="wdmam")` | 3 arc-minute World Digital Magnetic Anomaly Map, nT |
| `<stem>_relief.nc` | `load_earth_relief(resolution="15s")` | 15 arc-second relief, for plotting only |

These names are what `strapdown-sim` looks for beside each input CSV. Note that the shipped
recipes label the magnetic map `two_minutes` while the file holds a 3 arc-minute grid; as above,
the label has no numerical effect. When preprocessing splits a recording into segments without
`--getmaps`, each segment is given a copy of its parent recording's maps, which necessarily
cover it. Downloading needs the GMT C library and network access; nothing else in the toolchain
does.

## The Python mirror of these models

`analysis/src/analysis/geostats.py` (`analyze geostats`) characterises a data set's geophysical
readings against its maps: per-channel bias, measurement noise, signal-to-noise ratio and the
de-correlation length along track. For those statistics to describe the quantity the filter
actually sees, its anomaly functions (`normal_gravity`, `eotvos`, `gravity_anomaly_mgal`,
`observed_field_nt`, `wmm_total_field_nt`) mirror `core/src/earth.rs` and `geonav/src/lib.rs`
term for term, and its `self_check()` asserts that the two implementations agree on known cases
before any statistic is computed.
