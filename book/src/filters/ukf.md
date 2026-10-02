# Unscented Kalman Filter (UKF)

`kalman::UnscentedKalmanFilter` propagates a deterministic set of **sigma points** through the
nonlinear mechanization and the measurement models instead of linearizing them (Julier and
Uhlmann). It needs no transition Jacobian, and its update uses only each model's
`get_expected_measurement`. Run it with `strapdown-sim cl --filter ukf`.

## State

The full state vector of the [shared layout](./kalman.md#state-layout-and-units): nine
navigation states, the six IMU biases, then any extra states (`UkfConfig::other_states`, such as
geophysical map biases) and, last, the barometric bias when `estimate_baro_bias` is set.
`initialize_ukf` always builds the fifteen-state core; the constructor accepts an empty bias
slice for a nine-state filter, in which case no bias correction is applied.

## Sigma points and weights

For an $n$-element state the scaled unscented transform uses $2n+1$ points and

$$
\lambda = \alpha^2 (n + \kappa) - n ,
$$

$$
w^{(m)}_0 = \frac{\lambda}{n+\lambda}, \qquad
w^{(c)}_0 = \frac{\lambda}{n+\lambda} + (1 - \alpha^2 + \beta), \qquad
w^{(m)}_i = w^{(c)}_i = \frac{1}{2(n+\lambda)} .
$$

Point 0 is the mean; points $i$ and $i+n$ are the mean plus and minus column $i$ of a matrix
square root of $(n+\lambda)P$ (`linalg::matrix_square_root`: an equilibrated Cholesky, with
plain, jittered and eigenvalue fallbacks for a matrix that is not positive definite).

The attitude rows are not treated as a vector. Adding a rotation vector to an Euler triple is
not the same rotation as composing it, so for attitude each point is
$R_\text{mean}\operatorname{Exp}(\pm\text{column})$, re-expressed on the mean's branch of the
Euler angles. The same idea runs through the rest of the filter (#371):

- **Predict.** Each sigma point is bias-corrected with *its own* bias hypothesis
  ($\Delta v - b_a\Delta t$, $\Delta\theta - b_g\Delta t$) and mechanized. The mean of every
  non-attitude row is the weighted sum. The mean attitude is formed in the tangent space at
  propagated sigma point 0: the residuals $\operatorname{Log}(R_0^\top R_i)$ are summed with the
  mean weights and mapped back. The covariance uses tangent-space attitude residuals, and
  $q\ \Delta t$ is added to it.
- **Update.** Every sigma point is passed through `get_expected_measurement`, giving
  $\hat z$, $S$ and the state-measurement cross covariance (again with tangent-space attitude
  residuals). After the gate, $K = P_{xz} S^{-1}$; the attitude part of $K\nu$ is a rotation
  vector and is composed onto the mean attitude, the rest is added. The covariance becomes
  $P - K S K^\top$ and is then transported onto the new attitude's tangent space with the same
  right Jacobian the [ESKF](./eskf.md#update-injection-and-reset) uses (#398).

## Defaults: $\alpha = 0.1$, $\beta = 2$, $\kappa = 0$

| parameter | default | role |
| --- | --- | --- |
| `alpha` | `0.1` (`sim::DEFAULT_UKF_ALPHA`) | spread of the points about the mean |
| `beta` | `2.0` | prior knowledge of the distribution; 2 is optimal for a Gaussian |
| `kappa` | `0.0` | secondary spread |

They are set by `UkfConfig::{ukf_alpha, ukf_beta, ukf_kappa}` (each an `Option`, `None` taking
the default), by `[closed_loop]`'s `ukf_alpha`, `ukf_beta` and `ukf_kappa` in a configuration
file, or by `--ukf-alpha`, `--ukf-beta` and `--ukf-kappa` on `strapdown-sim cl`. All three routes
default to the same values.

### Why not the textbook alpha of 0.001

With $\kappa = 0$ the weights are $w_0 = 1 - \alpha^{-2}$ and $w_i = 1/(2n\alpha^2)$. They sum to
exactly 1 for any $\alpha$, but they cancel only in exact arithmetic. At $\alpha = 10^{-3}$ and
$n = 16$ the mean is formed as $-999{,}999\ x_0 + \sum 31{,}250\ x_i$: every term is about
$10^6$ times the answer, so six of `f64`'s sixteen significant digits go to cancellation on
every propagation step. At $\alpha = 0.1$, $w_0 = -99$ and the cancellation costs two digits.

That was the cause of #399, where the accuracy baseline moved under source changes that should
have been invisible. The investigation recorded in `ClosedLoopConfig::ukf_alpha`'s
documentation perturbed a single WGS84 constant by one unit in the last place and took the
worst response over the gated metrics:

| configuration | worst response to one ulp |
| --- | ---: |
| `alpha = 1e-3`, plain Cholesky (before #399) | 9.77% |
| `alpha = 1e-3`, equilibrated Cholesky | 3.57% |
| `alpha = 0.1`, plain Cholesky | 0.000297% |
| `alpha = 0.1`, equilibrated Cholesky (shipped) | 0.000140% |

`alpha` accounts for almost all of it; the equilibrated Cholesky that landed in the same change
for a further factor of two. Over the same perturbations the EKF, ESKF and RBPF, none of which
forms a weighted sigma-point mean, moved by less than $10^{-6}$%. The value is 0.1 rather than
1.0 because $\alpha$ still has a job: it keeps the points near the mean (0.4 sigma at 0.1), so
the transform samples the local nonlinearity rather than a shell several sigma out.

### The conditioning tests

`core/tests/ukf_conditioning.rs` holds both ends of that:

- `one_ulp_of_input_cannot_move_the_solution_by_a_centimetre` runs the UKF twice through a
  180 s, 50 Hz synthetic scenario with GNSS duty-cycled 60 s off and 60 s on (no start phase,
  so it opens with the outage), the second time with the initial latitude moved by one ulp, and requires the final positions to agree within 1 cm. The test's
  own documentation records 1.76 m at the old $\alpha$ and 0.18 mm at 0.1.
- `the_shipped_sigma_point_weights_do_not_cancel_away_the_mantissa` checks that the default
  weights spend no more than 2.5 of `f64`'s digits on cancellation, so lowering $\alpha$ again
  fails a test that names its own cause.

```bash
cargo test -p strapdown-core --test ukf_conditioning
```

## Construction

```rust,ignore
{{#include ../../../core/examples/kalman_filters.rs:ukf_new}}
```

The arguments are an `InitialState`, the six initial bias estimates, optional extra states, the
covariance diagonal (one entry per state, in the units of the
[shared layout](./kalman.md#state-layout-and-units)), the process-noise density $q$ as a
matrix, and $\alpha$, $\beta$, $\kappa$. `sim::initialize_ukf(&record, UkfConfig)` builds the
same filter from a `TestDataRecord`; [Kalman Filters](./kalman.md#the-siminitialize_-helpers)
lists the config fields and the $P_0$ it derives.

The same run as a configuration file (verified with `strapdown-sim --config` on a `syn`
trajectory):

```toml
input = "synthetic.csv"
output = "cl_ukf_config.csv"
mode = "closed-loop"

[closed_loop]
filter = "ukf"
ukf_alpha = 0.1
ukf_beta = 2.0
ukf_kappa = 0.0
```

## Things to know

- The update calls `get_expected_measurement` on all $2n+1$ points and never `get_jacobian`,
  so a model's Jacobian does not affect this filter. `MeasurementModel` still requires one,
  because the other filters use it.
- Each predict and each update factorizes the covariance once and evaluates the mechanization
  or the measurement model at all $2n+1$ points. No timing comparison is published for this
  crate.
- The reported Euler angles are on the principal branch after both predict and update.

## References

- Julier, S. J. and Uhlmann, J. K., "Unscented Filtering and Nonlinear Estimation",
  *Proceedings of the IEEE* 92(3), 2004.
- `UnscentedKalmanFilter` in `core/src/kalman.rs`, and `ClosedLoopConfig::ukf_alpha` in
  `core/src/sim.rs` for the full #399 record.
