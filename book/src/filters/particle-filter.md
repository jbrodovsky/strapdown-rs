# Particle Filters

## Overview

This document outlines the design for a particle filter-based Inertial Navigation System (INS) following the architecture established in the UKF implementation (`kalman.rs`). The design addresses vertical channel instability through two alternative approaches: third-order damping and 2.5D navigation.

## Executive Summary: Key Architectural Decisions

### 1. Custom Forward Propagation Function Required

**Critical insight**: For 2.5D navigation, we **cannot reuse** the standard `forward()` function from `lib.rs`. A new `forward_2_5d()` function must be implemented in `particle.rs`.

**Why?**
- Standard strapdown mechanization (Groves Eq 5.54) fully integrates vertical acceleration into vertical velocity
- This creates deterministic coupling: `v_v(t+dt) = v_v(t) + ∫[a_z + g - Coriolis] dt`
- In 2.5D navigation, we want vertical velocity to be a **random walk** constrained by measurements, not deterministically coupled to accelerometer readings

**Solution**: `forward_2_5d()` implements:
- **Full strapdown** for: attitude, horizontal velocities (v_n, v_e), horizontal position (lat, lon)
- **Simplified** for: vertical velocity (random walk), altitude (integration of v_v only)

### 2. Vertical Channel Treatment

Two approaches are provided:

| Approach | States | Vertical Velocity | Complexity | Recommended |
|----------|--------|-------------------|------------|-------------|
| **2.5D Simplified** | 15 | Random walk with large process noise | Low | **Yes** - Start here |
| Third-Order Damping | 17 | Damped feedback control | High | Optional - if 2.5D insufficient |

### 3. Implementation Priority

1. First: Implement `forward_2_5d()` - this is the foundation
2. Second: Implement particle filter with 2.5D mode
3. Later: Optionally add third-order damping if needed

## State Representation

### Standard 15-State Model

Following the UKF implementation, the baseline particle filter uses a 15-state model:

**Navigation states (9):**
- Position: `[latitude, longitude, altitude]` (rad, rad, m)
- Velocity: `[v_north, v_east, v_vertical]` (m/s)
- Attitude: `[roll, pitch, yaw]` (rad) - represented internally as DCM

**Bias states (6):**
- Accelerometer biases: `[b_ax, b_ay, b_az]` (m/s²)
- Gyroscope biases: `[b_gx, b_gy, b_gz]` (rad/s)

### Extended State for Vertical Channel Damping

For third-order vertical channel damping, add **optional states**:
- Altitude error estimate: `δh` (m)
- Altitude error rate: `δh_dot` (m/s)

Total state dimension: **15 + 2 = 17 states** (when using damping)

**Implementation note**: Use `Particle.other_states: Option<DVector<f64>>` for the damping states.

## Vertical Channel Approaches

### Approach A: Third-Order Vertical Channel Damping

**Motivation**: The vertical channel in INS is inherently unstable due to coupling between altitude and vertical velocity errors. Third-order damping provides feedback to stabilize this channel.

**State augmentation:**
```
x = [lat, lon, h, v_n, v_e, v_v, φ, θ, ψ, b_a, b_g, δh, δh_dot]
    └─────────────────────────────────────┘  └─────┘  └────────┘
           15 standard states                 biases    damping
```

**Dynamics model:**
- Altitude error dynamics: `δh_dot = δv_v`
- Altitude error rate dynamics: `δh_ddot = -k₁·δh - k₂·δh_dot + w_h`
  - `k₁`, `k₂`: Damping coefficients (tunable)
  - `w_h`: Process noise on altitude error

**Feedback mechanism:**
During propagation (`predict` step):
1. Propagate particles through standard strapdown equations
2. Update altitude using damping feedback:
   ```
   h_corrected = h_propagated - k_fb · δh
   v_v_corrected = v_v_propagated - k_fb · δh_dot
   ```
3. During measurement update with GPS altitude:
   - Innovation: `z - h_expected` updates both `h` and `δh`
   - Damping states receive weighted update based on particle importance

**Pros:**
- Theoretically rigorous approach based on observability analysis
- Provides smooth vertical channel behavior
- Well-suited for long GNSS outages

**Cons:**
- Additional states increase computational cost
- Requires careful tuning of damping coefficients
- Increased complexity in implementation

### Approach B: 2.5D Navigation (Simplified Vertical Treatment)

**Motivation**: Treat vertical acceleration as primarily noise-driven rather than deterministic, simplifying the vertical channel dynamics. This is the recommended approach based on past success.

**State representation:**
```
x = [lat, lon, h, v_n, v_e, v_v, φ, θ, ψ, b_a, b_g]
    └────────────────────────────────────────────┘
              Standard 15 states only
```

**Modified dynamics:**
- **Horizontal navigation**: Full 6-DOF strapdown mechanization for position, velocity, and attitude
- **Vertical channel**: Simplified treatment:
  - Altitude propagation: `h(t+dt) = h(t) + v_v(t)·dt`
  - Vertical velocity: Integrate vertical acceleration with **high process noise**
  - Vertical acceleration noise: Model as white noise with large variance
    ```
    Q_vertical = [Q_h, Q_v_v] where Q_v_v >> Q_v_n, Q_v_e
    ```

**Process noise tuning:**
- Position (lat/lon): `1e-6` (tight)
- Altitude: `1e-3` to `1e-2` (loose, allows drift)
- Horizontal velocity: `1e-3`
- Vertical velocity: `1e-2` to `1e-1` (very loose, absorbs vertical uncertainty)
- Biases: `1e-6` to `1e-8` (standard)

**Measurement handling:**
- GPS altitude measurements directly constrain `h` with measurement noise
- Barometric altitude (if available) provides additional vertical constraint
- Vertical velocity measurements (if available) constrain `v_v`
- During GNSS outages: Altitude drifts according to high process noise

**Pros:**
- Simpler implementation (no additional states)
- Computationally efficient
- Leverages particle filter's ability to handle non-Gaussian distributions
- Works well with intermittent GPS
- Avoids complex tuning of damping coefficients

**Cons:**
- Less theoretically rigorous
- Vertical channel accumulates uncertainty faster during outages
- Requires careful process noise tuning to balance stability and responsiveness

## Particle Filter Architecture

### Core Components

#### Particle struct (already defined in `particle.rs`)
```rust
pub struct Particle {
    pub nav_state: StrapdownState,        // 9 nav states
    pub accel_bias: DVector<f64>,         // 3 accel biases
    pub gyro_bias: DVector<f64>,          // 3 gyro biases
    pub other_states: Option<DVector<f64>>, // Optional: damping states
    pub state_size: usize,                // Total dimension
    pub weight: f64,                      // Importance weight
}
```

#### ParticleFilter struct
```rust
pub struct ParticleFilter {
    particles: Vec<Particle>,
    num_particles: usize,
    process_noise: ProcessNoise,
    resampling_strategy: ResamplingStrategy,
    averaging_strategy: AveragingStrategy,
    effective_particle_threshold: f64,  // e.g., 0.5 * num_particles
    rng: StdRng,  // Seeded RNG for reproducibility
    is_enu: bool,

    // Vertical channel specific
    vertical_channel_mode: VerticalChannelMode,  // ThirdOrder or Simplified
    damping_coefficients: Option<(f64, f64)>,    // (k1, k2) if using damping
}
```

#### ProcessNoise struct
```rust
pub struct ProcessNoise {
    pub position_std: Vector3<f64>,       // [σ_lat, σ_lon, σ_h]
    pub velocity_std: Vector3<f64>,       // [σ_vn, σ_ve, σ_vv]
    pub attitude_std: Vector3<f64>,       // [σ_φ, σ_θ, σ_ψ]
    pub accel_bias_std: Vector3<f64>,     // [σ_ba_x, σ_ba_y, σ_ba_z]
    pub gyro_bias_std: Vector3<f64>,      // [σ_bg_x, σ_bg_y, σ_bg_z]
    pub damping_states_std: Option<Vector2<f64>>,  // [σ_δh, σ_δh_dot]
}
```

#### Enums
```rust
pub enum VerticalChannelMode {
    ThirdOrderDamping { k1: f64, k2: f64 },
    Simplified,  // 2.5D navigation
}

pub enum ResamplingStrategy {
    Systematic,
    Stratified,
    Residual,
    Multinomial,
}

pub enum AveragingStrategy {
    WeightedMean,        // Standard weighted average
    MaximumWeight,       // Use highest weight particle
    MeanWithTrimming,    // Remove outliers then average
}
```

### Core Algorithm

#### Initialization
```rust
impl ParticleFilter {
    pub fn new(
        initial_state: InitialState,
        imu_biases: Vec<f64>,
        covariance_diagonal: Vec<f64>,
        process_noise: ProcessNoise,
        num_particles: usize,
        vertical_mode: VerticalChannelMode,
        seed: Option<u64>,
    ) -> Self;
}
```

**Initialization steps:**
1. Create `num_particles` particles from initial state
2. Sample each particle's state from `N(μ₀, P₀)` where:
   - `μ₀` = initial mean state
   - `P₀` = diag(covariance_diagonal)
3. Initialize all weights to `1.0 / num_particles`
4. Seed RNG for reproducibility

#### Predict Step

**Key architectural decision**: For 2.5D navigation, we **cannot** use the standard `forward()` function from `lib.rs`. The standard strapdown mechanization fully integrates vertical acceleration to update vertical velocity (Equation 5.54 from Groves), which creates deterministic coupling we want to avoid in 2.5D.

**Why a new forward function is needed:**

The standard `forward()` computes vertical velocity as:
```rust
// velocity_update() in lib.rs, line 532
velocity + (specific_force + gravity - r * (transport_rate + 2.0 * rotation_rate) * velocity) * dt
```

This tightly couples vertical acceleration to vertical velocity. In 2.5D navigation, we want:
- **Horizontal dynamics**: Full strapdown (deterministic acceleration → velocity → position)
- **Vertical dynamics**: Decoupled or weakly coupled (vertical velocity as random walk, altitude from velocity integration only)

**Solution**: Implement `forward_2_5d()` that modifies vertical channel propagation.

#### Update Step
```rust
impl NavigationFilter for ParticleFilter {
    fn update<M: MeasurementModel + ?Sized>(&mut self, measurement: &M) {
        // 1. Compute weights based on measurement likelihood
        for particle in &mut self.particles {
            let predicted_meas = measurement.get_expected_measurement(
                &particle.to_state_vector()
            );
            let innovation = measurement.get_vector() - predicted_meas;
            let meas_cov = measurement.get_noise();

            // Likelihood: p(z|x) ∝ exp(-0.5 * innovation^T * R^-1 * innovation)
            let likelihood = self.compute_likelihood(&innovation, &meas_cov);
            particle.weight *= likelihood;
        }

        // 2. Normalize weights
        self.normalize_weights();

        // 3. Check effective particle count
        let n_eff = self.effective_particle_count();

        // 4. Resample if needed
        if n_eff < self.effective_particle_threshold {
            self.resample();
        }
    }
}
```

#### Resampling

Systematic resampling is the recommended strategy:

```rust
fn systematic_resample(&mut self) {
    let n = self.num_particles;
    let mut new_particles = Vec::with_capacity(n);

    // Cumulative sum of weights
    let cumsum: Vec<f64> = self.particles.iter()
        .scan(0.0, |sum, p| { *sum += p.weight; Some(*sum) })
        .collect();

    // Systematic sampling
    let step = 1.0 / n as f64;
    let start = self.rng.gen_range(0.0..step);

    for i in 0..n {
        let u = start + i as f64 * step;
        let idx = cumsum.iter().position(|&cs| cs >= u).unwrap_or(n - 1);
        let mut new_particle = self.particles[idx].clone();
        new_particle.weight = 1.0 / n as f64;
        new_particles.push(new_particle);
    }

    self.particles = new_particles;
}
```

#### State Estimation

```rust
fn weighted_mean_estimate(&self) -> DVector<f64> {
    let mut mean_state = DVector::zeros(self.particles[0].state_size);

    for particle in &self.particles {
        let state_vec = particle.to_state_vector();
        mean_state += particle.weight * state_vec;
    }

    // Special handling for circular quantities (angles)
    self.wrap_angles(&mut mean_state);

    mean_state
}
```

## The forward_2_5d() Function

This function implements modified strapdown mechanization for 2.5D navigation. It should be added to `particle.rs` as a module-level function.

### Mathematical Formulation

**Full strapdown components (unchanged from Groves Ch 5.4-5.5):**
1. **Attitude update** (Equation 5.46):
   ```
   C(t+dt) = C(t) * [I + Ω_ib * dt] - [Ω_ie + Ω_el] * C(t) * dt
   ```
   - Full implementation using gyros, transport rate, Earth rate

2. **Horizontal velocity update** (Equation 5.54, horizontal components only):
   ```
   v_n(t+dt) = v_n(t) + [f_n + g_n - Coriolis_n - transport_n] * dt
   v_e(t+dt) = v_e(t) + [f_e + g_e - Coriolis_e - transport_e] * dt
   ```
   - Full mechanization for northward and eastward velocities

3. **Horizontal position update** (Equation 5.56, lat/lon only):
   ```
   lat(t+dt) = lat(t) + [v_n / (R_N + h)] * dt
   lon(t+dt) = lon(t) + [v_e / ((R_E + h) * cos(lat))] * dt
   ```

**Simplified vertical components (2.5D modification):**

4. **Vertical velocity**: Random walk model (NOT integrated from acceleration)
   ```
   v_v(t+dt) = v_v(t) + w_v * sqrt(dt)
   ```
   - `w_v ~ N(0, σ_vv²)` - large process noise
   - **No deterministic integration of vertical specific force**
   - Vertical velocity drifts according to process noise only

5. **Altitude update**: Simple integration of vertical velocity
   ```
   h(t+dt) = h(t) + v_v(t) * dt
   ```
   - Uses trapezoidal rule: `h(t+dt) = h(t) + 0.5 * [v_v(t) + v_v(t+dt)] * dt`

### Implementation Details

```rust
/// Modified forward propagation for 2.5D navigation.
///
/// Implements full strapdown mechanization for horizontal navigation (position,
/// velocity, attitude) while treating the vertical channel with simplified dynamics.
/// Vertical acceleration is NOT integrated into vertical velocity; instead, vertical
/// velocity follows a random walk model constrained by measurements and process noise.
///
/// # Arguments
/// * `state` - Mutable reference to the navigation state
/// * `imu_data` - Bias-corrected IMU measurements
/// * `dt` - Time step in seconds
///
/// # Mathematical Basis
/// - Horizontal: Full Groves Equations 5.46, 5.54, 5.56
/// - Vertical: Simplified dynamics, vertical velocity as random walk
pub fn forward_2_5d(state: &mut StrapdownState, imu_data: IMUData, dt: f64) {
    // See full implementation in source code
}
```

### Key Differences from Standard forward()

| Component | Standard `forward()` | `forward_2_5d()` |
|-----------|---------------------|------------------|
| Attitude | Full mechanization (Eq 5.46) | **Same** - Full mechanization |
| Specific force transform | Full (Eq 5.47) | **Same** - Full transform |
| Horizontal velocity | Full mechanization (Eq 5.54) | **Same** - Full mechanization |
| **Vertical velocity** | **Integrated from vertical accel** | **NOT integrated - random walk** |
| Horizontal position | Full mechanization (Eq 5.56) | **Same** - Full mechanization |
| Altitude | Integrated from vertical velocity | **Same** - Simple integration |

**Critical insight**: The ONLY difference is that `v_v(t+dt) ≈ v_v(t)` in the propagation step. Changes to vertical velocity come entirely from:
1. Process noise injection (large `σ_vv`)
2. Measurement updates (GPS altitude/velocity)

This makes vertical velocity behave like a "free parameter" that is constrained by measurements rather than deterministic dynamics.

## Integration with Existing Codebase

### Mirroring UKF Structure

The particle filter implements the same `NavigationFilter` trait as UKF:
```rust
pub trait NavigationFilter {
    fn predict(&mut self, imu_data: IMUData, dt: f64);
    fn update<M: MeasurementModel + ?Sized>(&mut self, measurement: &M);
    fn get_estimate(&self) -> DVector<f64>;
    fn get_certainty(&self) -> DMatrix<f64>;  // Empirical covariance from particles
}
```

### Simulation Integration

In `sim.rs`, add particle filter option to `closed_loop` function:
```rust
pub enum FilterType {
    UKF { alpha: f64, beta: f64, kappa: f64 },
    ParticleFilter {
        num_particles: usize,
        vertical_mode: VerticalChannelMode,
        resampling: ResamplingStrategy,
    },
}
```

## Testing Strategy

### Unit Tests
- Test particle initialization from `InitialState`
- Test process noise injection
- Test resampling algorithms (verify particle diversity)
- Test weight computation and normalization
- Test vertical channel damping (if used)
- Test state estimation with different averaging strategies

### Integration Tests
- Compare PF vs UKF on same trajectory
- Test with GNSS dropout scenarios
- Test vertical channel stability (hover, climb, descent scenarios)
- Verify deterministic results with seeded RNG

### Vertical Channel Tests
Create tests similar to UKF tests in `kalman.rs`:
- `pf_free_fall_motion()` - vertical dynamics with no support
- `pf_hover_motion()` - stationary altitude
- `pf_climb_descent()` - vertical motion profile
- `pf_altitude_drift_during_outage()` - verify controlled drift with 2.5D

## Recommended Implementation Order

1. ✅ Define `Particle` struct (already done)
2. **Implement `forward_2_5d()` function in `particle.rs`**
   - Copy `attitude_update()` from `lib.rs` or make it public
   - Implement `velocity_update_horizontal()` helper
   - Reuse `position_update()` from `lib.rs` (make public if needed)
3. Implement `ProcessNoise` struct
4. Implement `ParticleFilter` struct with basic initialization
5. Implement `predict()` using `forward_2_5d()`
6. Implement `update()` with systematic resampling
7. Implement `get_estimate()` with weighted mean
8. Write unit tests for each component
9. Integration with `sim.rs`
10. Test on real datasets
11. (Optional) Implement third-order damping if 2.5D is insufficient

## Tuning Guidance

### Number of Particles
- Start with **500-1000** particles for smartphone IMU
- Increase to **2000-5000** for high-precision applications
- Monitor computational cost vs. accuracy tradeoff

### Process Noise (2.5D)
**Conservative (high uncertainty, smooth vertical):**
- `σ_h = 0.1 m`, `σ_vv = 0.2 m/s`

**Aggressive (low uncertainty, responsive vertical):**
- `σ_h = 0.01 m`, `σ_vv = 0.05 m/s`

**Recommended starting point:**
- `σ_h = 0.05 m`, `σ_vv = 0.1 m/s`

### Resampling Threshold
- Effective particle threshold: `0.5 * num_particles`
- Resample more frequently if seeing particle degeneracy
- Monitor `n_eff` over time to diagnose issues

## Expected Performance

### Advantages over UKF
- Better handling of non-Gaussian distributions
- More robust to outliers in measurements
- Can represent multimodal posteriors (e.g., during ambiguous scenarios)
- 2.5D approach naturally handles vertical channel uncertainty

### Computational Cost
- PF with N=1000 particles ≈ 10-50× slower than UKF
- Use Rayon for parallel particle propagation if needed
- Profile and optimize hot paths (propagation, resampling)

### Memory Requirements
- Each particle: ~200 bytes (15 states × 8 bytes + overhead)
- 1000 particles ≈ 200 KB (negligible for modern systems)

## References

- Groves, P. D. (2013). *Principles of GNSS, Inertial, and Multisensor Integrated Navigation Systems, 2nd Edition*. Chapters 14-15 (INS error models, vertical channel instability).
- Gustafsson, F., et al. (2002). "Particle filters for positioning, navigation, and tracking." *IEEE Transactions on Signal Processing*.
- Existing `kalman.rs` implementation for architectural patterns.
