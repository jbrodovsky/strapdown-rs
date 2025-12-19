# Particle Filter INS Architecture Design

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

## 1. State Representation

### 1.1 Standard 15-State Model

Following the UKF implementation, the baseline particle filter uses a 15-state model:

**Navigation states (9):**
- Position: `[latitude, longitude, altitude]` (rad, rad, m)
- Velocity: `[v_north, v_east, v_vertical]` (m/s)
- Attitude: `[roll, pitch, yaw]` (rad) - represented internally as DCM

**Bias states (6):**
- Accelerometer biases: `[b_ax, b_ay, b_az]` (m/s²)
- Gyroscope biases: `[b_gx, b_gy, b_gz]` (rad/s)

### 1.2 Extended State for Vertical Channel Damping

For third-order vertical channel damping, add **optional states**:
- Altitude error estimate: `δh` (m)
- Altitude error rate: `δh_dot` (m/s)

Total state dimension: **15 + 2 = 17 states** (when using damping)

**Implementation note**: Use `Particle.other_states: Option<DVector<f64>>` for the damping states.

## 2. Vertical Channel Approaches

### 2.1 Approach A: Third-Order Vertical Channel Damping

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

### 2.2 Approach B: 2.5D Navigation (Simplified Vertical Treatment)

**Motivation**: Treat vertical acceleration as primarily noise-driven rather than deterministic, simplifying the vertical channel dynamics. This is your preference based on past success.

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
- Works well with intermittent GPS (your use case)
- Avoids complex tuning of damping coefficients

**Cons:**
- Less theoretically rigorous
- Vertical channel accumulates uncertainty faster during outages
- Requires careful process noise tuning to balance stability and responsiveness

## 3. Particle Filter Architecture

### 3.1 Core Components (following `kalman.rs` structure)

#### 3.1.1 `Particle` struct (already defined in `particle.rs`)
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

#### 3.1.2 `ParticleFilter` struct
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

#### 3.1.3 `ProcessNoise` struct
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

#### 3.1.4 Enums
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

### 3.2 Core Algorithm

#### 3.2.1 Initialization
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

#### 3.2.2 Predict Step (`predict` method)

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

```rust
impl NavigationFilter for ParticleFilter {
    fn predict(&mut self, imu_data: IMUData, dt: f64) {
        for particle in &mut self.particles {
            // 1. Bias-corrected IMU data
            let corrected_imu = IMUData {
                accel: imu_data.accel - &particle.accel_bias,
                gyro: imu_data.gyro - &particle.gyro_bias,
            };

            // 2. Modified strapdown propagation based on vertical channel mode
            match self.vertical_channel_mode {
                VerticalChannelMode::ThirdOrderDamping { k1, k2 } => {
                    // Use standard forward for damped case
                    forward(&mut particle.nav_state, corrected_imu, dt);
                    self.apply_vertical_damping(particle, k1, k2, dt);
                },
                VerticalChannelMode::Simplified => {
                    // Use 2.5D forward (horizontal + simplified vertical)
                    forward_2_5d(&mut particle.nav_state, corrected_imu, dt);
                },
            }

            // 3. Add process noise
            self.add_process_noise(particle, dt);

            // 4. Random walk on biases (small drift)
            self.propagate_biases(particle, dt);
        }
    }
}
```

**Process noise injection (2.5D approach):**
```rust
fn add_process_noise(&mut self, particle: &mut Particle, dt: f64) {
    let sqrt_dt = dt.sqrt();

    // Standard noise for horizontal states
    particle.nav_state.latitude +=
        self.rng.sample(Normal::new(0.0, self.process_noise.position_std[0])?) * sqrt_dt;
    particle.nav_state.longitude +=
        self.rng.sample(Normal::new(0.0, self.process_noise.position_std[1])?) * sqrt_dt;

    // INCREASED noise for vertical states (2.5D key feature)
    particle.nav_state.altitude +=
        self.rng.sample(Normal::new(0.0, self.process_noise.position_std[2])?) * sqrt_dt;
    particle.nav_state.velocity_vertical +=
        self.rng.sample(Normal::new(0.0, self.process_noise.velocity_std[2])?) * sqrt_dt;

    // Horizontal velocities
    particle.nav_state.velocity_north +=
        self.rng.sample(Normal::new(0.0, self.process_noise.velocity_std[0])?) * sqrt_dt;
    // ... (continue for other states)
}
```

#### 3.2.3 Update Step (`update` method)
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

**Weight computation:**
```rust
fn compute_likelihood(&self, innovation: &DVector<f64>, cov: &DMatrix<f64>) -> f64 {
    // Multivariate Gaussian likelihood
    let inv_cov = robust_spd_solve(cov, &DMatrix::identity(cov.nrows(), cov.ncols()));
    let mahalanobis = (innovation.transpose() * inv_cov * innovation)[(0, 0)];
    (-0.5 * mahalanobis).exp()
}
```

#### 3.2.4 Resampling
```rust
fn resample(&mut self) {
    match self.resampling_strategy {
        ResamplingStrategy::Systematic => self.systematic_resample(),
        ResamplingStrategy::Stratified => self.stratified_resample(),
        // ...
    }
}

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

#### 3.2.5 State Estimation
```rust
fn get_estimate(&self) -> DVector<f64> {
    match self.averaging_strategy {
        AveragingStrategy::WeightedMean => {
            self.weighted_mean_estimate()
        },
        AveragingStrategy::MaximumWeight => {
            self.max_weight_estimate()
        },
        AveragingStrategy::MeanWithTrimming => {
            self.trimmed_mean_estimate()
        },
    }
}

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

### 3.3 The `forward_2_5d()` Function

This function implements modified strapdown mechanization for 2.5D navigation. It should be added to `particle.rs` as a module-level function.

#### 3.3.1 Mathematical Formulation

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

#### 3.3.2 Implementation

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
    // 1. ATTITUDE UPDATE (unchanged from standard forward)
    //    Equation 5.46 - uses gyros, accounts for Earth rate and transport rate
    let c_0: Rotation3<f64> = state.attitude;
    let c_1: Matrix3<f64> = attitude_update(state, imu_data.gyro, dt);

    // 2. SPECIFIC FORCE TRANSFORMATION (unchanged)
    //    Equation 5.47 - transform body frame accel to nav frame
    let f: Vector3<f64> = 0.5 * (c_0.matrix() + c_1) * imu_data.accel;

    // 3. HORIZONTAL VELOCITY UPDATE (modified - only horizontal components)
    //    Full mechanization for v_n and v_e, IGNORE v_v integration
    let velocity_horizontal = velocity_update_horizontal(state, f, dt);

    // 4. VERTICAL VELOCITY (simplified - no deterministic integration)
    //    Keep current vertical velocity unchanged (process noise will be added later)
    let velocity_vertical = state.velocity_vertical;

    // 5. POSITION UPDATE
    //    - Horizontal: Full mechanization using updated horizontal velocities
    //    - Vertical: Simple integration of vertical velocity
    let velocity = Vector3::new(velocity_horizontal[0], velocity_horizontal[1], velocity_vertical);
    let (lat_1, lon_1, alt_1) = position_update(state, velocity, dt);

    // 6. UPDATE STATE
    state.attitude = Rotation3::from_matrix(&c_1);
    state.velocity_north = velocity_horizontal[0];
    state.velocity_east = velocity_horizontal[1];
    state.velocity_vertical = velocity_vertical;  // Unchanged from predict
    state.latitude = lat_1;
    state.longitude = lon_1;
    state.altitude = alt_1;
}

/// Horizontal velocity update for 2.5D navigation.
///
/// Implements full strapdown velocity update (Equation 5.54) but only for
/// horizontal components (north and east). Vertical component is ignored.
///
/// # Returns
/// Vector2 containing [v_n, v_e] in m/s
fn velocity_update_horizontal(
    state: &StrapdownState,
    specific_force: Vector3<f64>,
    dt: f64,
) -> Vector2<f64> {
    // Compute transport rate and Earth rotation rate
    let transport_rate: Matrix3<f64> = earth::vector_to_skew_symmetric(&earth::transport_rate(
        &state.latitude.to_degrees(),
        &state.altitude,
        &Vector3::from_vec(vec![
            state.velocity_north,
            state.velocity_east,
            state.velocity_vertical,
        ]),
    ));

    let rotation_rate: Matrix3<f64> =
        earth::vector_to_skew_symmetric(&earth::earth_rate_lla(&state.latitude.to_degrees()));

    // Get ECEF position vector for Coriolis calculation
    let r = earth::ecef_to_lla(&state.latitude.to_degrees(), &state.longitude.to_degrees());

    let velocity: Vector3<f64> = Vector3::new(
        state.velocity_north,
        state.velocity_east,
        state.velocity_vertical,
    );

    // Gravity (only used for horizontal components, but computed for consistency)
    let gravity = Vector3::new(
        0.0,
        0.0,
        earth::gravity(&state.latitude.to_degrees(), &state.altitude),
    );
    let gravity = if state.is_enu { -gravity } else { gravity };

    // Full velocity update equation (Equation 5.54)
    let velocity_full = velocity
        + (specific_force + gravity - r * (transport_rate + 2.0 * rotation_rate) * velocity) * dt;

    // RETURN ONLY HORIZONTAL COMPONENTS
    Vector2::new(velocity_full[0], velocity_full[1])
}
```

#### 3.3.3 Key Differences from Standard `forward()`

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

#### 3.3.4 Alternative: Weak Coupling Option

If you want some influence from vertical acceleration (but much weaker than standard strapdown):

```rust
// In forward_2_5d(), replace step 4 with:
let vertical_accel_contribution = specific_force[2] + gravity[2]
    - r[2] * (transport_rate + 2.0 * rotation_rate)[2] * velocity[2];

// Weak coupling: only use a fraction (e.g., 10%) of the acceleration
let coupling_strength = 0.1;  // Tunable: 0.0 = no coupling, 1.0 = full strapdown
let velocity_vertical = state.velocity_vertical
    + coupling_strength * vertical_accel_contribution * dt;
```

This provides a middle ground between full strapdown and pure random walk.

## 4. Vertical Channel Implementation Details

### 4.1 Third-Order Damping Implementation

```rust
fn apply_vertical_damping(&mut self, particle: &mut Particle, k1: f64, k2: f64, dt: f64) {
    if let Some(ref mut damping_states) = particle.other_states {
        let delta_h = damping_states[0];
        let delta_h_dot = damping_states[1];

        // Feedback to altitude and vertical velocity
        particle.nav_state.altitude -= k1 * delta_h * dt;
        particle.nav_state.velocity_vertical -= k2 * delta_h_dot * dt;

        // Propagate damping states
        // δh_dot = δv_v
        // δh_ddot = -k1·δh - k2·δh_dot
        let delta_h_ddot = -k1 * delta_h - k2 * delta_h_dot;

        damping_states[0] += delta_h_dot * dt;
        damping_states[1] += delta_h_ddot * dt;

        // Add process noise to damping states
        if let Some(ref damping_noise) = self.process_noise.damping_states_std {
            damping_states[0] += self.rng.sample(Normal::new(0.0, damping_noise[0])?);
            damping_states[1] += self.rng.sample(Normal::new(0.0, damping_noise[1])?);
        }
    }
}
```

### 4.2 2.5D Process Noise Configuration

**Recommended process noise for 2.5D approach:**
```rust
ProcessNoise {
    position_std: Vector3::new(1e-3, 1e-3, 5e-2),  // [lat, lon, h] - note large σ_h
    velocity_std: Vector3::new(1e-2, 1e-2, 1e-1),  // [v_n, v_e, v_v] - note large σ_vv
    attitude_std: Vector3::new(1e-3, 1e-3, 1e-3),
    accel_bias_std: Vector3::new(1e-3, 1e-3, 1e-3),
    gyro_bias_std: Vector3::new(1e-4, 1e-4, 1e-4),
    damping_states_std: None,  // Not used in 2.5D
}
```

Key insight: `σ_h` and `σ_vv` are 10-100× larger than horizontal counterparts, allowing the particle filter to naturally handle vertical uncertainty without explicit damping.

## 5. Integration with Existing Codebase

### 5.1 Mirroring UKF Structure

The particle filter should implement the same `NavigationFilter` trait as UKF:
```rust
pub trait NavigationFilter {
    fn predict(&mut self, imu_data: IMUData, dt: f64);
    fn update<M: MeasurementModel + ?Sized>(&mut self, measurement: &M);
    fn get_estimate(&self) -> DVector<f64>;
    fn get_certainty(&self) -> DMatrix<f64>;  // Empirical covariance from particles
}
```

### 5.2 Simulation Integration

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

pub fn closed_loop<F: NavigationFilter>(
    filter: &mut F,
    data: &[TestDataRecord],
    // ... existing parameters
) -> Vec<NavigationResult> {
    // Same structure as current implementation
}
```

## 6. Testing Strategy

### 6.1 Unit Tests
- Test particle initialization from `InitialState`
- Test process noise injection
- Test resampling algorithms (verify particle diversity)
- Test weight computation and normalization
- Test vertical channel damping (if used)
- Test state estimation with different averaging strategies

### 6.2 Integration Tests
- Compare PF vs UKF on same trajectory
- Test with GNSS dropout scenarios
- Test vertical channel stability (hover, climb, descent scenarios)
- Verify deterministic results with seeded RNG

### 6.3 Vertical Channel Tests
Create tests similar to UKF tests in `kalman.rs`:
- `pf_free_fall_motion()` - vertical dynamics with no support
- `pf_hover_motion()` - stationary altitude
- `pf_climb_descent()` - vertical motion profile
- `pf_altitude_drift_during_outage()` - verify controlled drift with 2.5D

## 7. Recommended Implementation Order

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
8. Write unit tests for each component:
   - Test `forward_2_5d()` vs `forward()` to verify vertical decoupling
   - Test particle propagation with process noise
   - Test resampling maintains particle diversity
9. Integration with `sim.rs`
10. Test on real datasets
11. (Optional) Implement third-order damping if 2.5D is insufficient

**Critical first step**: Getting `forward_2_5d()` correct is essential. This function is the architectural foundation that enables 2.5D navigation.

## 8. Tuning Guidance

### 8.1 Number of Particles
- Start with **500-1000** particles for smartphone IMU
- Increase to **2000-5000** for high-precision applications
- Monitor computational cost vs. accuracy tradeoff

### 8.2 Process Noise (2.5D)
**Conservative (high uncertainty, smooth vertical):**
- `σ_h = 0.1 m`, `σ_vv = 0.2 m/s`

**Aggressive (low uncertainty, responsive vertical):**
- `σ_h = 0.01 m`, `σ_vv = 0.05 m/s`

**Recommended starting point:**
- `σ_h = 0.05 m`, `σ_vv = 0.1 m/s`

### 8.3 Resampling Threshold
- Effective particle threshold: `0.5 * num_particles`
- Resample more frequently if seeing particle degeneracy
- Monitor `n_eff` over time to diagnose issues

## 9. Expected Performance

### 9.1 Advantages over UKF
- Better handling of non-Gaussian distributions
- More robust to outliers in measurements
- Can represent multimodal posteriors (e.g., during ambiguous scenarios)
- 2.5D approach naturally handles vertical channel uncertainty

### 9.2 Computational Cost
- PF with N=1000 particles ≈ 10-50× slower than UKF
- Use Rayon for parallel particle propagation if needed
- Profile and optimize hot paths (propagation, resampling)

### 9.3 Memory Requirements
- Each particle: ~200 bytes (15 states × 8 bytes + overhead)
- 1000 particles ≈ 200 KB (negligible for modern systems)

## 10. References

- Groves, P. D. (2013). *Principles of GNSS, Inertial, and Multisensor Integrated Navigation Systems, 2nd Edition*. Chapters 14-15 (INS error models, vertical channel instability).
- Gustafsson, F., et al. (2002). "Particle filters for positioning, navigation, and tracking." *IEEE Transactions on Signal Processing*.
- Your existing `kalman.rs` implementation for architectural patterns.
