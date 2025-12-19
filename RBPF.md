# Rao-Blackwellized Particle Filter Implementation Plan

## Overview

Implement a Rao-Blackwellized Particle Filter (RBPF) that partitions the 15-state INS into nonlinear states (position: 3) estimated via particles and conditionally linear states (velocity: 3, attitude: 3, biases: 6) estimated via per-particle UKF filters. This reduces computational burden while improving bias estimation and maintaining the existing 2.5D vertical channel approach.

## State Partitioning

**Nonlinear states (particles):**
- Latitude (rad)
- Longitude (rad)
- Altitude (m)

**Linear/Conditionally-Gaussian states (per-particle UKF):**
- Velocity: v_north, v_east, v_vertical (m/s)
- Attitude: roll, pitch, yaw (rad)
- Accelerometer biases: b_ax, b_ay, b_az (m/s²)
- Gyroscope biases: b_gx, b_gy, b_gz (rad/s)

## Implementation Steps

### Step 1: Create new structs for RBPF state representation

**Location:** `core/src/particle.rs`

#### 1.1 Define `RBParticle` struct

```rust
pub struct RBParticle {
    /// Position states: [latitude (rad), longitude (rad), altitude (m)]
    pub position: Vector3<f64>,
    
    /// Per-particle UKF for linear states (9-state: velocity, attitude, biases)
    pub kalman_filter: PerParticleUKF,
    
    /// Importance weight (unnormalized likelihood)
    pub weight: f64,
}
```

**Methods to implement:**
- `to_state_vector(&self) -> DVector<f64>` - Convert to 15-state vector for compatibility
- `from_state_vector(state: &DVector<f64>) -> Self` - Create from state vector
- `get_position(&self) -> Vector3<f64>` - Access position
- `get_linear_states(&self) -> DVector<f64>` - Access velocity, attitude, biases from UKF

#### 1.2 Define `PerParticleUKF` struct

```rust
pub struct PerParticleUKF {
    /// Mean state: [v_n, v_e, v_v, roll, pitch, yaw, b_ax, b_ay, b_az, b_gx, b_gy, b_gz]
    pub mean_state: DVector<f64>,
    
    /// Covariance matrix (9×9)
    pub covariance: DMatrix<f64>,
    
    /// UKF parameters (alpha, beta, kappa)
    alpha: f64,
    beta: f64,
    kappa: f64,
    
    /// Cached UKF weights
    weights_mean: DVector<f64>,
    weights_cov: DVector<f64>,
}
```

**Methods to implement:**
- `new(initial_state: DVector<f64>, initial_cov: DMatrix<f64>, alpha, beta, kappa) -> Self`
- `predict(&mut self, position: &Vector3<f64>, imu_data: IMUData, process_noise: &DMatrix<f64>, dt: f64)` - UKF predict step conditioned on particle position
- `update<M: MeasurementModel>(&mut self, position: &Vector3<f64>, measurement: &M) -> f64` - UKF update, returns marginal likelihood for particle weighting
- `get_sigma_points(&self) -> DMatrix<f64>` - Generate sigma points for 9-state UKF
- `get_estimate(&self) -> DVector<f64>` - Return mean state
- `get_covariance(&self) -> DMatrix<f64>` - Return covariance

#### 1.3 Define `RBProcessNoise` struct

```rust
pub struct RBProcessNoise {
    /// Position noise standard deviations [σ_lat, σ_lon, σ_h] (rad, rad, m)
    pub position_std: Vector3<f64>,
    
    /// Linear state noise covariance matrix (9×9) for per-particle UKF
    /// Includes: velocity (3), attitude (3), accel biases (3), gyro biases (3)
    pub linear_states_covariance: DMatrix<f64>,
}
```

**Default implementation:**
```rust
impl Default for RBProcessNoise {
    fn default() -> Self {
        // Position noise (for particle diffusion)
        let position_std = Vector3::new(1e-3, 1e-3, 5e-2);
        
        // Linear state noise (for UKF predict)
        let linear_noise_diag = vec![
            1e-2, 1e-2, 1e-1,  // velocity (2.5D: large v_v)
            1e-3, 1e-3, 1e-3,  // attitude
            1e-3, 1e-3, 1e-3,  // accel biases
            1e-4, 1e-4, 1e-4,  // gyro biases
        ];
        let linear_states_covariance = DMatrix::from_diagonal(&DVector::from_vec(linear_noise_diag));
        
        RBProcessNoise {
            position_std,
            linear_states_covariance,
        }
    }
}
```

### Step 2: Implement `RaoBlackwellizedParticleFilter` struct

**Location:** `core/src/particle.rs`

#### 2.1 Main struct definition

```rust
pub struct RaoBlackwellizedParticleFilter {
    /// Ensemble of RB particles
    particles: Vec<RBParticle>,
    
    /// Number of particles
    num_particles: usize,
    
    /// Process noise parameters
    process_noise: RBProcessNoise,
    
    /// Vertical channel mode (2.5D simplified)
    vertical_channel_mode: VerticalChannelMode,
    
    /// Resampling strategy
    resampling_strategy: ResamplingStrategy,
    
    /// Effective particle threshold for resampling
    effective_particle_threshold: f64,
    
    /// Random number generator (seeded)
    rng: StdRng,
    
    /// Coordinate frame flag
    is_enu: bool,
    
    /// UKF parameters for per-particle filters
    ukf_alpha: f64,
    ukf_beta: f64,
    ukf_kappa: f64,
}
```

#### 2.2 Constructor

```rust
impl RaoBlackwellizedParticleFilter {
    pub fn new(
        initial_state: InitialState,
        imu_biases: Vec<f64>,
        covariance_diagonal: Vec<f64>,  // 15-element: 3 position + 9 linear + 3 biases
        process_noise: RBProcessNoise,
        num_particles: usize,
        vertical_mode: VerticalChannelMode,
        resampling_strategy: ResamplingStrategy,
        ukf_alpha: f64,
        ukf_beta: f64,
        ukf_kappa: f64,
        seed: Option<u64>,
    ) -> Self {
        // 1. Initialize RNG
        // 2. Sample initial positions from Gaussian (first 3 elements of covariance_diagonal)
        // 3. Initialize identical per-particle UKFs with velocity, attitude, biases
        //    (elements 3-14 of covariance_diagonal)
        // 4. Set uniform weights
        // ...
    }
}
```

#### 2.3 Implement `NavigationFilter` trait

```rust
impl NavigationFilter for RaoBlackwellizedParticleFilter {
    fn predict(&mut self, imu_data: IMUData, dt: f64) {
        // For each particle:
        for particle in &mut self.particles {
            // 1. Get current velocity and attitude from UKF
            let linear_states = particle.kalman_filter.get_estimate();
            let velocity = linear_states.fixed_rows::<3>(0);
            let attitude = linear_states.fixed_rows::<3>(3);
            
            // 2. Propagate position using forward_2_5d_rbpf (attitude, velocity given)
            forward_2_5d_rbpf(
                &mut particle.position,
                &velocity,
                &attitude,
                dt
            );
            
            // 3. Add process noise to position
            add_position_noise(&mut particle.position, &self.process_noise.position_std, dt, &mut self.rng);
            
            // 4. UKF predict for linear states (conditioned on new position)
            let biases = linear_states.fixed_rows::<6>(3);
            let bias_corrected_imu = IMUData {
                accel: imu_data.accel - biases.fixed_rows::<3>(0),
                gyro: imu_data.gyro - biases.fixed_rows::<3>(3),
            };
            particle.kalman_filter.predict(
                &particle.position,
                bias_corrected_imu,
                &self.process_noise.linear_states_covariance,
                dt
            );
        }
    }
    
    fn update<M: MeasurementModel + ?Sized>(&mut self, measurement: &M) {
        // 1. Compute marginal likelihood for each particle
        for particle in &mut self.particles {
            // UKF update returns marginal likelihood p(z|position)
            let log_likelihood = particle.kalman_filter.update(
                &particle.position,
                measurement
            );
            particle.weight *= log_likelihood.exp();
        }
        
        // 2. Normalize weights
        self.normalize_weights();
        
        // 3. Check ESS and resample if needed
        let ess = self.effective_particle_count();
        if ess < self.effective_particle_threshold {
            self.resample();
        }
    }
    
    fn get_estimate(&self) -> DVector<f64> {
        // 1. Weighted average of positions
        let mean_position = self.weighted_position_mean();
        
        // 2. Weighted average of per-particle UKF means
        let mean_linear_states = self.weighted_linear_states_mean();
        
        // 3. Concatenate into 15-state vector
        let mut state = DVector::zeros(15);
        state.fixed_rows_mut::<3>(0).copy_from(&mean_position);
        state.fixed_rows_mut::<12>(3).copy_from(&mean_linear_states);
        state
    }
    
    fn get_certainty(&self) -> DMatrix<f64> {
        // Combined covariance from:
        // 1. Particle position covariance (empirical from weighted samples)
        // 2. Weighted average of per-particle UKF covariances
        // 3. Cross terms (position-linear states correlation)
        self.compute_combined_covariance()
    }
}
```

### Step 3: Add RBPF-specific propagation functions

**Location:** `core/src/particle.rs`

#### 3.1 Position propagation function

```rust
/// Propagate position states for RBPF given velocity and attitude from UKF.
///
/// This is the position-only component of 2.5D navigation. The velocity and
/// attitude are provided as inputs (from per-particle UKF) rather than being
/// propagated together with position.
///
/// # Arguments
/// * `position` - Mutable reference to position [lat, lon, alt]
/// * `velocity` - Velocity vector [v_n, v_e, v_v] from UKF
/// * `attitude` - Attitude angles [roll, pitch, yaw] from UKF
/// * `dt` - Time step
pub fn forward_2_5d_rbpf(
    position: &mut Vector3<f64>,
    velocity: &Vector3<f64>,
    attitude: &Vector3<f64>,
    dt: f64,
) {
    // 1. Update horizontal position (lat, lon) using geodetic equations
    //    Similar to position_update() but only using horizontal velocities
    
    // 2. Update altitude using vertical velocity (2.5D: simple integration)
    //    alt(t+dt) = alt(t) + v_v * dt
    
    // Uses earth model and geodetic calculations from crate::earth
}
```

#### 3.2 UKF predict with position conditioning

```rust
impl PerParticleUKF {
    /// UKF predict step conditioned on particle position.
    ///
    /// Propagates the 9-state linear system: velocity, attitude, biases.
    /// The nonlinear coupling with position comes through gravity, Earth rate,
    /// and transport rate terms which are evaluated at the given position.
    pub fn predict(
        &mut self,
        position: &Vector3<f64>,  // Given by particle
        imu_data: IMUData,
        process_noise: &DMatrix<f64>,
        dt: f64,
    ) {
        // 1. Generate sigma points for 9-state UKF
        let sigma_points = self.get_sigma_points();
        
        // 2. Propagate each sigma point
        //    - Velocity: use velocity_update_horizontal and vertical velocity dynamics
        //    - Attitude: use attitude_update with gyro measurements
        //    - Biases: random walk
        let propagated_sigma_points = self.propagate_sigma_points(
            &sigma_points,
            position,  // Earth parameters evaluated here
            imu_data,
            dt
        );
        
        // 3. Compute predicted mean and covariance
        let mut mean_predicted = DVector::zeros(9);
        for (i, sigma_point) in propagated_sigma_points.column_iter().enumerate() {
            mean_predicted += self.weights_mean[i] * sigma_point;
        }
        
        let mut cov_predicted = DMatrix::zeros(9, 9);
        for (i, sigma_point) in propagated_sigma_points.column_iter().enumerate() {
            let diff = sigma_point - &mean_predicted;
            cov_predicted += self.weights_cov[i] * (&diff * diff.transpose());
        }
        cov_predicted += process_noise;
        
        self.mean_state = mean_predicted;
        self.covariance = symmetrize(&cov_predicted);
    }
}
```

#### 3.3 UKF update with marginal likelihood

```rust
impl PerParticleUKF {
    /// UKF update step, returns marginal log-likelihood for particle weighting.
    ///
    /// Performs standard UKF measurement update on the linear states and
    /// computes the marginal likelihood p(z | position) which is used to
    /// update the particle weight.
    pub fn update<M: MeasurementModel + ?Sized>(
        &mut self,
        position: &Vector3<f64>,
        measurement: &M,
    ) -> f64 {
        // 1. Generate measurement sigma points
        // 2. Compute predicted measurement and innovation covariance
        // 3. Compute Kalman gain
        // 4. Update mean and covariance
        // 5. Compute marginal likelihood:
        //    log p(z|position) = -0.5 * (innovation^T * S^-1 * innovation + log|S| + k*log(2π))
        //    where S is the innovation covariance
        
        // ... (Standard UKF update equations)
        
        // Return log-likelihood for particle weighting
        let log_likelihood = -0.5 * (
            innovation.dot(&s_inv_innovation) +
            s.determinant().ln() +
            innovation.len() as f64 * (2.0 * std::f64::consts::PI).ln()
        );
        
        log_likelihood
    }
}
```

### Step 4: Extend sim.rs with initialization helper

**Location:** `core/src/sim.rs`

```rust
/// Initialize a Rao-Blackwellized particle filter for simulation.
///
/// This function creates and initializes a `RaoBlackwellizedParticleFilter` with the
/// given parameters, using UKF for per-particle linear state estimation.
///
/// # Arguments
///
/// * `initial_pose` - A `TestDataRecord` containing the initial pose information.
/// * `num_particles` - Number of particles in the ensemble (recommend 50-200 for RBPF).
/// * `vertical_mode` - Vertical channel mode (Simplified for 2.5D).
/// * `attitude_covariance` - Optional initial attitude covariance.
/// * `imu_biases` - Optional initial IMU biases.
/// * `imu_biases_covariance` - Optional IMU bias covariance.
/// * `process_noise` - Optional `RBProcessNoise` struct.
/// * `resampling_strategy` - Resampling algorithm (default: Systematic).
/// * `ukf_alpha` - UKF spread parameter (default: 1e-3).
/// * `ukf_beta` - UKF distribution parameter (default: 2.0).
/// * `ukf_kappa` - UKF secondary scaling (default: 0.0).
/// * `seed` - Optional random seed for reproducibility.
///
/// # Returns
///
/// * `RaoBlackwellizedParticleFilter` - Initialized RBPF instance.
#[allow(clippy::too_many_arguments)]
pub fn initialize_rbpf(
    initial_pose: &TestDataRecord,
    num_particles: usize,
    vertical_mode: VerticalChannelMode,
    attitude_covariance: Option<Vec<f64>>,
    imu_biases: Option<Vec<f64>>,
    imu_biases_covariance: Option<Vec<f64>>,
    process_noise: Option<RBProcessNoise>,
    resampling_strategy: Option<ResamplingStrategy>,
    ukf_alpha: Option<f64>,
    ukf_beta: Option<f64>,
    ukf_kappa: Option<f64>,
    seed: Option<u64>,
) -> RaoBlackwellizedParticleFilter {
    // Implementation parallel to initialize_particle_filter
    // but using RBProcessNoise and RBPF constructor
    // ...
}
```

### Step 5: Add integration tests

**Location:** `core/tests/integration_tests.rs`

#### 5.1 Basic closed-loop test

```rust
#[test]
fn test_rbpf_closed_loop_on_real_data() {
    // Load test data
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let test_data_path = Path::new(manifest_dir).join("tests/test_data.csv");
    let records = load_test_data(&test_data_path);
    assert!(!records.is_empty());

    // Initialize RBPF with 100 particles (fewer than standard PF)
    let mut rbpf = initialize_rbpf(
        &records[0],
        100,  // Reduced particle count vs standard PF
        VerticalChannelMode::Simplified,
        None,
        None,
        None,
        None,
        Some(ResamplingStrategy::Systematic),
        Some(1e-3),  // UKF alpha
        Some(2.0),   // UKF beta
        Some(0.0),   // UKF kappa
        Some(42),    // Seed
    );

    // Build event stream with passthrough GNSS
    let cfg = GnssDegradationConfig {
        scheduler: GnssScheduler::PassThrough,
        fault_model: GnssFaultModel::None,
    };
    let stream = build_event_stream(&records, &cfg);

    // Run closed-loop
    let results = run_closed_loop(&mut rbpf, stream, None)
        .expect("RBPF closed-loop should complete");

    // Compute error metrics
    let stats = compute_error_metrics(&results, &records);

    // Print statistics
    println!("\n=== RBPF Closed-Loop Error Statistics ===");
    println!("Horizontal Error: mean={:.2}m, max={:.2}m, rms={:.2}m",
        stats.mean_horizontal_error,
        stats.max_horizontal_error,
        stats.rms_horizontal_error
    );
    
    // Assert reasonable error bounds (should match or beat UKF)
    assert!(
        stats.rms_horizontal_error < 50.0,
        "RBPF RMS horizontal error too large: {:.2}m",
        stats.rms_horizontal_error
    );
}
```

#### 5.2 GNSS dropout test

```rust
#[test]
fn test_rbpf_with_gnss_dropout() {
    // Similar to particle filter dropout test
    // Use periodic scheduler with 60s on / 30s off
    // Verify particle diversity maintained during outages
    // ...
}
```

#### 5.3 Comparison test

```rust
#[test]
fn test_rbpf_vs_standard_pf() {
    // Load same data
    // Run RBPF with 100 particles
    // Run standard PF with 500 particles
    // Compare:
    //   - Accuracy (RBPF should match or beat)
    //   - Bias estimation (RBPF should be better)
    //   - Computation time (not directly tested, but note particle count)
    // ...
}
```

### Step 6: Documentation

#### 6.1 Design document

**Location:** `docs/PARTICLE_FILTER_DESIGN.md`

Add new section:

```markdown
## Rao-Blackwellized Particle Filter (RBPF)

### Overview

The RBPF exploits the conditional linear structure of the INS equations to reduce
computational burden while maintaining or improving estimation performance. It
partitions the 15-state vector into:

- **Nonlinear states (3)**: Position [lat, lon, alt] → represented by particles
- **Linear states (12)**: Velocity, attitude, biases → per-particle UKF

### Key Advantage

By marginalizing the linear states analytically (via Kalman filter), the RBPF
requires significantly fewer particles than a standard PF:

- Standard PF: 500-1000 particles typical
- RBPF: 50-200 particles sufficient

This 5-10× reduction in particles translates directly to computational savings
while often improving bias estimation quality.

### State Partitioning Rationale

**Why position as nonlinear?**
- Coupled with Earth model (geodetic coordinates)
- Subject to GPS outages → multimodal posteriors
- Nonlinear transformation from body to navigation frame

**Why velocity/attitude/biases as linear?**
- Conditionally Gaussian given position
- Biases evolve as random walks (Gaussian)
- Attitude dynamics approximately linear over short timesteps
- UKF handles residual nonlinearity efficiently

### 2.5D Integration

The RBPF maintains the 2.5D vertical channel approach:
- Vertical velocity in per-particle UKF with large process noise
- Position altitude propagated via simple integration
- Horizontal dynamics use full strapdown mechanization

### Algorithm Summary

**Predict:**
```
For each particle i:
  1. Get velocity, attitude from particle_i.ukf
  2. Propagate position using forward_2_5d_rbpf(position, velocity, attitude, dt)
  3. Add position process noise ~ N(0, Q_position)
  4. UKF predict for linear states: ukf.predict(new_position, imu, Q_linear, dt)
```

**Update:**
```
For each particle i:
  1. Compute marginal likelihood: p(z | position_i) from UKF innovation
  2. Update particle weight: w_i *= p(z | position_i)
  3. UKF update linear states: ukf.update(position_i, z)

Normalize weights
If ESS < threshold: resample particles
```

**Estimate:**
```
position_estimate = Σ(w_i * position_i)
linear_estimate = Σ(w_i * ukf_i.mean)
covariance = particle_cov(position) + Σ(w_i * ukf_i.cov) + cross_terms
```

### When to Use RBPF vs Standard PF

**Use RBPF when:**
- Computational resources limited
- Bias estimation critical (e.g., low-cost IMU)
- GNSS intermittent but not completely denied
- Need real-time performance

**Use Standard PF when:**
- Highly nonlinear dynamics (aggressive maneuvers)
- Extreme non-Gaussian posteriors expected
- Computational cost not a concern
- Research/benchmarking (simpler algorithm)
```

#### 6.2 User guide

**Location:** `docs/USER_GUIDE.md`

Add RBPF example:

```markdown
### Rao-Blackwellized Particle Filter

For improved efficiency with similar accuracy:

```rust
use strapdown::particle::{RaoBlackwellizedParticleFilter, RBProcessNoise, VerticalChannelMode, ResamplingStrategy};
use strapdown::kalman::InitialState;

let initial_state = InitialState { /* ... */ };
let process_noise = RBProcessNoise::default();

let rbpf = RaoBlackwellizedParticleFilter::new(
    initial_state,
    vec![0.0; 6],  // IMU biases
    vec![1e-3; 15],  // Initial covariance diagonal
    process_noise,
    100,  // Fewer particles than standard PF
    VerticalChannelMode::Simplified,
    ResamplingStrategy::Systematic,
    1e-3,  // UKF alpha
    2.0,   // UKF beta  
    0.0,   // UKF kappa
    Some(42),  // Random seed
);
```

The RBPF uses per-particle UKF for velocity, attitude, and biases, requiring
only ~100 particles compared to 500-1000 for standard PF.
```

## Implementation Notes

### Measurement Routing

Different measurements update different components:

- **GPS Position** → Updates particle weights via marginal likelihood, UKF updates velocity correlation
- **GPS Velocity** → UKF update only (linear in velocity space)
- **GPS Position+Velocity** → Combined: position for weights, velocity for UKF
- **Barometric Altitude** → Updates particle altitude weights, UKF vertical velocity

### Numerical Stability

1. **Log-likelihood computation**: Use log-space for particle weights to avoid underflow
2. **Covariance symmetrization**: Apply `symmetrize()` to UKF covariance matrices
3. **Robust SPD solve**: Use `robust_spd_solve()` for Kalman gain computation
4. **Weight normalization**: Subtract max log-weight before exp() to prevent overflow

### Performance Considerations

**Expected particle counts:**
- Continuous GNSS: 50 particles sufficient
- Intermittent GNSS (5-10s updates): 100 particles
- Long outages (>30s): 200 particles
- Extreme scenarios: 500 particles (but standard PF may be better)

**Computational complexity per timestep:**
- RBPF: O(N_particles × [position_prop + UKF_prop + UKF_update])
- Standard PF: O(N_particles × full_state_prop)
- UKF speedup: ~2-3× per particle, but RBPF UKF is 9-state vs 15-state
- Overall: RBPF with 100 particles ≈ Standard PF with 200-300 particles in cost

### Compatibility

The RBPF implements the `NavigationFilter` trait, making it a drop-in replacement for UKF or standard PF in:
- `run_closed_loop()` simulation
- `dead_reckoning()` comparison
- Event stream processing
- All measurement models

## Future Enhancements

1. **Adaptive particle count**: Increase/decrease N based on ESS or GPS availability
2. **Regularization**: Add roughening/jitter to prevent particle collapse
3. **Parallel UKF**: Use Rayon to parallelize per-particle UKF operations
4. **Alternative partitioning**: Position+Attitude as nonlinear (4D particles, 6D UKF)
5. **Auxiliary particle filter**: Use auxiliary variable for improved proposal

## References

1. Doucet, A., et al. "Rao-Blackwellised Particle Filtering for Dynamic Bayesian Networks." UAI 2000.
2. Schön, T., et al. "Marginalized Particle Filters for Mixed Linear/Nonlinear State-Space Models." IEEE Trans. Signal Processing, 2005.
3. Gustafsson, F., et al. "Particle filters for positioning, navigation, and tracking." IEEE Trans. Signal Processing, 2002.
