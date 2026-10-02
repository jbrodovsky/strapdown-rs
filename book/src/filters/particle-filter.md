# Particle Building Blocks

`core/src/particle.rs` is not a particle filter. There is no canonical particle-filter INS, so
the module provides pieces for assembling one: a trait for a particle, strategy enums, and four
resampling algorithms. The one concrete particle filter in the crate is the
[Rao-Blackwellized particle filter](./rbpf.md) in `core/src/rbpf.rs`, and it uses only the
resampling half of this module.

## What the module provides

| item | kind | purpose |
| --- | --- | --- |
| `Particle` | trait | a weighted state hypothesis: `new(&DVector<f64>, weight)`, `state()`, `set_state(..)`, `weight()`, `set_weight(..)`, plus `as_any` downcasting helpers |
| `ParticleFilter` | trait | one method, `resample(&mut self)`, for a filter built from these parts |
| `ParticleResamplingStrategy` | enum | `Multinomial`, `Systematic` (the default), `Stratified`, `Residual` |
| `ParticleAveragingStrategy` | enum | `Mean` (ignores weights), `WeightedMean` (the default), `HighestWeight` (the maximum-a-posteriori particle) |
| `multinomial_resample`, `systematic_resample`, `stratified_resample`, `residual_resample` | functions | draw ancestor indices from a weight vector |

Each resampling function takes `(weights: &[f64], num_samples: usize, rng: &mut R)` for any
`R: rand::Rng` and returns a `Vec<usize>` of indices into the original particles; a particle
whose index appears $k$ times is copied $k$ times. The weights must already be normalized to sum
to one, and that is the caller's job.

| algorithm | how it draws | variance |
| --- | --- | --- |
| multinomial | $N$ independent draws from the weight distribution | highest |
| systematic | one random offset, then $N$ evenly spaced points through the cumulative weights | lower |
| stratified | one random point in each of $N$ equal strata of $[0, 1)$ | lower than multinomial |
| residual | $\lfloor N w_i \rfloor$ copies of each particle deterministically, the remainder systematically from the fractional parts | lower than multinomial |

The algorithms are deterministic given the RNG state, so a seeded `StdRng` makes resampling
reproducible.

## How the RBPF uses them

The RBPF does not implement `Particle` or `ParticleFilter`, and it does not use
`ParticleAveragingStrategy`. Its particles (`rbpf::RbpfParticle`) carry a two-element horizontal
position error, the conditional mean of the linear states and a weight, which is a different
shape from a `Particle`'s single state vector; its estimate is a weighted mean with the linear
states' covariance added by the law of total covariance (see [RBPF](./rbpf.md)).

What it does use is the resampling:

- `RbpfConfig::resampling_strategy` is a `ParticleResamplingStrategy`, defaulting to
  `Systematic`.
- After each measurement update the filter computes the effective sample size,
  $N_\text{eff} = 1 / \sum_i w_i^2$ (`RaoBlackwellizedParticleFilter::effective_sample_size`),
  and resamples when it falls below `effective_sample_threshold` times the particle count.
  The default threshold, 1.0, resamples after every update, as Canciani and Raquet do.
- The selected strategy's function draws the indices; the survivors are copied with uniform
  weights and then roughened by `roughening_factor` (see [RBPF](./rbpf.md#configuration)).
- The draws come from the filter's own `StdRng`, seeded by `RbpfConfig::seed`, so a run is
  reproducible.

`resampling_strategy` is a field of the library's `RbpfConfig` only. The simulator's
`[particle_filter]` section and the `pf` subcommand do not expose it, so `strapdown-sim pf`
always resamples systematically.

## Building your own

A filter assembled from these parts implements `Particle` for its state type, keeps its own
weights normalized, calls one of the resampling functions inside its `ParticleFilter::resample`,
and implements [`NavigationFilter`](./kalman.md#the-navigationfilter-trait) so it can be driven
by `sim::run_closed_loop` or [`InsEngine`](../user-guide/library.md). The RBPF is the worked
example of the last two steps; `core/src/rbpf.rs` documents the rest.
