//! Particle filter implementation for nonlinear Bayesian state estimation.
//!
//! This module provides a particle filter (also known as Sequential Monte Carlo) implementation
//! for strapdown inertial navigation. Unlike Kalman filters which assume Gaussian distributions,
//! particle filters can represent arbitrary probability distributions through weighted samples.
//! This makes them well-suited for highly nonlinear systems and non-Gaussian noise characteristics.
//! However, they typically require more computational resources due to the need for many particles
//! to accurately approximate the posterior distribution. As such, they are often used in scenarios
//! where Kalman filters struggle, such as in the presence of multimodal distributions or severe
//! nonlinearities with state dynamics and dimensions that are manageable.
//!
//! A full 15-state canonical strapdown inertial navigation system is feasible with particle filters
//! for moderate particle counts (e.g., 1000-5000 particles) on modern hardware with *reasonably
//! good* IMU quality. For more challenging scenarios, such as low-quality IMUs or more complex
//! motion dynamics, hybrid approaches like Rao-Blackwellized Particle Filters (RBPF) can
//! be employed. RBPFs use particles to represent a subset of the state (e.g., position and velocity)
//! while using Kalman filters for the remaining states (e.g., attitude and sensor biases), reducing
//! the dimensionality of the particle filter and improving efficiency.
//!
//! # Key Components
//!
//! - [`Particle`]: A trait defining the required methods for particle state representation
//! - [`ParticleFilter`]: The main particle filter struct managing particles and weights
//! - [`ParticleAveragingStrategy`]: Methods for extracting state estimates from particles
//! - [`ParticleResamplingStrategy`]: Algorithms for combating particle degeneracy

use std::any::Any;

use crate::measurements::MeasurementModel;

use nalgebra::DVector;

pub trait Particle: Any {
    /// Downcast helper method to allow for type-safe downcasting
    fn as_any(&self) -> &dyn Any;
    /// Downcast helper method for mutable references
    fn as_any_mut(&mut self) -> &mut dyn Any;
    fn new(initial_state: &DVector<f64>, weight: f64) -> Self
    where
        Self: Sized;
    /// Returns the state vector of the particle.
    fn state(&self) -> DVector<f64>;

    /// Updates the particle weight based on the provided measurement model and actual measurement.
    fn update_weight<M: MeasurementModel>(&mut self, measurement: &M);

    /// Returns the weight of the particle.
    fn weight(&self) -> f64;

    /// Sets the weight of the particle.
    fn set_weight(&mut self, weight: f64);
}
