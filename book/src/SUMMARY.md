# Summary

[Introduction](./introduction.md)

# Getting Started

- [Installation](./installation/installation.md)
  - [System Requirements](./installation/requirements.md)
  - [Installing from Crates.io](./installation/crates-io.md)
  - [Building from Source](./installation/building.md)
- [Quick Start](./quick-start.md)

# User Guide

- [Overview](./user-guide/overview.md)
- [Core Concepts](./user-guide/concepts.md)
  - [Strapdown Mechanization](./user-guide/strapdown-mechanization.md)
  - [Coordinate Frames](./user-guide/coordinate-frames.md)
  - [State Representation](./user-guide/state-representation.md)
- [Running Simulations](./user-guide/simulations.md)
  - [Open-Loop (Dead Reckoning)](./user-guide/open-loop.md)
  - [Closed-Loop (UKF/EKF)](./user-guide/closed-loop.md)
  - [Particle Filter](./user-guide/particle-filter.md)
- [Input Data Format](./user-guide/data-format.md)
- [Configuration Files](./user-guide/configuration.md)
- [Logging](./user-guide/logging.md)

# Navigation Filters

- [Kalman Filters](./filters/kalman.md)
  - [Extended Kalman Filter (EKF)](./filters/ekf.md)
  - [Unscented Kalman Filter (UKF)](./filters/ukf.md)
  - [Comparison: EKF vs UKF](./filters/comparison.md)
- [Particle Filters](./filters/particle-filter.md)
  - [Rao-Blackwellized Particle Filter](./filters/rbpf.md)
- [Measurement Models](./filters/measurements.md)

# Geophysical Navigation

- [Overview](./geonav/overview.md)
- [Gravity Anomaly Navigation](./geonav/gravity.md)
- [Magnetic Anomaly Navigation](./geonav/magnetic.md)
- [Data Sources and Maps](./geonav/data-sources.md)

# GNSS Degradation Scenarios

- [Fault Simulation](./gnss/fault-simulation.md)
- [Dropout Scenarios](./gnss/dropouts.md)
- [Reduced Update Rates](./gnss/reduced-rates.md)
- [Measurement Corruption](./gnss/corruption.md)

# API Reference

- [strapdown-core](./api/core.md)
  - [earth Module](./api/earth.md)
  - [kalman Module](./api/kalman.md)
  - [measurements Module](./api/measurements.md)
  - [particles Module](./api/particles.md)
  - [sim Module](./api/sim.md)
- [strapdown-sim](./api/sim-binary.md)
- [strapdown-geonav](./api/geonav.md)

# Examples and Tutorials

- [Example Configurations](./examples/configurations.md)
- [Tutorial: Basic INS Simulation](./examples/tutorial-basic.md)
- [Tutorial: GPS Degradation](./examples/tutorial-gps-degradation.md)
- [Tutorial: Using the Particle Filter](./examples/tutorial-particle-filter.md)

# Development

- [Contributing](./development/contributing.md)
- [Building and Testing](./development/building.md)
- [Architecture](./development/architecture.md)
- [Project Structure](./development/structure.md)

# FAQ

- [Frequently Asked Questions](./faq.md)

# Additional Resources

- [Publications](./resources/publications.md)
- [External Links](./resources/links.md)
- [Glossary](./resources/glossary.md)
