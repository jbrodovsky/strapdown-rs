# Introduction

Welcome to the **Strapdown-rs** documentation! This guide will help you understand and use the Strapdown-rs library, a high-performance Rust implementation of strapdown inertial navigation system (INS) algorithms.

[![Crates.io](https://img.shields.io/crates/v/strapdown-rs.svg)](https://crates.io/crates/strapdown-rs)
[![Documentation](https://docs.rs/strapdown-rs/badge.svg)](https://docs.rs/strapdown-rs)
[![License](https://img.shields.io/crates/l/strapdown-rs.svg)](https://crates.io/crates/strapdown-rs)

## What is Strapdown-rs?

Strapdown-rs is a straightforward strapdown inertial navigation system (INS) implementation in Rust. It is designed to be simple, performant, and easy to understand, making it an excellent starting point for those interested in learning about or implementing strapdown INS algorithms.

The project provides:

1. **A source library** of strapdown INS algorithms and utilities (`strapdown-core`)
2. **A simulation program** for simulating INS performance in various GNSS conditions (`strapdown-sim`)
3. **Experimental geophysical navigation** capabilities for GNSS-denied environments (`strapdown-geonav`)

## Key Features

- **Modern Rust Implementation**: Memory-safe, high-performance code with cross-platform support
- **Multiple Navigation Filters**: Extended Kalman Filter (EKF), Unscented Kalman Filter (UKF), and Particle Filters
- **GNSS Degradation Scenarios**: Simulate dropouts, reduced update rates, and measurement corruption
- **Geophysical Navigation**: Experimental work on gravity and magnetic anomaly navigation
- **Comprehensive Documentation**: Well-documented code with examples and tutorials
- **Research-Focused**: Designed for research, teaching, and development purposes

## Target Audience

This library is intended for:

- **Researchers** working on navigation systems and sensor fusion
- **Engineers** developing autonomous systems, robotics, or aerospace applications
- **Students** learning about inertial navigation and state estimation
- **Developers** needing a high-performance INS implementation in Rust

## Project Status

Strapdown-rs is under active development as part of ongoing PhD research. The project prioritizes correctness, numerical stability, and performance while maintaining extensibility for researchers and engineers.

## Getting Help

- **User Guide**: Start with the [Quick Start](./quick-start.md) guide
- **API Documentation**: See the [API Reference](./api/core.md) for detailed function documentation
- **Examples**: Check out the [Examples and Tutorials](./examples/configurations.md)
- **Issues**: Report bugs or request features on [GitHub](https://github.com/jbrodovsky/strapdown-rs/issues)

## Citation

If you use Strapdown-rs in your research, please cite:

[![JOSS Paper](https://joss.theoj.org/papers/5079592cc860d1435482a4a7764edcd4/status.svg)](https://joss.theoj.org/papers/5079592cc860d1435482a4a7764edcd4)

## License

Strapdown-rs is licensed under the MIT License. See the [LICENSE](https://github.com/jbrodovsky/strapdown-rs/blob/main/LICENSE) file for details.
