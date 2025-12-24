# User Guide Overview

Welcome to the Strapdown-rs User Guide! This section provides comprehensive information on using the library and simulation tools.

## What You'll Learn

This guide covers:

1. **Core Concepts**: Understanding strapdown INS, coordinate frames, and state representation
2. **Running Simulations**: How to use the `strapdown-sim` binary
3. **Data Formats**: Preparing and formatting your input data
4. **Configuration**: Setting up simulations with TOML config files
5. **Logging and Debugging**: Monitoring simulation progress

## Quick Navigation

### For Beginners

If you're new to strapdown INS or this library:

1. Start with [Core Concepts](./concepts.md) to understand the fundamentals
2. Learn about [Coordinate Frames](./coordinate-frames.md) used in the library
3. Review [State Representation](./state-representation.md) to understand the 9-state and 15-state models
4. Try the [Quick Start](../quick-start.md) tutorial

### For Experienced Users

If you're familiar with INS and want to jump in:

1. Check [Input Data Format](./data-format.md) to prepare your data
2. Review [Configuration Files](./configuration.md) for advanced options
3. Explore [Running Simulations](./simulations.md) for different modes
4. See [Logging](./logging.md) for debugging and monitoring

## Simulation Modes

Strapdown-rs supports three main simulation modes:

### Open-Loop (Dead Reckoning)

Pure inertial navigation without corrections. Useful for:
- Understanding INS error growth
- Baseline comparisons
- Testing IMU data quality

See: [Open-Loop Mode](./open-loop.md)

### Closed-Loop (Kalman Filtering)

INS with GNSS corrections using EKF or UKF. Best for:
- Realistic navigation scenarios
- GNSS degradation studies
- Production-like simulations

See: [Closed-Loop Mode](./closed-loop.md)

### Particle Filter

Non-parametric Bayesian filtering for non-Gaussian distributions. Useful for:
- Multimodal uncertainty
- Highly nonlinear scenarios
- Research applications

See: [Particle Filter Mode](./particle-filter.md)

## Navigation Filters

The library provides multiple filter implementations:

- **Extended Kalman Filter (EKF)**: Fast, efficient, works well for mildly nonlinear systems
- **Unscented Kalman Filter (UKF)**: Better accuracy for nonlinear systems, 2-3x slower
- **Particle Filter**: Handles non-Gaussian distributions, computationally intensive
- **Rao-Blackwellized Particle Filter (RBPF)**: Hybrid approach combining particles and EKF

Learn more: [Navigation Filters](../filters/kalman.md)

## State Models

### 9-State Model

The basic navigation-only model:
- **Position**: latitude, longitude, altitude
- **Velocity**: north, east, down
- **Attitude**: roll, pitch, yaw

### 15-State Model

Extended model with IMU bias estimation:
- 9 navigation states (as above)
- **Accelerometer biases**: 3 states
- **Gyroscope biases**: 3 states

The 15-state model provides better long-term accuracy by estimating and correcting sensor biases.

## Typical Workflow

1. **Collect or prepare IMU/GNSS data** in CSV format
2. **Create a configuration file** specifying simulation parameters
3. **Run the simulation** using `strapdown-sim`
4. **Analyze results** from the output CSV
5. **Iterate** by adjusting parameters as needed

## Common Use Cases

### Research and Development

- Testing new navigation algorithms
- Comparing filter performance
- Studying error characteristics
- Publishing research results

### Education

- Teaching INS fundamentals
- Demonstrating sensor fusion
- Illustrating error sources
- Hands-on learning

### System Development

- Prototyping navigation systems
- Evaluating sensor requirements
- Testing GNSS-denied scenarios
- Performance benchmarking

## Getting Help

- **FAQ**: Check the [Frequently Asked Questions](../faq.md)
- **Examples**: Browse [Example Configurations](../examples/configurations.md)
- **API Docs**: See [API Reference](../api/core.md) for detailed documentation
- **Issues**: Report problems on [GitHub](https://github.com/jbrodovsky/strapdown-rs/issues)

## Next Steps

Choose your path:

- **New to INS?** → [Core Concepts](./concepts.md)
- **Ready to simulate?** → [Running Simulations](./simulations.md)
- **Need data format info?** → [Input Data Format](./data-format.md)
- **Want advanced features?** → [Configuration Files](./configuration.md)
