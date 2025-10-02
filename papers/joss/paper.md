---
title: 'strapdown-rs: A Simple Strapdown INS Implementation in Rust'
tags:
    - strapdown-ins
    - inertial-navigation
    - rust
    - robotics
    - aerospace
authors:
    - name: James Brodovsky
      orcid: 0000-0002-1371-9044
      equal-contrib: true
      corresponding: true
      affiliation: 1
affiliations:
    - name: Temple University, United States
      index: 1
date: 5 June 2025
bibliography: paper.bib
---

## Summary

Inertial navigation systems (INSs) are critical for many applications in robotics, aerospace, and autonomous systems. They provide real-time estimates of position, velocity, and orientation using data from an inertial measurement unit (IMU) as well as other aiding sensors like GPS. Strapdown implementations of INSs are becoming increasingly common due to the proliferation of modern IMUs that do not require gyroscopic stabilization such as micro-electromagnetic systems (MEMS), fiber optic gyroscopes (FOGs), and ring laser gyroscopes (RLGs). These IMUs are popular for their lower size, weight, and power (SWaP) characteristics, making them suitable for a wide range of applications including drones, robotics, and mobile devices. FOGs and RLGs in particular are becoming increasingly popular in the aerospace, maritime, and defense sectors due to their improved performance and lower cost. That said, such INSs are still highly reliant on GNSS position fixes for long-term accuracy, as even high-quality IMUs will drift over time without external corrections.

`strapdown-rs` is a Rust-based software library and simulation toolkit for implementing and testing strapdown inertial navigation systems (INS). It provides core functionality for processing inertial measurement unit (IMU) data to estimate position, velocity, and orientation using a strapdown mechanization model that is typical of modern systems, particularly in the low SWaP domain (cell phones, drones, robotics, UAVs, UUVs, etc.). Additionally, it provides a comprehensive simulation framework for safely testing INS performance under degraded and denied GNSS conditions—scenarios increasingly relevant for autonomous systems operating in contested or signal-denied environments.

`strapdown-rs` prioritizes correctness, numerical stability, and performance. It is built with extensibility in mind, allowing researchers and engineers to implement additional filtering, sensor fusion, or aiding algorithms on top of the base INS framework. This library is not intended to be a full-featured INS solution. Notably it does not have code for processing raw IMU or GPS signals from hardware and currently only implements a loosely-coupled INS.

The toolbox is designed for research, teaching, and development purposes and aims to serve the broader robotics, aerospace, and autonomous systems communities. The intent is to provide a high-performance, memory-safe, and modern cross-platform implementation of strapdown INS algorithms that can be easily integrated into existing systems or research workflows. The simulation framework enables users to generate synthetic navigation scenarios with controllable GNSS degradation, allowing for systematic testing and verification of INS algorithms under various failure modes.

## Statement of Need

Existing inertial navigation system (INS) implementations are often fragmented across proprietary MATLAB code, legacy C/C++ systems, and Python prototypes—each with significant drawbacks for open, reproducible, and high-performance research. MATLAB and Python are widely used for prototyping but are unsuitable for production or real-time systems, while C/C++ implementations are difficult to maintain and extend, lack modern tooling, and are rarely open source. This landscape forces researchers to translate algorithms between languages or pay for expensive commercial solutions, increasing complexity and the risk of errors, and makes it challenging to build reusable, extensible, and high-performance INS libraries. There is a clear need for a modern, open, and memory-safe toolkit that bridges the gap between rapid prototyping and robust deployment, enabling both research and production use without sacrificing performance, safety, or reproducibility. `strapdown-rs` addresses this gap by providing a cross-platform, open-source INS library and simulation framework that is both accessible and suitable for high-performance applications.

Reliable and reproducible simulation of GNSS-denied scenarios is a critical need for the navigation community, as collecting real-world data under signal denial typically requires expensive and logistically challenging field tests with specialized jamming hardware. Such tests are not only costly and difficult to repeat, but also raise regulatory and safety concerns, limiting their accessibility for most researchers. A robust simulation framework enables systematic evaluation of navigation algorithms under a wide range of degraded or denied GNSS conditions, supporting fair comparison, rapid iteration, and transparent reporting of results. By providing configurable, open-source tools for simulating GNSS outages, degradations, and spoofing, `strapdown-rs` empowers researchers to develop and validate robust navigation solutions without the barriers of hardware-based field testing.

## Overview of Functionality

`strapdown-rs` provides both core navigation algorithms and simulation capabilities through two main components: the `strapdown-core` library and the `strapdown-sim` binary. The library contains four primary modules: `earth`, `strapdown`, `filter`, and `sim` (for simulation utilities). The binary provides a command-line interface for running navigation simulations with configurable GNSS degradation scenarios.

### Core Library Modules

The `earth` module contains constants and functions related to the Earth's shape and other geophysical features (gravity and magnetic field). The Earth is modeled as an ellipsoid with a semi-major axis and a semi-minor axis [@wgs84]. The Earth's gravity is modeled as a function of the latitude and altitude using the Somigliana method. The Earth's magnetic field is modeled using a dipole model [@wmm]. This module relies on the nav-types crate [@nav-types] for the coordinate types and conversions, but provides additional functionality for calculating rotations for the strapdown navigation filters. This permits the transformation of additional quantities (velocity, acceleration, etc.) between the Earth-centered Earth-fixed (ECEF) frame and the local-level frame.

The `strapdown` module provides the forward mechanization equations for strapdown inertial navigation systems. It provides a set of structs for modeling both IMU data and the base nine-element strapdown state (latitude, longitude, and altitude; velocities north, east, and down; and attitude). It includes an implementation for the local-level frame forward mechanization, which is a common approach for strapdown INS and follows the equations from Chapter 5.4 of [@groves]. This module serves as the foundation for both dead reckoning and filtered navigation solutions.

The `filter` module contains the core functionality for implementing strapdown INS algorithms, primarily a loosely-coupled integration architecture according to Chapter 14.1.2 of [@groves]. This module contains implementations of various inertial navigation filters, including an Unscented Kalman Filter (UKF) and particle filter. These filters are used to estimate the state of a strapdown inertial navigation system based on IMU measurements and other sensor data. The filters use the strapdown equations (provided by the strapdown module) to propagate the state in the local level frame. The module also provides measurement models for GPS position and velocity updates.

The `sim` module provides utilities for simulation and testing, including data structures for test data records (compatible with the Sensor Logger app format), navigation results, and simulation functions for both dead reckoning and closed-loop navigation. Critically, this module includes the event stream framework that enables simulation of GNSS degradation scenarios.

The `messages` module implements an event-driven architecture for simulating GNSS degradation. It allows users to convert data contained in a tabular formate (i.g. CSV files) into a sequence of timestamped events that can be processed by the navigation filters. This module provides:

- **GNSS Schedulers**: Control measurement availability through passthrough (all measurements), fixed-interval sampling, or duty-cycle patterns (on/off periods)
- **Fault Models**: Simulate measurement corruption through various mechanisms including degraded measurements (AR(1) noise processes), slow bias drift (position/velocity drift with rotation), and hijacking/spoofing (position offset injection)
- **Event Streams**: Transform raw sensor data into sequences of IMU prediction steps and measurement updates, with fault injection applied according to the configured degradation model

### Simulation Binary

The `strapdown-sim` binary provides a command-line interface for running navigation simulations. It can be operated in two modes. In open-loop mode (dead reckoning) the system propagates the state using the strapdown equations and IMU measurements without any corrections from aiding sensors. This mode is only recommended when analyzing high-quality IMUs, as MEMS-grade sensors will accumulate significant drift within seconds to minutes. In closed-loop mode (GNSS-aided INS) the system uses a 15-state UKF (9 navigation states + 6 IMU bias states) to estimate the state and correct it using GPS measurements. This mode supports extensive GNSS degradation simulation capabilities that can be used to evaluate the robustness of navigation algorithms under various failure modes such as:

- **Signal dropouts**: Simulated via duty-cycle schedulers with configurable on/off periods
- **Reduced update rates**: Implemented through fixed-interval schedulers
- **Measurement degradation**: AR(1) colored noise processes applied to position and velocity measurements with configurable correlation and noise levels
- **Slow bias/drift**: Gradual position and velocity bias accumulation, optionally with rotation to simulate realistic drift patterns
- **Spoofing/hijacking**: Position offset injection over specified time windows to simulate spoofing attacks

These degradation scenarios can be configured individually or combined, and can be specified either through command-line arguments or configuration files (JSON, YAML, or TOML). This enables systematic testing of INS performance under controlled degradation conditions—critical for developing robust navigation systems for contested environments.

## Key Technical Contributions

The primary technical contributions of `strapdown-rs` to the open-source navigation community are a 

1. A well-documented, numerically stable implementation of the local-level frame strapdown equations following [@groves], providing a reusable foundation for navigation algorithm development.

2. A reference implementation of a 15-state UKF-based INS with GPS position and velocity aiding, suitable for MEMS-grade IMUs commonly found in robotics and low-SWaP applications.

3. A comprehensive toolkit for safely simulating INS performance under various GNSS degradation scenarios without requiring real-world testing in denied environments. The event-driven architecture separates concerns between measurement scheduling, fault injection, and navigation filtering, enabling researchers to systematically evaluate algorithm robustness.

These contributions make `strapdown-rs` particularly valuable for researchers developing alternative or complimentary navigation algorithms for autonomous systems that must operate in contested or signal-denied environments, such as urban canyons, indoor spaces, or GPS-jammed areas.

## References
