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

Inertial navigation systems (INSs) are critical for many applications in robotics, aerospace, and autonomous systems. They provide real-time estimates of position, velocity, and orientation using data from an inertial measurement unit (IMU) as well as other aiding sensors like GPS. Strapdown implementations of INSs are becoming increasingly common due to the proliferation of modern IMUs that do not require gyroscopic stabilization such as micro-electromagnetic systems (MEMS), fiber optic gyroscopes (FOGs), and ring laser gyroscopes (RLGs). These IMUs are popular for their lower size, weight, and power (SWaP) characteristics compared to historical spinning-mass gyroscopes and pendulous accelerometers, making them suitable for a wide range of applications including drones, robotics, and mobile devices. FOGs and RLGs in particular are becoming increasingly popular in the aerospace, maritime, and defense sectors due to their improved performance and lower cost. That said, such INSs are still highly reliant on GNSS position fixes for long-term accuracy, as even high-quality IMUs will drift over time without external corrections.

`strapdown-rs` is a Rust-based software library and simulation toolkit for implementing and testing strapdown inertial navigation systems (INS). It provides core functionality for processing inertial measurement unit (IMU) data to estimate position, velocity, and orientation using a strapdown mechanization model that is typical of modern systems, particularly in the low SWaP domain (cell phones, drones, robotics, UAVs, UUVs, etc.). Additionally, it provides a comprehensive simulation framework for safely testing INS performance under degraded and denied GNSS conditions—scenarios increasingly relevant for autonomous systems operating in contested or signal-denied environments.

`strapdown-rs` prioritizes correctness, numerical stability, and performance. It is built with extensibility in mind, allowing researchers and engineers to implement additional filtering, sensor fusion, or aiding algorithms on top of the base INS framework. This library is not intended to be a full-featured INS solution. Notably it does not have code for processing raw IMU or GPS signals from hardware and currently only implements a loosely-coupled INS.

The toolbox is designed for research, teaching, and development purposes and aims to serve the broader robotics, aerospace, and autonomous systems communities. The intent is to provide a high-performance, memory-safe, and modern cross-platform implementation of strapdown INS algorithms that can be easily integrated into existing systems or research workflows. The simulation framework enables users to generate synthetic navigation scenarios with controllable GNSS degradation, allowing for systematic testing and verification of INS algorithms under various failure modes.

## Statement of Need

Existing inertial navigation system (INS) implementations are often fragmented across proprietary MATLAB code, legacy C/C++ systems, and Python prototypes—each with significant drawbacks for open, reproducible, and high-performance research. MATLAB and Python are widely used for prototyping but are unsuitable for production or real-time systems, while C/C++ implementations are difficult to maintain and extend, lack modern tooling, and are rarely open source. This landscape forces researchers to translate algorithms between languages or pay for expensive commercial solutions, increasing complexity and the risk of errors, and makes it challenging to build reusable, extensible, and high-performance INS libraries. The motivating example for the development of this library is the notion of "vectorized" calculations.

Research in navigation algorithms, particularly alternative Positioning, Navigation, and Timing (PNT) methods, often necessitates the evaluation of other filter architectures such as particle filters or factor graph methods. Implementing these algorithms in an object-oriented manner in Python or MATLAB is straightforward, but the performance constraints make it difficult to evaluate them under realistic conditions. A common workaround is to implement the mathematics in a vectorized manner using NumPy or MATLAB's array operations. This can provide significant performance improvements, but it also introduces complexity, duplicates functionality, and can make the code less transparent and harder to debug. Additionally, vectorized implementations may not be suitable for real-time applications. There is a clear need for a modern, open, and high-performance toolkit that bridges the gap between rapid prototyping and compiled performance with a production environment in mind, enabling both research and production use without sacrificing performance, safety, or reproducibility. `strapdown-rs` addresses this gap by providing an INS library and simulation framework that is both accessible and suitable for high-performance applications.

The second need motivating the development of this library is to produce reliable and reproducible simulation of GNSS-denied scenarios. Such scenarios are a critical need for the navigation community, as collecting real-world data under signal denial typically requires expensive and logistically challenging field tests with specialized jamming hardware. Such tests are not only costly and difficult to repeat, but also raise regulatory and safety concerns, limiting their accessibility for most researchers. A robust simulation framework enables systematic evaluation of navigation algorithms under a wide range of degraded or denied GNSS conditions, supporting fair comparison, rapid iteration, and transparent reporting of results. By providing configurable, open-source tools for simulating GNSS outages, degradation, and spoofing, `strapdown-rs` empowers researchers to develop and validate robust navigation solutions without the barriers of hardware-based field-testing.

## State of the Field

Several open-source tools exist for inertial navigation research, but each carries significant limitations for the use cases addressed by `strapdown-rs`. The `gnss-ins-sim` Python package [@gnss-ins-sim] offers a simulation environment that produces similar outputs but is limited to Python's performance constraints and does not support GNSS degradation fault modeling. GTSAM [@gtsam] provides a powerful factor-graph framework in C++ and Python, but is a more general-purpose computer vision library rather than a dedicated INS toolkit, requiring substantial configuration overhead for straightforward strapdown navigation tasks. OpenVINS [@geneva2020openvins] is a high-quality visual-inertial odometry system, but is tightly coupled to ROS and oriented toward camera-IMU fusion rather than GNSS-aided INS research. Numerous MATLAB implementations exist in academic repositories, but these are proprietary, non-portable, and unsuitable for deployment in performance-sensitive applications.

No existing open-source tool in a single package provides: a standalone library for strapdown mechanization suitable for production, a configurable GNSS degradation simulation framework with reproducible, seeded scenarios, and a configurable INS implementation. Contributing these capabilities to an existing project was not viable because the language choice, which is central to the safety and performance goals of this work, is incompatible with Python- or C++-based frameworks. `strapdown-rs` fills this gap as a self-contained, high-performance, reproducible toolkit aimed specifically at GNSS-denied navigation research.

## Overview of Functionality

`strapdown-rs` provides both core navigation algorithms and simulation capabilities through two main components: the `strapdown-core` library and the `strapdown-sim` binary. The library contains four primary modules: `earth`, `strapdown`, `filter`, and `sim` (for simulation utilities). The binary provides a command-line interface for running navigation simulations with configurable GNSS degradation scenarios.

### Core Library Modules

The `earth` module contains constants and functions related to the Earth's shape and other geophysical features (gravity and magnetic field). The Earth is modeled as an ellipsoid with a semi-major axis and a semi-minor axis [@wgs84]. The Earth's gravity is modeled as a function of the latitude and altitude using the Somigliana method. The Earth's magnetic field is modeled using a dipole model [@wmm]. This module relies on the nav-types crate [@nav-types] for the coordinate types and conversions, but provides additional functionality for calculating rotations for the strapdown navigation filters. This permits the transformation of additional quantities (velocity, acceleration, etc.) between the Earth-centered Earth-fixed (ECEF) frame and the local-level frame.

The `strapdown` module provides the forward mechanization equations for strapdown inertial navigation systems. It provides a set of structs for modeling both IMU data and the base nine-element strapdown state (latitude, longitude, and altitude; velocities north, east, and down; and attitude). It includes an implementation for the local-level frame forward mechanization, which is a common approach for strapdown INS and follows the equations from Chapter 5.4 of [@groves]. This module serves as the foundation for both dead reckoning and filtered navigation solutions.

The `filter` module contains the core functionality for implementing strapdown INS algorithms, primarily a loosely-coupled integration architecture according to Chapter 14.1.2 of [@groves]. This module contains implementations of various inertial navigation filters, including an Unscented Kalman Filter (UKF) and particle filter. These filters are used to estimate the state of a strapdown inertial navigation system based on IMU measurements and other sensor data. The filters use the strapdown equations (provided by the strapdown module) to propagate the state in the local level frame. The module also provides measurement models for GPS position and velocity updates.

The `sim` module provides utilities for simulation and testing, including data structures for test data records (compatible with the Sensor Logger app format), navigation results, and simulation functions for both dead reckoning and closed-loop navigation. Critically, this module includes the event stream framework that enables simulation of GNSS degradation scenarios.

The `messages` module implements an event-driven architecture for simulating GNSS degradation. It allows users to convert data contained in a tabular format (e.g. CSV files) into a sequence of timestamped events that can be processed by the navigation filters, mimicking real-time sensor data streams and allowing for reproducible testing of degraded input. This module provides:

- **GNSS Schedulers**: Control measurement availability through passthrough (all measurements), fixed-interval sampling, or duty-cycle patterns (on/off periods)
- **Fault Models**: Simulate measurement corruption through various mechanisms including degraded measurements (AR(1) noise processes), slow bias drift (position/velocity drift with rotation), and hijacking/spoofing (position offset injection)
- **Event Streams**: Transform raw sensor data into sequences of IMU prediction steps and measurement updates, with fault injection applied according to the configured degradation model

### Simulation Binary

The `strapdown-sim` binary provides a command-line interface for running navigation simulations and generating synthetic datasets. It can be operated in three modes. In synthetic data generation mode it can generate realistic trajetories from perfect kinematics based on an IMU classification. In open-loop mode (dead reckoning) the system propagates the state using the strapdown equations and IMU measurements without any corrections from aiding sensors. This mode is only recommended when analyzing high-quality IMUs, as MEMS-grade sensors will accumulate significant drift within seconds to minutes. In closed-loop mode (GNSS-aided INS) the system uses a 15-state UKF (9 navigation states + 6 IMU bias states) to estimate the state and correct it using GPS measurements. This mode supports extensive GNSS degradation simulation capabilities that can be used to evaluate the robustness of navigation algorithms under various failure modes such as:

- **Signal dropouts**: Simulated via duty-cycle schedulers with configurable on/off periods
- **Reduced update rates**: Implemented through fixed-interval schedulers
- **Measurement degradation**: AR(1) colored noise processes applied to position and velocity measurements with configurable correlation and noise levels
- **Slow bias/drift**: Gradual position and velocity bias accumulation, optionally with rotation to simulate realistic drift patterns
- **Spoofing/hijacking**: Position offset injection over specified time windows to simulate spoofing attacks

These degradation scenarios can be configured individually or combined, and can be specified either through command-line arguments or configuration files (JSON, YAML, or TOML). This enables systematic testing of INS performance under controlled degradation conditions—critical for developing robust navigation systems for contested environments.

## Software Design

The central design decision in `strapdown-rs` is the choice of Rust as the implementation language. Rust provides memory safety guarantees without garbage collection, eliminating entire classes of bugs—buffer overflows, use-after-free, and data races—that are common in navigation C/C++ codebases and difficult to detect in safety-critical systems. This comes at no runtime performance cost relative to C/C++, making `strapdown-rs` suitable for future real-time deployment, unlike Python-based alternatives.

The workspace is split into two crates: `strapdown-core` (library) and `strapdown-sim` (CLI binary). This separation ensures that researchers who want only the navigation algorithms—without the simulation scaffolding—can depend on the core library independently. Within `strapdown-core`, trait-based abstractions (`NavigationFilter`, `MeasurementModel`) decouple the filter implementations from specific sensor models, allowing new filters or measurement sources to be added without modifying existing code.

The UKF was chosen over the more common Extended Kalman Filter (EKF) because the local-level frame strapdown equations are nonlinear, and the unscented transform avoids first-order linearization errors without requiring Jacobian derivations. The 9-state core strapdown state is extended to 15 states in the UKF to estimate IMU accelerometer and gyroscope biases, which are the dominant error sources for MEMS-grade sensors. The event-driven GNSS simulation architecture deliberately separates measurement scheduling, fault injection, and filter update logic into independent components, enabling systematic scenario construction and reproducible experiments.

## Key Technical Contributions

The primary technical contributions of `strapdown-rs` to the open-source navigation community are:

1. A well-documented, numerically stable implementation of the local-level frame strapdown equations following [@groves], providing a reusable foundation for navigation algorithm development.
1. A reference implementation of a 15-state UKF-based INS with GPS position and velocity aiding, suitable for MEMS-grade IMUs commonly found in robotics and low-SWaP applications.
1. A comprehensive toolkit for safely simulating INS performance under various GNSS degradation scenarios without requiring real-world testing in denied environments. The event-driven architecture separates concerns between measurement scheduling, fault injection, and navigation filtering, enabling researchers to systematically evaluate algorithm robustness.

These contributions make `strapdown-rs` particularly valuable for researchers developing alternative or complementary navigation algorithms for autonomous systems that must operate in contested or signal-denied environments, such as urban canyons, indoor spaces, or GPS-jammed areas.

## Research Impact Statement

`strapdown-rs` has been developed in direct support of ongoing navigation research at Temple University. The particle filter implementation and GNSS degradation simulation framework have been used in research exploring geophysical anomaly based navigation in a UKF [@anom-ukf] and Rao-Blackwellized particle filter [@anom-pf] approaches to geophysical anomaly-aided navigation, with associated data processing pipelines and analysis built on top of the `strapdown-core` library. This work has produced processed datasets and reproducible experimental configurations that are archived alongside the software, providing the community-readiness signals that support external replication.

The software's design for reproducibility—seeded random number generation, configuration-file-driven scenarios, and deterministic strapdown propagation—directly addresses a gap identified by the navigation research community: the lack of a publicly available, language-agnostic, reproducible baseline for evaluating GNSS-denied INS algorithms. Researchers can clone the repository, obtain the pre-processed input datasets, and reproduce published simulation results without access to proprietary tools or hardware.

As a Cargo-published crate, `strapdown-core` is available for direct dependency by other Rust projects. The clean trait-based interface for filters and measurement models is designed to lower the barrier for researchers wishing to implement and compare novel filtering algorithms—such as factor graph methods or learned observation models—against the provided UKF baseline, using identical strapdown mechanization and GNSS degradation scenarios.

## AI Usage Disclosure

Generative AI tools, specifically Claude (Anthropic), were used during the development of this software and paper. AI was used in a pair-programming capacity during implementation of several modules in `strapdown-core` and `strapdown-sim`, and as a copy editor for documentation strings and this paper. All AI-generated or AI-assisted code was reviewed by the author, validated against the reference equations in [@groves], and verified through the project's unit and integration test suite (`cargo test --workspace`). All AI-assisted prose was reviewed and edited by the author for technical accuracy. The core algorithmic content (strapdown mechanization equations, UKF sigma-point formulation, and particle filter resampling strategies) is based directly on established references and was verified by the author independently of any AI-generated suggestions.

## References