# Strapdown - A simple strapdown INS implementation

HTML: <a href="https://joss.theoj.org/papers/5079592cc860d1435482a4a7764edcd4"><img src="https://joss.theoj.org/papers/5079592cc860d1435482a4a7764edcd4/status.svg"></a>

Markdown: [![status](https://joss.theoj.org/papers/5079592cc860d1435482a4a7764edcd4/status.svg)](https://joss.theoj.org/papers/5079592cc860d1435482a4a7764edcd4)

[![Crates.io](https://img.shields.io/crates/v/strapdown-rs.svg)](https://crates.io/crates/strapdown-rs)
[![Documentation](https://docs.rs/strapdown-rs/badge.svg)](https://docs.rs/strapdown-rs)
[![License](https://img.shields.io/crates/l/strapdown-rs.svg)](https://crates.io/crates/strapdown-rs)

Strapdown-rs is a straightforward strapdown inertial navigation system (INS) implementation in Rust. It is designed to be simple and easy to understand, making it a great starting point for those interested in learning about or implementing strapdown INS algorithms. It is currently under active development.

## Installation

To use `strapdown-rs`, you can add it as a dependency in your `Cargo.toml` file: `cargo add strapdown-rs` or install it directly via `cargo install strapdown-rs`.

## Summary

`strapdown-rs` is a Rust-based software library for implementing strapdown inertial navigation systems (INS). It provides core functionality for processing inertial measurement unit (IMU) data to estimate position, velocity, and orientation using a strapdown mechanization model that is typical of modern systems particularly in the low size, weight, and power (low SWaP) domain (cell phones, drones, robotics, UAVs, UUVs, etc.).

`strapdown-rs` prioritizes correctness, numerical stability, and performance. It is built with extensibility in mind, allowing researchers and engineers to implement additional filtering, sensor fusion, or aiding algorithms on top of the base INS framework. This library is not intended to be a full-featured INS solution, notably it treats IMU output as pre-filtered measurements of relative motion. The toolbox is designed for research, teaching, and development purposes and aims to serve the broader robotics, aerospace, and autonomous systems communities. The intent is to provide a high-performance, memory-safe, and cross-platform implementation of strapdown INS algorithms that can be easily integrated into existing systems.

For basic reference, the library includes a basic GPS-based loosely couple INS measurement model, as well as a generic zero-mean normal distribution measurement model.

## Statement of Need

Strapdown INS implementations are commonly written in MATLAB, Python, or C++, and are typically *proprietary* or are heavily integrated into an existing architecture or framework. This project provides a high-performance, memory-safe, and cross-platform implementation in Rust — a modern systems language well-suited for embedded and real-time applications. Why Rust and why another INS library? Several reasons that are pertinent critiques of each language:

### MATLAB

Many proprietary research INS implementation exist in the maritime, aerospace, and defense engineering sectors which have robust experience with developing and researching INS algorithms. While MATLAB is great for prototyping, it is not suitable for production systems due to performance and deployment issues nor is it a systems programming language. This makes for the common refrain of "prototype in MATLAB, implement in C/C++". This generate additional workload and complexity for industry researchers and engineers as it introduces additional complexity and potential for bugs via the translation process. MATLAB is also antithetical to open science being a proprietary language with a closed-source ecosystem.

### C/C++

While different, C and C++ are often used interchangeably in the context of INS implementations. C is a low-level language that provides direct access to hardware and memory, making it suitable for performance-critical applications. However, C lacks modern features such as memory safety, concurrency, and high-level abstractions, which can lead to complex and error-prone code. C++ offers some of these features but is often criticized for its complexity and steep learning curve. Both languages also have a steep learning curve for those who are not familiar with systems programming. Simply put, these languages do not have the same level of safety and ease of use as higher-level languages like Python or MATLAB. This can make it difficult for researchers and engineers to implement and maintain complex algorithms, especially in the context where performance is still needed. Furthermore, these languages lack modern tooling making managing your dependencies, building, and testing more difficult.

### Python

Python is a great language for rapid prototyping and development, but it is not suitable for performance-critical applications. It also has issues with memory management and real-time constraints due to it's garbage collected nature. While Python is widely used and has many high-performance libraries (namely NumPy) for numerical computing, some applications simply cannot be vectorized appropriately to take advantage of the underlying C libraries or through additional tools like Numba. When running simulations, there is sometimes no avoiding a loop, something Python is notoriously slow at executing.

Specifically, one algorithm that is frequently used in navigation is a Particle Filter (PF). Particle filters are often used for state estimation in non-linear systems, and they require a large number of particles to be effective, particularly when used in systems with large state vectors. This introduces the primary problem that motivated the development of `strapdown-rs`. It makes sense to re-use the same code for typical local-level frame forward mechanization. This is a standard set of equations that can be used in multiple different INS architectures. For a Kalman Filter based INS, this is relatively simple and you can typically use whatever language's linear algebra library you prefer to store the data. You can also do the same for the particle filter, and have list of vectors that represent the particles. However, this forces you into the trap of Python: iterating through the the list. 

Alternatively, you could vectorize the operations, but this introduces additional complexity and requires you to test and verify that the vectorize operations match the original forward mechanization equations. This makes it difficult to swap out the filtering algorithm or the forward mechanization equations without rewriting large portions of the code.

Thus what is needed is a modern, compiled, systems programming language with a useful linear algebra library.

### Rust

Rust is a modern systems programming language that combines the performance of C/C++ with the safety and concurrency features of higher-level garbage-collected languages. It is designed for performance-critical applications and has a strong focus on memory safety, making it an ideal choice for implementing strapdown INS algorithms. From a scientific development perspective, Rust puts guardrails on scientist-developers who's primary skill set isn't in writing production-grade memory safe code. By following basic good practices in Rust, you get the benefits of modern tooling and language syntax that you get with Python, Java, and Go with the performance of C or C++, with the additional guarantee of if it compiles, the only bugs are *logic* bugs.

### Open Source

The `strapdown-rs` library is open source, which means that it is freely available for anyone to use, modify, and distribute. This is important for scientific research and development, as it allows researchers to share their work and collaborate with others in the field. Many such comprehensive INS implementations are often developed in-house, are proprietary, and closed source. Open source software also promotes transparency and reproducibility, which are essential for scientific research. By releasing `strapdown-rs` as an open-source library, it provides a reusable foundation for anyone building INS pipelines, sensor fusion stacks, or GNSS-denied navigation systems — particularly for research involving robotics, aerospace vehicles, or embedded autonomy platforms.

## Functionality

`strapdown-rs` is intended to be both a source code library included into your INS software or simulation environment as well as very light-weight INS analyzer (mostly to demonstrate how to use the library's API). It's functionality includes:

- A 15-state INS model with IMU bias estimation
- Navigation filters for inertial navigation including an Unscented Kalman Filter (UKF) and a particle filter (an error-state EKF is in the works)
- Mechanization equations for position, velocity, and orientation updates
- Examples and unit tests to verify correctness along with a basic reference dataset
- A straight forward simulation API for analyzing/testing the INS performance
- A simple CLI tool for running an open-loop (dead reckoning) or closed-loop (full state UKF) INS simulation
