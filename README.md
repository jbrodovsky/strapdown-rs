# Strapdown - A simple strapdown INS implementation

Strapdown-rs is a straightforward strapdown inertial navigation system (INS) implementation in Rust. It is designed to be simple and easy to understand, making it a great starting point for those interested in learning about strapdown INS algorithms. It is currently under active development.

# Summary

`strapdown-rs` is a Rust-based software library for implementing strapdown inertial navigation systems (INS). It provides core functionality for processing inertial measurement unit (IMU) data to estimate position, velocity, and orientation using a strapdown mechanization model that is typical of modern systems particularly in the low size, weight, and power (low SWaP) domain (cell phones, drones, robotics, UAVs, UUVs, etc.).

The toolbox is designed for research, teaching, and development purposes and aims to serve the broader robotics, aerospace, and autonomous systems communities. It includes a Rust-native core and optional Python and C bindings. The intent is to provide a high-performance, memory-safe, and cross-platform implementation of strapdown INS algorithms that can be easily integrated into existing systems.

`strapdown-rs` prioritizes correctness, numerical stability, and performance. It is built with extensibility in mind, allowing researchers and engineers to implement additional filtering, sensor fusion, or aiding algorithms on top of the base INS framework.

# Statement of Need

Strapdown INS implementations are commonly written in MATLAB, Python, or C++. This project provides a high-performance, memory-safe, and cross-platform implementation in Rust — a modern systems language well-suited for embedded and real-time applications. Why Rust and why another INS library? Several reasons that are pertinent critiques of each language:

- **MATLAB**: While MATLAB is great for prototyping, it is not suitable for production systems due to performance and deployment issues. It is also not a systems programming language. It is also antithetical to open science being a proprietary language with a closed-source ecosystem.
- **Python**: Python is a great language for rapid prototyping and development, but it is not suitable for performance-critical applications. It also has issues with memory management and real-time constraints. While Python is widely used and has many high-performance libraries (namely NumPy) for numerical computing, some applications simply cannot be vectorized appropriately to take advantage of the underlying C libraries. When running simulations, there is sometimes no avoiding a loop.
- **C++**: C++ is a powerful language for systems programming, but it has a steep learning curve and can be error-prone due to its complexity. It also lacks modern features like memory safety and concurrency support.
- **C**: C is a low-level systems programming language that is widely used for embedded systems and real-time applications. However, it lacks modern features like memory safety and concurrency support, making it more error-prone than other languages. C is also not as expressive as higher-level languages, which can make it harder to write and maintain complex algorithms.
- **Rust**: Rust is a modern systems programming language that combines the performance of C++ with the safety and concurrency features of higher-level languages. It is designed for performance-critical applications and has a strong focus on memory safety, making it an ideal choice for implementing strapdown INS algorithms. From a scientific development perspective, Rust puts guardrails on scientist-developers who's primary skill set isn't in writing production-grade memory safe code. By following basic good practices, in Rust you get the benfits of modern tooling that you get with Python with the performance of C and C++, with the additional guaruntee of, it it compiles, the only bugs are logic bugs.
- **Open Source**: The strapdown-rs library is open source, which means that it is freely available for anyone to use, modify, and distribute. This is important for scientific research and development, as it allows researchers to share their work and collaborate with others in the field. Many such comprehensive INS implementations are often developed in-house and are proprietary and closed. Open source software also promotes transparency and reproducibility, which are essential for scientific research.

By releasing `strapdown-rs` as an open-source library, we provide a reusable foundation for anyone building INS pipelines, sensor fusion stacks, or GNSS-denied navigation systems — particularly for research involving robotics, aerospace vehicles, or embedded autonomy platforms.

# Functionality

`strapdown-rs` is intended to be both a source code library included into your INS software or simulation environement as well as very light-weight INS analyzer (mostly to demonstrate how to use the library's API). It's functionality includes:

- A 15-state INS model with bias estimation
- Mechanization equations for velocity and orientation updates
- Support for inertial integration over time
- Integration-ready Python bindings (via `pyo3` / `maturin`) and C bindings (via `cbindgen`)
- Examples and unit tests to verify correctness
- A straight forward simulation API for analyzing/testing the INS performance