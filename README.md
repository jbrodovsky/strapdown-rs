# Strapdown - A simple strapdown INS implementation

Strapdown-rs is a straightforward strapdown inertial navigation system (INS) implementation in Rust. It is designed to be simple and easy to understand, making it a great starting point for those interested in learning about or implementing strapdown INS algorithms. It is currently under active development.

## Summary

`strapdown-rs` is a Rust-based software library for implementing strapdown inertial navigation systems (INS). It provides core functionality for processing inertial measurement unit (IMU) data to estimate position, velocity, and orientation using a strapdown mechanization model that is typical of modern systems particularly in the low size, weight, and power (low SWaP) domain (cell phones, drones, robotics, UAVs, UUVs, etc.).

The toolbox is designed for research, teaching, and development purposes and aims to serve the broader robotics, aerospace, and autonomous systems communities. It includes a Rust-native core and optional Python and C bindings. The intent is to provide a high-performance, memory-safe, and cross-platform implementation of strapdown INS algorithms that can be easily integrated into existing systems.

`strapdown-rs` prioritizes correctness, numerical stability, and performance. It is built with extensibility in mind, allowing researchers and engineers to implement additional filtering, sensor fusion, or aiding algorithms on top of the base INS framework. This library is not intended to be a full-featured INS solution, notably it treats IMU output as pre-filtered measurements of relative motion. Additionally, for basic reference, the library includes a basic GPS-based loosely couple INS measurement model, as well as a generic zero-mean normal distribution measurement model. 

## Statement of Need

Strapdown INS implementations are commonly written in MATLAB, Python, or C++, and are typically *proprietary* or are heavily integrated into an existing architecture or framework. This project provides a high-performance, memory-safe, and cross-platform implementation in Rust — a modern systems language well-suited for embedded and real-time applications. Why Rust and why another INS library? Several reasons that are pertinent critiques of each language:

### MATLAB

Many proprietary research INS implementation exist in the maritime, aerospace, and defense engineering sectors which have robust experience with developing and researching INS algorithms. While MATLAB is great for prototyping, it is not suitable for production systems due to performance and deployment issues nor is it a systems programming language. This makes for the common refrain of "prototype in MATLAB, implement in C/C++". This generate additional workload and complexity for industry researchers and engineers as it introduces additional complexity and potential for bugs via the translation process. MATLAB is also antithetical to open science being a proprietary language with a closed-source ecosystem.

### C/C++

While different, C and C++ are often used interchangeably in the context of INS implementations. C is a low-level language that provides direct access to hardware and memory, making it suitable for performance-critical applications. However, C lacks modern features such as memory safety, concurrency, and high-level abstractions, which can lead to complex and error-prone code. C++ offers some of these features but is often criticized for its complexity and steep learning curve. Both languages also have a steep learning curve for those who are not familiar with systems programming. Simply put, these languages do not have the same level of safety and ease of use as higher-level languages like Python or MATLAB. This can make it difficult for researchers and engineers to implement and maintain complex algorithms, especially in the context where performance is still needed. Furthermore, these languages lack modern tooling making managing your dependencies, building, and testing more difficult.

### Python

Python is a great language for rapid prototyping and development, but it is not suitable for performance-critical applications. It also has issues with memory management and real-time constraints due to it's garbage collected nature. While Python is widely used and has many high-performance libraries (namely NumPy) for numerical computing, some applications simply cannot be vectorized appropriately to take advantage of the underlying C libraries or through additional tools like Numba. When running simulations, there is sometimes no avoiding a loop, something Python is notoriously slow at executing.

### Rust

Rust is a modern systems programming language that combines the performance of C/C++ with the safety and concurrency features of higher-level garbage-collected languages. It is designed for performance-critical applications and has a strong focus on memory safety, making it an ideal choice for implementing strapdown INS algorithms. From a scientific development perspective, Rust puts guardrails on scientist-developers who's primary skill set isn't in writing production-grade memory safe code. By following basic good practices, in Rust you get the benefits of modern tooling and language syntax that you get with Python, Java, and Go with the performance of C or C++, with the additional guarantee of, it it compiles, the only bugs are logic bugs.

Frankly, the big selling point of Rust - that of memory safety - is a secondary point. The tooling and ecosystem around it are the key features that prompted it's selection for this library. It was easier to manage dependencies, build, and test the library in Rust than it was in C/C++. Rust also has tools for creating bindings to C, C++, and Python, making it easy to integrate with existing systems. The fact that it is also memory safe is a bonus. This additional bonus is particularly useful for those teams who are actively researching and developing new algorithms to be implmented as it permits a single pivot point for production. R&D teams may work with the Python bindings for rapid prototyping and development, while the production teams can work with the Rust-native core. This allows for a more seamless transition from research to production, as the same codebase can be used in both environments.


### Open Source

The strapdown-rs library is open source, which means that it is freely available for anyone to use, modify, and distribute. This is important for scientific research and development, as it allows researchers to share their work and collaborate with others in the field. Many such comprehensive INS implementations are often developed in-house and are proprietary and closed. Open source software also promotes transparency and reproducibility, which are essential for scientific research. By releasing `strapdown-rs` as an open-source library, it provides a reusable foundation for anyone building INS pipelines, sensor fusion stacks, or GNSS-denied navigation systems — particularly for research involving robotics, aerospace vehicles, or embedded autonomy platforms.

## Functionality

`strapdown-rs` is intended to be both a source code library included into your INS software or simulation environment as well as very light-weight INS analyzer (mostly to demonstrate how to use the library's API). It's functionality includes:

- A 15-state INS model with bias estimation
- Navigation filters for inertial navigation including an Unscented Kalman Filter (UKF) and a particle filter (an error-state EKF is in the works)
- Mechanization equations for velocity and orientation updates
- Support for inertial integration over time (dead reckoning)
- Integration-ready Python bindings (via `pyo3` / `maturin`) and C bindings (via `cbindgen`)
- Examples and unit tests to verify correctness
- A straight forward simulation API for analyzing/testing the INS performance
- Basic GPS-based loosely coupled INS measurement model and corresponding INS simulation
- A generic zero-mean normal distribution measurement model
