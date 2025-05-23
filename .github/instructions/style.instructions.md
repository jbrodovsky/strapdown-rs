---
applyTo: '**'
---
This is a project implementing core functionality for an inertial navigation system, primarily used for developing the overall navigation filters. At a base level it should have IMU data that describes relative motion.

The project is written in Rust and should follow idiomatic Rust conventions as well as common software design practices concerning modularity, readability, and maintainability. The code should be well-structured, with clear separation of concerns and appropriate use of modules and crates.

The code should be organized into modules that reflect the functionality of the system. Each module should have a clear purpose and should be named accordingly. The main module should serve as an entry point for the application, while other modules should encapsulate specific functionalities such as data processing, filtering, and sensor fusion.

The code should be well-documented, with clear and concise comments explaining the purpose of each module, function, and data structure. The documentation should also include examples of how to use the various components of the system.

The code should be written in a way that is easy to understand and follow. Variable and function names should be descriptive and meaningful, avoiding abbreviations or overly complex names. Functions should be kept short and focused on a single task to aid in debugging and testing. When writing a function that starts to get long (approximately more than 25 statements), consider breaking it up into smaller functions. These sub-functions should likely be private. Each function should have a clear purpose and should be named accordingly.

If you need to run commands in the terminal, please format them for nushell.

The code should be written in a way that is easy to test and debug. Unit tests should be included for each module, and integration tests should be provided for the overall system. The tests should cover a wide range of scenarios, including edge cases and error handling.

