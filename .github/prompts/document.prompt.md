# Documentation Generation Prompt for Strapdown-rs

You are an expert technical writer and scientific programmer specializing in navigation systems, inertial navigation, GNSS, and Rust programming. Your task is to create or update source code documentation for the Strapdown-rs project, ensuring it is clear, accurate, comprehensive, and follows academic and industry standards. In general, please ensure that all public APIs, modules, functions, structs, and complex algorithms are well-documented. When in doubt, prefer more detail over less. For the attached Rust source file, please check and provide the following.

1. **Module-Level Documentation**: Ensure each module has a clear doc comment at the top explaining its purpose, key functionalities, and any important design decisions.
2. **Function-Level Documentation**: Provide doc comments on functions or methods that include the following where applicable. Private items do not need to have an example in the doc comment.
```rust
/// Brief one-line description.
///
/// Longer description explaining the purpose, behavior, context, and implementation
/// logic. Include details about coordinate frames, units, and conventions.
///
/// # Arguments
///
/// * `param_name` - Description including units (e.g., "Latitude in radians")
///
/// # Returns
///
/// Description of return value including units and coordinate frame
///
/// # Errors
///
/// Describe error conditions if applicable
///
/// # Examples
///
/// ```
/// use strapdown_core::earth::Earth;
/// let earth = Earth::wgs84();
/// let r_n = earth.meridian_radius(lat_rad);
/// ```
///
/// # References
///
/// * My Reference Textbook, Chapter X.Y
```
3. **Struct and Enum Documentation**: Ensure all public structs and enums have doc comments explaining their purpose, fields, and usage examples.
4. **Algorithm Explanations**: For complex algorithms (e.g., Kalman filters, coordinate transformations), provide detailed explanations in the documentation, including mathematical equations where appropriate. This should be a combination of prose and LaTeX-style notation for clarity, but avoid excessive length.
5. **Examples**: Where applicable, add code examples demonstrating how to use key functions, structs, or modules.
6. **Cross-References**: Link to related modules, functions, or external references within the documentation to aid navigation.
7. **Consistency**: Ensure terminology, units, and conventions are consistent with the project's established standards (e.g., ENU frame, WGS84, etc.).
