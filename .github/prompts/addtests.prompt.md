You are an expert Rust developer working on the `strapdown-rs` project. Your goal is to add comprehensive unit tests to the active file, aiming for >= 90% code coverage. Use `cargo tarpaulin` to measure coverage and ensure edge cases, error conditions, and typical usage scenarios are tested.

## Project Guidelines
- **Location**: Unit tests must be placed in the same file as the code they test, within a `#[cfg(test)] mod tests { ... }` block.
- **Naming**: Use descriptive names for test functions that indicate the scenario and expected outcome.
- **Assertions**: Use `assert_eq!` and `assert!` macros. For floating-point comparisons, ensure you handle precision correctly (e.g., checking the difference is within a small epsilon). You may also use the `assert_approx_eq` crate for better floating-point assertions.

## Instructions
1.  **Analyze the Code**:
    -   Identify all public functions and critical private helper functions.
    -   Determine the "happy path" (expected valid usage).
    -   Identify edge cases (boundary values, empty collections, zeros, etc.).
    -   Identify error conditions (Result::Err, Option::None).

2.  **Generate Tests**:
    -   Create a `#[cfg(test)] mod tests` block if one does not exist.
    -   Import the parent module's contents: `use super::*;`.
    -   Write a test function for each identified case.
    -   Ensure tests are self-contained and do not depend on external state (files, network) unless mocked.

3.  **Coverage Strategy**:
    -   Ensure every branch of `if/else` and `match` statements is exercised.
    -   Test loop boundaries (0 iterations, 1 iteration, many iterations).
    -   Aim to execute every line of code in the file at least once.

4.  **Output**:
    -   Provide the complete code for the `tests` module.
    -   If modifying existing tests, use `// ...existing code...` to indicate unchanged parts.

Please generate the unit tests now.