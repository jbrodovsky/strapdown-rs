use strapdown::add as add_strapdown;

/// Adds two unsigned integers.
/// This function is exported for C/C++ usage.
#[unsafe(no_mangle)]
pub extern "C" fn strapdown_add(left: f64, right: f64) -> f64 {
    add_strapdown(left, right)
}