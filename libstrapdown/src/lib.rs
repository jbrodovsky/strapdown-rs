use std::os::raw::c_double;

#[unsafe(no_mangle)]
pub extern "C" fn ins_add(a: c_double, b: c_double) -> c_double {
    strapdown_core::add(a, b)
}