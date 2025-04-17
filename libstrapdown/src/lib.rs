use std::os::raw::c_double;

#[unsafe(no_mangle)]
pub extern "C" fn ins_add(a: c_double, b: c_double) -> c_double {
    strapdown::add(a, b)
}