//! Optional arguments that must be known at compile time (`endpoint=`,
//! `num=`, `axis=`, `k=`, `M=` ...).
//!
//! An absent argument (or `None`) takes NumPy's default. A value known only
//! at run time is rejected: substituting the default for it would compute
//! a different array than NumPy does.

use crate::types::Value;

/// The compile-time Boolean `arg`, or `default` when it is absent.
pub fn static_bool_arg(arg: Option<&Value>, default: bool, what: &str) -> bool {
    match arg.filter(|v| !matches!(v, Value::None)) {
        None => default,
        Some(v) => v
            .bool_val()
            .or_else(|| v.int_val().map(|i| i != 0))
            .unwrap_or_else(|| panic!("{what} must be a compile-time constant")),
    }
}

/// The compile-time integer `arg`, or `default` when it is absent.
pub fn static_int_arg(arg: Option<&Value>, default: i64, what: &str) -> i64 {
    match arg.filter(|v| !matches!(v, Value::None)) {
        None => default,
        Some(v) => v
            .int_val()
            .unwrap_or_else(|| panic!("{what} must be a compile-time integer")),
    }
}
