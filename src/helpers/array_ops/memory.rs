//! Unified memory operations (filter, repeat).

use std::collections::HashMap;

use crate::builder::IRBuilder;
use crate::types::Value;

use super::promote;

/// Unified filter. Always produces a dynamic array (runtime-dependent length).
pub fn filter(b: &mut IRBuilder, val: &Value, args: &[Value]) -> Value {
    let d = match val {
        Value::DynamicNDArray(d) => d.clone(),
        _ => promote(b, val),
    };
    crate::ops::dyn_ndarray::memory_ops::dyn_filter(b, &d, args)
}

/// Unified repeat.
pub fn repeat(
    b: &mut IRBuilder,
    val: &Value,
    args: &[Value],
    kwargs: &HashMap<String, Value>,
) -> Value {
    if let Value::DynamicNDArray(d) = val {
        return crate::ops::dyn_ndarray::memory_ops::dyn_repeat(b, d, args, kwargs);
    }

    let repeats_dynamic = args.first().map_or(false, |v| v.int_val().is_none());
    if repeats_dynamic {
        let d = promote(b, val);
        crate::ops::dyn_ndarray::memory_ops::dyn_repeat(b, &d, args, kwargs)
    } else {
        crate::ops::static_ndarray_ops::ndarray_repeat(b, val, args, kwargs)
    }
}
