//! Unified reduction operations (sum, prod, min, max, argmax, argmin, etc.).

use crate::builder::IRBuilder;
use crate::types::Value;

use super::{agg_kind, promote};

/// Unified reduce: handles static arrays, dynamic arrays, static axis,
/// dynamic axis, and no axis.
pub fn reduce(
    b: &mut IRBuilder,
    op: &str,
    val: &Value,
    axis_arg: Option<&Value>,
) -> Value {
    if let Value::DynamicNDArray(d) = val {
        return crate::ops::dyn_ndarray::aggregation::dyn_aggregate(
            b, d, axis_arg, agg_kind(op),
        );
    }

    match axis_arg {
        Some(ax_val) if !matches!(ax_val, Value::None) => {
            if let Some(ax) = ax_val.int_val() {
                crate::ops::static_ndarray_ops::reduce_with_axis(b, op, val, ax)
            } else {
                let d = promote(b, val);
                crate::ops::dyn_ndarray::aggregation::dyn_aggregate(
                    b, &d, Some(ax_val), agg_kind(op),
                )
            }
        }
        _ => crate::helpers::ndarray::builtin_reduce(b, op, val),
    }
}

/// Unified argmax/argmin.
pub fn argmax_argmin(
    b: &mut IRBuilder,
    val: &Value,
    axis_arg: Option<&Value>,
    is_max: bool,
) -> Value {
    let op = if is_max { "argmax" } else { "argmin" };

    if let Value::DynamicNDArray(d) = val {
        return crate::ops::dyn_ndarray::aggregation::dyn_aggregate(
            b, d, axis_arg, agg_kind(op),
        );
    }

    match axis_arg {
        Some(ax_val) if !matches!(ax_val, Value::None) => {
            if let Some(ax) = ax_val.int_val() {
                crate::ops::static_ndarray_ops::ndarray_argmax_argmin_with_axis(b, val, ax, is_max)
            } else {
                let d = promote(b, val);
                crate::ops::dyn_ndarray::aggregation::dyn_aggregate(
                    b, &d, Some(ax_val), agg_kind(op),
                )
            }
        }
        _ => crate::helpers::ndarray::ndarray_argmax_argmin(b, val, &[], is_max),
    }
}
