//! Operator implementations for builtins, `math`, NumPy functions and array
//! methods.
//!
//! `registry::build_op` maps a call (name and namespace) to an `Op`, whose
//! `build` dispatches on the argument values (scalar, array, list, ...) and
//! emits IR through the `IRBuilder`.

pub mod dyn_ndarray;
pub mod math_ops;
pub mod np_like;
pub mod registry;
pub mod static_ndarray_ops;

use std::collections::HashMap;

use crate::builder::IRBuilder;
use crate::types::Value;

// ---------------------------------------------------------------------------
// OpArgs
// ---------------------------------------------------------------------------

/// Operator arguments by parameter name.
#[derive(Debug, Clone)]
pub struct OpArgs {
    pub kwargs: HashMap<String, Value>,
}

impl OpArgs {
    pub fn new(kwargs: HashMap<String, Value>) -> Self {
        Self { kwargs }
    }

    pub fn require(&self, key: &str) -> &Value {
        self.kwargs
            .get(key)
            .unwrap_or_else(|| panic!("Missing required argument: {}", key))
    }
}

// ---------------------------------------------------------------------------
// Op trait
// ---------------------------------------------------------------------------

/// Trait for all operation definitions.
pub trait Op {
    /// Build the IR for this operation.
    fn build(&self, builder: &mut IRBuilder, args: &OpArgs) -> Value;
}

/// Dispatch a binary operation over two numeric values.
/// Handles Int+Int, Float+Float, and mixed Int+Float (promotes to Float).
/// Used by the NumPy-like arithmetic and comparison operators.
pub fn dispatch_binary_numeric(
    builder: &mut IRBuilder,
    lhs: &Value,
    rhs: &Value,
    int_op: fn(&mut IRBuilder, &Value, &Value) -> Value,
    float_op: fn(&mut IRBuilder, &Value, &Value) -> Value,
) -> Value {
    match (lhs, rhs) {
        (Value::Integer(_) | Value::Boolean(_), Value::Integer(_) | Value::Boolean(_)) => {
            int_op(builder, lhs, rhs)
        }
        (Value::Float(_), Value::Float(_)) => float_op(builder, lhs, rhs),
        (Value::Integer(_) | Value::Boolean(_), Value::Float(_)) => {
            let lf = builder.ir_float_cast(lhs);
            float_op(builder, &lf, rhs)
        }
        (Value::Float(_), Value::Integer(_) | Value::Boolean(_)) => {
            let rf = builder.ir_float_cast(rhs);
            float_op(builder, lhs, &rf)
        }
        _ => panic!(
            "dispatch_binary_numeric: unsupported types {:?} and {:?}",
            lhs.zinnia_type(),
            rhs.zinnia_type()
        ),
    }
}
