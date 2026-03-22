//! Operation definitions — ports Python `zinnia/op_def/` (223 files).
//!
//! Each Python `AbstractOp` subclass maps to a variant in the `Op` enum.
//! The `Op::build()` method dispatches by value type (scalar, ndarray, list, etc.)
//! calling the appropriate `IRBuilder` methods.

pub mod arithmetic;
pub mod cast;
pub mod comparison;
pub mod dyn_ndarray;
pub mod internal;
pub mod logical;
pub mod math_ops;
pub mod static_ndarray_ops;
pub mod nocls;
pub mod np_like;
pub mod registry;

use std::collections::HashMap;

use crate::builder::IRBuilder;
use crate::types::Value;

// ---------------------------------------------------------------------------
// OpArgsContainer — mirrors Python `OpArgsContainer`
// ---------------------------------------------------------------------------

/// Parsed operator arguments with an optional path condition.
#[derive(Debug, Clone)]
pub struct OpArgs {
    pub kwargs: HashMap<String, Value>,
    pub condition: Option<Value>,
}

impl OpArgs {
    pub fn new(kwargs: HashMap<String, Value>) -> Self {
        Self {
            kwargs,
            condition: None,
        }
    }

    pub fn with_condition(kwargs: HashMap<String, Value>, condition: Value) -> Self {
        Self {
            kwargs,
            condition: Some(condition),
        }
    }

    pub fn get(&self, key: &str) -> Option<&Value> {
        self.kwargs.get(key)
    }

    pub fn require(&self, key: &str) -> &Value {
        self.kwargs
            .get(key)
            .unwrap_or_else(|| panic!("Missing required argument: {}", key))
    }
}

// ---------------------------------------------------------------------------
// ParamEntry — mirrors Python `AbstractOp._ParamEntry`
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct ParamEntry {
    pub name: &'static str,
    pub optional: bool,
}

impl ParamEntry {
    pub const fn required(name: &'static str) -> Self {
        Self {
            name,
            optional: false,
        }
    }

    pub const fn optional(name: &'static str) -> Self {
        Self {
            name,
            optional: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Op trait — mirrors Python `AbstractOp`
// ---------------------------------------------------------------------------

/// Trait for all operation definitions.
pub trait Op {
    /// Operator name (e.g., "add", "int", "select").
    fn name(&self) -> &'static str;

    /// Operator signature (e.g., "add", "math.exp").
    fn signature(&self) -> &'static str { self.name() }

    /// Whether this op modifies values in place.
    fn is_inplace(&self) -> bool { false }

    /// Whether this op requires a path condition.
    fn requires_condition(&self) -> bool { false }

    /// Parameter definitions for argparse.
    fn params(&self) -> &[ParamEntry];

    /// Build the IR for this operation.
    fn build(&self, builder: &mut IRBuilder, args: &OpArgs) -> Value;
}

/// Dispatch a binary operation over two numeric values.
/// Handles Int+Int, Float+Float, and mixed Int+Float (promotes to Float).
/// Used by both arithmetic and comparison operators.
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

/// Parse positional + keyword arguments according to param definitions.
/// Mirrors Python `AbstractOp.argparse()`.
pub fn argparse(
    params: &[ParamEntry],
    positional: &[Value],
    kwargs: &HashMap<String, Value>,
) -> Result<HashMap<String, Value>, String> {
    let mut mapping: HashMap<String, Value> = HashMap::new();
    let mut filled: Vec<&str> = Vec::new();

    // Fill positional args
    for (i, val) in positional.iter().enumerate() {
        if i >= params.len() {
            return Err("Too many positional arguments".to_string());
        }
        mapping.insert(params[i].name.to_string(), val.clone());
        filled.push(params[i].name);
    }

    // Fill keyword args
    for (key, val) in kwargs {
        if filled.contains(&key.as_str()) {
            return Err(format!("Duplicate argument: {}", key));
        }
        let valid = params.iter().any(|p| p.name == key.as_str());
        if !valid {
            return Err(format!("Unexpected keyword argument: {}", key));
        }
        mapping.insert(key.clone(), val.clone());
        filled.push(params.iter().find(|p| p.name == key.as_str()).unwrap().name);
    }

    // Check required args
    for param in params {
        if !param.optional && !filled.contains(&param.name) {
            return Err(format!("Missing required argument: {}", param.name));
        }
    }

    Ok(mapping)
}
