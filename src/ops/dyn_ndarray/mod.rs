//! DynamicNDArray operations.
//!
//! Free functions that operate on `IRBuilder` for all bounded-dynamic array operations.
//! These mirror the Python `zinnia/op_def/dynamic_ndarray/` operators.

use std::collections::HashMap;

use crate::ir_gen::IRGenerator;
use crate::types::{NumberType, ScalarValue, Value};

pub mod metadata;
pub mod constructors;
pub mod reshape;
pub mod aggregation;
pub mod memory_ops;

#[derive(Debug, Clone, Copy)]
pub enum DynAggKind {
    Sum,
    Prod,
    Max,
    Min,
    All,
    Any,
    Argmax,
    Argmin,
}

// ═══════════════════════════════════════════════════════════════════════════
// Utility functions (pure computation, no IR emission)
// ═══════════════════════════════════════════════════════════════════════════

/// Compute row-major strides from shape.
pub fn dyn_row_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// Decode a linear index into multi-dimensional coordinates.
pub fn dyn_decode_coords(linear: usize, shape: &[usize], strides: &[usize]) -> Vec<usize> {
    shape
        .iter()
        .zip(strides.iter())
        .map(|(&dim, &stride)| (linear / stride) % dim)
        .collect()
}

/// Encode multi-dimensional coordinates into a linear index.
pub fn dyn_encode_coords(coords: &[usize], strides: &[usize]) -> usize {
    coords.iter().zip(strides.iter()).map(|(&c, &s)| c * s).sum()
}

/// Product of shape dimensions.
pub fn dyn_num_elements(shape: &[usize]) -> usize {
    shape.iter().product()
}

// ═══════════════════════════════════════════════════════════════════════════
// Helper: convert between Value and ScalarValue<i64>
// ═══════════════════════════════════════════════════════════════════════════

/// Extract ScalarValue<i64> from a Value::Integer (or coerce Boolean).
pub fn value_to_scalar_i64(val: &Value) -> ScalarValue<i64> {
    match val {
        Value::Integer(sv) => sv.clone(),
        Value::Boolean(sv) => ScalarValue::new(
            sv.static_val.map(|b| if b { 1 } else { 0 }),
            sv.ptr,
        ),
        Value::Float(sv) => ScalarValue::new(
            sv.static_val.map(|f| f as i64),
            sv.ptr,
        ),
        _ => ScalarValue::new(None, None),
    }
}

/// Convert a stored element (ScalarValue<i64>) back to a Value.
pub fn scalar_i64_to_value(elem: &ScalarValue<i64>, dtype: NumberType) -> Value {
    match dtype {
        NumberType::Integer => Value::Integer(elem.clone()),
        NumberType::Float => {
            Value::Float(ScalarValue::new(
                elem.static_val.map(|v| v as f64),
                elem.ptr,
            ))
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Dispatch router on IRGenerator (thin wrapper that delegates to free functions)
// ═══════════════════════════════════════════════════════════════════════════

impl IRGenerator {
    /// Main dispatch for DynamicNDArray method calls.
    /// Delegates to the corresponding free functions.
    pub fn dispatch_dyn_ndarray_method(
        &mut self,
        val: Value,
        method: &str,
        args: &[Value],
        kwargs: &HashMap<String, Value>,
    ) -> Value {
        // Extract data — we need ownership for some ops
        let data = match &val {
            Value::DynamicNDArray(d) => d.clone(),
            _ => panic!("dispatch_dyn_ndarray_method called on non-DynamicNDArray"),
        };

        match method {
            // Pure metadata
            "ndim" => metadata::dyn_ndim(&mut self.builder, &data),
            "dtype" => metadata::dyn_dtype(&data),
            "shape" => metadata::dyn_shape(&mut self.builder, &data),
            "size" => metadata::dyn_size(&mut self.builder, &data),

            // Simple value ops
            "astype" => metadata::dyn_astype(&mut self.builder, &data, args),
            "flatten" => metadata::dyn_flatten_to_list(&data),
            "flat" => metadata::dyn_flat(&data),
            "tolist" => metadata::dyn_tolist(&data),
            "T" => reshape::dyn_transpose(&mut self.builder, &data, &[]),
            "transpose" => {
                let axes_args = if let Some(axes_val) = kwargs.get("axes") {
                    vec![axes_val.clone()]
                } else {
                    args.to_vec()
                };
                reshape::dyn_transpose(&mut self.builder, &data, &axes_args)
            }
            "moveaxis" => reshape::dyn_moveaxis(&mut self.builder, &data, args),

            // Aggregation ops
            "sum" => {
                let axis = kwargs.get("axis").or_else(|| args.first());
                aggregation::dyn_aggregate(&mut self.builder, &data, axis, DynAggKind::Sum)
            }
            "prod" => {
                let axis = kwargs.get("axis").or_else(|| args.first());
                aggregation::dyn_aggregate(&mut self.builder, &data, axis, DynAggKind::Prod)
            }
            "max" => {
                let axis = kwargs.get("axis").or_else(|| args.first());
                aggregation::dyn_aggregate(&mut self.builder, &data, axis, DynAggKind::Max)
            }
            "min" => {
                let axis = kwargs.get("axis").or_else(|| args.first());
                aggregation::dyn_aggregate(&mut self.builder, &data, axis, DynAggKind::Min)
            }
            "all" => {
                let axis = kwargs.get("axis").or_else(|| args.first());
                aggregation::dyn_aggregate(&mut self.builder, &data, axis, DynAggKind::All)
            }
            "any" => {
                let axis = kwargs.get("axis").or_else(|| args.first());
                aggregation::dyn_aggregate(&mut self.builder, &data, axis, DynAggKind::Any)
            }
            "argmax" => {
                let axis = kwargs.get("axis").or_else(|| args.first());
                aggregation::dyn_aggregate(&mut self.builder, &data, axis, DynAggKind::Argmax)
            }
            "argmin" => {
                let axis = kwargs.get("axis").or_else(|| args.first());
                aggregation::dyn_aggregate(&mut self.builder, &data, axis, DynAggKind::Argmin)
            }

            // Memory-heavy ops
            "filter" => memory_ops::dyn_filter(&mut self.builder, &data, args),
            "repeat" => memory_ops::dyn_repeat(&mut self.builder, &data, args, kwargs),

            _ => panic!(
                "DynamicNDArray.{} not yet implemented in Rust IR generator",
                method
            ),
        }
    }
}
