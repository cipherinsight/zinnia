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
pub mod binary;

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
// Utility functions (pure shape computation)
// ═══════════════════════════════════════════════════════════════════════════
//
// The bodies of these utilities have moved to `helpers::shape_arith` so the
// static and dynamic ndarray surfaces can share them. The `dyn_*` names are
// kept as aliases here so existing call sites don't change. Phase 1
// (envelope migration) will retire the aliases in favour of direct uses of
// `shape_arith::*`.

pub use crate::helpers::shape_arith::{
    decode_coords as dyn_decode_coords, encode_coords as dyn_encode_coords,
    num_elements as dyn_num_elements, row_major_strides as dyn_row_major_strides,
};

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
        // Extract data
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
            "flatten" => metadata::dyn_flatten_to_list(&mut self.builder, &data),
            "flat" => metadata::dyn_flat(&mut self.builder, &data),
            "tolist" => metadata::dyn_tolist(&mut self.builder, &data),
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
            "reshape" => reshape::dyn_reshape(&mut self.builder, &data, args),

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
