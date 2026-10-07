//! DynamicNDArray operations.
//!
//! Free functions that operate on `IRBuilder` for all bounded-dynamic array operations.
//! These mirror the Python `zinnia/op_def/dynamic_ndarray/` operators.

use std::collections::HashMap;

use crate::ir_gen::IRGenerator;
use crate::types::{NumberType, ScalarValue, Value};

pub mod aggregation;
pub mod binary;
pub mod constructors;
pub mod memory_ops;
pub mod metadata;
pub mod reshape;

/// True if `meta` describes a contiguous row-major view from offset 0.
/// Consumers that read via `read_all_typed` + `row_major_strides` need
/// this to hold; if it doesn't, call [`force_contiguous`] first.
/// Whether a contiguous runtime-shaped array keeps element
/// `(c_0, ..., c_{n-1})` at its row-major position in the buffer shape
/// `logical_shape` (a "padded" layout: the runtime extents are a prefix
/// rectangle of the buffer), rather than in a compact buffer addressed by
/// C-order strides over the runtime extents (`runtime_strides`).
///
/// Padded: the buffer shape covers every axis's maximum extent and the
/// buffer holds the whole rectangle — bounded inputs with per-axis bounds,
/// runtime-bounds slices, compactions (rank 1), merged branches. Compact:
/// constructors whose total bound is below the rectangle, and runtime
/// reshapes with a runtime inner extent (their `logical_shape` is a
/// degenerate skeleton). The test reads the shapes only; a padded array's
/// `runtime_strides` are its buffer strides as well.
pub(crate) fn has_padded_layout(d: &crate::types::DynamicNDArrayData) -> bool {
    let covers_envelope = d
        .meta
        .logical_shape
        .iter()
        .zip(&d.envelope.dims)
        .all(|(&n, dim)| n >= dim.max);
    let rectangle: usize = d.meta.logical_shape.iter().product();
    covers_envelope && d.envelope.total_bound >= rectangle
}

/// Whether a padded runtime-shaped array may have holes in its buffer: an
/// axis after the first whose runtime extent is not statically its buffer
/// width. Then the element of C-order position `k` is not at buffer slot
/// `k`, so operations that read the buffer as a flat C-order sequence
/// (reshape, a flat argmax index) must map positions through the runtime
/// extents, or refuse the array.
pub(crate) fn padded_with_holes(d: &crate::types::DynamicNDArrayData) -> bool {
    has_padded_layout(d)
        && d.meta
            .runtime_shape
            .iter()
            .zip(&d.meta.logical_shape)
            .skip(1)
            .any(|(extent, &width)| extent.static_val != Some(width as i64))
}

/// The panic (an IR-generation error naming the construct) for an operation
/// that would read a padded buffer with holes as a flat C-order sequence.
pub(crate) fn refuse_padded_with_holes(op: &str) -> ! {
    panic!(
        "{op} is not supported on an array with a runtime shape held in a padded buffer \
         (a bounded input with per-axis bounds, or a slice with runtime bounds on an inner axis): \
         its elements are not contiguous in C order"
    )
}

pub fn is_contiguous(meta: &crate::types::DynArrayMeta) -> bool {
    if meta.logical_offset != 0 {
        return false;
    }
    let expected = crate::helpers::shape_arith::row_major_strides(&meta.logical_shape);
    meta.logical_strides == expected
}

/// Materialise a possibly-non-contiguous DynamicNDArray view into a fresh
/// contiguous row-major segment. No-op (returns a clone) if already
/// contiguous. Used by ops whose downstream code assumes
/// `read_all_typed` returns cells in row-major logical order.
pub fn force_contiguous(
    b: &mut crate::builder::IRBuilder,
    data: &crate::types::DynamicNDArrayData,
) -> crate::types::DynamicNDArrayData {
    use crate::types::{DynArrayMeta, DynamicNDArrayData, ValueId};
    if is_contiguous(&data.meta) {
        return data.clone();
    }
    let cells = crate::helpers::segment::read_all_logical_typed(
        b,
        data.segment_id,
        &data.meta.logical_shape,
        &data.meta.logical_strides,
        data.meta.logical_offset,
        data.dtype,
    );
    let new_seg = crate::helpers::segment::alloc_and_write_values(b, &cells, data.dtype);
    let new_strides = crate::helpers::shape_arith::row_major_strides(&data.meta.logical_shape);
    DynamicNDArrayData {
        envelope: data.envelope.clone(),
        dtype: data.dtype,
        segment_id: new_seg,
        meta: DynArrayMeta {
            logical_shape: data.meta.logical_shape.clone(),
            logical_offset: 0,
            logical_strides: new_strides.clone(),
            runtime_length: data.meta.runtime_length.clone(),
            runtime_rank: data.meta.runtime_rank.clone(),
            runtime_shape: data.meta.runtime_shape.clone(),
            runtime_strides: new_strides
                .iter()
                .map(|&s| ScalarValue::new(Some(s as i64), None))
                .collect(),
            runtime_offset: ScalarValue::new(Some(0), None),
        },
        value_id: ValueId::next(),
    }
}

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
        Value::Boolean(sv) => {
            ScalarValue::new(sv.static_val.map(|b| if b { 1 } else { 0 }), sv.stmt_id)
        }
        // The compile-time value survives only when it is an integer: a
        // truncated one would claim a constant the statement does not hold
        // (2.5 read back as the float 2.0).
        Value::Float(sv) => ScalarValue::new(
            sv.static_val.filter(|f| f.fract() == 0.0 && f.abs() < 9.0e15).map(|f| f as i64),
            sv.stmt_id,
        ),
        _ => ScalarValue::new(None, None),
    }
}

/// `v` as a value of `dtype` without IR: the same statement under the other
/// type. A value of the same type is itself, and so is a bool read as an
/// integer (a cell of a NumPy bool array stays a bool cell,
/// `builder/bool_dtype.rs`); any other reading keeps no compile-time value
/// except zero (whose field element is the same in every encoding), so no
/// reading invents a constant the statement does not hold.
pub fn retag(v: &Value, dtype: NumberType) -> Value {
    match (dtype, v) {
        (NumberType::Integer, Value::Integer(_) | Value::Boolean(_)) | (NumberType::Float, Value::Float(_)) => v.clone(),
        (NumberType::Integer, Value::Float(s)) => {
            Value::Integer(ScalarValue::new(s.static_val.filter(|f| *f == 0.0).map(|_| 0), s.stmt_id))
        }
        (NumberType::Float, Value::Integer(s)) => Value::Float(ScalarValue::new(s.static_val.filter(|i| *i == 0).map(|_| 0.0), s.stmt_id)),
        (NumberType::Float, Value::Boolean(s)) => Value::Float(ScalarValue::new(s.static_val.filter(|b| !*b).map(|_| 0.0), s.stmt_id)),
        _ => scalar_i64_to_value(&value_to_scalar_i64(v), dtype),
    }
}

/// Convert a stored element (ScalarValue<i64>) back to a Value.
pub fn scalar_i64_to_value(elem: &ScalarValue<i64>, dtype: NumberType) -> Value {
    match dtype {
        NumberType::Integer => Value::Integer(elem.clone()),
        NumberType::Float => Value::Float(ScalarValue::new(
            elem.static_val.map(|v| v as f64),
            elem.stmt_id,
        )),
        NumberType::Complex => Value::Complex {
            real: ScalarValue::new(elem.static_val.map(|v| v as f64), elem.stmt_id),
            imag: ScalarValue::new(Some(0.0), None),
        },
    }
}

pub fn value_to_complex_parts(val: &Value) -> (ScalarValue<i64>, ScalarValue<i64>) {
    // A float part keeps its compile-time value only when it is an integer
    // (as `value_to_scalar_i64`); the imaginary part of a real number is the
    // constant 0 (no statement: `ensure_ptr` makes the constant).
    let part = |s: &ScalarValue<f64>| value_to_scalar_i64(&Value::Float(s.clone()));
    let zero = || ScalarValue::new(Some(0), None);
    match val {
        Value::Complex { real, imag } => (part(real), part(imag)),
        Value::Float(sv) => (part(sv), zero()),
        Value::Integer(sv) => (sv.clone(), zero()),
        Value::Boolean(sv) => (ScalarValue::new(sv.static_val.map(i64::from), sv.stmt_id), zero()),
        _ => (ScalarValue::new(None, None), ScalarValue::new(None, None)),
    }
}

pub fn complex_parts_to_value(real: &ScalarValue<i64>, imag: &ScalarValue<i64>) -> Value {
    Value::Complex {
        real: ScalarValue::new(real.static_val.map(|v| v as f64), real.stmt_id),
        imag: ScalarValue::new(imag.static_val.map(|v| v as f64), imag.stmt_id),
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

        // `keepdims=True` is not threaded through the bounded-dynamic
        // reduction lowering (the axis is unconditionally removed from the
        // output meta). Silently ignoring it would mis-shape the result
        // (numpy keeps the reduced axis as extent 1). Reject loudly rather
        // than return the wrong rank. Static arrays DO honour keepdims via
        // `static_array::reductions`.
        let keepdims = kwargs
            .get("keepdims")
            .and_then(|v| v.bool_val().or_else(|| v.int_val().map(|i| i != 0)))
            .unwrap_or(false);
        if keepdims
            && matches!(
                method,
                "sum" | "prod" | "max" | "min" | "all" | "any" | "argmax" | "argmin" | "mean"
            )
        {
            panic!(
                "DynamicNDArray.{}(keepdims=True) is not supported for \
                 bounded-dynamic arrays; the reduced axis is dropped (keepdims \
                 is not honoured). Use keepdims=False, or convert to a static \
                 array before this reduction.",
                method
            );
        }

        match method {
            // Pure metadata
            "ndim" => metadata::dyn_ndim(&mut self.builder, &data),
            "dtype" => metadata::dyn_dtype(&mut self.builder, &data),
            "shape" => metadata::dyn_shape(&mut self.builder, &data),
            "size" => metadata::dyn_size(&mut self.builder, &data),

            // Simple value ops
            "astype" => metadata::dyn_astype(&mut self.builder, &data, args),
            "flatten" => metadata::dyn_flatten_to_list(&mut self.builder, &data),
            "flat" => metadata::dyn_flat(&mut self.builder, &data),
            "tolist" => metadata::dyn_tolist(&mut self.builder, &data),
            "real" => metadata::dyn_complex_real(&mut self.builder, &data),
            "imag" => metadata::dyn_complex_imag(&mut self.builder, &data),
            "conjugate" | "conj" => metadata::dyn_complex_conjugate(&mut self.builder, &data),
            // NumPy views; a layout Zinnia cannot give as one (a runtime-
            // strided buffer) is a copy, linked to the array so that writes
            // through either are tracked (`ir_gen::detached_copies`).
            "T" | "transpose" | "moveaxis" | "swapaxes" => {
                let result = match method {
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
                    _ => reshape::dyn_swapaxes(&mut self.builder, &data, args),
                };
                self.record_view_result(&val, &result);
                result
            }
            "squeeze" => {
                let squeeze_args = if let Some(axis_val) = kwargs.get("axis") {
                    vec![axis_val.clone()]
                } else {
                    args.to_vec()
                };
                reshape::dyn_squeeze(&mut self.builder, &data, &squeeze_args)
            }
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
            "mean" => {
                let axis = kwargs.get("axis").or_else(|| args.first());
                aggregation::dyn_mean(&mut self.builder, &data, axis)
            }
            "var" | "std" | "cumsum" | "cumprod" => panic!(
                "DynamicNDArray.{} is not yet supported for bounded-dynamic arrays; use a static array or keep this operation outside the bounded-dynamic surface",
                method
            ),

            // Memory-heavy ops
            "filter" => memory_ops::dyn_filter(&mut self.builder, &data, args),
            "repeat" => memory_ops::dyn_repeat(&mut self.builder, &data, args, kwargs),

            _ => panic!(
                "DynamicNDArray.{} is not yet supported for bounded-dynamic arrays; use a supported dynamic operator or convert to a static array before this operation",
                method
            ),
        }
    }
}
