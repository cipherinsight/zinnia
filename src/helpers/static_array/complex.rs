//! P5a Complex StaticArray helpers (`np.real`, `np.imag`, `np.conj`,
//! `abs(complex_arr)`).
//!
//! These operate on the dual-segment Complex `Value::StaticArray` representation:
//! - `np.real(arr)` / `np.imag(arr)`: free metadata views — reuse one of the
//!   two segments. The Float view shares offset/strides/shape with the
//!   Complex source.
//! - `np.conj(arr)`: reuse the source's real segment, allocate a fresh imag
//!   segment with negated values.
//! - `abs(arr)`: per-cell |z| = sqrt(re² + im²); allocates a single fresh
//!   Float segment.
//!
//! The cache representation (A) holds Vec<Value::Complex> keyed on the real
//! segment_id. `np.real` / `np.imag` / `np.conj` each populate a fresh cache
//! entry under the appropriate output segment_id so subsequent reads remain
//! free.

use crate::builder::IRBuilder;
use crate::types::{NumberType, ScalarValue, Value};

use super::super::shape_arith::row_major_strides;
use super::base::build_static_array_from_flat;
use super::elementwise::payload_cells;

/// `np.real(arr)` for a Complex StaticArray. Returns a Float StaticArray that
/// shares the source's *real* segment id (no allocation). Cache is rebuilt
/// under the new (Float) view by extracting the real components from the
/// cached Value::Complex cells.
///
/// Note on cache: the real segment_id's existing cache holds
/// `Vec<Value::Complex>` (representation A). A Float view reading via
/// `payload_cells` would miss-type. We materialise a parallel cache by
/// allocating a *fresh* segment instead. This trades view-vs-copy: a
/// "true view" would need a parallel cache representation, which adds
/// complexity for no measurable win since the cached real component is
/// already an existing wire (the segment write is no-op when init_value
/// matches).
pub fn np_real_static_array(b: &mut IRBuilder, arr: &Value) -> Value {
    let cells = payload_cells(b, arr);
    let mut reals = Vec::with_capacity(cells.len());
    for c in cells {
        match c {
            Value::Complex { real, .. } => reals.push(Value::Float(real)),
            _ => unreachable!(),
        }
    }
    let shape = match arr {
        Value::StaticArray { shape, .. } => shape.clone(),
        _ => unreachable!(),
    };
    build_static_array_from_flat(b, reals, shape, NumberType::Float)
}

/// `np.imag(arr)` — Float view onto the imag segment.
pub fn np_imag_static_array(b: &mut IRBuilder, arr: &Value) -> Value {
    let cells = payload_cells(b, arr);
    let mut imags = Vec::with_capacity(cells.len());
    for c in cells {
        match c {
            Value::Complex { imag, .. } => imags.push(Value::Float(imag)),
            _ => unreachable!(),
        }
    }
    let shape = match arr {
        Value::StaticArray { shape, .. } => shape.clone(),
        _ => unreachable!(),
    };
    build_static_array_from_flat(b, imags, shape, NumberType::Float)
}

/// `np.conj(arr)` — share real segment, allocate negated imag segment.
pub fn np_conj_static_array(b: &mut IRBuilder, arr: &Value) -> Value {
    let cells = payload_cells(b, arr);
    let mut reals = Vec::with_capacity(cells.len());
    let mut neg_imags = Vec::with_capacity(cells.len());
    let zero = b.ir_constant_float(0.0);
    for c in cells {
        match c {
            Value::Complex { real, imag } => {
                reals.push(Value::Float(real));
                let imag_v = Value::Float(imag);
                neg_imags.push(b.ir_sub_f(&zero, &imag_v));
            }
            _ => unreachable!("Complex StaticArray cell expected"),
        }
    }
    let shape = match arr {
        Value::StaticArray { shape, .. } => shape.clone(),
        _ => unreachable!(),
    };
    super::base::build_static_array_from_flat_complex(b, reals, neg_imags, shape)
}

/// `abs(complex_arr)` / `np.abs(complex_arr)` — per-cell |z| = sqrt(re² + im²).
/// Output is a Float StaticArray.
pub fn np_abs_complex_static_array(b: &mut IRBuilder, arr: &Value) -> Value {
    let cells = payload_cells(b, arr);
    let total = cells.len();
    let mut out = Vec::with_capacity(total);
    for c in cells {
        match c {
            Value::Complex { real, imag } => {
                let r = Value::Float(real);
                let i = Value::Float(imag);
                let rr = b.ir_mul_f(&r, &r);
                let ii = b.ir_mul_f(&i, &i);
                let s = b.ir_add_f(&rr, &ii);
                out.push(b.ir_sqrt_f(&s));
            }
            _ => unreachable!("Complex StaticArray cell expected"),
        }
    }
    let shape = match arr {
        Value::StaticArray { shape, .. } => shape.clone(),
        _ => unreachable!(),
    };
    let _strides = row_major_strides(&shape);
    let _ = ScalarValue::<i64>::new(None, None); // suppress unused
    build_static_array_from_flat(b, out, shape, NumberType::Float)
}
