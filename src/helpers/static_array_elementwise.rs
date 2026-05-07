//! Native elementwise binary / unary / comparison ops for `Value::StaticArray`.
//!
//! P4a of `compiler.epic-segment-native-static-arrays`: this module replaces
//! the boundary shim that used to convert a `StaticArray` into a nested
//! `Value::List` before dispatching to the legacy elementwise machinery in
//! `helpers::value_ops` and `ops::static_ndarray_ops::elementwise_binary`.
//!
//! Behaviour:
//! - Two `Value::StaticArray` operands of equal shape: iterate the flat
//!   payloads, emit one per-cell op into a fresh segment, return a new
//!   `Value::StaticArray` of the broadcast (here equal) shape.
//! - One `StaticArray` and one numeric scalar: iterate the array's payload,
//!   apply the scalar op per cell, return a new `Value::StaticArray`.
//! - Two `StaticArray`s with mismatched but broadcast-compatible shapes:
//!   compute `broadcast_shapes(lhs, rhs)`, iterate the output, and derive
//!   each input's source flat index from the output coordinates using
//!   stride-0 axes for broadcast dimensions. No `materialize_to_shape` is
//!   needed — broadcasting happens via the per-cell address arithmetic.
//!
//! Output caching: every fresh output is built via
//! [`build_static_array_from_flat`], which writes the flat payload into a new
//! segment AND populates `IRBuilder::static_array_payload` so static-index
//! reads on the result remain free.
//!
//! Constant folding: when each input cell carries a compile-time
//! `static_val`, the per-cell `ir_*` op preserves it (this is true for the
//! existing scalar dispatch in `apply_scalar_binary_op`). The cached payload
//! therefore holds the folded values, which the boundary shim returns to
//! legacy callers when they materialise via `to_value_list`.

use crate::builder::IRBuilder;
use crate::ops::dyn_ndarray::value_to_scalar_i64;
use crate::types::{NumberType, Value};

use super::broadcast::broadcast_shapes;
use super::shape_arith::{decode_coords, row_major_strides};
use super::static_array::{build_static_array_from_flat, to_value_list};
use super::value_ops::apply_scalar_binary_op;

// ────────────────────────────────────────────────────────────────────────
// Cell helpers
// ────────────────────────────────────────────────────────────────────────

/// Pull the flat payload of a `Value::StaticArray` as a `Vec<Value>`. Reuses
/// the cached wires when present; otherwise emits one `ir_read_memory` per
/// cell. Caller passes the visible shape (so we know how many cells the
/// view sees).
pub fn payload_cells(b: &mut IRBuilder, arr: &Value) -> Vec<Value> {
    let (dtype, shape, segment_id, _strides, offset) = match arr {
        Value::StaticArray {
            dtype,
            shape,
            segment_id,
            strides,
            offset,
        } => (*dtype, shape.clone(), *segment_id, strides.clone(), *offset),
        _ => panic!("payload_cells: expected Value::StaticArray"),
    };
    let total: usize = shape.iter().product();
    if let Some(cached) = b.static_array_payload.get(&segment_id) {
        return cached
            .iter()
            .skip(offset)
            .take(total)
            .cloned()
            .collect();
    }
    // Fallback: read every cell from the segment.
    let mut tmp = Vec::with_capacity(total);
    for i in 0..total {
        let addr = b.ir_constant_int((offset + i) as i64);
        let raw = b.ir_read_memory(segment_id, &addr);
        let sv = value_to_scalar_i64(&raw);
        tmp.push(crate::ops::dyn_ndarray::scalar_i64_to_value(&sv, dtype));
    }
    tmp
}

/// Determine the result dtype for a binary op given two operand dtypes /
/// scalar leaves. Mirrors the legacy promotion: any `Float` input produces
/// `Float`; otherwise `Integer`.
fn promote_dtype(lhs: NumberType, rhs: NumberType) -> NumberType {
    if matches!(lhs, NumberType::Float) || matches!(rhs, NumberType::Float) {
        NumberType::Float
    } else {
        NumberType::Integer
    }
}

fn dtype_of_static_array(val: &Value) -> NumberType {
    match val {
        Value::StaticArray { dtype, .. } => *dtype,
        _ => panic!("dtype_of_static_array: expected StaticArray"),
    }
}

fn shape_of_static_array(val: &Value) -> Vec<usize> {
    match val {
        Value::StaticArray { shape, .. } => shape.clone(),
        _ => panic!("shape_of_static_array: expected StaticArray"),
    }
}

/// Numeric-scalar dtype tag for promotion. Booleans count as Integer.
fn dtype_of_scalar(val: &Value) -> Option<NumberType> {
    match val {
        Value::Integer(_) | Value::Boolean(_) => Some(NumberType::Integer),
        Value::Float(_) => Some(NumberType::Float),
        _ => None,
    }
}

// ────────────────────────────────────────────────────────────────────────
// Output dtype rules for a given op
// ────────────────────────────────────────────────────────────────────────

fn is_comparison_op(op: &str) -> bool {
    matches!(op, "eq" | "ne" | "lt" | "lte" | "gt" | "gte")
}

/// `true` for ops that return a Boolean array regardless of operand dtype.
/// Comparisons fall in this bucket.
fn op_yields_boolean(op: &str) -> bool {
    is_comparison_op(op)
}

/// `true` for ops we route natively here (binary arithmetic + bitwise +
/// comparison). `mat_mul` is excluded — it routes through the legacy path
/// after a boundary conversion.
fn is_native_binary_op(op: &str) -> bool {
    matches!(
        op,
        "add" | "sub" | "mul" | "div" | "mod" | "pow" | "floor_div"
        | "bit_and" | "bit_or" | "bit_xor" | "shl" | "shr"
        | "eq" | "ne" | "lt" | "lte" | "gt" | "gte"
    )
}

// ────────────────────────────────────────────────────────────────────────
// Public entry points
// ────────────────────────────────────────────────────────────────────────

/// Try to apply a binary op natively to two `Value::StaticArray`s, or one
/// `StaticArray` and one numeric scalar. Returns `Some(result)` when this
/// path handles the case; `None` otherwise (caller should fall back to the
/// legacy materialised-list path).
///
/// The result is always a fresh `Value::StaticArray` with the broadcast
/// output shape (or the scalar-broadcast shape).
pub fn try_apply_binary_op(
    b: &mut IRBuilder,
    op: &str,
    lhs: &Value,
    rhs: &Value,
) -> Option<Value> {
    if !is_native_binary_op(op) {
        return None;
    }

    // Refuse Complex on either side (StaticArray of Complex isn't a thing
    // today; complex scalar + StaticArray defers to legacy until P5a).
    let lhs_dtype = match lhs {
        Value::StaticArray { dtype, .. } => Some(*dtype),
        _ => dtype_of_scalar(lhs),
    };
    let rhs_dtype = match rhs {
        Value::StaticArray { dtype, .. } => Some(*dtype),
        _ => dtype_of_scalar(rhs),
    };
    if matches!(lhs_dtype, Some(NumberType::Complex))
        || matches!(rhs_dtype, Some(NumberType::Complex))
    {
        return None;
    }

    match (lhs, rhs) {
        // Both StaticArray.
        (Value::StaticArray { .. }, Value::StaticArray { .. }) => {
            let lhs_shape = shape_of_static_array(lhs);
            let rhs_shape = shape_of_static_array(rhs);
            let lhs_dtype = dtype_of_static_array(lhs);
            let rhs_dtype = dtype_of_static_array(rhs);

            let out_shape = match broadcast_shapes(&lhs_shape, &rhs_shape) {
                Some(s) => s,
                None => panic!(
                    "operands could not be broadcast together with shapes {:?} {:?}",
                    lhs_shape, rhs_shape
                ),
            };
            let out_dtype = if op_yields_boolean(op) {
                NumberType::Integer
            } else {
                promote_dtype(lhs_dtype, rhs_dtype)
            };
            Some(elementwise_two_arrays(
                b, op, lhs, rhs, &lhs_shape, &rhs_shape, &out_shape, out_dtype,
            ))
        }
        // StaticArray + scalar (numeric).
        (Value::StaticArray { .. }, _) if rhs.is_number() => {
            let arr_shape = shape_of_static_array(lhs);
            let arr_dtype = dtype_of_static_array(lhs);
            let scalar_dtype = dtype_of_scalar(rhs).expect("rhs is_number guarantees scalar dtype");
            let out_dtype = if op_yields_boolean(op) {
                NumberType::Integer
            } else {
                promote_dtype(arr_dtype, scalar_dtype)
            };
            Some(elementwise_array_scalar(b, op, lhs, rhs, true, &arr_shape, out_dtype))
        }
        // scalar (numeric) + StaticArray.
        (_, Value::StaticArray { .. }) if lhs.is_number() => {
            let arr_shape = shape_of_static_array(rhs);
            let arr_dtype = dtype_of_static_array(rhs);
            let scalar_dtype = dtype_of_scalar(lhs).expect("lhs is_number guarantees scalar dtype");
            let out_dtype = if op_yields_boolean(op) {
                NumberType::Integer
            } else {
                promote_dtype(arr_dtype, scalar_dtype)
            };
            Some(elementwise_array_scalar(b, op, rhs, lhs, false, &arr_shape, out_dtype))
        }
        _ => None,
    }
}

/// Try to apply a unary op natively on a `Value::StaticArray`. Returns
/// `Some(result)` for the migrated unary set (`usub`, `uadd`, `not`,
/// `invert`); `None` otherwise. The caller falls back to the legacy path
/// if `None` is returned.
pub fn try_apply_unary_op(
    b: &mut IRBuilder,
    op: &str,
    operand: &Value,
) -> Option<Value> {
    let arr_dtype = match operand {
        Value::StaticArray { dtype, .. } => *dtype,
        _ => return None,
    };
    if matches!(arr_dtype, NumberType::Complex) {
        // Complex StaticArray isn't constructed today; defer to legacy.
        return None;
    }

    match op {
        "uadd" => Some(operand.clone()),
        "usub" | "not" | "invert" => {
            let arr_shape = shape_of_static_array(operand);
            let out_dtype = match op {
                "not" | "invert" => NumberType::Integer,
                _ => arr_dtype,
            };
            Some(unary_array(b, op, operand, &arr_shape, out_dtype))
        }
        _ => None,
    }
}

// ────────────────────────────────────────────────────────────────────────
// Implementation details
// ────────────────────────────────────────────────────────────────────────

/// Element-wise op on two `StaticArray`s with potentially different shapes
/// (broadcasted). Always produces a fresh segment-backed result.
fn elementwise_two_arrays(
    b: &mut IRBuilder,
    op: &str,
    lhs: &Value,
    rhs: &Value,
    lhs_shape: &[usize],
    rhs_shape: &[usize],
    out_shape: &[usize],
    out_dtype: NumberType,
) -> Value {
    let lhs_cells = payload_cells(b, lhs);
    let rhs_cells = payload_cells(b, rhs);

    let total: usize = out_shape.iter().product();
    let out_strides = row_major_strides(out_shape);
    let lhs_idx = make_broadcast_indexer(lhs_shape, out_shape);
    let rhs_idx = make_broadcast_indexer(rhs_shape, out_shape);

    let mut out: Vec<Value> = Vec::with_capacity(total);
    for flat in 0..total {
        let coords = decode_coords(flat, out_shape, &out_strides);
        let li = lhs_idx.flat_index(&coords);
        let ri = rhs_idx.flat_index(&coords);
        let l = &lhs_cells[li];
        let r = &rhs_cells[ri];
        out.push(apply_scalar_binary_op(b, op, l, r));
    }

    build_static_array_from_flat(b, out, out_shape.to_vec(), out_dtype)
}

/// Element-wise op on a `StaticArray` and a numeric scalar. `arr_is_lhs`
/// matches the numpy operand order (so `op(arr, scalar)` and
/// `op(scalar, arr)` differ for non-commutative ops like `sub`, `div`).
fn elementwise_array_scalar(
    b: &mut IRBuilder,
    op: &str,
    arr: &Value,
    scalar: &Value,
    arr_is_lhs: bool,
    arr_shape: &[usize],
    out_dtype: NumberType,
) -> Value {
    let cells = payload_cells(b, arr);
    let total: usize = arr_shape.iter().product();
    let mut out: Vec<Value> = Vec::with_capacity(total);
    for i in 0..total {
        let cell = &cells[i];
        let res = if arr_is_lhs {
            apply_scalar_binary_op(b, op, cell, scalar)
        } else {
            apply_scalar_binary_op(b, op, scalar, cell)
        };
        out.push(res);
    }
    build_static_array_from_flat(b, out, arr_shape.to_vec(), out_dtype)
}

/// Apply a unary op cell-by-cell.
fn unary_array(
    b: &mut IRBuilder,
    op: &str,
    arr: &Value,
    arr_shape: &[usize],
    out_dtype: NumberType,
) -> Value {
    let cells = payload_cells(b, arr);
    let total: usize = arr_shape.iter().product();
    let mut out: Vec<Value> = Vec::with_capacity(total);
    for i in 0..total {
        let v = &cells[i];
        let r = match op {
            "usub" => {
                if matches!(v, Value::Float(_)) {
                    let zero = b.ir_constant_float(0.0);
                    b.ir_sub_f(&zero, v)
                } else {
                    let zero = b.ir_constant_int(0);
                    b.ir_sub_i(&zero, v)
                }
            }
            "not" => b.ir_logical_not(v),
            "invert" => {
                if matches!(v, Value::Float(_)) {
                    panic!("Bitwise NOT (~) requires an integer/boolean operand, got float");
                }
                b.ir_bit_not_i(v)
            }
            _ => unreachable!("unary_array called with unsupported op {}", op),
        };
        out.push(r);
    }
    build_static_array_from_flat(b, out, arr_shape.to_vec(), out_dtype)
}

// ────────────────────────────────────────────────────────────────────────
// Broadcast indexer
// ────────────────────────────────────────────────────────────────────────

/// Maps an output coordinate (in the broadcast shape) to a flat source index
/// into the source array's payload. Uses stride-0 for broadcast (size-1) and
/// padded leading axes — no source materialisation needed.
struct BroadcastIndexer {
    /// Stride into the source's flat payload, indexed by *output* axis. A
    /// stride of 0 means the axis is broadcasted.
    out_axis_strides: Vec<usize>,
    /// True only for the "all out_axis_strides are zero AND source has 1
    /// element" case — used to skip the per-coord computation.
    is_scalar_source: bool,
}

impl BroadcastIndexer {
    fn flat_index(&self, out_coords: &[usize]) -> usize {
        if self.is_scalar_source {
            return 0;
        }
        let mut acc: usize = 0;
        for (ax, c) in out_coords.iter().enumerate() {
            acc += c * self.out_axis_strides[ax];
        }
        acc
    }
}

/// Build a `BroadcastIndexer` for a source shape against the target output
/// shape. Source shape is right-aligned with the output (left-padded with 1s
/// for missing leading axes).
fn make_broadcast_indexer(src_shape: &[usize], out_shape: &[usize]) -> BroadcastIndexer {
    let rank = out_shape.len();
    let pad = rank - src_shape.len();
    // Natural row-major strides over the *original* source. Used only for
    // unpadded, non-broadcast axes.
    let src_strides_nat = row_major_strides(src_shape);

    let mut out_axis_strides = vec![0usize; rank];
    for d in 0..rank {
        if d < pad {
            // Synthesised leading axis on the source — broadcast.
            out_axis_strides[d] = 0;
        } else {
            let sd = d - pad;
            out_axis_strides[d] = if src_shape[sd] == 1 { 0 } else { src_strides_nat[sd] };
        }
    }
    let src_total: usize = src_shape.iter().product();
    BroadcastIndexer {
        out_axis_strides,
        is_scalar_source: src_total == 1,
    }
}

// ────────────────────────────────────────────────────────────────────────
// Boundary helper
// ────────────────────────────────────────────────────────────────────────

/// Catch-all for ops where we can't (or shouldn't) handle natively but at
/// least one operand is a `Value::StaticArray`. Materialises both sides via
/// `to_value_list` so the legacy dispatcher works. Used by the boundary
/// shim in `apply_binary_op` after `try_apply_binary_op` returns `None`.
pub fn materialise_pair(b: &mut IRBuilder, lhs: &Value, rhs: &Value) -> (Value, Value) {
    let l = to_value_list(b, lhs);
    let r = to_value_list(b, rhs);
    (l, r)
}

// ────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::CompositeData;

    fn list_of(values: Vec<Value>) -> Value {
        let types = values.iter().map(|v| v.zinnia_type()).collect();
        Value::List(CompositeData {
            elements_type: types,
            values,
        })
    }

    fn make_1d_int(b: &mut IRBuilder, vals: &[i64]) -> Value {
        let leaves: Vec<Value> = vals.iter().map(|n| b.ir_constant_int(*n)).collect();
        let lst = list_of(leaves);
        super::super::static_array::to_static_array(b, &lst).expect("StaticArray")
    }

    #[test]
    fn binary_two_arrays_equal_shape_constant_folds() {
        let mut b = IRBuilder::new();
        let a = make_1d_int(&mut b, &[1, 2, 3]);
        let c = make_1d_int(&mut b, &[10, 20, 30]);
        let out = try_apply_binary_op(&mut b, "add", &a, &c).expect("native path");
        match &out {
            Value::StaticArray { dtype, shape, .. } => {
                assert_eq!(*dtype, NumberType::Integer);
                assert_eq!(*shape, vec![3]);
            }
            _ => panic!("expected StaticArray"),
        }
        // Re-read via the cache and check the folded values.
        let cells = payload_cells(&mut b, &out);
        let folded: Vec<Option<i64>> = cells.iter().map(|v| v.int_val()).collect();
        assert_eq!(folded, vec![Some(11), Some(22), Some(33)]);
    }

    #[test]
    fn binary_array_scalar_constant_folds() {
        let mut b = IRBuilder::new();
        let a = make_1d_int(&mut b, &[1, 2, 3]);
        let scalar = b.ir_constant_int(10);
        let out = try_apply_binary_op(&mut b, "mul", &a, &scalar).expect("native path");
        let cells = payload_cells(&mut b, &out);
        let folded: Vec<Option<i64>> = cells.iter().map(|v| v.int_val()).collect();
        assert_eq!(folded, vec![Some(10), Some(20), Some(30)]);
    }

    #[test]
    fn binary_broadcast_3x1_plus_1x4_constant_folds() {
        let mut b = IRBuilder::new();
        // (3, 1)
        let row0 = list_of(vec![b.ir_constant_int(1)]);
        let row1 = list_of(vec![b.ir_constant_int(2)]);
        let row2 = list_of(vec![b.ir_constant_int(3)]);
        let lhs = super::super::static_array::to_static_array(
            &mut b, &list_of(vec![row0, row1, row2]),
        ).unwrap();
        // (1, 4)
        let r0 = list_of((0..4).map(|i| b.ir_constant_int(10 + i * 10)).collect());
        let rhs = super::super::static_array::to_static_array(
            &mut b, &list_of(vec![r0]),
        ).unwrap();
        let out = try_apply_binary_op(&mut b, "add", &lhs, &rhs).expect("native path");
        match &out {
            Value::StaticArray { shape, .. } => assert_eq!(*shape, vec![3, 4]),
            _ => panic!("expected StaticArray"),
        }
        let cells = payload_cells(&mut b, &out);
        let folded: Vec<i64> = cells.iter().map(|v| v.int_val().unwrap()).collect();
        // Row 0: 1 + [10, 20, 30, 40] = [11, 21, 31, 41]
        // Row 2: 3 + [10, 20, 30, 40] = [13, 23, 33, 43]
        assert_eq!(folded[0], 11);
        assert_eq!(folded[3], 41);
        assert_eq!(folded[8], 13);
        assert_eq!(folded[11], 43);
    }

    #[test]
    fn unary_negation_constant_folds() {
        let mut b = IRBuilder::new();
        let a = make_1d_int(&mut b, &[1, -2, 3]);
        let out = try_apply_unary_op(&mut b, "usub", &a).expect("native path");
        let cells = payload_cells(&mut b, &out);
        let folded: Vec<Option<i64>> = cells.iter().map(|v| v.int_val()).collect();
        assert_eq!(folded, vec![Some(-1), Some(2), Some(-3)]);
    }

    #[test]
    fn comparison_returns_boolean_array_with_static_folds() {
        let mut b = IRBuilder::new();
        let a = make_1d_int(&mut b, &[1, 5, 3]);
        let c = make_1d_int(&mut b, &[2, 2, 2]);
        let out = try_apply_binary_op(&mut b, "gt", &a, &c).expect("native path");
        match &out {
            Value::StaticArray { dtype, shape, .. } => {
                assert_eq!(*dtype, NumberType::Integer);
                assert_eq!(*shape, vec![3]);
            }
            _ => panic!("expected StaticArray"),
        }
        let cells = payload_cells(&mut b, &out);
        let folded: Vec<Option<bool>> = cells.iter().map(|v| v.bool_val()).collect();
        assert_eq!(folded, vec![Some(false), Some(true), Some(true)]);
    }

    #[test]
    fn unary_pass_through_uadd() {
        let mut b = IRBuilder::new();
        let a = make_1d_int(&mut b, &[1, 2, 3]);
        let out = try_apply_unary_op(&mut b, "uadd", &a).expect("native path");
        match (&a, &out) {
            (Value::StaticArray { segment_id: a_seg, .. },
             Value::StaticArray { segment_id: o_seg, .. }) => assert_eq!(a_seg, o_seg),
            _ => panic!("expected StaticArray"),
        }
    }

    #[test]
    fn returns_none_for_unmigrated_ops() {
        let mut b = IRBuilder::new();
        let a = make_1d_int(&mut b, &[1, 2, 3]);
        let c = make_1d_int(&mut b, &[4, 5, 6]);
        // mat_mul intentionally falls through to legacy.
        assert!(try_apply_binary_op(&mut b, "mat_mul", &a, &c).is_none());
    }
}
