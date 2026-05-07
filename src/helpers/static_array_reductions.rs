//! Native reduction / aggregation paths for `Value::StaticArray`.
//!
//! P4b of `compiler.epic-segment-native-static-arrays`: this module replaces
//! the boundary shim that used to convert a `StaticArray` into a nested
//! `Value::List` before dispatching to `helpers::ndarray::builtin_reduce` or
//! `ops::static_ndarray_ops::reduce_with_axis` / `ndarray_argmax_argmin`.
//!
//! Behaviour:
//! - **Whole-array reductions** (`sum`, `prod`, `min`, `max`, `mean`,
//!   `any`, `all`): walk the cached payload window starting at `view.offset`
//!   and fold per-cell using the existing IR ops. Result is a scalar value
//!   (`Value::Integer` / `Value::Float` / `Value::Boolean`).
//! - **Axis-aware reductions**: iterate over the output shape; for each
//!   output cell, walk the reduced axis with `strides[axis]` increments and
//!   fold. Output is a fresh segment-backed `Value::StaticArray` of one
//!   lower rank (or the same rank with `keepdims=True`).
//! - **Argmax / argmin** (whole-array → `Value::Integer`; axis-aware →
//!   Integer `Value::StaticArray`).
//!
//! Constant folding: per-cell ops route through the existing IR ops which
//! preserve `static_val` when both inputs carry one. So `np.sum([1, 2, 3])`
//! returns a `Value::Integer` whose `static_val == Some(6)`.
//!
//! Cache awareness: every reduction reads via `payload_cells`, which uses
//! the `IRBuilder::static_array_payload` cache when present and falls back
//! to N `ir_read_memory` ops if the cache was invalidated by a P3 dynamic
//! write.

use crate::builder::IRBuilder;
use crate::types::{NumberType, Value};

use super::shape_arith::row_major_strides;
use super::static_array::build_static_array_from_flat;
use super::static_array_elementwise::payload_cells;

// ────────────────────────────────────────────────────────────────────────
// Per-cell folding helpers
// ────────────────────────────────────────────────────────────────────────

fn dtype_of(val: &Value) -> NumberType {
    match val {
        Value::StaticArray { dtype, .. } => *dtype,
        _ => panic!("dtype_of: expected StaticArray"),
    }
}

fn shape_of(val: &Value) -> Vec<usize> {
    match val {
        Value::StaticArray { shape, .. } => shape.clone(),
        _ => panic!("shape_of: expected StaticArray"),
    }
}

/// Fold two values for `op` (sum, prod, min, max, any, all). Cells must be
/// scalar leaves (Integer / Float / Boolean).
fn fold_two(b: &mut IRBuilder, op: &str, acc: &Value, elem: &Value) -> Value {
    match op {
        "sum" => super::value_ops::apply_binary_op(b, "add", acc, elem),
        "prod" => super::value_ops::apply_binary_op(b, "mul", acc, elem),
        "min" => {
            let cond = super::value_ops::apply_binary_op(b, "lt", acc, elem);
            super::value_ops::select_value(b, &cond, acc, elem)
        }
        "max" => {
            let cond = super::value_ops::apply_binary_op(b, "gt", acc, elem);
            super::value_ops::select_value(b, &cond, acc, elem)
        }
        "any" => {
            let acc_b = super::value_ops::to_scalar_bool(b, acc);
            let elem_b = super::value_ops::to_scalar_bool(b, elem);
            b.ir_logical_or(&acc_b, &elem_b)
        }
        "all" => {
            let acc_b = super::value_ops::to_scalar_bool(b, acc);
            let elem_b = super::value_ops::to_scalar_bool(b, elem);
            b.ir_logical_and(&acc_b, &elem_b)
        }
        _ => panic!("fold_two: unsupported op {}", op),
    }
}

/// Reduce a slice of cells with the given op; honours empty-array semantics.
fn reduce_cells(b: &mut IRBuilder, op: &str, cells: &[Value]) -> Value {
    if cells.is_empty() {
        return match op {
            "sum" => b.ir_constant_int(0),
            "prod" => b.ir_constant_int(1),
            "any" => b.ir_constant_bool(false),
            "all" => b.ir_constant_bool(true),
            "min" | "max" => Value::None,
            _ => Value::None,
        };
    }
    if matches!(op, "any" | "all") {
        let mut acc = super::value_ops::to_scalar_bool(b, &cells[0]);
        for elem in &cells[1..] {
            let elem_b = super::value_ops::to_scalar_bool(b, elem);
            acc = if op == "any" {
                b.ir_logical_or(&acc, &elem_b)
            } else {
                b.ir_logical_and(&acc, &elem_b)
            };
        }
        return acc;
    }
    let mut acc = cells[0].clone();
    for elem in &cells[1..] {
        acc = fold_two(b, op, &acc, elem);
    }
    acc
}

/// Floor-rounded mean: divides the (float-promoted) sum by N. The result is
/// always a float so it can hold non-integer means cleanly.
fn mean_from_sum(b: &mut IRBuilder, total_sum: &Value, n: usize) -> Value {
    let total_f = match total_sum {
        Value::Float(_) => total_sum.clone(),
        _ => b.ir_float_cast(total_sum),
    };
    let n_val = b.ir_constant_float(n as f64);
    b.ir_div_f(&total_f, &n_val)
}

// ────────────────────────────────────────────────────────────────────────
// Whole-array reductions
// ────────────────────────────────────────────────────────────────────────

fn whole_array_reduce(b: &mut IRBuilder, op: &str, arr: &Value) -> Value {
    let cells = payload_cells(b, arr);
    reduce_cells(b, op, &cells)
}

fn whole_array_mean(b: &mut IRBuilder, arr: &Value) -> Value {
    let shape = shape_of(arr);
    let n: usize = shape.iter().product::<usize>().max(1);
    let s = whole_array_reduce(b, "sum", arr);
    mean_from_sum(b, &s, n)
}

// ────────────────────────────────────────────────────────────────────────
// Axis-aware reductions (stride-based, policy α)
// ────────────────────────────────────────────────────────────────────────

/// Resolve a (possibly negative) axis against rank, panicking on out-of-range.
fn resolve_axis(axis: i64, rank: usize) -> usize {
    let resolved = if axis < 0 { rank as i64 + axis } else { axis };
    if resolved < 0 || resolved >= rank as i64 {
        panic!(
            "reduce: axis {} is out of bounds for array of rank {}",
            axis, rank
        );
    }
    resolved as usize
}

/// Compute the axis-reduced output shape (axis dropped, or kept as 1 with
/// `keepdims=True`).
fn reduced_shape(shape: &[usize], axis: usize, keepdims: bool) -> Vec<usize> {
    let mut out = Vec::with_capacity(shape.len());
    for (ax, &dim) in shape.iter().enumerate() {
        if ax == axis {
            if keepdims {
                out.push(1);
            }
        } else {
            out.push(dim);
        }
    }
    out
}

/// Per-output-cell, iterate along the reduced axis and fold.
///
/// Implementation policy α (stride-based): for each output coordinate, derive
/// the corresponding *base* address into the source's flat payload by
/// summing `coord_k * src_strides[k]` over non-reduced axes; then walk the
/// reduced axis with `src_strides[axis]` increments and fold the cells.
fn axis_reduce(
    b: &mut IRBuilder,
    op: &str,
    arr: &Value,
    axis: usize,
    keepdims: bool,
) -> Value {
    let shape = shape_of(arr);
    let dtype = dtype_of(arr);
    let rank = shape.len();

    // Special case: rank-0 (scalar) — just whole-array reduce.
    if rank == 0 {
        return whole_array_reduce(b, op, arr);
    }

    // The source uses *natural* row-major strides over `shape`. We can't use
    // the StaticArray's `strides` field directly because views can have
    // arbitrary strides — `payload_cells` already collapses to a contiguous
    // logical view, so we re-derive strides over `shape`.
    let src_strides = row_major_strides(&shape);
    let cells = payload_cells(b, arr);

    let out_shape = reduced_shape(&shape, axis, keepdims);
    let out_total: usize = out_shape.iter().product();
    let axis_len = shape[axis];
    let axis_stride = src_strides[axis];

    // Walk the *non-reduced* axes in the same row-major order they appear in
    // `out_shape`. For each output flat index, compute the source base by
    // unflattening into the non-reduced coordinates.
    let kept_axes: Vec<usize> = (0..rank).filter(|&a| a != axis).collect();
    let kept_strides: Vec<usize> = kept_axes.iter().map(|&a| src_strides[a]).collect();
    let kept_dims: Vec<usize> = kept_axes.iter().map(|&a| shape[a]).collect();
    let kept_out_strides = row_major_strides(&kept_dims);

    let mut out_flat: Vec<Value> = Vec::with_capacity(out_total);
    for out_idx in 0..out_total {
        // Decode out_idx into kept coords; for keepdims, the axis position
        // contributes a coord of 0 (size 1), so the same kept_out_strides
        // covers both keepdims=True and keepdims=False — but for
        // keepdims=True the out_total includes the size-1 axis, which is 1,
        // so the iteration count matches kept_dims' product. Build a
        // dedicated decode that uses kept_out_strides directly (size-1 axis
        // contributes 0).
        let mut remaining = out_idx;
        let mut base: usize = 0;
        for (k, &stride) in kept_out_strides.iter().enumerate() {
            let coord = if stride == 0 { 0 } else { remaining / stride };
            remaining %= stride.max(1);
            base += coord * kept_strides[k];
        }
        // Collect the fiber along axis.
        let mut fiber: Vec<Value> = Vec::with_capacity(axis_len);
        for j in 0..axis_len {
            let idx = base + j * axis_stride;
            fiber.push(cells[idx].clone());
        }
        out_flat.push(reduce_cells(b, op, &fiber));
    }

    // Output dtype: same as input (sum/prod/min/max/any/all preserve dtype).
    // mean is handled separately.
    let out_dtype = dtype;
    build_static_array_from_flat(b, out_flat, out_shape, out_dtype)
}

fn axis_mean(b: &mut IRBuilder, arr: &Value, axis: usize, keepdims: bool) -> Value {
    let shape = shape_of(arr);
    let n = shape[axis];
    // Reduce-sum along the axis, then divide each cell by n. The cells are
    // float-promoted to match numpy mean semantics.
    let summed = axis_reduce(b, "sum", arr, axis, keepdims);
    // summed is a StaticArray; build a new StaticArray by dividing each cell
    // by n (using float cast).
    let cells = payload_cells(b, &summed);
    let out_shape = match &summed {
        Value::StaticArray { shape, .. } => shape.clone(),
        _ => panic!(),
    };
    let n_val = b.ir_constant_float(n as f64);
    let mut out: Vec<Value> = Vec::with_capacity(cells.len());
    for c in &cells {
        let cf = match c {
            Value::Float(_) => c.clone(),
            _ => b.ir_float_cast(c),
        };
        out.push(b.ir_div_f(&cf, &n_val));
    }
    build_static_array_from_flat(b, out, out_shape, NumberType::Float)
}

// ────────────────────────────────────────────────────────────────────────
// Argmax / argmin
// ────────────────────────────────────────────────────────────────────────

fn whole_array_argmax_argmin(b: &mut IRBuilder, arr: &Value, is_max: bool) -> Value {
    let cells = payload_cells(b, arr);
    if cells.is_empty() {
        return b.ir_constant_int(0);
    }
    let dtype = dtype_of(arr);
    let mut best_idx = b.ir_constant_int(0);
    let mut best_val = cells[0].clone();
    for (i, elem) in cells.iter().enumerate().skip(1) {
        let cond = if dtype == NumberType::Float {
            if is_max {
                b.ir_greater_than_f(elem, &best_val)
            } else {
                b.ir_less_than_f(elem, &best_val)
            }
        } else if is_max {
            b.ir_greater_than_i(elem, &best_val)
        } else {
            b.ir_less_than_i(elem, &best_val)
        };
        let idx_val = b.ir_constant_int(i as i64);
        best_idx = b.ir_select_i(&cond, &idx_val, &best_idx);
        best_val = if dtype == NumberType::Float {
            b.ir_select_f(&cond, elem, &best_val)
        } else {
            b.ir_select_i(&cond, elem, &best_val)
        };
    }
    best_idx
}

/// Axis-aware argmax / argmin. Output is an Integer `StaticArray` with the
/// reduced axis dropped (or kept as 1 with `keepdims=True`).
fn axis_argmax_argmin(
    b: &mut IRBuilder,
    arr: &Value,
    axis: usize,
    is_max: bool,
    keepdims: bool,
) -> Value {
    let shape = shape_of(arr);
    let dtype = dtype_of(arr);
    let rank = shape.len();
    if rank == 0 {
        return whole_array_argmax_argmin(b, arr, is_max);
    }

    let src_strides = row_major_strides(&shape);
    let cells = payload_cells(b, arr);

    let out_shape = reduced_shape(&shape, axis, keepdims);
    let out_total: usize = out_shape.iter().product();
    let axis_len = shape[axis];
    let axis_stride = src_strides[axis];

    let kept_axes: Vec<usize> = (0..rank).filter(|&a| a != axis).collect();
    let kept_strides: Vec<usize> = kept_axes.iter().map(|&a| src_strides[a]).collect();
    let kept_dims: Vec<usize> = kept_axes.iter().map(|&a| shape[a]).collect();
    let kept_out_strides = row_major_strides(&kept_dims);

    let mut out_flat: Vec<Value> = Vec::with_capacity(out_total);
    for out_idx in 0..out_total {
        let mut remaining = out_idx;
        let mut base: usize = 0;
        for (k, &stride) in kept_out_strides.iter().enumerate() {
            let coord = if stride == 0 { 0 } else { remaining / stride };
            remaining %= stride.max(1);
            base += coord * kept_strides[k];
        }
        // Inline argmax/argmin over this fiber.
        let mut best_idx = b.ir_constant_int(0);
        let mut best_val = cells[base].clone();
        for j in 1..axis_len {
            let elem = &cells[base + j * axis_stride];
            let cond = if dtype == NumberType::Float {
                if is_max {
                    b.ir_greater_than_f(elem, &best_val)
                } else {
                    b.ir_less_than_f(elem, &best_val)
                }
            } else if is_max {
                b.ir_greater_than_i(elem, &best_val)
            } else {
                b.ir_less_than_i(elem, &best_val)
            };
            let idx_val = b.ir_constant_int(j as i64);
            best_idx = b.ir_select_i(&cond, &idx_val, &best_idx);
            best_val = if dtype == NumberType::Float {
                b.ir_select_f(&cond, elem, &best_val)
            } else {
                b.ir_select_i(&cond, elem, &best_val)
            };
        }
        out_flat.push(best_idx);
    }
    build_static_array_from_flat(b, out_flat, out_shape, NumberType::Integer)
}

// ────────────────────────────────────────────────────────────────────────
// Public entry points
// ────────────────────────────────────────────────────────────────────────

/// Try to apply a reduction op (`sum` / `prod` / `min` / `max` / `mean` /
/// `any` / `all`) natively on a `Value::StaticArray`. Returns `Some(result)`
/// when this path handles the case; `None` otherwise (caller should fall
/// back to the legacy materialised-list path).
///
/// `axis_arg` follows the named-attr convention: `None` or `Value::None`
/// means whole-array reduction; an integer means axis-aware.
pub fn try_apply_reduce(
    b: &mut IRBuilder,
    op: &str,
    val: &Value,
    axis_arg: Option<&Value>,
    keepdims: bool,
) -> Option<Value> {
    if !matches!(val, Value::StaticArray { .. }) {
        return None;
    }
    if !matches!(op, "sum" | "prod" | "min" | "max" | "mean" | "any" | "all") {
        return None;
    }
    // Complex StaticArray isn't constructed today; defer to legacy.
    if matches!(dtype_of(val), NumberType::Complex) {
        return None;
    }

    // Extract a static int axis, if any.
    let axis_int: Option<i64> = match axis_arg {
        None => None,
        Some(Value::None) => None,
        Some(v) => match v.int_val() {
            Some(i) => Some(i),
            // Non-static axis: defer to legacy.
            None => return None,
        },
    };

    if let Some(ax) = axis_int {
        let shape = shape_of(val);
        if shape.is_empty() {
            // 0-rank: ignore axis (numpy errors, but match legacy lenience).
            return Some(whole_array_reduce(b, op, val));
        }
        let axis = resolve_axis(ax, shape.len());
        if op == "mean" {
            return Some(axis_mean(b, val, axis, keepdims));
        }
        return Some(axis_reduce(b, op, val, axis, keepdims));
    }

    // Whole-array.
    if op == "mean" {
        return Some(whole_array_mean(b, val));
    }
    Some(whole_array_reduce(b, op, val))
}

/// Try to apply argmax / argmin natively on a `Value::StaticArray`. Same
/// `axis_arg` convention as `try_apply_reduce`.
pub fn try_apply_argmax_argmin(
    b: &mut IRBuilder,
    val: &Value,
    axis_arg: Option<&Value>,
    is_max: bool,
    keepdims: bool,
) -> Option<Value> {
    if !matches!(val, Value::StaticArray { .. }) {
        return None;
    }
    if matches!(dtype_of(val), NumberType::Complex) {
        return None;
    }
    let axis_int: Option<i64> = match axis_arg {
        None => None,
        Some(Value::None) => None,
        Some(v) => match v.int_val() {
            Some(i) => Some(i),
            None => return None,
        },
    };
    if let Some(ax) = axis_int {
        let shape = shape_of(val);
        if shape.is_empty() {
            return Some(whole_array_argmax_argmin(b, val, is_max));
        }
        let axis = resolve_axis(ax, shape.len());
        return Some(axis_argmax_argmin(b, val, axis, is_max, keepdims));
    }
    Some(whole_array_argmax_argmin(b, val, is_max))
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

    fn make_2d_int(b: &mut IRBuilder, rows: &[&[i64]]) -> Value {
        let row_lists: Vec<Value> = rows
            .iter()
            .map(|r| list_of(r.iter().map(|n| b.ir_constant_int(*n)).collect()))
            .collect();
        let lst = list_of(row_lists);
        super::super::static_array::to_static_array(b, &lst).expect("StaticArray")
    }

    #[test]
    fn whole_array_sum_constant_folds() {
        let mut b = IRBuilder::new();
        let a = make_1d_int(&mut b, &[1, 2, 3, 4]);
        let out = try_apply_reduce(&mut b, "sum", &a, None, false).expect("native");
        assert_eq!(out.int_val(), Some(10));
    }

    #[test]
    fn whole_array_prod_constant_folds() {
        let mut b = IRBuilder::new();
        let a = make_1d_int(&mut b, &[1, 2, 3, 4]);
        let out = try_apply_reduce(&mut b, "prod", &a, None, false).expect("native");
        assert_eq!(out.int_val(), Some(24));
    }

    #[test]
    fn whole_array_min_max_constant_folds() {
        let mut b = IRBuilder::new();
        let a = make_1d_int(&mut b, &[3, 1, 4, 1, 5, 9, 2, 6]);
        let mn = try_apply_reduce(&mut b, "min", &a, None, false).expect("native");
        let mx = try_apply_reduce(&mut b, "max", &a, None, false).expect("native");
        assert_eq!(mn.int_val(), Some(1));
        assert_eq!(mx.int_val(), Some(9));
    }

    #[test]
    fn axis_sum_2d_axis0_constant_folds() {
        let mut b = IRBuilder::new();
        // [[1, 2, 3], [4, 5, 6]] → axis=0 → [5, 7, 9]
        let a = make_2d_int(&mut b, &[&[1, 2, 3], &[4, 5, 6]]);
        let axis = b.ir_constant_int(0);
        let out = try_apply_reduce(&mut b, "sum", &a, Some(&axis), false).expect("native");
        match &out {
            Value::StaticArray { shape, .. } => assert_eq!(*shape, vec![3]),
            _ => panic!("expected StaticArray"),
        }
        let cells = payload_cells(&mut b, &out);
        let v: Vec<Option<i64>> = cells.iter().map(|c| c.int_val()).collect();
        assert_eq!(v, vec![Some(5), Some(7), Some(9)]);
    }

    #[test]
    fn axis_sum_2d_axis1_constant_folds() {
        let mut b = IRBuilder::new();
        let a = make_2d_int(&mut b, &[&[1, 2, 3], &[4, 5, 6]]);
        let axis = b.ir_constant_int(1);
        let out = try_apply_reduce(&mut b, "sum", &a, Some(&axis), false).expect("native");
        match &out {
            Value::StaticArray { shape, .. } => assert_eq!(*shape, vec![2]),
            _ => panic!("expected StaticArray"),
        }
        let cells = payload_cells(&mut b, &out);
        let v: Vec<Option<i64>> = cells.iter().map(|c| c.int_val()).collect();
        assert_eq!(v, vec![Some(6), Some(15)]);
    }

    #[test]
    fn axis_sum_negative_axis() {
        let mut b = IRBuilder::new();
        let a = make_2d_int(&mut b, &[&[1, 2, 3], &[4, 5, 6]]);
        let axis = b.ir_constant_int(-1);
        let out = try_apply_reduce(&mut b, "sum", &a, Some(&axis), false).expect("native");
        let cells = payload_cells(&mut b, &out);
        let v: Vec<Option<i64>> = cells.iter().map(|c| c.int_val()).collect();
        assert_eq!(v, vec![Some(6), Some(15)]);
    }

    #[test]
    fn whole_array_argmax() {
        let mut b = IRBuilder::new();
        let a = make_1d_int(&mut b, &[3, 1, 4, 1, 5, 9, 2, 6]);
        let out = try_apply_argmax_argmin(&mut b, &a, None, true, false).expect("native");
        assert_eq!(out.int_val(), Some(5));
    }

    #[test]
    fn whole_array_argmin() {
        let mut b = IRBuilder::new();
        let a = make_1d_int(&mut b, &[3, 1, 4, 1, 5, 9, 2, 6]);
        // numpy returns the *first* min, so argmin([3,1,4,1,...]) == 1.
        let out = try_apply_argmax_argmin(&mut b, &a, None, false, false).expect("native");
        assert_eq!(out.int_val(), Some(1));
    }

    #[test]
    fn whole_array_any_all_bool() {
        let mut b = IRBuilder::new();
        let a = make_1d_int(&mut b, &[0, 0, 1]);
        let any_v = try_apply_reduce(&mut b, "any", &a, None, false).expect("native");
        let all_v = try_apply_reduce(&mut b, "all", &a, None, false).expect("native");
        assert_eq!(any_v.bool_val(), Some(true));
        assert_eq!(all_v.bool_val(), Some(false));
    }

    #[test]
    fn keepdims_axis_reduction() {
        let mut b = IRBuilder::new();
        let a = make_2d_int(&mut b, &[&[1, 2, 3], &[4, 5, 6]]);
        let axis = b.ir_constant_int(0);
        let out = try_apply_reduce(&mut b, "sum", &a, Some(&axis), true).expect("native");
        match &out {
            Value::StaticArray { shape, .. } => assert_eq!(*shape, vec![1, 3]),
            _ => panic!("expected StaticArray"),
        }
    }

    #[test]
    fn returns_none_for_non_static_array() {
        let mut b = IRBuilder::new();
        let lst = list_of(vec![b.ir_constant_int(1), b.ir_constant_int(2)]);
        assert!(try_apply_reduce(&mut b, "sum", &lst, None, false).is_none());
    }
}
