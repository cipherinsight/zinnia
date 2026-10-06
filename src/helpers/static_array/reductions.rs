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

use super::super::shape_arith::row_major_strides;
use super::base::{build_static_array_from_flat, build_static_array_from_flat_complex};
use super::elementwise::{payload_cells, product_provenance_payload_cells};

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
        "sum" => super::super::value_ops::apply_binary_op(b, "add", acc, elem),
        "prod" => super::super::value_ops::apply_binary_op(b, "mul", acc, elem),
        "min" => {
            let cond = super::super::value_ops::apply_binary_op(b, "lt", acc, elem);
            let out = super::super::value_ops::select_value(b, &cond, acc, elem);
            b.deposit_minmax_facts(&out, acc, elem, false);
            out
        }
        "max" => {
            let cond = super::super::value_ops::apply_binary_op(b, "gt", acc, elem);
            let out = super::super::value_ops::select_value(b, &cond, acc, elem);
            b.deposit_minmax_facts(&out, acc, elem, true);
            out
        }
        "any" => {
            let acc_b = super::super::value_ops::to_scalar_bool(b, acc);
            let elem_b = super::super::value_ops::to_scalar_bool(b, elem);
            b.ir_logical_or(&acc_b, &elem_b)
        }
        "all" => {
            let acc_b = super::super::value_ops::to_scalar_bool(b, acc);
            let elem_b = super::super::value_ops::to_scalar_bool(b, elem);
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
        let mut acc = super::super::value_ops::to_scalar_bool(b, &cells[0]);
        for elem in &cells[1..] {
            let elem_b = super::super::value_ops::to_scalar_bool(b, elem);
            acc = if op == "any" {
                b.ir_logical_or(&acc, &elem_b)
            } else {
                b.ir_logical_and(&acc, &elem_b)
            };
        }
        return acc;
    }
    if matches!(op, "min" | "max") {
        if let Some(extreme) = sorted_extreme(b, op == "max", cells) {
            return extreme;
        }
    }
    if op == "sum" && cells.iter().all(|cell| matches!(cell, Value::Float(_))) {
        let mut level: Vec<Value> = cells.to_vec();
        while level.len() > 1 {
            let mut next = Vec::with_capacity(level.len().div_ceil(4));
            for chunk in level.chunks(4) {
                next.push(b.ir_sum_f(chunk));
            }
            level = next;
        }
        return level.pop().expect("non-empty level");
    }
    let mut acc = cells[0].clone();
    for elem in &cells[1..] {
        acc = fold_two(b, op, &acc, elem);
    }
    acc
}

/// The smallest (`largest`: the largest) of two or more numeric cells, as
/// the output of rank 0 (`n - 1`) of their sort: the sort lowering then
/// picks the cheapest construction for the target (the scan of the
/// adjacent-swap network's first pass, counts, or a hint bounded by asserted
/// comparisons and a membership product). The facts a select chain deposits
/// (the result bounds every element) are deposited for every element.
/// `None` for mixed or Boolean cells, which keep the chain.
fn sorted_extreme(b: &mut IRBuilder, largest: bool, cells: &[Value]) -> Option<Value> {
    let floats = cells.iter().all(|c| matches!(c, Value::Float(_)));
    let ints = cells.iter().all(|c| matches!(c, Value::Integer(_)));
    if cells.len() < 2 || !(floats || ints) {
        return None;
    }
    let elements: std::sync::Arc<[Value]> = cells.iter().map(|c| b.ensure_ptr(c)).collect();
    let rank = if largest { cells.len() as u32 - 1 } else { 0 };
    let out = b.ir_sorted(&elements, rank, false);
    for element in cells {
        b.deposit_minmax_facts(&out, element, element, largest);
    }
    Some(out)
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

/// Fold a slice of Complex cells with the given binary op (`add` for sum or
/// `mul` for prod) using the existing component-wise complex IR.
fn complex_reduce_cells(b: &mut IRBuilder, bin_op: &str, cells: &[Value]) -> Value {
    if cells.is_empty() {
        let zero = b.ir_constant_float(0.0);
        let one = b.ir_constant_float(1.0);
        let (rr, ii) = if bin_op == "add" {
            (zero.clone(), zero)
        } else {
            (one, zero)
        };
        return crate::helpers::value_ops::pack_complex_value(rr, ii);
    }
    let mut acc = cells[0].clone();
    for c in &cells[1..] {
        acc = crate::helpers::value_ops::apply_complex_binary_op(b, bin_op, &acc, c);
    }
    acc
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

fn try_resolve_axis(axis: i64, rank: usize) -> Option<usize> {
    let resolved = if axis < 0 { rank as i64 + axis } else { axis };
    if resolved < 0 || resolved >= rank as i64 {
        return None;
    }
    Some(resolved as usize)
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

/// Resolve a static scalar/tuple/list axis argument. `None` means whole-array
/// reduction; `Some(vec)` means a static axis set. Dynamic composite elements,
/// duplicate axes, and empty tuples fail open to the legacy path.
fn parse_axis_spec(axis_arg: Option<&Value>, rank: usize) -> Option<Option<Vec<usize>>> {
    match axis_arg {
        None | Some(Value::None) => Some(None),
        Some(Value::Tuple(data)) | Some(Value::List(data)) => {
            if data.values.is_empty() {
                return None;
            }
            let mut axes = Vec::with_capacity(data.values.len());
            for v in &data.values {
                let axis = try_resolve_axis(v.int_val()?, rank)?;
                if axes.contains(&axis) {
                    return None;
                }
                axes.push(axis);
            }
            axes.sort_unstable();
            Some(Some(axes))
        }
        Some(v) => Some(Some(vec![resolve_axis(v.int_val()?, rank)])),
    }
}

fn reduced_shape_multi(shape: &[usize], axes: &[usize], keepdims: bool) -> Vec<usize> {
    let mut out = Vec::with_capacity(shape.len());
    for (ax, &dim) in shape.iter().enumerate() {
        if axes.contains(&ax) {
            if keepdims {
                out.push(1);
            }
        } else {
            out.push(dim);
        }
    }
    out
}

fn multi_axis_reduce(
    b: &mut IRBuilder,
    op: &str,
    arr: &Value,
    axes: &[usize],
    keepdims: bool,
) -> Value {
    if op == "sum" {
        if let Value::StaticArray {
            segment_id,
            offset,
            dtype,
            ..
        } = arr
        {
            if *offset == 0 && !matches!(dtype, NumberType::Complex) {
                if let Some(provenance) = b.static_array_product_provenance.get(segment_id).cloned()
                {
                    return multi_axis_reduce_product_sum_provenance(
                        b,
                        provenance,
                        axes,
                        keepdims,
                        *segment_id,
                    );
                }
            }
        }
    }

    if axes.len() == 1 {
        return axis_reduce(b, op, arr, axes[0], keepdims);
    }

    let shape = shape_of(arr);
    let dtype = dtype_of(arr);
    let rank = shape.len();
    if rank == 0 {
        return whole_array_reduce(b, op, arr);
    }

    let src_strides = row_major_strides(&shape);
    let cells = payload_cells(b, arr);

    let out_shape = reduced_shape_multi(&shape, axes, keepdims);
    let out_total: usize = out_shape.iter().product();
    let reduced_axes: Vec<usize> = axes.to_vec();
    let kept_axes: Vec<usize> = (0..rank).filter(|a| !reduced_axes.contains(a)).collect();
    let kept_strides: Vec<usize> = kept_axes.iter().map(|&a| src_strides[a]).collect();
    let kept_dims: Vec<usize> = kept_axes.iter().map(|&a| shape[a]).collect();
    let kept_out_strides = row_major_strides(&kept_dims);
    let reduced_dims: Vec<usize> = reduced_axes.iter().map(|&a| shape[a]).collect();
    let reduced_src_strides: Vec<usize> = reduced_axes.iter().map(|&a| src_strides[a]).collect();
    let reduced_out_strides = row_major_strides(&reduced_dims);
    let reduced_total: usize = reduced_dims.iter().product();

    let mut out_flat: Vec<Value> = Vec::with_capacity(out_total);
    for out_idx in 0..out_total {
        let mut remaining = out_idx;
        let mut base: usize = 0;
        for (k, &stride) in kept_out_strides.iter().enumerate() {
            let coord = if stride == 0 { 0 } else { remaining / stride };
            remaining %= stride.max(1);
            base += coord * kept_strides[k];
        }

        let mut fiber: Vec<Value> = Vec::with_capacity(reduced_total);
        for red_idx in 0..reduced_total {
            let mut remaining = red_idx;
            let mut idx = base;
            for (k, &stride) in reduced_out_strides.iter().enumerate() {
                let coord = if stride == 0 { 0 } else { remaining / stride };
                remaining %= stride.max(1);
                idx += coord * reduced_src_strides[k];
            }
            fiber.push(cells[idx].clone());
        }
        out_flat.push(reduce_cells(b, op, &fiber));
    }

    build_static_array_from_flat(b, out_flat, out_shape, dtype)
}

fn multi_axis_reduce_product_sum_provenance(
    b: &mut IRBuilder,
    provenance: crate::builder::StaticArrayProductProvenance,
    axes: &[usize],
    keepdims: bool,
    source_segment_id: u32,
) -> Value {
    let shape = provenance.out_shape.clone();
    let dtype = provenance.dtype;
    let rank = shape.len();
    if rank == 0 {
        let cells = product_provenance_payload_cells(b, &provenance);
        b.static_array_payload
            .insert(source_segment_id, cells.clone());
        return reduce_cells(b, "sum", &cells);
    }

    let src_strides = row_major_strides(&shape);
    let out_shape = reduced_shape_multi(&shape, axes, keepdims);
    let out_total: usize = out_shape.iter().product();
    let reduced_axes: Vec<usize> = axes.to_vec();
    let kept_axes: Vec<usize> = (0..rank).filter(|a| !reduced_axes.contains(a)).collect();
    let kept_strides: Vec<usize> = kept_axes.iter().map(|&a| src_strides[a]).collect();
    let kept_dims: Vec<usize> = kept_axes.iter().map(|&a| shape[a]).collect();
    let kept_out_strides = row_major_strides(&kept_dims);
    let reduced_dims: Vec<usize> = reduced_axes.iter().map(|&a| shape[a]).collect();
    let reduced_src_strides: Vec<usize> = reduced_axes.iter().map(|&a| src_strides[a]).collect();
    let reduced_out_strides = row_major_strides(&reduced_dims);
    let reduced_total: usize = reduced_dims.iter().product();

    let out_product_strides = row_major_strides(&provenance.out_shape);
    let lhs_idx = super::elementwise::make_product_broadcast_indexer(
        &provenance.lhs_shape,
        &provenance.out_shape,
    );
    let rhs_idx = super::elementwise::make_product_broadcast_indexer(
        &provenance.rhs_shape,
        &provenance.out_shape,
    );

    let mut out_flat: Vec<Value> = Vec::with_capacity(out_total);
    for out_idx in 0..out_total {
        let mut remaining = out_idx;
        let mut base: usize = 0;
        for (k, &stride) in kept_out_strides.iter().enumerate() {
            let coord = if stride == 0 { 0 } else { remaining / stride };
            remaining %= stride.max(1);
            base += coord * kept_strides[k];
        }

        if dtype == NumberType::Float {
            let mut pairs: Vec<(Value, Value)> = Vec::with_capacity(reduced_total);
            for red_idx in 0..reduced_total {
                let mut remaining = red_idx;
                let mut idx = base;
                for (k, &stride) in reduced_out_strides.iter().enumerate() {
                    let coord = if stride == 0 { 0 } else { remaining / stride };
                    remaining %= stride.max(1);
                    idx += coord * reduced_src_strides[k];
                }
                let coords = super::super::shape_arith::decode_coords(
                    idx,
                    &provenance.out_shape,
                    &out_product_strides,
                );
                let lhs = &provenance.lhs_cells[lhs_idx.flat_index(&coords)];
                let rhs = &provenance.rhs_cells[rhs_idx.flat_index(&coords)];
                pairs.push((ensure_float_operand(b, lhs), ensure_float_operand(b, rhs)));
            }
            out_flat.push(if pairs.is_empty() {
                b.ir_constant_float(0.0)
            } else {
                b.ir_product_sum_f(&pairs)
            });
        } else {
            let mut fiber: Vec<Value> = Vec::with_capacity(reduced_total);
            for red_idx in 0..reduced_total {
                let mut remaining = red_idx;
                let mut idx = base;
                for (k, &stride) in reduced_out_strides.iter().enumerate() {
                    let coord = if stride == 0 { 0 } else { remaining / stride };
                    remaining %= stride.max(1);
                    idx += coord * reduced_src_strides[k];
                }
                let coords = super::super::shape_arith::decode_coords(
                    idx,
                    &provenance.out_shape,
                    &out_product_strides,
                );
                let l = &provenance.lhs_cells[lhs_idx.flat_index(&coords)];
                let r = &provenance.rhs_cells[rhs_idx.flat_index(&coords)];
                fiber.push(super::super::value_ops::apply_scalar_binary_op(
                    b, "mul", l, r,
                ));
            }
            out_flat.push(reduce_cells(b, "sum", &fiber));
        }
    }

    build_static_array_from_flat(b, out_flat, out_shape, dtype)
}

fn ensure_float_operand(b: &mut IRBuilder, value: &Value) -> Value {
    match value {
        Value::Float(_) => value.clone(),
        _ => b.ir_float_cast(value),
    }
}

fn fire_var_std_contract_for_output(b: &mut IRBuilder, op: &str, out: &Value, reduced_len: usize) {
    let len = b.ir_constant_int(reduced_len as i64);
    let Some(len_vid) = len.value_id() else {
        return;
    };
    let mut formals = std::collections::HashMap::new();
    formals.insert("len_arr".to_string(), len_vid);
    match out {
        Value::StaticArray { .. } => {
            for cell in payload_cells(b, out) {
                if let Some(out_vid) = cell.value_id() {
                    b.fire_contract(op, out_vid, &formals);
                }
            }
        }
        _ => {
            if let Some(out_vid) = out.value_id() {
                b.fire_contract(op, out_vid, &formals);
            }
        }
    }
}

fn parse_static_ddof(ddof_arg: Option<&Value>) -> i64 {
    match ddof_arg {
        None | Some(Value::None) => 0,
        Some(v) => v
            .int_val()
            .unwrap_or_else(|| panic!("var/std: ddof must be a static integer")),
    }
}

fn checked_var_divisor(reduced_len: usize, ddof: i64) -> usize {
    let divisor = reduced_len as i64 - ddof;
    if divisor <= 0 {
        panic!(
            "var/std: invalid ddof {}; reduced length {} gives non-positive divisor",
            ddof, reduced_len
        );
    }
    divisor as usize
}

fn variance_from_fiber(b: &mut IRBuilder, fiber: &[Value], ddof: i64) -> Value {
    let reduced_len = fiber.len();
    let divisor = checked_var_divisor(reduced_len, ddof);
    let sum = reduce_cells(b, "sum", fiber);
    let mean = mean_from_sum(b, &sum, reduced_len);
    let squares: Vec<(Value, Value)> = fiber
        .iter()
        .map(|cell| {
            let cf = ensure_float_operand(b, cell);
            let diff = b.ir_sub_f(&cf, &mean);
            (diff.clone(), diff)
        })
        .collect();
    let sq_sum = b.ir_dot_f(&squares);
    let denom = b.ir_constant_float(divisor as f64);
    b.ir_div_f(&sq_sum, &denom)
}

fn multi_axis_mean(b: &mut IRBuilder, arr: &Value, axes: &[usize], keepdims: bool) -> Value {
    if axes.len() == 1 {
        return axis_mean(b, arr, axes[0], keepdims);
    }

    let shape = shape_of(arr);
    let n: usize = axes.iter().map(|&a| shape[a]).product::<usize>().max(1);
    let summed = multi_axis_reduce(b, "sum", arr, axes, keepdims);
    let cells = payload_cells(b, &summed);
    let out_shape = shape_of(&summed);
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

fn multi_axis_var_std(
    b: &mut IRBuilder,
    op: &str,
    arr: &Value,
    axes: &[usize],
    keepdims: bool,
    ddof: i64,
) -> Value {
    let shape = shape_of(arr);
    let rank = shape.len();
    if rank == 0 {
        let cells = payload_cells(b, arr);
        let var = variance_from_fiber(b, &cells, ddof);
        let out = if op == "std" { b.ir_sqrt_f(&var) } else { var };
        fire_var_std_contract_for_output(b, op, &out, cells.len());
        return out;
    }

    let src_strides = row_major_strides(&shape);
    let cells = payload_cells(b, arr);
    let out_shape = reduced_shape_multi(&shape, axes, keepdims);
    let out_total: usize = out_shape.iter().product();
    let reduced_axes: Vec<usize> = axes.to_vec();
    let kept_axes: Vec<usize> = (0..rank).filter(|a| !reduced_axes.contains(a)).collect();
    let kept_strides: Vec<usize> = kept_axes.iter().map(|&a| src_strides[a]).collect();
    let kept_dims: Vec<usize> = kept_axes.iter().map(|&a| shape[a]).collect();
    let kept_out_strides = row_major_strides(&kept_dims);
    let reduced_dims: Vec<usize> = reduced_axes.iter().map(|&a| shape[a]).collect();
    let reduced_src_strides: Vec<usize> = reduced_axes.iter().map(|&a| src_strides[a]).collect();
    let reduced_out_strides = row_major_strides(&reduced_dims);
    let reduced_total: usize = reduced_dims.iter().product();
    checked_var_divisor(reduced_total, ddof);

    let mut out_flat: Vec<Value> = Vec::with_capacity(out_total);
    for out_idx in 0..out_total {
        let mut remaining = out_idx;
        let mut base: usize = 0;
        for (k, &stride) in kept_out_strides.iter().enumerate() {
            let coord = if stride == 0 { 0 } else { remaining / stride };
            remaining %= stride.max(1);
            base += coord * kept_strides[k];
        }

        let mut fiber: Vec<Value> = Vec::with_capacity(reduced_total);
        for red_idx in 0..reduced_total {
            let mut remaining = red_idx;
            let mut idx = base;
            for (k, &stride) in reduced_out_strides.iter().enumerate() {
                let coord = if stride == 0 { 0 } else { remaining / stride };
                remaining %= stride.max(1);
                idx += coord * reduced_src_strides[k];
            }
            fiber.push(cells[idx].clone());
        }
        let var = variance_from_fiber(b, &fiber, ddof);
        out_flat.push(if op == "std" { b.ir_sqrt_f(&var) } else { var });
    }

    let out = build_static_array_from_flat(b, out_flat, out_shape, NumberType::Float);
    fire_var_std_contract_for_output(b, op, &out, reduced_total);
    out
}

/// Per-output-cell, iterate along the reduced axis and fold.
///
/// Implementation policy α (stride-based): for each output coordinate, derive
/// the corresponding *base* address into the source's flat payload by
/// summing `coord_k * src_strides[k]` over non-reduced axes; then walk the
/// reduced axis with `src_strides[axis]` increments and fold the cells.
fn axis_reduce(b: &mut IRBuilder, op: &str, arr: &Value, axis: usize, keepdims: bool) -> Value {
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

/// Complex variant of [`axis_reduce`]. It mirrors the same stride/fiber
/// traversal, folding each Complex payload cell with component-wise IR and
/// rebuilding the result as a dual-segment Complex StaticArray.
fn axis_reduce_complex(
    b: &mut IRBuilder,
    bin_op: &str,
    arr: &Value,
    axis: usize,
    keepdims: bool,
) -> Value {
    let shape = shape_of(arr);
    let rank = shape.len();
    if rank == 0 {
        let cells = payload_cells(b, arr);
        return complex_reduce_cells(b, bin_op, &cells);
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

        let mut fiber: Vec<Value> = Vec::with_capacity(axis_len);
        for j in 0..axis_len {
            let idx = base + j * axis_stride;
            fiber.push(cells[idx].clone());
        }
        out_flat.push(complex_reduce_cells(b, bin_op, &fiber));
    }

    let mut reals = Vec::with_capacity(out_flat.len());
    let mut imags = Vec::with_capacity(out_flat.len());
    for c in &out_flat {
        let (real, imag) = crate::helpers::value_ops::unpack_value_to_complex_parts(b, c);
        reals.push(real);
        imags.push(imag);
    }
    build_static_array_from_flat_complex(b, reals, imags, out_shape)
}

fn axis_mean_complex(b: &mut IRBuilder, arr: &Value, axis: usize, keepdims: bool) -> Value {
    let shape = shape_of(arr);
    let n = shape[axis];
    let summed = axis_reduce_complex(b, "add", arr, axis, keepdims);
    let cells = payload_cells(b, &summed);
    let out_shape = shape_of(&summed);
    let n_val = b.ir_constant_float(n as f64);

    let mut reals = Vec::with_capacity(cells.len());
    let mut imags = Vec::with_capacity(cells.len());
    for c in &cells {
        let (sr, si) = crate::helpers::value_ops::unpack_value_to_complex_parts(b, c);
        reals.push(b.ir_div_f(&sr, &n_val));
        imags.push(b.ir_div_f(&si, &n_val));
    }

    build_static_array_from_flat_complex(b, reals, imags, out_shape)
}

// ────────────────────────────────────────────────────────────────────────
// Argmax / argmin
// ────────────────────────────────────────────────────────────────────────

/// A comparison cell `lhs cmp rhs`: its operands' statements and values,
/// the direction and the strictness.
struct Comparison {
    lhs: crate::types::StmtId,
    rhs: crate::types::StmtId,
    /// The operands with float casts looked through, to recognize one
    /// scalar cast once per element.
    lhs_source: crate::types::StmtId,
    rhs_source: crate::types::StmtId,
    lhs_value: Value,
    rhs_value: Value,
    /// True for `>` and `>=`, false for `<` and `<=`.
    greater: bool,
    /// `==` operands make it False (`>`, `<`).
    strict: bool,
}

/// The statement a float cast reads (`FloatCast(x)` is `x` for identity:
/// a broadcast casts the same scalar once per element).
fn uncast(b: &IRBuilder, stmt: crate::types::StmtId) -> crate::types::StmtId {
    match b.stmts.get(stmt as usize) {
        Some(s) if matches!(s.ir, crate::ir_defs::IR::FloatCast) => s.arguments[0],
        _ => stmt,
    }
}

fn comparison_of(b: &IRBuilder, cell: &Value) -> Option<Comparison> {
    use crate::ir_defs::IR;
    let stmt = b.stmts.get(cell.stmt_id()? as usize)?;
    let (greater, strict) = match stmt.ir {
        IR::GtF | IR::GtI => (true, true),
        IR::GteF | IR::GteI => (true, false),
        IR::LtF | IR::LtI => (false, true),
        IR::LteF | IR::LteI => (false, false),
        _ => return None,
    };
    let [lhs, rhs] = stmt.arguments[..] else { return None };
    let [lhs_vid, rhs_vid] = stmt.arg_values[..] else { return None };
    Some(Comparison {
        lhs,
        rhs,
        lhs_source: uncast(b, lhs),
        rhs_source: uncast(b, rhs),
        lhs_value: b.value_of_vid(lhs_vid)?,
        rhs_value: b.value_of_vid(rhs_vid)?,
        greater,
        strict,
    })
}

/// `argmax` of Booleans `x_k cmp e` over the outputs `x_k` of one sort in
/// rank order (`(np.sort(a) > e).argmax()`, the `searchsorted` idiom) and
/// one scalar `e`, without the sorted outputs.
///
/// A rising comparison (`x_k > e`, `x_k >= e`) of an ascending sequence is
/// False up to some position and True after it, so its first True is the
/// number of Falses, and an all-False sequence has argmax 0 (NumPy). The
/// sorted outputs hold the sort's elements, so the Falses are counted over
/// the elements (`a_j <= e` for `>`, `a_j < e` for `>=`) and the sort itself
/// is not needed. A falling comparison (`x_k < e`) is True up to some
/// position: its first True, when there is one, is position 0, and so is the
/// argmax of an all-False sequence.
fn first_true_of_sorted_comparisons(b: &mut IRBuilder, cells: &[Value]) -> Option<Value> {
    let n = cells.len();
    let comparisons: Vec<Comparison> = cells
        .iter()
        .map(|c| comparison_of(b, c))
        .collect::<Option<_>>()?;
    let first = &comparisons[0];
    let same_kind = comparisons.iter().all(|c| c.greater == first.greater && c.strict == first.strict);
    // The scalar is the operand shared by every cell; `x_k cmp e` is rising
    // for `>`/`>=`, and `e cmp x_k` for `<`/`<=`.
    let (elements, scalar, rising): (Vec<crate::types::StmtId>, Value, bool) =
        if n > 1 && comparisons.iter().all(|c| c.rhs_source == first.rhs_source) {
            (comparisons.iter().map(|c| c.lhs).collect(), first.rhs_value.clone(), first.greater)
        } else if n > 1 && comparisons.iter().all(|c| c.lhs_source == first.lhs_source) {
            (comparisons.iter().map(|c| c.rhs).collect(), first.lhs_value.clone(), !first.greater)
        } else {
            return None;
        };
    if !same_kind {
        return None;
    }
    let sort_elements = ascending_sort_outputs(b, &elements)?;
    if !rising {
        return Some(b.ir_constant_int(0));
    }
    // The Falses, counted over the sort's elements: `a_j <= e` for a strict
    // comparison, `a_j < e` otherwise.
    let falses: Vec<Value> = sort_elements
        .iter()
        .map(|a| {
            if first.strict {
                let above = less_than_by_type(b, &scalar, a);
                b.ir_logical_not(&above)
            } else {
                less_than_by_type(b, a, &scalar)
            }
        })
        .collect();
    let mut count = b.ir_constant_int(0);
    for f in &falses {
        let as_int = b.ir_int_cast(f);
        count = b.ir_add_i(&count, &as_int);
    }
    let length = b.ir_constant_int(n as i64);
    let any_true = b.ir_less_than_i(&count, &length);
    let zero = b.ir_constant_int(0);
    Some(b.ir_select_i(&any_true, &count, &zero))
}

/// `x < y` in the number type of the operands (float if either is).
fn less_than_by_type(b: &mut IRBuilder, x: &Value, y: &Value) -> Value {
    if matches!(x, Value::Float(_)) || matches!(y, Value::Float(_)) {
        b.ir_less_than_f(x, y)
    } else {
        b.ir_less_than_i(x, y)
    }
}

/// The elements of the sort whose outputs of ranks `0, 1, ..` are exactly
/// `stmts`, in that order.
fn ascending_sort_outputs(b: &IRBuilder, stmts: &[crate::types::StmtId]) -> Option<std::sync::Arc<[Value]>> {
    // A sorted-value lane at a constant rank: its elements and rank.
    let value_lane = |stmt| {
        let p = b.array_op_placeholder_at(stmt)?;
        let is_value = p.op.kind == crate::lower::OpKind::Sort && p.op.output == crate::lower::sort::VALUES;
        is_value.then(|| (p.shared.clone(), p.op.position))
    };
    let (elements, _) = value_lane(stmts[0])?;
    if elements.len() != stmts.len() {
        return None;
    }
    for (rank, &stmt) in stmts.iter().enumerate() {
        let (other, position) = value_lane(stmt)?;
        if position != Some(rank as u32) || !std::sync::Arc::ptr_eq(&other, &elements) {
            return None;
        }
    }
    Some(elements)
}

fn whole_array_argmax_argmin(b: &mut IRBuilder, arr: &Value, is_max: bool) -> Value {
    let cells = payload_cells(b, arr);
    if cells.is_empty() {
        return b.ir_constant_int(0);
    }
    let dtype = dtype_of(arr);
    let len = cells.len();
    // Boolean-aware fast path: argmax of `{0,1}` cells is "first index where
    // elem == 1". The generic body would emit `ir_greater_than_i` per step
    // (sign-aware range checks ~30 gates each); the bool form needs one mul
    // per step (NOT/AND/OR). Triggered by `(x==a)&(y==b)).argmax()` patterns.
    let best_idx = if is_max && cells.iter().all(|c| matches!(c, Value::Boolean(_))) {
        if let Some(first_true) = first_true_of_sorted_comparisons(b, &cells) {
            return first_true;
        }
        let mut idx = b.ir_constant_int(0);
        let mut found = cells[0].clone();
        for (i, elem) in cells.iter().enumerate().skip(1) {
            let not_found = b.ir_logical_not(&found);
            let take = b.ir_logical_and(&not_found, elem);
            let idx_val = b.ir_constant_int(i as i64);
            idx = b.ir_select_i(&take, &idx_val, &idx);
            found = b.ir_logical_or(&found, elem);
        }
        idx
    } else {
        let mut idx = b.ir_constant_int(0);
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
            idx = b.ir_select_i(&cond, &idx_val, &idx);
            best_val = if dtype == NumberType::Float {
                b.ir_select_f(&cond, elem, &best_val)
            } else {
                b.ir_select_i(&cond, elem, &best_val)
            };
        }
        idx
    };
    // Publish `0 <= best_idx < len` via the registered `dyn_argextremum`
    // contract so downstream `discharge_index_in_range(best_idx, 0, len)`
    // calls (e.g. `arr[arr.argmax()]`) prove the bound from facts instead
    // of emitting a u128 range assert. Without this, kernels like 0480
    // pay the ~2700-gate u128 lookup-table cost for an index that's
    // structurally bounded by the argmax loop.
    if let Some(idx_vid) = best_idx.value_id() {
        let len_arr_val = b.ir_constant_int(len as i64);
        if let Some(len_arr_vid) = len_arr_val.value_id() {
            let mut formals = std::collections::HashMap::new();
            formals.insert("len_arr".to_string(), len_arr_vid);
            b.fire_contract("dyn_argextremum", idx_vid, &formals);
        }
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
/// If the input `arr` is a StaticArray whose cells trace back to a
/// single input parameter (via `IR::ReadInteger` / `ReadFloat` with
/// `path.segments = [Index(_)]`), return that parameter's name.
/// Used to bind reduction outputs to `@requires(forall(<name>, ...))`
/// preconditions for element-bound propagation.
fn input_param_name_of_static_array(b: &IRBuilder, arr: &Value) -> Option<String> {
    use crate::circuit_input::PathSegment;
    use crate::ir_defs::IR;
    let seg = match arr {
        Value::StaticArray { segment_id, .. } => *segment_id,
        _ => return None,
    };
    let cached = b.static_array_payload.get(&seg)?;
    let first = cached.first()?;
    let stmt_id = first.value_id().and_then(|_| first.stmt_id())?;
    let stmt = b.stmts.get(stmt_id as usize)?;
    let path = match &stmt.ir {
        IR::ReadInteger { path, .. } | IR::ReadFloat { path, .. } => path,
        _ => return None,
    };
    if !matches!(path.segments.first(), Some(PathSegment::Index(_))) {
        return None;
    }
    Some(path.param.clone())
}

/// Propagate `forall(<input>, in_range/lt_bound/is_nonneg)` element
/// bounds to a reduction's scalar output, by depositing `Cmp` facts on
/// the output's `value_id`. Currently fires for `min`/`max` only —
/// those preserve element bounds exactly (output is one of the input
/// elements). Other reductions (`sum`, `mean`, etc.) need different
/// bound rules and are out of scope for this first cut.
/// Read the tightest known element bounds for an array `Value`, drawing
/// from BOTH (a) `@requires(forall(input_name, in_range/lt_bound/
/// is_nonneg))` structural-predicate atoms attached to the input name
/// the array traces back to, and (b) `PredicateApp { kind:
/// "array_in_range", args: [Var(Value(arr_vid)), lo, hi] }` facts
/// deposited on the array's `value_id` by other ops (e.g. the in-place
/// mutation contract, the boolean-mask-read pass-through).
///
/// Returns `(lo, hi)` if either source provides a bound; if both
/// provide bounds, the tightest intersection is returned.
pub(crate) fn read_array_element_bounds(b: &IRBuilder, arr: &Value) -> Option<(i64, i64)> {
    use crate::ir_defs::IR;
    use crate::optim::predicates::formula::{ContractTerm, ContractVar};

    let arr_vid = arr.value_id();
    let input_name = input_param_name_of_static_array(b, arr);

    let mut lo: Option<i64> = None;
    let mut hi: Option<i64> = None;

    // (a) Structural-predicate scan (works for original input arrays
    // whose payload cache is still populated).
    if let Some(ref name) = input_name {
        for s in &b.stmts {
            if let IR::StructuralPredicate { kind, args, .. } = &s.ir {
                if args.first().map(|s| s.as_str()) != Some(name.as_str()) {
                    continue;
                }
                match kind.as_str() {
                    "forall_in_range" if args.len() == 3 => {
                        if let (Ok(l), Ok(h)) = (args[1].parse::<i64>(), args[2].parse::<i64>()) {
                            lo = Some(lo.map_or(l, |x| x.max(l)));
                            hi = Some(hi.map_or(h, |x| x.min(h)));
                        }
                    }
                    "forall_lt_bound" if args.len() == 2 => {
                        if let Ok(h) = args[1].parse::<i64>() {
                            let new_hi = h - 1;
                            hi = Some(hi.map_or(new_hi, |x| x.min(new_hi)));
                        }
                    }
                    "forall_is_nonneg" if args.len() == 1 => {
                        lo = Some(lo.map_or(0, |x| x.max(0)));
                    }
                    _ => {}
                }
            }
        }
    }

    // (b) array_in_range fact scan keyed on the array's value_id.
    // Works for derived / in-place-mutated arrays whose origin can't
    // be traced back to a single input name.
    if let Some(vid) = arr_vid {
        for fact in b.facts.visible_facts() {
            if let ContractTerm::PredicateApp { kind, args } = fact {
                if kind != "array_in_range" || args.len() != 3 {
                    continue;
                }
                let matches_vid = matches!(
                    &args[0],
                    ContractTerm::Var(ContractVar::Value(v)) if *v == vid
                );
                if !matches_vid {
                    continue;
                }
                if let (ContractTerm::LitInt(l), ContractTerm::LitInt(h)) = (&args[1], &args[2]) {
                    lo = Some(lo.map_or(*l, |x| x.max(*l)));
                    hi = Some(hi.map_or(*h, |x| x.min(*h)));
                }
            }
        }
    }

    match (lo, hi) {
        (Some(l), Some(h)) if l <= h => Some((l, h)),
        _ => None,
    }
}

/// Deposit an `array_in_range(arr_vid, lo, hi)` fact on `arr_vid`'s
/// FactStack entry. Idempotent — duplicate facts are filtered out by
/// `prove` consumers via the connectivity fast-path's set semantics.
pub(crate) fn deposit_array_in_range_fact(
    b: &mut IRBuilder,
    arr_vid: crate::types::ValueId,
    lo: i64,
    hi: i64,
) {
    use crate::optim::predicates::formula::{ContractTerm, ContractVar};
    b.facts.insert_for(
        arr_vid,
        ContractTerm::PredicateApp {
            kind: "array_in_range".to_string(),
            args: vec![
                ContractTerm::Var(ContractVar::Value(arr_vid)),
                ContractTerm::LitInt(lo),
                ContractTerm::LitInt(hi),
            ],
        },
    );
}

fn propagate_forall_bounds_to_reduction_output(
    b: &mut IRBuilder,
    op: &str,
    input: &Value,
    output: &Value,
) {
    use crate::optim::predicates::formula::{CmpOp, ContractTerm, ContractVar};
    if !matches!(op, "min" | "max") {
        return;
    }
    let out_vid = match output.value_id() {
        Some(v) => v,
        None => return,
    };
    let (lo, hi) = match read_array_element_bounds(b, input) {
        Some((l, h)) => (Some(l), Some(h)),
        None => return,
    };
    if let Some(l) = lo {
        b.facts.insert_for(
            out_vid,
            ContractTerm::Cmp {
                op: CmpOp::Ge,
                lhs: Box::new(ContractTerm::Var(ContractVar::Value(out_vid))),
                rhs: Box::new(ContractTerm::LitInt(l)),
            },
        );
    }
    if let Some(h) = hi {
        b.facts.insert_for(
            out_vid,
            ContractTerm::Cmp {
                op: CmpOp::Le,
                lhs: Box::new(ContractTerm::Var(ContractVar::Value(out_vid))),
                rhs: Box::new(ContractTerm::LitInt(h)),
            },
        );
    }
}

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
    if matches!(dtype_of(val), NumberType::Complex) {
        // Reject reductions that aren't well-defined on Complex.
        if matches!(op, "min" | "max" | "any" | "all") {
            panic!(
                "Reduction `{}` is not defined on Complex (numpy raises TypeError)",
                op
            );
        }
        let shape = shape_of(val);
        let axis_spec = parse_axis_spec(axis_arg, shape.len())?;
        if let Some(axes) = axis_spec {
            if axes.len() != 1 {
                return None;
            }
            if shape.is_empty() {
                return Some(whole_array_reduce(b, op, val));
            }
            let axis = axes[0];
            if op == "mean" {
                return Some(axis_mean_complex(b, val, axis, keepdims));
            }
            let bin_op = if op == "sum" { "add" } else { "mul" };
            return Some(axis_reduce_complex(b, bin_op, val, axis, keepdims));
        }
        // Whole-array sum / prod / mean: walk Complex cells, fold with the
        // existing component-wise complex IR.
        let cells = payload_cells(b, val);
        if op == "mean" {
            let n: usize = shape_of(val).iter().product::<usize>().max(1);
            let s = complex_reduce_cells(b, "add", &cells);
            // Divide both components by n (real division).
            let n_val = b.ir_constant_float(n as f64);
            let (sr, si) = crate::helpers::value_ops::unpack_value_to_complex_parts(b, &s);
            let mr = b.ir_div_f(&sr, &n_val);
            let mi = b.ir_div_f(&si, &n_val);
            return Some(crate::helpers::value_ops::pack_complex_value(mr, mi));
        }
        let bin_op = if op == "sum" { "add" } else { "mul" };
        return Some(complex_reduce_cells(b, bin_op, &cells));
    }

    let shape = shape_of(val);
    let axis_spec = parse_axis_spec(axis_arg, shape.len())?;
    if let Some(axes) = axis_spec {
        if shape.is_empty() {
            // 0-rank: ignore axis (numpy errors, but match legacy lenience).
            return Some(whole_array_reduce(b, op, val));
        }
        if op == "mean" {
            return Some(multi_axis_mean(b, val, &axes, keepdims));
        }
        return Some(multi_axis_reduce(b, op, val, &axes, keepdims));
    }

    // Whole-array.
    if op == "mean" {
        return Some(whole_array_mean(b, val));
    }
    let out = whole_array_reduce(b, op, val);
    // Op-contract enrichment: for min/max, propagate `forall(input,
    // in_range/lt_bound/is_nonneg)` element bounds from the input's
    // @requires precondition onto the output scalar. This is what
    // lets `np.zeros(np.amax(x)+1)` compile when the user annotates
    // `@requires(forall(x, in_range(0, K)))`. See
    // `compiler.op-contract-forall-to-reduction-output`.
    propagate_forall_bounds_to_reduction_output(b, op, val, &out);
    Some(out)
}

/// Try to apply `np.var` / `np.std` natively on a `Value::StaticArray`.
/// `ddof` must be statically known; every output cell uses the reduced
/// fiber length, not the flattened input length, for its divisor contract.
pub fn try_apply_var_std(
    b: &mut IRBuilder,
    op: &str,
    val: &Value,
    axis_arg: Option<&Value>,
    ddof_arg: Option<&Value>,
    keepdims: bool,
) -> Option<Value> {
    if !matches!(val, Value::StaticArray { .. }) {
        return None;
    }
    if !matches!(op, "var" | "std") {
        return None;
    }
    if matches!(dtype_of(val), NumberType::Complex) {
        panic!("{}: Complex StaticArray input is deferred", op);
    }

    let shape = shape_of(val);
    let axis_spec = parse_axis_spec(axis_arg, shape.len()).unwrap_or_else(|| {
        panic!(
            "{}: axis must be a static integer or tuple/list of static integers",
            op
        )
    });
    let ddof = parse_static_ddof(ddof_arg);
    let axes = axis_spec.unwrap_or_else(|| (0..shape.len()).collect());
    let reduced_len = axes.iter().map(|&a| shape[a]).product::<usize>().max(1);
    checked_var_divisor(reduced_len, ddof);
    Some(multi_axis_var_std(b, op, val, &axes, keepdims, ddof))
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
        panic!(
            "argmax / argmin is not defined on Complex (no canonical ordering; numpy raises TypeError)"
        );
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
    use crate::ir_defs::IR;
    use crate::types::{CompositeData, ValueId};

    fn list_of(values: Vec<Value>) -> Value {
        let types = values.iter().map(|v| v.zinnia_type()).collect();
        Value::List(CompositeData {
            elements_type: types,
            values,

            value_id: ValueId::next(),
        })
    }

    fn make_1d_int(b: &mut IRBuilder, vals: &[i64]) -> Value {
        let leaves: Vec<Value> = vals.iter().map(|n| b.ir_constant_int(*n)).collect();
        let lst = list_of(leaves);
        super::super::base::to_static_array(b, &lst).expect("StaticArray")
    }

    fn make_2d_int(b: &mut IRBuilder, rows: &[&[i64]]) -> Value {
        let row_lists: Vec<Value> = rows
            .iter()
            .map(|r| list_of(r.iter().map(|n| b.ir_constant_int(*n)).collect()))
            .collect();
        let lst = list_of(row_lists);
        super::super::base::to_static_array(b, &lst).expect("StaticArray")
    }

    fn make_2d_complex(b: &mut IRBuilder, rows: &[&[(f64, f64)]]) -> Value {
        let row_lists: Vec<Value> = rows
            .iter()
            .map(|r| {
                list_of(
                    r.iter()
                        .map(|(re, im)| {
                            crate::helpers::value_ops::pack_complex_value(
                                b.ir_constant_float(*re),
                                b.ir_constant_float(*im),
                            )
                        })
                        .collect(),
                )
            })
            .collect();
        let lst = list_of(row_lists);
        super::super::base::to_static_array(b, &lst).expect("StaticArray")
    }

    fn make_4d_float(b: &mut IRBuilder, shape: [usize; 4]) -> Value {
        let total: usize = shape.iter().product();
        let flat: Vec<Value> = (0..total)
            .map(|i| b.ir_constant_float((i + 1) as f64))
            .collect();
        build_static_array_from_flat(b, flat, shape.to_vec(), NumberType::Float)
    }

    fn tuple_of(values: Vec<Value>) -> Value {
        let types = values.iter().map(|v| v.zinnia_type()).collect();
        Value::Tuple(CompositeData {
            elements_type: types,
            values,
            value_id: ValueId::next(),
        })
    }

    fn complex_static_vals(b: &mut IRBuilder, val: &Value) -> Vec<(Option<f64>, Option<f64>)> {
        match val {
            Value::StaticArray {
                dtype,
                imag_segment_id,
                ..
            } => {
                assert_eq!(*dtype, NumberType::Complex);
                assert!(imag_segment_id.is_some());
            }
            _ => panic!("expected Complex StaticArray"),
        }
        payload_cells(b, val)
            .iter()
            .map(|c| match c {
                Value::Complex { real, imag } => (real.static_val, imag.static_val),
                _ => panic!("expected Complex cell"),
            })
            .collect()
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
    fn multi_axis_float_sum_reduces_tuple_axes_without_axis_shift() {
        let mut b = IRBuilder::new();
        let a = make_4d_float(&mut b, [2, 2, 3, 4]);
        let axis = tuple_of(vec![
            b.ir_constant_int(1),
            b.ir_constant_int(2),
            b.ir_constant_int(3),
        ]);
        let out = try_apply_reduce(&mut b, "sum", &a, Some(&axis), false).expect("native");
        match &out {
            Value::StaticArray { shape, dtype, .. } => {
                assert_eq!(*shape, vec![2]);
                assert_eq!(*dtype, NumberType::Float);
            }
            _ => panic!("expected StaticArray"),
        }
        let vals: Vec<Option<f64>> = payload_cells(&mut b, &out)
            .iter()
            .map(|c| c.float_val())
            .collect();
        assert_eq!(vals, vec![Some(300.0), Some(876.0)]);
    }

    #[test]
    fn multi_axis_float_sum_keepdims_preserves_original_rank() {
        let mut b = IRBuilder::new();
        let a = make_4d_float(&mut b, [2, 2, 3, 4]);
        let axis = tuple_of(vec![b.ir_constant_int(1), b.ir_constant_int(-1)]);
        let out = try_apply_reduce(&mut b, "sum", &a, Some(&axis), true).expect("native");
        match &out {
            Value::StaticArray { shape, dtype, .. } => {
                assert_eq!(*shape, vec![2, 1, 3, 1]);
                assert_eq!(*dtype, NumberType::Float);
            }
            _ => panic!("expected StaticArray"),
        }
        let vals: Vec<Option<f64>> = payload_cells(&mut b, &out)
            .iter()
            .map(|c| c.float_val())
            .collect();
        assert_eq!(
            vals,
            vec![
                Some(68.0),
                Some(100.0),
                Some(132.0),
                Some(260.0),
                Some(292.0),
                Some(324.0)
            ]
        );
    }

    #[test]
    fn multi_axis_sum_consumes_product_provenance_without_materializing_product_payload() {
        let mut b = IRBuilder::new();
        let a = make_4d_float(&mut b, [1, 2, 2, 2]);
        let weights = make_4d_float(&mut b, [1, 2, 2, 2]);
        let product = crate::helpers::static_array::elementwise::try_apply_binary_op(
            &mut b, "mul", &a, &weights,
        )
        .expect("native");
        let product_segment = match &product {
            Value::StaticArray { segment_id, .. } => segment_id,
            _ => panic!("expected StaticArray"),
        };
        assert!(b
            .static_array_product_provenance
            .contains_key(&product_segment));
        let axis = tuple_of(vec![
            b.ir_constant_int(1),
            b.ir_constant_int(2),
            b.ir_constant_int(3),
        ]);
        let out = try_apply_reduce(&mut b, "sum", &product, Some(&axis), false).expect("native");
        assert!(b
            .static_array_product_provenance
            .contains_key(&product_segment));
        assert!(b.static_array_payload.get(&product_segment).is_none());
        assert!(!b.stmts.iter().any(|stmt| matches!(stmt.ir, IR::MulF)));
        let product_sum_stmt = b
            .stmts
            .iter()
            .find(|stmt| matches!(stmt.ir, IR::ProductSumF { .. }))
            .expect("expected ProductSumF for Float product-provenance sum");
        assert_eq!(product_sum_stmt.ir, IR::ProductSumF { len: 8 });
        assert_eq!(product_sum_stmt.arguments.len(), 16);
        let arg_values: Vec<f64> = product_sum_stmt
            .arguments
            .iter()
            .map(|arg| match b.stmts[*arg as usize].ir {
                IR::ConstantFloat { value } => value,
                ref other => panic!("expected ConstantFloat operand, got {other:?}"),
            })
            .collect();
        assert_eq!(
            arg_values,
            vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 6.0, 7.0, 7.0, 8.0, 8.0]
        );
        match &out {
            Value::StaticArray { shape, dtype, .. } => {
                assert_eq!(*shape, vec![1]);
                assert_eq!(*dtype, NumberType::Float);
            }
            _ => panic!("expected StaticArray"),
        }
        let vals: Vec<Option<f64>> = payload_cells(&mut b, &out)
            .iter()
            .map(|c| c.float_val())
            .collect();
        assert_eq!(vals, vec![None]);
    }

    #[test]
    fn product_provenance_snapshots_operand_cells_before_later_write() {
        let mut b = IRBuilder::new();
        let a = make_2d_int(&mut b, &[&[1, 2], &[3, 4]]);
        let weights = make_2d_int(&mut b, &[&[10, 20], &[30, 40]]);
        let product = crate::helpers::static_array::elementwise::try_apply_binary_op(
            &mut b, "mul", &a, &weights,
        )
        .expect("native");
        let idx = vec![
            crate::types::SliceIndex::Single(b.ir_constant_int(0)),
            crate::types::SliceIndex::Single(b.ir_constant_int(0)),
        ];
        let replacement = b.ir_constant_int(99);
        crate::helpers::static_array::write::static_array_setitem(&mut b, &a, &idx, &replacement);

        let axis = tuple_of(vec![b.ir_constant_int(0), b.ir_constant_int(1)]);
        let out = try_apply_reduce(&mut b, "sum", &product, Some(&axis), false).expect("native");
        assert!(!b
            .stmts
            .iter()
            .any(|stmt| matches!(stmt.ir, IR::ProductSumF { .. })));
        let vals: Vec<Option<i64>> = payload_cells(&mut b, &out)
            .iter()
            .map(|c| c.int_val())
            .collect();
        assert_eq!(vals, vec![Some(300)]);
    }

    #[test]
    fn fused_sum_does_not_consume_product_for_later_force() {
        let mut b = IRBuilder::new();
        let a = make_2d_int(&mut b, &[&[1, 2], &[3, 4]]);
        let weights = make_2d_int(&mut b, &[&[10, 20], &[30, 40]]);
        let product = crate::helpers::static_array::elementwise::try_apply_binary_op(
            &mut b, "mul", &a, &weights,
        )
        .expect("native");
        let axis = tuple_of(vec![b.ir_constant_int(0), b.ir_constant_int(1)]);
        let _sum = try_apply_reduce(&mut b, "sum", &product, Some(&axis), false).expect("native");

        let product_cells = payload_cells(&mut b, &product);
        let vals: Vec<Option<i64>> = product_cells.iter().map(|c| c.int_val()).collect();
        assert_eq!(vals, vec![Some(10), Some(40), Some(90), Some(160)]);
    }

    #[test]
    fn newaxis_static_array_view_feeds_product_provenance() {
        let mut b = IRBuilder::new();
        let a = make_2d_int(&mut b, &[&[1, 2], &[3, 4]]);
        let weights = make_2d_int(&mut b, &[&[10, 20], &[30, 40]]);
        let new_axis = crate::types::SliceIndex::NewAxis;
        let full = crate::types::SliceIndex::Range(None, None, None);
        let a_view = crate::helpers::static_array::read::static_array_subscript(
            &mut b,
            &a,
            &[full.clone(), full.clone(), new_axis.clone()],
        );
        let w_view = crate::helpers::static_array::read::static_array_subscript(
            &mut b,
            &weights,
            &[new_axis, full.clone(), full],
        );
        match &a_view {
            Value::StaticArray { shape, strides, .. } => {
                assert_eq!(*shape, vec![2, 2, 1]);
                assert_eq!(*strides, vec![2, 1, 0]);
            }
            _ => panic!("expected StaticArray view"),
        }
        let product = crate::helpers::static_array::elementwise::try_apply_binary_op(
            &mut b, "mul", &a_view, &w_view,
        )
        .expect("native");
        match &product {
            Value::StaticArray {
                segment_id, shape, ..
            } => {
                assert_eq!(*shape, vec![2, 2, 2]);
                assert!(b.static_array_product_provenance.contains_key(segment_id));
            }
            _ => panic!("expected StaticArray product"),
        }
    }

    #[test]
    fn multi_axis_reduce_fails_open_for_duplicate_or_dynamic_axis() {
        let mut b = IRBuilder::new();
        let a = make_4d_float(&mut b, [2, 2, 3, 4]);
        let dup_axis = tuple_of(vec![b.ir_constant_int(1), b.ir_constant_int(-3)]);
        assert!(try_apply_reduce(&mut b, "sum", &a, Some(&dup_axis), false).is_none());

        let dyn_axis = Value::Integer(crate::types::ScalarValue::new(None, None));
        let dynamic_axis = tuple_of(vec![b.ir_constant_int(1), dyn_axis]);
        assert!(try_apply_reduce(&mut b, "sum", &a, Some(&dynamic_axis), false).is_none());
    }

    #[test]
    fn complex_axis_sum_uses_native_static_array_path() {
        let mut b = IRBuilder::new();
        let a = make_2d_complex(
            &mut b,
            &[&[(1.0, 2.0), (3.0, 4.0)], &[(5.0, 6.0), (7.0, 8.0)]],
        );
        let axis = b.ir_constant_int(0);
        let out = try_apply_reduce(&mut b, "sum", &a, Some(&axis), false).expect("native");
        match &out {
            Value::StaticArray { shape, .. } => assert_eq!(*shape, vec![2]),
            _ => panic!("expected StaticArray"),
        }
        let cells = payload_cells(&mut b, &out);
        let vals: Vec<(Option<f64>, Option<f64>)> = cells
            .iter()
            .map(|c| match c {
                Value::Complex { real, imag } => (real.static_val, imag.static_val),
                _ => panic!("expected Complex cell"),
            })
            .collect();
        assert_eq!(vals, vec![(Some(6.0), Some(8.0)), (Some(10.0), Some(12.0))]);
    }

    #[test]
    fn complex_axis_mean_keepdims_uses_native_static_array_path() {
        let mut b = IRBuilder::new();
        let a = make_2d_complex(
            &mut b,
            &[&[(1.0, 2.0), (3.0, 4.0)], &[(5.0, 6.0), (7.0, 8.0)]],
        );
        let axis = b.ir_constant_int(1);
        let out = try_apply_reduce(&mut b, "mean", &a, Some(&axis), true).expect("native");
        match &out {
            Value::StaticArray { shape, .. } => assert_eq!(*shape, vec![2, 1]),
            _ => panic!("expected StaticArray"),
        }
        assert_eq!(
            complex_static_vals(&mut b, &out),
            vec![(Some(2.0), Some(3.0)), (Some(6.0), Some(7.0))]
        );
    }

    #[test]
    fn returns_none_for_non_static_array() {
        let mut b = IRBuilder::new();
        let lst = list_of(vec![b.ir_constant_int(1), b.ir_constant_int(2)]);
        assert!(try_apply_reduce(&mut b, "sum", &lst, None, false).is_none());
    }
}
