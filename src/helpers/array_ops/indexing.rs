//! Dynamic array indexing and slicing: element access, range slicing,
//! and fancy indexing.
//!
//! Sibling modules:
//! * [`super::bounded_axis`] — stride-layout dispatch + strict-mode env var.
//! * [`super::boolean_mask`] — boolean-mask classification predicate.

use crate::builder::IRBuilder;
use crate::helpers::shape_arith::row_major_strides;
use crate::types::{ValueId,
    DynArrayMeta, DynamicNDArrayData, NumberType, ScalarValue, SliceIndex, Value,
};

use super::bounded_axis::{bounded_axis_strict, select_stride_mode, stride_value, StrideMode};
use super::boolean_mask::is_boolean_mask;
use super::memory::filter;

// ── Main dispatch ───────────────────────────────────────────────────────

/// Subscript a DynamicNDArray with a list of SliceIndex values.
pub fn dyn_subscript(
    b: &mut IRBuilder,
    data: &DynamicNDArrayData,
    indices: &[SliceIndex],
) -> Value {
    // Single-index special cases: boolean mask, fancy indexing.
    if indices.len() == 1 {
        if let SliceIndex::Single(idx_val) = &indices[0] {
            if is_boolean_mask(idx_val) {
                return filter(b, &Value::DynamicNDArray(data.clone()), &[idx_val.clone()]);
            }
            if is_fancy_index(idx_val) {
                return dyn_fancy_index(b, data, idx_val);
            }
        }
    }

    // Multi-dim fancy: dyn[[r0,r1], [c0,c1]].
    if indices.len() >= 2 {
        let all_fancy = indices.iter().all(|s| match s {
            SliceIndex::Single(v) => is_fancy_index(v),
            _ => false,
        });
        if all_fancy {
            let idx_arrays: Vec<&Value> = indices.iter().map(|s| match s {
                SliceIndex::Single(v) => v,
                _ => unreachable!(),
            }).collect();
            return dyn_fancy_index_multidim(b, data, &idx_arrays);
        }
    }

    let all_single = indices.iter().all(|s| matches!(s, SliceIndex::Single(_)));

    if all_single && indices.len() == data.envelope.rank() {
        dyn_getitem_element(b, data, indices)
    } else if all_single && indices.len() == 1 {
        dyn_getitem_row(b, data, indices)
    } else if indices.len() == 1 {
        if let SliceIndex::Range(start, stop, step) = &indices[0] {
            dyn_slice_1d(b, data, start.as_ref(), stop.as_ref(), step.as_ref())
        } else {
            panic!("dyn_subscript: unsupported single index type")
        }
    } else {
        dyn_subscript_multidim(b, data, indices)
    }
}

// ── Classification ──────────────────────────────────────────────────────

fn is_fancy_index(val: &Value) -> bool {
    match val {
        Value::List(d) | Value::Tuple(d) => {
            !d.values.is_empty()
                && d.values.iter().all(|v| matches!(v, Value::Integer(_)))
        }
        _ => false,
    }
}

fn extract_index_values(val: &Value) -> Vec<Value> {
    match val {
        Value::List(d) | Value::Tuple(d) => d.values.clone(),
        _ => panic!("extract_index_values: expected List/Tuple"),
    }
}

// ── Element access ──────────────────────────────────────────────────────

fn dyn_getitem_element(
    b: &mut IRBuilder,
    data: &DynamicNDArrayData,
    indices: &[SliceIndex],
) -> Value {
    let addr = compute_flat_addr(b, data, indices);
    // Load-bearing index-in-range discharge (Group 5a). Replaces the
    // informational `probe_in_range` with Phase E enforcement at the
    // dyn-ndarray subscript chokepoint.
    crate::optim::resolver::discharge_index_in_range(
        b,
        &addr,
        0,
        data.envelope.total_bound as i64,
        "dyn_getitem",
    );
    let raw = b.ir_read_memory(data.segment_id, &addr);
    match data.dtype {
        NumberType::Float => Value::Float(ScalarValue::new(
            raw.int_val().map(|v| v as f64),
            raw.stmt_id(),
        )),
        NumberType::Integer => raw,
        NumberType::Complex => panic!(
            "DynamicNDArray of Complex is not yet supported (compiler.complex-ndarray-ops scope)"
        ),
    }
}

fn dyn_getitem_row(
    b: &mut IRBuilder,
    data: &DynamicNDArrayData,
    indices: &[SliceIndex],
) -> Value {
    let rank = data.envelope.rank();
    if rank == 1 {
        return dyn_getitem_element(b, data, indices);
    }

    let idx_val = match &indices[0] {
        SliceIndex::Single(v) => v,
        _ => unreachable!(),
    };

    let out_shape: Vec<usize> = data.meta.logical_shape[1..].to_vec();
    let out_total: usize = out_shape.iter().product();
    let out_strides = row_major_strides(&out_shape);

    let mode = select_stride_mode(data);
    let base = match &mode {
        StrideMode::LiteralLogical(strides) => {
            let row_stride = strides[0];
            if let Some(i) = idx_val.int_val() {
                let i = if i < 0 { data.meta.logical_shape[0] as i64 + i } else { i };
                b.ir_constant_int(i * row_stride as i64)
            } else {
                let stride_val = b.ir_constant_int(row_stride as i64);
                b.ir_mul_i(idx_val, &stride_val)
            }
        }
        StrideMode::SymbolicRuntime(_) => {
            let idx_v = if let Some(i) = idx_val.int_val() {
                let i = if i < 0 { data.meta.logical_shape[0] as i64 + i } else { i };
                b.ir_constant_int(i)
            } else {
                idx_val.clone()
            };
            let stride_val = stride_value(b, &mode, 0);
            b.ir_mul_i(&idx_v, &stride_val)
        }
    };

    let mut out_elements = Vec::with_capacity(out_total);
    for j in 0..out_total {
        let offset = b.ir_constant_int(j as i64);
        let addr = b.ir_add_i(&base, &offset);
        let elem = b.ir_read_memory(data.segment_id, &addr);
        out_elements.push(crate::ops::dyn_ndarray::value_to_scalar_i64(&elem));
    }

    let segment_id = crate::helpers::segment::alloc_and_write(b, &out_elements, data.dtype);
    let out_dims: Vec<crate::types::Dim> = data.envelope.dims[1..].to_vec();
    let envelope = crate::types::Envelope::new_with_bound(out_dims, out_total.min(data.envelope.total_bound));

    Value::DynamicNDArray(DynamicNDArrayData {
        envelope,
        dtype: data.dtype,
        segment_id,
        meta: DynArrayMeta {
            logical_shape: out_shape.clone(),
            logical_offset: 0,
            logical_strides: out_strides.clone(),
            runtime_length: ScalarValue::new(Some(out_total as i64), None),
            runtime_rank: ScalarValue::new(Some(out_shape.len() as i64), None),
            runtime_shape: out_shape.iter().map(|&s| ScalarValue::new(Some(s as i64), None)).collect(),
            runtime_strides: out_strides.iter().map(|&s| ScalarValue::new(Some(s as i64), None)).collect(),
            runtime_offset: ScalarValue::new(Some(0), None),
        },
        value_id: ValueId::next(),
    })
}

// ── Range slicing ───────────────────────────────────────────────────────

fn dyn_slice_1d(
    b: &mut IRBuilder,
    data: &DynamicNDArrayData,
    start: Option<&Value>,
    stop: Option<&Value>,
    step: Option<&Value>,
) -> Value {
    let len = data.meta.logical_shape[0];

    let is_present = |v: Option<&Value>| -> bool {
        matches!(v, Some(val) if !matches!(val, Value::None))
    };
    let static_val = |v: Option<&Value>| -> Option<i64> {
        v.and_then(|val| if matches!(val, Value::None) { None } else { val.int_val() })
    };

    let start_static = static_val(start);
    let stop_static = static_val(stop);
    let step_static = static_val(step);

    let all_static = (!is_present(start) || start_static.is_some())
        && (!is_present(stop) || stop_static.is_some())
        && (!is_present(step) || step_static.is_some());

    if all_static {
        return dyn_slice_1d_static(b, data, start_static, stop_static, step_static);
    }

    // Load-bearing slice-bound discharge (compiler.fuzz-finding-v2-slice-oob-witness-miss).
    // The pad-and-mask path below silently swallows OOB `start` / `stop` —
    // without an explicit discharge, an OOB witness slips through and
    // returns `satisfied = True`. Mirrors the scalar dyn-getitem path's
    // `discharge_index_in_range` call (Group 5a).
    //
    // Bound is `envelope.total_bound`, not `logical_shape[0]`, because a
    // reshape with runtime dims over-approximates `logical_shape` (e.g.
    // a `(2,4) -> (a,b)` reshape produces `logical_shape = [8, 1]` even
    // though a valid runtime shape is `(2,4)`). `total_bound` is the
    // max possible storage extent and matches the scalar dyn-getitem path.
    let bound = data.envelope.total_bound;
    crate::optim::resolver::discharge_slice_bound(b, start, bound, "dyn_slice_start");
    crate::optim::resolver::discharge_slice_bound(b, stop, bound, "dyn_slice_stop");

    // Dynamic path: pad-and-mask.
    let max_out_len = len;

    fn val_to_ir(b: &mut IRBuilder, v: Option<&Value>, default: i64) -> Value {
        match v {
            Some(val) if !matches!(val, Value::None) => {
                if let Some(s) = val.int_val() { b.ir_constant_int(s) } else { val.clone() }
            }
            _ => b.ir_constant_int(default),
        }
    }
    let start_ir = val_to_ir(b, start, 0);
    let stop_ir = val_to_ir(b, stop, len as i64);
    let step_ir = val_to_ir(b, step, 1);

    let default_val = crate::ops::dyn_ndarray::metadata::dyn_default_value(b, data.dtype);
    let zero = b.ir_constant_int(0);
    let len_val = b.ir_constant_int(len as i64);
    let mut out_elements = Vec::with_capacity(max_out_len);

    for i in 0..max_out_len {
        let i_const = b.ir_constant_int(i as i64);
        let offset = b.ir_mul_i(&i_const, &step_ir);
        let src_idx = b.ir_add_i(&start_ir, &offset);

        let ge_zero = b.ir_greater_than_or_equal_i(&src_idx, &zero);
        let lt_len = b.ir_less_than_i(&src_idx, &len_val);
        let in_range = b.ir_logical_and(&ge_zero, &lt_len);

        let step_pos = b.ir_greater_than_i(&step_ir, &zero);
        let lt_stop = b.ir_less_than_i(&src_idx, &stop_ir);
        let gt_stop = b.ir_greater_than_i(&src_idx, &stop_ir);
        let stop_ok = b.ir_select_i(&step_pos, &lt_stop, &gt_stop);
        let stop_bool = b.ir_bool_cast(&stop_ok);
        let in_bounds = b.ir_logical_and(&in_range, &stop_bool);

        let max_idx = b.ir_constant_int(len as i64 - 1);
        let is_neg = b.ir_less_than_i(&src_idx, &zero);
        let is_over = b.ir_greater_than_i(&src_idx, &max_idx);
        let clamped_hi = b.ir_select_i(&is_over, &max_idx, &src_idx);
        let clamped = b.ir_select_i(&is_neg, &zero, &clamped_hi);

        let elem = b.ir_read_memory(data.segment_id, &clamped);
        let masked = if data.dtype == NumberType::Float {
            b.ir_select_f(&in_bounds, &elem, &default_val)
        } else {
            b.ir_select_i(&in_bounds, &elem, &default_val)
        };
        out_elements.push(crate::ops::dyn_ndarray::value_to_scalar_i64(&masked));
    }

    // runtime_length
    let step_pos_2 = b.ir_greater_than_i(&step_ir, &zero);
    let stop_minus_start = b.ir_sub_i(&stop_ir, &start_ir);
    let start_minus_stop = b.ir_sub_i(&start_ir, &stop_ir);
    let diff = b.ir_select_i(&step_pos_2, &stop_minus_start, &start_minus_stop);
    let neg_step = b.ir_sub_i(&zero, &step_ir);
    let abs_step = b.ir_select_i(&step_pos_2, &step_ir, &neg_step);
    let diff_pos = b.ir_greater_than_i(&diff, &zero);
    let one = b.ir_constant_int(1);
    let abs_step_m1 = b.ir_sub_i(&abs_step, &one);
    let numerator = b.ir_add_i(&diff, &abs_step_m1);
    let divided = b.ir_div_i(&numerator, &abs_step);
    let runtime_length = b.ir_select_i(&diff_pos, &divided, &zero);

    let segment_id = crate::helpers::segment::alloc_and_write(b, &out_elements, data.dtype);
    let runtime_len_sv = crate::ops::dyn_ndarray::value_to_scalar_i64(&runtime_length);
    let envelope = crate::types::Envelope::new_with_bound(
        vec![crate::types::Dim::new_dynamic(&mut b.dim_table, 0, max_out_len)],
        max_out_len,
    );

    Value::DynamicNDArray(DynamicNDArrayData {
        envelope, dtype: data.dtype, segment_id,
        meta: DynArrayMeta {
            logical_shape: vec![max_out_len], logical_offset: 0, logical_strides: vec![1],
            runtime_length: runtime_len_sv.clone(),
            runtime_rank: ScalarValue::new(Some(1), None),
            runtime_shape: vec![runtime_len_sv],
            runtime_strides: vec![ScalarValue::new(Some(1), None)],
            runtime_offset: ScalarValue::new(Some(0), None),
        },
        value_id: ValueId::next(),
    })
}

fn dyn_slice_1d_static(
    b: &mut IRBuilder,
    data: &DynamicNDArrayData,
    start: Option<i64>, stop: Option<i64>, step: Option<i64>,
) -> Value {
    let len = data.meta.logical_shape[0] as i64;
    let start_i = start.unwrap_or(0);
    let stop_i = stop.unwrap_or(len);
    let step_i = step.unwrap_or(1);
    let start_i = if start_i < 0 { (len + start_i).max(0) } else { start_i.min(len) };
    let stop_i = if stop_i < 0 { (len + stop_i).max(0) } else { stop_i.min(len) };
    assert!(step_i != 0, "slice step cannot be zero");

    let mut src_indices: Vec<i64> = Vec::new();
    if step_i > 0 {
        let mut i = start_i;
        while i < stop_i { src_indices.push(i); i += step_i; }
    } else {
        let mut i = start_i;
        while i > stop_i { src_indices.push(i); i += step_i; }
    }

    let out_len = src_indices.len();
    let mut out_elements = Vec::with_capacity(out_len);
    for &src_i in &src_indices {
        let addr = b.ir_constant_int(src_i);
        let elem = b.ir_read_memory(data.segment_id, &addr);
        out_elements.push(crate::ops::dyn_ndarray::value_to_scalar_i64(&elem));
    }

    let segment_id = crate::helpers::segment::alloc_and_write(b, &out_elements, data.dtype);
    let envelope = crate::types::Envelope::from_static_shape(&mut b.dim_table, &[out_len]);

    Value::DynamicNDArray(DynamicNDArrayData {
        envelope, dtype: data.dtype, segment_id,
        meta: DynArrayMeta {
            logical_shape: vec![out_len], logical_offset: 0, logical_strides: vec![1],
            runtime_length: ScalarValue::new(Some(out_len as i64), None),
            runtime_rank: ScalarValue::new(Some(1), None),
            runtime_shape: vec![ScalarValue::new(Some(out_len as i64), None)],
            runtime_strides: vec![ScalarValue::new(Some(1), None)],
            runtime_offset: ScalarValue::new(Some(0), None),
        },
        value_id: ValueId::next(),
    })
}

// ── Multi-dim subscript ─────────────────────────────────────────────────

/// Multi-dim subscript with mixed Single/Range indices.
///
/// Strategy: check if any Range has dynamic bounds. If so, apply the
/// dynamic range on its axis first (via dyn_slice_axis0 for axis 0, or
/// transpose trick for other axes), producing an intermediate with that
/// axis sliced. Then recurse with the remaining indices (now all static).
///
/// If all ranges are static, use direct coordinate-based reads.
fn dyn_subscript_multidim(
    b: &mut IRBuilder, data: &DynamicNDArrayData, indices: &[SliceIndex],
) -> Value {
    // Check if any Range has dynamic bounds.
    let first_dynamic_range = indices.iter().enumerate().find(|(_, idx)| {
        if let SliceIndex::Range(start, stop, step) = idx {
            let resolve = |v: &Option<Value>| -> bool {
                match v.as_ref() {
                    Some(Value::None) | None => false,
                    Some(val) => val.int_val().is_none(),
                }
            };
            resolve(start) || resolve(stop) || resolve(step)
        } else {
            false
        }
    });

    if let Some((dyn_ax, _)) = first_dynamic_range {
        // Dynamic range at axis `dyn_ax`. Process it first.
        return dyn_subscript_with_dynamic_axis(b, data, indices, dyn_ax);
    }

    // All ranges are static — use direct coordinate-based reads.
    dyn_subscript_multidim_static(b, data, indices)
}

/// Handle multi-dim subscript when axis `dyn_ax` has a dynamic Range.
/// Slices the dynamic axis directly (no transpose), then recurses with
/// the remaining indices on the sliced result.
fn dyn_subscript_with_dynamic_axis(
    b: &mut IRBuilder,
    data: &DynamicNDArrayData,
    indices: &[SliceIndex],
    dyn_ax: usize,
) -> Value {
    let (start, stop, step) = match &indices[dyn_ax] {
        SliceIndex::Range(s, e, st) => (s.as_ref(), e.as_ref(), st.as_ref()),
        _ => unreachable!(),
    };

    let sliced = dyn_slice_axis(b, data, dyn_ax, start, stop, step);

    // Build remaining indices: replace the dynamic Range with full range `:`.
    let remaining: Vec<SliceIndex> = indices.iter().enumerate().map(|(i, idx)| {
        if i == dyn_ax {
            SliceIndex::Range(None, None, None)
        } else {
            idx.clone()
        }
    }).collect();

    let all_trivial = remaining.iter().all(|idx| match idx {
        SliceIndex::Range(s, e, st) => {
            matches!(s, None | Some(Value::None)) &&
            matches!(e, None | Some(Value::None)) &&
            matches!(st, None | Some(Value::None))
        }
        _ => false,
    });

    if all_trivial {
        return sliced;
    }

    let s_data = match &sliced {
        Value::DynamicNDArray(d) => d,
        _ => unreachable!(),
    };
    dyn_subscript(b, s_data, &remaining)
}

/// Slice along any axis of a multi-dim array with dynamic bounds.
/// No transpose — computes source addresses directly using the axis stride.
///
/// For each output element, the source address is:
///   base_from_other_axes + (start + slice_idx * step) * stride[axis]
/// where other axes use their full coordinate ranges.
fn dyn_slice_axis(
    b: &mut IRBuilder,
    data: &DynamicNDArrayData,
    axis: usize,
    start: Option<&Value>,
    stop: Option<&Value>,
    step: Option<&Value>,
) -> Value {
    let shape = &data.meta.logical_shape;
    let rank = shape.len();

    if rank == 1 {
        let out = dyn_slice_1d(b, data, start, stop, step);
        if let Some(out_vid) = out.value_id() {
            crate::optim::resolver::relay_forall_eq_const_from_input(b, data.value_id, out_vid);
        }
        return out;
    }

    let mode = select_stride_mode(data);
    let axis_len = shape[axis];
    let max_slice_len = axis_len; // upper bound on output size along this axis

    // Load-bearing slice-bound discharge (compiler.fuzz-finding-v2-slice-oob-witness-miss).
    // See `dyn_slice_1d` for rationale; multi-dim path uses `total_bound`
    // for the same reshape over-approximation reason.
    let bound = data.envelope.total_bound;
    crate::optim::resolver::discharge_slice_bound(b, start, bound, "dyn_slice_start");
    crate::optim::resolver::discharge_slice_bound(b, stop, bound, "dyn_slice_stop");

    fn val_to_ir(b: &mut IRBuilder, v: Option<&Value>, default: i64) -> Value {
        match v {
            Some(val) if !matches!(val, Value::None) => {
                if let Some(s) = val.int_val() { b.ir_constant_int(s) } else { val.clone() }
            }
            _ => b.ir_constant_int(default),
        }
    }
    let start_ir = val_to_ir(b, start, 0);
    let stop_ir = val_to_ir(b, stop, axis_len as i64);
    let step_ir = val_to_ir(b, step, 1);

    // Output shape: same as input but with the sliced axis replaced by max_slice_len.
    let mut out_shape = shape.clone();
    out_shape[axis] = max_slice_len;

    // Build coordinate ranges for non-sliced axes.
    let other_axes: Vec<(usize, Vec<usize>)> = (0..rank)
        .filter(|&ax| ax != axis)
        .map(|ax| (ax, (0..shape[ax]).collect()))
        .collect();

    // Total output = product of all dims in out_shape.
    let out_total: usize = out_shape.iter().product();
    let out_strides = row_major_strides(&out_shape);

    let default_val = crate::ops::dyn_ndarray::metadata::dyn_default_value(b, data.dtype);
    let zero = b.ir_constant_int(0);
    let axis_len_val = b.ir_constant_int(axis_len as i64);
    let axis_stride_val = stride_value(b, &mode, axis);

    let mut out_elements = Vec::with_capacity(out_total);

    for flat_out in 0..out_total {
        let out_coords = crate::helpers::shape_arith::decode_coords(flat_out, &out_shape, &out_strides);

        // The coordinate along the sliced axis is the slice index.
        let slice_idx = out_coords[axis];
        let slice_idx_val = b.ir_constant_int(slice_idx as i64);

        // Source index along the sliced axis: start + slice_idx * step.
        let offset = b.ir_mul_i(&slice_idx_val, &step_ir);
        let src_axis_idx = b.ir_add_i(&start_ir, &offset);

        // In-bounds check.
        let ge_zero = b.ir_greater_than_or_equal_i(&src_axis_idx, &zero);
        let lt_len = b.ir_less_than_i(&src_axis_idx, &axis_len_val);
        let in_range = b.ir_logical_and(&ge_zero, &lt_len);

        let step_pos = b.ir_greater_than_i(&step_ir, &zero);
        let lt_stop = b.ir_less_than_i(&src_axis_idx, &stop_ir);
        let gt_stop = b.ir_greater_than_i(&src_axis_idx, &stop_ir);
        let stop_ok = b.ir_select_i(&step_pos, &lt_stop, &gt_stop);
        let stop_bool = b.ir_bool_cast(&stop_ok);
        let in_bounds = b.ir_logical_and(&in_range, &stop_bool);

        // Clamp for safe read.
        let max_idx = b.ir_constant_int(axis_len as i64 - 1);
        let is_neg = b.ir_less_than_i(&src_axis_idx, &zero);
        let is_over = b.ir_greater_than_i(&src_axis_idx, &max_idx);
        let clamped_hi = b.ir_select_i(&is_over, &max_idx, &src_axis_idx);
        let clamped = b.ir_select_i(&is_neg, &zero, &clamped_hi);

        // Compute full source flat address: other axes at their coords + clamped axis.
        // The stride layout (literal vs symbolic-runtime) is selected by
        // `select_stride_mode` above; literal mode folds the other-axis
        // contribution at compile time, symbolic mode emits per-axis
        // `ir_mul_i` against `runtime_strides`.
        let other_offset = match &mode {
            StrideMode::LiteralLogical(strides) => {
                let mut addr_static: i64 = 0;
                for &(ax, _) in &other_axes {
                    addr_static += out_coords[ax] as i64 * strides[ax] as i64;
                }
                b.ir_constant_int(addr_static)
            }
            StrideMode::SymbolicRuntime(_) => {
                let mut acc: Option<Value> = None;
                for &(ax, _) in &other_axes {
                    let coord = b.ir_constant_int(out_coords[ax] as i64);
                    let stride_v = stride_value(b, &mode, ax);
                    let contrib = b.ir_mul_i(&coord, &stride_v);
                    acc = Some(match acc.take() {
                        None => contrib,
                        Some(prev) => b.ir_add_i(&prev, &contrib),
                    });
                }
                acc.unwrap_or_else(|| b.ir_constant_int(0))
            }
        };
        let axis_contrib = b.ir_mul_i(&clamped, &axis_stride_val);
        let addr = b.ir_add_i(&other_offset, &axis_contrib);

        let elem = b.ir_read_memory(data.segment_id, &addr);
        let masked = if data.dtype == NumberType::Float {
            b.ir_select_f(&in_bounds, &elem, &default_val)
        } else {
            b.ir_select_i(&in_bounds, &elem, &default_val)
        };
        out_elements.push(crate::ops::dyn_ndarray::value_to_scalar_i64(&masked));
    }

    // Runtime length along sliced axis.
    let step_pos_2 = b.ir_greater_than_i(&step_ir, &zero);
    let s_m_s = b.ir_sub_i(&stop_ir, &start_ir);
    let s_m_s2 = b.ir_sub_i(&start_ir, &stop_ir);
    let diff = b.ir_select_i(&step_pos_2, &s_m_s, &s_m_s2);
    let neg_step = b.ir_sub_i(&zero, &step_ir);
    let abs_step = b.ir_select_i(&step_pos_2, &step_ir, &neg_step);
    let diff_pos = b.ir_greater_than_i(&diff, &zero);
    let one = b.ir_constant_int(1);
    let abs_step_m1 = b.ir_sub_i(&abs_step, &one);
    let numerator = b.ir_add_i(&diff, &abs_step_m1);
    let divided = b.ir_div_i(&numerator, &abs_step);
    let runtime_axis_len = b.ir_select_i(&diff_pos, &divided, &zero);

    // Build output metadata.
    let segment_id = crate::helpers::segment::alloc_and_write(b, &out_elements, data.dtype);

    let mut dims: Vec<crate::types::Dim> = Vec::with_capacity(rank);
    let mut rt_shape: Vec<ScalarValue<i64>> = Vec::with_capacity(rank);
    for ax in 0..rank {
        if ax == axis {
            dims.push(crate::types::Dim::new_dynamic(&mut b.dim_table, 0, max_slice_len));
            rt_shape.push(crate::ops::dyn_ndarray::value_to_scalar_i64(&runtime_axis_len));
        } else {
            dims.push(crate::types::Dim::new_static(&mut b.dim_table, shape[ax]));
            rt_shape.push(ScalarValue::new(Some(shape[ax] as i64), None));
        }
    }
    let envelope = crate::types::Envelope::new_with_bound(dims, out_total.min(data.envelope.total_bound));

    // Runtime total length.
    let other_product: i64 = (0..rank).filter(|&ax| ax != axis).map(|ax| shape[ax] as i64).product();
    let other_prod_val = b.ir_constant_int(other_product);
    let runtime_length = b.ir_mul_i(&runtime_axis_len, &other_prod_val);
    let runtime_len_sv = crate::ops::dyn_ndarray::value_to_scalar_i64(&runtime_length);

    let out = Value::DynamicNDArray(DynamicNDArrayData {
        envelope, dtype: data.dtype, segment_id,
        meta: DynArrayMeta {
            logical_shape: out_shape.clone(), logical_offset: 0, logical_strides: out_strides.clone(),
            runtime_length: runtime_len_sv,
            runtime_rank: ScalarValue::new(Some(rank as i64), None),
            runtime_shape: rt_shape,
            runtime_strides: out_strides.iter().map(|&s| ScalarValue::new(Some(s as i64), None)).collect(),
            runtime_offset: ScalarValue::new(Some(0), None),
        },
        value_id: ValueId::next(),
    });
    if let Some(out_vid) = out.value_id() {
        crate::optim::resolver::relay_forall_eq_const_from_input(b, data.value_id, out_vid);
    }
    out
}

/// Static multi-dim subscript: all Range bounds are compile-time known.
/// Uses direct coordinate-based reads.
fn dyn_subscript_multidim_static(
    b: &mut IRBuilder, data: &DynamicNDArrayData, indices: &[SliceIndex],
) -> Value {
    let shape = &data.meta.logical_shape;
    let mode = select_stride_mode(data);
    let rank = shape.len();

    let mut out_shape: Vec<usize> = Vec::new();
    let mut axis_ranges: Vec<(usize, Vec<usize>)> = Vec::new();

    for (ax, idx) in indices.iter().enumerate() {
        match idx {
            SliceIndex::Single(_) => {}
            SliceIndex::Range(start, stop, step) => {
                let dim = shape[ax] as i64;
                let resolve = |v: &Option<Value>, default: i64| -> i64 {
                    match v.as_ref() {
                        Some(Value::None) | None => default,
                        Some(val) => val.int_val().unwrap_or(default),
                    }
                };
                let s = resolve(start, 0);
                let e = resolve(stop, dim);
                let st = resolve(step, 1);
                let s = if s < 0 { (dim + s).max(0) } else { s.min(dim) } as usize;
                let e = if e < 0 { (dim + e).max(0) } else { e.min(dim) } as usize;
                let mut coords = Vec::new();
                let mut i = s;
                if st > 0 {
                    while i < e { coords.push(i); i += st as usize; }
                } else {
                    while i > e { coords.push(i); i = i.wrapping_add(st as usize); }
                }
                out_shape.push(coords.len());
                axis_ranges.push((ax, coords));
            }
            _ => panic!("DynamicNDArray: Ellipsis/NewAxis indexing not yet supported"),
        }
    }

    for ax in indices.len()..rank {
        let coords: Vec<usize> = (0..shape[ax]).collect();
        out_shape.push(coords.len());
        axis_ranges.push((ax, coords));
    }

    let out_total: usize = out_shape.iter().product();
    let out_strides_out = row_major_strides(&out_shape);
    let mut out_elements = Vec::with_capacity(out_total);

    // Base offset from Single indices. Literal-mode folds static
    // contributions at compile time; symbolic-mode emits an `ir_mul_i`
    // chain against `runtime_strides`.
    let base_offset = match &mode {
        StrideMode::LiteralLogical(strides) => {
            let mut fixed_offset_static: i64 = 0;
            let mut dynamic_offset_parts: Vec<Value> = Vec::new();
            for (ax, idx) in indices.iter().enumerate() {
                if let SliceIndex::Single(v) = idx {
                    if let Some(i) = v.int_val() {
                        let i = if i < 0 { shape[ax] as i64 + i } else { i };
                        fixed_offset_static += i * strides[ax] as i64;
                    } else {
                        let stride_val = b.ir_constant_int(strides[ax] as i64);
                        let contrib = b.ir_mul_i(v, &stride_val);
                        dynamic_offset_parts.push(contrib);
                    }
                }
            }
            if dynamic_offset_parts.is_empty() {
                b.ir_constant_int(fixed_offset_static)
            } else {
                let mut acc = b.ir_constant_int(fixed_offset_static);
                for part in &dynamic_offset_parts { acc = b.ir_add_i(&acc, part); }
                acc
            }
        }
        StrideMode::SymbolicRuntime(_) => {
            let mut acc: Option<Value> = None;
            for (ax, idx) in indices.iter().enumerate() {
                if let SliceIndex::Single(v) = idx {
                    let idx_val = if let Some(i) = v.int_val() {
                        let i = if i < 0 { shape[ax] as i64 + i } else { i };
                        b.ir_constant_int(i)
                    } else {
                        v.clone()
                    };
                    let stride_v = stride_value(b, &mode, ax);
                    let contrib = b.ir_mul_i(&idx_val, &stride_v);
                    acc = Some(match acc.take() {
                        None => contrib,
                        Some(prev) => b.ir_add_i(&prev, &contrib),
                    });
                }
            }
            acc.unwrap_or_else(|| b.ir_constant_int(0))
        }
    };

    for flat_out in 0..out_total {
        let out_coords = crate::helpers::shape_arith::decode_coords(flat_out, &out_shape, &out_strides_out);
        let offset_val = match &mode {
            StrideMode::LiteralLogical(strides) => {
                let mut src_offset: i64 = 0;
                for (out_ax, &(src_ax, ref coords)) in axis_ranges.iter().enumerate() {
                    src_offset += coords[out_coords[out_ax]] as i64 * strides[src_ax] as i64;
                }
                b.ir_constant_int(src_offset)
            }
            StrideMode::SymbolicRuntime(_) => {
                let mut acc: Option<Value> = None;
                for (out_ax, &(src_ax, ref coords)) in axis_ranges.iter().enumerate() {
                    let coord = b.ir_constant_int(coords[out_coords[out_ax]] as i64);
                    let stride_v = stride_value(b, &mode, src_ax);
                    let contrib = b.ir_mul_i(&coord, &stride_v);
                    acc = Some(match acc.take() {
                        None => contrib,
                        Some(prev) => b.ir_add_i(&prev, &contrib),
                    });
                }
                acc.unwrap_or_else(|| b.ir_constant_int(0))
            }
        };
        let addr = b.ir_add_i(&base_offset, &offset_val);
        let elem = b.ir_read_memory(data.segment_id, &addr);
        out_elements.push(crate::ops::dyn_ndarray::value_to_scalar_i64(&elem));
    }

    if out_shape.is_empty() {
        return b.ir_read_memory(data.segment_id, &base_offset);
    }

    let segment_id = crate::helpers::segment::alloc_and_write(b, &out_elements, data.dtype);
    let envelope = crate::types::Envelope::from_static_shape(&mut b.dim_table, &out_shape);

    Value::DynamicNDArray(DynamicNDArrayData {
        envelope, dtype: data.dtype, segment_id,
        meta: DynArrayMeta {
            logical_shape: out_shape.clone(), logical_offset: 0, logical_strides: out_strides_out.clone(),
            runtime_length: ScalarValue::new(Some(out_total as i64), None),
            runtime_rank: ScalarValue::new(Some(out_shape.len() as i64), None),
            runtime_shape: out_shape.iter().map(|&s| ScalarValue::new(Some(s as i64), None)).collect(),
            runtime_strides: out_strides_out.iter().map(|&s| ScalarValue::new(Some(s as i64), None)).collect(),
            runtime_offset: ScalarValue::new(Some(0), None),
        },
        value_id: ValueId::next(),
    })
}


// ── Fancy indexing ──────────────────────────────────────────────────────

fn dyn_fancy_index(b: &mut IRBuilder, data: &DynamicNDArrayData, idx_array: &Value) -> Value {
    let indices = extract_index_values(idx_array);
    let rank = data.envelope.rank();

    if rank == 1 {
        let out_len = indices.len();
        let mut out_elements = Vec::with_capacity(out_len);
        for idx in &indices {
            let addr = if let Some(i) = idx.int_val() {
                let i = if i < 0 { data.meta.logical_shape[0] as i64 + i } else { i };
                b.ir_constant_int(i)
            } else { idx.clone() };
            let elem = b.ir_read_memory(data.segment_id, &addr);
            out_elements.push(crate::ops::dyn_ndarray::value_to_scalar_i64(&elem));
        }
        let segment_id = crate::helpers::segment::alloc_and_write(b, &out_elements, data.dtype);
        let envelope = crate::types::Envelope::from_static_shape(&mut b.dim_table, &[out_len]);
        return Value::DynamicNDArray(DynamicNDArrayData {
            envelope, dtype: data.dtype, segment_id,
            meta: DynArrayMeta {
                logical_shape: vec![out_len], logical_offset: 0, logical_strides: vec![1],
                runtime_length: ScalarValue::new(Some(out_len as i64), None),
                runtime_rank: ScalarValue::new(Some(1), None),
                runtime_shape: vec![ScalarValue::new(Some(out_len as i64), None)],
                runtime_strides: vec![ScalarValue::new(Some(1), None)],
                runtime_offset: ScalarValue::new(Some(0), None),
            },
            value_id: ValueId::next(),
        });
    }

    // Multi-dim: each index selects a row along axis 0.
    let mode = select_stride_mode(data);
    let row_shape: Vec<usize> = data.meta.logical_shape[1..].to_vec();
    let row_size: usize = row_shape.iter().product();
    let num_indices = indices.len();
    let out_total = num_indices * row_size;
    let mut out_elements = Vec::with_capacity(out_total);

    for idx in &indices {
        let base = match &mode {
            StrideMode::LiteralLogical(strides) => {
                let row_stride = strides[0];
                if let Some(i) = idx.int_val() {
                    let i = if i < 0 { data.meta.logical_shape[0] as i64 + i } else { i };
                    b.ir_constant_int(i * row_stride as i64)
                } else {
                    let stride_val = b.ir_constant_int(row_stride as i64);
                    b.ir_mul_i(idx, &stride_val)
                }
            }
            StrideMode::SymbolicRuntime(_) => {
                let idx_v = if let Some(i) = idx.int_val() {
                    let i = if i < 0 { data.meta.logical_shape[0] as i64 + i } else { i };
                    b.ir_constant_int(i)
                } else {
                    idx.clone()
                };
                let stride_v = stride_value(b, &mode, 0);
                b.ir_mul_i(&idx_v, &stride_v)
            }
        };
        for j in 0..row_size {
            let offset = b.ir_constant_int(j as i64);
            let addr = b.ir_add_i(&base, &offset);
            let elem = b.ir_read_memory(data.segment_id, &addr);
            out_elements.push(crate::ops::dyn_ndarray::value_to_scalar_i64(&elem));
        }
    }

    let mut out_shape = vec![num_indices];
    out_shape.extend(&row_shape);
    let out_strides = row_major_strides(&out_shape);
    let segment_id = crate::helpers::segment::alloc_and_write(b, &out_elements, data.dtype);
    let envelope = crate::types::Envelope::from_static_shape(&mut b.dim_table, &out_shape);

    Value::DynamicNDArray(DynamicNDArrayData {
        envelope, dtype: data.dtype, segment_id,
        meta: DynArrayMeta {
            logical_shape: out_shape.clone(), logical_offset: 0, logical_strides: out_strides.clone(),
            runtime_length: ScalarValue::new(Some(out_total as i64), None),
            runtime_rank: ScalarValue::new(Some(out_shape.len() as i64), None),
            runtime_shape: out_shape.iter().map(|&s| ScalarValue::new(Some(s as i64), None)).collect(),
            runtime_strides: out_strides.iter().map(|&s| ScalarValue::new(Some(s as i64), None)).collect(),
            runtime_offset: ScalarValue::new(Some(0), None),
        },
        value_id: ValueId::next(),
    })
}

fn dyn_fancy_index_multidim(
    b: &mut IRBuilder, data: &DynamicNDArrayData, idx_arrays: &[&Value],
) -> Value {
    let mode = select_stride_mode(data);
    let shape = &data.meta.logical_shape;
    let arrays: Vec<Vec<Value>> = idx_arrays.iter().map(|v| extract_index_values(v)).collect();
    let out_len = arrays[0].len();
    assert!(arrays.iter().all(|a| a.len() == out_len), "fancy indexing: all index arrays must have the same length");

    let mut out_elements = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let addr = match &mode {
            StrideMode::LiteralLogical(strides) => {
                let mut static_sum: i64 = 0;
                let mut dynamic_parts: Vec<Value> = Vec::new();
                for (ax, arr) in arrays.iter().enumerate() {
                    let idx_val = &arr[i];
                    let stride = strides[ax] as i64;
                    if let Some(v) = idx_val.int_val() {
                        let v = if v < 0 { shape[ax] as i64 + v } else { v };
                        static_sum += v * stride;
                    } else {
                        let stride_val = b.ir_constant_int(stride);
                        dynamic_parts.push(b.ir_mul_i(idx_val, &stride_val));
                    }
                }
                if dynamic_parts.is_empty() {
                    b.ir_constant_int(static_sum)
                } else {
                    let mut acc = b.ir_constant_int(static_sum);
                    for part in &dynamic_parts { acc = b.ir_add_i(&acc, part); }
                    acc
                }
            }
            StrideMode::SymbolicRuntime(_) => {
                let mut acc: Option<Value> = None;
                for (ax, arr) in arrays.iter().enumerate() {
                    let idx_val = &arr[i];
                    let v = if let Some(vi) = idx_val.int_val() {
                        let vi = if vi < 0 { shape[ax] as i64 + vi } else { vi };
                        b.ir_constant_int(vi)
                    } else {
                        idx_val.clone()
                    };
                    let stride_v = stride_value(b, &mode, ax);
                    let contrib = b.ir_mul_i(&v, &stride_v);
                    acc = Some(match acc.take() {
                        None => contrib,
                        Some(prev) => b.ir_add_i(&prev, &contrib),
                    });
                }
                acc.unwrap_or_else(|| b.ir_constant_int(0))
            }
        };
        out_elements.push(crate::ops::dyn_ndarray::value_to_scalar_i64(&b.ir_read_memory(data.segment_id, &addr)));
    }

    let segment_id = crate::helpers::segment::alloc_and_write(b, &out_elements, data.dtype);
    let envelope = crate::types::Envelope::from_static_shape(&mut b.dim_table, &[out_len]);

    Value::DynamicNDArray(DynamicNDArrayData {
        envelope, dtype: data.dtype, segment_id,
        meta: DynArrayMeta {
            logical_shape: vec![out_len], logical_offset: 0, logical_strides: vec![1],
            runtime_length: ScalarValue::new(Some(out_len as i64), None),
            runtime_rank: ScalarValue::new(Some(1), None),
            runtime_shape: vec![ScalarValue::new(Some(out_len as i64), None)],
            runtime_strides: vec![ScalarValue::new(Some(1), None)],
            runtime_offset: ScalarValue::new(Some(0), None),
        },
        value_id: ValueId::next(),
    })
}

// ── Helpers ─────────────────────────────────────────────────────────────

/// Compute a flat address into `data`'s segment from a list of single
/// indices.
///
/// Three-way dispatch on [`select_stride_mode`]:
/// * [`StrideMode::LiteralLogical`] — compile-time constant strides; static
///   indices fold into `static_sum`, dynamic indices contribute one
///   `ir_mul_i` per axis. Zero regression for non-bounded arrays.
/// * [`StrideMode::SymbolicRuntime`] — SSA-`Value` strides from the
///   compact buffer's runtime layout. Every axis pays an `ir_mul_i` plus
///   an `ir_add_i` (no constant folding when one operand is an SSA
///   `Value`). Strict mode additionally emits per-axis
///   `0 <= idx < runtime_shape[axis]` assertions via `ir_assert`.
pub(crate) fn compute_flat_addr(b: &mut IRBuilder, data: &DynamicNDArrayData, indices: &[SliceIndex]) -> Value {
    let mode = select_stride_mode(data);
    let shape = &data.meta.logical_shape;

    match mode {
        StrideMode::LiteralLogical(strides) => {
            let mut static_sum: i64 = 0;
            let mut dynamic_parts: Vec<Value> = Vec::new();

            for (ax, idx) in indices.iter().enumerate() {
                let stride = strides[ax] as i64;
                match idx {
                    SliceIndex::Single(v) => {
                        if let Some(i) = v.int_val() {
                            let i = if i < 0 { shape[ax] as i64 + i } else { i };
                            static_sum += i * stride;
                        } else {
                            let stride_val = b.ir_constant_int(stride);
                            dynamic_parts.push(b.ir_mul_i(v, &stride_val));
                        }
                    }
                    _ => panic!("compute_flat_addr: expected Single index"),
                }
            }

            if dynamic_parts.is_empty() {
                b.ir_constant_int(static_sum)
            } else {
                let mut acc = b.ir_constant_int(static_sum);
                for part in &dynamic_parts { acc = b.ir_add_i(&acc, part); }
                acc
            }
        }
        StrideMode::SymbolicRuntime(_) => {
            // Compact-buffer mode: every axis pays an ir_mul_i with the
            // SSA stride. Static indices still bypass `ir_int_cast`-style
            // wrapping by going through `ir_constant_int(i)`, but the
            // resulting multiplication cannot be constant-folded.
            let strict = bounded_axis_strict();
            let zero = if strict { Some(b.ir_constant_int(0)) } else { None };

            let mut acc: Option<Value> = None;
            for (ax, idx) in indices.iter().enumerate() {
                let SliceIndex::Single(v) = idx else {
                    panic!("compute_flat_addr: expected Single index");
                };
                let idx_val = if let Some(i) = v.int_val() {
                    let i = if i < 0 { shape[ax] as i64 + i } else { i };
                    b.ir_constant_int(i)
                } else {
                    v.clone()
                };
                let stride_val = stride_value(b, &mode, ax);
                let contrib = b.ir_mul_i(&idx_val, &stride_val);
                acc = Some(match acc.take() {
                    None => contrib,
                    Some(prev) => b.ir_add_i(&prev, &contrib),
                });

                if strict {
                    let zero_ref = zero.as_ref().expect("strict mode allocates zero");
                    let len_val = match data.meta.runtime_shape[ax].static_val {
                        Some(v) => b.ir_constant_int(v),
                        None => match data.meta.runtime_shape[ax].stmt_id {
                            Some(ptr) => Value::Integer(ScalarValue::new(None, Some(ptr))),
                            None => b.ir_constant_int(shape[ax] as i64),
                        },
                    };
                    let ge = b.ir_greater_than_or_equal_i(&idx_val, zero_ref);
                    let lt = b.ir_less_than_i(&idx_val, &len_val);
                    let in_range = b.ir_logical_and(&ge, &lt);
                    b.ir_assert(&in_range);
                }
            }

            acc.unwrap_or_else(|| b.ir_constant_int(0))
        }
    }
}

// ── Tests (multi-dim Case B Tier 1) ─────────────────────────────────────

#[cfg(test)]
mod tests_case_b {
    use super::*;
    use crate::ir_defs::IR;
    use crate::ops::dyn_ndarray::constructors::{
        dyn_fill, dyn_from_values_with_active_compact, dyn_from_values_with_active_nd,
    };
    use crate::ops::dyn_ndarray::value_to_scalar_i64;
    use crate::types::{CompositeData, Envelope, NumberType, ZinniaType};
    use std::collections::HashMap;

    /// Drop a stmts-prefix-snapshot in front of a closure: returns the new
    /// stmts emitted between the snapshot and the closure's exit.
    fn emitted_during<R>(b: &mut IRBuilder, f: impl FnOnce(&mut IRBuilder) -> R) -> (R, Vec<IR>) {
        let before = b.stmts.len();
        let result = f(b);
        let irs = b.stmts[before..].iter().map(|s| s.ir.clone()).collect();
        (result, irs)
    }

    fn make_compact_2d_dyn(b: &mut IRBuilder, m_max: usize, n_max: usize, bound: usize) -> Value {
        // Build a compact-mode 2-D dyn ndarray with symbolic m, n axes.
        let m_sv = value_to_scalar_i64(&b.ir_constant_int(m_max as i64));
        let n_sv = value_to_scalar_i64(&b.ir_constant_int(n_max as i64));
        // Force runtime_shape entries to look symbolic (clear static_val
        // so the dispatch sees `any_bounded = true`).
        let m_symbolic = ScalarValue::new(None, m_sv.stmt_id);
        let n_symbolic = ScalarValue::new(None, n_sv.stmt_id);
        let zero_sv = value_to_scalar_i64(&b.ir_constant_int(0));
        dyn_from_values_with_active_compact(
            b,
            zero_sv,
            vec![m_max, n_max],
            vec![m_symbolic, n_symbolic],
            bound,
            NumberType::Integer,
        )
    }

    #[test]
    fn compute_flat_addr_uses_runtime_strides_when_bounded_imbalanced() {
        let mut b = IRBuilder::new();
        let arr = make_compact_2d_dyn(&mut b, 100, 100, 10);
        let data = match &arr {
            Value::DynamicNDArray(d) => d.clone(),
            _ => unreachable!(),
        };
        // Sanity: dispatch picks SymbolicRuntime.
        let mode = select_stride_mode(&data);
        assert!(matches!(mode, StrideMode::SymbolicRuntime(_)));

        let indices = vec![
            SliceIndex::Single(b.ir_constant_int(0)),
            SliceIndex::Single(b.ir_constant_int(0)),
        ];
        let (_addr, irs) = emitted_during(&mut b, |b| compute_flat_addr(b, &data, &indices));
        // Compact dispatch emits at least one MulI per axis (here: 2 axes
        // ⇒ at least 2 MulIs). The literal path would emit none for two
        // static-zero indices.
        let muli_count = irs.iter().filter(|ir| matches!(ir, IR::MulI)).count();
        assert!(
            muli_count >= 2,
            "expected at least 2 MulI emissions in SymbolicRuntime mode, got {muli_count}; irs = {irs:?}"
        );
    }

    #[test]
    fn compute_flat_addr_uses_logical_strides_when_not_bounded() {
        let mut b = IRBuilder::new();
        // Build a static-shape dyn-ndarray via the standard fill path.
        let shape = Value::Tuple(CompositeData {
            elements_type: vec![ZinniaType::Integer, ZinniaType::Integer],
            values: vec![b.ir_constant_int(3), b.ir_constant_int(4)],
        
            value_id: ValueId::next(),
        });
        let result = dyn_fill(&mut b, &[shape], &HashMap::new(), 0);
        let data = match &result {
            Value::DynamicNDArray(d) => d.clone(),
            _ => unreachable!(),
        };
        let mode = select_stride_mode(&data);
        assert!(matches!(mode, StrideMode::LiteralLogical(_)));

        let indices = vec![
            SliceIndex::Single(b.ir_constant_int(1)),
            SliceIndex::Single(b.ir_constant_int(2)),
        ];
        let (_addr, irs) = emitted_during(&mut b, |b| compute_flat_addr(b, &data, &indices));
        // Literal-stride path folds two static indices into a single
        // ConstantInt — no MulI/AddI in the static-static case.
        let muli_count = irs.iter().filter(|ir| matches!(ir, IR::MulI)).count();
        assert_eq!(muli_count, 0, "literal-stride dispatch should fold static indices; got irs = {irs:?}");
    }

    #[test]
    fn np_fill_multi_dim_uses_compact_buffer_when_total_bound_proven_tighter() {
        let mut b = IRBuilder::new();
        // Plant m, n with bounded ranges and a product fact m*n <= 10.
        // We synthesise this directly by constructing a compact array via
        // the constructor primitive — equivalent to what np_fill's
        // multi-dim path emits when resolve_int_or_bounded proves the
        // product bound. (np_fill itself runs end-to-end through visitors
        // that aren't exercised at this seam test.)
        let arr = make_compact_2d_dyn(&mut b, 10, 10, 10);
        let data = match &arr {
            Value::DynamicNDArray(d) => d,
            _ => unreachable!(),
        };
        assert_eq!(data.envelope.total_bound, 10);
        // The compact buffer has total_bound slots, not product(max_shape).
        // The static `max_total` derived from the envelope is the cap on
        // observable addresses.
        let max_product: usize = data.meta.logical_shape.iter().product();
        assert_eq!(max_product, 100);
        assert!(data.envelope.total_bound < max_product);
    }

    #[test]
    fn transpose_2d_bounded_preserves_active_region() {
        let mut b = IRBuilder::new();
        // Build a static-shape 2-D bounded array (non-compact, so transpose
        // can materialize through the literal-stride path).
        let m_sv = value_to_scalar_i64(&b.ir_constant_int(5));
        let n_sv = value_to_scalar_i64(&b.ir_constant_int(3));
        let zero_sv = value_to_scalar_i64(&b.ir_constant_int(0));
        let len_val = b.ir_constant_int(15);
        let arr = dyn_from_values_with_active_nd(
            &mut b,
            vec![zero_sv; 15],
            vec![5, 3],
            vec![m_sv.clone(), n_sv.clone()],
            len_val,
            NumberType::Integer,
        );
        let data = match &arr {
            Value::DynamicNDArray(d) => d.clone(),
            _ => unreachable!(),
        };
        let total_bound_in = data.envelope.total_bound;

        let transposed = crate::ops::dyn_ndarray::reshape::dyn_transpose(&mut b, &data, &[]);
        let out = match &transposed {
            Value::DynamicNDArray(d) => d,
            _ => unreachable!(),
        };
        // runtime_shape must be permuted (axis 0 ↔ axis 1 for 2-D).
        assert_eq!(out.meta.runtime_shape[0].static_val, Some(3));
        assert_eq!(out.meta.runtime_shape[1].static_val, Some(5));
        // total_bound is preserved (transpose is a metadata-permutation
        // operation; the active region's size doesn't change).
        assert_eq!(out.envelope.total_bound, total_bound_in);
    }

    #[test]
    fn elementwise_add_on_2d_bounded_arrays() {
        let mut b = IRBuilder::new();
        let m_sv = value_to_scalar_i64(&b.ir_constant_int(3));
        let n_sv = value_to_scalar_i64(&b.ir_constant_int(4));
        let zero_sv = value_to_scalar_i64(&b.ir_constant_int(0));
        let len_lhs = b.ir_constant_int(12);
        let lhs = dyn_from_values_with_active_nd(
            &mut b,
            vec![zero_sv.clone(); 12],
            vec![3, 4],
            vec![m_sv.clone(), n_sv.clone()],
            len_lhs,
            NumberType::Integer,
        );
        let len_rhs = b.ir_constant_int(12);
        let rhs = dyn_from_values_with_active_nd(
            &mut b,
            vec![zero_sv.clone(); 12],
            vec![3, 4],
            vec![m_sv, n_sv],
            len_rhs,
            NumberType::Integer,
        );
        let lhs_d = match &lhs { Value::DynamicNDArray(d) => d.clone(), _ => unreachable!() };
        let rhs_d = match &rhs { Value::DynamicNDArray(d) => d.clone(), _ => unreachable!() };
        let out = crate::ops::dyn_ndarray::binary::dyn_binary_op(&mut b, "add", &lhs_d, &rhs_d);
        let out_d = match &out { Value::DynamicNDArray(d) => d, _ => unreachable!() };
        assert_eq!(out_d.meta.logical_shape, vec![3, 4]);
        assert_eq!(out_d.meta.runtime_shape[0].static_val, Some(3));
        assert_eq!(out_d.meta.runtime_shape[1].static_val, Some(4));
    }

    #[test]
    fn strict_mode_constructor_emits_total_bound_invariant() {
        // Save+restore env var to keep the test thread-safe(ish). The
        // strict-mode check is read once per constructor call, so setting
        // immediately before is sufficient.
        let _guard = ScopedEnvVar::set("ZINNIA_BOUNDED_AXIS_STRICT", "1");
        let mut b = IRBuilder::new();
        let (_arr, irs) = emitted_during(&mut b, |b| make_compact_2d_dyn(b, 100, 100, 10));
        // The compact constructor in strict mode emits an Assert on the
        // <=-comparison `runtime_length <= total_bound`. There should be
        // exactly one Assert in the trace.
        let assert_count = irs.iter().filter(|ir| matches!(ir, IR::Assert)).count();
        assert!(
            assert_count >= 1,
            "strict-mode compact constructor should emit at least one Assert; irs = {irs:?}"
        );
    }

    #[test]
    fn strict_mode_oob_subscript_emits_assertion() {
        let _guard = ScopedEnvVar::set("ZINNIA_BOUNDED_AXIS_STRICT", "1");
        let mut b = IRBuilder::new();
        let arr = make_compact_2d_dyn(&mut b, 100, 100, 10);
        let data = match &arr {
            Value::DynamicNDArray(d) => d.clone(),
            _ => unreachable!(),
        };
        let indices = vec![
            SliceIndex::Single(b.ir_constant_int(0)),
            SliceIndex::Single(b.ir_constant_int(0)),
        ];
        let (_addr, irs) = emitted_during(&mut b, |b| compute_flat_addr(b, &data, &indices));
        // Per-axis bounds assertion: 2 axes ⇒ at least 2 Asserts.
        let assert_count = irs.iter().filter(|ir| matches!(ir, IR::Assert)).count();
        assert!(
            assert_count >= 2,
            "strict-mode subscript should emit per-axis Asserts; irs = {irs:?}"
        );
    }

    /// Shared mutex that serializes env-var-mutating tests. `cargo test`
    /// runs tests in parallel by default; without serialization, two tests
    /// that both `set_var`/`remove_var` on the same key race — one's
    /// `Drop`-restore overwrites the other's mid-flight set.
    static STRICT_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// RAII guard for env-var scoping inside tests. Acquires the shared
    /// `STRICT_ENV_LOCK` so concurrent strict-mode tests serialize.
    struct ScopedEnvVar {
        key: &'static str,
        previous: Option<String>,
        _lock: std::sync::MutexGuard<'static, ()>,
    }
    impl ScopedEnvVar {
        fn set(key: &'static str, value: &str) -> Self {
            let lock = STRICT_ENV_LOCK
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let previous = std::env::var(key).ok();
            std::env::set_var(key, value);
            ScopedEnvVar { key, previous, _lock: lock }
        }
    }
    impl Drop for ScopedEnvVar {
        fn drop(&mut self) {
            match &self.previous {
                Some(v) => std::env::set_var(self.key, v),
                None => std::env::remove_var(self.key),
            }
        }
    }

    // Suppress dead-code lint on Envelope import.
    #[allow(dead_code)]
    fn _envelope_unused() -> Option<Envelope> { None }
}
