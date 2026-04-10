//! Dynamic array indexing and slicing: element access, range slicing,
//! boolean masking, and fancy indexing.

use crate::builder::IRBuilder;
use crate::helpers::shape_arith::row_major_strides;
use crate::types::{
    DynArrayMeta, DynamicNDArrayData, NumberType, ScalarValue, SliceIndex, Value,
};

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

/// Check if a value looks like a boolean mask.
/// Only `Value::Boolean` leaves qualify — `Value::Integer(0/1)` is fancy indexing.
pub fn is_boolean_mask(val: &Value) -> bool {
    match val {
        Value::DynamicNDArray(_) => true,
        Value::List(d) | Value::Tuple(d) => {
            d.values.iter().all(|v| match v {
                Value::Boolean(_) => true,
                Value::List(_) | Value::Tuple(_) => is_boolean_mask(v),
                _ => false,
            })
        }
        _ => false,
    }
}

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
    let raw = b.ir_read_memory(data.segment_id, &addr);
    match data.dtype {
        NumberType::Float => Value::Float(ScalarValue::new(
            raw.int_val().map(|v| v as f64),
            raw.ptr(),
        )),
        NumberType::Integer => raw,
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
    let row_stride = data.meta.logical_strides[0];

    let base = if let Some(i) = idx_val.int_val() {
        let i = if i < 0 { data.meta.logical_shape[0] as i64 + i } else { i };
        b.ir_constant_int(i * row_stride as i64)
    } else {
        let stride_val = b.ir_constant_int(row_stride as i64);
        b.ir_mul_i(idx_val, &stride_val)
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
    let shape = &data.meta.logical_shape;
    let strides = &data.meta.logical_strides;
    let rank = shape.len();

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

    if let Some((dyn_ax, dyn_idx)) = first_dynamic_range {
        // Dynamic range at axis `dyn_ax`. Process it first.
        return dyn_subscript_with_dynamic_axis(b, data, indices, dyn_ax);
    }

    // All ranges are static — use direct coordinate-based reads.
    dyn_subscript_multidim_static(b, data, indices)
}

/// Handle multi-dim subscript when axis `dyn_ax` has a dynamic Range.
/// 1. Apply the dynamic range on axis `dyn_ax` via axis-0 trick
///    (transpose dyn_ax to front, slice axis 0, transpose back).
/// 2. Build remaining indices (with dyn_ax removed/replaced) and recurse.
fn dyn_subscript_with_dynamic_axis(
    b: &mut IRBuilder,
    data: &DynamicNDArrayData,
    indices: &[SliceIndex],
    dyn_ax: usize,
) -> Value {
    let rank = data.envelope.rank();

    // Extract the dynamic range.
    let (start, stop, step) = match &indices[dyn_ax] {
        SliceIndex::Range(s, e, st) => (s.as_ref(), e.as_ref(), st.as_ref()),
        _ => unreachable!(),
    };

    // If dyn_ax == 0, we can slice directly.
    // Otherwise, transpose dyn_ax to position 0, slice, transpose back.
    let (sliced, axis_was_transposed) = if dyn_ax == 0 {
        let s = dyn_slice_axis0(b, data, start, stop, step);
        (s, false)
    } else {
        // Transpose: move dyn_ax to position 0.
        let mut perm: Vec<usize> = (0..rank).collect();
        perm.remove(dyn_ax);
        perm.insert(0, dyn_ax);
        let perm_vals: Vec<Value> = perm.iter().map(|&p|
            Value::Integer(ScalarValue::new(Some(p as i64), None))
        ).collect();
        let perm_list = Value::List(crate::types::CompositeData {
            elements_type: vec![crate::types::ZinniaType::Integer; rank],
            values: perm_vals,
        });
        let transposed = crate::ops::dyn_ndarray::reshape::dyn_transpose(b, data, &[perm_list]);
        let t_data = match &transposed {
            Value::DynamicNDArray(d) => d,
            _ => unreachable!(),
        };

        // Slice axis 0 of the transposed array.
        let sliced = dyn_slice_axis0(b, t_data, start, stop, step);
        let s_data = match &sliced {
            Value::DynamicNDArray(d) => d,
            _ => unreachable!(),
        };

        // Transpose back: inverse permutation.
        let mut inv_perm = vec![0usize; rank];
        for (i, &p) in perm.iter().enumerate() {
            inv_perm[p] = i;
        }
        let inv_vals: Vec<Value> = inv_perm.iter().map(|&p|
            Value::Integer(ScalarValue::new(Some(p as i64), None))
        ).collect();
        let inv_list = Value::List(crate::types::CompositeData {
            elements_type: vec![crate::types::ZinniaType::Integer; rank],
            values: inv_vals,
        });
        let back = crate::ops::dyn_ndarray::reshape::dyn_transpose(b, s_data, &[inv_list]);
        (back, true)
    };

    // Build remaining indices: replace the dynamic Range with a full range `:`.
    let remaining: Vec<SliceIndex> = indices.iter().enumerate().map(|(i, idx)| {
        if i == dyn_ax {
            // Already sliced — use full range to keep this axis.
            SliceIndex::Range(None, None, None)
        } else {
            idx.clone()
        }
    }).collect();

    // Check if remaining indices are all trivial (full ranges or need processing).
    let all_trivial = remaining.iter().all(|idx| match idx {
        SliceIndex::Range(None, None, None) => true,
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

    // Recurse with remaining indices on the sliced result.
    let s_data = match &sliced {
        Value::DynamicNDArray(d) => d,
        _ => unreachable!(),
    };
    dyn_subscript(b, s_data, &remaining)
}

/// Static multi-dim subscript: all Range bounds are compile-time known.
/// Uses direct coordinate-based reads.
fn dyn_subscript_multidim_static(
    b: &mut IRBuilder, data: &DynamicNDArrayData, indices: &[SliceIndex],
) -> Value {
    let shape = &data.meta.logical_shape;
    let strides = &data.meta.logical_strides;
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

    let base_offset = if dynamic_offset_parts.is_empty() {
        b.ir_constant_int(fixed_offset_static)
    } else {
        let mut acc = b.ir_constant_int(fixed_offset_static);
        for part in &dynamic_offset_parts { acc = b.ir_add_i(&acc, part); }
        acc
    };

    for flat_out in 0..out_total {
        let out_coords = crate::helpers::shape_arith::decode_coords(flat_out, &out_shape, &out_strides_out);
        let mut src_offset: i64 = 0;
        for (out_ax, &(src_ax, ref coords)) in axis_ranges.iter().enumerate() {
            src_offset += coords[out_coords[out_ax]] as i64 * strides[src_ax] as i64;
        }
        let offset_val = b.ir_constant_int(src_offset);
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
    })
}

/// Slice along axis 0 of a multi-dim array: select rows start:stop:step.
/// For rank-1, delegates to dyn_slice_1d. For rank-N, reads selected rows
/// and packs them into a new (N-dim) array.
fn dyn_slice_axis0(
    b: &mut IRBuilder,
    data: &DynamicNDArrayData,
    start: Option<&Value>,
    stop: Option<&Value>,
    step: Option<&Value>,
) -> Value {
    let rank = data.envelope.rank();
    if rank == 1 {
        return dyn_slice_1d(b, data, start, stop, step);
    }

    let axis0_len = data.meta.logical_shape[0];
    let row_stride = data.meta.logical_strides[0];
    let row_shape: Vec<usize> = data.meta.logical_shape[1..].to_vec();
    let row_size: usize = row_shape.iter().product();

    // Determine which rows to select — static or dynamic.
    let static_val = |v: Option<&Value>| -> Option<i64> {
        v.and_then(|val| if matches!(val, Value::None) { None } else { val.int_val() })
    };
    let is_present = |v: Option<&Value>| -> bool {
        matches!(v, Some(val) if !matches!(val, Value::None))
    };

    let start_s = static_val(start);
    let stop_s = static_val(stop);
    let step_s = static_val(step);

    let all_static = (!is_present(start) || start_s.is_some())
        && (!is_present(stop) || stop_s.is_some())
        && (!is_present(step) || step_s.is_some());

    if all_static {
        // Static path: compute row indices at compile time.
        let len = axis0_len as i64;
        let s = start_s.unwrap_or(0);
        let e = stop_s.unwrap_or(len);
        let st = step_s.unwrap_or(1);
        let s = if s < 0 { (len + s).max(0) } else { s.min(len) };
        let e = if e < 0 { (len + e).max(0) } else { e.min(len) };
        assert!(st != 0);

        let mut row_indices: Vec<i64> = Vec::new();
        if st > 0 {
            let mut i = s;
            while i < e { row_indices.push(i); i += st; }
        } else {
            let mut i = s;
            while i > e { row_indices.push(i); i += st; }
        }

        let num_rows = row_indices.len();
        let out_total = num_rows * row_size;
        let mut out_elements = Vec::with_capacity(out_total);

        for &row_i in &row_indices {
            let base = row_i * row_stride as i64;
            for j in 0..row_size {
                let addr = b.ir_constant_int(base + j as i64);
                let elem = b.ir_read_memory(data.segment_id, &addr);
                out_elements.push(crate::ops::dyn_ndarray::value_to_scalar_i64(&elem));
            }
        }

        let mut out_shape = vec![num_rows];
        out_shape.extend(&row_shape);
        let out_strides = row_major_strides(&out_shape);
        let segment_id = crate::helpers::segment::alloc_and_write(b, &out_elements, data.dtype);
        let envelope = crate::types::Envelope::from_static_shape(&mut b.dim_table, &out_shape);

        return Value::DynamicNDArray(DynamicNDArrayData {
            envelope, dtype: data.dtype, segment_id,
            meta: DynArrayMeta {
                logical_shape: out_shape.clone(), logical_offset: 0, logical_strides: out_strides.clone(),
                runtime_length: ScalarValue::new(Some(out_total as i64), None),
                runtime_rank: ScalarValue::new(Some(out_shape.len() as i64), None),
                runtime_shape: out_shape.iter().map(|&s| ScalarValue::new(Some(s as i64), None)).collect(),
                runtime_strides: out_strides.iter().map(|&s| ScalarValue::new(Some(s as i64), None)).collect(),
                runtime_offset: ScalarValue::new(Some(0), None),
            },
        });
    }

    // Dynamic path: pad-and-mask, selecting up to axis0_len rows.
    let max_rows = axis0_len;
    let max_out_total = max_rows * row_size;

    fn val_to_ir(b: &mut IRBuilder, v: Option<&Value>, default: i64) -> Value {
        match v {
            Some(val) if !matches!(val, Value::None) => {
                if let Some(s) = val.int_val() { b.ir_constant_int(s) } else { val.clone() }
            }
            _ => b.ir_constant_int(default),
        }
    }
    let start_ir = val_to_ir(b, start, 0);
    let stop_ir = val_to_ir(b, stop, axis0_len as i64);
    let step_ir = val_to_ir(b, step, 1);

    let default_val = crate::ops::dyn_ndarray::metadata::dyn_default_value(b, data.dtype);
    let zero = b.ir_constant_int(0);
    let len_val = b.ir_constant_int(axis0_len as i64);
    let mut out_elements = Vec::with_capacity(max_out_total);

    for row_i in 0..max_rows {
        let i_const = b.ir_constant_int(row_i as i64);
        let offset = b.ir_mul_i(&i_const, &step_ir);
        let src_row = b.ir_add_i(&start_ir, &offset);

        // In-bounds check (same as dyn_slice_1d).
        let ge_zero = b.ir_greater_than_or_equal_i(&src_row, &zero);
        let lt_len = b.ir_less_than_i(&src_row, &len_val);
        let in_range = b.ir_logical_and(&ge_zero, &lt_len);

        let step_pos = b.ir_greater_than_i(&step_ir, &zero);
        let lt_stop = b.ir_less_than_i(&src_row, &stop_ir);
        let gt_stop = b.ir_greater_than_i(&src_row, &stop_ir);
        let stop_ok = b.ir_select_i(&step_pos, &lt_stop, &gt_stop);
        let stop_bool = b.ir_bool_cast(&stop_ok);
        let in_bounds = b.ir_logical_and(&in_range, &stop_bool);

        // Clamp row index for safe read.
        let max_row = b.ir_constant_int(axis0_len as i64 - 1);
        let is_neg = b.ir_less_than_i(&src_row, &zero);
        let is_over = b.ir_greater_than_i(&src_row, &max_row);
        let clamped_hi = b.ir_select_i(&is_over, &max_row, &src_row);
        let clamped_row = b.ir_select_i(&is_neg, &zero, &clamped_hi);

        // Read all elements in this row, masked.
        let row_stride_val = b.ir_constant_int(row_stride as i64);
        let base = b.ir_mul_i(&clamped_row, &row_stride_val);

        for j in 0..row_size {
            let offset = b.ir_constant_int(j as i64);
            let addr = b.ir_add_i(&base, &offset);
            let elem = b.ir_read_memory(data.segment_id, &addr);
            let masked = if data.dtype == NumberType::Float {
                b.ir_select_f(&in_bounds, &elem, &default_val)
            } else {
                b.ir_select_i(&in_bounds, &elem, &default_val)
            };
            out_elements.push(crate::ops::dyn_ndarray::value_to_scalar_i64(&masked));
        }
    }

    // runtime row count
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
    let runtime_rows = b.ir_select_i(&diff_pos, &divided, &zero);
    let row_size_val = b.ir_constant_int(row_size as i64);
    let runtime_length = b.ir_mul_i(&runtime_rows, &row_size_val);

    let mut out_shape = vec![max_rows];
    out_shape.extend(&row_shape);
    let out_strides = row_major_strides(&out_shape);
    let segment_id = crate::helpers::segment::alloc_and_write(b, &out_elements, data.dtype);

    let mut dims: Vec<crate::types::Dim> = vec![
        crate::types::Dim::new_dynamic(&mut b.dim_table, 0, max_rows),
    ];
    for &s in &row_shape {
        dims.push(crate::types::Dim::new_static(&mut b.dim_table, s));
    }
    let envelope = crate::types::Envelope::new_with_bound(dims, max_out_total.min(data.envelope.total_bound));

    let runtime_len_sv = crate::ops::dyn_ndarray::value_to_scalar_i64(&runtime_length);
    let runtime_rows_sv = crate::ops::dyn_ndarray::value_to_scalar_i64(&runtime_rows);
    let mut rt_shape = vec![runtime_rows_sv];
    for &s in &row_shape {
        rt_shape.push(ScalarValue::new(Some(s as i64), None));
    }

    Value::DynamicNDArray(DynamicNDArrayData {
        envelope, dtype: data.dtype, segment_id,
        meta: DynArrayMeta {
            logical_shape: out_shape.clone(), logical_offset: 0, logical_strides: out_strides.clone(),
            runtime_length: runtime_len_sv,
            runtime_rank: ScalarValue::new(Some(out_shape.len() as i64), None),
            runtime_shape: rt_shape,
            runtime_strides: out_strides.iter().map(|&s| ScalarValue::new(Some(s as i64), None)).collect(),
            runtime_offset: ScalarValue::new(Some(0), None),
        },
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
        });
    }

    // Multi-dim: each index selects a row along axis 0.
    let row_stride = data.meta.logical_strides[0];
    let row_shape: Vec<usize> = data.meta.logical_shape[1..].to_vec();
    let row_size: usize = row_shape.iter().product();
    let num_indices = indices.len();
    let out_total = num_indices * row_size;
    let mut out_elements = Vec::with_capacity(out_total);

    for idx in &indices {
        let base = if let Some(i) = idx.int_val() {
            let i = if i < 0 { data.meta.logical_shape[0] as i64 + i } else { i };
            b.ir_constant_int(i * row_stride as i64)
        } else {
            let stride_val = b.ir_constant_int(row_stride as i64);
            b.ir_mul_i(idx, &stride_val)
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
    })
}

fn dyn_fancy_index_multidim(
    b: &mut IRBuilder, data: &DynamicNDArrayData, idx_arrays: &[&Value],
) -> Value {
    let strides = &data.meta.logical_strides;
    let shape = &data.meta.logical_shape;
    let arrays: Vec<Vec<Value>> = idx_arrays.iter().map(|v| extract_index_values(v)).collect();
    let out_len = arrays[0].len();
    assert!(arrays.iter().all(|a| a.len() == out_len), "fancy indexing: all index arrays must have the same length");

    let mut out_elements = Vec::with_capacity(out_len);
    for i in 0..out_len {
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
        let addr = if dynamic_parts.is_empty() {
            b.ir_constant_int(static_sum)
        } else {
            let mut acc = b.ir_constant_int(static_sum);
            for part in &dynamic_parts { acc = b.ir_add_i(&acc, part); }
            acc
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
    })
}

// ── Helpers ─────────────────────────────────────────────────────────────

pub(crate) fn compute_flat_addr(b: &mut IRBuilder, data: &DynamicNDArrayData, indices: &[SliceIndex]) -> Value {
    let strides = &data.meta.logical_strides;
    let shape = &data.meta.logical_shape;
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
