//! Dynamic array indexing and slicing: element access, range slicing,
//! and fancy indexing.
//!
//! Sibling modules:
//! * [`super::bounded_axis`] — stride-layout dispatch + strict-mode env var.
//! * [`super::array_index`] — how an array-valued index is read (mask or positions).

use crate::builder::IRBuilder;
use crate::helpers::shape_arith::row_major_strides;
use crate::types::{
    DynArrayMeta, DynamicNDArrayData, NumberType, ScalarValue, SliceIndex, Value, ValueId,
};

use super::array_index::{array_index_kind, ArrayIndexKind};
use super::bounded_axis::{bounded_axis_strict, select_stride_mode, stride_value, StrideMode};
use super::memory::filter;

// ── Main dispatch ───────────────────────────────────────────────────────

/// Subscript a DynamicNDArray with a list of SliceIndex values.
pub fn dyn_subscript(
    b: &mut IRBuilder,
    data: &DynamicNDArrayData,
    indices: &[SliceIndex],
) -> Value {
    // Ellipsis expands to the missing full ranges; NewAxis is stripped
    // here and re-inserted as a unit axis on the result.
    if indices
        .iter()
        .any(|i| matches!(i, SliceIndex::Ellipsis | SliceIndex::NewAxis))
    {
        let rank = data.envelope.rank();
        let consumed = indices
            .iter()
            .filter(|i| matches!(i, SliceIndex::Single(_) | SliceIndex::Range(_, _, _)))
            .count();
        let mut expanded: Vec<SliceIndex> = Vec::new();
        let mut newaxis_out: Vec<usize> = Vec::new();
        let mut out_axis = 0usize;
        let mut seen_ellipsis = false;
        for idx in indices {
            match idx {
                SliceIndex::Ellipsis => {
                    assert!(
                        !seen_ellipsis,
                        "an index can only have a single ellipsis ('...')"
                    );
                    seen_ellipsis = true;
                    for _ in 0..rank.saturating_sub(consumed) {
                        expanded.push(SliceIndex::Range(None, None, None));
                        out_axis += 1;
                    }
                }
                SliceIndex::NewAxis => {
                    newaxis_out.push(out_axis);
                    out_axis += 1;
                }
                SliceIndex::Single(_) => expanded.push(idx.clone()),
                SliceIndex::Range(_, _, _) => {
                    expanded.push(idx.clone());
                    out_axis += 1;
                }
            }
        }
        let mut cur = dyn_subscript(b, data, &expanded);
        for &pos in &newaxis_out {
            match &cur {
                Value::DynamicNDArray(d) => {
                    let pos_v = Value::Integer(ScalarValue::new(Some(pos as i64), None));
                    cur =
                        crate::ops::dyn_ndarray::reshape::dyn_expand_dims(b, d, &[pos_v]);
                }
                _ => panic!(
                    "DynamicNDArray: NewAxis on a scalar subscript result is not supported"
                ),
            }
        }
        return cur;
    }

    // A tuple index addresses one axis per entry (`a[t]` is
    // `a[t[0], t[1], ...]`); only a list is a fancy index.
    if let [SliceIndex::Single(Value::Tuple(tuple))] = indices {
        let expanded: Vec<SliceIndex> =
            tuple.values.iter().cloned().map(SliceIndex::Single).collect();
        return dyn_subscript(b, data, &expanded);
    }
    let rank = data.envelope.rank();
    if indices.len() > rank {
        panic!(
            "too many indices for array: array is {rank}-dimensional, but {} were indexed",
            indices.len()
        );
    }

    // Array-valued indices: a boolean mask, or integer index arrays.
    if let Some(out) = dyn_subscript_with_index_arrays(b, data, indices) {
        return out;
    }

    let all_single = indices.iter().all(|s| matches!(s, SliceIndex::Single(_)));

    if all_single && indices.len() == data.envelope.rank() {
        dyn_getitem_element(b, data, indices)
    } else if all_single && indices.len() == 1 {
        dyn_getitem_row(b, data, indices)
    } else if indices.len() == 1 {
        if let SliceIndex::Range(start, stop, step) = &indices[0] {
            // One range on a rank >= 2 array slices axis 0 (rows), not the
            // flattened buffer.
            if data.meta.logical_shape.len() >= 2 {
                dyn_slice_axis(b, data, 0, start.as_ref(), stop.as_ref(), step.as_ref())
            } else {
                dyn_slice_1d(b, data, start.as_ref(), stop.as_ref(), step.as_ref())
            }
        } else {
            panic!("dyn_subscript: unsupported single index type")
        }
    } else {
        dyn_subscript_multidim(b, data, indices)
    }
}

// ── Element access ──────────────────────────────────────────────────────

fn dyn_getitem_element(
    b: &mut IRBuilder,
    data: &DynamicNDArrayData,
    indices: &[SliceIndex],
) -> Value {
    let indices = wrap_and_check_dynamic_indices(b, data, indices, "dyn_getitem");
    let addr = compute_flat_addr(b, data, &indices);
    if data.envelope.total_bound > 0 {
        crate::optim::resolver::discharge_index_in_range(
            b,
            &addr,
            0,
            data.envelope.total_bound as i64,
            "dyn_getitem",
        );
    }
    crate::helpers::segment::read_typed(b, data.segment_id, &addr, data.dtype)
}

fn dyn_getitem_row(b: &mut IRBuilder, data: &DynamicNDArrayData, indices: &[SliceIndex]) -> Value {
    let rank = data.envelope.rank();
    if rank == 1 {
        return dyn_getitem_element(b, data, indices);
    }

    // The row index must be in range for the RUNTIME row count (it used
    // to be unchecked, so an out-of-range row read padding or another
    // row's cells).
    let wrapped = wrap_and_check_dynamic_indices(b, data, &indices[..1], "dyn_getitem_row");
    let idx_val = match &wrapped[0] {
        SliceIndex::Single(v) => v,
        _ => unreachable!(),
    };

    let out_shape: Vec<usize> = data.meta.logical_shape[1..].to_vec();
    let out_total: usize = out_shape.iter().product();
    let out_strides = row_major_strides(&out_shape);

    let mode = select_stride_mode(data);
    if rank > 2 && matches!(mode, StrideMode::SymbolicRuntime(_)) {
        panic!(
            "indexing one row of a compact runtime-shaped array of rank {rank} is not supported"
        );
    }
    let base = match &mode {
        StrideMode::LiteralLogical(strides) => {
            let row_stride = strides[0];
            if let Some(i) = idx_val.int_val() {
                let i = if i < 0 {
                    data.meta.logical_shape[0] as i64 + i
                } else {
                    i
                };
                b.ir_constant_int(i * row_stride as i64)
            } else {
                let stride_val = b.ir_constant_int(row_stride as i64);
                b.ir_mul_i(idx_val, &stride_val)
            }
        }
        StrideMode::SymbolicRuntime(_) => {
            let idx_v = if let Some(i) = idx_val.int_val() {
                let i = if i < 0 {
                    data.meta.logical_shape[0] as i64 + i
                } else {
                    i
                };
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
        let elem = crate::helpers::segment::read_typed(b, data.segment_id, &addr, data.dtype);
        out_elements.push(elem.clone());
    }

    let segment_id = crate::helpers::segment::alloc_and_write_values(b, &out_elements, data.dtype);
    let mut row_length = b.ir_constant_int(1);
    for dim in &data.meta.runtime_shape[1..] {
        let dim = scalar_i64_value(b, dim);
        row_length = b.ir_mul_i(&row_length, &dim);
    }
    let row_runtime_length = crate::ops::dyn_ndarray::value_to_scalar_i64(&row_length);
    let out_dims: Vec<crate::types::Dim> = data.envelope.dims[1..].to_vec();
    let envelope =
        crate::types::Envelope::new_with_bound(out_dims, out_total.min(data.envelope.total_bound));

    Value::DynamicNDArray(DynamicNDArrayData {
        envelope,
        dtype: data.dtype,
        segment_id,
        meta: DynArrayMeta {
            logical_shape: out_shape.clone(),
            logical_offset: 0,
            logical_strides: out_strides.clone(),
            // The row has the runtime shape of the remaining axes (the
            // copied envelope row is padded past it).
            runtime_length: row_runtime_length,
            runtime_rank: ScalarValue::new(Some(out_shape.len() as i64), None),
            runtime_shape: data.meta.runtime_shape[1..].to_vec(),
            runtime_strides: out_strides
                .iter()
                .map(|&s| ScalarValue::new(Some(s as i64), None))
                .collect(),
            runtime_offset: ScalarValue::new(Some(0), None),
        },
        value_id: ValueId::next(),
    })
}

// ── Range slicing ───────────────────────────────────────────────────────

/// A slice `start:stop:step` normalized against a runtime extent: the
/// first source index, the step, and the number of selected elements.
struct RuntimeSlice {
    start: Value,
    step: Value,
    length: Value,
}

/// The bounds of a slice `start:stop:step` as Python reads them against an
/// axis of `len` elements (`slice.indices`): missing bounds take the
/// defaults of the step's sign, negative bounds count from `len`, and both
/// are clamped into `[0, len]` (step > 0) or `[-1, len - 1]` (step < 0).
/// The selected indices are then `start, start + step, ...` while
/// `index < stop` (step > 0) or `index > stop` (step < 0), all inside
/// `[0, len)`. A zero step is rejected (statically, or by an in-circuit
/// assert). With `static_len`, a runtime bound the compiler can prove inside
/// the window is used as it is (no wrap or clamp to emit).
pub(crate) struct PythonSlice {
    pub start: Value,
    pub stop: Value,
    pub step: Value,
    /// The step's sign when known at compile time.
    pub step_positive: Option<bool>,
}

pub(crate) fn python_slice_bounds(
    b: &mut IRBuilder,
    start: Option<&Value>,
    stop: Option<&Value>,
    step: Option<&Value>,
    len: &Value,
    static_len: Option<usize>,
) -> PythonSlice {
    let is_present =
        |v: Option<&Value>| -> bool { matches!(v, Some(val) if !matches!(val, Value::None)) };
    let step_static = step.and_then(|v| if matches!(v, Value::None) { None } else { v.int_val() });

    let zero = b.ir_constant_int(0);
    let one = b.ir_constant_int(1);
    let neg_one = b.ir_constant_int(-1);
    let len_m1 = b.ir_sub_i(len, &one);

    // Step: static when possible; a runtime step is supported via
    // sign-dependent selects. step == 0 is rejected (statically or by an
    // in-circuit assert, mirroring python).
    let step_ir = match (is_present(step), step_static) {
        (false, _) => b.ir_constant_int(1),
        (true, Some(s)) => {
            assert!(s != 0, "slice step cannot be zero");
            b.ir_constant_int(s)
        }
        (true, None) => {
            let s = step.unwrap().clone();
            let nonzero = b.ir_not_equal_i(&s, &zero);
            b.ir_assert(&nonzero);
            s
        }
    };
    let step_positive: Option<bool> = step_static.map(|s| s > 0).or(if is_present(step) {
        None
    } else {
        Some(true)
    });
    // Step-sign-dependent quantities: fold statically when the sign is
    // known, otherwise select on the runtime sign bit.
    let by_sign = |b: &mut IRBuilder, pos: &Value, neg: &Value| -> Value {
        match step_positive {
            Some(true) => pos.clone(),
            Some(false) => neg.clone(),
            None => {
                let sp = b.ir_greater_than_i(&step_ir, &zero);
                b.ir_select_i(&sp, pos, neg)
            }
        }
    };
    let window_lo = by_sign(b, &zero, &neg_one);
    let window_hi = by_sign(b, len, &len_m1);
    let start_default = by_sign(b, &zero, &len_m1);
    let stop_default = by_sign(b, len, &neg_one);
    // The non-negative values that need neither the wrap nor the clamp.
    let identity_window = match (static_len, step_positive) {
        (Some(n), Some(true)) => Some(n as i64 + 1),
        (Some(n), Some(false)) => Some(n as i64),
        _ => None,
    };

    let normalize = |b: &mut IRBuilder, v: Option<&Value>, default: &Value| -> Value {
        let raw = match v {
            Some(val) if !matches!(val, Value::None) => {
                if let Some(s) = val.int_val() {
                    b.ir_constant_int(s)
                } else {
                    val.clone()
                }
            }
            _ => return default.clone(),
        };
        if identity_window.is_some_and(|hi| crate::optim::resolver::provably_in_range(b, &raw, 0, hi)) {
            return raw;
        }
        let shifted = b.ir_add_i(&raw, len);
        let is_neg = b.ir_less_than_i(&raw, &zero);
        let adjusted = b.ir_select_i(&is_neg, &shifted, &raw);
        let lt_lo = b.ir_less_than_i(&adjusted, &window_lo);
        let above = b.ir_select_i(&lt_lo, &window_lo, &adjusted);
        let gt_hi = b.ir_greater_than_i(&above, &window_hi);
        b.ir_select_i(&gt_hi, &window_hi, &above)
    };
    let start_ir = normalize(b, start, &start_default);
    let stop_ir = normalize(b, stop, &stop_default);
    PythonSlice { start: start_ir, stop: stop_ir, step: step_ir, step_positive }
}

/// Python's slice semantics against the runtime extent `len_rt` of the
/// sliced axis ([`python_slice_bounds`]) and the number of selected
/// elements, `max(0, ceil((stop - start) / step))`.
fn normalize_runtime_slice(
    b: &mut IRBuilder,
    start: Option<&Value>,
    stop: Option<&Value>,
    step: Option<&Value>,
    len_rt: &Value,
) -> RuntimeSlice {
    let slice = python_slice_bounds(b, start, stop, step, len_rt, None);
    let (start_ir, stop_ir, step_ir) = (slice.start, slice.stop, slice.step);
    let zero = b.ir_constant_int(0);
    let one = b.ir_constant_int(1);
    let by_sign = |b: &mut IRBuilder, pos: &Value, neg: &Value| -> Value {
        match slice.step_positive {
            Some(true) => pos.clone(),
            Some(false) => neg.clone(),
            None => {
                let sp = b.ir_greater_than_i(&step_ir, &zero);
                b.ir_select_i(&sp, pos, neg)
            }
        }
    };

    // out_len = max(0, ceil(diff / |step|)) with diff oriented by step sign.
    let stop_minus_start = b.ir_sub_i(&stop_ir, &start_ir);
    let start_minus_stop = b.ir_sub_i(&start_ir, &stop_ir);
    let diff = by_sign(b, &stop_minus_start, &start_minus_stop);
    let neg_step = b.ir_sub_i(&zero, &step_ir);
    let abs_step = by_sign(b, &step_ir, &neg_step);
    let abs_step_m1 = b.ir_sub_i(&abs_step, &one);
    let numerator = b.ir_add_i(&diff, &abs_step_m1);
    let divided = b.ir_floor_div_i(&numerator, &abs_step);
    let diff_pos = b.ir_greater_than_i(&diff, &zero);
    let runtime_length = b.ir_select_i(&diff_pos, &divided, &zero);

    RuntimeSlice { start: start_ir, step: step_ir, length: runtime_length }
}

fn dyn_slice_1d(
    b: &mut IRBuilder,
    data: &DynamicNDArrayData,
    start: Option<&Value>,
    stop: Option<&Value>,
    step: Option<&Value>,
) -> Value {
    let len = data.meta.logical_shape[0];

    let is_present =
        |v: Option<&Value>| -> bool { matches!(v, Some(val) if !matches!(val, Value::None)) };
    let static_val = |v: Option<&Value>| -> Option<i64> {
        v.and_then(|val| {
            if matches!(val, Value::None) {
                None
            } else {
                val.int_val()
            }
        })
    };

    let start_static = static_val(start);
    let stop_static = static_val(stop);
    let step_static = static_val(step);

    let all_static = (!is_present(start) || start_static.is_some())
        && (!is_present(stop) || stop_static.is_some())
        && (!is_present(step) || step_static.is_some());

    // The fully-static fast path is only sound when the array's runtime
    // length is itself a compile-time constant: slice defaults and negative
    // bounds are relative to the *actual* length, not the padded bound.
    if all_static && data.meta.runtime_length.static_val.is_some() {
        return dyn_slice_1d_static(b, data, start_static, stop_static, step_static);
    }

    // Dynamic path: pad-and-mask, with slice defaults / negative bounds /
    // clamping all relative to the array's *runtime* length (python slice
    // semantics: any integer bound is valid), not the padded bound.
    let max_out_len = len;
    let len_rt = Value::Integer(data.meta.runtime_length.clone());
    let RuntimeSlice { start: start_ir, step: step_ir, length: runtime_length } =
        normalize_runtime_slice(b, start, stop, step, &len_rt);
    let zero = b.ir_constant_int(0);

    let default_val = crate::ops::dyn_ndarray::metadata::dyn_default_value(b, data.dtype);
    let max_idx = b.ir_constant_int(len as i64 - 1);
    let mut out_elements = Vec::with_capacity(max_out_len);

    // View composition: if the input is itself a materialised slice of a
    // base segment (element j = base[off + (v.start + j*v.step)*stride])
    // and this slice has a static step, the composed source index is
    // v.start + v.step*start + (v.step*step)*i. When the composed start
    // resolves to a compile-time constant (the SMT resolver folds
    // cancelling dynamic terms, e.g. reversal of a to-the-end slice), every
    // element becomes a constant-address read of the base — no clamping,
    // no intermediate hop.
    let flat_input = data.meta.logical_strides == vec![1]
        && data.meta.logical_offset == 0
        && data.meta.logical_shape.len() == 1;
    let composed: Option<(crate::builder::SliceView, i64)> = match (
        flat_input,
        step_static.or(if is_present(step) { None } else { Some(1) }),
        b.provenance.slice_view(data.segment_id).cloned(),
    ) {
        (true, Some(step_c), Some(view)) if data.dtype != NumberType::Complex => {
            let vstep = b.ir_constant_int(view.step);
            let scaled_start = b.ir_mul_i(&start_ir, &vstep);
            let comp_start = b.ir_add_i(&view.start, &scaled_start);
            match crate::optim::resolver::require_static_int(
                b,
                &comp_start,
                crate::optim::resolver::SiteKind::SliceStart,
                None,
            ) {
                Ok(s0) => {
                    if std::env::var("ZINNIA_DEBUG_SLICE_VIEW").is_ok() {
                        eprintln!("[slice-view] composed start resolved to {}", s0.get());
                    }
                    Some((
                        crate::builder::SliceView {
                            start: b.ir_constant_int(s0.get()),
                            step: view.step * step_c,
                            ..view
                        },
                        s0.get(),
                    ))
                }
                Err(e) => {
                    if std::env::var("ZINNIA_DEBUG_SLICE_VIEW").is_ok() {
                        eprintln!("[slice-view] resolution failed: {e}");
                    }
                    None
                }
            }
        }
        _ => { if std::env::var("ZINNIA_DEBUG_SLICE_VIEW").is_ok() { eprintln!("[slice-view] no composition (flat/step/view gate)"); } None }
    };

    if let Some((cview, s0)) = &composed {
        for i in 0..max_out_len {
            let i_const = b.ir_constant_int(i as i64);
            let in_bounds = b.ir_less_than_i(&i_const, &runtime_length);
            let axis_idx = s0 + (i as i64) * cview.step;
            // Statically out-of-range source positions can only belong to
            // masked elements (valid i < out_len always maps in-range), so
            // they take the default without a read.
            let masked = if axis_idx >= 0 && (axis_idx as usize) < cview.axis_len {
                let slot = cview.base_offset + (axis_idx as usize) * cview.stride;
                let addr = b.ir_constant_int(slot as i64);
                let elem =
                    crate::helpers::segment::read_typed(b, cview.base_segment, &addr, data.dtype);
                if data.dtype == NumberType::Float {
                    b.ir_select_f(&in_bounds, &elem, &default_val)
                } else {
                    b.ir_select_i(&in_bounds, &elem, &default_val)
                }
            } else {
                default_val.clone()
            };
            out_elements.push(masked);
        }
    } else {
        for i in 0..max_out_len {
            let i_const = b.ir_constant_int(i as i64);
            // Element i is valid iff i < out_len; that single check subsumes
            // the start/stop/range conditions because the bounds are clamped.
            let in_bounds = b.ir_less_than_i(&i_const, &runtime_length);

            let offset = match step_static {
                Some(s) => b.ir_constant_int(i as i64 * s),
                None if !is_present(step) => b.ir_constant_int(i as i64),
                None => b.ir_mul_i(&i_const, &step_ir),
            };
            let src_idx = b.ir_add_i(&start_ir, &offset);
            let is_neg = b.ir_less_than_i(&src_idx, &zero);
            let is_over = b.ir_greater_than_i(&src_idx, &max_idx);
            let clamped_hi = b.ir_select_i(&is_over, &max_idx, &src_idx);
            let clamped = b.ir_select_i(&is_neg, &zero, &clamped_hi);

            let elem = crate::helpers::segment::read_typed(b, data.segment_id, &clamped, data.dtype);
            let masked = if data.dtype == NumberType::Float {
                b.ir_select_f(&in_bounds, &elem, &default_val)
            } else {
                b.ir_select_i(&in_bounds, &elem, &default_val)
            };
            out_elements.push(masked.clone());
        }
    }

    let segment_id = crate::helpers::segment::alloc_and_write_values(b, &out_elements, data.dtype);
    // Provenance for further composition: either the composed base view, or
    // this slice's own map over its (flat) input.
    if let Some((cview, _)) = composed {
        b.provenance.set_slice_view(segment_id, cview);
    } else if flat_input && data.dtype != NumberType::Complex {
        if let Some(step_c) = step_static.or(if is_present(step) { None } else { Some(1) }) {
            b.provenance.set_slice_view(
                segment_id,
                crate::builder::SliceView {
                    base_segment: data.segment_id,
                    axis_len: len,
                    stride: 1,
                    base_offset: 0,
                    start: start_ir.clone(),
                    step: step_c,
                },
            );
        }
    }
    let runtime_len_sv = crate::ops::dyn_ndarray::value_to_scalar_i64(&runtime_length);
    let envelope = crate::types::Envelope::new_with_bound(
        vec![crate::types::Dim::new_dynamic(
            &mut b.dim_table,
            0,
            max_out_len,
        )],
        max_out_len,
    );

    Value::DynamicNDArray(DynamicNDArrayData {
        envelope,
        dtype: data.dtype,
        segment_id,
        meta: DynArrayMeta {
            logical_shape: vec![max_out_len],
            logical_offset: 0,
            logical_strides: vec![1],
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
    start: Option<i64>,
    stop: Option<i64>,
    step: Option<i64>,
) -> Value {
    // Only sound when runtime_length is static (the caller guarantees it):
    // slice defaults and negative bounds are relative to the actual length.
    let len = data
        .meta
        .runtime_length
        .static_val
        .expect("dyn_slice_1d_static requires a static runtime length");
    let step_i = step.unwrap_or(1);
    assert!(step_i != 0, "slice step cannot be zero");
    // Python slice-bound defaults depend on step sign:
    //   step > 0 → start=0, stop=len
    //   step < 0 → start=len-1, stop=-1 (i.e. "before index 0")
    let start_i = start.unwrap_or(if step_i > 0 { 0 } else { len - 1 });
    let stop_i = stop.unwrap_or(if step_i > 0 { len } else { -1 });
    let (lo, hi) = if step_i > 0 { (0, len) } else { (-1, len - 1) };
    let start_i = if start.is_some() && start_i < 0 {
        (len + start_i).clamp(lo, hi)
    } else {
        start_i.clamp(lo, hi)
    };
    let stop_i = if stop.is_some() && stop_i < 0 {
        (len + stop_i).clamp(lo, hi)
    } else if stop.is_some() {
        stop_i.clamp(lo, hi)
    } else {
        stop_i
    };

    let mut src_indices: Vec<i64> = Vec::new();
    if step_i > 0 {
        let mut i = start_i;
        while i < stop_i {
            src_indices.push(i);
            i += step_i;
        }
    } else {
        let mut i = start_i;
        while i > stop_i {
            src_indices.push(i);
            i += step_i;
        }
    }

    let out_len = src_indices.len();
    let mut out_elements = Vec::with_capacity(out_len);
    for &src_i in &src_indices {
        let addr = b.ir_constant_int(src_i);
        let elem = crate::helpers::segment::read_typed(b, data.segment_id, &addr, data.dtype);
        out_elements.push(elem.clone());
    }

    let segment_id = crate::helpers::segment::alloc_and_write_values(b, &out_elements, data.dtype);
    let envelope = crate::types::Envelope::from_static_shape(&mut b.dim_table, &[out_len]);

    Value::DynamicNDArray(DynamicNDArrayData {
        envelope,
        dtype: data.dtype,
        segment_id,
        meta: DynArrayMeta {
            logical_shape: vec![out_len],
            logical_offset: 0,
            logical_strides: vec![1],
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
    b: &mut IRBuilder,
    data: &DynamicNDArrayData,
    indices: &[SliceIndex],
) -> Value {
    // Single indices wrap against, and are checked below, the runtime
    // extent of their axis (NumPy's IndexError has no proof), not the
    // buffer width: `x[:, -1]` of a 2x2 array in a 4x3 buffer is column 1.
    let indices: &Vec<SliceIndex> = &indices
        .iter()
        .enumerate()
        .map(|(axis, idx)| match idx {
            SliceIndex::Single(raw @ Value::Integer(_)) => {
                let hi = scalar_i64_value(b, &data.meta.runtime_shape[axis]);
                let wrapped = wrap_negative_index(b, raw, &hi);
                SliceIndex::Single(crate::optim::resolver::discharge_index_below_value(
                    b, &wrapped, 0, &hi, "dyn_subscript",
                ))
            }
            other => other.clone(),
        })
        .collect();
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

    // A Range with static bounds is still RUNTIME-dependent on an axis
    // whose runtime extent is dynamic, unless it is the whole axis: the
    // bounds clamp to the runtime extent, and negative steps and bounds
    // resolve against it, not the envelope width (`a[0:3, :]` on 2 runtime
    // rows has 2 rows; `m[:, ::-1]` on a runtime-ncols array reverses the
    // first `ncols` cells, binding-audit case 0419).
    let runtime_dependent = indices.iter().enumerate().find(|(ax, idx)| {
        if let SliceIndex::Range(start, stop, step) = idx {
            let static_of = |v: &Option<Value>| -> Option<i64> {
                match v.as_ref() {
                    Some(Value::None) | None => None,
                    Some(val) => val.int_val(),
                }
            };
            let unset = |v: &Option<Value>| matches!(v, None | Some(Value::None));
            let whole_axis = (unset(start) || static_of(start) == Some(0))
                && unset(stop)
                && (unset(step) || static_of(step) == Some(1));
            let axis_runtime_dynamic = data
                .meta
                .runtime_shape
                .get(*ax)
                .is_some_and(|sv| sv.static_val.is_none());
            axis_runtime_dynamic && !whole_axis
        } else {
            false
        }
    });
    if let Some((dyn_ax, _)) = runtime_dependent {
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
    let remaining: Vec<SliceIndex> = indices
        .iter()
        .enumerate()
        .map(|(i, idx)| {
            if i == dyn_ax {
                SliceIndex::Range(None, None, None)
            } else {
                idx.clone()
            }
        })
        .collect();

    let all_trivial = remaining.iter().all(|idx| match idx {
        SliceIndex::Range(s, e, st) => {
            matches!(s, None | Some(Value::None))
                && matches!(e, None | Some(Value::None))
                && matches!(st, None | Some(Value::None))
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
        return out;
    }

    let mode = select_stride_mode(data);
    let axis_len = shape[axis];
    let max_slice_len = axis_len; // upper bound on output size along this axis

    // Python slice semantics against the RUNTIME extent of this axis (the
    // envelope width is padding): bounds are wrapped and clamped, so every
    // selected index lies inside the runtime extent and the selected count
    // never exceeds it.
    let axis_rt_sv = data.meta.runtime_shape.get(axis).cloned();
    let axis_len_rt: Value = match &axis_rt_sv {
        Some(sv) if sv.static_val.is_none() => Value::Integer(sv.clone()),
        Some(sv) => b.ir_constant_int(sv.static_val.unwrap()),
        None => b.ir_constant_int(axis_len as i64),
    };
    let RuntimeSlice { start: start_ir, step: step_ir, length: runtime_axis_len } =
        normalize_runtime_slice(b, start, stop, step, &axis_len_rt);

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
    let axis_stride_val = stride_value(b, &mode, axis);

    let mut out_elements = Vec::with_capacity(out_total);

    for flat_out in 0..out_total {
        let out_coords =
            crate::helpers::shape_arith::decode_coords(flat_out, &out_shape, &out_strides);

        // The coordinate along the sliced axis is the slice index.
        let slice_idx = out_coords[axis];
        let slice_idx_val = b.ir_constant_int(slice_idx as i64);

        // Source index along the sliced axis: start + slice_idx * step.
        let offset = b.ir_mul_i(&slice_idx_val, &step_ir);
        let src_axis_idx = b.ir_add_i(&start_ir, &offset);

        // Output row `slice_idx` holds data iff it is below the selected
        // count; its source index is then inside the runtime extent.
        let in_bounds = b.ir_less_than_i(&slice_idx_val, &runtime_axis_len);

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

        let elem = crate::helpers::segment::read_typed(b, data.segment_id, &addr, data.dtype);
        let masked = if data.dtype == NumberType::Float {
            b.ir_select_f(&in_bounds, &elem, &default_val)
        } else {
            b.ir_select_i(&in_bounds, &elem, &default_val)
        };
        out_elements.push(masked.clone());
    }


    // Build output metadata.
    let segment_id = crate::helpers::segment::alloc_and_write_values(b, &out_elements, data.dtype);

    let mut dims: Vec<crate::types::Dim> = Vec::with_capacity(rank);
    let mut rt_shape: Vec<ScalarValue<i64>> = Vec::with_capacity(rank);
    for ax in 0..rank {
        if ax == axis {
            dims.push(crate::types::Dim::new_dynamic(
                &mut b.dim_table,
                0,
                max_slice_len,
            ));
            rt_shape.push(crate::ops::dyn_ndarray::value_to_scalar_i64(
                &runtime_axis_len,
            ));
        } else {
            // Non-sliced axes keep their (possibly runtime) extents —
            // collapsing them to the envelope width here is what leaked
            // static skeleton dims into output metadata.
            dims.push(data.envelope.dims[ax]);
            rt_shape.push(
                data.meta
                    .runtime_shape
                    .get(ax)
                    .cloned()
                    .unwrap_or_else(|| ScalarValue::new(Some(shape[ax] as i64), None)),
            );
        }
    }
    let envelope =
        crate::types::Envelope::new_with_bound(dims, out_total.min(data.envelope.total_bound));

    // Runtime total length = product of runtime extents.
    let mut runtime_length = runtime_axis_len.clone();
    for (ax, sv) in rt_shape.iter().enumerate() {
        if ax == axis {
            continue;
        }
        let dim_v: Value = if let Some(c) = sv.static_val {
            b.ir_constant_int(c)
        } else {
            Value::Integer(sv.clone())
        };
        runtime_length = b.ir_mul_i(&runtime_length, &dim_v);
    }
    let runtime_len_sv = crate::ops::dyn_ndarray::value_to_scalar_i64(&runtime_length);

    let out = Value::DynamicNDArray(DynamicNDArrayData {
        envelope,
        dtype: data.dtype,
        segment_id,
        meta: DynArrayMeta {
            logical_shape: out_shape.clone(),
            logical_offset: 0,
            logical_strides: out_strides.clone(),
            runtime_length: runtime_len_sv,
            runtime_rank: ScalarValue::new(Some(rank as i64), None),
            runtime_shape: rt_shape,
            runtime_strides: out_strides
                .iter()
                .map(|&s| ScalarValue::new(Some(s as i64), None))
                .collect(),
            runtime_offset: ScalarValue::new(Some(0), None),
        },
        value_id: ValueId::next(),
    });
    out
}

/// Static multi-dim subscript: all Range bounds are compile-time known.
/// Uses direct coordinate-based reads.
fn dyn_subscript_multidim_static(
    b: &mut IRBuilder,
    data: &DynamicNDArrayData,
    indices: &[SliceIndex],
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
                let given = |v: &Option<Value>| -> Option<i64> {
                    match v.as_ref() {
                        Some(Value::None) | None => None,
                        Some(val) => val.int_val(),
                    }
                };
                let st = given(step).unwrap_or(1);
                assert!(st != 0, "slice step cannot be zero");
                // Python slice-bound defaults and clamp windows depend on
                // the step sign (step < 0 walks dim-1 down to "before 0").
                // The old sign-agnostic defaults made `[::-1]` produce an
                // EMPTY axis (binding-audit case 0419).
                let (lo, hi) = if st > 0 { (0i64, dim) } else { (-1i64, dim - 1) };
                let s = match given(start) {
                    Some(v) if v < 0 => (dim + v).clamp(lo, hi),
                    Some(v) => v.clamp(lo, hi),
                    None => {
                        if st > 0 {
                            0
                        } else {
                            dim - 1
                        }
                    }
                };
                let e = match given(stop) {
                    Some(v) if v < 0 => (dim + v).clamp(lo, hi),
                    Some(v) => v.clamp(lo, hi),
                    None => {
                        if st > 0 {
                            dim
                        } else {
                            -1
                        }
                    }
                };
                let mut coords: Vec<usize> = Vec::new();
                let mut i = s;
                if st > 0 {
                    while i < e {
                        coords.push(i as usize);
                        i += st;
                    }
                } else {
                    while i > e {
                        coords.push(i as usize);
                        i += st;
                    }
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
                for part in &dynamic_offset_parts {
                    acc = b.ir_add_i(&acc, part);
                }
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
        let out_coords =
            crate::helpers::shape_arith::decode_coords(flat_out, &out_shape, &out_strides_out);
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
        let elem = crate::helpers::segment::read_typed(b, data.segment_id, &addr, data.dtype);
        out_elements.push(elem.clone());
    }

    if out_shape.is_empty() {
        return crate::helpers::segment::read_typed(b, data.segment_id, &base_offset, data.dtype);
    }

    let segment_id = crate::helpers::segment::alloc_and_write_values(b, &out_elements, data.dtype);

    // Every Range reaching this path on an axis with a runtime extent is the
    // whole axis (see `dyn_subscript_multidim`), so the output axis has that
    // runtime extent; the output buffer is padded (static strides of
    // `out_shape`), its cells past the extent unused.
    let runtime_shape: Vec<ScalarValue<i64>> = axis_ranges
        .iter()
        .map(|(src_ax, coords)| match data.meta.runtime_shape.get(*src_ax) {
            Some(extent) if extent.static_val.is_none() && coords.len() == shape[*src_ax] => extent.clone(),
            _ => ScalarValue::new(Some(coords.len() as i64), None),
        })
        .collect();
    let dims: Vec<crate::types::Dim> = runtime_shape
        .iter()
        .zip(&out_shape)
        .map(|(extent, &width)| match extent.static_val {
            Some(_) => crate::types::Dim::new_static(&mut b.dim_table, width),
            None => crate::types::Dim::new_dynamic(&mut b.dim_table, 0, width),
        })
        .collect();
    let envelope = crate::types::Envelope::new_with_bound(dims, out_total);
    let runtime_length = if runtime_shape.iter().all(|e| e.static_val.is_some()) {
        ScalarValue::new(Some(out_total as i64), None)
    } else {
        let mut length = b.ir_constant_int(1);
        for extent in &runtime_shape {
            let extent = scalar_i64_value(b, extent);
            length = b.ir_mul_i(&length, &extent);
        }
        crate::ops::dyn_ndarray::value_to_scalar_i64(&length)
    };

    Value::DynamicNDArray(DynamicNDArrayData {
        envelope,
        dtype: data.dtype,
        segment_id,
        meta: DynArrayMeta {
            logical_shape: out_shape.clone(),
            logical_offset: 0,
            logical_strides: out_strides_out.clone(),
            runtime_length,
            runtime_rank: ScalarValue::new(Some(out_shape.len() as i64), None),
            runtime_shape,
            runtime_strides: out_strides_out
                .iter()
                .map(|&s| ScalarValue::new(Some(s as i64), None))
                .collect(),
            runtime_offset: ScalarValue::new(Some(0), None),
        },
        value_id: ValueId::next(),
    })
}

// ── Array-valued indices ────────────────────────────────────────────────

/// `data[indices]` when an index is array-valued (`array_index_kind`): a
/// boolean mask keeps the elements where it is true (`filter`); integer
/// index arrays gather (`dyn_gather_rows`, `dyn_gather_elements`). `None`
/// when no index is array-valued.
fn dyn_subscript_with_index_arrays(
    b: &mut IRBuilder,
    data: &DynamicNDArrayData,
    indices: &[SliceIndex],
) -> Option<Value> {
    let kinds: Vec<Option<ArrayIndexKind>> = indices
        .iter()
        .map(|idx| match idx {
            SliceIndex::Single(v) => array_index_kind(b, v),
            _ => None,
        })
        .collect();
    if kinds.iter().all(Option::is_none) {
        return None;
    }
    let arrays: Vec<&Value> = indices
        .iter()
        .filter_map(|idx| match idx {
            SliceIndex::Single(v) => Some(v),
            _ => None,
        })
        .collect();
    match kinds.as_slice() {
        [Some(ArrayIndexKind::Mask)] => {
            Some(filter(b, &Value::DynamicNDArray(data.clone()), &[arrays[0].clone()]))
        }
        [Some(ArrayIndexKind::Positions)] => Some(dyn_gather_rows(b, data, arrays[0])),
        _ if kinds.iter().all(|k| *k == Some(ArrayIndexKind::Positions)) => {
            Some(dyn_gather_elements(b, data, &arrays))
        }
        _ => panic!(
            "indexing a runtime-shaped array with index arrays mixed with slices, \
             scalars or boolean masks is not supported"
        ),
    }
}

/// The entries of an integer index array, with the shape NumPy gives it and
/// its runtime number of entries: all of them for a list or static array; for
/// a runtime-shaped (rank-one) array, its first `len` slots, the others
/// being padding.
struct IndexArray {
    entries: Vec<Value>,
    shape: Vec<usize>,
    /// `None` when every entry is in use.
    runtime_len: Option<Value>,
    /// The envelope dims of a runtime-shaped index array (its shape's
    /// symbolic length), reused by the result.
    dims: Option<Vec<crate::types::Dim>>,
}

fn index_array(b: &mut IRBuilder, idx: &Value) -> IndexArray {
    match idx {
        Value::List(_) | Value::Tuple(_) => IndexArray {
            entries: crate::helpers::composite::flatten_composite(idx)
                .iter()
                .map(|v| integer_index(b, v))
                .collect(),
            shape: crate::helpers::composite::get_composite_shape(idx),
            runtime_len: None,
            dims: None,
        },
        Value::StaticArray { shape, .. } => IndexArray {
            entries: crate::helpers::static_array::elementwise::payload_cells(b, idx)
                .iter()
                .map(|v| integer_index(b, v))
                .collect(),
            shape: shape.clone(),
            runtime_len: None,
            dims: None,
        },
        Value::DynamicNDArray(d) => {
            assert!(
                d.envelope.rank() == 1,
                "indexing with a runtime-shaped index array of rank {} is not supported",
                d.envelope.rank()
            );
            let d = crate::ops::dyn_ndarray::force_contiguous(b, d);
            let len = d.meta.runtime_length.clone();
            let all_in_use = len.static_val == Some(d.max_length() as i64);
            IndexArray {
                entries: crate::helpers::segment::read_all_typed(
                    b,
                    d.segment_id,
                    d.max_length(),
                    d.dtype,
                ),
                shape: vec![d.max_length()],
                runtime_len: (!all_in_use).then(|| scalar_i64_value(b, &len)),
                dims: Some(d.envelope.dims.clone()),
            }
        }
        _ => unreachable!("array_index_kind classified a non-array index"),
    }
}

/// An entry of an integer index array as an integer (NumPy converts the
/// index list `[True, 2]` to the integer array `[1, 2]`).
fn integer_index(b: &mut IRBuilder, v: &Value) -> Value {
    match v {
        Value::Boolean(_) => b.ir_int_cast(v),
        _ => v.clone(),
    }
}

/// Run `f` for slot `k` of an index array: where the slot is padding
/// (`k >= len`), its index is not checked and the access reads row 0.
fn for_index_slot<T>(
    b: &mut IRBuilder,
    index: &IndexArray,
    k: usize,
    f: impl FnOnce(&mut IRBuilder) -> T,
) -> T {
    match &index.runtime_len {
        None => f(b),
        Some(len) => {
            let k_v = b.ir_constant_int(k as i64);
            let in_use = b.ir_less_than_i(&k_v, len);
            b.under_condition(&in_use, f)
        }
    }
}

/// `data[idx]` for an integer index array `idx`: NumPy integer-array
/// indexing on axis 0. Entry `k` of `idx` selects row `idx[k]` of `data`,
/// counted from the end when negative, and the result has shape
/// `idx.shape + data.shape[1:]`. Every index in use must lie in
/// `[-n, n)` for the runtime length `n` of axis 0
/// (`wrap_and_check_dynamic_indices`, checked where the access runs).
///
/// When the facts prove every index of a fully used `idx` in `[0, n)` and
/// `n` is a compile-time constant, the checks are skipped
/// (`gather_skip_bounds_check`).
fn dyn_gather_rows(b: &mut IRBuilder, data: &DynamicNDArrayData, idx: &Value) -> Value {
    let axis0_static = data.meta.runtime_shape[0].static_val
        == Some(data.meta.logical_shape[0] as i64);
    let fully_used = !matches!(idx, Value::DynamicNDArray(d)
        if d.meta.runtime_length.static_val != Some(d.max_length() as i64));
    if idx.value_id().is_none() || !(axis0_static && fully_used) {
        return dyn_gather_rows_lower(b, data, idx, true);
    }
    // The array facts of the indices' current contents decide whether every
    // one is in `[0, n)` (`builder/array_facts.rs`).
    let in_range = b.array_within(idx, 0, data.meta.logical_shape[0] as i64 - 1);
    let inputs = DynGatherInputs { data: data.clone(), idx: idx.clone() };
    let skip = crate::optim::OpStrategy {
        name: "gather_skip_bounds_check",
        precondition: crate::optim::decided(),
        lower: |b, inputs: &DynGatherInputs| dyn_gather_rows_lower(b, &inputs.data, &inputs.idx, false),
    };
    let set = crate::optim::OpStrategySet {
        strategies: if in_range { vec![skip] } else { Vec::new() },
        default: |b, inputs: &DynGatherInputs| {
            dyn_gather_rows_lower(b, &inputs.data, &inputs.idx, true)
        },
    };
    crate::optim::dispatch_strategy(b, "dyn_fancy_index", &inputs, &set)
}

#[derive(Clone)]
struct DynGatherInputs {
    data: DynamicNDArrayData,
    idx: Value,
}

fn dyn_gather_rows_lower(
    b: &mut IRBuilder,
    data: &DynamicNDArrayData,
    idx: &Value,
    check_bounds: bool,
) -> Value {
    let index = index_array(b, idx);
    let data = &crate::ops::dyn_ndarray::force_contiguous(b, data);
    let mode = select_stride_mode(data);
    if data.envelope.rank() >= 2 && matches!(mode, StrideMode::SymbolicRuntime(_)) {
        panic!("indexing the rows of a compact runtime-shaped array with an index array is not supported");
    }
    let row_shape: Vec<usize> = data.meta.logical_shape[1..].to_vec();
    let row_size: usize = row_shape.iter().product();

    let mut cells = Vec::with_capacity(index.entries.len() * row_size);
    for (k, entry) in index.entries.iter().enumerate() {
        let row_cells = for_index_slot(b, &index, k, |b| {
            let row = if check_bounds {
                checked_index(b, data, 0, entry)
            } else {
                entry.clone()
            };
            let base = row_base_address(b, &mode, &row);
            (0..row_size)
                .map(|j| {
                    let offset = b.ir_constant_int(j as i64);
                    let addr = b.ir_add_i(&base, &offset);
                    crate::helpers::segment::read_typed(b, data.segment_id, &addr, data.dtype)
                })
                .collect::<Vec<_>>()
        });
        cells.extend(row_cells);
    }
    let segment_id = crate::helpers::segment::alloc_and_write_values(b, &cells, data.dtype);

    // Result shape: the index array's shape, then a row of `data`.
    let mut out_shape = index.shape.clone();
    out_shape.extend(&row_shape);
    let mut dims = match &index.dims {
        Some(dims) => dims.clone(),
        None => index
            .shape
            .iter()
            .map(|&n| crate::types::Dim::new_static(&mut b.dim_table, n))
            .collect(),
    };
    dims.extend(data.envelope.dims[1..].iter().cloned());
    let mut runtime_shape: Vec<ScalarValue<i64>> = match &index.runtime_len {
        Some(len) => vec![crate::ops::dyn_ndarray::value_to_scalar_i64(len)],
        None => index.shape.iter().map(|&n| ScalarValue::new(Some(n as i64), None)).collect(),
    };
    runtime_shape.extend(data.meta.runtime_shape[1..].iter().cloned());
    dyn_array_with_shape(b, segment_id, data.dtype, out_shape, dims, runtime_shape)
}

/// `data[i0, i1, ...]` with one integer index array per leading axis, all of
/// one length `L`: entry `k` of the result is
/// `data[i0[k], i1[k], ...]`, each index wrapped and checked against its
/// axis's runtime length. Index arrays of different shapes (NumPy would
/// broadcast them) are rejected.
fn dyn_gather_elements(b: &mut IRBuilder, data: &DynamicNDArrayData, arrays: &[&Value]) -> Value {
    let axes: Vec<IndexArray> = arrays.iter().map(|v| index_array(b, v)).collect();
    let shape = axes[0].shape.clone();
    assert!(
        axes.iter().all(|a| a.shape == shape && a.runtime_len.is_none()),
        "indexing a runtime-shaped array with several index arrays requires index arrays \
         of one static shape"
    );
    assert!(
        arrays.len() == data.envelope.rank(),
        "indexing a runtime-shaped array with fewer index arrays than axes is not supported"
    );
    let cells: Vec<Value> = (0..axes[0].entries.len())
        .map(|k| {
            let element: Vec<SliceIndex> =
                axes.iter().map(|a| SliceIndex::Single(a.entries[k].clone())).collect();
            dyn_getitem_element(b, data, &element)
        })
        .collect();
    let segment_id = crate::helpers::segment::alloc_and_write_values(b, &cells, data.dtype);
    let dims = shape
        .iter()
        .map(|&n| crate::types::Dim::new_static(&mut b.dim_table, n))
        .collect();
    let runtime_shape = shape.iter().map(|&n| ScalarValue::new(Some(n as i64), None)).collect();
    dyn_array_with_shape(b, segment_id, data.dtype, shape, dims, runtime_shape)
}

/// Index `idx` on `axis`, wrapped (NumPy's one negative wrap) and checked
/// against the axis's runtime length where the access runs.
fn checked_index(b: &mut IRBuilder, data: &DynamicNDArrayData, axis: usize, idx: &Value) -> Value {
    let mut padded = vec![SliceIndex::Range(None, None, None); axis];
    padded.push(SliceIndex::Single(idx.clone()));
    match wrap_and_check_dynamic_indices(b, data, &padded, "fancy_index").pop() {
        Some(SliceIndex::Single(v)) => v,
        _ => unreachable!(),
    }
}

/// The flat address of the first cell of row `row` (already wrapped and in
/// range) of a contiguous array with stride mode `mode`.
fn row_base_address(b: &mut IRBuilder, mode: &StrideMode, row: &Value) -> Value {
    let stride = match mode {
        StrideMode::LiteralLogical(strides) => b.ir_constant_int(strides[0] as i64),
        StrideMode::SymbolicRuntime(_) => stride_value(b, mode, 0),
    };
    match (row.int_val(), stride.int_val()) {
        (Some(r), Some(s)) => b.ir_constant_int(r * s),
        _ => b.ir_mul_i(row, &stride),
    }
}

/// A fresh, contiguous `DynamicNDArray` over `segment_id` with padded
/// shape `shape`, envelope `dims` and runtime shape `runtime_shape`.
fn dyn_array_with_shape(
    b: &mut IRBuilder,
    segment_id: u32,
    dtype: NumberType,
    shape: Vec<usize>,
    dims: Vec<crate::types::Dim>,
    runtime_shape: Vec<ScalarValue<i64>>,
) -> Value {
    let total: usize = shape.iter().product();
    let strides = row_major_strides(&shape);
    let mut runtime_length = b.ir_constant_int(1);
    for dim in &runtime_shape {
        let dim = scalar_i64_value(b, dim);
        runtime_length = b.ir_mul_i(&runtime_length, &dim);
    }
    Value::DynamicNDArray(DynamicNDArrayData {
        envelope: crate::types::Envelope::new_with_bound(dims, total),
        dtype,
        segment_id,
        meta: DynArrayMeta {
            logical_shape: shape.clone(),
            logical_offset: 0,
            logical_strides: strides.clone(),
            runtime_length: crate::ops::dyn_ndarray::value_to_scalar_i64(&runtime_length),
            runtime_rank: ScalarValue::new(Some(shape.len() as i64), None),
            runtime_shape,
            runtime_strides: strides
                .iter()
                .map(|&s| ScalarValue::new(Some(s as i64), None))
                .collect(),
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
pub(crate) fn compute_flat_addr(
    b: &mut IRBuilder,
    data: &DynamicNDArrayData,
    indices: &[SliceIndex],
) -> Value {
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
                for part in &dynamic_parts {
                    acc = b.ir_add_i(&acc, part);
                }
                acc
            }
        }
        StrideMode::SymbolicRuntime(_) => {
            // Compact-buffer mode: every axis pays an ir_mul_i with the
            // SSA stride. Static indices still bypass `ir_int_cast`-style
            // wrapping by going through `ir_constant_int(i)`, but the
            // resulting multiplication cannot be constant-folded.
            let strict = bounded_axis_strict();
            let zero = if strict {
                Some(b.ir_constant_int(0))
            } else {
                None
            };

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

pub(crate) fn scalar_i64_value(b: &mut IRBuilder, sv: &ScalarValue<i64>) -> Value {
    if let Some(v) = sv.static_val {
        b.ir_constant_int(v)
    } else if sv.stmt_id.is_some() {
        Value::Integer(sv.clone())
    } else {
        b.ir_constant_int(0)
    }
}

/// Wrap each `Single` index against the RUNTIME size of its axis and
/// require it in range: NumPy reads `x[-1]` of a runtime-shaped array as
/// its last runtime element, so a negative index becomes
/// `i + runtime_shape[ax]` (it used to be wrapped with the envelope size,
/// pointing into the padding), and `0 <= wrapped < runtime_shape[ax]` is
/// discharged (no IR when the facts prove it, an in-circuit check
/// otherwise). Returns the wrapped indices for address computation (each
/// as returned by the discharge: in range even where the access does not
/// run).
pub(crate) fn wrap_and_check_dynamic_indices(
    b: &mut IRBuilder,
    data: &DynamicNDArrayData,
    indices: &[SliceIndex],
    op_name: &'static str,
) -> Vec<SliceIndex> {
    indices
        .iter()
        .enumerate()
        .map(|(axis, idx)| {
            let SliceIndex::Single(raw) = idx else {
                return idx.clone();
            };
            let hi = scalar_i64_value(b, &data.meta.runtime_shape[axis]);
            let wrapped = wrap_negative_index(b, raw, &hi);
            SliceIndex::Single(crate::optim::resolver::discharge_index_below_value(
                b, &wrapped, 0, &hi, op_name,
            ))
        })
        .collect()
}

/// `idx + len` for a negative `idx`, else `idx` (one NumPy wrap).
fn wrap_negative_index(b: &mut IRBuilder, idx: &Value, len: &Value) -> Value {
    if let Some(i) = idx.int_val() {
        if i >= 0 {
            return idx.clone();
        }
        return match len.int_val() {
            Some(n) => b.ir_constant_int(n + i),
            None => {
                let offset = b.ir_constant_int(i);
                b.ir_add_i(len, &offset)
            }
        };
    }
    if b.proves_at_least(idx, 0) {
        return idx.clone();
    }
    let zero = b.ir_constant_int(0);
    let is_negative = b.ir_less_than_i(idx, &zero);
    let shifted = b.ir_add_i(idx, len);
    b.ir_select_i(&is_negative, &shifted, idx)
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

    fn make_2d_dyn(b: &mut IRBuilder, rows: usize, cols: usize) -> Value {
        let zero = value_to_scalar_i64(&b.ir_constant_int(0));
        let values = vec![zero; rows * cols];
        let runtime_shape = vec![
            ScalarValue::new(Some(rows as i64), None),
            ScalarValue::new(Some(cols as i64), None),
        ];
        let runtime_length = b.ir_constant_int((rows * cols) as i64);
        dyn_from_values_with_active_nd(
            b,
            values,
            vec![rows, cols],
            runtime_shape,
            runtime_length,
            NumberType::Integer,
        )
    }

    fn make_dynamic_index_list(b: &mut IRBuilder, len: usize) -> Value {
        let values: Vec<Value> = (0..len)
            .map(|i| {
                b.ir_read_integer(
                    crate::circuit_input::InputPath::new(format!("idx{i}"), vec![]),
                    false,
                )
            })
            .collect();
        let types = values.iter().map(|v| v.zinnia_type()).collect();
        Value::List(CompositeData::new(types, values))
    }

    /// Bounds checks: an index check is one `AssertU128Lt` (or an `Assert`
    /// of a comparison chain for a non-zero lower bound).
    fn count_asserts(irs: &[IR]) -> usize {
        irs.iter().filter(|ir| matches!(ir, IR::Assert | IR::AssertU128Lt)).count()
    }

    struct ScopedTelemetryDir {
        dir: std::path::PathBuf,
        previous: Option<String>,
    }

    impl ScopedTelemetryDir {
        fn enabled() -> Self {
            let previous = std::env::var("ZINNIA_TELEMETRY_DIR").ok();
            let nonce = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0);
            let dir = std::env::temp_dir()
                .join(format!("zinnia-gather-tel-{}-{nonce}", std::process::id()));
            std::fs::create_dir_all(&dir).expect("tmp telemetry dir creatable");
            std::env::set_var("ZINNIA_TELEMETRY_DIR", &dir);
            Self { dir, previous }
        }
    }

    impl Drop for ScopedTelemetryDir {
        fn drop(&mut self) {
            match &self.previous {
                Some(v) => std::env::set_var("ZINNIA_TELEMETRY_DIR", v),
                None => std::env::remove_var("ZINNIA_TELEMETRY_DIR"),
            }
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    fn read_sink_file(path: &std::path::Path) -> String {
        std::fs::read_to_string(path).unwrap_or_default()
    }

    #[test]
    fn dyn_fancy_index_without_fact_emits_per_index_bounds_checks() {
        let mut b = IRBuilder::new();
        let arr = make_2d_dyn(&mut b, 4, 2);
        let data = match &arr {
            Value::DynamicNDArray(d) => d.clone(),
            _ => unreachable!(),
        };
        let indices = make_dynamic_index_list(&mut b, 3);

        let (_out, irs) = emitted_during(&mut b, |b| dyn_gather_rows(b, &data, &indices));
        let assert_count = count_asserts(&irs);
        assert_eq!(
            assert_count, 3,
            "default dyn fancy gather must emit one bounds Assert per index; irs = {irs:?}",
        );
    }

    #[test]
    fn dyn_fancy_index_with_forall_in_range_fact_skips_bounds_checks() {
        let _tel = ScopedTelemetryDir::enabled();
        let mut b = IRBuilder::new();
        let sink_path = b.telemetry().expect("telemetry enabled").path.clone();
        let arr = make_2d_dyn(&mut b, 4, 2);
        let data = match &arr {
            Value::DynamicNDArray(d) => d.clone(),
            _ => unreachable!(),
        };
        let indices = make_dynamic_index_list(&mut b, 3);
        // The indices' cells are in [0, 3] (an array fact, as a
        // `@requires(forall(idx, in_range(0, 3)))` states on an input's cells).
        let cells = b.leaf_cells(&indices).expect("an index list of scalars");
        b.state_defined_facts(
            &cells,
            crate::optim::array_facts::Segmentation::uniform(
                cells.len(),
                crate::optim::array_facts::ElemFacts::range(crate::optim::array_facts::Range::new(0, 3)),
            ),
        );

        let (_out, irs) = emitted_during(&mut b, |b| dyn_gather_rows(b, &data, &indices));
        let assert_count = count_asserts(&irs);
        assert_eq!(
            assert_count, 0,
            "forall_in_range-proved dyn fancy gather must skip per-index bounds Asserts; irs = {irs:?}",
        );
        drop(b);
        let telemetry = read_sink_file(&sink_path);
        assert!(
            telemetry.contains(r#""event":"strategy_dispatch""#)
                && telemetry.contains(r#""op":"dyn_fancy_index""#)
                && telemetry.contains(r#""strategy":"gather_skip_bounds_check""#),
            "proved gather must record the skip-check strategy dispatch; telemetry = {telemetry}",
        );
    }

    #[test]
    #[should_panic(expected = "index 9 out of range")]
    fn dyn_fancy_index_default_rejects_literal_out_of_range_index() {
        let mut b = IRBuilder::new();
        let values = (0..4)
            .map(|n| value_to_scalar_i64(&b.ir_constant_int(n)))
            .collect();
        let runtime_shape = vec![ScalarValue::new(Some(4), None)];
        let runtime_length = b.ir_constant_int(4);
        let arr = dyn_from_values_with_active_nd(
            &mut b,
            values,
            vec![4],
            runtime_shape,
            runtime_length,
            NumberType::Integer,
        );
        let data = match &arr {
            Value::DynamicNDArray(d) => d.clone(),
            _ => unreachable!(),
        };
        let indices = Value::List(CompositeData::new(
            vec![ZinniaType::Integer],
            vec![b.ir_constant_int(9)],
        ));

        let _ = dyn_gather_rows(&mut b, &data, &indices);
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
        assert_eq!(
            muli_count, 0,
            "literal-stride dispatch should fold static indices; got irs = {irs:?}"
        );
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
        let lhs_d = match &lhs {
            Value::DynamicNDArray(d) => d.clone(),
            _ => unreachable!(),
        };
        let rhs_d = match &rhs {
            Value::DynamicNDArray(d) => d.clone(),
            _ => unreachable!(),
        };
        let out = crate::ops::dyn_ndarray::binary::dyn_binary_op(&mut b, "add", &lhs_d, &rhs_d);
        let out_d = match &out {
            Value::DynamicNDArray(d) => d,
            _ => unreachable!(),
        };
        assert_eq!(out_d.meta.logical_shape, vec![3, 4]);
        assert_eq!(out_d.meta.runtime_shape[0].static_val, Some(3));
        assert_eq!(out_d.meta.runtime_shape[1].static_val, Some(4));
    }

    #[test]
    fn compact_constructor_checks_total_bound_without_facts() {
        // With no facts about the runtime shape, the compact constructor
        // must check `0 <= runtime_length <= total_bound` in-circuit, in
        // every mode (it used to do so only under
        // `ZINNIA_BOUNDED_AXIS_STRICT=1`).
        let mut b = IRBuilder::new();
        let (_arr, irs) = emitted_during(&mut b, |b| make_compact_2d_dyn(b, 100, 100, 10));
        let bound_checks = irs.iter().filter(|ir| matches!(ir, IR::AssertU128Lt)).count();
        assert!(
            bound_checks >= 1,
            "compact constructor should range-check its runtime length; irs = {irs:?}"
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
            ScopedEnvVar {
                key,
                previous,
                _lock: lock,
            }
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
    fn _envelope_unused() -> Option<Envelope> {
        None
    }
}
