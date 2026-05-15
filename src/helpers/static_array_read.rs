//! Native read paths for `Value::StaticArray`.
//!
//! P2 of `compiler.epic-segment-native-static-arrays`: reads (element,
//! row/column view, multi-dim element, slice, for-loop iteration) operate
//! directly on the segment-backed representation without first materialising
//! the array via `to_value_list`. Static-index reads prefer the cached wire
//! held in `IRBuilder::static_array_payload` (free); dynamic-index reads emit
//! a single `ir_read_memory` op (O(1) constraint cost vs the legacy O(N) mux
//! chain in `dynamic_list_subscript`). View vs copy policy:
//!
//! - **Single-axis-0 indexing** producing a (D-1)-rank slice (`arr[i]` on a
//!   ≥2-D array): view — same segment, adjusted offset, dropped axis-0
//!   stride/shape. Cheap and matches NumPy semantics; the chained subscript
//!   form `arr[i][j]` lands here on the first step.
//! - **Static slice** (`arr[a:b]`, `arr[a:b, c]`):
//!   - step=1, contiguous in memory: view (same segment, adjusted
//!     offset+shape).
//!   - step != 1, reversed, or non-contiguous slab: materialise into a fresh
//!     segment so downstream consumers see contiguous storage. This matches
//!     the policy used by `dyn_ndarray::reshape::dyn_transpose`.
//! - **Dynamic slice** (`arr[a:b]` with runtime `a` / `b`): always
//!   materialise (mirrors `dyn_slice_axis`).
//!
//! Cache policy: views produced by this module DO inherit the original cache
//! entry's segment_id — so reads via `static_array_payload.get(segment_id)`
//! still return the original wires. The `offset` field of the view shifts
//! the cache lookup window correctly.

use crate::builder::IRBuilder;
use crate::ops::dyn_ndarray::{scalar_i64_to_value, value_to_scalar_i64};
use crate::types::{NumberType, ScalarValue, SliceIndex, Value, ValueId};

use super::shape_arith::{decode_coords, row_major_strides};

// ────────────────────────────────────────────────────────────────────────
// Element / row helpers
// ────────────────────────────────────────────────────────────────────────

/// Read a single leaf from a `Value::StaticArray` at the given flat (linear)
/// offset relative to `offset` (i.e. element `flat_idx` of the logical
/// array). Prefers the cached original wire; falls back to a single
/// `ir_read_memory` if no cache entry exists.
fn read_leaf_at_flat(
    b: &mut IRBuilder,
    segment_id: u32,
    base_offset: usize,
    flat_idx: usize,
    dtype: NumberType,
) -> Value {
    let abs = base_offset + flat_idx;
    if let Some(cached) = b.static_array_payload.get(&segment_id) {
        if abs < cached.len() {
            return cached[abs].clone();
        }
    }
    let addr = b.ir_constant_int(abs as i64);
    let raw = b.ir_read_memory(segment_id, &addr);
    scalar_i64_to_value(&value_to_scalar_i64(&raw), dtype)
}

/// Read a single leaf at a runtime address. Always emits one
/// `ir_read_memory` op.
///
/// Runs Group 5a's `discharge_index_in_range` against the dynamic
/// address: a literal out of `[0, total_size)` panics at compile time,
/// a Disproved discharge panics, and Unknown emits a witness assertion
/// under lenient mode (panics under `ZINNIA_OP_REQUIRES_STRICT=1`). The
/// memory-trace permutation argument remains the prover-side backstop.
fn read_leaf_at_dynamic(
    b: &mut IRBuilder,
    segment_id: u32,
    addr: &Value,
    dtype: NumberType,
    total_size: usize,
) -> Value {
    // Load-bearing index-in-range discharge (Group 5a). Replaces the
    // informational `probe_in_range` with Phase E enforcement at the
    // static-array dynamic-read chokepoint.
    crate::optim::resolver::discharge_index_in_range(
        b,
        addr,
        0,
        total_size as i64,
        "static_array_read",
    );
    let raw = b.ir_read_memory(segment_id, addr);
    scalar_i64_to_value(&value_to_scalar_i64(&raw), dtype)
}

/// Build a runtime address from `offset + sum_k(idx[k] * strides[k])`.
fn compute_addr(
    b: &mut IRBuilder,
    base_offset: usize,
    strides: &[usize],
    indices: &[Value],
    shape: &[usize],
) -> Value {
    let mut static_sum: i64 = base_offset as i64;
    let mut dynamic_parts: Vec<Value> = Vec::new();
    for (ax, idx) in indices.iter().enumerate() {
        let stride = strides[ax] as i64;
        if let Some(i) = idx.int_val() {
            let i = if i < 0 { shape[ax] as i64 + i } else { i };
            static_sum += i * stride;
        } else {
            let stride_val = b.ir_constant_int(stride);
            dynamic_parts.push(b.ir_mul_i(idx, &stride_val));
        }
    }
    if dynamic_parts.is_empty() {
        b.ir_constant_int(static_sum)
    } else {
        let mut acc = b.ir_constant_int(static_sum);
        for p in &dynamic_parts {
            acc = b.ir_add_i(&acc, p);
        }
        acc
    }
}

// ────────────────────────────────────────────────────────────────────────
// Public dispatch
// ────────────────────────────────────────────────────────────────────────

/// Subscript a `Value::StaticArray` natively (no materialisation to List).
/// Mirrors the responsibilities of `dyn_subscript` for `DynamicNDArray`, but
/// stays on the StaticArray representation when it can.
pub fn static_array_subscript(b: &mut IRBuilder, val: &Value, indices: &[SliceIndex]) -> Value {
    let (dtype, shape, segment_id, strides, offset, imag_seg) = match val {
        Value::StaticArray { dtype, shape, segment_id, strides, offset, imag_segment_id, value_id: _ } => {
            (*dtype, shape.clone(), *segment_id, strides.clone(), *offset, *imag_segment_id)
        }
        _ => panic!("static_array_subscript: expected Value::StaticArray"),
    };

    if indices.is_empty() {
        return val.clone();
    }

    // Expand a single Ellipsis (`...`) into the right number of full-range
    // slices, based on the source rank.
    if indices.iter().any(|i| matches!(i, SliceIndex::Ellipsis)) {
        let consumed: usize = indices
            .iter()
            .filter(|i| matches!(i, SliceIndex::Single(_) | SliceIndex::Range(_, _, _)))
            .count();
        let num_colons = shape.len().saturating_sub(consumed);
        let mut expanded: Vec<SliceIndex> = Vec::with_capacity(indices.len() - 1 + num_colons);
        let mut seen = false;
        for idx in indices {
            match idx {
                SliceIndex::Ellipsis if !seen => {
                    seen = true;
                    for _ in 0..num_colons {
                        expanded.push(SliceIndex::Range(None, None, None));
                    }
                }
                SliceIndex::Ellipsis => panic!("an index can only have a single ellipsis ('...')"),
                other => expanded.push(other.clone()),
            }
        }
        return static_array_subscript(b, val, &expanded);
    }

    // For NewAxis, fall back to materialised List path — `to_value_list`
    // followed by `multidim_subscript`. NewAxis on a StaticArray is rare.
    if indices.iter().any(|i| matches!(i, SliceIndex::NewAxis)) {
        let lst = super::static_array::to_value_list(b, val);
        let data = match &lst {
            Value::List(d) => d.clone(),
            _ => return lst,
        };
        return super::ndarray::multidim_subscript(b, &data, indices);
    }

    // ── Fast path: all-Single, fully-resolved index across all axes →
    //    leaf read (static or dynamic).
    let all_single = indices.iter().all(|s| matches!(s, SliceIndex::Single(_)));
    if all_single && indices.len() == shape.len() {
        // All indices present.
        let idx_vals: Vec<Value> = indices.iter().map(|s| match s {
            SliceIndex::Single(v) => v.clone(),
            _ => unreachable!(),
        }).collect();

        // If all static, read from cache or build static address.
        let all_static = idx_vals.iter().all(|v| v.int_val().is_some());
        if all_static {
            // Compute flat (relative) index into payload, plus negative-norm.
            let mut flat: i64 = 0;
            for (ax, v) in idx_vals.iter().enumerate() {
                let i = v.int_val().unwrap();
                let i = if i < 0 { shape[ax] as i64 + i } else { i };
                flat += i * strides[ax] as i64;
            }
            if dtype == NumberType::Complex {
                return read_complex_leaf(b, segment_id, imag_seg.expect("Complex StaticArray missing imag_segment_id"), offset + flat as usize);
            }
            return read_leaf_at_flat(b, segment_id, offset, flat as usize, dtype);
        }
        let addr = compute_addr(b, offset, &strides, &idx_vals, &shape);
        if dtype == NumberType::Complex {
            return read_complex_leaf_dynamic(b, segment_id, imag_seg.expect("Complex StaticArray missing imag_segment_id"), &addr);
        }
        let total: usize = shape.iter().product();
        return read_leaf_at_dynamic(b, segment_id, &addr, dtype, offset + total);
    }

    // ── Single-index, ndim==1 → leaf or panic
    if indices.len() == 1 && shape.len() == 1 {
        match &indices[0] {
            SliceIndex::Single(v) => {
                if let Some(i) = v.int_val() {
                    let i = if i < 0 { shape[0] as i64 + i } else { i };
                    if dtype == NumberType::Complex {
                        return read_complex_leaf(b, segment_id, imag_seg.expect("Complex StaticArray missing imag_segment_id"), offset + i as usize * strides[0]);
                    }
                    return read_leaf_at_flat(b, segment_id, offset, i as usize, dtype);
                }
                // Dynamic 1-D: address = offset + i.
                // (stride is 1 for the only axis.)
                let off_val = b.ir_constant_int(offset as i64);
                let stride_val = b.ir_constant_int(strides[0] as i64);
                let scaled = b.ir_mul_i(v, &stride_val);
                let addr = b.ir_add_i(&off_val, &scaled);
                if dtype == NumberType::Complex {
                    return read_complex_leaf_dynamic(b, segment_id, imag_seg.expect("Complex StaticArray missing imag_segment_id"), &addr);
                }
                let total: usize = shape.iter().product();
                return read_leaf_at_dynamic(b, segment_id, &addr, dtype, offset + total);
            }
            SliceIndex::Range(s, e, st) => {
                return slice_axis_static_or_dynamic_1d(
                    b, val, dtype, shape, segment_id, strides, offset,
                    s.as_ref(), e.as_ref(), st.as_ref(),
                );
            }
            _ => unreachable!("Ellipsis/NewAxis already handled"),
        }
    }

    // ── Single-index on ndim>=2: row select on axis 0 → return a view
    //    (no materialisation). The chained `arr[i][j]` form lands here.
    if indices.len() == 1 && shape.len() >= 2 {
        match &indices[0] {
            SliceIndex::Single(v) => {
                let new_shape = shape[1..].to_vec();
                let new_strides = strides[1..].to_vec();
                if let Some(i) = v.int_val() {
                    let i = if i < 0 { shape[0] as i64 + i } else { i };
                    let new_offset = offset + (i as usize) * strides[0];
                    return Value::StaticArray {
                        dtype,
                        shape: new_shape,
                        segment_id,
                        strides: new_strides,
                        offset: new_offset,
                        imag_segment_id: imag_seg,
                        value_id: ValueId::next(),
                    };
                }
                // Dynamic axis-0 index: materialise the row into a fresh
                // contiguous segment. We can't represent a dynamic-offset
                // view in the StaticArray variant.
                return materialise_axis0_dynamic_row(
                    b, dtype, &shape, segment_id, imag_seg, &strides, offset, v,
                );
            }
            SliceIndex::Range(s, e, st) => {
                return slice_axis(
                    b, val, dtype, &shape, segment_id, &strides, offset,
                    0, s.as_ref(), e.as_ref(), st.as_ref(),
                );
            }
            _ => unreachable!(),
        }
    }

    // ── Multi-index, mixed Single / Range across multiple axes.
    //    Strategy: if all Range bounds are static and all Single ints are
    //    static → return a contiguous view when possible, else materialise.
    //    Otherwise route through the general multidim materialiser.
    multidim_subscript_static_array(b, val, indices)
}

// ────────────────────────────────────────────────────────────────────────
// Slice helpers
// ────────────────────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn slice_axis_static_or_dynamic_1d(
    b: &mut IRBuilder,
    src: &Value,
    dtype: NumberType,
    shape: Vec<usize>,
    segment_id: u32,
    strides: Vec<usize>,
    offset: usize,
    start: Option<&Value>,
    stop: Option<&Value>,
    step: Option<&Value>,
) -> Value {
    let len = shape[0];
    let static_val = |v: Option<&Value>| -> Option<i64> {
        v.and_then(|val| if matches!(val, Value::None) { None } else { val.int_val() })
    };
    let is_present = |v: Option<&Value>| -> bool {
        matches!(v, Some(val) if !matches!(val, Value::None))
    };
    let s_static = static_val(start);
    let e_static = static_val(stop);
    let st_static = static_val(step);

    let all_static = (!is_present(start) || s_static.is_some())
        && (!is_present(stop) || e_static.is_some())
        && (!is_present(step) || st_static.is_some());

    let out = if !all_static {
        // Dynamic bounds → materialise.
        materialise_dynamic_1d_slice(
            b, dtype, len, segment_id, strides[0], offset,
            start, stop, step,
        )
    } else {
        let len_i = len as i64;
        let s = s_static.unwrap_or(0);
        let e = e_static.unwrap_or(len_i);
        let st = st_static.unwrap_or(1);
        let s = if s < 0 { (len_i + s).max(0) } else { s.min(len_i) };
        let e = if e < 0 { (len_i + e).max(0) } else { e.min(len_i) };
        assert!(st != 0, "slice step cannot be zero");

        if st == 1 && strides[0] == 1 {
            // Contiguous step=1 slab → view.
            let out_len = (e - s).max(0) as usize;
            let new_offset = offset + s as usize;
            // Preserve dual-segment imag for Complex views.
            let imag_seg = if let Value::StaticArray { imag_segment_id, .. } = src {
                *imag_segment_id
            } else { None };
            Value::StaticArray {
                dtype,
                shape: vec![out_len],
                segment_id,
                strides: vec![strides[0]],
                offset: new_offset,
                imag_segment_id: imag_seg,
                value_id: ValueId::next(),
            }
        } else {
            // Non-contiguous: materialise.
            let mut indices: Vec<i64> = Vec::new();
            if st > 0 {
                let mut i = s;
                while i < e { indices.push(i); i += st; }
            } else {
                let mut i = s;
                while i > e { indices.push(i); i += st; }
            }
            let out_len = indices.len();
            if dtype == NumberType::Complex {
                let imag_seg = if let Value::StaticArray { imag_segment_id, .. } = src {
                    imag_segment_id.expect("Complex StaticArray missing imag_segment_id")
                } else { unreachable!() };
                let mut reals: Vec<Value> = Vec::with_capacity(out_len);
                let mut imags: Vec<Value> = Vec::with_capacity(out_len);
                for src_i in &indices {
                    let flat = (*src_i as usize) * strides[0];
                    let abs = offset + flat;
                    let leaf = read_complex_leaf(b, segment_id, imag_seg, abs);
                    if let Value::Complex { real, imag } = leaf {
                        reals.push(Value::Float(real));
                        imags.push(Value::Float(imag));
                    }
                }
                super::static_array::build_static_array_from_flat_complex(b, reals, imags, vec![out_len])
            } else {
                let mut leaves: Vec<Value> = Vec::with_capacity(out_len);
                for src_i in &indices {
                    let flat = (*src_i as usize) * strides[0];
                    leaves.push(read_leaf_at_flat(b, segment_id, offset, flat, dtype));
                }
                super::static_array::build_static_array_from_flat(b, leaves, vec![out_len], dtype)
            }
        }
    };
    if let (Some(in_vid), Some(out_vid)) = (src.value_id(), out.value_id()) {
        crate::optim::resolver::relay_forall_eq_const_from_input(b, in_vid, out_vid);
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn materialise_dynamic_1d_slice(
    b: &mut IRBuilder,
    dtype: NumberType,
    len: usize,
    segment_id: u32,
    axis_stride: usize,
    offset: usize,
    start: Option<&Value>,
    stop: Option<&Value>,
    step: Option<&Value>,
) -> Value {
    let max_out_len = len;

    // Load-bearing slice-bound discharge (compiler.fuzz-finding-v2-slice-oob-witness-miss).
    // The internal mask-and-clamp below silently swallows OOB `start` / `stop`
    // — without an explicit discharge, an OOB witness slips through and
    // returns `satisfied = True`. Mirrors the scalar arm's
    // `discharge_index_in_range` (Group 5a) on the dyn-read chokepoint.
    crate::optim::resolver::discharge_slice_bound(b, start, len, "static_array_slice_start");
    crate::optim::resolver::discharge_slice_bound(b, stop, len, "static_array_slice_stop");

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

    let default_val = crate::ops::dyn_ndarray::metadata::dyn_default_value(b, dtype);
    let zero = b.ir_constant_int(0);
    let len_val = b.ir_constant_int(len as i64);
    let stride_val = b.ir_constant_int(axis_stride as i64);
    let offset_val = b.ir_constant_int(offset as i64);

    let mut leaves: Vec<Value> = Vec::with_capacity(max_out_len);
    for i in 0..max_out_len {
        let i_const = b.ir_constant_int(i as i64);
        let off2 = b.ir_mul_i(&i_const, &step_ir);
        let src_idx = b.ir_add_i(&start_ir, &off2);

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

        let scaled = b.ir_mul_i(&clamped, &stride_val);
        let addr = b.ir_add_i(&offset_val, &scaled);
        let elem = b.ir_read_memory(segment_id, &addr);
        let masked = if dtype == NumberType::Float {
            b.ir_select_f(&in_bounds, &elem, &default_val)
        } else {
            b.ir_select_i(&in_bounds, &elem, &default_val)
        };
        leaves.push(scalar_i64_to_value(&value_to_scalar_i64(&masked), dtype));
    }

    super::static_array::build_static_array_from_flat(
        b, leaves, vec![max_out_len], dtype,
    )
}

#[allow(clippy::too_many_arguments)]
fn slice_axis(
    b: &mut IRBuilder,
    src: &Value,
    dtype: NumberType,
    shape: &[usize],
    segment_id: u32,
    strides: &[usize],
    offset: usize,
    axis: usize,
    start: Option<&Value>,
    stop: Option<&Value>,
    step: Option<&Value>,
) -> Value {
    if shape.len() == 1 {
        // The inner 1-D helper already emits the content-relay on return,
        // so don't double-relay here.
        return slice_axis_static_or_dynamic_1d(
            b, src, dtype, shape.to_vec(), segment_id, strides.to_vec(), offset,
            start, stop, step,
        );
    }
    let static_val = |v: Option<&Value>| -> Option<i64> {
        v.and_then(|val| if matches!(val, Value::None) { None } else { val.int_val() })
    };
    let is_present = |v: Option<&Value>| -> bool {
        matches!(v, Some(val) if !matches!(val, Value::None))
    };
    let s_static = static_val(start);
    let e_static = static_val(stop);
    let st_static = static_val(step);
    let all_static = (!is_present(start) || s_static.is_some())
        && (!is_present(stop) || e_static.is_some())
        && (!is_present(step) || st_static.is_some());

    let out = if !all_static {
        materialise_dynamic_axis_slice(
            b, dtype, shape, segment_id, strides, offset, axis, start, stop, step,
        )
    } else {
        let len_i = shape[axis] as i64;
        let s = s_static.unwrap_or(0);
        let e = e_static.unwrap_or(len_i);
        let st = st_static.unwrap_or(1);
        let s = if s < 0 { (len_i + s).max(0) } else { s.min(len_i) };
        let e = if e < 0 { (len_i + e).max(0) } else { e.min(len_i) };
        assert!(st != 0, "slice step cannot be zero");

        if st == 1 && axis == 0 {
            // Slice along axis 0 with step 1: contiguous view.
            let out_len = (e - s).max(0) as usize;
            let mut new_shape = shape.to_vec();
            new_shape[0] = out_len;
            let new_offset = offset + (s as usize) * strides[0];
            let imag_seg = if let Value::StaticArray { imag_segment_id, .. } = src {
                *imag_segment_id
            } else { None };
            Value::StaticArray {
                dtype,
                shape: new_shape,
                segment_id,
                strides: strides.to_vec(),
                offset: new_offset,
                imag_segment_id: imag_seg,
                value_id: ValueId::next(),
            }
        } else {
            // Non-contiguous along this axis OR slicing inner axis: materialise.
            let mut indices: Vec<i64> = Vec::new();
            if st > 0 {
                let mut i = s;
                while i < e { indices.push(i); i += st; }
            } else {
                let mut i = s;
                while i > e { indices.push(i); i += st; }
            }
            let out_axis_len = indices.len();
            let mut new_shape = shape.to_vec();
            new_shape[axis] = out_axis_len;
            let total: usize = new_shape.iter().product();
            let new_strides = row_major_strides(&new_shape);

            let mut leaves: Vec<Value> = Vec::with_capacity(total);
            for flat_out in 0..total {
                let coords = decode_coords(flat_out, &new_shape, &new_strides);
                let mut src_flat: usize = 0;
                for ax in 0..shape.len() {
                    let src_coord = if ax == axis {
                        indices[coords[ax]] as usize
                    } else {
                        coords[ax]
                    };
                    src_flat += src_coord * strides[ax];
                }
                leaves.push(read_leaf_at_flat(b, segment_id, offset, src_flat, dtype));
            }
            super::static_array::build_static_array_from_flat(b, leaves, new_shape, dtype)
        }
    };
    if let (Some(in_vid), Some(out_vid)) = (src.value_id(), out.value_id()) {
        crate::optim::resolver::relay_forall_eq_const_from_input(b, in_vid, out_vid);
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn materialise_dynamic_axis_slice(
    b: &mut IRBuilder,
    dtype: NumberType,
    shape: &[usize],
    segment_id: u32,
    strides: &[usize],
    offset: usize,
    axis: usize,
    start: Option<&Value>,
    stop: Option<&Value>,
    step: Option<&Value>,
) -> Value {
    let axis_len = shape[axis];
    let max_axis = axis_len;

    // Load-bearing slice-bound discharge (compiler.fuzz-finding-v2-slice-oob-witness-miss).
    // See `materialise_dynamic_1d_slice` for rationale; multi-dim path is
    // analogous, discharging against the length of the sliced axis.
    crate::optim::resolver::discharge_slice_bound(b, start, axis_len, "static_array_slice_start");
    crate::optim::resolver::discharge_slice_bound(b, stop, axis_len, "static_array_slice_stop");

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

    let default_val = crate::ops::dyn_ndarray::metadata::dyn_default_value(b, dtype);
    let zero = b.ir_constant_int(0);
    let axis_len_val = b.ir_constant_int(axis_len as i64);
    let axis_stride_val = b.ir_constant_int(strides[axis] as i64);

    let mut new_shape = shape.to_vec();
    new_shape[axis] = max_axis;
    let total: usize = new_shape.iter().product();
    let new_strides = row_major_strides(&new_shape);

    let mut leaves: Vec<Value> = Vec::with_capacity(total);
    for flat_out in 0..total {
        let coords = decode_coords(flat_out, &new_shape, &new_strides);
        let slice_idx = coords[axis];
        let slice_idx_val = b.ir_constant_int(slice_idx as i64);

        let off2 = b.ir_mul_i(&slice_idx_val, &step_ir);
        let src_axis_idx = b.ir_add_i(&start_ir, &off2);

        let ge_zero = b.ir_greater_than_or_equal_i(&src_axis_idx, &zero);
        let lt_len = b.ir_less_than_i(&src_axis_idx, &axis_len_val);
        let in_range = b.ir_logical_and(&ge_zero, &lt_len);
        let step_pos = b.ir_greater_than_i(&step_ir, &zero);
        let lt_stop = b.ir_less_than_i(&src_axis_idx, &stop_ir);
        let gt_stop = b.ir_greater_than_i(&src_axis_idx, &stop_ir);
        let stop_ok = b.ir_select_i(&step_pos, &lt_stop, &gt_stop);
        let stop_bool = b.ir_bool_cast(&stop_ok);
        let in_bounds = b.ir_logical_and(&in_range, &stop_bool);

        let max_idx = b.ir_constant_int(axis_len as i64 - 1);
        let is_neg = b.ir_less_than_i(&src_axis_idx, &zero);
        let is_over = b.ir_greater_than_i(&src_axis_idx, &max_idx);
        let clamped_hi = b.ir_select_i(&is_over, &max_idx, &src_axis_idx);
        let clamped = b.ir_select_i(&is_neg, &zero, &clamped_hi);

        let mut addr_static: i64 = offset as i64;
        for ax in 0..shape.len() {
            if ax == axis {
                continue;
            }
            addr_static += coords[ax] as i64 * strides[ax] as i64;
        }
        let other_offset = b.ir_constant_int(addr_static);
        let axis_contrib = b.ir_mul_i(&clamped, &axis_stride_val);
        let addr = b.ir_add_i(&other_offset, &axis_contrib);

        let elem = b.ir_read_memory(segment_id, &addr);
        let masked = if dtype == NumberType::Float {
            b.ir_select_f(&in_bounds, &elem, &default_val)
        } else {
            b.ir_select_i(&in_bounds, &elem, &default_val)
        };
        leaves.push(scalar_i64_to_value(&value_to_scalar_i64(&masked), dtype));
    }
    super::static_array::build_static_array_from_flat(b, leaves, new_shape, dtype)
}

/// Materialise a single dynamic-axis-0 row into a fresh segment. For Complex
/// dtype, two parallel reads/writes per element across the dual segments.
fn materialise_axis0_dynamic_row(
    b: &mut IRBuilder,
    dtype: NumberType,
    shape: &[usize],
    segment_id: u32,
    imag_seg: Option<u32>,
    strides: &[usize],
    offset: usize,
    idx: &Value,
) -> Value {
    let row_stride = strides[0];
    let row_shape = shape[1..].to_vec();
    let row_total: usize = row_shape.iter().product();
    let stride_val = b.ir_constant_int(row_stride as i64);
    let offset_val = b.ir_constant_int(offset as i64);
    let base = b.ir_mul_i(idx, &stride_val);
    let base = b.ir_add_i(&offset_val, &base);

    if dtype == NumberType::Complex {
        let imag_seg = imag_seg.expect("Complex StaticArray missing imag_segment_id");
        let mut real_leaves: Vec<Value> = Vec::with_capacity(row_total);
        let mut imag_leaves: Vec<Value> = Vec::with_capacity(row_total);
        for j in 0..row_total {
            let off2 = b.ir_constant_int(j as i64);
            let addr = b.ir_add_i(&base, &off2);
            let r = b.ir_read_memory(segment_id, &addr);
            let im = b.ir_read_memory(imag_seg, &addr);
            real_leaves.push(scalar_i64_to_value(&value_to_scalar_i64(&r), NumberType::Float));
            imag_leaves.push(scalar_i64_to_value(&value_to_scalar_i64(&im), NumberType::Float));
        }
        return super::static_array::build_static_array_from_flat_complex(
            b, real_leaves, imag_leaves, row_shape,
        );
    }

    let mut leaves: Vec<Value> = Vec::with_capacity(row_total);
    for j in 0..row_total {
        let off2 = b.ir_constant_int(j as i64);
        let addr = b.ir_add_i(&base, &off2);
        let elem = b.ir_read_memory(segment_id, &addr);
        leaves.push(scalar_i64_to_value(&value_to_scalar_i64(&elem), dtype));
    }
    super::static_array::build_static_array_from_flat(b, leaves, row_shape, dtype)
}

/// Multi-axis subscript with mixed Single + Range. This is the general case;
/// covers `arr[i, j]`, `arr[:, j]`, `arr[a:b, c]`, etc.
fn multidim_subscript_static_array(
    b: &mut IRBuilder,
    val: &Value,
    indices: &[SliceIndex],
) -> Value {
    let (dtype, shape, segment_id, strides, offset, imag_seg) = match val {
        Value::StaticArray { dtype, shape, segment_id, strides, offset, imag_segment_id, value_id: _ } => {
            (*dtype, shape.clone(), *segment_id, strides.clone(), *offset, *imag_segment_id)
        }
        _ => unreachable!(),
    };

    let rank = shape.len();
    if indices.len() > rank {
        // Trailing indices can only be NewAxis (handled at top-level) — at
        // this depth, treat the rest as scalar-trailing, which means too
        // many indices.
        panic!("too many indices for array: array is {}-dimensional, but {} were indexed", rank, indices.len());
    }

    // If every position is Single and resolved (covered earlier), so reach
    // here for at least one Range. Compute a per-axis "selection" — either
    // Single (consumes the axis) or Range coords.
    let mut out_shape: Vec<usize> = Vec::new();
    let mut axis_kind: Vec<AxisSel> = Vec::with_capacity(rank);

    for ax in 0..rank {
        if ax >= indices.len() {
            // Trailing axes default to full range.
            out_shape.push(shape[ax]);
            axis_kind.push(AxisSel::FullRange);
            continue;
        }
        match &indices[ax] {
            SliceIndex::Single(v) => {
                axis_kind.push(AxisSel::Single(v.clone()));
            }
            SliceIndex::Range(s, e, st) => {
                let len_i = shape[ax] as i64;
                let resolve = |v: &Option<Value>, def: i64| -> Option<i64> {
                    match v.as_ref() {
                        Some(Value::None) | None => Some(def),
                        Some(val) => val.int_val(),
                    }
                };
                let st_static = resolve(st, 1);
                let s_static = resolve(s, 0);
                let e_static = resolve(e, len_i);
                if let (Some(ss), Some(ee), Some(stst)) = (s_static, e_static, st_static) {
                    let ss = if ss < 0 { (len_i + ss).max(0) } else { ss.min(len_i) };
                    let ee = if ee < 0 { (len_i + ee).max(0) } else { ee.min(len_i) };
                    assert!(stst != 0, "slice step cannot be zero");
                    let mut coords: Vec<usize> = Vec::new();
                    if stst > 0 {
                        let mut i = ss;
                        while i < ee { coords.push(i as usize); i += stst; }
                    } else {
                        let mut i = ss;
                        while i > ee { coords.push(i as usize); i += stst; }
                    }
                    out_shape.push(coords.len());
                    axis_kind.push(AxisSel::Coords(coords));
                } else {
                    // Dynamic-bounds range on this axis: fall back to the
                    // dynamic-axis materialiser, then recurse.
                    let sliced = slice_axis(
                        b, val, dtype, &shape, segment_id, &strides, offset,
                        ax, s.as_ref(), e.as_ref(), st.as_ref(),
                    );
                    let remaining: Vec<SliceIndex> = indices.iter().enumerate().map(|(i, idx)| {
                        if i == ax { SliceIndex::Range(None, None, None) } else { idx.clone() }
                    }).collect();
                    return static_array_subscript(b, &sliced, &remaining);
                }
            }
            _ => unreachable!("Ellipsis/NewAxis already handled at top-level"),
        }
    }

    // Compute base from Single static contributions; collect dynamic Single
    // contributions; collect axis-coord arrays for Range axes.
    let mut static_base: i64 = offset as i64;
    let mut dynamic_parts: Vec<Value> = Vec::new();
    let mut range_axes: Vec<(usize, Vec<usize>)> = Vec::new();

    for ax in 0..rank {
        match &axis_kind[ax] {
            AxisSel::Single(v) => {
                if let Some(i) = v.int_val() {
                    let i = if i < 0 { shape[ax] as i64 + i } else { i };
                    static_base += i * strides[ax] as i64;
                } else {
                    let stride_val = b.ir_constant_int(strides[ax] as i64);
                    dynamic_parts.push(b.ir_mul_i(v, &stride_val));
                }
            }
            AxisSel::Coords(coords) => {
                range_axes.push((ax, coords.clone()));
            }
            AxisSel::FullRange => {
                let coords: Vec<usize> = (0..shape[ax]).collect();
                range_axes.push((ax, coords));
            }
        }
    }

    let total: usize = out_shape.iter().product();
    let out_strides = row_major_strides(&out_shape);

    // Special case: out_shape empty (all axes were Single) — that's the
    // all-single fast path covered earlier; should not reach here. Guard.
    if total == 0 || range_axes.is_empty() {
        // All Single but indices.len() < rank → out is a view into the
        // remaining axes. Build a view StaticArray.
        if range_axes.is_empty() && indices.len() < rank {
            // Shouldn't happen — we filled trailing axes as FullRange.
            unreachable!();
        }
        // Empty dimension somewhere → empty array.
        let empty: Vec<Value> = Vec::new();
        return super::static_array::build_static_array_from_flat(b, empty, out_shape, dtype);
    }

    let mut leaves: Vec<Value> = Vec::with_capacity(total);
    for flat_out in 0..total {
        let out_coords = decode_coords(flat_out, &out_shape, &out_strides);
        let mut src_flat = static_base;
        for (out_ax, (src_ax, coords)) in range_axes.iter().enumerate() {
            src_flat += coords[out_coords[out_ax]] as i64 * strides[*src_ax] as i64;
        }
        if dynamic_parts.is_empty() {
            // Fully static address (after relative-to-segment normalisation).
            // src_flat is already absolute (offset baked into static_base).
            let flat_idx = src_flat as usize;
            if dtype == NumberType::Complex {
                let im = imag_seg.expect("Complex StaticArray missing imag_segment_id");
                leaves.push(read_complex_leaf(b, segment_id, im, flat_idx));
            } else {
                // read_leaf_at_flat takes a base+rel; pass abs as 0+abs.
                leaves.push(read_leaf_at_flat(b, segment_id, 0, flat_idx, dtype));
            }
        } else {
            let static_part = b.ir_constant_int(src_flat);
            let mut acc = static_part;
            for p in &dynamic_parts {
                acc = b.ir_add_i(&acc, p);
            }
            if dtype == NumberType::Complex {
                let im = imag_seg.expect("Complex StaticArray missing imag_segment_id");
                leaves.push(read_complex_leaf_dynamic(b, segment_id, im, &acc));
            } else {
                let src_total: usize = shape.iter().product();
                leaves.push(read_leaf_at_dynamic(b, segment_id, &acc, dtype, offset + src_total));
            }
        }
    }
    if dtype == NumberType::Complex {
        let mut reals: Vec<Value> = Vec::with_capacity(leaves.len());
        let mut imags: Vec<Value> = Vec::with_capacity(leaves.len());
        for v in leaves {
            match v {
                Value::Complex { real, imag } => {
                    reals.push(Value::Float(real));
                    imags.push(Value::Float(imag));
                }
                _ => unreachable!(),
            }
        }
        return super::static_array::build_static_array_from_flat_complex(b, reals, imags, out_shape);
    }
    super::static_array::build_static_array_from_flat(b, leaves, out_shape, dtype)
}

enum AxisSel {
    Single(Value),
    Coords(Vec<usize>),
    FullRange,
}

// ────────────────────────────────────────────────────────────────────────
// Iteration helpers
// ────────────────────────────────────────────────────────────────────────

/// Build the per-iteration value for `for x in arr` over a `Value::StaticArray`.
/// For a 1-D array this is a leaf (Integer / Float / Boolean) read at index
/// `i`. For a ≥2-D array this is a (D-1)-rank `Value::StaticArray` view
/// sharing the same segment.
pub fn iter_element(b: &mut IRBuilder, arr: &Value, i: usize) -> Value {
    let (dtype, shape, segment_id, strides, offset, imag_seg) = match arr {
        Value::StaticArray { dtype, shape, segment_id, strides, offset, imag_segment_id, value_id: _ } => {
            (*dtype, shape.clone(), *segment_id, strides.clone(), *offset, *imag_segment_id)
        }
        _ => panic!("iter_element: expected StaticArray"),
    };
    if shape.len() == 1 {
        // For Complex 1-D, return Value::Complex {real, imag}.
        if dtype == NumberType::Complex {
            return read_complex_leaf(b, segment_id, imag_seg.expect("Complex array missing imag_segment_id"), offset + i * strides[0]);
        }
        return read_leaf_at_flat(b, segment_id, offset, i * strides[0], dtype);
    }
    // ≥2-D → view sharing segment(s).
    let new_shape = shape[1..].to_vec();
    let new_strides = strides[1..].to_vec();
    let new_offset = offset + i * strides[0];
    Value::StaticArray {
        dtype,
        shape: new_shape,
        segment_id,
        strides: new_strides,
        offset: new_offset,
        imag_segment_id: imag_seg,
        value_id: ValueId::next(),
    }
}

/// Read a single Complex element from a dual-segment Complex StaticArray at
/// the given absolute flat offset. Prefers cached Value::Complex wires.
pub fn read_complex_leaf(
    b: &mut IRBuilder,
    real_seg: u32,
    imag_seg: u32,
    abs_offset: usize,
) -> Value {
    if let Some(cached) = b.static_array_payload.get(&real_seg) {
        if abs_offset < cached.len() {
            // Cache (rep A) holds Value::Complex directly.
            return cached[abs_offset].clone();
        }
    }
    let addr = b.ir_constant_int(abs_offset as i64);
    let real_raw = b.ir_read_memory(real_seg, &addr);
    let imag_raw = b.ir_read_memory(imag_seg, &addr);
    let real_sv = match scalar_i64_to_value(&value_to_scalar_i64(&real_raw), NumberType::Float) {
        Value::Float(s) => s,
        _ => unreachable!(),
    };
    let imag_sv = match scalar_i64_to_value(&value_to_scalar_i64(&imag_raw), NumberType::Float) {
        Value::Float(s) => s,
        _ => unreachable!(),
    };
    Value::Complex { real: real_sv, imag: imag_sv }
}

/// Read a Complex element at a runtime address (sum of `offset + sum_k(idx*stride)`).
pub fn read_complex_leaf_dynamic(
    b: &mut IRBuilder,
    real_seg: u32,
    imag_seg: u32,
    addr: &Value,
) -> Value {
    let real_raw = b.ir_read_memory(real_seg, addr);
    let imag_raw = b.ir_read_memory(imag_seg, addr);
    let real_sv = match scalar_i64_to_value(&value_to_scalar_i64(&real_raw), NumberType::Float) {
        Value::Float(s) => s,
        _ => unreachable!(),
    };
    let imag_sv = match scalar_i64_to_value(&value_to_scalar_i64(&imag_raw), NumberType::Float) {
        Value::Float(s) => s,
        _ => unreachable!(),
    };
    Value::Complex { real: real_sv, imag: imag_sv }
}
