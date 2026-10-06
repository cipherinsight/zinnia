//! Native write paths for `Value::StaticArray`.
//!
//! P3 of `compiler.epic-segment-native-static-arrays`: writes (element setitem
//! at static and dynamic indices, multi-dim writes, slice setitem, and
//! augmented assignment) operate directly on the segment-backed
//! representation.
//!
//! The headline win is replacing the legacy `dynamic_list_set_item` mux
//! chain (O(N) per write) with a single `ir_write_memory(segment_id, addr,
//! value)` (O(1) per write). For sort / queens / nested-write benchmarks
//! that loop with dynamic-index writes, this turns "compile timed out" into
//! "compiles in seconds".
//!
//! Cache policy (`IRBuilder::static_array_payload`):
//!
//! - **Element write at static index**: the cache cell at the written offset
//!   is updated to the new wire (option (b) from the card). Other cached
//!   cells stay valid; static-index reads after the write continue to land
//!   as cheap cached lookups.
//! - **Small element/slice writes at dynamic indices**: the cache is updated
//!   as a symbolic overlay (`select(idx == coord, new, old)`) and the segment
//!   is marked lazy. Static consumers keep using direct wires; a future
//!   memory-backed dynamic read/write flushes the overlay through
//!   `materialize_static_array_segment`.
//! - **Large or unsupported writes**: the segment is materialised and the
//!   cache entry is removed, preserving the existing zkRAM fallback.
//!
//! View-write fidelity: `Value::StaticArray` views (e.g. `arr[i]` for ≥2-D
//! `arr`, or `arr[a:b]` static contiguous slices from P2) inherit the
//! source segment_id with an adjusted offset. A write into the view writes
//! through to the source segment at `view.offset + flat_idx`. No new
//! segment is allocated.

use crate::builder::IRBuilder;
use crate::types::{NumberType, SliceIndex, Value, ValueId};

use super::super::shape_arith::{decode_coords, row_major_strides};

// ────────────────────────────────────────────────────────────────────────
// Helpers
// ────────────────────────────────────────────────────────────────────────

/// Conditional memory writes read the previous physical cell. A lazy
/// StaticArray keeps that content only in its cached payload until forced;
/// without this boundary, skipped writes select allocation zeros instead.
/// Unconditional static writes still update the cache without forcing it.
fn materialize_before_predicated_write(b: &mut IRBuilder, val: &Value) {
    let conditional = b
        .write_predicate()
        .is_some_and(|p| p.bool_val().or_else(|| p.int_val().map(|v| v != 0)) != Some(true));
    if conditional {
        super::base::materialize_static_array_if_lazy(b, val);
    }
}

/// `v` converted to the dtype of the array stored in `segment_id` before it
/// is stored there (NumPy converts an assigned value to the array's dtype).
/// A bool array stores truth values as bool cells
/// (`IRBuilder::bool_dtype_cell`), which keeps its cached cells bool cells
/// (see `builder/bool_dtype.rs`).
fn cast_for_segment(b: &mut IRBuilder, segment_id: u32, v: &Value, dtype: NumberType) -> Value {
    if dtype == NumberType::Integer && b.is_bool_segment(segment_id) {
        return b.bool_dtype_cell(v);
    }
    cast_to_dtype(b, v, dtype)
}

/// Cast a value to match the target dtype before storing it in a segment.
fn cast_to_dtype(b: &mut IRBuilder, v: &Value, dtype: NumberType) -> Value {
    match dtype {
        NumberType::Float => {
            if matches!(v, Value::Float(_)) {
                v.clone()
            } else {
                b.ir_float_cast(v)
            }
        }
        NumberType::Integer => {
            if matches!(v, Value::Integer(_)) {
                v.clone()
            } else if matches!(v, Value::Boolean(_)) {
                // `ir_int_cast` reads `float_val()` for inference, so it
                // loses Boolean compile-time values. Build the Integer
                // wire and explicitly carry over the static_val so the
                // cached cell's `int_val()` stays compile-time-resolvable.
                // Without this, `[True] * limit` followed by `a[i] = False`
                // strips the static `False` and breaks downstream control
                // flow (range / enumerate over the cell's truthiness).
                let casted = b.ir_int_cast(v);
                if let (Value::Integer(mut s), Some(b_val)) = (casted.clone(), v.bool_val()) {
                    s.static_val = Some(if b_val { 1 } else { 0 });
                    Value::Integer(s)
                } else {
                    casted
                }
            } else {
                b.ir_int_cast(v)
            }
        }
        // Complex casting is component-aware. For non-Complex sources we
        // promote via `unpack_to_complex_parts`. For Complex sources we pass
        // through. The dispatch in `static_array_setitem` for Complex dtype
        // splits the write into two per-component segment writes; this
        // helper is *only* called by the per-component (Float) path, never
        // with `dtype == Complex`. We surface a clear panic if it ever does.
        NumberType::Complex => panic!(
            "cast_to_dtype(Complex) is not used for dual-segment StaticArray writes; \
             component-level writes use NumberType::Float per cell"
        ),
    }
}

/// Update the payload cache cell at the given absolute offset to the new
/// wire. Used after a static-index write so cached-wire reads stay coherent.
fn cache_set_cell(b: &mut IRBuilder, segment_id: u32, abs_offset: usize, new_val: Value) {
    if b.static_array_product_provenance.contains_key(&segment_id) {
        super::base::force_static_array_product_provenance(b, segment_id);
    }
    if let Some(cells) = b.static_array_payload.get_mut(&segment_id) {
        if abs_offset < cells.len() {
            cells[abs_offset] = new_val;
        }
    }
}

/// Invalidate (drop) the payload cache for a segment after a write whose
/// destination cell isn't statically known. Subsequent reads via
/// `to_value_list` or `read_leaf_at_flat` will fall through to issuing
/// `ir_read_memory` ops, which see the up-to-date post-write zkRAM state.
fn cache_invalidate(b: &mut IRBuilder, segment_id: u32) {
    if b.static_array_product_provenance.contains_key(&segment_id) {
        super::base::force_static_array_product_provenance(b, segment_id);
    }
    b.static_array_payload.remove(&segment_id);
}

const SMALL_DYNAMIC_WRITE_OVERLAY_LIMIT: usize = 1024;

fn select_typed(
    b: &mut IRBuilder,
    dtype: NumberType,
    cond: &Value,
    true_val: &Value,
    false_val: &Value,
) -> Value {
    match dtype {
        NumberType::Float => b.ir_select_f(cond, true_val, false_val),
        NumberType::Integer => {
            let out = b.ir_select_i(cond, true_val, false_val);
            // A choice between two NumPy bools is a bool (a write into a
            // bool array keeps its cached cells bool cells).
            if b.is_bool_cell(true_val) && b.is_bool_cell(false_val) {
                b.record_bool_cell(&out);
            }
            out
        }
        NumberType::Complex => {
            panic!("select_typed does not handle Complex StaticArray overlays")
        }
    }
}

use super::base::{index_equals_coord, normalize_dynamic_index};

/// Bounds chokepoint for the dynamic-index write paths. For every runtime
/// (non-literal) index axis, discharge numpy's validity contract
/// `-axis_len <= i < axis_len` through the Phase-E resolver — the write-side
/// analogue of the read chokepoints (`read.rs:82-88, 254-260, 531-564`).
///
/// This is load-bearing soundness: without it an out-of-range dynamic write
/// silently no-ops on the overlay path (the per-cell `index_equals_coord`
/// matches nothing) or writes an unconstrained address on the memory-write
/// fallback; the read path has discharged at its chokepoint all along.
///
/// We discharge against the *raw* index (no normalize wire) rather than the
/// numpy-wrapped value: `-n <= i < n` is exactly the numpy validity window,
/// and discharging it directly anchors on the input wire's `value_id` while
/// avoiding the `to_signed`/select normalization fan-out per element (which
/// would balloon scatter cases like 0318 that recurse here per index). When
/// the index is provably non-negative (`ask_bounds`), we tighten to `[0, n)`
/// so the `AssertU128Lt` interval hook (`builder.rs`) can deposit the same
/// `[0, n)` interval the read side relies on. The actual write address is
/// numpy-wrapped downstream — `index_equals_coord` (overlay) and
/// `normalize_for_address` (memory-write fallback) both handle the negative
/// wrap — so the discharged window and the written address agree.
fn discharge_dynamic_write_indices(b: &mut IRBuilder, shape: &[usize], idx_vals: &[Value]) {
    for (ax, idx) in idx_vals.iter().enumerate() {
        discharge_write_index(b, idx, shape[ax]);
    }
}

/// Discharge NumPy's validity window for one index of a write into an axis
/// of length `axis_len` (see [`discharge_dynamic_write_indices`]). Literal
/// indices are compile-time checked against the `[-n, n)` window; a runtime
/// index is checked in-circuit (under the execution predicate, see
/// `builder/guard.rs`) unless the facts prove it.
fn discharge_write_index(b: &mut IRBuilder, idx: &Value, axis_len: usize) {
    let axis_len = axis_len as i64;
    let non_negative = idx
        .int_val()
        .map(|v| v >= 0)
        .or_else(|| b.ask_bounds(idx).map(|(min, _)| min >= 0))
        .unwrap_or(false);
    let lo = if non_negative { 0 } else { -axis_len };
    crate::optim::resolver::discharge_index_in_range(b, idx, lo, axis_len, "static_array_setitem");
}

/// A slice assignment `a[..., i, ...] = v` with a runtime integer `i` on
/// some axes raises `IndexError` in NumPy unless `-n <= i < n`. Every
/// slice-assignment lowering selects the written cells by `i == coord`,
/// which matches nothing for an out-of-range `i`; without this check such
/// an `i` would prove the unchanged array.
fn discharge_runtime_single_indices(b: &mut IRBuilder, shape: &[usize], indices: &[SliceIndex]) {
    for (ax, idx) in indices.iter().enumerate() {
        if let SliceIndex::Single(v) = idx {
            if v.int_val().is_none() {
                discharge_write_index(b, v, shape[ax]);
            }
        }
    }
}

/// Numpy-wrap a runtime index for use in an address computation: negative
/// indices become `i + axis_len`. Statically-non-negative indices pass
/// through. Used on the memory-write fallback where `compute_addr` consumes
/// the index directly (unlike the overlay, whose `index_equals_coord` wraps
/// internally).
fn normalize_for_address(b: &mut IRBuilder, idx: &Value, axis_len: usize) -> Value {
    if idx.int_val().is_some() {
        return idx.clone();
    }
    normalize_dynamic_index(b, idx, axis_len)
}

fn and_all(b: &mut IRBuilder, parts: Vec<Value>) -> Value {
    let mut it = parts.into_iter();
    let Some(mut acc) = it.next() else {
        return b.ir_constant_bool(true);
    };
    for part in it {
        acc = b.ir_logical_and(&acc, &part);
    }
    acc
}

/// The offset of the element at the compile-time indices `idx_vals` (one
/// per axis; negative ones count from the end). An index outside its axis
/// is NumPy's `IndexError` where the write runs (`checked_axis_index`):
/// a compile error on a path that always runs; under a runtime predicate
/// the circuit asserts the write does not run and index 0 stands in. (The
/// offset used to be summed unchecked, so `a[0, 5] = v` on a 4x4 array
/// wrote `a[1, 1]`.)
fn constant_element_offset(
    b: &mut IRBuilder,
    shape: &[usize],
    strides: &[usize],
    idx_vals: &[Value],
) -> usize {
    idx_vals
        .iter()
        .enumerate()
        .map(|(ax, idx)| {
            let checked = super::base::checked_axis_index(b, idx, ax, shape[ax], "setitem");
            let i = checked.int_val().expect("a constant index stays constant") as usize;
            i * strides[ax]
        })
        .sum()
}

fn abs_offset_for_coords(offset: usize, strides: &[usize], coords: &[usize]) -> usize {
    let mut abs = offset;
    for (ax, coord) in coords.iter().enumerate() {
        abs += *coord * strides[ax];
    }
    abs
}

/// Compute a runtime address from `offset + sum_k(idx[k] * strides[k])`.
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

/// Setitem entry point for `Value::StaticArray`. Handles element setitem at
/// static and dynamic indices, multi-dim element setitem, and slice
/// setitem. Returns the (still segment-backed) array — segment writes are
/// in-place from the user's view, so we just hand the same shape/segment
/// back to the variable binding.
pub fn static_array_setitem(
    b: &mut IRBuilder,
    val: &Value,
    indices: &[SliceIndex],
    value: &Value,
) -> Value {
    let (dtype, shape, segment_id, strides, offset, imag_seg) = match val {
        Value::StaticArray {
            dtype,
            shape,
            segment_id,
            strides,
            offset,
            imag_segment_id,
            value_id: _,
        } => (
            *dtype,
            shape.clone(),
            *segment_id,
            strides.clone(),
            *offset,
            *imag_segment_id,
        ),
        _ => panic!("static_array_setitem: expected Value::StaticArray"),
    };

    if indices.is_empty() {
        return val.clone();
    }
    // A statically dead mutation must not update either storage or its
    // cached payload. ir_write_memory intentionally emits no write in this
    // case; caching its return value would otherwise replace the old cell.
    if b.write_predicate()
        .and_then(|p| p.bool_val().or_else(|| p.int_val().map(|v| v != 0)))
        == Some(false)
    {
        return val.clone();
    }
    // A tuple index: `a[np.nonzero(m)] = v` (also `np.where(m)`) is
    // NumPy's `a[m] = v`; any other tuple `t` addresses one entry per
    // axis, `a[t[0], t[1], ...] = v`.
    match super::read::tuple_index(b, indices, &shape) {
        super::read::TupleIndex::NotATuple => {}
        super::read::TupleIndex::Mask(mask) => {
            return super::mask::try_apply_boolean_mask_write(b, val, &mask, value)
                .expect("nonzero mask has the array's shape")
        }
        super::read::TupleIndex::Axes(expanded) => {
            return static_array_setitem(b, val, &expanded, value)
        }
    }

    // Expand a single Ellipsis (`...`) into the right number of full-range
    // slices, based on the source rank. Mirrors the read-path behaviour.
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
                SliceIndex::Ellipsis => {
                    panic!("an index can only have a single ellipsis ('...')");
                }
                other => expanded.push(other.clone()),
            }
        }
        return static_array_setitem(b, val, &expanded, value);
    }

    // NewAxis or other unusual indices on a setitem target: defer to the
    // legacy List path. Rare for StaticArray writes.
    if indices.iter().any(|i| matches!(i, SliceIndex::NewAxis)) {
        let lst = super::base::to_value_list(b, val);
        // The caller (do_recursive_assign) will run the legacy
        // set_nested_value path. Returning the materialised List signals
        // that path.
        return set_nested_via_list(b, &lst, indices, value);
    }

    let has_range = indices
        .iter()
        .any(|i| matches!(i, SliceIndex::Range(_, _, _)));

    // An index array on any axis, read by its dtype
    // (`array_ops::array_index_kind`): integer positions make this a
    // scatter, `val[i0[k], i1[k], ...] = rhs[k]` (`static_array_fancy_setitem`).
    // A boolean mask is written by the visitor's mask path
    // (`mask::try_apply_boolean_mask_write`); one that reaches here (with
    // other indices, or a value that path does not take) is rejected rather
    // than read as positions. Runtime-shaped index arrays are written
    // through the visitor too (`assignment::setitem_runtime_positions`).
    let mut any_fancy = false;
    for i in indices {
        let SliceIndex::Single(v) = i else { continue };
        match crate::helpers::array_ops::array_index_kind(b, v) {
            Some(crate::helpers::array_ops::ArrayIndexKind::Mask) => {
                panic!("this assignment through a boolean mask is not supported")
            }
            Some(_) if matches!(v, Value::DynamicNDArray(_)) => panic!(
                "assigning through a runtime-shaped index array combined with other \
                 indices is not supported"
            ),
            Some(crate::helpers::array_ops::ArrayIndexKind::Positions) => any_fancy = true,
            None => {}
        }
    }
    if any_fancy {
        return static_array_fancy_setitem(
            b, val, dtype, &shape, segment_id, &strides, offset, indices, value,
        );
    }

    if has_range {
        // Every real slice write goes through the segment (`static_array_
        // slice_setitem`), so the array stays an ndarray and views of it see
        // the write. (Large slices with compile-time bounds used to be
        // written on a materialised List that the variable was rebound to:
        // `a[:, :] = b` on a 40x40 `a` made `2 * a` repeat the list, and
        // large arrays with runtime bounds had them resolved to the full
        // axis, so `a[i:i + 2] = 5.0` wrote all of `a`.)
        // A runtime single index (`a[i, :] = v`) is checked here: both
        // lowerings below select the written cells by `i == coord`, which
        // matches nothing for an out-of-range `i`.
        discharge_runtime_single_indices(b, &shape, indices);
        if dtype == NumberType::Complex {
            return complex_slice_setitem_via_list(b, val, &shape, indices, value);
        }
        return static_array_slice_setitem(
            b, val, dtype, &shape, segment_id, &strides, offset, indices, value,
        );
    }

    // All-Single indices.
    if dtype == NumberType::Complex {
        return complex_element_setitem(
            b,
            val,
            &shape,
            segment_id,
            imag_seg.expect("Complex StaticArray missing imag_segment_id"),
            &strides,
            offset,
            indices,
            value,
        );
    }
    static_array_element_setitem(
        b, val, dtype, &shape, segment_id, &strides, offset, indices, value,
    )
}

/// Component-wise element setitem for a dual-segment Complex StaticArray.
/// Promotes the value to (real_f, imag_f) parts, then writes to the real
/// segment and imag segment at the same address.
#[allow(clippy::too_many_arguments)]
fn complex_element_setitem(
    b: &mut IRBuilder,
    val: &Value,
    shape: &[usize],
    real_seg: u32,
    imag_seg: u32,
    strides: &[usize],
    offset: usize,
    indices: &[SliceIndex],
    value: &Value,
) -> Value {
    // Convert each SliceIndex::Single into a Value (we already vetted this is
    // an all-Single setitem at the dispatcher).
    let idx_vals: Vec<Value> = indices
        .iter()
        .map(|s| match s {
            SliceIndex::Single(v) => v.clone(),
            _ => unreachable!("complex_element_setitem expects all-Single indices"),
        })
        .collect();

    // Partial indexing (`z[i] = row` on a 2-D `z`) writes the sub-array
    // `z[i]`: one element write per position of it, of the value broadcast
    // to its shape.
    if idx_vals.len() < shape.len() {
        let trailing = &shape[idx_vals.len()..];
        let trailing_strides = row_major_strides(trailing);
        for (flat, element) in region_values(b, value, trailing).iter().enumerate() {
            let mut full = indices.to_vec();
            for coord in decode_coords(flat, trailing, &trailing_strides) {
                full.push(SliceIndex::Single(b.ir_constant_int(coord as i64)));
            }
            complex_element_setitem(
                b, val, shape, real_seg, imag_seg, strides, offset, &full, element,
            );
        }
        return val.clone();
    }
    if idx_vals.len() > shape.len() {
        panic!(
            "too many indices for array: array is {}-dimensional, but {} were indexed",
            shape.len(),
            idx_vals.len()
        );
    }
    // Pull out per-component (real_f, imag_f) Float wires.
    let (re_v, im_v) = crate::helpers::value_ops::unpack_value_to_complex_parts(b, value);
    // Static fast path: known offset → both component writes are direct.
    let all_static = idx_vals.iter().all(|v| v.int_val().is_some());
    if all_static {
        materialize_before_predicated_write(b, val);
        let flat = (offset + constant_element_offset(b, shape, strides, &idx_vals)) as i64;
        let addr = b.ir_constant_int(flat);
        // Cache the values actually stored (under a conditional-write
        // predicate these are `select(p, new, old)`, not the raw inputs).
        let re_s = b.ir_write_memory(real_seg, &addr, &re_v);
        let im_s = b.ir_write_memory(imag_seg, &addr, &im_v);
        let abs = flat as usize;
        let new_complex = match (&re_s, &im_s) {
            (Value::Float(r), Value::Float(im)) => Value::Complex {
                real: r.clone(),
                imag: im.clone(),
            },
            _ => unreachable!(),
        };
        cache_set_cell(b, real_seg, abs, new_complex);
        return val.clone();
    }
    // Dynamic: check the indices (`-n <= i < n` where the write runs), drop
    // the cache, then issue two parallel ir_write_memory ops at the wrapped
    // address. (The address used to be formed from the raw indices, so a
    // negative or out-of-range index wrote outside the array.)
    discharge_dynamic_write_indices(b, shape, &idx_vals);
    let wrapped: Vec<Value> = idx_vals
        .iter()
        .enumerate()
        .map(|(ax, idx)| normalize_for_address(b, idx, shape[ax]))
        .collect();
    let addr = compute_addr(b, offset, strides, &wrapped, shape);
    b.ir_write_memory(real_seg, &addr, &re_v);
    b.ir_write_memory(imag_seg, &addr, &im_v);
    cache_invalidate(b, real_seg);
    val.clone()
}

/// Slice setitem for Complex StaticArray. Routes through materialise →
/// list-based set_nested_value → rebind. Mirrors the legacy Complex slice
/// behaviour; segment-write fidelity for slice writes on Complex is
/// covered by the round-trip cache (the result is rebuilt as a fresh
/// Complex StaticArray when possible).
fn complex_slice_setitem_via_list(
    b: &mut IRBuilder,
    val: &Value,
    shape: &[usize],
    indices: &[SliceIndex],
    value: &Value,
) -> Value {
    // The list writer resolves only compile-time indices; with a runtime
    // bound or index it would write the wrong cells (or none).
    let Some(region) = static_region_shape(shape, indices) else {
        panic!(
            "slice assignment into a complex array with a runtime index or slice bound is \
             not supported; use compile-time slice bounds"
        );
    };
    let lst = super::base::to_value_list(b, val);
    let value_lst = assigned_value_as_region_list(b, value, &region);
    let updated = legacy_set_nested(b, &lst, indices, &value_lst);
    // Rebound as a List the array would pick up Python list semantics.
    super::base::to_static_array(b, &updated).unwrap_or(updated)
}

/// Fallback used only when an exotic index (NewAxis) is encountered on a
/// setitem target — convert to a List and run the legacy set_nested_value
/// path via the caller.
fn set_nested_via_list(
    _b: &mut IRBuilder,
    val: &Value,
    _indices: &[SliceIndex],
    _value: &Value,
) -> Value {
    // The caller in do_recursive_assign will detect that we returned a
    // non-StaticArray and run the legacy set_nested_value path; we can
    // signal that by simply returning the List.
    val.clone()
}

/// Inline version of the legacy IRGenerator::set_nested_value, taking a
/// `Value::List` and producing an updated `Value::List`, for the complex
/// slice write. Replicated here (vs called from the IR generator) because
/// helpers can't reach back into the visitor layer cleanly. Handles
/// compile-time indices only, with a number or a value already broadcast
/// to the region's shape (`assigned_value_as_region_list`).
fn legacy_set_nested(
    b: &mut IRBuilder,
    current: &Value,
    indices: &[SliceIndex],
    value: &Value,
) -> Value {
    if indices.is_empty() {
        return value.clone();
    }
    let data = match current {
        Value::List(d) | Value::Tuple(d) => d.clone(),
        _ => return current.clone(),
    };
    let is_tuple = matches!(current, Value::Tuple(_));
    match &indices[0] {
        SliceIndex::Single(idx_val) => {
            if let Some(i) = idx_val.int_val() {
                let i = if i < 0 {
                    (data.values.len() as i64 + i) as usize
                } else {
                    i as usize
                };
                if i < data.values.len() {
                    let mut new_values = data.values.clone();
                    let mut new_types = data.elements_type.clone();
                    if indices.len() == 1 {
                        new_values[i] = value.clone();
                    } else {
                        new_values[i] = legacy_set_nested(b, &new_values[i], &indices[1..], value);
                    }
                    new_types[i] = new_values[i].zinnia_type();
                    return if is_tuple {
                        Value::Tuple(crate::types::CompositeData {
                            elements_type: new_types,
                            values: new_values,
                            value_id: ValueId::next(),
                        })
                    } else {
                        Value::List(crate::types::CompositeData {
                            elements_type: new_types,
                            values: new_values,
                            value_id: ValueId::next(),
                        })
                    };
                }
            }
            current.clone()
        }
        SliceIndex::Range(start, stop, step) => {
            let coords = super::super::shape_arith::resolve_slice(
                data.values.len(),
                start.as_ref().and_then(|v| v.int_val()),
                stop.as_ref().and_then(|v| v.int_val()),
                step.as_ref().and_then(|v| v.int_val()),
            );

            let mut new_values = data.values.clone();
            let mut new_types = data.elements_type.clone();
            if indices.len() > 1 {
                if let Value::List(rhs_data) | Value::Tuple(rhs_data) = value {
                    for (rhs_idx, &i) in coords.iter().enumerate() {
                        if rhs_idx >= rhs_data.values.len() {
                            break;
                        }
                        new_values[i] = legacy_set_nested(
                            b,
                            &new_values[i],
                            &indices[1..],
                            &rhs_data.values[rhs_idx],
                        );
                        new_types[i] = new_values[i].zinnia_type();
                    }
                } else {
                    // Scalar RHS — broadcast inner.
                    for &i in &coords {
                        new_values[i] = legacy_set_nested(b, &new_values[i], &indices[1..], value);
                        new_types[i] = new_values[i].zinnia_type();
                    }
                }
            } else if let Value::List(rhs_data) | Value::Tuple(rhs_data) = value {
                for (rhs_idx, &i) in coords.iter().enumerate() {
                    if rhs_idx >= rhs_data.values.len() {
                        break;
                    }
                    new_values[i] = rhs_data.values[rhs_idx].clone();
                    new_types[i] = new_values[i].zinnia_type();
                }
            } else {
                for &i in &coords {
                    new_values[i] = value.clone();
                    new_types[i] = new_values[i].zinnia_type();
                }
            }
            if is_tuple {
                Value::Tuple(crate::types::CompositeData {
                    elements_type: new_types,
                    values: new_values,
                    value_id: ValueId::next(),
                })
            } else {
                Value::List(crate::types::CompositeData {
                    elements_type: new_types,
                    values: new_values,
                    value_id: ValueId::next(),
                })
            }
        }
        _ => current.clone(),
    }
}

// ────────────────────────────────────────────────────────────────────────
// Broadcasting the assigned value of a slice assignment
// ────────────────────────────────────────────────────────────────────────

/// The shape of the region `a[indices]` selects in an array of `shape`
/// when every index is a compile-time constant (`None` otherwise): one
/// axis per slice and per trailing unindexed axis; an integer index drops
/// its axis.
fn static_region_shape(shape: &[usize], indices: &[SliceIndex]) -> Option<Vec<usize>> {
    let constant = |bound: &Option<Value>| match bound {
        None | Some(Value::None) => Some(None),
        Some(v) => v.int_val().map(Some),
    };
    let mut region = Vec::new();
    for (axis, index) in indices.iter().enumerate() {
        match index {
            SliceIndex::Single(v) => {
                v.int_val()?;
            }
            SliceIndex::Range(start, stop, step) => {
                let coords = super::super::shape_arith::resolve_slice(
                    shape[axis],
                    constant(start)?,
                    constant(stop)?,
                    constant(step)?,
                );
                region.push(coords.len());
            }
            _ => return None,
        }
    }
    region.extend_from_slice(&shape[indices.len().min(shape.len())..]);
    Some(region)
}

/// NumPy's broadcast of an assigned value to the region a slice assignment
/// writes (`a[idx] = v`). Aligned at the trailing axes, each axis of `v`'s
/// shape (`source`) must equal the region's or be 1 (repeated along it);
/// axes of `v` beyond the region's rank must be 1. `elements` are `v`'s
/// elements in row-major order; the result is the value written at each
/// position of the region, in row-major order. The error is NumPy's.
fn broadcast_assigned_elements(
    elements: &[Value],
    source: &[usize],
    region: &[usize],
) -> Result<Vec<Value>, String> {
    let error = || {
        format!(
            "ValueError: could not broadcast input array from shape {:?} into shape {:?}",
            source, region
        )
    };
    if elements.len() != source.iter().product::<usize>() {
        return Err(error());
    }
    let extra = source.len().saturating_sub(region.len());
    if source[..extra].iter().any(|&dim| dim != 1) {
        return Err(error());
    }
    let source = &source[extra..];
    let pad = region.len() - source.len();
    let source_strides = row_major_strides(source);
    // Position in `elements` per region axis: 0 along a repeated axis.
    let mut strides = vec![0usize; region.len()];
    for (axis, &dim) in source.iter().enumerate() {
        if dim == region[pad + axis] {
            strides[pad + axis] = source_strides[axis];
        } else if dim != 1 {
            return Err(error());
        }
    }
    let region_strides = row_major_strides(region);
    let total: usize = region.iter().product();
    Ok((0..total)
        .map(|flat| {
            let coords = decode_coords(flat, region, &region_strides);
            let at: usize = coords.iter().zip(&strides).map(|(c, s)| c * s).sum();
            elements[at].clone()
        })
        .collect())
}

/// The elements written into a region of shape `region` (the native slice
/// writer's order) from the array-like value `value` with `elements`.
/// A value with a static shape is broadcast; one written into a region with
/// runtime bounds (whose shape here is the largest possible) or a value
/// without a static shape must have exactly one element per position.
fn region_elements(
    value: &Value,
    elements: Vec<Value>,
    region: &[usize],
    runtime_bounds: bool,
) -> Vec<Value> {
    let source = match value {
        Value::StaticArray { shape, .. } => Some(shape.clone()),
        Value::List(_) | Value::Tuple(_) => {
            Some(super::super::composite::get_composite_shape(value))
        }
        _ => None,
    };
    match source {
        Some(source) if !runtime_bounds => {
            broadcast_assigned_elements(&elements, &source, region).unwrap_or_else(|e| panic!("{e}"))
        }
        _ if elements.len() == region.iter().product::<usize>() => elements,
        _ => panic!(
            "slice assignment: the value has {} elements but the target region has shape {:?}; \
             broadcasting into a slice with runtime bounds, or from an array with a runtime \
             shape, is not supported",
            elements.len(),
            region
        ),
    }
}

/// The value written at each position of a region of shape `region`, in
/// row-major order: a number at every position, an array-like value
/// broadcast to the region (`broadcast_assigned_elements`).
fn region_values(b: &mut IRBuilder, value: &Value, region: &[usize]) -> Vec<Value> {
    let (elements, source) = match value {
        Value::StaticArray { shape, .. } => {
            (super::elementwise::payload_cells(b, value), shape.clone())
        }
        Value::List(_) | Value::Tuple(_) => (
            super::super::composite::flatten_composite_with_builder(b, value),
            super::super::composite::get_composite_shape(value),
        ),
        Value::DynamicNDArray(_) => super::super::composite::reject_runtime_shaped_operand(),
        number => return vec![number.clone(); region.iter().product()],
    };
    broadcast_assigned_elements(&elements, &source, region).unwrap_or_else(|e| panic!("{e}"))
}

/// The assigned value for the list-based slice writer: a number as it is,
/// an array-like value broadcast to a nested list of the region's shape.
fn assigned_value_as_region_list(b: &mut IRBuilder, value: &Value, region: &[usize]) -> Value {
    if !matches!(
        value,
        Value::StaticArray { .. } | Value::List(_) | Value::Tuple(_) | Value::DynamicNDArray(_)
    ) {
        return value.clone();
    }
    let broadcast = region_values(b, value, region);
    let types = broadcast.iter().map(|v| v.zinnia_type()).collect();
    super::super::composite::build_nested_value(broadcast, types, region)
}

// ────────────────────────────────────────────────────────────────────────
// Fancy-index setitem
// ────────────────────────────────────────────────────────────────────────

/// Cells in row-major order as the value of that shape: the number itself
/// for shape `()`, else a nested list.
fn nested_cells(cells: Vec<Value>, shape: &[usize]) -> Value {
    if shape.is_empty() {
        return cells.into_iter().next().expect("a 0-d value has one cell");
    }
    let types = cells.iter().map(|v| v.zinnia_type()).collect();
    super::super::composite::build_nested_value(cells, types, shape)
}

/// The rows (first-axis items) of a value of `shape` given its cells.
fn cells_as_rows(cells: Vec<Value>, shape: &[usize]) -> Vec<Value> {
    let row_len: usize = shape[1..].iter().product();
    (0..shape[0])
        .map(|k| nested_cells(cells[k * row_len..(k + 1) * row_len].to_vec(), &shape[1..]))
        .collect()
}

/// Extract the per-element Values of an integer index source. Accepts a
/// StaticArray (segment-backed) or a List/Tuple (composite-backed).
fn read_index_array_elements(b: &mut IRBuilder, src: &Value) -> Vec<Value> {
    match src {
        Value::StaticArray { .. } => {
            let list = super::base::to_value_list(b, src);
            match list {
                Value::List(d) => d.values,
                _ => vec![],
            }
        }
        Value::List(d) | Value::Tuple(d) => d.values.clone(),
        _ => vec![],
    }
}

/// Scatter writes for fancy-index assignment on a StaticArray destination.
/// Handles the NumPy semantics:
///   for k in 0..L: dst[idx_axes[k]] = rhs[k] (or rhs broadcast scalar).
/// Per-axis indices may mix fancy (array-valued Single) with simple Single
/// scalars or Range slices; non-fancy axes are carried through unchanged.
#[allow(clippy::too_many_arguments)]
fn static_array_fancy_setitem(
    b: &mut IRBuilder,
    val: &Value,
    _dtype: NumberType,
    shape: &[usize],
    _segment_id: u32,
    _strides: &[usize],
    _offset: usize,
    indices: &[SliceIndex],
    value: &Value,
) -> Value {
    // Collect per-axis fancy index materializations; None for non-fancy axes.
    let mut fancy_arrays: Vec<Option<Vec<Value>>> = Vec::with_capacity(indices.len());
    let mut fancy_len: Option<usize> = None;
    for idx in indices {
        let elems = match idx {
            SliceIndex::Single(v @ Value::StaticArray { .. })
            | SliceIndex::Single(v @ Value::List(_))
            | SliceIndex::Single(v @ Value::Tuple(_)) => {
                // NumPy reads the index list `[True, 2]` as positions `[1, 2]`.
                let entries = read_index_array_elements(b, v);
                Some(
                    entries
                        .iter()
                        .map(|e| if matches!(e, Value::Boolean(_)) { b.ir_int_cast(e) } else { e.clone() })
                        .collect::<Vec<_>>(),
                )
            }
            _ => None,
        };
        if let Some(ref es) = elems {
            if let Some(prev) = fancy_len {
                assert!(
                    prev == es.len(),
                    "fancy-index axes must broadcast: got lengths {} and {}",
                    prev,
                    es.len()
                );
            } else {
                fancy_len = Some(es.len());
            }
        }
        fancy_arrays.push(elems);
    }
    let l = fancy_len.expect("static_array_fancy_setitem called without a fancy axis");

    // NumPy writes `value`, broadcast to the shape `(l, *region)`, where
    // `region` is what each index `k` selects: a value with one axis more
    // than `region` and `l` rows gives row `k` to index `k`; a value of the
    // region's rank or less is broadcast to every `k` (the per-`k` write
    // checks it). (A value of rank 2 used to be handed whole to every `k`
    // and rejected, `M[[i, j]] = M[[j, i]]` (pythran lu); a 1-D value
    // with `l` elements fed element `k` to row `k`, where NumPy broadcasts
    // it to every row.)
    let region_rank = indices.iter().filter(|i| matches!(i, SliceIndex::Range(..))).count()
        + shape.len().saturating_sub(indices.len());
    let value_shape = match value {
        Value::StaticArray { shape, .. } => shape.clone(),
        Value::List(_) | Value::Tuple(_) => super::super::composite::get_composite_shape(value),
        _ => Vec::new(),
    };
    // The value's cells are read once, before any write: NumPy evaluates
    // the right-hand side first, so `M[[1, 2]] = M[0:2]` writes the old rows.
    let value_cells = match value {
        Value::StaticArray { .. } => Some(super::elementwise::payload_cells(b, value)),
        Value::List(_) | Value::Tuple(_) => {
            Some(super::super::composite::flatten_composite_with_builder(b, value))
        }
        _ => None,
    };
    let value_per_k: Vec<Value> = match value_cells {
        Some(cells) if value_shape.len() == region_rank + 1 && value_shape[0] == l => {
            cells_as_rows(cells, &value_shape)
        }
        Some(cells) => vec![nested_cells(cells, &value_shape); l],
        None => vec![value.clone(); l],
    };

    for k in 0..l {
        let mut k_indices: Vec<SliceIndex> = Vec::with_capacity(indices.len());
        for (ax, idx) in indices.iter().enumerate() {
            match &fancy_arrays[ax] {
                Some(arr) => k_indices.push(SliceIndex::Single(arr[k].clone())),
                None => k_indices.push(idx.clone()),
            }
        }
        let _ = static_array_setitem(b, val, &k_indices, &value_per_k[k]);
    }

    val.clone()
}

// ────────────────────────────────────────────────────────────────────────
// Element setitem
// ────────────────────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn static_array_element_setitem(
    b: &mut IRBuilder,
    val: &Value,
    dtype: NumberType,
    shape: &[usize],
    segment_id: u32,
    strides: &[usize],
    offset: usize,
    indices: &[SliceIndex],
    value: &Value,
) -> Value {
    // Extract index Values out of SliceIndex::Single.
    let idx_vals: Vec<Value> = indices
        .iter()
        .map(|s| match s {
            SliceIndex::Single(v) => v.clone(),
            _ => unreachable!("static_array_element_setitem: non-Single index"),
        })
        .collect();

    let rank = shape.len();

    // Partial indexing (fewer indices than rank): the LHS targets a row /
    // sub-array. Rewrite as a slice-setitem with trailing full ranges.
    if idx_vals.len() < rank {
        let mut full: Vec<SliceIndex> = idx_vals.iter().cloned().map(SliceIndex::Single).collect();
        for _ in idx_vals.len()..rank {
            full.push(SliceIndex::Range(None, None, None));
        }
        return static_array_slice_setitem(
            b, val, dtype, shape, segment_id, strides, offset, &full, value,
        );
    }

    if idx_vals.len() > rank {
        panic!(
            "too many indices for array: array is {}-dimensional, but {} were indexed",
            rank,
            idx_vals.len()
        );
    }

    // Empty-composite RHS (e.g. matmul of two empty 1-D arrays evaluates
    // to an empty list at the IR layer in some legacy code paths) →
    // there's nothing to write. Match the legacy `set_nested_value` no-op
    // semantics rather than panicking inside `cast_to_dtype`.
    if let Value::List(d) | Value::Tuple(d) = value {
        if d.values.is_empty() {
            return val.clone();
        }
    }

    // Cast the RHS value to the segment's dtype.
    let cast_val = cast_for_segment(b, segment_id, value, dtype);

    // All-static path: use the cached-wire-update strategy.
    let all_static = idx_vals.iter().all(|v| v.int_val().is_some());
    if all_static {
        materialize_before_predicated_write(b, val);
        let abs_offset = offset + constant_element_offset(b, shape, strides, &idx_vals);
        let addr = b.ir_constant_int(abs_offset as i64);
        // Cache what was actually stored (predicated under a branch).
        let stored = b.ir_write_memory(segment_id, &addr, &cast_val);
        cache_set_cell(b, segment_id, abs_offset, stored);
        return val.clone();
    }

    // Dynamic-index write chokepoint. Discharge numpy's validity window
    // `-n <= i < n` per axis BEFORE any of the three dynamic write paths
    // (overlay, memory-write fallback, fancy scatter — which recurses back
    // through here). Without this, an OOB dynamic write silently no-ops on
    // the overlay path or writes an unconstrained address on the memory-write
    // fallback; the read path has discharged at its chokepoint all along.
    // See C09.
    discharge_dynamic_write_indices(b, shape, &idx_vals);

    // Overlay path: `index_equals_coord` numpy-wraps each index internally,
    // so it consumes the raw (possibly-negative) indices directly.
    if try_overlay_dynamic_element_setitem(
        b, dtype, shape, segment_id, strides, offset, &idx_vals, &cast_val,
    ) {
        return val.clone();
    }

    // At least one dynamic index → invalidate cache, single segment write.
    // The memory-write `compute_addr` does NOT wrap dynamic indices, so we
    // numpy-wrap each axis here before forming the address.
    super::base::materialize_static_array_if_lazy(b, val);
    let wrapped: Vec<Value> = idx_vals
        .iter()
        .enumerate()
        .map(|(ax, idx)| normalize_for_address(b, idx, shape[ax]))
        .collect();
    let addr = compute_addr(b, offset, strides, &wrapped, shape);
    b.ir_write_memory(segment_id, &addr, &cast_val);
    cache_invalidate(b, segment_id);
    val.clone()
}

fn try_overlay_dynamic_element_setitem(
    b: &mut IRBuilder,
    dtype: NumberType,
    shape: &[usize],
    segment_id: u32,
    strides: &[usize],
    offset: usize,
    idx_vals: &[Value],
    cast_val: &Value,
) -> bool {
    if dtype == NumberType::Complex {
        return false;
    }
    let total: usize = shape.iter().product();
    if total > SMALL_DYNAMIC_WRITE_OVERLAY_LIMIT {
        return false;
    }
    if b.static_array_product_provenance.contains_key(&segment_id) {
        super::base::force_static_array_product_provenance(b, segment_id);
    }
    let Some(mut cells) = b.static_array_payload.get(&segment_id).cloned() else {
        return false;
    };
    let logical_strides = row_major_strides(shape);
    for flat in 0..total {
        let coords = decode_coords(flat, shape, &logical_strides);
        let abs = abs_offset_for_coords(offset, strides, &coords);
        if abs >= cells.len() {
            return false;
        }
        let mut conds = Vec::with_capacity(idx_vals.len());
        for (ax, idx) in idx_vals.iter().enumerate() {
            conds.push(index_equals_coord(b, idx, coords[ax], shape[ax]));
        }
        // This path mutates only the compile-time overlay (no
        // ir_write_memory until lazy materialisation, which may happen
        // outside the branch), so the conditional-write predicate must
        // be folded into the per-cell select here.
        if let Some(p) = b.write_predicate().cloned() {
            let static_true = p
                .bool_val()
                .or_else(|| p.int_val().map(|v| v != 0))
                .unwrap_or(false);
            if !static_true {
                conds.push(p);
            }
        }
        let cond = and_all(b, conds);
        let old = cells[abs].clone();
        cells[abs] = select_typed(b, dtype, &cond, cast_val, &old);
    }
    b.static_array_payload.insert(segment_id, cells);
    b.lazy_static_array_segments.insert(segment_id);
    true
}

// ────────────────────────────────────────────────────────────────────────
// Slice setitem
// ────────────────────────────────────────────────────────────────────────

/// Per-axis specification for slice setitem. Mirrors the structure used by
/// `dyn_setitem_slice` for `DynamicNDArrayData`.
enum AxisCoords {
    /// One coordinate (consumes the axis in the LHS).
    Single(Value),
    /// Compile-time-known list of coordinates.
    Static(Vec<usize>),
    /// Runtime-bound range; resolved per output position with masking.
    Dynamic {
        start: Value,
        stop: Value,
        step: Value,
        axis_len: usize,
    },
}

#[allow(clippy::too_many_arguments)]
fn static_array_slice_setitem(
    b: &mut IRBuilder,
    val: &Value,
    dtype: NumberType,
    shape: &[usize],
    segment_id: u32,
    strides: &[usize],
    offset: usize,
    indices: &[SliceIndex],
    value: &Value,
) -> Value {
    let rank = shape.len();
    if indices.len() > rank {
        panic!(
            "too many indices for array: array is {}-dimensional, but {} were indexed",
            rank,
            indices.len()
        );
    }

    // Build per-axis coord specs and the LHS value shape.
    let mut axis_specs: Vec<(usize, AxisCoords)> = Vec::new(); // (src_axis, spec)
    let mut value_shape: Vec<usize> = Vec::new();
    let mut has_dynamic_range = false;

    for (ax, idx) in indices.iter().enumerate() {
        match idx {
            SliceIndex::Single(v) => {
                axis_specs.push((ax, AxisCoords::Single(v.clone())));
            }
            SliceIndex::Range(start, stop, step) => {
                let dim = shape[ax] as i64;
                let resolve = |v: &Option<Value>| -> Option<i64> {
                    match v.as_ref() {
                        Some(Value::None) | None => None,
                        Some(val) => val.int_val(),
                    }
                };
                let s_static = resolve(start);
                let e_static = resolve(stop);
                let st_static = resolve(step);

                let is_present = |v: &Option<Value>| -> bool {
                    matches!(v, Some(val) if !matches!(val, Value::None))
                };
                let all_static = (!is_present(start) || s_static.is_some())
                    && (!is_present(stop) || e_static.is_some())
                    && (!is_present(step) || st_static.is_some());

                if all_static {
                    let coords = super::super::shape_arith::resolve_slice(
                        shape[ax], s_static, e_static, st_static,
                    );
                    value_shape.push(coords.len());
                    axis_specs.push((ax, AxisCoords::Static(coords)));
                } else {
                    has_dynamic_range = true;
                    fn to_ir(b: &mut IRBuilder, v: &Option<Value>, default: i64) -> Value {
                        match v.as_ref() {
                            Some(val) if !matches!(val, Value::None) => {
                                if let Some(s) = val.int_val() {
                                    b.ir_constant_int(s)
                                } else {
                                    val.clone()
                                }
                            }
                            _ => b.ir_constant_int(default),
                        }
                    }
                    let start_ir = to_ir(b, start, 0);
                    let stop_ir = to_ir(b, stop, dim);
                    let step_ir = to_ir(b, step, 1);
                    let max_len = shape[ax];
                    value_shape.push(max_len);
                    axis_specs.push((
                        ax,
                        AxisCoords::Dynamic {
                            start: start_ir,
                            stop: stop_ir,
                            step: step_ir,
                            axis_len: shape[ax],
                        },
                    ));
                }
            }
            _ => panic!("static_array slice setitem: Ellipsis/NewAxis already handled"),
        }
    }

    // Trailing axes default to full range.
    for ax in indices.len()..rank {
        let coords: Vec<usize> = (0..shape[ax]).collect();
        value_shape.push(coords.len());
        axis_specs.push((ax, AxisCoords::Static(coords)));
    }

    // Materialise RHS values.
    let value_is_scalar = value.is_number();
    let value_elements: Vec<Value> = if value_is_scalar {
        Vec::new()
    } else {
        match value {
            Value::StaticArray { .. } => {
                // Use to_value_list to flatten — the cache-aware materialiser
                // returns the original wires when possible.
                super::elementwise::payload_cells(b, value)
            }
            Value::DynamicNDArray(vd) => {
                let vd_owned = crate::ops::dyn_ndarray::force_contiguous(b, vd);
                super::super::segment::read_all(b, vd_owned.segment_id, vd_owned.max_length())
            }
            Value::List(_) | Value::Tuple(_) => {
                super::super::composite::flatten_composite_with_builder(b, value)
            }
            _ => panic!("slice assignment: value must be scalar or array-like"),
        }
    };

    // NumPy broadcasts the value to the region the indices select.
    let value_elements = if value_is_scalar {
        value_elements
    } else {
        region_elements(value, value_elements, &value_shape, has_dynamic_range)
    };

    let value_total: usize = value_shape.iter().product();
    let value_strides = row_major_strides(&value_shape);

    if !has_dynamic_range {
        let has_dynamic_single = axis_specs
            .iter()
            .any(|(_, s)| matches!(s, AxisCoords::Single(v) if v.int_val().is_none()));
        if has_dynamic_single
            && try_overlay_dynamic_single_slice_setitem(
                b,
                dtype,
                shape,
                segment_id,
                strides,
                offset,
                &axis_specs,
                &value_shape,
                &value_strides,
                value_is_scalar,
                value,
                &value_elements,
            )
        {
            return val.clone();
        }
        if has_dynamic_single {
            super::base::materialize_static_array_if_lazy(b, val);
        } else {
            materialize_before_predicated_write(b, val);
        }

        // All static bounds: iterate over target positions and emit
        // constant-address writes. This is also the fast / contiguous-view
        // path mentioned in the card decisions: each LHS cell is written
        // with a single segment write to the underlying segment, so a slice
        // write into a contiguous-view target writes through with no
        // intermediate segment.
        for val_flat in 0..value_total {
            let val_coords = decode_coords(val_flat, &value_shape, &value_strides);
            let mut addr_static: i64 = offset as i64;
            let mut val_coord_idx = 0;
            let mut had_dynamic_single = false;
            let mut dynamic_addr_parts: Vec<Value> = Vec::new();

            for (src_axis, spec) in &axis_specs {
                let stride = strides[*src_axis] as i64;
                match spec {
                    AxisCoords::Single(v) => {
                        if let Some(i) = v.int_val() {
                            let i = if i < 0 {
                                shape[*src_axis] as i64 + i
                            } else {
                                i
                            };
                            addr_static += i * stride;
                        } else {
                            had_dynamic_single = true;
                            let stride_val = b.ir_constant_int(stride);
                            dynamic_addr_parts.push(b.ir_mul_i(v, &stride_val));
                        }
                    }
                    AxisCoords::Static(coords) => {
                        let coord = coords[val_coords[val_coord_idx]];
                        addr_static += coord as i64 * stride;
                        val_coord_idx += 1;
                    }
                    AxisCoords::Dynamic { .. } => unreachable!("checked above"),
                }
            }

            let addr = if had_dynamic_single {
                let mut acc = b.ir_constant_int(addr_static);
                for p in &dynamic_addr_parts {
                    acc = b.ir_add_i(&acc, p);
                }
                acc
            } else {
                b.ir_constant_int(addr_static)
            };

            let write_val = if value_is_scalar {
                cast_for_segment(b, segment_id, value, dtype)
            } else {
                cast_for_segment(b, segment_id, &value_elements[val_flat], dtype)
            };
            // Cache what was actually stored (predicated under a branch).
            let stored = b.ir_write_memory(segment_id, &addr, &write_val);

            if !had_dynamic_single {
                cache_set_cell(b, segment_id, addr_static as usize, stored);
            }
        }
        if axis_specs
            .iter()
            .any(|(_, s)| matches!(s, AxisCoords::Single(v) if v.int_val().is_none()))
        {
            // A dynamic Single index made addresses runtime-only.
            cache_invalidate(b, segment_id);
        }
    } else {
        super::base::materialize_static_array_if_lazy(b, val);
        // Has dynamic Range bounds: iterate over max positions, mask out
        // out-of-bounds writes by reading-then-selecting-then-writing.
        for val_flat in 0..value_total {
            let val_coords = decode_coords(val_flat, &value_shape, &value_strides);

            let mut addr_parts_static: i64 = offset as i64;
            let mut addr_parts_dynamic: Vec<Value> = Vec::new();
            let mut in_bounds_parts: Vec<Value> = Vec::new();
            let mut val_coord_idx = 0;

            for (src_axis, spec) in &axis_specs {
                let stride = strides[*src_axis] as i64;
                match spec {
                    AxisCoords::Single(v) => {
                        if let Some(i) = v.int_val() {
                            let i = if i < 0 {
                                shape[*src_axis] as i64 + i
                            } else {
                                i
                            };
                            addr_parts_static += i * stride;
                        } else {
                            let stride_val = b.ir_constant_int(stride);
                            addr_parts_dynamic.push(b.ir_mul_i(v, &stride_val));
                        }
                    }
                    AxisCoords::Static(coords) => {
                        let coord = coords[val_coords[val_coord_idx]];
                        addr_parts_static += coord as i64 * stride;
                        val_coord_idx += 1;
                    }
                    AxisCoords::Dynamic {
                        start,
                        stop,
                        step,
                        axis_len,
                    } => {
                        let idx_in_slice = val_coords[val_coord_idx] as i64;
                        let idx_const = b.ir_constant_int(idx_in_slice);
                        let off2 = b.ir_mul_i(&idx_const, step);
                        let src_idx = b.ir_add_i(start, &off2);

                        let zero = b.ir_constant_int(0);
                        let len_val = b.ir_constant_int(*axis_len as i64);
                        let ge_zero = b.ir_greater_than_or_equal_i(&src_idx, &zero);
                        let lt_len = b.ir_less_than_i(&src_idx, &len_val);
                        let in_range = b.ir_logical_and(&ge_zero, &lt_len);
                        let step_pos = b.ir_greater_than_i(step, &zero);
                        let lt_stop = b.ir_less_than_i(&src_idx, stop);
                        let gt_stop = b.ir_greater_than_i(&src_idx, stop);
                        let stop_ok = b.ir_select_i(&step_pos, &lt_stop, &gt_stop);
                        let stop_bool = b.ir_bool_cast(&stop_ok);
                        let in_bounds = b.ir_logical_and(&in_range, &stop_bool);
                        in_bounds_parts.push(in_bounds);

                        let max_idx = b.ir_constant_int(*axis_len as i64 - 1);
                        let is_neg = b.ir_less_than_i(&src_idx, &zero);
                        let is_over = b.ir_greater_than_i(&src_idx, &max_idx);
                        let clamped_hi = b.ir_select_i(&is_over, &max_idx, &src_idx);
                        let clamped = b.ir_select_i(&is_neg, &zero, &clamped_hi);

                        let stride_val = b.ir_constant_int(stride);
                        addr_parts_dynamic.push(b.ir_mul_i(&clamped, &stride_val));
                        val_coord_idx += 1;
                    }
                }
            }

            let mut addr = b.ir_constant_int(addr_parts_static);
            for part in &addr_parts_dynamic {
                addr = b.ir_add_i(&addr, part);
            }

            let in_bounds = if in_bounds_parts.is_empty() {
                b.ir_constant_bool(true)
            } else {
                let mut acc = in_bounds_parts[0].clone();
                for ib in &in_bounds_parts[1..] {
                    acc = b.ir_logical_and(&acc, ib);
                }
                acc
            };

            let current = b.ir_read_memory(segment_id, &addr);
            let write_val = if value_is_scalar {
                cast_for_segment(b, segment_id, value, dtype)
            } else {
                cast_for_segment(b, segment_id, &value_elements[val_flat], dtype)
            };
            let selected = if dtype == NumberType::Float {
                b.ir_select_f(&in_bounds, &write_val, &current)
            } else {
                b.ir_select_i(&in_bounds, &write_val, &current)
            };
            b.ir_write_memory(segment_id, &addr, &selected);
        }
        cache_invalidate(b, segment_id);
    }

    val.clone()
}

#[allow(clippy::too_many_arguments)]
/// A slice write through exactly one runtime index (`a[i, :] = v`; the
/// other axes static), lowered by the scatter family: every cell it may
/// touch becomes an `OpKind::IndexWrite` lane over the index's pinned
/// one-hot (`lower::scatter::write`), which also proves `-n <= i < n` (a
/// slice write had no bounds check of its own). Integer and float arrays.
struct OneHotWriteAxis {
    axis: usize,
    shared: std::sync::Arc<[Value]>,
    kind: crate::lower::OpKind,
    /// The write's execution predicate, when it is a runtime value.
    predicate: Option<Value>,
}

impl OneHotWriteAxis {
    fn of(b: &mut IRBuilder, dtype: NumberType, shape: &[usize], axis_specs: &[(usize, AxisCoords)]) -> Option<Self> {
        if !matches!(dtype, NumberType::Integer | NumberType::Float) {
            return None;
        }
        let mut runtime = axis_specs.iter().filter_map(|(axis, spec)| match spec {
            AxisCoords::Single(v) if v.int_val().is_none() => Some((*axis, v)),
            _ => None,
        });
        let (axis, index) = runtime.next()?;
        if runtime.next().is_some() || axis_specs.iter().any(|(_, s)| matches!(s, AxisCoords::Dynamic { .. })) {
            return None;
        }
        let predicate = b.write_predicate().cloned().filter(|p| {
            p.bool_val().or_else(|| p.int_val().map(|v| v != 0)) != Some(true)
        });
        let index = b.guard_operand(index, 0);
        let index = b.ensure_ptr(&index);
        let kind = crate::lower::OpKind::IndexWrite { axis_len: shape[axis] as u32 };
        Some(OneHotWriteAxis { axis, shared: std::sync::Arc::from(vec![index]), kind, predicate })
    }

    /// The new value of the cell at `coords` (holding `old`).
    fn cell(&self, b: &mut IRBuilder, coords: &[usize], old: &Value, written: &Value) -> Value {
        let mut own = vec![old.clone(), written.clone()];
        if let Some(p) = &self.predicate {
            own.push(p.clone());
        }
        let op = crate::lower::ArrayOp {
            kind: self.kind,
            shared: 1,
            output: 0,
            position: Some(coords[self.axis] as u32),
        };
        b.ir_array_op(op, &self.shared, &own)
    }
}

fn try_overlay_dynamic_single_slice_setitem(
    b: &mut IRBuilder,
    dtype: NumberType,
    shape: &[usize],
    segment_id: u32,
    strides: &[usize],
    offset: usize,
    axis_specs: &[(usize, AxisCoords)],
    value_shape: &[usize],
    value_strides: &[usize],
    value_is_scalar: bool,
    value: &Value,
    value_elements: &[Value],
) -> bool {
    if dtype == NumberType::Complex {
        return false;
    }
    let total: usize = shape.iter().product();
    if total > SMALL_DYNAMIC_WRITE_OVERLAY_LIMIT {
        return false;
    }
    if b.static_array_product_provenance.contains_key(&segment_id) {
        super::base::force_static_array_product_provenance(b, segment_id);
    }
    let Some(mut cells) = b.static_array_payload.get(&segment_id).cloned() else {
        return false;
    };
    let logical_strides = row_major_strides(shape);
    let one_hot_write = OneHotWriteAxis::of(b, dtype, shape, axis_specs);
    for flat in 0..total {
        let coords = decode_coords(flat, shape, &logical_strides);
        let abs = abs_offset_for_coords(offset, strides, &coords);
        if abs >= cells.len() {
            return false;
        }

        let mut conds: Vec<Value> = Vec::new();
        let mut rhs_coords: Vec<usize> = Vec::with_capacity(value_shape.len());
        let mut targeted = true;
        for (src_axis, spec) in axis_specs {
            match spec {
                // The one-hot write tests its runtime index itself.
                AxisCoords::Single(_) if one_hot_write.as_ref().is_some_and(|w| w.axis == *src_axis) => {}
                AxisCoords::Single(v) => {
                    conds.push(index_equals_coord(
                        b,
                        v,
                        coords[*src_axis],
                        shape[*src_axis],
                    ));
                }
                AxisCoords::Static(valid_coords) => {
                    if let Some(pos) = valid_coords.iter().position(|c| *c == coords[*src_axis]) {
                        rhs_coords.push(pos);
                    } else {
                        targeted = false;
                        break;
                    }
                }
                AxisCoords::Dynamic { .. } => return false,
            }
        }
        if !targeted {
            continue;
        }
        if let Some(axis) = &one_hot_write {
            // The other indices are constants here: their tests are known.
            if conds.iter().any(|c| c.bool_val() == Some(false)) {
                continue;
            }
            let raw_write = if value_is_scalar {
                value.clone()
            } else {
                let rhs_flat: usize = rhs_coords.iter().zip(value_strides).map(|(c, s)| c * s).sum();
                value_elements[rhs_flat].clone()
            };
            let write_val = cast_for_segment(b, segment_id, &raw_write, dtype);
            cells[abs] = axis.cell(b, &coords, &cells[abs], &write_val);
            continue;
        }
        // Compile-time-overlay-only path: fold the conditional-write
        // predicate into the per-cell select (same as
        // `try_overlay_dynamic_element_setitem`).
        if let Some(p) = b.write_predicate().cloned() {
            let static_true = p
                .bool_val()
                .or_else(|| p.int_val().map(|v| v != 0))
                .unwrap_or(false);
            if !static_true {
                conds.push(p);
            }
        }
        let cond = and_all(b, conds);
        let raw_write = if value_is_scalar {
            value.clone()
        } else {
            let mut rhs_flat = 0usize;
            for (ax, coord) in rhs_coords.iter().enumerate() {
                rhs_flat += *coord * value_strides[ax];
            }
            value_elements[rhs_flat].clone()
        };
        let write_val = cast_for_segment(b, segment_id, &raw_write, dtype);
        let old = cells[abs].clone();
        cells[abs] = select_typed(b, dtype, &cond, &write_val, &old);
    }
    b.static_array_payload.insert(segment_id, cells);
    b.lazy_static_array_segments.insert(segment_id);
    true
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

            value_id: ValueId::next(),
        })
    }

    fn make_1d(b: &mut IRBuilder, vals: &[i64]) -> Value {
        let leaves: Vec<Value> = vals.iter().map(|n| b.ir_constant_int(*n)).collect();
        let lst = list_of(leaves);
        super::super::base::to_static_array(b, &lst).expect("StaticArray")
    }

    fn read_back_1d(b: &mut IRBuilder, val: &Value) -> Vec<Option<i64>> {
        let lst = super::super::base::to_value_list(b, val);
        match lst {
            Value::List(d) => d.values.iter().map(|v| v.int_val()).collect(),
            _ => panic!("expected List"),
        }
    }

    #[test]
    fn static_index_write_updates_cache() {
        let mut b = IRBuilder::new();
        let arr = make_1d(&mut b, &[1, 2, 3, 4]);
        // arr[2] = 99
        let new_val = b.ir_constant_int(99);
        let indices = vec![SliceIndex::Single(b.ir_constant_int(2))];
        let updated = static_array_setitem(&mut b, &arr, &indices, &new_val);
        let read = read_back_1d(&mut b, &updated);
        assert_eq!(read, vec![Some(1), Some(2), Some(99), Some(4)]);
    }

    #[test]
    fn dynamic_index_write_keeps_overlay_cache() {
        let mut b = IRBuilder::new();
        let arr = make_1d(&mut b, &[10, 20, 30, 40]);
        let segment_id = match &arr {
            Value::StaticArray { segment_id, .. } => *segment_id,
            _ => panic!(),
        };
        // simulate a runtime index — fresh wire with no static_val
        let runtime_idx =
            b.ir_read_integer(crate::circuit_input::InputPath::new("j", vec![]), true);
        let new_val = b.ir_constant_int(77);
        let indices = vec![SliceIndex::Single(runtime_idx)];
        static_array_setitem(&mut b, &arr, &indices, &new_val);
        // Small dynamic writes stay as a logical overlay so static consumers
        // can avoid memory reads; a later dynamic memory access will flush it.
        assert!(b.static_array_payload.get(&segment_id).is_some());
        assert!(b.lazy_static_array_segments.contains(&segment_id));
    }
}
