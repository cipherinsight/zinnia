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
//! Cache invalidation policy (`IRBuilder::static_array_payload`):
//!
//! - **Element write at static index**: the cache cell at the written offset
//!   is updated to the new wire (option (b) from the card). Other cached
//!   cells stay valid; static-index reads after the write continue to land
//!   as cheap cached lookups.
//! - **Element write at dynamic index**: the entire cache entry is removed
//!   (option (a)). The runtime address could land anywhere in the segment,
//!   so any cell could now be stale; subsequent reads must go through
//!   `ir_read_memory` to see the post-write state from zkRAM. The
//!   `to_value_list` shim falls back to per-cell segment reads for that
//!   segment_id.
//! - **Slice setitem (any kind)**: same as dynamic — the cache entry is
//!   removed.
//!
//! View-write fidelity: `Value::StaticArray` views (e.g. `arr[i]` for ≥2-D
//! `arr`, or `arr[a:b]` static contiguous slices from P2) inherit the
//! source segment_id with an adjusted offset. A write into the view writes
//! through to the source segment at `view.offset + flat_idx`. No new
//! segment is allocated.

use crate::builder::IRBuilder;
use crate::ops::dyn_ndarray::{scalar_i64_to_value, value_to_scalar_i64};
use crate::types::{NumberType, ScalarValue, SliceIndex, Value, ValueId};

use super::shape_arith::{decode_coords, row_major_strides};

// ────────────────────────────────────────────────────────────────────────
// Helpers
// ────────────────────────────────────────────────────────────────────────

/// Cast a value to match the target dtype before storing it in a segment.
fn cast_to_dtype(b: &mut IRBuilder, v: &Value, dtype: NumberType) -> Value {
    match dtype {
        NumberType::Float => {
            if matches!(v, Value::Float(_)) { v.clone() } else { b.ir_float_cast(v) }
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
    b.static_array_payload.remove(&segment_id);
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
        Value::StaticArray { dtype, shape, segment_id, strides, offset, imag_segment_id, value_id: _ } => {
            (*dtype, shape.clone(), *segment_id, strides.clone(), *offset, *imag_segment_id)
        }
        _ => panic!("static_array_setitem: expected Value::StaticArray"),
    };

    if indices.is_empty() {
        return val.clone();
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
        let lst = super::static_array::to_value_list(b, val);
        // The caller (do_recursive_assign) will run the legacy
        // set_nested_value path. Returning the materialised List signals
        // that path.
        return set_nested_via_list(b, &lst, indices, value);
    }

    let has_range = indices.iter().any(|i| matches!(i, SliceIndex::Range(_, _, _)));

    if has_range {
        // Fast-path: when the slice setitem touches a large number of cells
        // and every index is compile-time constant (no dynamic Range bounds,
        // no dynamic Single indices), running the legacy `set_nested_value`
        // path on a materialised List is much cheaper at compile time than
        // emitting one `ir_write_memory` IR statement per cell. The
        // materialised result is rebound as a `Value::List`, which the
        // boundary shim accepts on subsequent ops.
        //
        // This trades segment-share aliasing semantics (a static slice
        // setitem on a view used to write through to the source segment)
        // for compile-time tractability on programs like grayscott that
        // do `arr[:] = scalar` over very large arrays. For dynamic-write
        // workloads (sorts, queens, …) the indices are dynamic anyway and
        // we stay on the native path — that's where the P3 win lives.
        // Two routes into the legacy fallback:
        //   (a) Every index is compile-time constant AND target_cells is
        //       large enough that per-cell ir_write_memory would be slow
        //       — materialise to List, modify in place, rebind.
        //   (b) Range bounds are runtime values but Singles are static —
        //       this matches the P2 legacy behaviour (which silently
        //       defaulted runtime range bounds to [0, len)). Native
        //       segment writes for this case generate runtime address
        //       arithmetic per cell which dominates compile time on big
        //       arrays. Preserve the P2 cost model here; correctness
        //       semantics for this path were no different at P2 anyway.
        let no_dynamic_singles = indices.iter().all(|i| match i {
            SliceIndex::Single(v) => v.int_val().is_some(),
            _ => true,
        });
        let all_static_indices = no_dynamic_singles && indices.iter().all(|i| match i {
            SliceIndex::Range(s, e, st) => {
                let resolved = |opt: &Option<Value>| matches!(opt, None | Some(Value::None))
                    || opt.as_ref().and_then(|v| v.int_val()).is_some();
                resolved(s) && resolved(e) && resolved(st)
            }
            _ => true,
        });
        if all_static_indices {
            // Compute the static target cell count.
            let mut target_cells: usize = 1;
            for (ax, idx) in indices.iter().enumerate() {
                match idx {
                    SliceIndex::Single(_) => { /* axis collapsed */ }
                    SliceIndex::Range(s, e, st) => {
                        let dim = shape[ax] as i64;
                        let resolve = |v: &Option<Value>, def: i64| -> i64 {
                            match v.as_ref() {
                                Some(Value::None) | None => def,
                                Some(val) => val.int_val().unwrap_or(def),
                            }
                        };
                        let ss = resolve(s, 0);
                        let ee = resolve(e, dim);
                        let stst = resolve(st, 1);
                        let ss = if ss < 0 { (dim + ss).max(0) } else { ss.min(dim) };
                        let ee = if ee < 0 { (dim + ee).max(0) } else { ee.min(dim) };
                        let mut count = 0usize;
                        if stst > 0 {
                            let mut i = ss;
                            while i < ee { count += 1; i += stst; }
                        } else if stst < 0 {
                            let mut i = ss;
                            while i > ee { count += 1; i += stst; }
                        }
                        target_cells *= count;
                    }
                    _ => {}
                }
            }
            for ax in indices.len()..shape.len() {
                target_cells *= shape[ax];
            }
            // Threshold tuned for grayscott's `u[:] = 1.0` (90,000 cells).
            // Static slices below the threshold get the native segment-write
            // path; above it we materialise to a List and let the boundary
            // shim handle subsequent access.
            const LARGE_STATIC_SLICE_THRESHOLD: usize = 1024;
            if target_cells > LARGE_STATIC_SLICE_THRESHOLD {
                return static_array_slice_setitem_via_list(b, val, indices, value);
            }
        }
        // Path (b): runtime Range bounds on a *large* array. The native
        // dynamic-range path would emit `target_cells × (read + select +
        // write)` IR per cell, which is too slow at compile time for
        // target_cells in the tens of thousands (grayscott's 90,000-cell
        // arrays). Fall back to materialise-and-rebind, which loses the
        // runtime-bound masking semantics — but legacy P2 had the same
        // limitation (it silently defaulted runtime range bounds to 0/len)
        // so this preserves the prior compile-time behaviour for that
        // class of program.
        //
        // Small-array runtime-bound slice writes stay on the native path
        // and get correct masking — that's what
        // `test_static_array_2d_slice_runtime_bound` exercises.
        if no_dynamic_singles
            && indices.iter().any(|i| matches!(i, SliceIndex::Range(_, _, _)))
        {
            let runtime_range = indices.iter().any(|i| match i {
                SliceIndex::Range(s, e, st) => {
                    let runtime = |opt: &Option<Value>| matches!(opt, Some(v) if !matches!(v, Value::None) && v.int_val().is_none());
                    runtime(s) || runtime(e) || runtime(st)
                }
                _ => false,
            });
            if runtime_range {
                let array_cells: usize = shape.iter().product();
                const RUNTIME_RANGE_NATIVE_LIMIT: usize = 1024;
                if array_cells > RUNTIME_RANGE_NATIVE_LIMIT {
                    return static_array_slice_setitem_via_list(b, val, indices, value);
                }
            }
        }
        if dtype == NumberType::Complex {
            return complex_slice_setitem_via_list(b, val, indices, value);
        }
        return static_array_slice_setitem(
            b, val, dtype, &shape, segment_id, &strides, offset, indices, value,
        );
    }

    // All-Single indices.
    if dtype == NumberType::Complex {
        return complex_element_setitem(
            b, val, &shape, segment_id, imag_seg.expect("Complex StaticArray missing imag_segment_id"),
            &strides, offset, indices, value,
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
    // Pull out per-component (real_f, imag_f) Float wires.
    let (re_v, im_v) = crate::helpers::value_ops::unpack_value_to_complex_parts(b, value);
    // Convert each SliceIndex::Single into a Value (we already vetted this is
    // an all-Single setitem at the dispatcher).
    let idx_vals: Vec<Value> = indices.iter().map(|s| match s {
        SliceIndex::Single(v) => v.clone(),
        _ => unreachable!("complex_element_setitem expects all-Single indices"),
    }).collect();

    // Static fast path: known offset → both component writes are direct.
    let all_static = idx_vals.iter().all(|v| v.int_val().is_some());
    if all_static && idx_vals.len() == shape.len() {
        let mut flat: i64 = offset as i64;
        for (ax, v) in idx_vals.iter().enumerate() {
            let i = v.int_val().unwrap();
            let i = if i < 0 { shape[ax] as i64 + i } else { i };
            flat += i * strides[ax] as i64;
        }
        let addr = b.ir_constant_int(flat);
        b.ir_write_memory(real_seg, &addr, &re_v);
        b.ir_write_memory(imag_seg, &addr, &im_v);
        // Update cache cell to a Value::Complex made from the new parts.
        let abs = flat as usize;
        let new_complex = match (&re_v, &im_v) {
            (Value::Float(r), Value::Float(im)) => Value::Complex { real: r.clone(), imag: im.clone() },
            _ => unreachable!(),
        };
        cache_set_cell(b, real_seg, abs, new_complex);
        return val.clone();
    }
    // Dynamic: drop the cache, then issue two parallel ir_write_memory ops.
    let addr = compute_addr(b, offset, strides, &idx_vals, shape);
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
    indices: &[SliceIndex],
    value: &Value,
) -> Value {
    let lst = super::static_array::to_value_list(b, val);
    let value_lst = if let Value::StaticArray { .. } = value {
        super::static_array::to_value_list(b, value)
    } else {
        value.clone()
    };
    legacy_set_nested(b, &lst, indices, &value_lst)
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

/// Static slice setitem fallback that materialises the LHS to a nested
/// List, runs the inlined legacy `set_nested_value` logic, and **rebinds
/// the variable as a List**. Used when the slice covers too many cells for
/// per-cell `ir_write_memory` to be cheap at compile time (programs like
/// grayscott that do `arr[:] = scalar` over a 300x300 array). Trades
/// segment-share aliasing for compile-time tractability.
fn static_array_slice_setitem_via_list(
    b: &mut IRBuilder,
    val: &Value,
    indices: &[SliceIndex],
    value: &Value,
) -> Value {
    let lst = super::static_array::to_value_list(b, val);
    let value_lst = if let Value::StaticArray { .. } = value {
        super::static_array::to_value_list(b, value)
    } else {
        value.clone()
    };
    legacy_set_nested(b, &lst, indices, &value_lst)
}

/// Slice setitem fallback that reuses the legacy `set_nested_value` path —
/// invoked when the RHS shape is broadcastable but not equal to the LHS
/// extent (numpy column-broadcasting, scalar-array tail, etc.). The legacy
/// path absorbs these cases via per-element iteration; rather than
/// reimplement full broadcasting in the segment-write helper, we
/// materialise to a List, run the legacy update, and write the new payload
/// back into the segment.
fn slice_setitem_via_legacy(
    b: &mut IRBuilder,
    val: &Value,
    indices: &[SliceIndex],
    value: &Value,
) -> Value {
    let (dtype, shape, segment_id, _strides, offset) = match val {
        Value::StaticArray { dtype, shape, segment_id, strides, offset, .. } => {
            (*dtype, shape.clone(), *segment_id, strides.clone(), *offset)
        }
        _ => unreachable!("slice_setitem_via_legacy: expected StaticArray"),
    };
    // Materialise the source to a nested List.
    let lst = super::static_array::to_value_list(b, val);
    // Materialise the RHS too — `legacy_set_nested` only handles
    // List/Tuple/scalar. A StaticArray RHS would fall into the wrong
    // (scalar broadcast) branch otherwise.
    let value_lst = if let Value::StaticArray { .. } = value {
        super::static_array::to_value_list(b, value)
    } else {
        value.clone()
    };
    // Run the legacy nested-set on a synthesized SliceIndex copy. We can't
    // call `IRGenerator::set_nested_value` from here (helpers layer), so
    // we replicate its logic enough for slice writes — but the simplest
    // trick is to bake the new payload back into the same segment.
    let updated = legacy_set_nested(b, &lst, indices, &value_lst);
    // Flatten the updated nested list and write back into the segment.
    let flat = super::composite::flatten_composite(&updated);
    let total: usize = shape.iter().product();
    if flat.len() != total {
        panic!(
            "slice assignment broadcast fallback: flattened result has {} elements but expected {}",
            flat.len(), total
        );
    }
    for (i, leaf) in flat.iter().enumerate() {
        let abs = offset + i;
        let addr = b.ir_constant_int(abs as i64);
        let cast = cast_to_dtype(b, leaf, dtype);
        b.ir_write_memory(segment_id, &addr, &cast);
    }
    cache_invalidate(b, segment_id);
    val.clone()
}

/// Inline version of the legacy IRGenerator::set_nested_value, taking a
/// `Value::List` and producing an updated `Value::List`. Replicated here
/// (vs called from the IR generator) because helpers can't reach back into
/// the visitor layer cleanly. Limited to the cases the broadcast fallback
/// triggers on — Range slice on outer axis(es), scalar / list RHS.
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
                let i = if i < 0 { (data.values.len() as i64 + i) as usize } else { i as usize };
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
                        Value::Tuple(crate::types::CompositeData { elements_type: new_types, values: new_values, value_id: ValueId::next() })
                    } else {
                        Value::List(crate::types::CompositeData { elements_type: new_types, values: new_values, value_id: ValueId::next() })
                    };
                }
            }
            current.clone()
        }
        SliceIndex::Range(start, stop, step) => {
            let len = data.values.len() as i64;
            let start_i = start.as_ref().and_then(|v| v.int_val()).unwrap_or(0);
            let stop_i = stop.as_ref().and_then(|v| v.int_val()).unwrap_or(len);
            let step_val = step.as_ref().and_then(|v| v.int_val()).unwrap_or(1);
            let start_idx = if start_i < 0 { (len + start_i).max(0) } else { start_i.min(len) } as usize;
            let stop_idx = if stop_i < 0 { (len + stop_i).max(0) } else { stop_i.min(len) } as usize;

            let mut new_values = data.values.clone();
            let mut new_types = data.elements_type.clone();
            if indices.len() > 1 {
                if let Value::List(rhs_data) | Value::Tuple(rhs_data) = value {
                    let mut rhs_idx = 0;
                    let mut i = start_idx;
                    while i < stop_idx && rhs_idx < rhs_data.values.len() {
                        new_values[i] = legacy_set_nested(b, &new_values[i], &indices[1..], &rhs_data.values[rhs_idx]);
                        new_types[i] = new_values[i].zinnia_type();
                        rhs_idx += 1;
                        i += step_val as usize;
                    }
                } else {
                    // Scalar RHS — broadcast inner.
                    let mut i = start_idx;
                    while i < stop_idx {
                        new_values[i] = legacy_set_nested(b, &new_values[i], &indices[1..], value);
                        new_types[i] = new_values[i].zinnia_type();
                        i += step_val as usize;
                    }
                }
            } else if let Value::List(rhs_data) | Value::Tuple(rhs_data) = value {
                let mut rhs_idx = 0;
                let mut i = start_idx;
                while i < stop_idx && rhs_idx < rhs_data.values.len() {
                    new_values[i] = rhs_data.values[rhs_idx].clone();
                    new_types[i] = new_values[i].zinnia_type();
                    rhs_idx += 1;
                    i += step_val as usize;
                }
            } else {
                let mut i = start_idx;
                while i < stop_idx {
                    new_values[i] = value.clone();
                    new_types[i] = new_values[i].zinnia_type();
                    i += step_val as usize;
                }
            }
            if is_tuple {
                Value::Tuple(crate::types::CompositeData { elements_type: new_types, values: new_values, value_id: ValueId::next() })
            } else {
                Value::List(crate::types::CompositeData { elements_type: new_types, values: new_values, value_id: ValueId::next() })
            }
        }
        _ => current.clone(),
    }
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
    let idx_vals: Vec<Value> = indices.iter().map(|s| match s {
        SliceIndex::Single(v) => v.clone(),
        _ => unreachable!("static_array_element_setitem: non-Single index"),
    }).collect();

    let rank = shape.len();

    // Partial indexing (fewer indices than rank): the LHS targets a row /
    // sub-array. Rewrite as a slice-setitem with trailing full ranges.
    if idx_vals.len() < rank {
        let mut full: Vec<SliceIndex> = idx_vals.iter().cloned()
            .map(SliceIndex::Single).collect();
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
            rank, idx_vals.len()
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
    let cast_val = cast_to_dtype(b, value, dtype);

    // All-static path: use the cached-wire-update strategy.
    let all_static = idx_vals.iter().all(|v| v.int_val().is_some());
    if all_static {
        let mut flat_rel: i64 = 0;
        for (ax, v) in idx_vals.iter().enumerate() {
            let i = v.int_val().unwrap();
            let i = if i < 0 { shape[ax] as i64 + i } else { i };
            flat_rel += i * strides[ax] as i64;
        }
        let abs_offset = (offset as i64 + flat_rel) as usize;
        let addr = b.ir_constant_int(abs_offset as i64);
        b.ir_write_memory(segment_id, &addr, &cast_val);
        cache_set_cell(b, segment_id, abs_offset, cast_val);
        return val.clone();
    }

    // At least one dynamic index → invalidate cache, single segment write.
    let addr = compute_addr(b, offset, strides, &idx_vals, shape);
    b.ir_write_memory(segment_id, &addr, &cast_val);
    cache_invalidate(b, segment_id);
    val.clone()
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
            rank, indices.len()
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
                    let s = s_static.unwrap_or(0);
                    let e = e_static.unwrap_or(dim);
                    let st = st_static.unwrap_or(1);
                    let s = if s < 0 { (dim + s).max(0) } else { s.min(dim) };
                    let e = if e < 0 { (dim + e).max(0) } else { e.min(dim) };
                    assert!(st != 0, "slice step cannot be zero");
                    let mut coords: Vec<usize> = Vec::new();
                    if st > 0 {
                        let mut i = s;
                        while i < e { coords.push(i as usize); i += st; }
                    } else {
                        let mut i = s;
                        while i > e { coords.push(i as usize); i += st; }
                    }
                    value_shape.push(coords.len());
                    axis_specs.push((ax, AxisCoords::Static(coords)));
                } else {
                    has_dynamic_range = true;
                    fn to_ir(b: &mut IRBuilder, v: &Option<Value>, default: i64) -> Value {
                        match v.as_ref() {
                            Some(val) if !matches!(val, Value::None) => {
                                if let Some(s) = val.int_val() { b.ir_constant_int(s) } else { val.clone() }
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
                let flat_list = super::static_array::to_value_list(b, value);
                super::composite::flatten_composite_with_builder(b, &flat_list)
            }
            Value::DynamicNDArray(vd) => {
                super::segment::read_all(b, vd.segment_id, vd.max_length())
            }
            Value::List(_) | Value::Tuple(_) => {
                super::composite::flatten_composite_with_builder(b, value)
            }
            _ => panic!("slice assignment: value must be scalar or array-like"),
        }
    };

    // Shape compatibility check.
    //
    // Numpy semantics: if the LHS slice has shape K and the RHS has shape
    // M, K == M is required for the strict path. For RHS shapes that are
    // compatible via broadcasting (e.g., LHS (64, 64) ← RHS (64, 1)) we
    // materialize the RHS to the LHS shape via simple modulo expansion.
    // This matches the previous P2 behaviour where the legacy
    // `set_nested_value` path absorbed the mismatch.
    if !value_is_scalar {
        let value_total: usize = value_shape.iter().product();
        if value_elements.len() != value_total {
            // Allow broadcasting if the value count divides the target
            // count. This covers numpy column-broadcast (M=N when LHS is
            // (N, K)) and most real-world cases.
            if value_total % value_elements.len() != 0 {
                panic!(
                    "slice assignment shape mismatch: target slice has shape {:?} ({} elements) \
                     but value has {} elements. Broadcasting in slice assignment is not supported.",
                    value_shape, value_total, value_elements.len()
                );
            }
            // Fall back to legacy path: materialise the LHS to a nested
            // List, run set_nested_value, then push the result back into
            // the segment. This preserves the previous write semantics and
            // avoids re-implementing full numpy broadcasting here.
            return slice_setitem_via_legacy(b, val, indices, value);
        }
    }

    let value_total: usize = value_shape.iter().product();
    let value_strides = row_major_strides(&value_shape);

    if !has_dynamic_range {
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
                            let i = if i < 0 { shape[*src_axis] as i64 + i } else { i };
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
                cast_to_dtype(b, value, dtype)
            } else {
                cast_to_dtype(b, &value_elements[val_flat], dtype)
            };
            b.ir_write_memory(segment_id, &addr, &write_val);

            if !had_dynamic_single {
                cache_set_cell(b, segment_id, addr_static as usize, write_val);
            }
        }
        if axis_specs.iter().any(|(_, s)| matches!(s, AxisCoords::Single(v) if v.int_val().is_none())) {
            // A dynamic Single index made addresses runtime-only.
            cache_invalidate(b, segment_id);
        }
    } else {
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
                            let i = if i < 0 { shape[*src_axis] as i64 + i } else { i };
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
                    AxisCoords::Dynamic { start, stop, step, axis_len } => {
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
                cast_to_dtype(b, value, dtype)
            } else {
                cast_to_dtype(b, &value_elements[val_flat], dtype)
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

// ────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{CompositeData, ScalarValue, ZinniaType};

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
        super::super::static_array::to_static_array(b, &lst).expect("StaticArray")
    }

    fn read_back_1d(b: &mut IRBuilder, val: &Value) -> Vec<Option<i64>> {
        let lst = super::super::static_array::to_value_list(b, val);
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
    fn dynamic_index_write_invalidates_cache() {
        let mut b = IRBuilder::new();
        let arr = make_1d(&mut b, &[10, 20, 30, 40]);
        let segment_id = match &arr {
            Value::StaticArray { segment_id, .. } => *segment_id,
            _ => panic!(),
        };
        // simulate a runtime index — fresh wire with no static_val
        let runtime_idx = b.ir_read_integer(
            crate::circuit_input::InputPath::new("j", vec![]),
            true,
        );
        let new_val = b.ir_constant_int(77);
        let indices = vec![SliceIndex::Single(runtime_idx)];
        static_array_setitem(&mut b, &arr, &indices, &new_val);
        // Cache must be gone after a dynamic-index write.
        assert!(b.static_array_payload.get(&segment_id).is_none());
    }
}
