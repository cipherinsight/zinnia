//! Dynamic array element and masked assignment.

use crate::builder::IRBuilder;
use crate::types::{DynamicNDArrayData, NumberType, SliceIndex, Value, ValueId};

use super::bounded_axis::{select_stride_mode, stride_value, StrideMode};
use super::indexing::{compute_flat_addr, wrap_and_check_dynamic_indices};

/// Single element assignment: dyn[i] = x or dyn[i, j] = x.
/// In-place write via `ir_write_memory` — O(1).
pub fn dyn_setitem(
    b: &mut IRBuilder,
    data: &DynamicNDArrayData,
    indices: &[SliceIndex],
    value: &Value,
) -> Value {
    // A tuple index addresses one axis per entry (`a[t] = v` is
    // `a[t[0], t[1], ...] = v`); only a list is a fancy index.
    if let [SliceIndex::Single(Value::Tuple(tuple))] = indices {
        let expanded: Vec<SliceIndex> =
            tuple.values.iter().cloned().map(SliceIndex::Single).collect();
        return dyn_setitem(b, data, &expanded, value);
    }
    // Fancy-index scatter: any axis indexed by an integer array (StaticArray /
    // List / Tuple) means `for k in 0..L: dst[idx_axes[k]] = rhs[k] | rhs`.
    // Without this branch, `compute_flat_addr` would try to multiply an array
    // by a stride and emit zero-arg / malformed IR statements.
    if indices.iter().any(|i| {
        matches!(
            i,
            SliceIndex::Single(Value::StaticArray { .. } | Value::List(_) | Value::Tuple(_))
        )
    }) {
        return dyn_fancy_setitem(b, data, indices, value);
    }

    let indices = &wrap_and_check_dynamic_indices(b, data, indices, "dyn_setitem");
    let addr = compute_flat_addr(b, data, indices);
    if data.envelope.total_bound > 0 {
        crate::optim::resolver::discharge_index_in_range(
            b,
            &addr,
            0,
            data.envelope.total_bound as i64,
            "dyn_setitem",
        );
    }

    let write_val = cast_to_dtype(b, value, data.dtype);
    crate::helpers::segment::write_typed(b, data.segment_id, &addr, &write_val, data.dtype);

    Value::DynamicNDArray(data.clone())
}

fn read_index_array_elements(b: &mut IRBuilder, src: &Value) -> Vec<Value> {
    match src {
        Value::StaticArray { .. } => {
            let list = crate::helpers::static_array::base::to_value_list(b, src);
            match list {
                Value::List(d) => d.values,
                _ => vec![],
            }
        }
        Value::List(d) | Value::Tuple(d) => d.values.clone(),
        _ => vec![],
    }
}

fn dyn_fancy_setitem(
    b: &mut IRBuilder,
    data: &DynamicNDArrayData,
    indices: &[SliceIndex],
    value: &Value,
) -> Value {
    let mut fancy_arrays: Vec<Option<Vec<Value>>> = Vec::with_capacity(indices.len());
    let mut fancy_len: Option<usize> = None;
    for idx in indices {
        let elems = match idx {
            SliceIndex::Single(v @ Value::StaticArray { .. })
            | SliceIndex::Single(v @ Value::List(_))
            | SliceIndex::Single(v @ Value::Tuple(_)) => Some(read_index_array_elements(b, v)),
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
    let l = fancy_len.expect("dyn_fancy_setitem called without a fancy axis");

    let value_per_k: Option<Vec<Value>> = match value {
        Value::StaticArray { shape: vs, .. } if !vs.is_empty() && vs[0] == l && vs.len() == 1 => {
            Some(read_index_array_elements(b, value))
        }
        Value::List(d) | Value::Tuple(d) if d.values.len() == l => Some(d.values.clone()),
        _ => None,
    };

    let mut current = data.clone();
    for k in 0..l {
        let mut k_indices: Vec<SliceIndex> = Vec::with_capacity(indices.len());
        for (ax, idx) in indices.iter().enumerate() {
            match &fancy_arrays[ax] {
                Some(arr) => k_indices.push(SliceIndex::Single(arr[k].clone())),
                None => k_indices.push(idx.clone()),
            }
        }
        let v_k = match &value_per_k {
            Some(vs) => vs[k].clone(),
            None => value.clone(),
        };
        let updated = dyn_setitem(b, &current, &k_indices, &v_k);
        if let Value::DynamicNDArray(d) = updated {
            current = d;
        }
    }
    Value::DynamicNDArray(current)
}

/// Masked assignment: dyn[mask] = x.
/// Reads all elements, selects new or old per mask, writes to fresh segment.
pub fn dyn_setitem_mask(
    b: &mut IRBuilder,
    data: &DynamicNDArrayData,
    mask: &Value,
    value: &Value,
) -> Value {
    let data_owned = crate::ops::dyn_ndarray::force_contiguous(b, data);
    let data = &data_owned;
    let max_len = data.max_length();
    let target_shape = &data.meta.logical_shape;

    let mask_elements: Vec<Value> = match mask {
        Value::DynamicNDArray(md) => {
            if md.meta.logical_shape != *target_shape {
                panic!(
                    "masked assignment shape mismatch: array has shape {:?} but mask has shape {:?}",
                    target_shape,
                    md.meta.logical_shape
                );
            }
            if md.max_length() != max_len {
                panic!(
                    "masked assignment shape mismatch: array has {} elements but mask has {}",
                    max_len,
                    md.max_length()
                );
            }
            let md_owned = crate::ops::dyn_ndarray::force_contiguous(b, md);
            crate::helpers::segment::read_all_typed(
                b,
                md_owned.segment_id,
                md_owned.max_length(),
                md_owned.dtype,
            )
        }
        Value::List(_) | Value::Tuple(_) | Value::StaticArray { .. } => {
            let shape = crate::helpers::composite::get_composite_shape(mask);
            if shape != *target_shape {
                panic!(
                    "masked assignment shape mismatch: array has shape {:?} but mask has shape {:?}",
                    target_shape, shape
                );
            }
            let mask_total: usize = shape.iter().product();
            if mask_total != max_len {
                panic!(
                    "masked assignment shape mismatch: array has {} elements but mask has {}",
                    max_len, mask_total
                );
            }
            crate::helpers::composite::flatten_composite_with_builder(b, mask)
        }
        _ => panic!("masked assignment: mask must be array-like"),
    };
    if mask_elements.len() != max_len {
        panic!(
            "masked assignment shape mismatch: array has {} elements but mask has {}",
            max_len,
            mask_elements.len()
        );
    }

    // Resolve the RHS. numpy `a[mask] = rhs` consumes `rhs` SEQUENTIALLY,
    // one element per true mask cell — so a non-scalar RHS is fundamentally
    // PACKED to the true count, NOT positionally aligned to the target.
    //
    // Three shapes (mirrors the StaticArray classification in
    // `helpers/static_array/mask.rs`):
    //   - `Scalar`     : broadcast across all true positions.
    //   - `FullShape`  : one cell per target element, written positionally.
    //     Only minted when an identity MaskReadView proves it (the
    //     `a[mask] op= v` round-trip cancellation: the packed-kept-sources
    //     invariant gives `rhs[#trues before i] == sources[i]` at every true
    //     position, so the positional write is correct and the pack/unpack
    //     shift network cancels) OR for a genuinely target-sized composite.
    //   - `Compacted`  : PACKED values (true-count-sized or a filter output
    //     whose view was dropped); aligned at write time by a running prefix
    //     count of trues.
    enum Rhs {
        Scalar(Value),
        FullShape(Vec<Value>),
        Compacted(Vec<Value>),
    }

    let rhs = if value.is_number() {
        Rhs::Scalar(value.clone())
    } else {
        match value {
            Value::DynamicNDArray(vd) => {
                // Round-trip cancellation: when the RHS carries a mask-read
                // view whose mask is THIS mask (same cell SSA values) and the
                // value is the packed array itself (identity meta + the
                // original runtime-length stmt), the packed invariant makes
                // the positional write correct with no unpack network.
                let meta_identity = vd.meta.logical_offset == 0
                    && vd.meta.runtime_offset.static_val == Some(0)
                    && vd.meta.logical_shape.as_slice() == [max_len]
                    && vd.meta.logical_strides.as_slice() == [1]
                    && vd.meta.runtime_length.stmt_id.is_some();
                let view_sources = b
                    .provenance
                    .mask_read_view(vd.segment_id)
                    .filter(|view| {
                        meta_identity
                            && view.count_stmt == vd.meta.runtime_length.stmt_id
                            && view.sources.len() == max_len
                            && view.mask_cells.len() == mask_elements.len()
                            && view.mask_cells.iter().zip(mask_elements.iter()).all(
                                |(a, c)| match (a.value_id(), c.value_id()) {
                                    (Some(x), Some(y)) => x == y,
                                    _ => false,
                                },
                            )
                    })
                    .map(|view| view.sources.clone());
                if let Some(sources) = view_sources {
                    Rhs::FullShape(sources)
                } else {
                    let vd_owned = crate::ops::dyn_ndarray::force_contiguous(b, vd);
                    let cells = crate::helpers::segment::read_all_typed(
                        b,
                        vd_owned.segment_id,
                        vd_owned.max_length(),
                        vd_owned.dtype,
                    );
                    // Discriminate a genuinely full-shape RHS from a packed
                    // (filter-output / true-count) one by its runtime length:
                    // a fully-populated array of the target shape has a
                    // STATIC runtime_length == max_len, so its cell i lands at
                    // target position i (the `arr[mask] = arr2` / np.where
                    // gated-assignment shape, matching the StaticArray
                    // `FullShape` case). A compaction output's runtime length
                    // is data-dependent (`static_val == None`) — its kept
                    // values are PACKED to the front and must be aligned by a
                    // prefix count of trues.
                    let is_full_shape = vd.meta.logical_shape == *target_shape
                        && vd.max_length() == max_len
                        && vd.meta.runtime_length.static_val == Some(max_len as i64);
                    if is_full_shape {
                        Rhs::FullShape(cells)
                    } else {
                        Rhs::Compacted(cells)
                    }
                }
            }
            Value::List(_) | Value::Tuple(_) | Value::StaticArray { .. } => {
                let shape = crate::helpers::composite::get_composite_shape(value);
                let cells = crate::helpers::composite::flatten_composite_with_builder(b, value);
                if shape == *target_shape && cells.len() == max_len {
                    // Genuinely target-sized composite (np.where-style gated
                    // assignment): positional.
                    Rhs::FullShape(cells)
                } else {
                    // True-count-sized (or otherwise compacted) RHS.
                    Rhs::Compacted(cells)
                }
            }
            _ => panic!("masked assignment: value must be scalar or array-like"),
        }
    };

    let current_vals =
        crate::helpers::segment::read_all_typed(b, data.segment_id, max_len, data.dtype);
    let mut out_elements = Vec::with_capacity(max_len);

    // Running count of true mask cells before position i — the packed-RHS
    // read pointer. Constant-folds away entirely for static masks.
    let mut prefix = b.ir_constant_int(0);
    for i in 0..max_len {
        let keep = b.ir_bool_cast(&mask_elements[i]);

        let new_val = match &rhs {
            Rhs::Scalar(v) => cast_to_dtype(b, v, data.dtype),
            Rhs::FullShape(cells) => cast_to_dtype(b, &cells[i], data.dtype),
            Rhs::Compacted(cells) => {
                // rhs[#trues before i], one-hot over the prefix counter (at
                // most i trues can precede position i, and no more than the
                // packed RHS length).
                let mut acc = current_vals[i].clone();
                for (k, c) in cells.iter().enumerate().take(i + 1) {
                    let k_c = b.ir_constant_int(k as i64);
                    let eq = b.ir_equal_i(&prefix, &k_c);
                    let c_cast = cast_to_dtype(b, c, data.dtype);
                    acc = crate::helpers::value_ops::select_value(b, &eq, &c_cast, &acc);
                }
                acc
            }
        };

        let selected =
            crate::helpers::value_ops::select_value(b, &keep, &new_val, &current_vals[i]);
        out_elements.push(selected);

        if matches!(rhs, Rhs::Compacted(_)) {
            let m_i = b.ir_int_cast(&keep);
            prefix = b.ir_add_i(&prefix, &m_i);
        }
    }

    let segment_id = crate::helpers::segment::alloc_and_write_values(b, &out_elements, data.dtype);

    Value::DynamicNDArray(DynamicNDArrayData {
        envelope: data.envelope.clone(),
        dtype: data.dtype,
        segment_id,
        meta: data.meta.clone(),
        value_id: ValueId::next(),
    })
}

/// Slice assignment: dyn[i:j] = x, dyn[i:j, k] = x, etc.
/// In-place writes at computed addresses. Supports mixed Single/Range
/// indices. Forbids broadcasting mismatch with clear error.
pub fn dyn_setitem_slice(
    b: &mut IRBuilder,
    data: &DynamicNDArrayData,
    indices: &[SliceIndex],
    value: &Value,
) -> Value {
    let shape = &data.meta.logical_shape;
    let mode = select_stride_mode(data);
    let rank = shape.len();

    // Build the set of target positions: for each index, compute the
    // coordinates that the assignment covers.
    //
    // Single(v): one coordinate on this axis (axis collapsed in the value).
    // Range(s,e,st): multiple coordinates on this axis.
    struct AxisSpec {
        coords: AxisCoords,
        src_axis: usize,
    }
    enum AxisCoords {
        Single(Value),      // one coordinate (static or dynamic)
        Static(Vec<usize>), // known list of coordinates
        Dynamic {
            // pad-and-mask
            start: Value,
            stop: Value,
            step: Value,
            axis_len: usize,
        },
    }

    let mut axis_specs: Vec<AxisSpec> = Vec::new();
    // Axes that appear in the VALUE shape (Range axes and trailing implicit axes).
    let mut value_shape: Vec<usize> = Vec::new();
    let mut has_dynamic_range = false;

    for (ax, idx) in indices.iter().enumerate() {
        match idx {
            SliceIndex::Single(v) => {
                axis_specs.push(AxisSpec {
                    coords: AxisCoords::Single(v.clone()),
                    src_axis: ax,
                });
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

                let all_static = start.is_none()
                    || matches!(start.as_ref(), Some(Value::None))
                    || s_static.is_some();
                let all_static = all_static
                    && (stop.is_none()
                        || matches!(stop.as_ref(), Some(Value::None))
                        || e_static.is_some());
                let all_static = all_static
                    && (step.is_none()
                        || matches!(step.as_ref(), Some(Value::None))
                        || st_static.is_some());

                if all_static {
                    let coords = crate::helpers::shape_arith::resolve_slice(
                        shape[ax],
                        s_static,
                        e_static,
                        st_static,
                    );
                    value_shape.push(coords.len());
                    axis_specs.push(AxisSpec {
                        coords: AxisCoords::Static(coords),
                        src_axis: ax,
                    });
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
                    value_shape.push(shape[ax]);
                    axis_specs.push(AxisSpec {
                        coords: AxisCoords::Dynamic {
                            start: start_ir,
                            stop: stop_ir,
                            step: step_ir,
                            axis_len: shape[ax],
                        },
                        src_axis: ax,
                    });
                }
            }
            _ => panic!("slice assignment: Ellipsis/NewAxis not yet supported"),
        }
    }

    // Trailing axes not in indices are full ranges.
    for ax in indices.len()..rank {
        let coords: Vec<usize> = (0..shape[ax]).collect();
        value_shape.push(coords.len());
        axis_specs.push(AxisSpec {
            coords: AxisCoords::Static(coords),
            src_axis: ax,
        });
    }

    // Get value elements.
    let value_is_scalar = value.is_number();
    let value_elements: Vec<Value> = if value_is_scalar {
        vec![]
    } else {
        match value {
            Value::DynamicNDArray(vd) => {
                let vd_owned = crate::ops::dyn_ndarray::force_contiguous(b, vd);
                crate::helpers::segment::read_all_typed(
                    b,
                    vd_owned.segment_id,
                    vd_owned.max_length(),
                    vd_owned.dtype,
                )
            }
            Value::List(_) | Value::Tuple(_) => crate::helpers::composite::flatten_composite(value),
            // P4a: array literals / np.asarray RHS arrive as
            // segment-backed StaticArrays.
            Value::StaticArray { .. } => {
                crate::helpers::static_array::elementwise::payload_cells(b, value)
            }
            _ => panic!("slice assignment: value must be scalar or array-like"),
        }
    };

    // Check shape compatibility (no broadcasting).
    if !value_is_scalar {
        let value_total: usize = value_shape.iter().product();
        if value_elements.len() != value_total {
            panic!(
                "slice assignment shape mismatch: target slice has shape {:?} ({} elements) \
                 but value has {} elements. Broadcasting in slice assignment is not supported.",
                value_shape,
                value_total,
                value_elements.len()
            );
        }
    }

    // Collect the Range/trailing axes for iteration.
    let _range_axes: Vec<(usize, &AxisSpec)> = axis_specs
        .iter()
        .enumerate()
        .filter(|(_, spec)| !matches!(spec.coords, AxisCoords::Single(_)))
        .collect();

    let value_total: usize = value_shape.iter().product();
    let value_strides = crate::helpers::shape_arith::row_major_strides(&value_shape);

    if !has_dynamic_range {
        // All static: iterate over all target positions and write. In
        // literal-stride mode the entire address folds at compile time;
        // in symbolic-runtime mode we emit one `ir_mul_i` per axis against
        // the compact buffer's `runtime_strides`.
        for val_flat in 0..value_total {
            let val_coords =
                crate::helpers::shape_arith::decode_coords(val_flat, &value_shape, &value_strides);

            let addr = match &mode {
                StrideMode::LiteralLogical(strides) => {
                    let mut addr_static: i64 = 0;
                    let mut dynamic_parts: Vec<Value> = Vec::new();
                    let mut val_coord_idx = 0;
                    for spec in &axis_specs {
                        match &spec.coords {
                            AxisCoords::Single(v) => {
                                if let Some(i) = v.int_val() {
                                    let i = if i < 0 {
                                        shape[spec.src_axis] as i64 + i
                                    } else {
                                        i
                                    };
                                    addr_static += i * strides[spec.src_axis] as i64;
                                } else {
                                    // Dynamic single index (e.g.
                                    // `out[row, :] = vals` with a runtime
                                    // row): contribute idx*stride; the
                                    // bound is discharged once below.
                                    let stride_val =
                                        b.ir_constant_int(strides[spec.src_axis] as i64);
                                    dynamic_parts.push(b.ir_mul_i(v, &stride_val));
                                }
                            }
                            AxisCoords::Static(coords) => {
                                let coord = coords[val_coords[val_coord_idx]];
                                addr_static += coord as i64 * strides[spec.src_axis] as i64;
                                val_coord_idx += 1;
                            }
                            AxisCoords::Dynamic { .. } => unreachable!("checked above"),
                        }
                    }
                    let mut acc = b.ir_constant_int(addr_static);
                    for part in &dynamic_parts {
                        acc = b.ir_add_i(&acc, part);
                    }
                    acc
                }
                StrideMode::SymbolicRuntime(_) => {
                    let mut acc: Option<Value> = None;
                    let mut val_coord_idx = 0;
                    for spec in &axis_specs {
                        let coord_val = match &spec.coords {
                            AxisCoords::Single(v) => {
                                if let Some(i) = v.int_val() {
                                    let i = if i < 0 {
                                        shape[spec.src_axis] as i64 + i
                                    } else {
                                        i
                                    };
                                    b.ir_constant_int(i)
                                } else {
                                    v.clone()
                                }
                            }
                            AxisCoords::Static(coords) => {
                                let coord = coords[val_coords[val_coord_idx]];
                                val_coord_idx += 1;
                                b.ir_constant_int(coord as i64)
                            }
                            AxisCoords::Dynamic { .. } => unreachable!("checked above"),
                        };
                        let stride_v = stride_value(b, &mode, spec.src_axis);
                        let contrib = b.ir_mul_i(&coord_val, &stride_v);
                        acc = Some(match acc.take() {
                            None => contrib,
                            Some(prev) => b.ir_add_i(&prev, &contrib),
                        });
                    }
                    acc.unwrap_or_else(|| b.ir_constant_int(0))
                }
            };
            let write_val = if value_is_scalar {
                cast_to_dtype(b, value, data.dtype)
            } else {
                cast_to_dtype(b, &value_elements[val_flat], data.dtype)
            };
            crate::helpers::segment::write_typed(b, data.segment_id, &addr, &write_val, data.dtype);
        }
    } else {
        // Has dynamic range: iterate over max positions, conditionally write.
        // Read current values first (for select on out-of-bounds positions).
        for val_flat in 0..value_total {
            let val_coords =
                crate::helpers::shape_arith::decode_coords(val_flat, &value_shape, &value_strides);

            let mut addr_parts_static: i64 = 0;
            let mut addr_parts_dynamic: Vec<Value> = Vec::new();
            let mut in_bounds_parts: Vec<Value> = Vec::new();
            let mut val_coord_idx = 0;

            for spec in &axis_specs {
                let (stride_lit, stride_v): (i64, Value) = match &mode {
                    StrideMode::LiteralLogical(strides) => {
                        let s = strides[spec.src_axis] as i64;
                        (s, b.ir_constant_int(s))
                    }
                    StrideMode::SymbolicRuntime(_) => {
                        // Symbolic stride: force the entire axis into the
                        // dynamic-parts list via `ir_mul_i` so the
                        // address-folding loop below sees no compile-time
                        // contribution from this axis. We pass `0` for the
                        // literal sentinel to ensure no folding happens.
                        (0, stride_value(b, &mode, spec.src_axis))
                    }
                };
                let use_symbolic = matches!(mode, StrideMode::SymbolicRuntime(_));
                match &spec.coords {
                    AxisCoords::Single(v) => {
                        if let Some(i) = v.int_val() {
                            let i = if i < 0 {
                                shape[spec.src_axis] as i64 + i
                            } else {
                                i
                            };
                            if use_symbolic {
                                let i_val = b.ir_constant_int(i);
                                addr_parts_dynamic.push(b.ir_mul_i(&i_val, &stride_v));
                            } else {
                                addr_parts_static += i * stride_lit;
                            }
                        } else {
                            addr_parts_dynamic.push(b.ir_mul_i(v, &stride_v));
                        }
                    }
                    AxisCoords::Static(coords) => {
                        let coord = coords[val_coords[val_coord_idx]];
                        if use_symbolic {
                            let c_val = b.ir_constant_int(coord as i64);
                            addr_parts_dynamic.push(b.ir_mul_i(&c_val, &stride_v));
                        } else {
                            addr_parts_static += coord as i64 * stride_lit;
                        }
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
                        let offset = b.ir_mul_i(&idx_const, step);
                        let src_idx = b.ir_add_i(start, &offset);

                        // In-bounds check.
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

                        // Clamp for safe address.
                        let max_idx = b.ir_constant_int(*axis_len as i64 - 1);
                        let is_neg = b.ir_less_than_i(&src_idx, &zero);
                        let is_over = b.ir_greater_than_i(&src_idx, &max_idx);
                        let clamped_hi = b.ir_select_i(&is_over, &max_idx, &src_idx);
                        let clamped = b.ir_select_i(&is_neg, &zero, &clamped_hi);

                        addr_parts_dynamic.push(b.ir_mul_i(&clamped, &stride_v));
                        val_coord_idx += 1;
                    }
                }
                let _ = stride_lit; // silence unused when symbolic
            }

            // Compute address.
            let mut addr = b.ir_constant_int(addr_parts_static);
            for part in &addr_parts_dynamic {
                addr = b.ir_add_i(&addr, part);
            }

            // Combined in-bounds.
            let in_bounds = if in_bounds_parts.is_empty() {
                b.ir_constant_bool(true)
            } else {
                let mut acc = in_bounds_parts[0].clone();
                for ib in &in_bounds_parts[1..] {
                    acc = b.ir_logical_and(&acc, ib);
                }
                acc
            };

            // Read current, select, write.
            let current =
                crate::helpers::segment::read_typed(b, data.segment_id, &addr, data.dtype);
            let write_val = if value_is_scalar {
                cast_to_dtype(b, value, data.dtype)
            } else {
                cast_to_dtype(b, &value_elements[val_flat], data.dtype)
            };
            let selected =
                crate::helpers::value_ops::select_value(b, &in_bounds, &write_val, &current);
            crate::helpers::segment::write_typed(b, data.segment_id, &addr, &selected, data.dtype);
        }
    }

    Value::DynamicNDArray(data.clone())
}

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
            } else {
                b.ir_int_cast(v)
            }
        }
        NumberType::Complex => {
            let (real, imag) = crate::helpers::value_ops::unpack_value_to_complex_parts(b, v);
            crate::helpers::value_ops::pack_complex_value(real, imag)
        }
    }
}
