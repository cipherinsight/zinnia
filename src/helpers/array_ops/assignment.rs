//! Dynamic array element and masked assignment.

use crate::builder::IRBuilder;
use crate::types::{
    DynArrayMeta, DynamicNDArrayData, NumberType, ScalarValue, SliceIndex, Value,
};

use super::indexing::compute_flat_addr;

/// Single element assignment: dyn[i] = x or dyn[i, j] = x.
/// In-place write via `ir_write_memory` — O(1).
pub fn dyn_setitem(
    b: &mut IRBuilder,
    data: &DynamicNDArrayData,
    indices: &[SliceIndex],
    value: &Value,
) -> Value {
    let addr = compute_flat_addr(b, data, indices);

    let write_val = cast_to_dtype(b, value, data.dtype);
    b.ir_write_memory(data.segment_id, &addr, &write_val);

    Value::DynamicNDArray(data.clone())
}

/// Masked assignment: dyn[mask] = x.
/// Reads all elements, selects new or old per mask, writes to fresh segment.
pub fn dyn_setitem_mask(
    b: &mut IRBuilder,
    data: &DynamicNDArrayData,
    mask: &Value,
    value: &Value,
) -> Value {
    let max_len = data.max_length();

    let mask_elements: Vec<Value> = match mask {
        Value::DynamicNDArray(md) => {
            crate::helpers::segment::read_all(b, md.segment_id, md.max_length())
        }
        Value::List(_) | Value::Tuple(_) => {
            crate::helpers::composite::flatten_composite(mask)
        }
        _ => panic!("masked assignment: mask must be array-like"),
    };

    let value_is_scalar = value.is_number();
    let value_elements: Vec<Value> = if value_is_scalar {
        vec![]
    } else {
        match value {
            Value::DynamicNDArray(vd) => {
                crate::helpers::segment::read_all(b, vd.segment_id, vd.max_length())
            }
            Value::List(_) | Value::Tuple(_) => {
                crate::helpers::composite::flatten_composite(value)
            }
            _ => panic!("masked assignment: value must be scalar or array-like"),
        }
    };

    let current_vals = crate::helpers::segment::read_all(b, data.segment_id, max_len);
    let mut out_elements = Vec::with_capacity(max_len);

    for i in 0..max_len {
        let mask_val = if i < mask_elements.len() {
            mask_elements[i].clone()
        } else {
            b.ir_constant_int(0)
        };
        let keep = b.ir_bool_cast(&mask_val);

        let new_val = if value_is_scalar {
            cast_to_dtype(b, value, data.dtype)
        } else if i < value_elements.len() {
            cast_to_dtype(b, &value_elements[i.min(value_elements.len() - 1)], data.dtype)
        } else {
            crate::ops::dyn_ndarray::metadata::dyn_default_value(b, data.dtype)
        };

        let selected = if data.dtype == NumberType::Float {
            b.ir_select_f(&keep, &new_val, &current_vals[i])
        } else {
            b.ir_select_i(&keep, &new_val, &current_vals[i])
        };
        out_elements.push(crate::ops::dyn_ndarray::value_to_scalar_i64(&selected));
    }

    let segment_id = crate::helpers::segment::alloc_and_write(b, &out_elements, data.dtype);

    Value::DynamicNDArray(DynamicNDArrayData {
        envelope: data.envelope.clone(),
        dtype: data.dtype,
        segment_id,
        meta: data.meta.clone(),
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
    let strides = &data.meta.logical_strides;
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
        Single(Value),          // one coordinate (static or dynamic)
        Static(Vec<usize>),     // known list of coordinates
        Dynamic {               // pad-and-mask
            start: Value,
            stop: Value,
            step: Value,
            max_len: usize,
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

                let all_static = start.is_none() || matches!(start.as_ref(), Some(Value::None)) || s_static.is_some();
                let all_static = all_static
                    && (stop.is_none() || matches!(stop.as_ref(), Some(Value::None)) || e_static.is_some());
                let all_static = all_static
                    && (step.is_none() || matches!(step.as_ref(), Some(Value::None)) || st_static.is_some());

                if all_static {
                    let s = s_static.unwrap_or(0);
                    let e = e_static.unwrap_or(dim);
                    let st = st_static.unwrap_or(1);
                    let s = if s < 0 { (dim + s).max(0) } else { s.min(dim) } as usize;
                    let e = if e < 0 { (dim + e).max(0) } else { e.min(dim) } as usize;
                    let mut coords = Vec::new();
                    if st > 0 {
                        let mut i = s;
                        while i < e { coords.push(i); i += st as usize; }
                    } else {
                        let mut i = s;
                        while i > e { coords.push(i); i = i.wrapping_sub((-st) as usize); }
                    }
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
                    axis_specs.push(AxisSpec {
                        coords: AxisCoords::Dynamic {
                            start: start_ir,
                            stop: stop_ir,
                            step: step_ir,
                            max_len,
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
                crate::helpers::segment::read_all(b, vd.segment_id, vd.max_length())
            }
            Value::List(_) | Value::Tuple(_) => {
                crate::helpers::composite::flatten_composite(value)
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
                value_shape, value_total, value_elements.len()
            );
        }
    }

    // Collect the Range/trailing axes for iteration.
    let range_axes: Vec<(usize, &AxisSpec)> = axis_specs.iter().enumerate()
        .filter(|(_, spec)| !matches!(spec.coords, AxisCoords::Single(_)))
        .collect();

    let value_total: usize = value_shape.iter().product();
    let value_strides = crate::helpers::shape_arith::row_major_strides(&value_shape);

    if !has_dynamic_range {
        // All static: iterate over all target positions and write.
        for val_flat in 0..value_total {
            let val_coords = crate::helpers::shape_arith::decode_coords(val_flat, &value_shape, &value_strides);

            // Compute target flat address.
            let mut addr_static: i64 = 0;
            let mut val_coord_idx = 0;

            for spec in &axis_specs {
                match &spec.coords {
                    AxisCoords::Single(v) => {
                        if let Some(i) = v.int_val() {
                            let i = if i < 0 { shape[spec.src_axis] as i64 + i } else { i };
                            addr_static += i * strides[spec.src_axis] as i64;
                        } else {
                            // Dynamic single index — need IR arithmetic.
                            // For simplicity, fall through to the dynamic path below.
                            panic!("slice assignment with dynamic single index + static range not yet optimized");
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

            let addr = b.ir_constant_int(addr_static);
            let write_val = if value_is_scalar {
                cast_to_dtype(b, value, data.dtype)
            } else {
                cast_to_dtype(b, &value_elements[val_flat], data.dtype)
            };
            b.ir_write_memory(data.segment_id, &addr, &write_val);
        }
    } else {
        // Has dynamic range: iterate over max positions, conditionally write.
        // Read current values first (for select on out-of-bounds positions).
        for val_flat in 0..value_total {
            let val_coords = crate::helpers::shape_arith::decode_coords(val_flat, &value_shape, &value_strides);

            let mut addr_parts_static: i64 = 0;
            let mut addr_parts_dynamic: Vec<Value> = Vec::new();
            let mut in_bounds_parts: Vec<Value> = Vec::new();
            let mut val_coord_idx = 0;

            for spec in &axis_specs {
                let stride = strides[spec.src_axis] as i64;
                match &spec.coords {
                    AxisCoords::Single(v) => {
                        if let Some(i) = v.int_val() {
                            let i = if i < 0 { shape[spec.src_axis] as i64 + i } else { i };
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
                    AxisCoords::Dynamic { start, stop, step, max_len, axis_len } => {
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

                        let stride_val = b.ir_constant_int(stride);
                        addr_parts_dynamic.push(b.ir_mul_i(&clamped, &stride_val));
                        val_coord_idx += 1;
                    }
                }
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
            let current = b.ir_read_memory(data.segment_id, &addr);
            let write_val = if value_is_scalar {
                cast_to_dtype(b, value, data.dtype)
            } else {
                cast_to_dtype(b, &value_elements[val_flat], data.dtype)
            };
            let selected = if data.dtype == NumberType::Float {
                b.ir_select_f(&in_bounds, &write_val, &current)
            } else {
                b.ir_select_i(&in_bounds, &write_val, &current)
            };
            b.ir_write_memory(data.segment_id, &addr, &selected);
        }
    }

    Value::DynamicNDArray(data.clone())
}

fn cast_to_dtype(b: &mut IRBuilder, v: &Value, dtype: NumberType) -> Value {
    match dtype {
        NumberType::Float => {
            if matches!(v, Value::Float(_)) { v.clone() } else { b.ir_float_cast(v) }
        }
        NumberType::Integer => {
            if matches!(v, Value::Integer(_)) { v.clone() } else { b.ir_int_cast(v) }
        }
    }
}
