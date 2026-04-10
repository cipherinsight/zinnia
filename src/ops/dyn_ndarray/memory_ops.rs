use std::collections::HashMap;

use crate::builder::IRBuilder;
use crate::types::{
    DynArrayMeta, DynamicNDArrayData, NumberType, ScalarValue, Value,
};

use super::{
    dyn_decode_coords, dyn_encode_coords, dyn_num_elements, dyn_row_major_strides,
    value_to_scalar_i64,
};

pub fn dyn_filter(b: &mut IRBuilder, data: &DynamicNDArrayData, args: &[Value]) -> Value {
    let mask = args
        .first()
        .expect("filter: requires a mask argument");
    let elements: Vec<Value> = crate::helpers::segment::read_all(b, data.segment_id, data.max_length());

    // Get mask elements — from segment if dynamic, from composite if static.
    let mask_elements: Vec<Value> = match mask {
        Value::DynamicNDArray(md) => {
            crate::helpers::segment::read_all(b, md.segment_id, md.max_length())
        }
        Value::List(_) | Value::Tuple(_) => {
            crate::helpers::composite::flatten_composite(mask)
        }
        _ => panic!("filter: mask must be array-like"),
    };

    let max_len = data.max_length();

    // Allocate output segment pre-filled with defaults.
    let default_val = super::metadata::dyn_default_value(b, data.dtype);
    let default_sv = value_to_scalar_i64(&default_val);
    let default_elements = vec![default_sv; max_len];
    let segment_id = crate::helpers::segment::alloc_and_write(b, &default_elements, data.dtype);

    // Compaction via ZKRAM write pointer: for each input element, if mask
    // is true, write the element at write_ptr and advance. If false, write
    // a default at write_ptr (will be overwritten by the next kept element).
    // O(N) writes instead of O(N²) selects.
    let mut write_ptr = b.ir_constant_int(0);

    for i in 0..max_len.min(elements.len()) {
        let mask_val = if i < mask_elements.len() {
            mask_elements[i].clone()
        } else {
            b.ir_constant_int(0)
        };
        let keep = b.ir_bool_cast(&mask_val);

        // Write element or default at write_ptr. Non-kept writes are
        // harmless — they'll be overwritten by the next kept element or
        // sit beyond runtime_length.
        let val_to_write = if data.dtype == NumberType::Float {
            b.ir_select_f(&keep, &elements[i], &default_val)
        } else {
            b.ir_select_i(&keep, &elements[i], &default_val)
        };
        b.ir_write_memory(segment_id, &write_ptr, &val_to_write);

        // Advance write_ptr only when keep is true.
        let one = b.ir_constant_int(1);
        let zero = b.ir_constant_int(0);
        let inc = b.ir_select_i(&keep, &one, &zero);
        write_ptr = b.ir_add_i(&write_ptr, &inc);
    }

    // After filter, the runtime length is unknown (depends on the mask) but
    // bounded by the input's max. total_bound conserved from source (§3.8).
    let envelope = crate::types::Envelope::new_with_bound(
        vec![crate::types::Dim::new_dynamic(&mut b.dim_table, 0, max_len)],
        data.envelope.total_bound,
    );
    let result = DynamicNDArrayData {
        envelope,
        dtype: data.dtype,
        segment_id,
        meta: DynArrayMeta {
            logical_shape: vec![max_len],
            logical_offset: 0,
            logical_strides: vec![1],
            runtime_length: value_to_scalar_i64(&write_ptr),
            runtime_rank: ScalarValue::new(Some(1), None),
            runtime_shape: vec![value_to_scalar_i64(&write_ptr)],
            runtime_strides: vec![ScalarValue::new(Some(1), None)],
            runtime_offset: ScalarValue::new(Some(0), None),
        },
    };
    Value::DynamicNDArray(result)
}

/// DynamicNDArray.repeat(repeats, axis=...)
pub fn dyn_repeat(
    b: &mut IRBuilder,
    data: &DynamicNDArrayData,
    args: &[Value],
    kwargs: &HashMap<String, Value>,
) -> Value {
    let repeats = args
        .first()
        .and_then(|v| v.int_val())
        .expect("repeat: repeats must be a constant integer") as usize;
    let axis = kwargs
        .get("axis")
        .or_else(|| args.get(1))
        .and_then(|v| v.int_val());

    let values = crate::helpers::segment::read_all(b, data.segment_id, data.max_length());
    let numel = dyn_num_elements(&data.meta.logical_shape);

    if let Some(_ax) = axis {
        // For axis-specific repeat, flatten first, repeat, return 1D
        // (full axis support would need coordinate transforms)
        let mut new_elements = Vec::new();
        for val in values.iter().take(numel) {
            let sv = value_to_scalar_i64(val);
            for _ in 0..repeats {
                new_elements.push(sv.clone());
            }
        }
        let new_len = new_elements.len();
        let segment_id = crate::helpers::segment::alloc_and_write(b, &new_elements, data.dtype);
        let envelope = crate::types::Envelope::from_static_shape(&mut b.dim_table, &[new_len]);
        let result = DynamicNDArrayData {
            envelope,
            dtype: data.dtype,
            segment_id,
            meta: DynArrayMeta {
                logical_shape: vec![new_len],
                logical_offset: 0,
                logical_strides: vec![1],
                runtime_length: ScalarValue::new(Some(new_len as i64), None),
                runtime_rank: ScalarValue::new(Some(1), None),
                runtime_shape: vec![ScalarValue::new(Some(new_len as i64), None)],
                runtime_strides: vec![ScalarValue::new(Some(1), None)],
                runtime_offset: ScalarValue::new(Some(0), None),
            },
        };
        Value::DynamicNDArray(result)
    } else {
        // No axis: flatten then repeat each element
        let mut new_elements = Vec::new();
        for val in values.iter().take(numel) {
            let sv = value_to_scalar_i64(val);
            for _ in 0..repeats {
                new_elements.push(sv.clone());
            }
        }
        let new_len = new_elements.len();
        let segment_id = crate::helpers::segment::alloc_and_write(b, &new_elements, data.dtype);
        let envelope = crate::types::Envelope::from_static_shape(&mut b.dim_table, &[new_len]);
        let result = DynamicNDArrayData {
            envelope,
            dtype: data.dtype,
            segment_id,
            meta: DynArrayMeta {
                logical_shape: vec![new_len],
                logical_offset: 0,
                logical_strides: vec![1],
                runtime_length: ScalarValue::new(Some(new_len as i64), None),
                runtime_rank: ScalarValue::new(Some(1), None),
                runtime_shape: vec![ScalarValue::new(Some(new_len as i64), None)],
                runtime_strides: vec![ScalarValue::new(Some(1), None)],
                runtime_offset: ScalarValue::new(Some(0), None),
            },
        };
        Value::DynamicNDArray(result)
    }
}

/// Coerce a Value to a 1-D DynamicNDArrayData for concat/stack scalar support.
/// Scalars become length-1 arrays; existing DynamicNDArrays pass through.
fn coerce_to_dyn(b: &mut IRBuilder, v: &Value) -> DynamicNDArrayData {
    match v {
        Value::DynamicNDArray(d) => d.clone(),
        Value::Integer(_) | Value::Float(_) | Value::Boolean(_) => {
            let dtype = if matches!(v, Value::Float(_)) {
                NumberType::Float
            } else {
                NumberType::Integer
            };
            let sv = value_to_scalar_i64(v);
            let segment_id = crate::helpers::segment::alloc_and_write(b, &[sv], dtype);
            let envelope = crate::types::Envelope::from_static_shape(&mut b.dim_table, &[1]);
            DynamicNDArrayData {
                envelope,
                dtype,
                segment_id,
                meta: DynArrayMeta {
                    logical_shape: vec![1],
                    logical_offset: 0,
                    logical_strides: vec![1],
                    runtime_length: ScalarValue::new(Some(1), None),
                    runtime_rank: ScalarValue::new(Some(1), None),
                    runtime_shape: vec![ScalarValue::new(Some(1), None)],
                    runtime_strides: vec![ScalarValue::new(Some(1), None)],
                    runtime_offset: ScalarValue::new(Some(0), None),
                },
            }
        }
        _ => panic!("concatenate/stack: unsupported element type {:?}", v.zinnia_type()),
    }
}

/// Resolve output dtype: float wins over integer (NumPy promotion rule).
fn resolve_dtype(arrays: &[DynamicNDArrayData]) -> NumberType {
    if arrays.iter().any(|a| a.dtype == NumberType::Float) {
        NumberType::Float
    } else {
        NumberType::Integer
    }
}

/// Cast a read element to the target dtype if needed.
fn cast_element(b: &mut IRBuilder, val: &Value, src_dtype: NumberType, dst_dtype: NumberType) -> ScalarValue<i64> {
    if src_dtype == dst_dtype {
        return value_to_scalar_i64(val);
    }
    // Need to cast: int→float or float→int.
    let casted = match dst_dtype {
        NumberType::Float => b.ir_float_cast(val),
        NumberType::Integer => b.ir_int_cast(val),
    };
    value_to_scalar_i64(&casted)
}

/// Concatenate dynamic arrays along an existing axis.
///
/// Supports: dtype promotion, axis=None (flatten then concat), scalar inputs.
/// Output envelope: non-concat dims from first array, concat axis = sum.
/// total_bound = sum of input total_bounds.
pub fn dyn_concatenate(
    b: &mut IRBuilder,
    args: &[Value],
    kwargs: &HashMap<String, Value>,
) -> Value {
    let arrays_val = args
        .first()
        .expect("concatenate: requires arrays argument");
    let raw_values: Vec<Value> = match arrays_val {
        Value::List(cd) | Value::Tuple(cd) => cd.values.clone(),
        _ => panic!("concatenate: first arg must be list/tuple of arrays"),
    };
    assert!(!raw_values.is_empty(), "concatenate: need at least one array");

    // Check axis=None (flatten all then concatenate as 1D).
    let axis_val = kwargs
        .get("axis")
        .or_else(|| args.get(1));
    let axis_is_none = matches!(axis_val, Some(Value::None));

    // Coerce scalars to 1-d arrays.
    let mut arrays: Vec<DynamicNDArrayData> = raw_values
        .iter()
        .map(|v| coerce_to_dyn(b, v))
        .collect();

    // If axis=None, flatten all arrays to 1-d first.
    if axis_is_none {
        arrays = arrays
            .iter()
            .map(|a| {
                let total: usize = a.meta.logical_shape.iter().product();
                if a.meta.logical_shape.len() == 1 && a.meta.logical_shape[0] == total {
                    a.clone()
                } else {
                    // Flatten: reinterpret as 1-d (always contiguous).
                    let strides = vec![1];
                    let envelope = crate::types::Envelope::new_with_bound(
                        vec![crate::types::Dim::new_static(&mut b.dim_table, total)],
                        a.envelope.total_bound,
                    );
                    DynamicNDArrayData {
                        envelope,
                        dtype: a.dtype,
                        segment_id: a.segment_id,
                        meta: DynArrayMeta {
                            logical_shape: vec![total],
                            logical_offset: 0,
                            logical_strides: strides,
                            runtime_length: a.meta.runtime_length.clone(),
                            runtime_rank: ScalarValue::new(Some(1), None),
                            runtime_shape: vec![a.meta.runtime_length.clone()],
                            runtime_strides: vec![ScalarValue::new(Some(1), None)],
                            runtime_offset: ScalarValue::new(Some(0), None),
                        },
                    }
                }
            })
            .collect();
    }

    let axis = if axis_is_none {
        0i64
    } else {
        axis_val.and_then(|v| v.int_val()).unwrap_or(0)
    };
    let ndim = arrays[0].meta.logical_shape.len();
    let ax = if axis < 0 {
        (ndim as i64 + axis) as usize
    } else {
        axis as usize
    };
    assert!(ax < ndim, "concatenate: axis {} out of bounds for rank {}", ax, ndim);

    // Validate non-concat axes match.
    let base_shape = &arrays[0].meta.logical_shape;
    for (i, arr) in arrays.iter().enumerate().skip(1) {
        assert_eq!(
            arr.meta.logical_shape.len(), ndim,
            "concatenate: array {} has rank {} but expected {}",
            i, arr.meta.logical_shape.len(), ndim
        );
        for d in 0..ndim {
            if d != ax {
                assert_eq!(
                    arr.meta.logical_shape[d], base_shape[d],
                    "concatenate: array {} has size {} on axis {} but expected {}",
                    i, arr.meta.logical_shape[d], d, base_shape[d]
                );
            }
        }
    }

    // Max output shape: sum of max axis dims.
    let max_concat_dim: usize = arrays.iter().map(|a| a.meta.logical_shape[ax]).sum();
    let mut out_shape = base_shape.clone();
    out_shape[ax] = max_concat_dim;
    let out_numel: usize = out_shape.iter().product();
    let out_strides = dyn_row_major_strides(&out_shape);
    let out_dtype = resolve_dtype(&arrays);
    let out_ax_stride = out_strides[ax] as i64;

    // Check if any input has a dynamic concat axis.
    let has_dynamic_axis = arrays.iter().any(|a| {
        a.envelope.dims[ax].min != a.envelope.dims[ax].max
    });

    // Build envelope.
    let total_bound: usize = arrays.iter().map(|a| a.envelope.total_bound).sum();
    let mut out_dims: Vec<crate::types::Dim> = arrays[0].envelope.dims.clone();
    if has_dynamic_axis {
        out_dims[ax] = crate::types::Dim::new_dynamic(&mut b.dim_table, 0, max_concat_dim);
    } else {
        out_dims[ax] = crate::types::Dim::new_static(&mut b.dim_table, max_concat_dim);
    }
    let envelope = crate::types::Envelope::new_with_bound(out_dims, total_bound);

    // Non-concat axes: product for per-axis-position element count.
    let other_product: usize = out_shape.iter().enumerate()
        .filter(|&(d, _)| d != ax)
        .map(|(_, &s)| s)
        .product::<usize>()
        .max(1);
    let mut other_dims: Vec<usize> = Vec::new();
    for (d, &s) in base_shape.iter().enumerate() {
        if d != ax { other_dims.push(s); }
    }
    let other_dim_strides = dyn_row_major_strides(&other_dims);

    // Allocate output segment with defaults.
    let default_val = super::metadata::dyn_default_value(b, out_dtype);
    let default_sv = value_to_scalar_i64(&default_val);
    let init_elements = vec![default_sv; out_numel];
    let segment_id = crate::helpers::segment::alloc_and_write(b, &init_elements, out_dtype);
    let max_out_addr = (out_numel as i64 - 1).max(0);

    // Single pass: iterate per input, write to output at cumulative_offset.
    // For each input k, position i on concat axis (0..max_axis_k):
    //   source_addr = i * src_stride[ax] + other_offset
    //   output_addr = (cumulative_offset + i) * out_stride[ax] + other_offset
    //   in_bounds = i < runtime_shape_k[ax]
    //   write selected value or default
    // After input k: cumulative_offset += runtime_shape_k[ax]
    //
    // Out-of-bounds writes from input k land at positions that will be
    // overwritten by input k+1 (since cumulative_offset advances by
    // runtime length, not max length).

    let mut cumulative_offset = b.ir_constant_int(0);
    let mut runtime_concat_len = b.ir_constant_int(0);

    for (k, arr) in arrays.iter().enumerate() {
        let src_shape = &arr.meta.logical_shape;
        let src_strides_k = dyn_row_major_strides(src_shape);
        let src_ax_stride = src_strides_k[ax] as i64;
        let src_max_ax = src_shape[ax];
        let src_max_len = arr.max_length() as i64;

        // Runtime axis length for this input (may be dynamic).
        let runtime_ax_len: Value = if let Some(sv) = arr.meta.runtime_shape.get(ax) {
            if let Some(s) = sv.static_val {
                b.ir_constant_int(s)
            } else if let Some(ptr) = sv.ptr {
                Value::Integer(ScalarValue::new(None, Some(ptr)))
            } else {
                b.ir_constant_int(src_max_ax as i64)
            }
        } else {
            b.ir_constant_int(src_max_ax as i64)
        };

        for i in 0..src_max_ax {
            let i_val = b.ir_constant_int(i as i64);
            let in_bounds = b.ir_less_than_i(&i_val, &runtime_ax_len);

            // Source address: i * src_stride[ax] (base for this axis position).
            let src_ax_base = i as i64 * src_ax_stride;

            // Output axis position = cumulative_offset + i (dynamic).
            let out_ax_pos = b.ir_add_i(&cumulative_offset, &i_val);
            let out_ax_stride_val = b.ir_constant_int(out_ax_stride);
            let out_ax_base = b.ir_mul_i(&out_ax_pos, &out_ax_stride_val);

            for other_flat in 0..other_product {
                let other_coords = dyn_decode_coords(other_flat, &other_dims, &other_dim_strides);

                // Compute non-concat address contributions (static for both src and out).
                let mut src_other_addr: i64 = 0;
                let mut out_other_addr: i64 = 0;
                let mut oc_idx = 0;
                for d in 0..ndim {
                    if d != ax {
                        src_other_addr += other_coords[oc_idx] as i64 * src_strides_k[d] as i64;
                        out_other_addr += other_coords[oc_idx] as i64 * out_strides[d] as i64;
                        oc_idx += 1;
                    }
                }

                // Source: read via ir_read_memory with clamped address.
                let src_addr_static = src_ax_base + src_other_addr;
                let clamped_src = src_addr_static.max(0).min(src_max_len - 1);
                let src_addr_val = b.ir_constant_int(clamped_src);
                let src_val = b.ir_read_memory(arr.segment_id, &src_addr_val);

                // Cast dtype if needed.
                let casted = if arr.dtype == out_dtype {
                    src_val
                } else {
                    match out_dtype {
                        NumberType::Float => b.ir_float_cast(&src_val),
                        NumberType::Integer => b.ir_int_cast(&src_val),
                    }
                };

                // Output address = out_ax_base + out_other_addr (dynamic + static).
                let out_other_val = b.ir_constant_int(out_other_addr);
                let out_addr = b.ir_add_i(&out_ax_base, &out_other_val);

                // Clamp output address for safety.
                let max_addr_val = b.ir_constant_int(max_out_addr);
                let zero_val = b.ir_constant_int(0);
                let is_neg = b.ir_less_than_i(&out_addr, &zero_val);
                let clamped_lo = b.ir_select_i(&is_neg, &zero_val, &out_addr);
                let is_over = b.ir_greater_than_i(&clamped_lo, &max_addr_val);
                let clamped_out = b.ir_select_i(&is_over, &max_addr_val, &clamped_lo);

                // Select value: in_bounds → casted source, else → default.
                let write_val = if out_dtype == NumberType::Float {
                    b.ir_select_f(&in_bounds, &casted, &default_val)
                } else {
                    b.ir_select_i(&in_bounds, &casted, &default_val)
                };

                b.ir_write_memory(segment_id, &clamped_out, &write_val);
            }
        }

        // Advance cumulative offset by this input's runtime axis length.
        cumulative_offset = b.ir_add_i(&cumulative_offset, &runtime_ax_len);
        runtime_concat_len = b.ir_add_i(&runtime_concat_len, &runtime_ax_len);
    }

    // Runtime metadata.
    let other_prod_val = b.ir_constant_int(other_product as i64);
    let runtime_length = b.ir_mul_i(&runtime_concat_len, &other_prod_val);

    let mut runtime_shape: Vec<ScalarValue<i64>> = out_shape
        .iter()
        .map(|&s| ScalarValue::new(Some(s as i64), None))
        .collect();
    if has_dynamic_axis {
        runtime_shape[ax] = value_to_scalar_i64(&runtime_concat_len);
    }

    Value::DynamicNDArray(DynamicNDArrayData {
        envelope,
        dtype: out_dtype,
        segment_id,
        meta: DynArrayMeta {
            logical_shape: out_shape.clone(),
            logical_offset: 0,
            logical_strides: out_strides.clone(),
            runtime_length: value_to_scalar_i64(&runtime_length),
            runtime_rank: ScalarValue::new(Some(ndim as i64), None),
            runtime_shape,
            runtime_strides: out_strides
                .iter()
                .map(|&s| ScalarValue::new(Some(s as i64), None))
                .collect(),
            runtime_offset: ScalarValue::new(Some(0), None),
        },
    })
}

/// Stack dynamic arrays along a new axis.
///
/// Supports: dtype promotion, scalar inputs (treated as 0-d → stacked as 1-d).
/// Output envelope: inserts new static dim (= num_arrays), other dims from first.
/// total_bound = N * T_input.
pub fn dyn_stack(
    b: &mut IRBuilder,
    args: &[Value],
    kwargs: &HashMap<String, Value>,
) -> Value {
    let arrays_val = args
        .first()
        .expect("stack: requires arrays argument");
    let raw_values: Vec<Value> = match arrays_val {
        Value::List(cd) | Value::Tuple(cd) => cd.values.clone(),
        _ => panic!("stack: first arg must be list/tuple of arrays"),
    };
    assert!(!raw_values.is_empty(), "stack: need at least one array");

    let arrays: Vec<DynamicNDArrayData> = raw_values
        .iter()
        .map(|v| coerce_to_dyn(b, v))
        .collect();

    let axis = kwargs
        .get("axis")
        .or_else(|| args.get(1))
        .and_then(|v| v.int_val())
        .unwrap_or(0);
    let base_shape = &arrays[0].meta.logical_shape;
    let ndim = base_shape.len();
    let ax = if axis < 0 {
        (ndim as i64 + 1 + axis) as usize
    } else {
        axis as usize
    };
    assert!(ax <= ndim, "stack: axis {} out of bounds for rank {}", ax, ndim);

    // Validate all arrays have the same shape.
    for (i, arr) in arrays.iter().enumerate().skip(1) {
        assert_eq!(
            &arr.meta.logical_shape, base_shape,
            "stack: array {} has shape {:?} but expected {:?}",
            i, arr.meta.logical_shape, base_shape
        );
    }

    let num_arrays = arrays.len();
    let mut out_shape = base_shape.clone();
    out_shape.insert(ax, num_arrays);
    let out_numel: usize = out_shape.iter().product();
    let out_strides = dyn_row_major_strides(&out_shape);
    let out_dtype = resolve_dtype(&arrays);

    // Build envelope.
    let total_bound = num_arrays.saturating_mul(arrays[0].envelope.total_bound);
    let mut out_dims = arrays[0].envelope.dims.clone();
    out_dims.insert(ax, crate::types::Dim::new_static(&mut b.dim_table, num_arrays));
    let envelope = crate::types::Envelope::new_with_bound(out_dims, total_bound);

    // Check if any input has dynamic element count.
    let has_dynamic_input = arrays.iter().any(|a| {
        a.envelope.dims.iter().any(|d| d.min != d.max)
    });

    // Allocate output segment with defaults.
    let default_val = super::metadata::dyn_default_value(b, out_dtype);
    let default_sv = value_to_scalar_i64(&default_val);
    let init_elements = vec![default_sv; out_numel];
    let segment_id = crate::helpers::segment::alloc_and_write(b, &init_elements, out_dtype);
    let max_out_addr = (out_numel as i64 - 1).max(0);

    // Per-input pass: for each input k, write its elements into the output
    // at stack axis position k. Use ir_read_memory for source access and
    // in_bounds checks based on runtime_length for dynamic inputs.
    let src_max_numel: usize = base_shape.iter().product();
    let src_strides_base = dyn_row_major_strides(base_shape);

    for (k, arr) in arrays.iter().enumerate() {
        let src_numel = arr.max_length();

        // Runtime length for this input (may be dynamic).
        let runtime_len: Value = if let Some(sv) = &arr.meta.runtime_length.static_val {
            b.ir_constant_int(*sv)
        } else if let Some(ptr) = arr.meta.runtime_length.ptr {
            Value::Integer(ScalarValue::new(None, Some(ptr)))
        } else {
            b.ir_constant_int(src_numel as i64)
        };

        for src_flat in 0..src_max_numel {
            let src_flat_val = b.ir_constant_int(src_flat as i64);
            let in_bounds = b.ir_less_than_i(&src_flat_val, &runtime_len);

            // Read from source via ir_read_memory.
            let clamped_src = (src_flat as i64).min(src_numel as i64 - 1).max(0);
            let src_addr = b.ir_constant_int(clamped_src);
            let src_val = b.ir_read_memory(arr.segment_id, &src_addr);

            // Cast dtype if needed.
            let casted = if arr.dtype == out_dtype {
                src_val
            } else {
                match out_dtype {
                    NumberType::Float => b.ir_float_cast(&src_val),
                    NumberType::Integer => b.ir_int_cast(&src_val),
                }
            };

            // Compute output address: decode src_flat to source coords,
            // insert k at stack axis, encode with output strides.
            let src_coords = dyn_decode_coords(src_flat, base_shape, &src_strides_base);
            let mut out_coords = src_coords.clone();
            out_coords.insert(ax, k);
            let out_flat = dyn_encode_coords(&out_coords, &out_strides) as i64;
            let out_addr = b.ir_constant_int(out_flat.min(max_out_addr).max(0));

            let write_val = if out_dtype == NumberType::Float {
                b.ir_select_f(&in_bounds, &casted, &default_val)
            } else {
                b.ir_select_i(&in_bounds, &casted, &default_val)
            };

            b.ir_write_memory(segment_id, &out_addr, &write_val);
        }
    }

    // Runtime metadata: stacked axis is always static (num_arrays).
    // Other dims may be dynamic — use first input's runtime_shape.
    let mut runtime_shape: Vec<ScalarValue<i64>> = Vec::new();
    let mut src_dim_idx = 0;
    for d in 0..out_shape.len() {
        if d == ax {
            runtime_shape.push(ScalarValue::new(Some(num_arrays as i64), None));
        } else {
            if src_dim_idx < arrays[0].meta.runtime_shape.len() {
                runtime_shape.push(arrays[0].meta.runtime_shape[src_dim_idx].clone());
            } else {
                runtime_shape.push(ScalarValue::new(Some(out_shape[d] as i64), None));
            }
            src_dim_idx += 1;
        }
    }

    // runtime_length = num_arrays * input_runtime_length.
    let num_arr_val = b.ir_constant_int(num_arrays as i64);
    let input_runtime_len = if let Some(sv) = &arrays[0].meta.runtime_length.static_val {
        b.ir_constant_int(*sv)
    } else if let Some(ptr) = arrays[0].meta.runtime_length.ptr {
        Value::Integer(ScalarValue::new(None, Some(ptr)))
    } else {
        b.ir_constant_int(src_max_numel as i64)
    };
    let runtime_length = b.ir_mul_i(&num_arr_val, &input_runtime_len);

    Value::DynamicNDArray(DynamicNDArrayData {
        envelope,
        dtype: out_dtype,
        segment_id,
        meta: DynArrayMeta {
            logical_shape: out_shape.clone(),
            logical_offset: 0,
            logical_strides: out_strides.clone(),
            runtime_length: value_to_scalar_i64(&runtime_length),
            runtime_rank: ScalarValue::new(Some(out_shape.len() as i64), None),
            runtime_shape,
            runtime_strides: out_strides
                .iter()
                .map(|&s| ScalarValue::new(Some(s as i64), None))
                .collect(),
            runtime_offset: ScalarValue::new(Some(0), None),
        },
    })
}

/// Split a dynamic array into parts along an axis.
///
/// `sections`: either an integer N (equal division) or a list of split indices.
/// Split indices may be static (compile-time) or dynamic (circuit wires).
/// `allow_unequal`: if true, integer N allows unequal chunks (array_split).
///
/// Static indices → O(N) fast path. Dynamic indices → O(K*axis_len) pad-and-mask.
/// Returns a List of DynamicNDArrays.
pub fn dyn_split(
    b: &mut IRBuilder,
    data: &DynamicNDArrayData,
    args: &[Value],
    kwargs: &HashMap<String, Value>,
) -> Value {
    dyn_split_impl(b, data, args, kwargs, false)
}

/// array_split: like split but allows unequal chunks for integer N.
pub fn dyn_array_split(
    b: &mut IRBuilder,
    data: &DynamicNDArrayData,
    args: &[Value],
    kwargs: &HashMap<String, Value>,
) -> Value {
    dyn_split_impl(b, data, args, kwargs, true)
}

fn dyn_split_impl(
    b: &mut IRBuilder,
    data: &DynamicNDArrayData,
    args: &[Value],
    kwargs: &HashMap<String, Value>,
    allow_unequal: bool,
) -> Value {
    let sections_val = args.get(0).expect("split: requires sections argument");
    let axis = kwargs
        .get("axis")
        .or_else(|| args.get(1))
        .and_then(|v| v.int_val())
        .unwrap_or(0);
    let ndim = data.meta.logical_shape.len();
    let ax = if axis < 0 {
        (ndim as i64 + axis) as usize
    } else {
        axis as usize
    };
    assert!(ax < ndim, "split: axis {} out of bounds for rank {}", ax, ndim);

    let axis_len = data.meta.logical_shape[ax];

    // Parse sections into split-point Values (may be static or dynamic).
    // Also track whether all are static for the fast path.
    let split_values: Vec<Value>;
    let num_chunks: usize;

    if let Some(n) = sections_val.int_val() {
        let n = n as usize;
        assert!(n > 0, "split: number of sections must be positive");
        if !allow_unequal {
            assert!(
                axis_len % n == 0,
                "split: array of size {} on axis {} cannot be split into {} equal parts",
                axis_len, ax, n
            );
        }
        // Compute split points for equal/unequal division.
        // array_split: first (axis_len % n) chunks get ceil(axis_len/n),
        // remaining get floor(axis_len/n).
        let mut points = Vec::new();
        let base = axis_len / n;
        let extra = axis_len % n;
        let mut pos = 0usize;
        for i in 0..n - 1 {
            pos += base + if i < extra { 1 } else { 0 };
            points.push(Value::Integer(ScalarValue::new(Some(pos as i64), None)));
        }
        split_values = points;
        num_chunks = n;
    } else {
        match sections_val {
            Value::List(cd) | Value::Tuple(cd) => {
                split_values = cd.values.clone();
                num_chunks = cd.values.len() + 1;
            }
            _ => panic!("split: sections must be int or list of ints"),
        }
    }

    let all_static = split_values.iter().all(|v| v.int_val().is_some());

    // Pre-read source segment.
    let src_vals = crate::helpers::segment::read_all(b, data.segment_id, data.max_length());
    let src_strides = dyn_row_major_strides(&data.meta.logical_shape);

    let mut result_arrays: Vec<Value> = Vec::with_capacity(num_chunks);

    if all_static {
        // ── Fast path: all split points are compile-time constants ──
        let static_points: Vec<usize> = split_values
            .iter()
            .map(|v| v.int_val().unwrap().min(axis_len as i64).max(0) as usize)
            .collect();

        let mut ranges: Vec<(usize, usize)> = Vec::new();
        let mut prev = 0;
        for &sp in &static_points {
            ranges.push((prev, sp.min(axis_len)));
            prev = sp.min(axis_len);
        }
        ranges.push((prev, axis_len));

        for (start, end) in &ranges {
            let chunk_len = end - start;
            let mut chunk_shape = data.meta.logical_shape.clone();
            chunk_shape[ax] = chunk_len;
            let chunk_numel: usize = chunk_shape.iter().product();
            let chunk_strides = dyn_row_major_strides(&chunk_shape);

            let mut chunk_elements = Vec::with_capacity(chunk_numel);
            for flat in 0..chunk_numel {
                let chunk_coords = dyn_decode_coords(flat, &chunk_shape, &chunk_strides);
                let mut src_coords = chunk_coords.clone();
                src_coords[ax] += start;
                let src_flat = dyn_encode_coords(&src_coords, &src_strides);
                chunk_elements.push(value_to_scalar_i64(&src_vals[src_flat]));
            }

            let segment_id = crate::helpers::segment::alloc_and_write(b, &chunk_elements, data.dtype);
            let mut chunk_dims = data.envelope.dims.clone();
            chunk_dims[ax] = crate::types::Dim::new_static(&mut b.dim_table, chunk_len);
            let chunk_bound = chunk_numel.min(data.envelope.total_bound);
            let envelope = crate::types::Envelope::new_with_bound(chunk_dims, chunk_bound);

            result_arrays.push(Value::DynamicNDArray(DynamicNDArrayData {
                envelope,
                dtype: data.dtype,
                segment_id,
                meta: DynArrayMeta {
                    logical_shape: chunk_shape.clone(),
                    logical_offset: 0,
                    logical_strides: chunk_strides.clone(),
                    runtime_length: ScalarValue::new(Some(chunk_numel as i64), None),
                    runtime_rank: ScalarValue::new(Some(ndim as i64), None),
                    runtime_shape: chunk_shape
                        .iter()
                        .map(|&s| ScalarValue::new(Some(s as i64), None))
                        .collect(),
                    runtime_strides: chunk_strides
                        .iter()
                        .map(|&s| ScalarValue::new(Some(s as i64), None))
                        .collect(),
                    runtime_offset: ScalarValue::new(Some(0), None),
                },
            }));
        }
    } else {
        // ── Dynamic path: direct address computation per chunk ──
        // For each chunk k with boundaries [start_k, end_k), output position i
        // maps to source axis position (start_k + i). We use ir_read_memory
        // with the computed source address — no iteration over all positions.
        //
        // Each chunk is allocated at max size (axis_len). Positions where
        // i >= (end_k - start_k) get default values.
        // Total IR ops: K * axis_len * other_product (reads + writes).

        // Build boundary Values: [0, sp[0], sp[1], ..., axis_len]
        let zero = b.ir_constant_int(0);
        let axis_len_val = b.ir_constant_int(axis_len as i64);
        let mut boundaries: Vec<Value> = Vec::with_capacity(num_chunks + 1);
        boundaries.push(zero);
        for sv in &split_values {
            let v = if let Some(s) = sv.int_val() {
                b.ir_constant_int(s)
            } else {
                sv.clone()
            };
            // Clamp to [0, axis_len].
            let clamped_lo = {
                let zero_v = b.ir_constant_int(0);
                let is_neg = b.ir_less_than_i(&v, &zero_v);
                b.ir_select_i(&is_neg, &zero_v, &v)
            };
            let clamped = {
                let is_over = b.ir_greater_than_i(&clamped_lo, &axis_len_val);
                b.ir_select_i(&is_over, &axis_len_val, &clamped_lo)
            };
            boundaries.push(clamped);
        }
        boundaries.push(axis_len_val);

        // Non-split axes dimensions and strides.
        let other_product: usize = data.meta.logical_shape.iter().enumerate()
            .filter(|&(i, _)| i != ax)
            .map(|(_, &s)| s)
            .product::<usize>()
            .max(1);
        let mut other_dims: Vec<usize> = Vec::new();
        for (d, &s) in data.meta.logical_shape.iter().enumerate() {
            if d != ax { other_dims.push(s); }
        }
        let other_strides = dyn_row_major_strides(&other_dims);

        let max_chunk_shape = {
            let mut s = data.meta.logical_shape.clone();
            s[ax] = axis_len;
            s
        };
        let max_chunk_numel = axis_len * other_product;
        let max_chunk_strides = dyn_row_major_strides(&max_chunk_shape);

        let default_val = super::metadata::dyn_default_value(b, data.dtype);
        let default_sv = value_to_scalar_i64(&default_val);
        let ax_stride = src_strides[ax] as i64;
        let max_addr = (data.max_length() as i64 - 1).max(0);

        for chunk_idx in 0..num_chunks {
            let chunk_start = &boundaries[chunk_idx];
            let chunk_end = &boundaries[chunk_idx + 1];

            // chunk_len = end - start (dynamic).
            let chunk_len = b.ir_sub_i(chunk_end, chunk_start);

            // Allocate output segment with defaults.
            let init_elements = vec![default_sv.clone(); max_chunk_numel];
            let segment_id = crate::helpers::segment::alloc_and_write(b, &init_elements, data.dtype);

            // For each output position i on the split axis, compute source
            // address = (start_k + i) * stride[ax] + other_offset.
            // In-bounds: i < chunk_len.
            for i in 0..axis_len {
                let i_val = b.ir_constant_int(i as i64);
                let in_bounds = b.ir_less_than_i(&i_val, &chunk_len);

                // source axis position = start_k + i
                let src_axis_pos = b.ir_add_i(chunk_start, &i_val);
                // Base address contribution from split axis.
                let ax_stride_val = b.ir_constant_int(ax_stride);
                let ax_addr_part = b.ir_mul_i(&src_axis_pos, &ax_stride_val);

                for other_flat in 0..other_product {
                    let other_coords = dyn_decode_coords(other_flat, &other_dims, &other_strides);

                    // Compute non-split address contribution (static).
                    let mut other_addr: i64 = 0;
                    let mut oc_idx = 0;
                    for d in 0..ndim {
                        if d != ax {
                            other_addr += other_coords[oc_idx] as i64 * src_strides[d] as i64;
                            oc_idx += 1;
                        }
                    }
                    let other_addr_val = b.ir_constant_int(other_addr);

                    // Full source address = ax_addr_part + other_addr.
                    let src_addr = b.ir_add_i(&ax_addr_part, &other_addr_val);

                    // Clamp address for out-of-bounds safety.
                    let max_addr_val = b.ir_constant_int(max_addr);
                    let zero_val = b.ir_constant_int(0);
                    let is_neg = b.ir_less_than_i(&src_addr, &zero_val);
                    let clamped_lo = b.ir_select_i(&is_neg, &zero_val, &src_addr);
                    let is_over = b.ir_greater_than_i(&clamped_lo, &max_addr_val);
                    let clamped_addr = b.ir_select_i(&is_over, &max_addr_val, &clamped_lo);

                    // Random access read from source.
                    let src_val = b.ir_read_memory(data.segment_id, &clamped_addr);

                    // Select: in-bounds → source value, else → default.
                    let write_val = if data.dtype == NumberType::Float {
                        b.ir_select_f(&in_bounds, &src_val, &default_val)
                    } else {
                        b.ir_select_i(&in_bounds, &src_val, &default_val)
                    };

                    // Direct output address: i * other_product + other_flat.
                    let out_addr = b.ir_constant_int((i * other_product + other_flat) as i64);
                    b.ir_write_memory(segment_id, &out_addr, &write_val);
                }
            }

            // Envelope and runtime metadata.
            let mut chunk_dims = data.envelope.dims.clone();
            chunk_dims[ax] = crate::types::Dim::new_dynamic(&mut b.dim_table, 0, axis_len);
            let chunk_bound = max_chunk_numel.min(data.envelope.total_bound);
            let envelope = crate::types::Envelope::new_with_bound(chunk_dims, chunk_bound);

            let mut runtime_shape: Vec<ScalarValue<i64>> = data.meta.logical_shape
                .iter()
                .map(|&s| ScalarValue::new(Some(s as i64), None))
                .collect();
            let runtime_axis_len = b.ir_sub_i(chunk_end, chunk_start);
            runtime_shape[ax] = value_to_scalar_i64(&runtime_axis_len);

            let other_prod_val = b.ir_constant_int(other_product as i64);
            let runtime_length = b.ir_mul_i(&runtime_axis_len, &other_prod_val);

            result_arrays.push(Value::DynamicNDArray(DynamicNDArrayData {
                envelope,
                dtype: data.dtype,
                segment_id,
                meta: DynArrayMeta {
                    logical_shape: max_chunk_shape.clone(),
                    logical_offset: 0,
                    logical_strides: max_chunk_strides.clone(),
                    runtime_length: value_to_scalar_i64(&runtime_length),
                    runtime_rank: ScalarValue::new(Some(ndim as i64), None),
                    runtime_shape,
                    runtime_strides: max_chunk_strides
                        .iter()
                        .map(|&s| ScalarValue::new(Some(s as i64), None))
                        .collect(),
                    runtime_offset: ScalarValue::new(Some(0), None),
                },
            }));
        }
    }

    let types = result_arrays.iter().map(|v| v.zinnia_type()).collect();
    Value::List(crate::types::CompositeData {
        elements_type: types,
        values: result_arrays,
    })
}
