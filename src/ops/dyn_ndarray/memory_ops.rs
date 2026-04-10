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

/// DynamicNDArray.concatenate(arrays, axis=0)
pub fn dyn_concatenate(
    b: &mut IRBuilder,
    args: &[Value],
    kwargs: &HashMap<String, Value>,
) -> Value {
    let arrays_val = args
        .first()
        .expect("concatenate: requires arrays argument");
    let arrays: Vec<DynamicNDArrayData> = match arrays_val {
        Value::List(cd) | Value::Tuple(cd) => cd
            .values
            .iter()
            .map(|v| match v {
                Value::DynamicNDArray(d) => d.clone(),
                _ => panic!("concatenate: all elements must be DynamicNDArray"),
            })
            .collect(),
        _ => panic!("concatenate: first arg must be list/tuple of arrays"),
    };
    if arrays.is_empty() {
        return Value::None;
    }

    let axis = kwargs
        .get("axis")
        .or_else(|| args.get(1))
        .and_then(|v| v.int_val())
        .unwrap_or(0);
    let ndim = arrays[0].meta.logical_shape.len();
    let ax = if axis < 0 {
        (ndim as i64 + axis) as usize
    } else {
        axis as usize
    };
    assert!(ax < ndim, "concatenate: axis out of bounds");

    // Compute output shape
    let base_shape = &arrays[0].meta.logical_shape;
    let mut out_shape = base_shape.clone();
    let concat_dim: usize = arrays.iter().map(|a| a.meta.logical_shape[ax]).sum();
    out_shape[ax] = concat_dim;
    let out_numel: usize = out_shape.iter().product();
    let out_strides = dyn_row_major_strides(&out_shape);

    // Build output elements by coordinate mapping
    let mut out_elements = Vec::with_capacity(out_numel);
    // Compute axis offsets for each source
    let mut axis_offsets = Vec::new();
    let mut offset = 0usize;
    for arr in &arrays {
        axis_offsets.push(offset);
        offset += arr.meta.logical_shape[ax];
    }

    let dtype = arrays[0].dtype;

    for i in 0..out_numel {
        let out_coords = dyn_decode_coords(i, &out_shape, &out_strides);

        // Find which source array this coordinate belongs to
        let ax_coord = out_coords[ax];
        let mut src_idx = 0;
        for (si, &off) in axis_offsets.iter().enumerate() {
            let next = if si + 1 < axis_offsets.len() {
                axis_offsets[si + 1]
            } else {
                concat_dim
            };
            if ax_coord >= off && ax_coord < next {
                src_idx = si;
                break;
            }
        }

        // Adjust coordinate for source
        let mut src_coords = out_coords.clone();
        src_coords[ax] -= axis_offsets[src_idx];

        // Read from source segment
        let src = &arrays[src_idx];
        let src_strides = dyn_row_major_strides(&src.meta.logical_shape);
        let src_flat = dyn_encode_coords(&src_coords, &src_strides);
        let src_linear = src.meta.logical_offset + src_flat;
        let addr = b.ir_constant_int(src_linear as i64);
        let elem_val = b.ir_read_memory(src.segment_id, &addr);
        out_elements.push(value_to_scalar_i64(&elem_val));
    }

    let _ = ndim;
    let _ = out_numel;
    let segment_id = crate::helpers::segment::alloc_and_write(b, &out_elements, dtype);
    let envelope = crate::types::Envelope::from_static_shape(&mut b.dim_table, &out_shape);
    let result = DynamicNDArrayData {
        envelope,
        dtype,
        segment_id,
        meta: DynArrayMeta {
            logical_shape: out_shape.clone(),
            logical_offset: 0,
            logical_strides: out_strides.clone(),
            runtime_length: ScalarValue::new(Some(out_numel as i64), None),
            runtime_rank: ScalarValue::new(Some(out_shape.len() as i64), None),
            runtime_shape: out_shape
                .iter()
                .map(|&s| ScalarValue::new(Some(s as i64), None))
                .collect(),
            runtime_strides: out_strides
                .iter()
                .map(|&s| ScalarValue::new(Some(s as i64), None))
                .collect(),
            runtime_offset: ScalarValue::new(Some(0), None),
        },
    };
    Value::DynamicNDArray(result)
}

/// DynamicNDArray.stack(arrays, axis=0)
pub fn dyn_stack(
    b: &mut IRBuilder,
    args: &[Value],
    kwargs: &HashMap<String, Value>,
) -> Value {
    let arrays_val = args
        .first()
        .expect("stack: requires arrays argument");
    let arrays: Vec<DynamicNDArrayData> = match arrays_val {
        Value::List(cd) | Value::Tuple(cd) => cd
            .values
            .iter()
            .map(|v| match v {
                Value::DynamicNDArray(d) => d.clone(),
                _ => panic!("stack: all elements must be DynamicNDArray"),
            })
            .collect(),
        _ => panic!("stack: first arg must be list/tuple of arrays"),
    };
    if arrays.is_empty() {
        return Value::None;
    }

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
    assert!(ax <= ndim, "stack: axis out of bounds");

    // Output shape: insert num_arrays at axis position
    let num_arrays = arrays.len();
    let mut out_shape = base_shape.clone();
    out_shape.insert(ax, num_arrays);
    let out_numel: usize = out_shape.iter().product();
    let out_strides = dyn_row_major_strides(&out_shape);
    let out_ndim = out_shape.len();

    let dtype = arrays[0].dtype;
    let mut out_elements = Vec::with_capacity(out_numel);

    for i in 0..out_numel {
        let out_coords = dyn_decode_coords(i, &out_shape, &out_strides);
        // The axis coordinate selects which source array
        let src_idx = out_coords[ax];
        // Remove axis coordinate to get source coordinates
        let mut src_coords: Vec<usize> = out_coords.clone();
        src_coords.remove(ax);

        let src = &arrays[src_idx];
        let src_strides = dyn_row_major_strides(&src.meta.logical_shape);
        let src_linear = src.meta.logical_offset + dyn_encode_coords(&src_coords, &src_strides);
        let addr = b.ir_constant_int(src_linear as i64);
        let elem_val = b.ir_read_memory(src.segment_id, &addr);
        out_elements.push(value_to_scalar_i64(&elem_val));
    }

    let _ = out_numel;
    let _ = out_ndim;
    let segment_id = crate::helpers::segment::alloc_and_write(b, &out_elements, dtype);
    let envelope = crate::types::Envelope::from_static_shape(&mut b.dim_table, &out_shape);
    let result = DynamicNDArrayData {
        envelope,
        dtype,
        segment_id,
        meta: DynArrayMeta {
            logical_shape: out_shape.clone(),
            logical_offset: 0,
            logical_strides: out_strides.clone(),
            runtime_length: ScalarValue::new(Some(out_numel as i64), None),
            runtime_rank: ScalarValue::new(Some(out_shape.len() as i64), None),
            runtime_shape: out_shape
                .iter()
                .map(|&s| ScalarValue::new(Some(s as i64), None))
                .collect(),
            runtime_strides: out_strides
                .iter()
                .map(|&s| ScalarValue::new(Some(s as i64), None))
                .collect(),
            runtime_offset: ScalarValue::new(Some(0), None),
        },
    };
    Value::DynamicNDArray(result)
}
