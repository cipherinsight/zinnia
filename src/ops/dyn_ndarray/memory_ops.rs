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
    let elements = super::metadata::dyn_elements_to_values(data);

    // Get mask elements
    let mask_elements: Vec<Value> = match mask {
        Value::DynamicNDArray(md) => super::metadata::dyn_elements_to_values(md),
        Value::List(cd) | Value::Tuple(cd) => cd.values.clone(),
        _ => panic!("filter: mask must be array-like"),
    };

    let max_len = data.max_length;

    // Build output via compaction with write pointer
    let mut write_ptr = b.ir_constant_int(0);
    let mut out_values: Vec<Value> = (0..max_len)
        .map(|_| super::metadata::dyn_default_value(b, data.dtype))
        .collect();

    for i in 0..max_len.min(elements.len()) {
        // Get mask value (or default false)
        let mask_val = if i < mask_elements.len() {
            mask_elements[i].clone()
        } else {
            b.ir_constant_int(0)
        };
        // Check if mask is truthy
        let keep = b.ir_bool_cast(&mask_val);

        // Conditionally place element at write_ptr position
        for j in 0..max_len {
            let j_const = b.ir_constant_int(j as i64);
            let is_target = b.ir_equal_i(&write_ptr, &j_const);
            let should_write = b.ir_logical_and(&keep, &is_target);
            out_values[j] = if data.dtype == NumberType::Float {
                b.ir_select_f(&should_write, &elements[i], &out_values[j])
            } else {
                b.ir_select_i(&should_write, &elements[i], &out_values[j])
            };
        }

        // Increment write_ptr if keep
        let one = b.ir_constant_int(1);
        let zero = b.ir_constant_int(0);
        let inc = b.ir_select_i(&keep, &one, &zero);
        write_ptr = b.ir_add_i(&write_ptr, &inc);
    }

    let new_elements: Vec<ScalarValue<i64>> =
        out_values.iter().map(|v| value_to_scalar_i64(v)).collect();

    Value::DynamicNDArray(DynamicNDArrayData {
        max_length: max_len,
        max_rank: 1,
        dtype: data.dtype,
        elements: new_elements,
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
    })
}

/// DynamicNDArray.repeat(repeats, axis=...)
pub fn dyn_repeat(
    _b: &mut IRBuilder,
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

    let flat = super::metadata::dyn_flatten_values(data);
    let numel = dyn_num_elements(&data.meta.logical_shape);

    if let Some(_ax) = axis {
        // For axis-specific repeat, flatten first, repeat, return 1D
        // (full axis support would need coordinate transforms)
        let mut new_elements = Vec::new();
        for elem in flat.iter().take(numel) {
            for _ in 0..repeats {
                new_elements.push(elem.clone());
            }
        }
        let new_len = new_elements.len();
        Value::DynamicNDArray(DynamicNDArrayData {
            max_length: new_len,
            max_rank: 1,
            dtype: data.dtype,
            elements: new_elements,
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
        })
    } else {
        // No axis: flatten then repeat each element
        let mut new_elements = Vec::new();
        for elem in flat.iter().take(numel) {
            for _ in 0..repeats {
                new_elements.push(elem.clone());
            }
        }
        let new_len = new_elements.len();
        Value::DynamicNDArray(DynamicNDArrayData {
            max_length: new_len,
            max_rank: 1,
            dtype: data.dtype,
            elements: new_elements,
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
        })
    }
}

/// DynamicNDArray.concatenate(arrays, axis=0)
pub fn dyn_concatenate(
    _b: &mut IRBuilder,
    args: &[Value],
    kwargs: &HashMap<String, Value>,
) -> Value {
    let arrays_val = args
        .first()
        .expect("concatenate: requires arrays argument");
    let arrays: Vec<&DynamicNDArrayData> = match arrays_val {
        Value::List(cd) | Value::Tuple(cd) => cd
            .values
            .iter()
            .map(|v| match v {
                Value::DynamicNDArray(d) => d,
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

        // Read from source
        let src = arrays[src_idx];
        let src_strides = dyn_row_major_strides(&src.meta.logical_shape);
        let src_flat = dyn_encode_coords(&src_coords, &src_strides);
        let src_linear = src.meta.logical_offset + src_flat;

        let elem = if src_linear < src.elements.len() {
            src.elements[src_linear].clone()
        } else {
            ScalarValue::new(Some(0), None)
        };
        out_elements.push(elem);
    }

    Value::DynamicNDArray(DynamicNDArrayData {
        max_length: out_numel,
        max_rank: ndim,
        dtype,
        elements: out_elements,
        meta: DynArrayMeta {
            logical_shape: out_shape.clone(),
            logical_offset: 0,
            logical_strides: out_strides.clone(),
            runtime_length: ScalarValue::new(Some(out_numel as i64), None),
            runtime_rank: ScalarValue::new(Some(ndim as i64), None),
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
    })
}

/// DynamicNDArray.stack(arrays, axis=0)
pub fn dyn_stack(
    _b: &mut IRBuilder,
    args: &[Value],
    kwargs: &HashMap<String, Value>,
) -> Value {
    let arrays_val = args
        .first()
        .expect("stack: requires arrays argument");
    let arrays: Vec<&DynamicNDArrayData> = match arrays_val {
        Value::List(cd) | Value::Tuple(cd) => cd
            .values
            .iter()
            .map(|v| match v {
                Value::DynamicNDArray(d) => d,
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

        let src = arrays[src_idx];
        let src_strides = dyn_row_major_strides(&src.meta.logical_shape);
        let src_linear = src.meta.logical_offset + dyn_encode_coords(&src_coords, &src_strides);

        let elem = if src_linear < src.elements.len() {
            src.elements[src_linear].clone()
        } else {
            ScalarValue::new(Some(0), None)
        };
        out_elements.push(elem);
    }

    Value::DynamicNDArray(DynamicNDArrayData {
        max_length: out_numel,
        max_rank: out_ndim,
        dtype,
        elements: out_elements,
        meta: DynArrayMeta {
            logical_shape: out_shape.clone(),
            logical_offset: 0,
            logical_strides: out_strides.clone(),
            runtime_length: ScalarValue::new(Some(out_numel as i64), None),
            runtime_rank: ScalarValue::new(Some(out_ndim as i64), None),
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
    })
}
