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
