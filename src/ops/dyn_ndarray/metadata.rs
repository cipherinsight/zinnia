use crate::builder::IRBuilder;
use crate::types::{
    CompositeData, DynArrayMeta, DynamicNDArrayData, NumberType, ScalarValue, Value, ZinniaType,
};

use super::{
    dyn_decode_coords, dyn_encode_coords, dyn_num_elements, dyn_row_major_strides,
    scalar_i64_to_value, value_to_scalar_i64,
};

use std::collections::HashMap;

// ── Phase 1: utility helpers ──

/// Create a default value for the given dtype.
pub fn dyn_default_value(b: &mut IRBuilder, dtype: NumberType) -> Value {
    match dtype {
        NumberType::Integer => b.ir_constant_int(0),
        NumberType::Float => b.ir_constant_float(0.0),
    }
}

/// Flatten a DynamicNDArray's elements respecting view metadata (offset, strides, shape).
pub fn dyn_flatten_values(data: &DynamicNDArrayData) -> Vec<ScalarValue<i64>> {
    let shape = &data.meta.logical_shape;
    let strides = &data.meta.logical_strides;
    let offset = data.meta.logical_offset;
    let numel = dyn_num_elements(shape);

    if numel == 0 {
        return vec![];
    }

    let row_strides = dyn_row_major_strides(shape);
    let mut result = Vec::with_capacity(numel);
    for i in 0..numel {
        let coords = dyn_decode_coords(i, shape, &row_strides);
        let src_idx = offset + dyn_encode_coords(&coords, strides);
        if src_idx < data.elements.len() {
            result.push(data.elements[src_idx].clone());
        } else {
            result.push(ScalarValue::new(Some(0), None));
        }
    }
    result
}

/// Convert a DynamicNDArray's flat elements to Vec<Value>.
pub fn dyn_elements_to_values(data: &DynamicNDArrayData) -> Vec<Value> {
    let flat = dyn_flatten_values(data);
    flat.iter()
        .map(|elem| scalar_i64_to_value(elem, data.dtype))
        .collect()
}

// ── Phase 2: pure metadata ops ────────────────────────────────────

pub fn dyn_ndim(b: &mut IRBuilder, data: &DynamicNDArrayData) -> Value {
    b.ir_constant_int(data.meta.logical_shape.len() as i64)
}

pub fn dyn_dtype(data: &DynamicNDArrayData) -> Value {
    match data.dtype {
        NumberType::Integer => Value::Class(ZinniaType::Integer),
        NumberType::Float => Value::Class(ZinniaType::Float),
    }
}

pub fn dyn_shape(b: &mut IRBuilder, data: &DynamicNDArrayData) -> Value {
    let shape = &data.meta.logical_shape;
    if shape.len() == 1 {
        // 1D: use runtime_length if available (dynamic), else constant
        let len_val = if let Some(v) = data.meta.runtime_length.static_val {
            b.ir_constant_int(v)
        } else if let Some(ptr) = data.meta.runtime_length.ptr {
            Value::Integer(ScalarValue::new(None, Some(ptr)))
        } else {
            b.ir_constant_int(shape[0] as i64)
        };
        let types = vec![ZinniaType::Integer];
        Value::Tuple(CompositeData {
            elements_type: types,
            values: vec![len_val],
        })
    } else {
        let vals: Vec<Value> = shape
            .iter()
            .map(|&s| Value::Integer(ScalarValue::new(Some(s as i64), None)))
            .collect();
        let types = vec![ZinniaType::Integer; vals.len()];
        Value::Tuple(CompositeData {
            elements_type: types,
            values: vals,
        })
    }
}

pub fn dyn_size(b: &mut IRBuilder, data: &DynamicNDArrayData) -> Value {
    let shape = &data.meta.logical_shape;
    if shape.len() == 1 {
        // 1D: return runtime_length if dynamic
        if let Some(v) = data.meta.runtime_length.static_val {
            b.ir_constant_int(v)
        } else if let Some(ptr) = data.meta.runtime_length.ptr {
            Value::Integer(ScalarValue::new(None, Some(ptr)))
        } else {
            b.ir_constant_int(shape[0] as i64)
        }
    } else {
        let total: usize = shape.iter().product();
        b.ir_constant_int(total as i64)
    }
}

// ── Phase 2: simple value ops ─────────────────────────────────────

pub fn dyn_astype(b: &mut IRBuilder, data: &DynamicNDArrayData, args: &[Value]) -> Value {
    let target_float = matches!(args.first(), Some(Value::Class(ZinniaType::Float)));
    let new_dtype = if target_float {
        NumberType::Float
    } else {
        NumberType::Integer
    };

    // Cast each element
    let flat = dyn_flatten_values(data);
    let new_elements: Vec<ScalarValue<i64>> = flat
        .iter()
        .map(|elem| {
            let val = scalar_i64_to_value(elem, data.dtype);
            let cast_val = if target_float {
                b.ir_float_cast(&val)
            } else {
                b.ir_int_cast(&val)
            };
            value_to_scalar_i64(&cast_val)
        })
        .collect();

    Value::DynamicNDArray(DynamicNDArrayData {
        envelope: data.envelope.clone(),
        dtype: new_dtype,
        elements: new_elements,
        meta: data.meta.clone(),
    })
}

pub fn dyn_flatten_to_list(data: &DynamicNDArrayData) -> Value {
    let values = dyn_elements_to_values(data);
    let types = values.iter().map(|v| v.zinnia_type()).collect();
    Value::List(CompositeData {
        elements_type: types,
        values,
    })
}

pub fn dyn_flat(b: &mut IRBuilder, data: &DynamicNDArrayData) -> Value {
    let flat = dyn_flatten_values(data);
    let max_length = data.max_length();
    // Flatten to a single dim of the same total max bound. Always Static
    // here because flatten loses any min-bound information from the
    // individual axes (we'd need a full intersection to recover it).
    let envelope = crate::types::Envelope::from_static_shape(&mut b.dim_table, &[max_length]);
    Value::DynamicNDArray(DynamicNDArrayData {
        envelope,
        dtype: data.dtype,
        elements: flat,
        meta: DynArrayMeta {
            logical_shape: vec![max_length],
            logical_offset: 0,
            logical_strides: vec![1],
            runtime_length: data.meta.runtime_length.clone(),
            runtime_rank: ScalarValue::new(Some(1), None),
            runtime_shape: vec![data.meta.runtime_length.clone()],
            runtime_strides: vec![ScalarValue::new(Some(1), None)],
            runtime_offset: ScalarValue::new(Some(0), None),
        },
    })
}

pub fn dyn_tolist(data: &DynamicNDArrayData) -> Value {
    let flat_values = dyn_elements_to_values(data);
    let shape = &data.meta.logical_shape;
    build_nested_list(&flat_values, shape)
}

/// Build nested Value::List from flat values and shape.
pub fn build_nested_list(flat: &[Value], shape: &[usize]) -> Value {
    if shape.len() <= 1 {
        let len = shape.first().copied().unwrap_or(flat.len());
        let vals: Vec<Value> = flat.iter().take(len).cloned().collect();
        let types = vals.iter().map(|v| v.zinnia_type()).collect();
        return Value::List(CompositeData {
            elements_type: types,
            values: vals,
        });
    }
    let inner_size: usize = shape[1..].iter().product();
    let mut rows = Vec::new();
    for chunk in flat.chunks(inner_size) {
        rows.push(build_nested_list(chunk, &shape[1..]));
    }
    let row_types = rows.iter().map(|v| v.zinnia_type()).collect();
    Value::List(CompositeData {
        elements_type: row_types,
        values: rows,
    })
}

// ── Phase 2: constructors (zeros, ones) ──────────────────────

/// DynamicNDArray.zeros(shape, dtype=...)
pub fn dyn_zeros(b: &mut IRBuilder, args: &[Value], kwargs: &HashMap<String, Value>) -> Value {
    super::constructors::dyn_fill(b, args, kwargs, 0)
}

/// DynamicNDArray.ones(shape, dtype=...)
pub fn dyn_ones(b: &mut IRBuilder, args: &[Value], kwargs: &HashMap<String, Value>) -> Value {
    super::constructors::dyn_fill(b, args, kwargs, 1)
}
