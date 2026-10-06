use crate::builder::IRBuilder;
use crate::types::{
    CompositeData, DynArrayMeta, DynamicNDArrayData, NumberType, ScalarValue, Value, ValueId,
    ZinniaType,
};

use std::collections::HashMap;

// ── Phase 1: utility helpers ──

/// Create a default value for the given dtype.
pub fn dyn_default_value(b: &mut IRBuilder, dtype: NumberType) -> Value {
    match dtype {
        NumberType::Integer => b.ir_constant_int(0),
        NumberType::Float => b.ir_constant_float(0.0),
        NumberType::Complex => Value::Complex {
            real: ScalarValue::new(Some(0.0), None),
            imag: ScalarValue::new(Some(0.0), None),
        },
    }
}

// ── Phase 2: pure metadata ops ────────────────────────────────────

pub fn dyn_ndim(b: &mut IRBuilder, data: &DynamicNDArrayData) -> Value {
    b.ir_constant_int(data.meta.logical_shape.len() as i64)
}

pub fn dyn_dtype(b: &IRBuilder, data: &DynamicNDArrayData) -> Value {
    match data.dtype {
        NumberType::Integer if b.is_bool_segment(data.segment_id) => Value::Class(ZinniaType::Boolean),
        NumberType::Integer => Value::Class(ZinniaType::Integer),
        NumberType::Float => Value::Class(ZinniaType::Float),
        NumberType::Complex => Value::Class(ZinniaType::Complex),
    }
}

pub fn dyn_shape(b: &mut IRBuilder, data: &DynamicNDArrayData) -> Value {
    let vals: Vec<Value> = data
        .meta
        .runtime_shape
        .iter()
        .enumerate()
        .map(|(axis, dim)| {
            if let Some(v) = dim.static_val {
                b.ir_constant_int(v)
            } else if let Some(ptr) = dim.stmt_id {
                Value::Integer(ScalarValue::new(None, Some(ptr)))
            } else {
                b.ir_constant_int(data.meta.logical_shape[axis] as i64)
            }
        })
        .collect();
    let types = vec![ZinniaType::Integer; vals.len()];
    Value::Tuple(CompositeData {
        elements_type: types,
        values: vals,

        value_id: ValueId::next(),
    })
}

pub fn dyn_size(b: &mut IRBuilder, data: &DynamicNDArrayData) -> Value {
    if let Some(v) = data.meta.runtime_length.static_val {
        b.ir_constant_int(v)
    } else if let Some(ptr) = data.meta.runtime_length.stmt_id {
        Value::Integer(ScalarValue::new(None, Some(ptr)))
    } else {
        b.ir_constant_int(data.meta.logical_shape.iter().product::<usize>() as i64)
    }
}

// ── Phase 2: simple value ops ─────────────────────────────────────

pub fn dyn_astype(b: &mut IRBuilder, data: &DynamicNDArrayData, args: &[Value]) -> Value {
    let to_bool = matches!(args.first(), Some(Value::Class(ZinniaType::Boolean)));
    let new_dtype = match args.first() {
        Some(Value::Class(ZinniaType::Complex)) => NumberType::Complex,
        Some(Value::Class(ZinniaType::Float)) => NumberType::Float,
        _ => NumberType::Integer,
    };

    let values =
        crate::helpers::segment::read_all_typed(b, data.segment_id, data.max_length(), data.dtype);
    let new_values: Vec<Value> = values
        .iter()
        .map(|val| match new_dtype {
            NumberType::Complex => {
                let (real, imag) = crate::helpers::value_ops::unpack_value_to_complex_parts(b, val);
                crate::helpers::value_ops::pack_complex_value(real, imag)
            }
            NumberType::Float => b.ir_float_cast(&val),
            // `astype(bool)` is each element's truth value (NumPy: `x != 0`);
            // `astype(int)` of a bool array has integer cells.
            NumberType::Integer if to_bool => b.bool_dtype_cell(val),
            NumberType::Integer if matches!(val, Value::Integer(_) | Value::Boolean(_)) => {
                b.int_dtype_cell(val)
            }
            NumberType::Integer => b.ir_int_cast(val),
        })
        .collect();

    let segment_id = crate::helpers::segment::alloc_and_write_values(b, &new_values, new_dtype);
    let result = DynamicNDArrayData {
        envelope: data.envelope.clone(),
        dtype: new_dtype,
        segment_id,
        meta: data.meta.clone(),
        value_id: ValueId::next(),
    };
    Value::DynamicNDArray(result)
}

pub fn dyn_flatten_to_list(b: &mut IRBuilder, data: &DynamicNDArrayData) -> Value {
    reject_if_runtime_shape_may_be_shorter(data, "flatten");
    let values =
        crate::helpers::segment::read_all_typed(b, data.segment_id, data.max_length(), data.dtype);
    let types = values.iter().map(|v| v.zinnia_type()).collect();
    Value::List(CompositeData {
        elements_type: types,
        values,

        value_id: ValueId::next(),
    })
}

pub fn dyn_flat(b: &mut IRBuilder, data: &DynamicNDArrayData) -> Value {
    let values =
        crate::helpers::segment::read_all_typed(b, data.segment_id, data.max_length(), data.dtype);
    let max_length = data.max_length();
    let segment_id = crate::helpers::segment::alloc_and_write_values(b, &values, data.dtype);
    // Flatten to a single dim of the same total max bound. Always Static
    // here because flatten loses any min-bound information from the
    // individual axes (we'd need a full intersection to recover it).
    let envelope = crate::types::Envelope::from_static_shape(&mut b.dim_table, &[max_length]);
    let result = DynamicNDArrayData {
        envelope,
        dtype: data.dtype,
        segment_id,
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
        value_id: ValueId::next(),
    };
    Value::DynamicNDArray(result)
}

pub fn dyn_tolist(b: &mut IRBuilder, data: &DynamicNDArrayData) -> Value {
    reject_if_runtime_shape_may_be_shorter(data, "tolist");
    let flat_values =
        crate::helpers::segment::read_all_typed(b, data.segment_id, data.max_length(), data.dtype);
    let shape = &data.meta.logical_shape.clone();
    build_nested_list(&flat_values, shape)
}

pub fn dyn_complex_real(b: &mut IRBuilder, data: &DynamicNDArrayData) -> Value {
    if data.dtype != NumberType::Complex {
        return Value::DynamicNDArray(data.clone());
    }
    let values =
        crate::helpers::segment::read_all_typed(b, data.segment_id, data.max_length(), data.dtype);
    let real_values: Vec<Value> = values
        .into_iter()
        .map(|v| match v {
            Value::Complex { real, .. } => Value::Float(real),
            _ => unreachable!("Complex DynamicNDArray read must yield Complex values"),
        })
        .collect();
    let segment_id =
        crate::helpers::segment::alloc_and_write_values(b, &real_values, NumberType::Float);
    Value::DynamicNDArray(DynamicNDArrayData {
        envelope: data.envelope.clone(),
        dtype: NumberType::Float,
        segment_id,
        meta: data.meta.clone(),
        value_id: ValueId::next(),
    })
}

pub fn dyn_complex_imag(b: &mut IRBuilder, data: &DynamicNDArrayData) -> Value {
    if data.dtype != NumberType::Complex {
        let zeros = vec![b.ir_constant_float(0.0); data.max_length()];
        let segment_id =
            crate::helpers::segment::alloc_and_write_values(b, &zeros, NumberType::Float);
        return Value::DynamicNDArray(DynamicNDArrayData {
            envelope: data.envelope.clone(),
            dtype: NumberType::Float,
            segment_id,
            meta: data.meta.clone(),
            value_id: ValueId::next(),
        });
    }
    let values =
        crate::helpers::segment::read_all_typed(b, data.segment_id, data.max_length(), data.dtype);
    let imag_values: Vec<Value> = values
        .into_iter()
        .map(|v| match v {
            Value::Complex { imag, .. } => Value::Float(imag),
            _ => unreachable!("Complex DynamicNDArray read must yield Complex values"),
        })
        .collect();
    let segment_id =
        crate::helpers::segment::alloc_and_write_values(b, &imag_values, NumberType::Float);
    Value::DynamicNDArray(DynamicNDArrayData {
        envelope: data.envelope.clone(),
        dtype: NumberType::Float,
        segment_id,
        meta: data.meta.clone(),
        value_id: ValueId::next(),
    })
}

pub fn dyn_complex_conjugate(b: &mut IRBuilder, data: &DynamicNDArrayData) -> Value {
    if data.dtype != NumberType::Complex {
        return Value::DynamicNDArray(data.clone());
    }
    let values =
        crate::helpers::segment::read_all_typed(b, data.segment_id, data.max_length(), data.dtype);
    let zero = b.ir_constant_float(0.0);
    let conj_values: Vec<Value> = values
        .into_iter()
        .map(|v| match v {
            Value::Complex { real, imag } => {
                let neg_imag = b.ir_sub_f(&zero, &Value::Float(imag));
                crate::helpers::value_ops::pack_complex_value(Value::Float(real), neg_imag)
            }
            _ => unreachable!("Complex DynamicNDArray read must yield Complex values"),
        })
        .collect();
    let segment_id =
        crate::helpers::segment::alloc_and_write_values(b, &conj_values, NumberType::Complex);
    Value::DynamicNDArray(DynamicNDArrayData {
        envelope: data.envelope.clone(),
        dtype: NumberType::Complex,
        segment_id,
        meta: data.meta.clone(),
        value_id: ValueId::next(),
    })
}

fn reject_if_runtime_shape_may_be_shorter(data: &DynamicNDArrayData, method: &str) {
    let runtime_may_be_shorter = data
        .meta
        .runtime_shape
        .iter()
        .zip(&data.meta.logical_shape)
        .any(|(runtime, logical)| runtime.static_val != Some(*logical as i64));
    if runtime_may_be_shorter {
        panic!(
            "DynamicNDArray.{} is not yet supported for bounded-dynamic arrays because it would expose max-buffer padding; use runtime-shape-aware array operations instead",
            method
        );
    }
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

            value_id: ValueId::next(),
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

        value_id: ValueId::next(),
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
