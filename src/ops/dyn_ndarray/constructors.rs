use std::collections::HashMap;

use crate::builder::IRBuilder;
use crate::types::{
    DynArrayMeta, DynamicNDArrayData, NumberType, ScalarValue, Value, ZinniaType,
};

use super::{dyn_row_major_strides, value_to_scalar_i64};

pub fn dyn_fill(
    b: &mut IRBuilder,
    args: &[Value],
    kwargs: &HashMap<String, Value>,
    fill_value: i64,
) -> Value {
    let shape = parse_shape_arg(args.first().expect("zeros/ones: requires shape arg"));
    let dtype = parse_dtype_kwarg(kwargs);
    let max_length: usize = shape.iter().product();
    let max_rank = shape.len();

    let fill_sv = match dtype {
        NumberType::Integer => {
            let v = b.ir_constant_int(fill_value);
            value_to_scalar_i64(&v)
        }
        NumberType::Float => {
            let v = b.ir_constant_float(fill_value as f64);
            value_to_scalar_i64(&v)
        }
    };
    let elements = vec![fill_sv; max_length];

    let strides = dyn_row_major_strides(&shape);
    let _ = max_rank;
    let envelope = crate::types::Envelope::from_static_shape(&mut b.dim_table, &shape);
    let mut result = DynamicNDArrayData {
        envelope,
        dtype,
        elements,
        segment_id: None,
        meta: DynArrayMeta {
            logical_shape: shape.clone(),
            logical_offset: 0,
            logical_strides: strides,
            runtime_length: ScalarValue::new(Some(max_length as i64), None),
            runtime_rank: ScalarValue::new(Some(shape.len() as i64), None),
            runtime_shape: shape
                .iter()
                .map(|&s| ScalarValue::new(Some(s as i64), None))
                .collect(),
            runtime_strides: dyn_row_major_strides(&shape)
                .iter()
                .map(|&s| ScalarValue::new(Some(s as i64), None))
                .collect(),
            runtime_offset: ScalarValue::new(Some(0), None),
        },
    };
    crate::helpers::segment::ensure_segment(b, &mut result);
    Value::DynamicNDArray(result)
}

/// DynamicNDArray.eye(N, M=None, dtype=...)
pub fn dyn_eye(b: &mut IRBuilder, args: &[Value], kwargs: &HashMap<String, Value>) -> Value {
    let n = args
        .first()
        .and_then(|v| v.int_val())
        .expect("eye: N must be constant int") as usize;
    let m = args
        .get(1)
        .or_else(|| kwargs.get("M"))
        .and_then(|v| v.int_val())
        .unwrap_or(n as i64) as usize;
    let dtype = parse_dtype_kwarg(kwargs);

    let max_length = n * m;
    let shape = vec![n, m];
    let strides = dyn_row_major_strides(&shape);

    let zero = match dtype {
        NumberType::Integer => {
            let v = b.ir_constant_int(0);
            value_to_scalar_i64(&v)
        }
        NumberType::Float => {
            let v = b.ir_constant_float(0.0);
            value_to_scalar_i64(&v)
        }
    };
    let one = match dtype {
        NumberType::Integer => {
            let v = b.ir_constant_int(1);
            value_to_scalar_i64(&v)
        }
        NumberType::Float => {
            let v = b.ir_constant_float(1.0);
            value_to_scalar_i64(&v)
        }
    };

    let mut elements = Vec::with_capacity(max_length);
    for i in 0..n {
        for j in 0..m {
            elements.push(if i == j { one.clone() } else { zero.clone() });
        }
    }

    let envelope = crate::types::Envelope::from_static_shape(&mut b.dim_table, &shape);
    let mut result = DynamicNDArrayData {
        envelope,
        dtype,
        elements,
        segment_id: None,
        meta: DynArrayMeta {
            logical_shape: shape.clone(),
            logical_offset: 0,
            logical_strides: strides,
            runtime_length: ScalarValue::new(Some(max_length as i64), None),
            runtime_rank: ScalarValue::new(Some(2), None),
            runtime_shape: shape
                .iter()
                .map(|&s| ScalarValue::new(Some(s as i64), None))
                .collect(),
            runtime_strides: dyn_row_major_strides(&shape)
                .iter()
                .map(|&s| ScalarValue::new(Some(s as i64), None))
                .collect(),
            runtime_offset: ScalarValue::new(Some(0), None),
        },
    };
    crate::helpers::segment::ensure_segment(b, &mut result);
    Value::DynamicNDArray(result)
}

pub fn parse_shape_arg(val: &Value) -> Vec<usize> {
    match val {
        Value::Tuple(data) | Value::List(data) => data
            .values
            .iter()
            .map(|v| v.int_val().expect("shape element must be constant int") as usize)
            .collect(),
        Value::Integer(_) => vec![val.int_val().unwrap() as usize],
        _ => panic!("shape must be tuple, list, or int"),
    }
}

pub fn parse_dtype_kwarg(kwargs: &HashMap<String, Value>) -> NumberType {
    if let Some(Value::Class(ZinniaType::Integer)) = kwargs.get("dtype") {
        NumberType::Integer
    } else if let Some(Value::Class(ZinniaType::Float)) = kwargs.get("dtype") {
        NumberType::Float
    } else {
        NumberType::Float // default to float like Python
    }
}
