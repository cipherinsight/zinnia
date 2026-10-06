//! Typed arithmetic the scatter constructions share: products of a 0/1 bit
//! and a cell, and sums, on int64 or Q32 cells.

use crate::builder::IRBuilder;
use crate::ir_defs::IR;
use crate::types::{NumberType, ScalarValue, Value};

/// `bit * value` for a 0/1 `bit`, typed as `value`'s dtype. A field product:
/// a Q32 `value` scaled by 0 or 1 needs no rescale.
pub fn bit_times(b: &mut IRBuilder, bit: &Value, value: &Value, dtype: NumberType) -> Value {
    if let Some(v) = bit.int_val() {
        return match v {
            0 => zero(b, dtype),
            _ => value.clone(),
        };
    }
    let product = b.create_ir(&IR::MulI, &[bit.clone(), value.clone()]);
    as_dtype(product, dtype)
}

/// The sum of `terms` (at least one), left to right.
pub fn sum(b: &mut IRBuilder, terms: &[Value], dtype: NumberType) -> Value {
    let (first, rest) = terms.split_first().expect("a sum has a term");
    rest.iter().fold(first.clone(), |acc, term| match dtype {
        NumberType::Float => b.ir_add_f(&acc, term),
        _ => b.ir_add_i(&acc, term),
    })
}

/// `x - y`.
pub fn difference(b: &mut IRBuilder, x: &Value, y: &Value, dtype: NumberType) -> Value {
    match dtype {
        NumberType::Float => b.ir_sub_f(x, y),
        _ => b.ir_sub_i(x, y),
    }
}

/// The constant 0 of `dtype`.
pub fn zero(b: &mut IRBuilder, dtype: NumberType) -> Value {
    match dtype {
        NumberType::Float => b.ir_constant_float(0.0),
        _ => b.ir_constant_int(0),
    }
}

/// A field-arithmetic result retyped as `dtype` (an integer product of a
/// bit and a Q32 cell is that Q32 cell scaled by the bit).
fn as_dtype(value: Value, dtype: NumberType) -> Value {
    match (value, dtype) {
        (Value::Integer(sv), NumberType::Float) => {
            Value::Float(ScalarValue { static_val: None, stmt_id: sv.stmt_id, value_id: sv.value_id })
        }
        (value, _) => value,
    }
}
