//! The array arguments of the cell-level NumPy functions (`diagonals`,
//! `products`, `polynomials`, `differences`, `determinant`): an argument
//! read as its cells in row-major order with its shape and dtype.

use crate::builder::IRBuilder;
use crate::types::{NumberType, Value};

/// A real array argument with a compile-time shape.
pub(super) struct StaticOperand {
    /// The cells in row-major order (through any view's strides).
    pub cells: Vec<Value>,
    pub shape: Vec<usize>,
    pub dtype: NumberType,
    /// Whether NumPy's dtype is `bool` (`builder/bool_dtype.rs`).
    pub is_bool: bool,
}

impl StaticOperand {
    /// `value` (an ndarray, a nested numeric list or a scalar) as an
    /// operand of `op`. An array with a runtime shape and a complex array
    /// are rejected.
    pub fn read(b: &mut IRBuilder, value: &Value, op: &str) -> Self {
        match value {
            Value::DynamicNDArray(_) => crate::helpers::composite::reject_runtime_shaped_operand(),
            Value::Integer(_) | Value::Float(_) | Value::Boolean(_) => StaticOperand {
                cells: vec![value.clone()],
                shape: vec![],
                dtype: if matches!(value, Value::Float(_)) { NumberType::Float } else { NumberType::Integer },
                is_bool: b.is_bool_cell(value),
            },
            Value::Complex { .. } => panic!("{op}: complex values are not supported"),
            // `np.array([])`: NumPy's empty array is float64.
            Value::List(d) | Value::Tuple(d) if d.values.is_empty() => StaticOperand {
                cells: vec![],
                shape: vec![0],
                dtype: NumberType::Float,
                is_bool: false,
            },
            _ => {
                let array = crate::helpers::static_array::base::to_static_array(b, value)
                    .unwrap_or_else(|| panic!("{op}: expected a rectangular numeric array"));
                let Value::StaticArray { shape, dtype, .. } = &array else { unreachable!() };
                if *dtype == NumberType::Complex {
                    panic!("{op}: complex arrays are not supported");
                }
                StaticOperand {
                    cells: crate::helpers::static_array::elementwise::payload_cells(b, &array),
                    shape: shape.clone(),
                    dtype: *dtype,
                    is_bool: crate::helpers::array_ops::is_bool_array(b, &array),
                }
            }
        }
    }

    /// Like [`StaticOperand::read`], rejecting a `bool` array: NumPy's
    /// arithmetic on bool arrays is logical (`+` is `or`, `-` raises), not
    /// the integer arithmetic of the cell-level functions.
    pub fn read_numeric(b: &mut IRBuilder, value: &Value, op: &str) -> Self {
        let operand = Self::read(b, value, op);
        if operand.is_bool {
            panic!("{op}: boolean arrays are not supported (NumPy's arithmetic on them is logical)");
        }
        operand
    }
}

/// The `dtype` array of `shape` holding `cells` (row-major), integer cells
/// of a float array converted.
pub(super) fn build_array(b: &mut IRBuilder, cells: Vec<Value>, shape: Vec<usize>, dtype: NumberType) -> Value {
    let cells = match dtype {
        NumberType::Float => cells
            .iter()
            .map(|c| crate::helpers::value_ops::ensure_float(b, c))
            .collect(),
        _ => cells,
    };
    crate::helpers::static_array::base::build_static_array_from_flat(b, cells, shape, dtype)
}

/// NumPy's result dtype of an arithmetic operation on two real operands.
pub(super) fn promoted(lhs: NumberType, rhs: NumberType) -> NumberType {
    if lhs == NumberType::Float || rhs == NumberType::Float {
        NumberType::Float
    } else {
        NumberType::Integer
    }
}

/// The constant zero of `dtype`.
pub(super) fn zero(b: &mut IRBuilder, dtype: NumberType) -> Value {
    match dtype {
        NumberType::Float => b.ir_constant_float(0.0),
        _ => b.ir_constant_int(0),
    }
}

/// The constant one of `dtype`.
pub(super) fn one(b: &mut IRBuilder, dtype: NumberType) -> Value {
    match dtype {
        NumberType::Float => b.ir_constant_float(1.0),
        _ => b.ir_constant_int(1),
    }
}

/// `lhs op rhs` on two cells (`value_ops::apply_scalar_binary_op`: integer
/// arithmetic on integers, fixed-point arithmetic once a float is involved).
pub(super) fn cell_op(b: &mut IRBuilder, op: &str, lhs: &Value, rhs: &Value) -> Value {
    crate::helpers::value_ops::apply_scalar_binary_op(b, op, lhs, rhs)
}

/// The row-major position, in an array of shape `source` broadcast to
/// `target` (NumPy's rules), of the cell at row-major position `index` of
/// `target`.
pub(super) fn broadcast_source_index(index: usize, target: &[usize], source: &[usize]) -> usize {
    let pad = target.len() - source.len();
    let target_strides = crate::helpers::shape_arith::row_major_strides(target);
    let source_strides = crate::helpers::shape_arith::row_major_strides(source);
    source
        .iter()
        .enumerate()
        .map(|(d, &extent)| {
            let coordinate = index / target_strides[pad + d] % target[pad + d];
            if extent == 1 { 0 } else { coordinate * source_strides[d] }
        })
        .sum()
}
