use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;

use super::ShapeDim;

/// Discriminant for numeric element types (Integer, Float, or Complex).
///
/// `Complex` is a logical dtype: a complex array keeps its real and
/// imaginary parts (both fixed-point Floats) in two parallel segments
/// (`Value::StaticArray::imag_segment_id`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NumberType {
    Integer,
    Float,
    Complex,
}

impl fmt::Display for NumberType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NumberType::Integer => write!(f, "Integer"),
            NumberType::Float => write!(f, "Float"),
            NumberType::Complex => write!(f, "Complex"),
        }
    }
}

/// The unified Zinnia type enum. The enum discriminant IS the type — this
/// eliminates the DTDescriptor class hierarchy entirely.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ZinniaType {
    Integer,
    Float,
    Boolean,
    /// Complex scalar. Lowered at the IR layer as a 2-tuple of Float
    /// (real, imag). The type tag carries the metadata that the underlying
    /// Tuple([Float, Float]) is a complex number, so `apply_binary_op` and
    /// other paths can dispatch correctly.
    Complex,
    String,
    None,
    Class,
    NDArray {
        shape: Vec<usize>,
        dtype: NumberType,
    },
    SymbolicNDArray {
        shape: Vec<ShapeDim>,
        dtype: NumberType,
        #[serde(default)]
        bindings: BTreeMap<String, usize>,
    },
    DynamicNDArray {
        dtype: NumberType,
        max_length: usize,
        max_rank: usize,
        #[serde(default)]
        max_shape: Option<Vec<usize>>,
    },
    List {
        elements: Vec<ZinniaType>,
    },
    Tuple {
        elements: Vec<ZinniaType>,
    },
    PoseidonHashed {
        dtype: Box<ZinniaType>,
    },
}

impl fmt::Display for ZinniaType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ZinniaType::Integer => write!(f, "Integer"),
            ZinniaType::Float => write!(f, "Float"),
            ZinniaType::Boolean => write!(f, "Boolean"),
            ZinniaType::Complex => write!(f, "Complex"),
            ZinniaType::String => write!(f, "String"),
            ZinniaType::None => write!(f, "None"),
            ZinniaType::Class => write!(f, "Class"),
            ZinniaType::NDArray { shape, dtype } => {
                let dims: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
                write!(f, "NDArray[{}, {}]", dtype, dims.join(", "))
            }
            ZinniaType::SymbolicNDArray { shape, dtype, .. } => {
                let dims: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
                write!(f, "SymbolicNDArray[{}, {}]", dtype, dims.join(", "))
            }
            ZinniaType::DynamicNDArray {
                dtype,
                max_length,
                max_rank,
                ..
            } => write!(f, "DynamicNDArray[{}, {}, {}]", dtype, max_length, max_rank),
            ZinniaType::List { elements } => {
                let elems: Vec<String> = elements.iter().map(|e| e.to_string()).collect();
                write!(f, "List[{}]", elems.join(", "))
            }
            ZinniaType::Tuple { elements } => {
                let elems: Vec<String> = elements.iter().map(|e| e.to_string()).collect();
                write!(f, "Tuple[{}]", elems.join(", "))
            }
            ZinniaType::PoseidonHashed { dtype } => write!(f, "PoseidonHashed[{}]", dtype),
        }
    }
}

impl ZinniaType {
    /// Returns true if this type is a numeric type (Integer, Float, Boolean, or Complex).
    pub fn is_number(&self) -> bool {
        matches!(
            self,
            ZinniaType::Integer | ZinniaType::Float | ZinniaType::Boolean | ZinniaType::Complex
        )
    }

    /// Returns true if this type is an integer-like type (Integer or Boolean).
    pub fn is_integer(&self) -> bool {
        matches!(self, ZinniaType::Integer | ZinniaType::Boolean)
    }

    /// Returns true if this type is an NDArray or DynamicNDArray.
    pub fn is_ndarray(&self) -> bool {
        matches!(
            self,
            ZinniaType::NDArray { .. }
                | ZinniaType::SymbolicNDArray { .. }
                | ZinniaType::DynamicNDArray { .. }
        )
    }

    /// Returns the `NumberType` for numeric types.
    pub fn number_type(&self) -> Option<NumberType> {
        match self {
            ZinniaType::Integer | ZinniaType::Boolean => Some(NumberType::Integer),
            ZinniaType::Float => Some(NumberType::Float),
            _ => Option::None,
        }
    }

    /// Check whether this type can be implicitly cast to `target`.
    /// Mirrors the Python DTDescriptor casting rules:
    /// - Boolean -> Integer (Boolean is a subtype of Integer)
    /// - Same type -> same type
    pub fn can_implicit_cast_to(&self, target: &ZinniaType) -> bool {
        if self == target {
            return true;
        }
        // Boolean can implicitly cast to Integer (BooleanDTDescriptor extends IntegerDTDescriptor)
        if *self == ZinniaType::Boolean && *target == ZinniaType::Integer {
            return true;
        }
        false
    }

    /// For NDArray types, returns the total number of elements.
    pub fn num_elements(&self) -> Option<usize> {
        match self {
            ZinniaType::NDArray { shape, .. } => Some(shape.iter().product()),
            ZinniaType::SymbolicNDArray {
                shape, bindings, ..
            } => super::concretize_shape_dims(shape, bindings)
                .ok()
                .map(|shape| shape.iter().product())
                .or_else(|| super::bounded_shape_total(shape, bindings).ok()),
            ZinniaType::DynamicNDArray { max_length, .. } => Some(*max_length),
            _ => Option::None,
        }
    }
}
