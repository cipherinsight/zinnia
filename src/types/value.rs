use super::zinnia_type::ZinniaType;
use super::scalar::{ScalarValue, StringValue};
use super::composite::{CompositeData, NDArrayData, DynamicNDArrayData};
use super::StmtId;

/// The unified value enum representing all Zinnia values during compilation.
/// Merges the Python Value hierarchy + DTDescriptor into a single tagged union.
#[derive(Debug, Clone)]
pub enum Value {
    Integer(ScalarValue<i64>),
    Float(ScalarValue<f64>),
    Boolean(ScalarValue<bool>),
    String(StringValue),
    None,
    Class(ZinniaType),
    NDArray(NDArrayData),
    DynamicNDArray(DynamicNDArrayData),
    List(CompositeData),
    Tuple(CompositeData),
    PoseidonHashed {
        dtype: Box<ZinniaType>,
        inner: Box<Value>,
    },
}

impl Value {
    /// Returns the `ZinniaType` of this value.
    pub fn zinnia_type(&self) -> ZinniaType {
        match self {
            Value::Integer(_) => ZinniaType::Integer,
            Value::Float(_) => ZinniaType::Float,
            Value::Boolean(_) => ZinniaType::Boolean,
            Value::String(_) => ZinniaType::String,
            Value::None => ZinniaType::None,
            Value::Class(inner_type) => inner_type.clone(),
            Value::NDArray(data) => ZinniaType::NDArray {
                shape: data.shape.clone(),
                dtype: data.dtype,
            },
            Value::DynamicNDArray(data) => ZinniaType::DynamicNDArray {
                dtype: data.dtype,
                max_length: data.max_length(),
                max_rank: data.max_rank(),
            },
            Value::List(data) => ZinniaType::List {
                elements: data.elements_type.clone(),
            },
            Value::Tuple(data) => ZinniaType::Tuple {
                elements: data.elements_type.clone(),
            },
            Value::PoseidonHashed { dtype, .. } => ZinniaType::PoseidonHashed {
                dtype: dtype.clone(),
            },
        }
    }

    /// Returns the IR statement pointer for atomic values, if available.
    pub fn ptr(&self) -> Option<StmtId> {
        match self {
            Value::Integer(s) => s.ptr,
            Value::Float(s) => s.ptr,
            Value::Boolean(s) => s.ptr,
            Value::String(s) => Some(s.ptr),
            Value::None => Option::None,
            Value::Class(_) => Option::None,
            _ => Option::None,
        }
    }

    /// Returns true if this value is a numeric type.
    pub fn is_number(&self) -> bool {
        matches!(self, Value::Integer(_) | Value::Float(_) | Value::Boolean(_))
    }

    /// Returns true if this value is an integer-like type.
    pub fn is_integer(&self) -> bool {
        matches!(self, Value::Integer(_) | Value::Boolean(_))
    }

    /// Returns true if this value is an NDArray or DynamicNDArray.
    pub fn is_ndarray(&self) -> bool {
        matches!(self, Value::NDArray(_) | Value::DynamicNDArray(_))
    }

    /// Returns the compile-time integer value, if known.
    pub fn int_val(&self) -> Option<i64> {
        match self {
            Value::Integer(s) => s.static_val,
            Value::Boolean(s) => s.static_val.map(|b| if b { 1 } else { 0 }),
            _ => Option::None,
        }
    }

    /// Returns the compile-time float value, if known.
    pub fn float_val(&self) -> Option<f64> {
        match self {
            Value::Float(s) => s.static_val,
            _ => Option::None,
        }
    }

    /// Returns the compile-time bool value, if known.
    pub fn bool_val(&self) -> Option<bool> {
        match self {
            Value::Boolean(s) => s.static_val,
            _ => Option::None,
        }
    }

    /// Returns the compile-time string value, if known.
    pub fn string_val(&self) -> Option<&str> {
        match self {
            Value::String(s) => Some(&s.val),
            _ => Option::None,
        }
    }
}
