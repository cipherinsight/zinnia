use serde::{Deserialize, Serialize};
use std::fmt;

/// Statement ID type used throughout the IR system.
pub type StmtId = u32;

// ---------------------------------------------------------------------------
// ZinniaType — unified type discriminant (replaces DTDescriptor hierarchy)
// ---------------------------------------------------------------------------

/// Discriminant for numeric element types (Integer or Float).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NumberType {
    Integer,
    Float,
}

impl fmt::Display for NumberType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NumberType::Integer => write!(f, "Integer"),
            NumberType::Float => write!(f, "Float"),
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
    String,
    None,
    Class,
    NDArray {
        shape: Vec<usize>,
        dtype: NumberType,
    },
    DynamicNDArray {
        dtype: NumberType,
        max_length: usize,
        max_rank: usize,
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
            ZinniaType::String => write!(f, "String"),
            ZinniaType::None => write!(f, "None"),
            ZinniaType::Class => write!(f, "Class"),
            ZinniaType::NDArray { shape, dtype } => {
                let dims: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
                write!(f, "NDArray[{}, {}]", dtype, dims.join(", "))
            }
            ZinniaType::DynamicNDArray {
                dtype,
                max_length,
                max_rank,
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
    /// Returns true if this type is a numeric type (Integer, Float, or Boolean).
    pub fn is_number(&self) -> bool {
        matches!(
            self,
            ZinniaType::Integer | ZinniaType::Float | ZinniaType::Boolean
        )
    }

    /// Returns true if this type is an integer-like type (Integer or Boolean).
    pub fn is_integer(&self) -> bool {
        matches!(self, ZinniaType::Integer | ZinniaType::Boolean)
    }

    /// Returns true if this type is a float type.
    pub fn is_float(&self) -> bool {
        matches!(self, ZinniaType::Float)
    }

    /// Returns true if this type is a boolean type.
    pub fn is_boolean(&self) -> bool {
        matches!(self, ZinniaType::Boolean)
    }

    /// Returns true if this type is an NDArray or DynamicNDArray.
    pub fn is_ndarray(&self) -> bool {
        matches!(
            self,
            ZinniaType::NDArray { .. } | ZinniaType::DynamicNDArray { .. }
        )
    }

    /// Returns true if this type is a DynamicNDArray.
    pub fn is_dynamic_ndarray(&self) -> bool {
        matches!(self, ZinniaType::DynamicNDArray { .. })
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

    /// Create a `ZinniaType` from a type annotation string and optional arguments.
    /// Mirrors DTDescriptorFactory.create().
    pub fn from_annotation(typename: &str, args: &[AnnotationArg]) -> Result<ZinniaType, String> {
        match typename {
            "Integer" | "int" | "Int" | "integer" | "Boolean" | "bool" | "Bool" | "boolean" => {
                // Note: Python treats bool annotations as Integer currently
                Ok(ZinniaType::Integer)
            }
            "Float" | "float" => Ok(ZinniaType::Float),
            "String" | "str" => Ok(ZinniaType::String),
            "None" => Ok(ZinniaType::None),
            "Class" => Ok(ZinniaType::Class),
            "List" | "list" => {
                if args.is_empty() {
                    return Err("Annotation `List` requires 1 or more arguments".to_string());
                }
                let mut elements = Vec::new();
                for arg in args {
                    match arg {
                        AnnotationArg::Type(t) => elements.push(t.clone()),
                        _ => {
                            return Err(
                                "Annotation `List` requires all type arguments to be a datatype"
                                    .to_string(),
                            )
                        }
                    }
                }
                Ok(ZinniaType::List { elements })
            }
            "Tuple" | "tuple" => {
                if args.is_empty() {
                    return Err("Annotation `Tuple` requires 1 or more arguments".to_string());
                }
                let mut elements = Vec::new();
                for arg in args {
                    match arg {
                        AnnotationArg::Type(t) => elements.push(t.clone()),
                        _ => {
                            return Err(
                                "Annotation `Tuple` requires all type arguments to be a datatype"
                                    .to_string(),
                            )
                        }
                    }
                }
                Ok(ZinniaType::Tuple { elements })
            }
            "NDArray" => {
                if args.len() <= 1 {
                    return Err(
                        "Annotation `NDArray` requires 2 or more arguments".to_string()
                    );
                }
                let dtype = match &args[0] {
                    AnnotationArg::Type(ZinniaType::Integer) => NumberType::Integer,
                    AnnotationArg::Type(ZinniaType::Float) => NumberType::Float,
                    _ => {
                        return Err(
                            "Annotation `NDArray` missing a required argument dtype".to_string()
                        )
                    }
                };
                let mut shape = Vec::new();
                for arg in &args[1..] {
                    match arg {
                        AnnotationArg::Int(n) => shape.push(*n as usize),
                        _ => {
                            return Err(
                                "Annotation `NDArray` only accepts integers as dimension sizes"
                                    .to_string(),
                            )
                        }
                    }
                }
                Ok(ZinniaType::NDArray { shape, dtype })
            }
            "DynamicNDArray" => {
                if args.len() != 3 {
                    return Err(
                        "Annotation `DynamicNDArray` requires exactly 3 arguments: dtype, max_length, max_rank".to_string()
                    );
                }
                let dtype = match &args[0] {
                    AnnotationArg::Type(ZinniaType::Integer) => NumberType::Integer,
                    AnnotationArg::Type(ZinniaType::Float) => NumberType::Float,
                    _ => {
                        return Err(
                            "Unsupported `DynamicNDArray` dtype".to_string()
                        )
                    }
                };
                let max_length = match &args[1] {
                    AnnotationArg::Int(n) if *n > 0 => *n as usize,
                    _ => {
                        return Err(
                            "Annotation `DynamicNDArray` requires `max_length` to be a positive integer".to_string()
                        )
                    }
                };
                let max_rank = match &args[2] {
                    AnnotationArg::Int(n) if *n > 0 => *n as usize,
                    _ => {
                        return Err(
                            "Annotation `DynamicNDArray` requires `max_rank` to be a positive integer".to_string()
                        )
                    }
                };
                Ok(ZinniaType::DynamicNDArray {
                    dtype,
                    max_length,
                    max_rank,
                })
            }
            "PoseidonHashed" => {
                if args.len() != 1 {
                    return Err(
                        "Annotation `PoseidonHashed` requires exactly 1 argument".to_string()
                    );
                }
                match &args[0] {
                    AnnotationArg::Type(t) => Ok(ZinniaType::PoseidonHashed {
                        dtype: Box::new(t.clone()),
                    }),
                    _ => Err(
                        "Annotation `PoseidonHashed` requires a type argument".to_string()
                    ),
                }
            }
            _ => Err(format!("`{}` is not a valid type name", typename)),
        }
    }

    /// Returns the list of type alias strings that map to this type.
    pub fn type_aliases(&self) -> Vec<&'static str> {
        match self {
            ZinniaType::Integer => vec!["Integer", "int", "Int", "integer"],
            ZinniaType::Float => vec!["Float", "float"],
            ZinniaType::Boolean => vec!["Boolean", "bool", "Bool", "boolean"],
            ZinniaType::String => vec!["String", "str"],
            ZinniaType::None => vec!["None"],
            ZinniaType::Class => vec!["Class"],
            ZinniaType::NDArray { .. } => vec!["NDArray"],
            ZinniaType::DynamicNDArray { .. } => vec!["DynamicNDArray"],
            ZinniaType::List { .. } => vec!["List", "list"],
            ZinniaType::Tuple { .. } => vec!["Tuple", "tuple"],
            ZinniaType::PoseidonHashed { .. } => vec!["PoseidonHashed"],
        }
    }

    /// Returns the canonical typename string (matching Python get_typename()).
    pub fn typename(&self) -> &'static str {
        match self {
            ZinniaType::Integer => "Integer",
            ZinniaType::Float => "Float",
            ZinniaType::Boolean => "Boolean",
            ZinniaType::String => "String",
            ZinniaType::None => "None",
            ZinniaType::Class => "Class",
            ZinniaType::NDArray { .. } => "NDArray",
            ZinniaType::DynamicNDArray { .. } => "DynamicNDArray",
            ZinniaType::List { .. } => "List",
            ZinniaType::Tuple { .. } => "Tuple",
            ZinniaType::PoseidonHashed { .. } => "PoseidonHashed",
        }
    }

    /// For NDArray types, returns the total number of elements.
    pub fn num_elements(&self) -> Option<usize> {
        match self {
            ZinniaType::NDArray { shape, .. } => Some(shape.iter().product()),
            ZinniaType::DynamicNDArray { max_length, .. } => Some(*max_length),
            _ => Option::None,
        }
    }
}

/// Argument type for annotation parsing (can be a type or an integer literal).
#[derive(Debug, Clone)]
pub enum AnnotationArg {
    Type(ZinniaType),
    Int(i64),
}

// ---------------------------------------------------------------------------
// Serde support for DTDescriptor-compatible dict serialization
// ---------------------------------------------------------------------------

/// Mirrors the Python DTDescriptorFactory.export() / import_from() format:
/// `{"__class__": "IntegerDTDescriptor", "dt_data": {...}}`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DTDescriptorDict {
    #[serde(rename = "__class__")]
    pub class_name: String,
    pub dt_data: serde_json::Value,
}

impl ZinniaType {
    /// Deserialize from the Python DTDescriptorFactory.export() dict format.
    pub fn from_dt_dict(dict: &DTDescriptorDict) -> Result<ZinniaType, String> {
        match dict.class_name.as_str() {
            "IntegerDTDescriptor" => Ok(ZinniaType::Integer),
            "FloatDTDescriptor" => Ok(ZinniaType::Float),
            "BooleanDTDescriptor" => Ok(ZinniaType::Boolean),
            "StringDTDescriptor" => Ok(ZinniaType::String),
            "NoneDTDescriptor" => Ok(ZinniaType::None),
            "ClassDTDescriptor" => Ok(ZinniaType::Class),
            "NDArrayDTDescriptor" => {
                let data = &dict.dt_data;
                let dtype_dict: DTDescriptorDict =
                    serde_json::from_value(data["dtype"].clone())
                        .map_err(|e| format!("NDArray dtype: {}", e))?;
                let dtype = match ZinniaType::from_dt_dict(&dtype_dict)? {
                    ZinniaType::Integer => NumberType::Integer,
                    ZinniaType::Float => NumberType::Float,
                    _ => return Err("NDArray dtype must be Integer or Float".to_string()),
                };
                let shape: Vec<usize> = data["shape"]
                    .as_array()
                    .ok_or("NDArray shape must be an array")?
                    .iter()
                    .map(|v| v.as_u64().unwrap_or(0) as usize)
                    .collect();
                Ok(ZinniaType::NDArray { shape, dtype })
            }
            "DynamicNDArrayDTDescriptor" => {
                let data = &dict.dt_data;
                let dtype_dict: DTDescriptorDict =
                    serde_json::from_value(data["dtype"].clone())
                        .map_err(|e| format!("DynamicNDArray dtype: {}", e))?;
                let dtype = match ZinniaType::from_dt_dict(&dtype_dict)? {
                    ZinniaType::Integer => NumberType::Integer,
                    ZinniaType::Float => NumberType::Float,
                    _ => {
                        return Err(
                            "DynamicNDArray dtype must be Integer or Float".to_string()
                        )
                    }
                };
                let max_length = data["max_length"].as_u64().unwrap_or(0) as usize;
                let max_rank = data["max_rank"].as_u64().unwrap_or(0) as usize;
                Ok(ZinniaType::DynamicNDArray {
                    dtype,
                    max_length,
                    max_rank,
                })
            }
            "ListDTDescriptor" => {
                let data = &dict.dt_data;
                let elements_raw = data["elements"]
                    .as_array()
                    .ok_or("List elements must be an array")?;
                let mut elements = Vec::new();
                for elem in elements_raw {
                    let elem_dict: DTDescriptorDict =
                        serde_json::from_value(elem.clone())
                            .map_err(|e| format!("List element: {}", e))?;
                    elements.push(ZinniaType::from_dt_dict(&elem_dict)?);
                }
                Ok(ZinniaType::List { elements })
            }
            "TupleDTDescriptor" => {
                let data = &dict.dt_data;
                let elements_raw = data["elements"]
                    .as_array()
                    .ok_or("Tuple elements must be an array")?;
                let mut elements = Vec::new();
                for elem in elements_raw {
                    let elem_dict: DTDescriptorDict =
                        serde_json::from_value(elem.clone())
                            .map_err(|e| format!("Tuple element: {}", e))?;
                    elements.push(ZinniaType::from_dt_dict(&elem_dict)?);
                }
                Ok(ZinniaType::Tuple { elements })
            }
            "PoseidonHashedDTDescriptor" => {
                let data = &dict.dt_data;
                let dtype_dict: DTDescriptorDict =
                    serde_json::from_value(data["dtype"].clone())
                        .map_err(|e| format!("PoseidonHashed dtype: {}", e))?;
                let dtype = ZinniaType::from_dt_dict(&dtype_dict)?;
                Ok(ZinniaType::PoseidonHashed {
                    dtype: Box::new(dtype),
                })
            }
            other => Err(format!("Unknown DTDescriptor class: {}", other)),
        }
    }

    /// Serialize to the Python DTDescriptorFactory.export() dict format.
    pub fn to_dt_dict(&self) -> DTDescriptorDict {
        match self {
            ZinniaType::Integer => DTDescriptorDict {
                class_name: "IntegerDTDescriptor".to_string(),
                dt_data: serde_json::json!({}),
            },
            ZinniaType::Float => DTDescriptorDict {
                class_name: "FloatDTDescriptor".to_string(),
                dt_data: serde_json::json!({}),
            },
            ZinniaType::Boolean => DTDescriptorDict {
                class_name: "BooleanDTDescriptor".to_string(),
                dt_data: serde_json::json!({}),
            },
            ZinniaType::String => DTDescriptorDict {
                class_name: "StringDTDescriptor".to_string(),
                dt_data: serde_json::json!({}),
            },
            ZinniaType::None => DTDescriptorDict {
                class_name: "NoneDTDescriptor".to_string(),
                dt_data: serde_json::json!({}),
            },
            ZinniaType::Class => DTDescriptorDict {
                class_name: "ClassDTDescriptor".to_string(),
                dt_data: serde_json::json!({}),
            },
            ZinniaType::NDArray { shape, dtype } => {
                let dtype_type = match dtype {
                    NumberType::Integer => ZinniaType::Integer,
                    NumberType::Float => ZinniaType::Float,
                };
                let dtype_dict = dtype_type.to_dt_dict();
                DTDescriptorDict {
                    class_name: "NDArrayDTDescriptor".to_string(),
                    dt_data: serde_json::json!({
                        "dtype": serde_json::to_value(&dtype_dict).unwrap(),
                        "shape": shape,
                    }),
                }
            }
            ZinniaType::DynamicNDArray {
                dtype,
                max_length,
                max_rank,
            } => {
                let dtype_type = match dtype {
                    NumberType::Integer => ZinniaType::Integer,
                    NumberType::Float => ZinniaType::Float,
                };
                let dtype_dict = dtype_type.to_dt_dict();
                DTDescriptorDict {
                    class_name: "DynamicNDArrayDTDescriptor".to_string(),
                    dt_data: serde_json::json!({
                        "dtype": serde_json::to_value(&dtype_dict).unwrap(),
                        "max_length": max_length,
                        "max_rank": max_rank,
                    }),
                }
            }
            ZinniaType::List { elements } => {
                let elems: Vec<serde_json::Value> = elements
                    .iter()
                    .map(|e| serde_json::to_value(e.to_dt_dict()).unwrap())
                    .collect();
                DTDescriptorDict {
                    class_name: "ListDTDescriptor".to_string(),
                    dt_data: serde_json::json!({ "elements": elems }),
                }
            }
            ZinniaType::Tuple { elements } => {
                let elems: Vec<serde_json::Value> = elements
                    .iter()
                    .map(|e| serde_json::to_value(e.to_dt_dict()).unwrap())
                    .collect();
                DTDescriptorDict {
                    class_name: "TupleDTDescriptor".to_string(),
                    dt_data: serde_json::json!({ "elements": elems }),
                }
            }
            ZinniaType::PoseidonHashed { dtype } => {
                let dtype_dict = dtype.to_dt_dict();
                DTDescriptorDict {
                    class_name: "PoseidonHashedDTDescriptor".to_string(),
                    dt_data: serde_json::json!({
                        "dtype": serde_json::to_value(&dtype_dict).unwrap(),
                    }),
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ScalarValue — compile-time constant + IR statement reference
// ---------------------------------------------------------------------------

/// Holds an optional compile-time constant value and an optional IR statement
/// pointer. Replaces the ValueTriplet pattern for atomic types.
#[derive(Debug, Clone)]
pub struct ScalarValue<T: Clone> {
    /// Compile-time constant value (None if unknown at compile time).
    pub static_val: Option<T>,
    /// IR statement ID that produces this value (None if pure constant).
    pub ptr: Option<StmtId>,
}

impl<T: Clone> ScalarValue<T> {
    pub fn new(static_val: Option<T>, ptr: Option<StmtId>) -> Self {
        Self { static_val, ptr }
    }

    pub fn constant(val: T) -> Self {
        Self {
            static_val: Some(val),
            ptr: Option::None,
        }
    }

    pub fn runtime(ptr: StmtId) -> Self {
        Self {
            static_val: Option::None,
            ptr: Some(ptr),
        }
    }

    pub fn known(val: T, ptr: StmtId) -> Self {
        Self {
            static_val: Some(val),
            ptr: Some(ptr),
        }
    }
}

impl<T: Clone + PartialEq> PartialEq for ScalarValue<T> {
    fn eq(&self, other: &Self) -> bool {
        self.static_val == other.static_val && self.ptr == other.ptr
    }
}

// ---------------------------------------------------------------------------
// StringValue
// ---------------------------------------------------------------------------

/// String value with compile-time known string and an IR statement reference.
#[derive(Debug, Clone)]
pub struct StringValue {
    pub val: String,
    pub ptr: StmtId,
}

// ---------------------------------------------------------------------------
// NDArrayData — storage for static-shaped arrays
// ---------------------------------------------------------------------------

/// Flat storage for NDArray values.
#[derive(Debug, Clone)]
pub struct NDArrayData {
    pub shape: Vec<usize>,
    pub dtype: NumberType,
    /// Flat storage of element values. Length = product of shape.
    pub elements: Vec<ScalarValue<i64>>,
}

impl NDArrayData {
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }
}

// ---------------------------------------------------------------------------
// DynArrayMeta — metadata for DynamicNDArray
// ---------------------------------------------------------------------------

/// Runtime metadata for dynamic-shaped arrays.
#[derive(Debug, Clone)]
pub struct DynArrayMeta {
    pub logical_shape: Vec<usize>,
    pub logical_offset: usize,
    pub logical_strides: Vec<usize>,
    pub runtime_length: ScalarValue<i64>,
    pub runtime_rank: ScalarValue<i64>,
    pub runtime_shape: Vec<ScalarValue<i64>>,
    pub runtime_strides: Vec<ScalarValue<i64>>,
    pub runtime_offset: ScalarValue<i64>,
}

// ---------------------------------------------------------------------------
// DynamicNDArrayData — storage for runtime-shaped arrays
// ---------------------------------------------------------------------------

/// Storage for DynamicNDArray values (extends NDArrayData with metadata).
#[derive(Debug, Clone)]
pub struct DynamicNDArrayData {
    pub max_length: usize,
    pub max_rank: usize,
    pub dtype: NumberType,
    /// Flat storage of element values. Length = max_length.
    pub elements: Vec<ScalarValue<i64>>,
    pub meta: DynArrayMeta,
}

// ---------------------------------------------------------------------------
// CompositeData — storage for List/Tuple values
// ---------------------------------------------------------------------------

/// Storage for List and Tuple composite values.
#[derive(Debug, Clone)]
pub struct CompositeData {
    pub elements_type: Vec<ZinniaType>,
    pub values: Vec<Value>,
}

// ---------------------------------------------------------------------------
// Value — the unified value enum (replaces Value class hierarchy)
// ---------------------------------------------------------------------------

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
                max_length: data.max_length,
                max_rank: data.max_rank,
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zinnia_type_display() {
        assert_eq!(ZinniaType::Integer.to_string(), "Integer");
        assert_eq!(ZinniaType::Float.to_string(), "Float");
        assert_eq!(ZinniaType::Boolean.to_string(), "Boolean");
        assert_eq!(
            ZinniaType::NDArray {
                shape: vec![3, 4],
                dtype: NumberType::Float,
            }
            .to_string(),
            "NDArray[Float, 3, 4]"
        );
        assert_eq!(
            ZinniaType::DynamicNDArray {
                dtype: NumberType::Integer,
                max_length: 100,
                max_rank: 2,
            }
            .to_string(),
            "DynamicNDArray[Integer, 100, 2]"
        );
        assert_eq!(
            ZinniaType::List {
                elements: vec![ZinniaType::Integer, ZinniaType::Float],
            }
            .to_string(),
            "List[Integer, Float]"
        );
        assert_eq!(
            ZinniaType::PoseidonHashed {
                dtype: Box::new(ZinniaType::Integer),
            }
            .to_string(),
            "PoseidonHashed[Integer]"
        );
    }

    #[test]
    fn test_type_predicates() {
        assert!(ZinniaType::Integer.is_number());
        assert!(ZinniaType::Float.is_number());
        assert!(ZinniaType::Boolean.is_number());
        assert!(!ZinniaType::String.is_number());

        assert!(ZinniaType::Integer.is_integer());
        assert!(ZinniaType::Boolean.is_integer());
        assert!(!ZinniaType::Float.is_integer());

        assert!(ZinniaType::NDArray {
            shape: vec![2],
            dtype: NumberType::Integer,
        }
        .is_ndarray());
        assert!(ZinniaType::DynamicNDArray {
            dtype: NumberType::Float,
            max_length: 10,
            max_rank: 1,
        }
        .is_ndarray());
    }

    #[test]
    fn test_implicit_cast() {
        assert!(ZinniaType::Boolean.can_implicit_cast_to(&ZinniaType::Integer));
        assert!(ZinniaType::Integer.can_implicit_cast_to(&ZinniaType::Integer));
        assert!(!ZinniaType::Integer.can_implicit_cast_to(&ZinniaType::Float));
        assert!(!ZinniaType::Float.can_implicit_cast_to(&ZinniaType::Integer));
    }

    #[test]
    fn test_from_annotation() {
        assert_eq!(
            ZinniaType::from_annotation("int", &[]).unwrap(),
            ZinniaType::Integer
        );
        assert_eq!(
            ZinniaType::from_annotation("Float", &[]).unwrap(),
            ZinniaType::Float
        );
        assert_eq!(
            ZinniaType::from_annotation("NDArray", &[
                AnnotationArg::Type(ZinniaType::Float),
                AnnotationArg::Int(3),
                AnnotationArg::Int(4),
            ])
            .unwrap(),
            ZinniaType::NDArray {
                shape: vec![3, 4],
                dtype: NumberType::Float,
            }
        );
        assert_eq!(
            ZinniaType::from_annotation("DynamicNDArray", &[
                AnnotationArg::Type(ZinniaType::Integer),
                AnnotationArg::Int(100),
                AnnotationArg::Int(2),
            ])
            .unwrap(),
            ZinniaType::DynamicNDArray {
                dtype: NumberType::Integer,
                max_length: 100,
                max_rank: 2,
            }
        );
        assert!(ZinniaType::from_annotation("NDArray", &[]).is_err());
        assert!(ZinniaType::from_annotation("unknown_type", &[]).is_err());
    }

    #[test]
    fn test_num_elements() {
        assert_eq!(
            ZinniaType::NDArray {
                shape: vec![3, 4],
                dtype: NumberType::Float,
            }
            .num_elements(),
            Some(12)
        );
        assert_eq!(
            ZinniaType::DynamicNDArray {
                dtype: NumberType::Integer,
                max_length: 100,
                max_rank: 2,
            }
            .num_elements(),
            Some(100)
        );
        assert_eq!(ZinniaType::Integer.num_elements(), Option::None);
    }

    #[test]
    fn test_dt_dict_round_trip() {
        let types = vec![
            ZinniaType::Integer,
            ZinniaType::Float,
            ZinniaType::Boolean,
            ZinniaType::String,
            ZinniaType::None,
            ZinniaType::Class,
            ZinniaType::NDArray {
                shape: vec![2, 3],
                dtype: NumberType::Float,
            },
            ZinniaType::DynamicNDArray {
                dtype: NumberType::Integer,
                max_length: 50,
                max_rank: 3,
            },
            ZinniaType::List {
                elements: vec![ZinniaType::Integer, ZinniaType::Float],
            },
            ZinniaType::Tuple {
                elements: vec![ZinniaType::String, ZinniaType::Boolean],
            },
            ZinniaType::PoseidonHashed {
                dtype: Box::new(ZinniaType::Integer),
            },
        ];

        for ty in types {
            let dict = ty.to_dt_dict();
            let restored = ZinniaType::from_dt_dict(&dict)
                .unwrap_or_else(|e| panic!("Failed to round-trip {:?}: {}", ty, e));
            assert_eq!(ty, restored, "Round-trip failed for {:?}", ty);
        }
    }

    #[test]
    fn test_value_creation() {
        let v = Value::Integer(ScalarValue::known(42, 0));
        assert_eq!(v.zinnia_type(), ZinniaType::Integer);
        assert_eq!(v.ptr(), Some(0));
        assert_eq!(v.int_val(), Some(42));
        assert!(v.is_number());
        assert!(v.is_integer());

        let v = Value::Float(ScalarValue::known(3.14, 1));
        assert_eq!(v.zinnia_type(), ZinniaType::Float);
        assert_eq!(v.float_val(), Some(3.14));

        let v = Value::Boolean(ScalarValue::known(true, 2));
        assert_eq!(v.zinnia_type(), ZinniaType::Boolean);
        assert_eq!(v.bool_val(), Some(true));
        assert!(v.is_integer()); // Boolean is integer-like

        let v = Value::None;
        assert_eq!(v.zinnia_type(), ZinniaType::None);
        assert_eq!(v.ptr(), Option::None);
    }

    #[test]
    fn test_number_type() {
        assert_eq!(ZinniaType::Integer.number_type(), Some(NumberType::Integer));
        assert_eq!(ZinniaType::Float.number_type(), Some(NumberType::Float));
        assert_eq!(ZinniaType::Boolean.number_type(), Some(NumberType::Integer));
        assert_eq!(ZinniaType::String.number_type(), Option::None);
    }
}
