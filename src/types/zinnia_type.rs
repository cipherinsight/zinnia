use serde::{Deserialize, Serialize};
use std::fmt;

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
