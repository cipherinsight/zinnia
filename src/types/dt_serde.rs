use serde::{Deserialize, Serialize};

use super::zinnia_type::{NumberType, ZinniaType};

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
