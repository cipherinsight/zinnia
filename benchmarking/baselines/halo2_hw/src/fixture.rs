//! Reading `fixture.json`, the shared per-case input file.
//!
//! Schema (owned by the dataset workstream, see the benchmark brief):
//!
//! ```json
//! { "case": "ds1000_case0300", "function": "kernel",
//!   "parameters": [ {"name": "a", "kind": "array", "dtype": "float", "shape": [20]},
//!                   {"name": "p", "kind": "scalar", "dtype": "int"} ],
//!   "inputs": { "A": {"a": [...], "p": 25}, "B": {"a": [...], "p": 37} } }
//! ```
//!
//! Arrays are nested lists in row-major order. A circuit asks for each
//! parameter by name together with the shape it was written for; a mismatch is
//! an error, never a silently different circuit.

use std::collections::BTreeMap;
use std::path::Path;

use serde::Deserialize;
use serde_json::Value;

use crate::Error;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Dtype {
    Float,
    Int,
    Bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ParamKind {
    Array,
    Scalar,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Parameter {
    pub name: String,
    pub kind: ParamKind,
    pub dtype: Dtype,
    #[serde(default)]
    pub shape: Vec<usize>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Fixture {
    pub case: String,
    pub parameters: Vec<Parameter>,
    pub inputs: BTreeMap<String, BTreeMap<String, Value>>,
}

impl Fixture {
    pub fn load(path: &Path) -> Result<Self, Error> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| Error::new(format!("cannot read fixture {}: {e}", path.display())))?;
        serde_json::from_str(&text)
            .map_err(|e| Error::new(format!("invalid fixture {}: {e}", path.display())))
    }

    /// The parameter values of one named input set (`"A"` or `"B"`).
    pub fn inputs(&self, input_name: &str) -> Result<Inputs, Error> {
        let raw = self.inputs.get(input_name).ok_or_else(|| {
            Error::new(format!("fixture for {} has no input set {input_name:?}", self.case))
        })?;
        let mut values = BTreeMap::new();
        for param in &self.parameters {
            let json = raw.get(&param.name).ok_or_else(|| {
                Error::new(format!("input set {input_name:?} lacks parameter {:?}", param.name))
            })?;
            values.insert(param.name.clone(), ParamValue::from_json(param, json)?);
        }
        Ok(Inputs { values })
    }
}

/// One parameter's value, flattened in row-major order.
#[derive(Clone, Debug)]
pub struct ParamValue {
    pub dtype: Dtype,
    /// Empty for scalars.
    pub shape: Vec<usize>,
    /// Leaf JSON values (numbers or booleans).
    pub data: Vec<Value>,
}

impl ParamValue {
    fn from_json(param: &Parameter, json: &Value) -> Result<Self, Error> {
        let mut data = Vec::new();
        let mut shape = Vec::new();
        flatten(json, 0, &mut shape, &mut data)
            .map_err(|e| Error::new(format!("parameter {:?}: {e}", param.name)))?;
        let declared: &[usize] = match param.kind {
            ParamKind::Scalar => &[],
            ParamKind::Array => &param.shape,
        };
        if shape != declared {
            return Err(Error::new(format!(
                "parameter {:?}: value has shape {shape:?}, fixture declares {declared:?}",
                param.name
            )));
        }
        Ok(ParamValue { dtype: param.dtype, shape, data })
    }
}

/// Walks a nested JSON list, recording the shape on the first descent and
/// checking that every sibling has the same shape.
fn flatten(json: &Value, depth: usize, shape: &mut Vec<usize>, out: &mut Vec<Value>) -> Result<(), String> {
    match json {
        Value::Array(items) => {
            if shape.len() == depth {
                shape.push(items.len());
            } else if shape.get(depth) != Some(&items.len()) {
                return Err("ragged nested list".into());
            }
            for item in items {
                flatten(item, depth + 1, shape, out)?;
            }
            Ok(())
        }
        Value::Number(_) | Value::Bool(_) => {
            if shape.len() != depth {
                return Err("ragged nested list".into());
            }
            out.push(json.clone());
            Ok(())
        }
        other => Err(format!("unsupported JSON value {other}")),
    }
}

/// The values of one input set, looked up by parameter name.
#[derive(Clone, Debug)]
pub struct Inputs {
    values: BTreeMap<String, ParamValue>,
}

impl Inputs {
    fn get(&self, name: &str, dtype: Dtype, shape: &[usize]) -> Result<&ParamValue, Error> {
        let value = self
            .values
            .get(name)
            .ok_or_else(|| Error::new(format!("no parameter named {name:?}")))?;
        if value.dtype != dtype {
            return Err(Error::new(format!(
                "parameter {name:?} has dtype {:?}, circuit expects {dtype:?}",
                value.dtype
            )));
        }
        if value.shape != shape {
            return Err(Error::new(format!(
                "parameter {name:?} has shape {:?}, circuit was written for {shape:?}",
                value.shape
            )));
        }
        Ok(value)
    }

    /// A float array of exactly `shape`, flattened row-major.
    pub fn floats(&self, name: &str, shape: &[usize]) -> Result<Vec<f64>, Error> {
        self.get(name, Dtype::Float, shape)?.data.iter().map(|v| to_f64(name, v)).collect()
    }

    /// An integer array of exactly `shape`, flattened row-major.
    pub fn ints(&self, name: &str, shape: &[usize]) -> Result<Vec<i64>, Error> {
        self.get(name, Dtype::Int, shape)?.data.iter().map(|v| to_i64(name, v)).collect()
    }

    /// A boolean array of exactly `shape`, flattened row-major.
    pub fn bools(&self, name: &str, shape: &[usize]) -> Result<Vec<bool>, Error> {
        self.get(name, Dtype::Bool, shape)?.data.iter().map(|v| to_bool(name, v)).collect()
    }

    pub fn float(&self, name: &str) -> Result<f64, Error> {
        to_f64(name, &self.get(name, Dtype::Float, &[])?.data[0])
    }

    pub fn int(&self, name: &str) -> Result<i64, Error> {
        to_i64(name, &self.get(name, Dtype::Int, &[])?.data[0])
    }
}

fn to_f64(name: &str, v: &Value) -> Result<f64, Error> {
    v.as_f64().ok_or_else(|| Error::new(format!("parameter {name:?}: {v} is not a number")))
}

fn to_i64(name: &str, v: &Value) -> Result<i64, Error> {
    v.as_i64().ok_or_else(|| Error::new(format!("parameter {name:?}: {v} is not a 64-bit integer")))
}

fn to_bool(name: &str, v: &Value) -> Result<bool, Error> {
    match v {
        Value::Bool(b) => Ok(*b),
        Value::Number(n) if n.as_i64() == Some(0) => Ok(false),
        Value::Number(n) if n.as_i64() == Some(1) => Ok(true),
        _ => Err(Error::new(format!("parameter {name:?}: {v} is not a boolean"))),
    }
}
