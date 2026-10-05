//! Loading kernel inputs from `datasets/ds1000/<case>/fixture.json`.
//!
//! Parameters come in the kernel's signature order. Float values are encoded
//! as Q32 (`round(v * 2^32)`, Zinnia's quantizer), integers as `i64`, and
//! booleans as `bool`. Arrays are flattened in row-major order and written as
//! one vector each, so the guest reads one value per parameter.

use std::{collections::BTreeMap, path::Path};

use anyhow::{bail, ensure, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::Q32;

#[derive(Debug, Deserialize)]
pub struct Fixture {
    pub case: String,
    pub function: String,
    pub parameters: Vec<Parameter>,
    pub inputs: BTreeMap<String, BTreeMap<String, Value>>,
}

#[derive(Debug, Deserialize)]
pub struct Parameter {
    pub name: String,
    pub kind: Kind,
    pub dtype: Dtype,
    #[serde(default)]
    pub shape: Option<Vec<usize>>,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Kind {
    Scalar,
    Array,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Dtype {
    Float,
    Int,
    Bool,
}

/// One encoded kernel argument.
#[derive(Clone, Debug, PartialEq)]
pub enum Arg {
    Float(Q32),
    Int(i64),
    Bool(bool),
    FloatArray(Vec<Q32>),
    IntArray(Vec<i64>),
    BoolArray(Vec<bool>),
}

/// A zkVM's private-input writer (`SP1Stdin`, RISC Zero's
/// `ExecutorEnvBuilder`, or the native runner).
pub trait ArgWriter {
    fn write<T: Serialize>(&mut self, value: &T);
}

impl Arg {
    pub fn write_to(&self, writer: &mut impl ArgWriter) {
        match self {
            Arg::Float(v) => writer.write(v),
            Arg::Int(v) => writer.write(v),
            Arg::Bool(v) => writer.write(v),
            Arg::FloatArray(v) => writer.write(v),
            Arg::IntArray(v) => writer.write(v),
            Arg::BoolArray(v) => writer.write(v),
        }
    }
}

impl Fixture {
    pub fn load(path: &Path) -> Result<Fixture> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading fixture {}", path.display()))?;
        serde_json::from_str(&text).with_context(|| format!("parsing fixture {}", path.display()))
    }

    /// The encoded arguments of input `name` (`A` or `B`), in signature order.
    pub fn args(&self, name: &str) -> Result<Vec<Arg>> {
        let values = self
            .inputs
            .get(name)
            .with_context(|| format!("fixture has no input named {name:?}"))?;
        ensure!(
            values.len() == self.parameters.len(),
            "input {name:?} has {} values for {} parameters",
            values.len(),
            self.parameters.len()
        );
        self.parameters
            .iter()
            .map(|param| {
                let value = values
                    .get(&param.name)
                    .with_context(|| format!("input {name:?} lacks parameter {:?}", param.name))?;
                param.encode(value).with_context(|| format!("parameter {:?}", param.name))
            })
            .collect()
    }
}

impl Parameter {
    fn encode(&self, value: &Value) -> Result<Arg> {
        Ok(match self.kind {
            Kind::Scalar => match self.dtype {
                Dtype::Float => Arg::Float(float(value)?),
                Dtype::Int => Arg::Int(int(value)?),
                Dtype::Bool => Arg::Bool(boolean(value)?),
            },
            Kind::Array => {
                let mut leaves = Vec::new();
                let shape = flatten(value, &mut leaves)?;
                if let Some(expected) = &self.shape {
                    ensure!(&shape == expected, "shape {shape:?}, expected {expected:?}");
                }
                match self.dtype {
                    Dtype::Float => Arg::FloatArray(leaves.iter().copied().map(float).collect::<Result<_>>()?),
                    Dtype::Int => Arg::IntArray(leaves.iter().copied().map(int).collect::<Result<_>>()?),
                    Dtype::Bool => Arg::BoolArray(leaves.iter().copied().map(boolean).collect::<Result<_>>()?),
                }
            }
        })
    }
}

/// Collects the leaves of a nested list in row-major order and returns its
/// shape; rejects ragged lists.
fn flatten<'a>(value: &'a Value, leaves: &mut Vec<&'a Value>) -> Result<Vec<usize>> {
    match value {
        Value::Array(items) => {
            let mut inner: Option<Vec<usize>> = None;
            for item in items {
                let shape = flatten(item, leaves)?;
                match &inner {
                    None => inner = Some(shape),
                    Some(first) => ensure!(*first == shape, "ragged array"),
                }
            }
            let mut shape = vec![items.len()];
            shape.extend(inner.unwrap_or_default());
            Ok(shape)
        }
        leaf => {
            leaves.push(leaf);
            Ok(Vec::new())
        }
    }
}

fn float(value: &Value) -> Result<Q32> {
    match value.as_f64() {
        Some(v) if v.is_finite() => Ok(Q32::from_f64(v)),
        _ => bail!("expected a finite number, got {value}"),
    }
}

fn int(value: &Value) -> Result<i64> {
    value.as_i64().with_context(|| format!("expected an integer, got {value}"))
}

fn boolean(value: &Value) -> Result<bool> {
    value.as_bool().with_context(|| format!("expected a boolean, got {value}"))
}
