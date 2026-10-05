//! Loading kernel inputs from `datasets/ds1000/<case>/fixture.json`.
//!
//! Parameters come in the kernel's signature order. Each becomes one buffer
//! of 64-bit words (see [`crate::Word`]): floats as Q32 (`round(v * 2^32)`,
//! Zinnia's quantizer), integers as themselves, booleans as 0 or 1; arrays
//! flattened in row-major order, a scalar as one word.

use std::{collections::BTreeMap, path::Path};

use anyhow::{bail, ensure, Context, Result};
use serde::Deserialize;
use serde_json::Value;

use crate::{Word, Q32};

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


impl Fixture {
    pub fn load(path: &Path) -> Result<Fixture> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading fixture {}", path.display()))?;
        serde_json::from_str(&text).with_context(|| format!("parsing fixture {}", path.display()))
    }

    /// The words of each argument of input `name` (`A` or `B`), in signature
    /// order, leaving out the parameters in `unused` (see
    /// [`crate::unused_params`]).
    pub fn args(&self, name: &str, unused: &[&str]) -> Result<Vec<Vec<i64>>> {
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
        for skipped in unused {
            ensure!(
                self.parameters.iter().any(|p| p.name == *skipped),
                "unused parameter {skipped:?} is not a kernel parameter"
            );
        }
        self.parameters
            .iter()
            .filter(|param| !unused.contains(&param.name.as_str()))
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
    /// The argument's words: one for a scalar, the row-major elements of an
    /// array.
    fn encode(&self, value: &Value) -> Result<Vec<i64>> {
        let mut leaves = Vec::new();
        match self.kind {
            Kind::Scalar => leaves.push(value),
            Kind::Array => {
                let shape = flatten(value, &mut leaves)?;
                if let Some(expected) = &self.shape {
                    ensure!(&shape == expected, "shape {shape:?}, expected {expected:?}");
                }
            }
        }
        leaves
            .into_iter()
            .map(|leaf| {
                Ok(match self.dtype {
                    Dtype::Float => float(leaf)?.to_word(),
                    Dtype::Int => int(leaf)?,
                    Dtype::Bool => boolean(leaf)?.to_word(),
                })
            })
            .collect()
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Fixture floats must be parsed correctly rounded (as Python and Rust's
    /// `str::parse` do); otherwise some Q32 encodings come out one step off
    /// from Zinnia's and the Cairo driver's.
    #[test]
    fn floats_are_parsed_exactly() {
        let mut state: u64 = 0x2545_f491_4f6c_dd1d;
        for _ in 0..20_000 {
            state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let v = (state >> 11) as f64 / (1u64 << 53) as f64 * 2e6 - 1e6;
            let text = format!("{v:.17e}");
            let exact: f64 = text.parse().unwrap();
            let parsed = serde_json::from_str::<Value>(&text).unwrap().as_f64().unwrap();
            assert_eq!(parsed.to_bits(), exact.to_bits(), "{text}");
        }
    }
}
