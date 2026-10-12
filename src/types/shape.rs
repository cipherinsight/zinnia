use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ShapeDim {
    Const(usize),
    Symbol(String),
    Bounded { symbol: String, upper: usize },
}

impl ShapeDim {

    pub fn upper_bound(&self, bindings: &BTreeMap<String, usize>) -> Option<usize> {
        match self {
            ShapeDim::Const(n) => Some(*n),
            ShapeDim::Symbol(name) => bindings.get(name).copied(),
            ShapeDim::Bounded { upper, .. } => Some(*upper),
        }
    }
}

impl fmt::Display for ShapeDim {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ShapeDim::Const(n) => write!(f, "{}", n),
            ShapeDim::Symbol(name) => write!(f, "{}", name),
            ShapeDim::Bounded { symbol, upper } => write!(f, "{}<={}", symbol, upper),
        }
    }
}

pub fn concretize_shape_dims(
    shape: &[ShapeDim],
    bindings: &BTreeMap<String, usize>,
) -> Result<Vec<usize>, Vec<String>> {
    let mut concrete = Vec::with_capacity(shape.len());
    let mut unresolved = Vec::new();
    for dim in shape {
        match dim {
            ShapeDim::Const(n) => concrete.push(*n),
            ShapeDim::Symbol(name) => match bindings.get(name) {
                Some(n) => concrete.push(*n),
                None => unresolved.push(name.clone()),
            },
            ShapeDim::Bounded { symbol, .. } => unresolved.push(symbol.clone()),
        }
    }
    if unresolved.is_empty() {
        Ok(concrete)
    } else {
        Err(unresolved)
    }
}

pub fn bounded_shape_total(
    shape: &[ShapeDim],
    bindings: &BTreeMap<String, usize>,
) -> Result<usize, Vec<String>> {
    let mut total = 1usize;
    let mut unresolved = Vec::new();
    for dim in shape {
        match dim.upper_bound(bindings) {
            Some(n) => total = total.saturating_mul(n),
            None => match dim {
                ShapeDim::Symbol(name) => unresolved.push(name.clone()),
                ShapeDim::Bounded { symbol, .. } => unresolved.push(symbol.clone()),
                ShapeDim::Const(_) => {}
            },
        }
    }
    if unresolved.is_empty() {
        Ok(total)
    } else {
        Err(unresolved)
    }
}
