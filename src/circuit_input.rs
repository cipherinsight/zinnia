//! Structured circuit input types.
//!
//! This module defines the nested, self-describing format for circuit inputs
//! that crosses the Python → Rust boundary. It replaces the legacy flat
//! key-value format (`"0_1_2" → Integer(42)`) with typed tree structures
//! carrying metadata (parameter name, type, public/private).
//!
//! # Key types
//!
//! - [`InputPath`] / [`PathSegment`]: address a scalar leaf in the input tree.
//!   Used by IR read instructions to identify which circuit input to read.
//! - [`InputNode`]: a recursive tree of actual input data, mirroring the
//!   structure of `ZinniaType` but using distinct variant names.
//! - [`InputParam`]: one circuit input parameter with metadata and value.
//! - [`CircuitInputs`]: top-level collection of all circuit input parameters.
//! - [`ResolvedWitness`]: the full prover witness (circuit inputs + external
//!   function results), ready for synthesis.

use serde::{Deserialize, Serialize};

use crate::prove::error::ProvingError;
use crate::prove::kernel;

// ---------------------------------------------------------------------------
// InputPath — how IR instructions address circuit input values
// ---------------------------------------------------------------------------

/// A path into the structured circuit input tree.
///
/// IR read instructions (`ReadInteger`, `ReadFloat`, `ReadHash`) carry an
/// `InputPath` that names the parameter and the path through its nested
/// structure to reach a scalar leaf.
///
/// # Examples
///
/// | Circuit parameter | Path | Display |
/// |-------------------|------|---------|
/// | `x: int` (scalar) | `{ param: "x", segments: [] }` | `x` |
/// | `arr: NDArray[int, 2, 3]`, element 5 | `{ param: "arr", segments: [Index(5)] }` | `arr.5` |
/// | `h: PoseidonHashed[int]`, inner | `{ param: "h", segments: [Inner] }` | `h.inner` |
/// | `h: PoseidonHashed[int]`, hash | `{ param: "h", segments: [Hash] }` | `h.hash` |
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct InputPath {
    /// The parameter name from the circuit definition.
    pub param: String,
    /// Path segments navigating into the nested value.
    pub segments: Vec<PathSegment>,
}

impl InputPath {
    pub fn new(param: impl Into<String>, segments: Vec<PathSegment>) -> Self {
        Self { param: param.into(), segments }
    }

    /// Human-readable display string, e.g. `"arr.5"`, `"h.inner.0"`.
    pub fn display(&self) -> String {
        if self.segments.is_empty() {
            return self.param.clone();
        }
        let mut s = self.param.clone();
        for seg in &self.segments {
            s.push('.');
            match seg {
                PathSegment::Index(i) => s.push_str(&i.to_string()),
                PathSegment::Inner => s.push_str("inner"),
                PathSegment::Hash => s.push_str("hash"),
            }
        }
        s
    }
}

impl std::fmt::Display for InputPath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.display())
    }
}

/// A single step in an [`InputPath`].
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PathSegment {
    /// Index into a list, tuple, NDArray, or DynamicNDArray.
    Index(u32),
    /// Navigate into the inner value of a `PoseidonHashed`.
    Inner,
    /// Navigate to the hash field of a `PoseidonHashed`.
    Hash,
}

// ---------------------------------------------------------------------------
// InputNode — the recursive circuit input data tree
// ---------------------------------------------------------------------------

/// A node in the structured circuit input tree.
///
/// Each variant holds actual data (not type metadata). The nesting mirrors
/// `ZinniaType` but uses distinct names to avoid confusion:
///
/// | ZinniaType variant | InputNode variant |
/// |--------------------|-------------------|
/// | `Integer` | `Int(i64)` |
/// | `Float` | `Float(f64)` |
/// | `Boolean` | `Bool(bool)` |
/// | `String` | `Hex(String)` |
/// | `NDArray` / `DynamicNDArray` | `Array { .. }` |
/// | `List` / `Tuple` | `Sequence(Vec<..>)` |
/// | `PoseidonHashed` | `Hashed { .. }` |
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InputNode {
    /// A scalar integer value.
    Int(i64),
    /// A scalar float value.
    Float(f64),
    /// A boolean value.
    Bool(bool),
    /// A hex-encoded field element string (used for Poseidon hashes).
    Hex(String),
    /// A flat array of elements with known shape (NDArray or DynamicNDArray).
    Array {
        shape: Vec<usize>,
        elements: Vec<InputNode>,
    },
    /// An ordered sequence of heterogeneous elements (List or Tuple).
    Sequence(Vec<InputNode>),
    /// A Poseidon-hashed wrapper: inner value + precomputed hash.
    Hashed {
        inner: Box<InputNode>,
        hash: String,
    },
}

impl InputNode {
    /// Navigate into this node by following path segments.
    ///
    /// Returns a reference to the leaf node, or an error if the path
    /// does not match the structure.
    pub fn navigate(&self, segments: &[PathSegment]) -> Result<&InputNode, ProvingError> {
        if segments.is_empty() {
            return Ok(self);
        }
        match (&segments[0], self) {
            (PathSegment::Index(i), InputNode::Sequence(elems)) => {
                let idx = *i as usize;
                elems.get(idx)
                    .ok_or_else(|| ProvingError::other(format!(
                        "Sequence index {} out of range (len {})", idx, elems.len()
                    )))?
                    .navigate(&segments[1..])
            }
            (PathSegment::Index(i), InputNode::Array { elements, .. }) => {
                let idx = *i as usize;
                elements.get(idx)
                    .ok_or_else(|| ProvingError::other(format!(
                        "Array index {} out of range (len {})", idx, elements.len()
                    )))?
                    .navigate(&segments[1..])
            }
            (PathSegment::Inner, InputNode::Hashed { inner, .. }) => {
                inner.navigate(&segments[1..])
            }
            (PathSegment::Hash, InputNode::Hashed { .. }) => {
                if segments.len() > 1 {
                    return Err(ProvingError::other(
                        "Hash segment must be terminal (no further navigation)"
                    ));
                }
                Ok(self)
            }
            (seg, node) => {
                Err(ProvingError::other(format!(
                    "Cannot navigate {:?} into {:?}", seg, std::mem::discriminant(node)
                )))
            }
        }
    }

    /// Convert a leaf node to an Fp field element.
    pub fn to_fp(&self, precision_bits: u32) -> Result<pasta_curves::Fp, ProvingError> {
        use pasta_curves::Fp;
        match self {
            InputNode::Int(v) => Ok(kernel::i64_to_fp(*v)),
            InputNode::Float(v) => Ok(kernel::quantize_to_fp(*v, precision_bits)),
            InputNode::Bool(v) => Ok(if *v { Fp::one() } else { Fp::zero() }),
            InputNode::Hex(s) => kernel::hex_str_to_fp(s),
            InputNode::Hashed { hash, .. } => kernel::hex_str_to_fp(hash),
            InputNode::Array { .. } | InputNode::Sequence(_) => {
                Err(ProvingError::other("Cannot convert composite InputNode to Fp"))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// InputParam / CircuitInputs — top-level input structure
// ---------------------------------------------------------------------------

/// A single circuit input parameter with metadata and its value tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputParam {
    /// Parameter name from the circuit definition (e.g., `"x"`, `"arr"`).
    pub name: String,
    /// Whether this parameter is public (exposed as instance column).
    pub is_public: bool,
    /// Type descriptor in ZinniaType serde form.
    pub dtype: serde_json::Value,
    /// The actual input data tree.
    pub value: InputNode,
}

/// All circuit input parameters, structured and self-describing.
///
/// This is what Python serializes to JSON and Rust deserializes in `prove_circuit`.
/// It represents the user-provided inputs, NOT the full ZK witness (which also
/// includes intermediate computation values and external function results).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitInputs {
    pub params: Vec<InputParam>,
}

impl CircuitInputs {
    pub fn new() -> Self {
        Self { params: Vec::new() }
    }

    /// Resolve an [`InputPath`] to an Fp field element by navigating the tree.
    pub fn resolve(&self, path: &InputPath, precision_bits: u32) -> Result<pasta_curves::Fp, ProvingError> {
        let param = self.params.iter()
            .find(|p| p.name == path.param)
            .ok_or_else(|| ProvingError::witness_missing(&path.display()))?;
        let leaf = param.value.navigate(&path.segments)?;
        leaf.to_fp(precision_bits)
    }
}

impl Default for CircuitInputs {
    fn default() -> Self { Self::new() }
}

// ---------------------------------------------------------------------------
// ResolvedWitness — the full prover witness, output of preprocessing
// ---------------------------------------------------------------------------

/// The full prover witness: structured circuit inputs plus external function results.
///
/// This is what synthesizers consume. Circuit inputs are resolved by navigating
/// the [`CircuitInputs`] tree. External function results are stored separately
/// in a flat map keyed by `(store_idx, output_idx)`.
///
/// Named "witness" deliberately — unlike [`CircuitInputs`] (which is just the
/// user-provided inputs), this struct represents the complete set of values the
/// prover needs, which is the standard ZK meaning of "witness".
#[derive(Debug, Clone)]
pub struct ResolvedWitness {
    /// The structured circuit inputs.
    pub input: CircuitInputs,
    /// External function results, keyed by (store_idx, output_idx).
    pub external_results: std::collections::HashMap<(u32, u32), pasta_curves::Fp>,
    /// Precision bits for fixed-point quantization.
    pub precision_bits: u32,
}

impl ResolvedWitness {
    pub fn new(input: CircuitInputs, precision_bits: u32) -> Self {
        Self {
            input,
            external_results: std::collections::HashMap::new(),
            precision_bits,
        }
    }

    /// Resolve a circuit input path to Fp.
    pub fn resolve_path(&self, path: &InputPath) -> Result<pasta_curves::Fp, ProvingError> {
        self.input.resolve(path, self.precision_bits)
    }

    /// Look up an external function result.
    pub fn resolve_external(&self, store_idx: u32, output_idx: u32) -> Result<pasta_curves::Fp, ProvingError> {
        self.external_results.get(&(store_idx, output_idx)).copied()
            .ok_or_else(|| ProvingError::witness_missing(
                &format!("__ext.{}.{}", store_idx, output_idx)
            ))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_input_path_display() {
        assert_eq!(InputPath::new("x", vec![]).display(), "x");
        assert_eq!(InputPath::new("arr", vec![PathSegment::Index(5)]).display(), "arr.5");
        assert_eq!(
            InputPath::new("h", vec![PathSegment::Inner, PathSegment::Index(0)]).display(),
            "h.inner.0"
        );
        assert_eq!(InputPath::new("h", vec![PathSegment::Hash]).display(), "h.hash");
    }

    #[test]
    fn test_navigate_scalar() {
        let node = InputNode::Int(42);
        let result = node.navigate(&[]).unwrap();
        assert!(matches!(result, InputNode::Int(42)));
    }

    #[test]
    fn test_navigate_sequence() {
        let node = InputNode::Sequence(vec![
            InputNode::Int(10),
            InputNode::Int(20),
            InputNode::Int(30),
        ]);
        let result = node.navigate(&[PathSegment::Index(1)]).unwrap();
        assert!(matches!(result, InputNode::Int(20)));
    }

    #[test]
    fn test_navigate_array() {
        let node = InputNode::Array {
            shape: vec![3],
            elements: vec![InputNode::Int(1), InputNode::Int(2), InputNode::Int(3)],
        };
        let result = node.navigate(&[PathSegment::Index(2)]).unwrap();
        assert!(matches!(result, InputNode::Int(3)));
    }

    #[test]
    fn test_navigate_hashed() {
        let node = InputNode::Hashed {
            inner: Box::new(InputNode::Int(42)),
            hash: "a".repeat(64),
        };
        // Navigate to inner
        let inner = node.navigate(&[PathSegment::Inner]).unwrap();
        assert!(matches!(inner, InputNode::Int(42)));
        // Navigate to hash (terminal)
        let hash = node.navigate(&[PathSegment::Hash]).unwrap();
        assert!(matches!(hash, InputNode::Hashed { .. }));
    }

    #[test]
    fn test_navigate_nested() {
        let node = InputNode::Sequence(vec![
            InputNode::Array {
                shape: vec![2],
                elements: vec![InputNode::Int(100), InputNode::Int(200)],
            },
            InputNode::Int(300),
        ]);
        let result = node.navigate(&[PathSegment::Index(0), PathSegment::Index(1)]).unwrap();
        assert!(matches!(result, InputNode::Int(200)));
    }

    #[test]
    fn test_navigate_index_out_of_range() {
        let node = InputNode::Sequence(vec![InputNode::Int(1)]);
        assert!(node.navigate(&[PathSegment::Index(5)]).is_err());
    }

    #[test]
    fn test_navigate_type_mismatch() {
        let node = InputNode::Int(42);
        assert!(node.navigate(&[PathSegment::Index(0)]).is_err());
        assert!(node.navigate(&[PathSegment::Inner]).is_err());
    }

    #[test]
    fn test_navigate_hash_not_terminal() {
        let node = InputNode::Hashed {
            inner: Box::new(InputNode::Int(1)),
            hash: "a".repeat(64),
        };
        assert!(node.navigate(&[PathSegment::Hash, PathSegment::Index(0)]).is_err());
    }

    #[test]
    fn test_to_fp_integer() {
        let node = InputNode::Int(42);
        let fp = node.to_fp(32).unwrap();
        assert_eq!(kernel::fp_to_i64(fp), 42);
    }

    #[test]
    fn test_to_fp_negative() {
        let node = InputNode::Int(-7);
        let fp = node.to_fp(32).unwrap();
        assert_eq!(kernel::fp_to_i64(fp), -7);
    }

    #[test]
    fn test_to_fp_bool() {
        assert_eq!(kernel::fp_to_i64(InputNode::Bool(true).to_fp(32).unwrap()), 1);
        assert_eq!(kernel::fp_to_i64(InputNode::Bool(false).to_fp(32).unwrap()), 0);
    }

    #[test]
    fn test_to_fp_composite_fails() {
        let node = InputNode::Sequence(vec![InputNode::Int(1)]);
        assert!(node.to_fp(32).is_err());
    }

    #[test]
    fn test_circuit_inputs_resolve() {
        let inputs = CircuitInputs {
            params: vec![
                InputParam {
                    name: "x".to_string(),
                    is_public: false,
                    dtype: serde_json::json!("Integer"),
                    value: InputNode::Int(99),
                },
                InputParam {
                    name: "arr".to_string(),
                    is_public: true,
                    dtype: serde_json::json!({"NDArray": {"shape": [3], "dtype": "Integer"}}),
                    value: InputNode::Array {
                        shape: vec![3],
                        elements: vec![InputNode::Int(10), InputNode::Int(20), InputNode::Int(30)],
                    },
                },
            ],
        };
        // Resolve scalar
        let fp = inputs.resolve(&InputPath::new("x", vec![]), 32).unwrap();
        assert_eq!(kernel::fp_to_i64(fp), 99);
        // Resolve array element
        let fp = inputs.resolve(&InputPath::new("arr", vec![PathSegment::Index(2)]), 32).unwrap();
        assert_eq!(kernel::fp_to_i64(fp), 30);
    }

    #[test]
    fn test_circuit_inputs_resolve_missing_param() {
        let inputs = CircuitInputs::new();
        assert!(inputs.resolve(&InputPath::new("nonexistent", vec![]), 32).is_err());
    }

    #[test]
    fn test_resolved_witness_external() {
        let mut rw = ResolvedWitness::new(CircuitInputs::new(), 32);
        rw.external_results.insert((1, 0), kernel::i64_to_fp(42));
        let fp = rw.resolve_external(1, 0).unwrap();
        assert_eq!(kernel::fp_to_i64(fp), 42);
        assert!(rw.resolve_external(99, 0).is_err());
    }

    #[test]
    fn test_serde_round_trip() {
        let inputs = CircuitInputs {
            params: vec![InputParam {
                name: "x".to_string(),
                is_public: true,
                dtype: serde_json::json!("Integer"),
                value: InputNode::Int(42),
            }],
        };
        let json = serde_json::to_string(&inputs).unwrap();
        let deserialized: CircuitInputs = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.params.len(), 1);
        assert_eq!(deserialized.params[0].name, "x");
    }
}
