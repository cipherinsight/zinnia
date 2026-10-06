//! How kernel inputs become `fn main` parameters of the emitted Noir circuit.
//!
//! Every kernel parameter is one `fn main` parameter: a single `Field` for a
//! scalar leaf, or a flat `[Field; N]` holding the parameter's scalar leaves in
//! the order of the witness tree (row-major for arrays; a complex number is the
//! pair `(real, imag)`, so a complex array interleaves them). Two things are
//! decided here, both from the parameter's type structure only (never from
//! input values, so the circuit text is the same for every input):
//!
//! - the Noir **identifier** of the parameter ([`noir_param_identifier`]), and
//! - the **leaf layout** ([`LeafLayout`]): where the scalar leaf an IR read
//!   addresses (an [`InputPath`]) sits in the flat parameter.

use crate::circuit_input::{InputNode, InputPath, PathSegment};

/// The prefix that keeps a kernel parameter name apart from every name the
/// emitter generates (see [`noir_param_identifier`]).
const ESCAPE_PREFIX: &str = "param_";

/// Names a kernel parameter may not take in `main.nr` as is: Noir keywords,
/// built-in type names and the module paths the circuit refers to.
const RESERVED: &[&str] = &[
    "as", "assert", "assert_eq", "bool", "break", "call_data", "char", "comptime", "constrain",
    "continue", "contract", "crate", "dep", "else", "enum", "false", "Field", "fmtstr", "fn",
    "for", "global", "if", "impl", "in", "let", "loop", "main", "match", "mod", "mut", "noir_fpc",
    "oracle", "pub", "quote", "return", "return_data", "self", "Self", "std", "str", "struct",
    "super", "trait", "true", "type", "unchecked", "unconstrained", "unsafe", "use", "where",
    "while",
];

/// The `fn main` parameter (and `Prover.toml` key) of the kernel parameter
/// `name`.
///
/// The emitter names its own values `v<N>` (`NoirEmitter::fresh`), so a kernel
/// parameter called `v2` would be shadowed by, or shadow, the circuit's value
/// `v2` (npbench gemver has parameters `u1, v1, u2, v2`). Such a name, a Noir
/// keyword or built-in name ([`RESERVED`]) and an integer type name
/// (`u8`, `i64`, ...) get [`ESCAPE_PREFIX`] in front; so does every name that
/// already starts with the prefix. That keeps the map one-to-one (a name
/// starting with the prefix is always an escaped one, and stripping the prefix
/// recovers the kernel name) and disjoint from the emitter's names, which
/// never start with it. Every other name is kept as is.
pub fn noir_param_identifier(name: &str) -> String {
    if needs_escape(name) {
        format!("{ESCAPE_PREFIX}{name}")
    } else {
        name.to_string()
    }
}

fn needs_escape(name: &str) -> bool {
    let is_digits = |s: &str| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit());
    let generated_value = name.strip_prefix('v').is_some_and(is_digits);
    let integer_type = name.strip_prefix(['u', 'i']).is_some_and(is_digits);
    generated_value || integer_type || RESERVED.contains(&name) || name.starts_with(ESCAPE_PREFIX)
}

/// The shape of an input's witness tree, reduced to what places its scalar
/// leaves in the flat `fn main` parameter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LeafLayout {
    /// One scalar (int, float, bool, hash string, Poseidon-hashed value).
    Leaf,
    /// `count` elements of the same layout (an array of scalars, or of
    /// complex numbers).
    Repeated { count: usize, element: Box<LeafLayout> },
    /// Elements of different layouts (a tuple, a list, a complex number).
    Fields(Vec<LeafLayout>),
}

impl LeafLayout {
    /// The layout of a witness tree. A `Hashed` node is one leaf: the circuit
    /// reads its hash only (`flat_leaves` writes it as one value).
    pub fn of(node: &InputNode) -> Self {
        match node {
            InputNode::Array { elements, .. } => Self::of_elements(elements),
            InputNode::Sequence(items) => Self::Fields(items.iter().map(Self::of).collect()),
            _ => Self::Leaf,
        }
    }

    fn of_elements(elements: &[InputNode]) -> Self {
        let layouts: Vec<Self> = elements.iter().map(Self::of).collect();
        match layouts.first() {
            Some(first) if layouts.iter().all(|l| l == first) => {
                Self::Repeated { count: layouts.len(), element: Box::new(first.clone()) }
            }
            None => Self::Repeated { count: 0, element: Box::new(Self::Leaf) },
            Some(_) => Self::Fields(layouts),
        }
    }

    /// Number of scalar leaves.
    pub fn leaves(&self) -> usize {
        match self {
            Self::Leaf => 1,
            Self::Repeated { count, element } => count * element.leaves(),
            Self::Fields(fields) => fields.iter().map(Self::leaves).sum(),
        }
    }

    /// Position, in the flat parameter, of the scalar leaf `segments`
    /// addresses; `None` when the path does not end on a leaf of this layout.
    pub fn flat_offset(&self, segments: &[PathSegment]) -> Option<usize> {
        let Some((first, rest)) = segments.split_first() else {
            return matches!(self, Self::Leaf).then_some(0);
        };
        let i = match first {
            PathSegment::Index(i) => i,
            // The hash of a Poseidon-hashed input is its one leaf.
            PathSegment::Hash if rest.is_empty() && matches!(self, Self::Leaf) => return Some(0),
            _ => return None,
        };
        let i = *i as usize;
        match self {
            Self::Leaf => None,
            Self::Repeated { count, element } => {
                (i < *count).then_some(())?;
                Some(i * element.leaves() + element.flat_offset(rest)?)
            }
            Self::Fields(fields) => {
                let before: usize = fields.get(..i)?.iter().map(Self::leaves).sum();
                Some(before + fields.get(i)?.flat_offset(rest)?)
            }
        }
    }
}

/// Where an IR read of `path` finds its value in a parameter of the given
/// layout, as a Noir expression: the parameter itself for a scalar, else the
/// flat element.
pub fn leaf_expr(identifier: &str, layout: &LeafLayout, path: &InputPath) -> Option<String> {
    let offset = layout.flat_offset(&path.segments)?;
    Some(match layout {
        LeafLayout::Leaf => identifier.to_string(),
        _ => format!("{identifier}[{offset}]"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn complex(re: f64, im: f64) -> InputNode {
        InputNode::Sequence(vec![InputNode::Float(re), InputNode::Float(im)])
    }

    fn path(segments: &[u32]) -> Vec<PathSegment> {
        segments.iter().map(|i| PathSegment::Index(*i)).collect()
    }

    #[test]
    fn generated_names_and_keywords_are_escaped_one_to_one() {
        // `u1` is also a Noir type name.
        for name in ["v2", "v0", "v1", "u1", "u8", "i64", "type", "fn", "std", "main", "param_x", "param_"] {
            assert_eq!(noir_param_identifier(name), format!("param_{name}"));
        }
        for name in ["v", "va", "v2x", "x", "A", "alpha", "_max", "u", "w", "params", "u2x"] {
            assert_eq!(noir_param_identifier(name), name);
        }
        // One-to-one on names that collide after escaping.
        assert_ne!(noir_param_identifier("v2"), noir_param_identifier("param_v2"));
    }

    #[test]
    fn complex_scalar_and_array_leaves_are_interleaved() {
        let scalar = LeafLayout::of(&complex(1.0, 2.0));
        assert_eq!(scalar.leaves(), 2);
        assert_eq!(scalar.flat_offset(&path(&[1])), Some(1));
        assert_eq!(scalar.flat_offset(&path(&[])), None);

        let array = InputNode::Array {
            shape: vec![2, 3],
            elements: (0..6).map(|k| complex(k as f64, -(k as f64))).collect(),
            runtime_shape: None,
        };
        let layout = LeafLayout::of(&array);
        assert_eq!(layout.leaves(), 12);
        // Element 4 (row 1, column 1), imaginary part.
        assert_eq!(layout.flat_offset(&path(&[4, 1])), Some(9));
        assert_eq!(layout.flat_offset(&path(&[6, 0])), None);
        assert_eq!(layout.flat_offset(&path(&[4])), None);
    }

    #[test]
    fn scalar_arrays_keep_their_flat_index() {
        let array = InputNode::Array {
            shape: vec![4],
            elements: (0..4).map(InputNode::Int).collect(),
            runtime_shape: None,
        };
        let layout = LeafLayout::of(&array);
        let read = InputPath::new("x", path(&[3]));
        assert_eq!(leaf_expr("x", &layout, &read).as_deref(), Some("x[3]"));
        let scalar = InputPath::new("n", vec![]);
        assert_eq!(leaf_expr("n", &LeafLayout::Leaf, &scalar).as_deref(), Some("n"));
    }

    #[test]
    fn heterogeneous_sequences_sum_the_preceding_leaves() {
        let tuple = InputNode::Sequence(vec![
            InputNode::Int(1),
            complex(0.0, 1.0),
            InputNode::Array { shape: vec![2], elements: vec![InputNode::Int(3), InputNode::Int(4)], runtime_shape: None },
        ]);
        let layout = LeafLayout::of(&tuple);
        assert_eq!(layout.leaves(), 5);
        assert_eq!(layout.flat_offset(&path(&[1, 1])), Some(2));
        assert_eq!(layout.flat_offset(&path(&[2, 1])), Some(4));
    }
}
