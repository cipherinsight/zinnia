//! The array-operator placeholder: one IR statement per live output lane
//! of an array operator (`IR::ArrayOp`), and what the rest of the compiler
//! may know about it before the layer lowers it.
//!
//! An operator instance (one `np.sort`, one crop, one `np.insert`) emits one
//! placeholder per output element it produces. Every placeholder of the
//! instance carries the same `kind` and the same leading `shared` operands
//! (the operator's inputs); together they identify the instance, which the
//! layer calls a *group*. The operands after the shared ones belong to the
//! lane alone (a runtime rank, a runtime index). Dead-code elimination
//! removes the placeholders nothing reads, so when the layer runs a group's
//! remaining placeholders are exactly the operator's live outputs.
//!
//! Before the layer runs, passes see a placeholder as an opaque pure value.
//! What they may know about it (its constant value when its operands are
//! constants, its bounds, the operands it is one of) comes from the
//! operator's [`OperatorSemantics`], which its family implements next to
//! its constructions. No backend accepts a placeholder.

use std::ops::Range;

use serde::{Deserialize, Serialize};

use crate::provenance::Interval;
use crate::types::{NumberType, Value};

/// One output lane of an array operator (the payload of `IR::ArrayOp`).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ArrayOp {
    /// The operator and its static parameters (shapes, axes), the same on
    /// every lane of one instance.
    pub kind: OpKind,
    /// How many leading operands are the instance's inputs, shared by every
    /// lane; the rest are this lane's own operands.
    pub shared: u32,
    /// Which output of the operator the lane belongs to (`np.sort`: 0 the
    /// sorted values, 1 the stable argsort).
    pub output: u8,
    /// The lane's position in that output, or `None` when this lane's own
    /// operands give it at runtime (a read of the output at a runtime
    /// index).
    pub position: Option<u32>,
}

/// The array operators the layer lowers. Each family adds its operators in
/// its own section; static parameters are fields of the variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum OpKind {
    // ── sort family (`lower::sort`) ───────────────────────────────────
    /// NumPy's stable ascending sort of the shared operands. Output 0 is
    /// the sorted values, output 1 the stable argsort (original positions,
    /// ties in original order). A runtime position is the lane's one own
    /// operand; one outside `[0, n)` has no proof (NumPy raises).
    Sort,
    // ── gather family (`lower::gather`) ───────────────────────────────
    // ── scatter family (`lower::scatter`) ─────────────────────────────
}

impl OpKind {
    /// The operator's semantics, implemented by its family.
    pub fn semantics(self) -> &'static dyn OperatorSemantics {
        match self {
            OpKind::Sort => &super::sort::SortSemantics,
        }
    }

    /// A short name for printing the IR.
    pub fn name(self) -> &'static str {
        match self {
            OpKind::Sort => "sort",
        }
    }
}

/// A compile-time constant lane value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Constant {
    Int(i64),
    Float(f64),
    Bool(bool),
}

/// What passes that run before the layer may know about a lane. Every
/// answer must hold for the value every construction of the operator
/// produces (they all compute NumPy's result).
pub trait OperatorSemantics: Sync {
    /// The lane's value when the operands it depends on are constants
    /// (NumPy's result), or `None`. `args` are all the lane's operands.
    fn fold(&self, op: &ArrayOp, args: &[Value]) -> Option<Constant>;

    /// The number type of the lane's value.
    fn number_type(&self, op: &ArrayOp, args: &[Value]) -> NumberType;

    /// Bounds of the lane's value from bounds of its operands (`None` where
    /// unknown), or `None`.
    fn bounds(&self, op: &ArrayOp, operands: &[Option<Interval>]) -> Option<Interval>;

    /// Operands the lane's value is always one of (a sorted value is one of
    /// the elements; a gathered value one of the source's), for analyses
    /// that bound widths. `None` when the value may be anything else.
    fn picks_from(&self, op: &ArrayOp) -> Option<Range<usize>>;
}

impl ArrayOp {
    /// The `IR::ArrayOp` statement's printed form.
    pub fn describe(&self) -> String {
        let position = self.position.map_or("runtime".to_string(), |p| p.to_string());
        format!("array_op[{}][{}][{}][{}]", self.kind.name(), self.shared, self.output, position)
    }
}
