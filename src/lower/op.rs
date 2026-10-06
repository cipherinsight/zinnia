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
    /// `np.searchsorted(table, t, side)` for a non-decreasing `table` (the
    /// shared operands, the producer guarantees the order) and one query
    /// `t` per lane (the lane's one own operand): the number of entries
    /// `<= t` (`right`) or `< t` (left). Output 0, position `None`.
    SearchSorted { right: bool },
    // ── statistics family (`lower::stats`) ────────────────────────────
    /// Pearson's `r` of `np.corrcoef(x, y)` (its off-diagonal entry) for two
    /// 1-D arrays of `n` elements. Shared operands: `x`, then `y`, then (when
    /// `guarded`) the execution predicate under which the call runs.
    /// `integer`: every element is an integer or a Boolean. One lane
    /// (output 0, position 0). No proof for a zero variance (NumPy: nan).
    Corrcoef { n: u32, integer: bool, guarded: bool },
    // ── gather family (`lower::gather`) ───────────────────────────────
    /// The first position whose shared 0/1 flag is set (`min` of
    /// `np.flatnonzero`, of an `np.argwhere` coordinate). One lane, at
    /// position 0. The frontend asserts that some flag is set.
    FirstSet,
    /// The last position whose flag is set (`max`).
    LastSet,
    /// NumPy's `argmax` of the shared 0/1 flags: the first set position, 0
    /// when none is set.
    ArgmaxOfFlags,
    /// The window `x[s_0:e_0, ...]` (step 1 on every axis) of a row-major
    /// array of shape `dims[..rank]`, at the top-left of a zero buffer of
    /// that shape. Shared operands: the cells, then `(start, stop)` per
    /// axis, each start in `[0, dims[a]]`. Lane `p` is the buffer's cell `p`.
    Window { rank: u8, dims: [u32; 3] },
    /// `x[i]` of the shared cells at the lane's own operand `i`, which is in
    /// `[0, n)` (the frontend checked it).
    Element,
    /// `c[p] = v` for a permutation `p` of `0..n`: lane `j` is `c[j]`, the
    /// `v[k]` with `p[k] == j`. Shared operands: `p`, then `v` (n each). The
    /// frontend emits it only where a precondition asserts `p` is a
    /// permutation.
    PermutationScatter,
    // ── scatter family (`lower::scatter`) ─────────────────────────────
    /// `np.insert` of one slice before a runtime position along an axis of
    /// `rows` slices, each of `lanes` cells. Shared operands: the source
    /// cells (`rows * lanes`, row-major by slice), the inserted slice
    /// (`lanes` cells) and the position. Output 0 has `(rows + 1) * lanes`
    /// cells, row-major by slice. A position outside `[-rows, rows]` has no
    /// proof (NumPy raises).
    Insert { rows: u32, lanes: u32 },
    /// The cells of an array after `a[..., i, ...] = v` at a runtime index
    /// `i` on an axis of length `axis_len` (the other axes static). Shared
    /// operand: the index (guarded by the write's execution predicate). Each
    /// lane is one cell the write may touch: own operands `[old, written]`,
    /// then the execution predicate when the write is conditional; its
    /// position is the cell's coordinate on the indexed axis. An index
    /// outside `[-axis_len, axis_len)` has no proof (NumPy raises).
    IndexWrite { axis_len: u32 },
    /// The cells of an array after a masked write `a[mask] = v`: no shared
    /// operand; each lane is one cell, own operands `[old, written, mask]`
    /// (the mask bit of that cell). Every masked write of the program is one
    /// group (a chain of writes over one cell is lowered together).
    MaskWrite,
    // ── memory family (`lower::memory`) ───────────────────────────────
    /// NumPy's position of a runtime index on an axis of length `len`: the
    /// shared operand is the index `i`; output 0 (position 0) is the
    /// position `w` (`i`, or `i + len` for a negative `i`), output 1 at
    /// position `k` the bit `[w == k]`. An `i` outside `[-len, len)` has no
    /// proof (NumPy raises; the program checks `0 <= w < len`).
    Wrap { len: u32 },
}

impl OpKind {
    /// The operator's semantics, implemented by its family.
    pub fn semantics(self) -> &'static dyn OperatorSemantics {
        match self {
            OpKind::Sort => &super::sort::SortSemantics,
            OpKind::SearchSorted { .. } => &super::sort::SearchSortedSemantics,
            OpKind::Corrcoef { .. } => &super::stats::CorrcoefSemantics,
            OpKind::FirstSet
            | OpKind::LastSet
            | OpKind::ArgmaxOfFlags
            | OpKind::Window { .. }
            | OpKind::Element
            | OpKind::PermutationScatter => &super::gather::GatherSemantics,
            OpKind::Insert { .. } => &super::scatter::InsertSemantics,
            OpKind::IndexWrite { .. } => &super::scatter::IndexWriteSemantics,
            OpKind::MaskWrite => &super::scatter::MaskWriteSemantics,
            OpKind::Wrap { .. } => &super::memory::wrap::WrapSemantics,
        }
    }

    /// A short name for printing the IR.
    pub fn name(self) -> &'static str {
        match self {
            OpKind::Sort => "sort",
            OpKind::SearchSorted { right: true } => "searchsorted_right",
            OpKind::SearchSorted { right: false } => "searchsorted_left",
            OpKind::Corrcoef { .. } => "corrcoef",
            OpKind::FirstSet => "first_set",
            OpKind::LastSet => "last_set",
            OpKind::ArgmaxOfFlags => "argmax_of_flags",
            OpKind::Window { .. } => "window",
            OpKind::Element => "element",
            OpKind::PermutationScatter => "permutation_scatter",
            OpKind::Insert { .. } => "insert",
            OpKind::IndexWrite { .. } => "index_write",
            OpKind::MaskWrite => "mask_write",
            OpKind::Wrap { .. } => "wrap",
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
