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
    /// NumPy's `argmax` (`max`) or `argmin` of the shared operands (at
    /// least two), compared as floats when `float`: the first position of
    /// the largest (smallest) value. One lane, output 0 at position 0
    /// (`sort::arg_extreme`).
    ArgExtreme { max: bool, float: bool },
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
    /// The reads of one line of known cells: `ArrayRead`s of an `ArrayInit`
    /// (`lower::arrays`), `x[i]` of the shared cells at the lane's own
    /// operand `i`, which is in `[0, n)` (the frontend checked it). A group
    /// kind only: no placeholder has it (the reads are `IR::ArrayRead`s).
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
    /// The cells of an array after a masked write `a[mask] = v`: no shared
    /// operand; each lane is one cell, own operands `[old, written, mask]`
    /// (the mask bit of that cell). Every masked write of the program is one
    /// group (a chain of writes over one cell is lowered together).
    MaskWrite,
    // ── linear family (`lower::linear`) ───────────────────────────────
    /// The full float correlation of `x` (`long` values) and `y` (`short`),
    /// the shared operands in that order: lane `k` in `[0, long + short -
    /// 1)` is `Σ_j x[j + k - (short - 1)]·y[j]` over the `j` that index
    /// `x`, rounded once (`ProductSumF`).
    Correlation { long: u32, short: u32 },
    /// The float product of a `rows × inner` and an `inner × cols` matrix
    /// (row-major, in that order): lane `i·cols + j` is `Σ_l A[i, l]·B[l,
    /// j]`, rounded once (`IRBuilder::ir_dot_f`).
    MatMul { rows: u32, inner: u32, cols: u32 },
    // ── permutation check (`lower::permutation`) ──────────────────────
    /// `is_permutation(p)` of the shared integers: one lane (output 0,
    /// position 0) whose value is 1; no proof unless `p` is a permutation
    /// of `0..n` (the frontend asserts the lane).
    IsPermutation,
    // ── memory family (`lower::memory`) ───────────────────────────────
    /// NumPy's position of a runtime index on an axis of length `len`: the
    /// shared operand is the index `i`; output 0 (position 0) is the
    /// position `w` (`i`, or `i + len` for a negative `i`), output 1 at
    /// position `k` the bit `[w == k]`. An `i` outside `[-len, len)` has no
    /// proof (NumPy raises; the program checks `0 <= w < len`).
    Wrap { len: u32 },
    /// The version tree of an array the program writes at runtime
    /// (`lower::memory::versions`): a group kind only, whose lanes are the
    /// tree's reads at constant addresses. No placeholder has it.
    Versions,
}

impl OpKind {
    /// The operator's semantics, implemented by its family.
    pub fn semantics(self) -> &'static dyn OperatorSemantics {
        match self {
            OpKind::Sort => &super::sort::SortSemantics,
            OpKind::SearchSorted { .. } => &super::sort::SearchSortedSemantics,
            OpKind::ArgExtreme { .. } => &super::sort::arg_extreme::ArgExtremeSemantics,
            OpKind::Corrcoef { .. } => &super::stats::CorrcoefSemantics,
            OpKind::FirstSet
            | OpKind::LastSet
            | OpKind::ArgmaxOfFlags
            | OpKind::Window { .. }
            | OpKind::Element
            | OpKind::PermutationScatter => &super::gather::GatherSemantics,
            OpKind::Insert { .. } => &super::scatter::InsertSemantics,
            OpKind::MaskWrite => &super::scatter::MaskWriteSemantics,
            OpKind::Correlation { .. } | OpKind::MatMul { .. } => &super::linear::LinearSemantics,
            OpKind::IsPermutation => &super::permutation::IsPermutationSemantics,
            OpKind::Wrap { .. } => &super::memory::wrap::WrapSemantics,
            OpKind::Versions => unreachable!("a memory group has no placeholder"),
        }
    }

    /// The family of operators this kind belongs to: the kinds whose groups
    /// choose among the same constructions. Measurement only (the layer
    /// report and the fixed configurations of `docs/ablations.md`).
    pub fn family(self) -> &'static str {
        match self {
            OpKind::Sort => "sort",
            OpKind::SearchSorted { .. } => "searchsorted",
            OpKind::ArgExtreme { .. } => "argextreme",
            OpKind::Corrcoef { .. } => "corrcoef",
            OpKind::FirstSet | OpKind::LastSet | OpKind::ArgmaxOfFlags => "scan",
            OpKind::Window { .. } => "window",
            OpKind::Element => "element",
            OpKind::PermutationScatter => "permutation scatter",
            OpKind::Insert { .. } => "insert",
            OpKind::MaskWrite => "mask write",
            OpKind::Correlation { .. } | OpKind::MatMul { .. } => "linear",
            OpKind::IsPermutation => "is_permutation",
            OpKind::Wrap { .. } => "wrap",
            OpKind::Versions => "memory",
        }
    }

    /// A short name for printing the IR.
    pub fn name(self) -> &'static str {
        match self {
            OpKind::Sort => "sort",
            OpKind::SearchSorted { right: true } => "searchsorted_right",
            OpKind::SearchSorted { right: false } => "searchsorted_left",
            OpKind::ArgExtreme { max: true, .. } => "argmax",
            OpKind::ArgExtreme { max: false, .. } => "argmin",
            OpKind::Corrcoef { .. } => "corrcoef",
            OpKind::FirstSet => "first_set",
            OpKind::LastSet => "last_set",
            OpKind::ArgmaxOfFlags => "argmax_of_flags",
            OpKind::Window { .. } => "window",
            OpKind::Element => "element",
            OpKind::PermutationScatter => "permutation_scatter",
            OpKind::Insert { .. } => "insert",
            OpKind::MaskWrite => "mask_write",
            OpKind::Correlation { .. } => "correlation",
            OpKind::MatMul { .. } => "matmul",
            OpKind::IsPermutation => "is_permutation",
            OpKind::Wrap { .. } => "wrap",
            OpKind::Versions => "versions",
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
