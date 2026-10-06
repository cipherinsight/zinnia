//! The linear family: sums of products whose terms follow an algebraic
//! structure a verifier challenge can check far more cheaply than the
//! products can be computed.
//!
//! Operators (`OpKind`):
//! * `Correlation { long, short }`: the full correlation of `x` (`long`
//!   values) and `y` (`short`), lane `k` (in `[0, long + short - 1)`)
//!   `Σ_j x[j + k - (short - 1)]·y[j]` over the `j` that index `x`, rounded
//!   once (`ProductSumF`). `np.correlate` in every mode and `np.convolve`
//!   (`correlate(longer, shorter[::-1])`) select lanes of it
//!   (`ir_gen::named_attr::numpy_funcs::correlation`; float operands only:
//!   integer ones keep their exact integer sums).
//! * `MatMul { rows, inner, cols }`: the float matrix product `A·B` of a
//!   `rows × inner` and an `inner × cols` matrix (2-D by 2-D, `@`, `np.dot`,
//!   `np.matmul`), lane `i·cols + j` `Σ_l A[i, l]·B[l, j]` rounded once
//!   (`IRBuilder::ir_dot_f`).
//!
//! Constructions:
//! * [`DirectProductSums`]: each live lane its own sum of products, exactly
//!   the IR the frontend emitted before this family existed. Always applies
//!   (the only one on Noir).
//! * [`PolynomialIdentity`] (correlations) and [`Freivalds`] (matrix
//!   products): every raw sum (before its rescale) is a prover hint pinned
//!   by one identity check at a verifier challenge, and each live lane is
//!   the exact rescale of its raw sum. halo2-base only (the program-wide
//!   challenge, `lower::challenge`).
//!
//! **Why the identity constructions are sound.** The direct construction
//! computes each raw sum `R = Σ a·b` in field arithmetic (`ProductSumF`:
//! the exact sum of the raw Q32 products) and rescales it: `floor(R / 2^32)`
//! with the remainder range-checked to 32 bits and the quotient to the
//! signed 64-bit window. The identity constructions rescale a hinted `R'`
//! with the same gadget (`IR::RescaleRawF`) and check the identity among the
//! hinted raw sums and the operands in field arithmetic: `R'` equals the
//! field sum `Σ a·b` for every lane, which is the very value the direct
//! construction rescales. So both accept exactly the same operands and
//! outputs, except with the identity check's probability `degree / |F|`
//! (`Identity::degree`): no bound on the operands is needed for the
//! argument, and the rounding is the same exact rounding. Every hint is
//! unique (each raw sum is determined by the operands).
//!
//! **Completeness.** The honest raw sums satisfy the identity whatever the
//! challenge, and their rescales are the direct construction's values.

mod semantics;
#[cfg(test)]
mod tests;

pub use semantics::LinearSemantics;

use std::sync::Arc;

use crate::builder::IRBuilder;
use crate::ir_defs::{HintKind, Identity};
use crate::types::{StmtId, Value};

use super::{Candidate, Construction, Context, EmitCx, Emitter, Group, Lane, OpKind, Requirement, Target};

/// The family's constructions, in the order the selection breaks ties.
pub fn constructions() -> Vec<&'static dyn Construction> {
    vec![&DirectProductSums, &PolynomialIdentity, &Freivalds]
}

// ── Frontend ────────────────────────────────────────────────────────────

/// The lanes `ks` of the full float correlation of `x` and `y` (see the
/// module doc), as placeholders over one instance. Operands are floats.
pub fn correlation_lanes(b: &mut IRBuilder, x: &[Value], y: &[Value], ks: std::ops::Range<usize>) -> Vec<Value> {
    let kind = OpKind::Correlation { long: x.len() as u32, short: y.len() as u32 };
    let shared: Arc<[Value]> = x.iter().chain(y).map(|v| b.ensure_ptr(v)).collect();
    ks.map(|k| {
        let op = super::ArrayOp { kind, shared: shared.len() as u32, output: 0, position: Some(k as u32) };
        b.ir_array_op(op, &shared, &[])
    })
    .collect()
}

/// The float product of the row-major matrices `a` (`rows × inner`) and `m`
/// (`inner × cols`), as placeholders over one instance, row-major. Operands
/// are floats.
pub fn matmul_lanes(b: &mut IRBuilder, a: &[Value], m: &[Value], rows: usize, inner: usize, cols: usize) -> Vec<Value> {
    assert_eq!((a.len(), m.len()), (rows * inner, inner * cols), "matmul operand shapes");
    let kind = OpKind::MatMul { rows: rows as u32, inner: inner as u32, cols: cols as u32 };
    let shared: Arc<[Value]> = a.iter().chain(m).map(|v| b.ensure_ptr(v)).collect();
    (0..rows * cols)
        .map(|p| {
            let op = super::ArrayOp { kind, shared: shared.len() as u32, output: 0, position: Some(p as u32) };
            b.ir_array_op(op, &shared, &[])
        })
        .collect()
}

// ── The structure of each operator ─────────────────────────────────────

/// The operand pairs of one lane (indices into the shared operands), in the
/// order the frontend sums them.
pub(crate) fn lane_pairs(kind: OpKind, position: usize) -> Vec<(usize, usize)> {
    match kind {
        OpKind::Correlation { long, short } => {
            let (n1, n2, k) = (long as usize, short as usize, position);
            // j with 0 <= j + k - (n2 - 1) < n1.
            let first = (n2 - 1).saturating_sub(k);
            let last = (n1 + n2 - 1 - k).min(n2);
            (first..last).map(|j| (j + k + 1 - n2, n1 + j)).collect()
        }
        OpKind::MatMul { rows, inner, cols } => {
            let (inner, cols) = (inner as usize, cols as usize);
            let (i, j) = (position / cols, position % cols);
            let b0 = rows as usize * inner;
            (0..inner).map(|l| (i * inner + l, b0 + l * cols + j)).collect()
        }
        other => unreachable!("{other:?} is not a linear operator"),
    }
}

/// The number of lanes of the operator (live or not).
fn lane_count(kind: OpKind) -> usize {
    match kind {
        OpKind::Correlation { long, short } => (long + short - 1) as usize,
        OpKind::MatMul { rows, cols, .. } => (rows * cols) as usize,
        other => unreachable!("{other:?} is not a linear operator"),
    }
}

fn is_linear(kind: OpKind) -> bool {
    matches!(kind, OpKind::Correlation { .. } | OpKind::MatMul { .. })
}

// ── Costs ───────────────────────────────────────────────────────────────

/// halo2-base cells (lookup width 8), fitted on forced constructions
/// (`tests::measure_linear_cells`; the shared input checks cancel): a sum of
/// `L` products 3 per product plus its rescale; a hinted raw sum one cell;
/// the identity checks as `prove/halo2_base/identities.rs` lays them out.
mod cost {
    pub const PRODUCT: u64 = 3;
    pub const RESCALE: u64 = 45;
    pub const HINT: u64 = 1;
    /// One Horner step (`mul_add`).
    pub const HORNER_STEP: u64 = 4;
    /// One term of an inner product, and its start.
    pub const INNER_TERM: u64 = 3;
    pub const INNER_START: u64 = 1;
    /// One power of the challenge (`mul`).
    pub const POWER: u64 = 4;
    /// An operand no live lane reads: the identity check still reads it, so
    /// it stays alive (its read cell; an input's window check is a program
    /// assertion, emitted anyway). Measured 5 cells per unchecked input.
    pub const UNREAD_OPERAND: u64 = 5;
    /// Noir gates of a sum of `L` products (unused for selection: the only
    /// construction there).
    pub const NOIR_PRODUCT: u64 = 1;
    pub const NOIR_SUM: u64 = 4;
}

fn direct_cost(group: &Group, target: Target) -> u64 {
    group
        .lanes
        .iter()
        .map(|lane| {
            let terms = lane_pairs(group.kind, lane.position.expect("a linear lane has a position") as usize).len() as u64;
            match target {
                Target::Halo2Base => terms * cost::PRODUCT + cost::RESCALE,
                Target::Noir => terms * cost::NOIR_PRODUCT + cost::NOIR_SUM,
            }
        })
        .sum()
}

/// The identity construction's cells: a hint per raw sum, a rescale per
/// live lane, the second-phase check, and the operands only the check reads.
fn identity_cost(group: &Group) -> u64 {
    let live = group.lanes.len() as u64;
    let read: std::collections::HashSet<usize> = group
        .lanes
        .iter()
        .flat_map(|lane| lane_pairs(group.kind, lane.position.expect("a position") as usize))
        .flat_map(|(x, y)| [x, y])
        .collect();
    let unread = (group.shared.len().max(operand_count(group.kind)) - read.len()) as u64 * cost::UNREAD_OPERAND;
    let hints = lane_count(group.kind) as u64 * cost::HINT;
    let check = match group.kind {
        OpKind::Correlation { long, short } => {
            let (n1, n2) = (u64::from(long), u64::from(short));
            // Horner over a (n1), b (n2) and c (n1 + n2 - 1), and one product.
            (n1 - 1 + n2 - 1 + n1 + n2 - 2) * cost::HORNER_STEP + cost::POWER
        }
        OpKind::MatMul { rows, inner, cols } => {
            let (rows, inner, cols) = (u64::from(rows), u64::from(inner), u64::from(cols));
            let inner_products = inner * (cost::INNER_START + cols * cost::INNER_TERM)
                + rows * (cost::INNER_START + inner * cost::INNER_TERM)
                + rows * (cost::INNER_START + cols * cost::INNER_TERM);
            inner_products + cols.saturating_sub(2) * cost::POWER
        }
        other => unreachable!("{other:?}"),
    };
    hints + live * cost::RESCALE + check + unread
}

/// The number of shared operands of the operator.
fn operand_count(kind: OpKind) -> usize {
    match kind {
        OpKind::Correlation { long, short } => (long + short) as usize,
        OpKind::MatMul { rows, inner, cols } => (rows * inner + inner * cols) as usize,
        other => unreachable!("{other:?} is not a linear operator"),
    }
}

// ── Constructions ───────────────────────────────────────────────────────

/// Each live lane as its own rescaled sum of products: `ProductSumF` for a
/// correlation, `IRBuilder::ir_dot_f` for a matrix product (a constant
/// when every operand is one, `MulF` for one product), as the frontend
/// emitted them. Preconditions: none. Postconditions: the lanes. Cost
/// (halo2-base): `3` per product and `45` per lane. Sound and complete:
/// each lane is the scalar IR of its definition.
pub struct DirectProductSums;

/// A correlation by one polynomial identity (`IR::IdentityCheck` of
/// `Identity::PolynomialProduct`): with `ŷ` the reversed `y`, the full
/// correlation's raw sums are the coefficients of `x(γ)·ŷ(γ)`; every raw
/// sum is hinted (`HintKind::ProductSum` of its pairs), the identity
/// `x(γ)·ŷ(γ) = c(γ)` is checked at a verifier challenge, and each live
/// lane is `RescaleRawF` of its raw sum.
///
/// Preconditions: the verifier challenge. Postconditions: the lanes.
/// Soundness: the module doc; the check's degree is `long + short - 2`.
/// Cost (halo2-base): a hint per coefficient (`long + short - 1`, live or
/// not), a rescale (45) per live lane, Horner steps of 4 cells over the
/// three polynomials' coefficients and one product (4). Against the direct
/// sums (3 cells per product): a full convolution of two length-`N` arrays
/// costs `~13 N` cells of identity against `~3 N^2`.
pub struct PolynomialIdentity;

/// A matrix product by Freivalds' check (`IR::IdentityCheck` of
/// `Identity::MatrixProduct`): every raw entry `R_ij = Σ_l A_il·B_lj` is
/// hinted (`HintKind::ProductSum`), `A·(B·r) = R·r` is checked for `r =
/// (1, γ, ..., γ^(cols-1))` at a verifier challenge, row by row, and each
/// live entry is `RescaleRawF` of its raw sum: the Q32 rescaling and its
/// rounding are checked exactly, outside the identity, which holds exactly
/// on the field integers the circuit computes (the module doc).
///
/// Preconditions: the verifier challenge. Postconditions: the lanes.
/// Soundness: the module doc; the check's degree is `cols - 1` (a nonzero
/// row of `A·B - R` against the powers of `γ`). Cost (halo2-base): a hint
/// per entry, a rescale per live entry, `3` per term of the inner products
/// `B·r` (`inner·cols`), `A·(B·r)` (`rows·inner`) and `R·r` (`rows·cols`),
/// `4` per power: `~9 N^2` against `3 N^3` for `N × N`.
pub struct Freivalds;

impl Construction for DirectProductSums {
    fn name(&self) -> &'static str {
        "linear: direct product sums"
    }

    fn candidates(&self, group: &Group, cx: &Context) -> Vec<Candidate> {
        if !is_linear(group.kind) {
            return Vec::new();
        }
        vec![Candidate {
            pre: Vec::new(),
            post: lanes(group),
            cost: direct_cost(group, cx.target),
            emitter: Box::new(DirectEmitter { kind: group.kind, shared: group.shared.clone() }),
        }]
    }
}

impl Construction for PolynomialIdentity {
    fn name(&self) -> &'static str {
        "linear: polynomial identity"
    }

    fn candidates(&self, group: &Group, cx: &Context) -> Vec<Candidate> {
        identity_candidates(group, cx, |kind| matches!(kind, OpKind::Correlation { .. }))
    }
}

impl Construction for Freivalds {
    fn name(&self) -> &'static str {
        "linear: freivalds"
    }

    fn candidates(&self, group: &Group, cx: &Context) -> Vec<Candidate> {
        identity_candidates(group, cx, |kind| matches!(kind, OpKind::MatMul { .. }))
    }
}

fn lanes(group: &Group) -> Vec<Requirement> {
    group.lanes.iter().map(|lane| Requirement::Lane(lane.stmt)).collect()
}

fn identity_candidates(group: &Group, cx: &Context, applies: impl Fn(OpKind) -> bool) -> Vec<Candidate> {
    if !applies(group.kind) || !super::challenge::available(cx.target) {
        return Vec::new();
    }
    vec![Candidate {
        pre: vec![super::challenge::CHALLENGE],
        post: lanes(group),
        cost: identity_cost(group),
        emitter: Box::new(IdentityEmitter { kind: group.kind, shared: group.shared.clone(), raw: Vec::new() }),
    }]
}

// ── Emitters ────────────────────────────────────────────────────────────

struct DirectEmitter {
    kind: OpKind,
    shared: Vec<StmtId>,
}

impl Emitter for DirectEmitter {
    fn lane(&mut self, b: &mut IRBuilder, cx: &mut EmitCx, lane: &Lane) -> Value {
        let operands = cx.values(&self.shared);
        let pairs: Vec<(Value, Value)> = lane_pairs(self.kind, lane.position.expect("a position") as usize)
            .into_iter()
            .map(|(x, y)| (operands[x].clone(), operands[y].clone()))
            .collect();
        match self.kind {
            OpKind::Correlation { .. } => b.ir_product_sum_f(&pairs),
            _ => b.ir_dot_f(&pairs),
        }
    }
}

struct IdentityEmitter {
    kind: OpKind,
    shared: Vec<StmtId>,
    /// Every lane's hinted raw sum, by position.
    raw: Vec<Value>,
}

impl Emitter for IdentityEmitter {
    /// Hints every raw sum, then asserts the identity among the operands and
    /// the raw sums (the check comes after its operands, before any use).
    fn begin(&mut self, b: &mut IRBuilder, cx: &mut EmitCx) {
        let operands = cx.values(&self.shared);
        let zero = b.ir_constant_int(0);
        self.raw = (0..lane_count(self.kind))
            .map(|p| {
                let pairs: Vec<Value> = lane_pairs(self.kind, p)
                    .into_iter()
                    .flat_map(|(x, y)| [operands[x].clone(), operands[y].clone()])
                    .collect();
                b.ir_hint(HintKind::ProductSum, &pairs, &zero)
            })
            .collect();
        let (identity, check_operands): (Identity, Vec<Value>) = match self.kind {
            OpKind::Correlation { long, short } => {
                let (x, y) = operands.split_at(long as usize);
                let reversed = y.iter().rev().cloned();
                let all = x.iter().cloned().chain(reversed).chain(self.raw.iter().cloned()).collect();
                (Identity::PolynomialProduct { n: long, m: short }, all)
            }
            OpKind::MatMul { rows, inner, cols } => {
                let all = operands.iter().cloned().chain(self.raw.iter().cloned()).collect();
                (Identity::MatrixProduct { rows, inner, cols }, all)
            }
            other => unreachable!("{other:?}"),
        };
        b.ir_identity_check(identity, &check_operands);
    }

    fn lane(&mut self, b: &mut IRBuilder, _cx: &mut EmitCx, lane: &Lane) -> Value {
        let raw = self.raw[lane.position.expect("a position") as usize].clone();
        b.ir_rescale_raw_f(&raw)
    }
}
