//! Array facts: claims about the values of a whole list of cells, where
//! they come from, and how the layer establishes them.
//!
//! The claims (`graph::Claim`) over a list `L` of values (`ListId`):
//!
//! | claim | meaning |
//! |---|---|
//! | `ElementsWithin { L, lo, hi }` | every value in `[lo, hi]` (an end may be open) |
//! | `NonDecreasing(L)` | `L[0] <= L[1] <= ...` |
//! | `StrictlyIncreasing(L)` | `L[0] < L[1] < ...` |
//! | `Distinct(L)` | no two values equal |
//! | `CountBound { L, max_nonzero }` | at most `max_nonzero` values nonzero |
//! | `Permutation(L, [0, ..., n-1])` | `L` is a permutation of `0..n` |
//!
//! and the scalar claim they decompose into, `Ordered { x, y, strict }`
//! (`x <= y`, or `x < y`). Values are integers or floats' fixed-point codes,
//! which are ordered as the floats.
//!
//! **How a claim is established** (the trust rule: a construction may rely
//! on a claim only if every accepted witness satisfies it):
//!
//! 1. **By definition** ([`definitional`]): the claim holds of the values
//!    whatever the witness, because of what the statements are. Constants;
//!    the value lanes of one sort at increasing constant ranks (every sort
//!    construction yields NumPy's stable sort, so they are non-decreasing);
//!    the index lanes of one sort (a stable argsort is a permutation of
//!    `0..n`, so its lanes are distinct, and all `n` of them are that
//!    permutation); squares, absolute values and sums of such terms are
//!    non-negative. These are free facts (`facts::FreeFacts`), and like a
//!    scalar gadget's postcondition they may discharge a program check.
//! 2. **By bounds**: every value's interval (`optim::RangeResolver`) lies in
//!    the claimed range, or two values' intervals are ordered. Free facts.
//! 3. **By the program's checks** ([`basis`]): an array claim is the
//!    conjunction of scalar claims about its values (`NonDecreasing(L)` is
//!    `Ordered { L[i], L[i+1] }` for every `i`; `ElementsWithin` is one
//!    `Within` per value and side). The pass adds a *composition node* that
//!    establishes the array claim from those scalar facts at no cost
//!    (`emit::add_composition_nodes`); each scalar fact is then justified as
//!    any other, typically by the program check that states it: the
//!    assertions `@requires(is_sorted(a))` or `forall(a, in_range(lo, hi))`
//!    emit (`optim/predicates/witness.rs`). A check that justifies a claim a
//!    selected construction relies on is kept.
//! 4. **By a construction's postcondition**: `Permutation(p, 0..n)` by the
//!    `is_permutation` check (`lower::permutation`).
//!
//! A construction that relies on a claim asks [`Context::established`]
//! first and offers its candidate only when the claim holds by one of these
//! routes without a new check: such claims are assumptions about the input,
//! not obligations NumPy's semantics impose, so the layer never emits a
//! check for one (that would reject inputs the program accepts).
//!
//! **Measurement switch.** `ZINNIA_ABLATE_ARRAY_FACTS=1` turns every array
//! claim off (`docs/ablations.md`): no claim is established, implied or
//! composed, and the program's order checks are not read as claims, so the
//! constructions keyed on them do not apply and the layer lowers as it did
//! before array claims existed.

use std::cell::Cell;

use crate::ir::IRGraph;
use crate::ir_defs::IR;
use crate::lower::op::OpKind;
use crate::types::StmtId;

use super::graph::{Bound, Claim, Fact, ListEntry, ListId, Scope};
use super::sort::{INDICES, VALUES};

thread_local! {
    static ENABLED: Cell<bool> = const { Cell::new(true) };
}

/// Whether array claims are on in this lowering (off under
/// `ZINNIA_ABLATE_ARRAY_FACTS=1`; set by `emit::lower_with`, per thread like
/// the list interner).
pub fn enabled() -> bool {
    ENABLED.with(Cell::get)
}

pub(super) fn set_enabled(on: bool) {
    ENABLED.with(|e| e.set(on));
}

// ── Implication ─────────────────────────────────────────────────────────

/// `true` when the array claim (or permutation) `a` implies `b`
/// syntactically: a claim about a list implies the same claim about a part
/// of it (`ElementsWithin`, `Distinct`: any sublist; the orders: a
/// subsequence in list order), the scalar claims about its values, and a
/// weaker claim of the same kind. `Permutation(p, 0..n)` implies
/// `ElementsWithin { p, 0, n - 1 }` and `Distinct(p)`.
pub fn implies(a: &Claim, b: &Claim) -> bool {
    use Claim::*;
    if let (Permutation(p, q), Permutation(r, s)) = (*a, *b) {
        return (p, q) == (r, s) || (p, q) == (s, r);
    }
    if !enabled() {
        return a == b;
    }
    match *a {
        Permutation(p, q) => {
            let values = match (range_len(q), range_len(p)) {
                (Some(_), _) => p,
                (None, Some(_)) => q,
                (None, None) => return false,
            };
            let n = values.len() as i64;
            implies(&ElementsWithin { list: values, lo: Some(0), hi: Some(n - 1) }, b) || implies(&Distinct(values), b)
        }
        ElementsWithin { list, lo, hi } => {
            let covers = |lo2: Option<i64>, hi2: Option<i64>| {
                lo2.is_none_or(|l| lo.is_some_and(|lo| lo >= l)) && hi2.is_none_or(|u| hi.is_some_and(|hi| hi <= u))
            };
            match *b {
                ElementsWithin { list: part, lo: lo2, hi: hi2 } => {
                    covers(lo2, hi2)
                        && part.entries().iter().all(|entry| match *entry {
                            ListEntry::Stmt(s) => list.contains(s),
                            ListEntry::Int(c) => lo2.is_none_or(|l| c >= l) && hi2.is_none_or(|u| c <= u),
                        })
                }
                Within { x, lo: lo2, hi: hi2 } => list.contains(x) && covers(lo2, hi2),
                InRange { x, lo: lo2, hi: Bound::Const(h) } => list.contains(x) && covers(Some(lo2), Some(h - 1)),
                NonNegative(x) => list.contains(x) && covers(Some(0), None),
                NonZero(x) => list.contains(x) && (covers(Some(1), None) || covers(None, Some(-1))),
                _ => false,
            }
        }
        NonDecreasing(list) => match *b {
            NonDecreasing(part) => is_subsequence(part, list),
            Ordered { x, y, strict: false } => ordered_positions(list, x, y, false),
            _ => false,
        },
        StrictlyIncreasing(list) => match *b {
            NonDecreasing(part) | StrictlyIncreasing(part) => is_subsequence(part, list),
            Distinct(part) => distinct_part(part, list),
            Ordered { x, y, strict } => ordered_positions(list, x, y, strict),
            _ => false,
        },
        Distinct(list) => matches!(*b, Distinct(part) if distinct_part(part, list)),
        CountBound { list, max_nonzero } => {
            matches!(*b, CountBound { list: l2, max_nonzero: m2 } if l2 == list && max_nonzero <= m2)
        }
        _ => a == b,
    }
}

/// `n` when `list` is `[0, 1, ..., n - 1]`.
pub fn range_len(list: ListId) -> Option<usize> {
    let entries = list.entries();
    entries.iter().enumerate().all(|(k, e)| *e == ListEntry::Int(k as i64)).then_some(entries.len())
}

/// Whether `part` is a subsequence of `list` (its values at increasing
/// positions of `list`); a list of at most one value always is (it is
/// trivially ordered).
fn is_subsequence(part: ListId, list: ListId) -> bool {
    let part = part.entries();
    if part.len() <= 1 {
        return true;
    }
    let list = list.entries();
    let mut rest = list.iter();
    part.iter().all(|wanted| rest.any(|e| e == wanted))
}

/// Whether `x` stands before `y` in `list` (strictly, or at the same
/// position too when not `strict`): then the order of `list` orders them.
fn ordered_positions(list: ListId, x: StmtId, y: StmtId, strict: bool) -> bool {
    if x == y {
        return !strict && list.contains(x);
    }
    let positions = list.positions();
    match (positions.get(&x), positions.get(&y)) {
        (Some(px), Some(py)) => px.iter().min() < py.iter().max(),
        _ => false,
    }
}

/// Whether `part`'s values are values of `list` at distinct positions.
fn distinct_part(part: ListId, list: ListId) -> bool {
    let entries = part.entries();
    let mut seen = std::collections::HashSet::new();
    entries.iter().all(|e| match *e {
        ListEntry::Stmt(s) => list.contains(s) && seen.insert(s),
        ListEntry::Int(_) => false,
    })
}

// ── Composition from scalar facts ───────────────────────────────────────

/// The scalar claims whose conjunction is `claim` (route 3 of the module
/// doc), or `None` when it has no such basis or a constant part of it is
/// false. Pairs and values that are integer constants are decided here and
/// left out.
pub fn basis(claim: &Claim) -> Option<Vec<Claim>> {
    match *claim {
        Claim::NonDecreasing(list) => order_basis(list, false),
        Claim::StrictlyIncreasing(list) => order_basis(list, true),
        Claim::ElementsWithin { list, lo, hi } => {
            let mut atoms = Vec::new();
            for entry in list.entries().iter() {
                match *entry {
                    ListEntry::Int(c) => {
                        if lo.is_some_and(|l| c < l) || hi.is_some_and(|u| c > u) {
                            return None;
                        }
                    }
                    ListEntry::Stmt(x) => {
                        // One atom per side: the program checks each side
                        // apart (`forall(a, in_range(lo, hi))` asserts
                        // `a >= lo` and `a <= hi`).
                        if lo.is_some() {
                            atoms.push(Claim::Within { x, lo, hi: None });
                        }
                        if hi.is_some() {
                            atoms.push(Claim::Within { x, lo: None, hi });
                        }
                    }
                }
            }
            Some(atoms)
        }
        _ => None,
    }
}

fn order_basis(list: ListId, strict: bool) -> Option<Vec<Claim>> {
    let entries = list.entries();
    let mut atoms = Vec::new();
    for pair in entries.windows(2) {
        match (pair[0], pair[1]) {
            (ListEntry::Int(a), ListEntry::Int(b)) => {
                if a > b || (strict && a == b) {
                    return None;
                }
            }
            (ListEntry::Stmt(x), ListEntry::Stmt(y)) => atoms.push(Claim::Ordered { x, y, strict }),
            (ListEntry::Int(c), ListEntry::Stmt(y)) => {
                atoms.push(Claim::Within { x: y, lo: Some(if strict { c.checked_add(1)? } else { c }), hi: None })
            }
            (ListEntry::Stmt(x), ListEntry::Int(c)) => {
                atoms.push(Claim::Within { x, lo: None, hi: Some(if strict { c.checked_sub(1)? } else { c }) })
            }
        }
    }
    Some(atoms)
}

/// The facts of `basis(fact.claim)`, in `fact`'s scope.
pub fn basis_facts(fact: &Fact) -> Option<Vec<Fact>> {
    Some(basis(&fact.claim)?.into_iter().map(|claim| Fact { claim, scope: fact.scope }).collect())
}

// ── Facts by definition ─────────────────────────────────────────────────

/// A statement's role in a sort, when it is a lane at a constant rank.
#[derive(Clone, Copy, PartialEq, Eq)]
struct SortLane<'g> {
    elements: &'g [StmtId],
    output: u8,
    rank: u32,
}

fn sort_lane(graph: &IRGraph, s: StmtId) -> Option<SortLane<'_>> {
    let stmt = graph.stmts.get(s as usize)?;
    match &stmt.ir {
        IR::ArrayOp(op) if op.kind == OpKind::Sort && (op.output == VALUES || op.output == INDICES) => Some(SortLane {
            elements: &stmt.arguments[..op.shared as usize],
            output: op.output,
            rank: op.position?,
        }),
        _ => None,
    }
}

/// The integer code of a constant statement: an integer or Boolean, or a
/// float's fixed-point code.
pub fn constant_code(graph: &IRGraph, s: StmtId) -> Option<i64> {
    match graph.stmts.get(s as usize)?.ir {
        IR::ConstantInt { value } => Some(value),
        IR::ConstantBool { value } => Some(i64::from(value)),
        IR::ConstantFloat { value } => {
            let precision = crate::prove::types::ProvingParams::default().precision_bits;
            i64::try_from(crate::prove::field::quantize(value, precision)).ok()
        }
        _ => None,
    }
}

/// The code of a list entry when it is a constant.
fn entry_code(graph: &IRGraph, entry: ListEntry) -> Option<i64> {
    match entry {
        ListEntry::Int(c) => Some(c),
        ListEntry::Stmt(s) => constant_code(graph, s),
    }
}

/// Whether `claim` holds by the definition of its statements (route 1 of
/// the module doc): on every witness the circuit accepts, whatever
/// construction lowers the statements, without relying on any check.
pub fn definitional(graph: &IRGraph, claim: &Claim) -> bool {
    if !enabled() {
        return false;
    }
    match *claim {
        Claim::Ordered { x, y, strict } => {
            if let (Some(a), Some(b)) = (constant_code(graph, x), constant_code(graph, y)) {
                return if strict { a < b } else { a <= b };
            }
            if x == y {
                return !strict;
            }
            // Two value lanes of one sort: the lower rank's value is at most
            // the higher rank's.
            matches!((sort_lane(graph, x), sort_lane(graph, y)),
                (Some(a), Some(b)) if !strict && a.output == VALUES && b.output == VALUES
                    && a.elements == b.elements && a.rank <= b.rank)
        }
        Claim::NonDecreasing(list) | Claim::StrictlyIncreasing(list) => {
            let strict = matches!(claim, Claim::StrictlyIncreasing(_));
            let entries = list.entries();
            entries.windows(2).all(|pair| match (entry_code(graph, pair[0]), entry_code(graph, pair[1])) {
                (Some(a), Some(b)) => if strict { a < b } else { a <= b },
                _ => match (pair[0], pair[1]) {
                    (ListEntry::Stmt(x), ListEntry::Stmt(y)) => definitional(graph, &Claim::Ordered { x, y, strict }),
                    _ => false,
                },
            })
        }
        Claim::Distinct(list) => {
            let entries = list.entries();
            let codes: Option<Vec<i64>> = entries.iter().map(|&e| entry_code(graph, e)).collect();
            if let Some(mut codes) = codes {
                codes.sort_unstable();
                return codes.windows(2).all(|w| w[0] < w[1]);
            }
            // Index lanes of one sort at distinct ranks (a permutation's
            // values are distinct).
            index_lanes(graph, list).is_some_and(|(_, ranks)| {
                let mut sorted = ranks.clone();
                sorted.sort_unstable();
                sorted.dedup();
                sorted.len() == ranks.len()
            })
        }
        Claim::Permutation(p, q) => {
            let (values, n) = match (range_len(q), range_len(p)) {
                (Some(n), _) => (p, n),
                (None, Some(n)) => (q, n),
                (None, None) => return false,
            };
            // All `n` index lanes of one sort over `n` elements (in any
            // order): the stable argsort, a permutation of `0..n`.
            index_lanes(graph, values).is_some_and(|(elements, mut ranks)| {
                ranks.sort_unstable();
                elements == n && ranks.iter().enumerate().all(|(k, &r)| r as usize == k)
            })
        }
        Claim::CountBound { list, max_nonzero } => list.len() <= max_nonzero as usize,
        Claim::NonNegative(x) => non_negative_by_definition(graph, x, 0),
        Claim::ElementsWithin { list, lo: Some(lo), hi: None } if lo <= 0 => {
            list.entries().iter().all(|&e| match e {
                ListEntry::Int(c) => c >= lo,
                ListEntry::Stmt(s) => non_negative_by_definition(graph, s, 0),
            })
        }
        _ => false,
    }
}

/// When every entry of `list` is an index lane of one sort: the number of
/// elements that sort has, and the lanes' ranks.
fn index_lanes(graph: &IRGraph, list: ListId) -> Option<(usize, Vec<u32>)> {
    let mut elements: Option<&[StmtId]> = None;
    let mut ranks = Vec::new();
    for entry in list.entries().iter() {
        let ListEntry::Stmt(s) = *entry else { return None };
        let lane = sort_lane(graph, s).filter(|l| l.output == INDICES)?;
        if elements.is_some_and(|e| e != lane.elements) {
            return None;
        }
        elements = Some(lane.elements);
        ranks.push(lane.rank);
    }
    Some((elements?.len(), ranks))
}

/// Whether `x >= 0` by its definition: a square (`x * x`, integer or
/// float), an absolute value, a non-negative constant, a comparison or
/// Boolean, a sum or product of such values, a cast of one, a select between
/// two, or a sort value of such elements; `depth` bounds the walk.
///
/// Soundness: integer arithmetic in the IR is exact in the field, so a value
/// is the integer its definition says as long as no intermediate value
/// wraps the field. [`magnitude_bits`] bounds every value involved below
/// `2^250` (far below `p / 2`), from the input domain (every input is in the
/// signed 64-bit window, `ir_gen/input_domain.rs`) and the operations alone.
/// Then a square or a product of non-negative integers is a non-negative
/// integer, and so is a sum of such. A float's square or product is the
/// floored quotient by `2^32` of the product of the codes, non-negative when
/// that product is; `abs` is `|x|`.
pub fn non_negative_by_definition(graph: &IRGraph, x: StmtId, depth: u32) -> bool {
    if depth > 64 || magnitude_bits(graph, x, 0).is_none_or(|m| m > MAX_MAGNITUDE_BITS) {
        return false;
    }
    let Some(stmt) = graph.stmts.get(x as usize) else { return false };
    let args = stmt.arguments.as_slice();
    let rec = |s: StmtId| non_negative_by_definition(graph, s, depth + 1);
    match &stmt.ir {
        IR::ConstantInt { value } => *value >= 0,
        IR::ConstantFloat { value } => *value >= 0.0,
        IR::ConstantBool { .. } => true,
        IR::MulI | IR::MulF => matches!(args, [a, b] if a == b) || args.iter().all(|&a| rec(a)),
        IR::AbsI | IR::AbsF => true,
        IR::AddI | IR::AddF | IR::SumF { .. } => args.iter().all(|&a| rec(a)),
        IR::IntCast | IR::FloatCast | IR::BoolCast => args.iter().all(|&a| rec(a)),
        IR::SelectI | IR::SelectF => args.len() == 3 && rec(args[1]) && rec(args[2]),
        IR::EqI | IR::NeI | IR::LtI | IR::LteI | IR::GtI | IR::GteI | IR::EqF | IR::NeF | IR::LtF | IR::LteF
        | IR::GtF | IR::GteF | IR::LogicalAnd | IR::LogicalOr | IR::LogicalNot => true,
        IR::ArrayOp(op) if op.kind == OpKind::Sort && op.output == VALUES => {
            args[..op.shared as usize].iter().all(|&a| rec(a))
        }
        _ => false,
    }
}

/// Values whose magnitude stays below `2^MAX_MAGNITUDE_BITS` are integers of
/// the field's signed range with room for a few more additions.
const MAX_MAGNITUDE_BITS: u32 = 250;

/// An upper bound `m` with `|x| < 2^m` for the integer (or float code) `x`
/// on every accepted witness, from its definition and the input domain
/// alone; `None` when unknown or too large to matter.
pub fn magnitude_bits(graph: &IRGraph, x: StmtId, depth: u32) -> Option<u32> {
    if depth > 64 {
        return None;
    }
    let stmt = graph.stmts.get(x as usize)?;
    let args = stmt.arguments.as_slice();
    let rec = |s: StmtId| magnitude_bits(graph, s, depth + 1);
    let max_of = |items: &[StmtId]| items.iter().map(|&a| rec(a)).try_fold(0u32, |m, b| Some(m.max(b?)));
    let bits_of = |v: u128| 128 - v.leading_zeros();
    let m = match &stmt.ir {
        IR::ConstantInt { value } => bits_of(value.unsigned_abs() as u128),
        IR::ConstantBool { .. } => 1,
        IR::ConstantFloat { .. } => bits_of(constant_code(graph, x)?.unsigned_abs() as u128),
        // The input domain: integers and float codes in the 64-bit window.
        IR::ReadInteger { .. } | IR::ReadFloat { .. } => 64,
        IR::EqI | IR::NeI | IR::LtI | IR::LteI | IR::GtI | IR::GteI | IR::EqF | IR::NeF | IR::LtF | IR::LteF
        | IR::GtF | IR::GteF | IR::LogicalAnd | IR::LogicalOr | IR::LogicalNot | IR::BoolCast => 1,
        IR::AddI | IR::SubI | IR::AddF | IR::SubF => max_of(args)? + 1,
        IR::SumF { .. } => max_of(args)? + bits_of(args.len() as u128),
        IR::MulI => rec(*args.first()?)? + rec(*args.get(1)?)?,
        // floor(a b / 2^32): |q| <= |a b| / 2^32 + 1.
        IR::MulF => (rec(*args.first()?)? + rec(*args.get(1)?)?).saturating_sub(32) + 1,
        IR::AbsI | IR::AbsF => rec(*args.first()?)?,
        IR::SelectI | IR::SelectF if args.len() == 3 => max_of(&args[1..])?,
        IR::ArrayOp(op) if op.kind == OpKind::Sort && op.output == VALUES => max_of(&args[..op.shared as usize])?,
        _ => return None,
    };
    (m <= MAX_MAGNITUDE_BITS).then_some(m)
}

/// The scope a composed fact holds in: the composition of facts holding
/// everywhere holds everywhere.
pub fn everywhere(claim: Claim) -> Fact {
    Fact { claim, scope: Scope::Everywhere }
}

// ── Facts the IR states (the interface for the IR generator) ────────────

/// The claim of an `IR::ArrayFact` statement about the values of its list
/// (the statement's first `len` operands), which the IR generator states
/// (`IRBuilder::ir_array_fact`, `docs/array-facts-lowering.md`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum ArrayFactKind {
    /// Every value in `[lo, hi]` (`None`: an open end).
    ElementsWithin { lo: Option<i64>, hi: Option<i64> },
    NonDecreasing,
    StrictlyIncreasing,
    Distinct,
    /// At most `max_nonzero` values nonzero.
    CountBound { max_nonzero: u32 },
    /// The values are a permutation of `0, 1, ..., len - 1`.
    Permutation,
}

/// The payload of `IR::ArrayFact`: the claim and how many of the
/// statement's operands are the list (the rest are its basis).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct ArrayFactSpec {
    pub kind: ArrayFactKind,
    pub len: u32,
}

impl ArrayFactSpec {
    /// The statement's printed form.
    pub fn describe(&self) -> String {
        format!("array_fact[{:?}][{}]", self.kind, self.len)
    }

    /// The claim about `list`.
    pub fn claim(&self, list: ListId) -> Claim {
        match self.kind {
            ArrayFactKind::ElementsWithin { lo, hi } => Claim::ElementsWithin { list, lo, hi },
            ArrayFactKind::NonDecreasing => Claim::NonDecreasing(list),
            ArrayFactKind::StrictlyIncreasing => Claim::StrictlyIncreasing(list),
            ArrayFactKind::Distinct => Claim::Distinct(list),
            ArrayFactKind::CountBound { max_nonzero } => Claim::CountBound { list, max_nonzero },
            ArrayFactKind::Permutation => Claim::Permutation(list, ListId::range(list.len())),
        }
    }
}

/// A fact an `IR::ArrayFact` statement states, and its basis: the Boolean
/// statements whose truth it rests on (each asserted by the program); none
/// when it holds by the definition of the list's values.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Stated {
    pub fact: Fact,
    pub basis: Vec<StmtId>,
}

/// The bounds each value of the stated facts' lists has (`Within` facts,
/// one per value), from the facts that hold on every accepted witness: the
/// definitional ones, and those whose every basis condition `asserted`
/// says the program asserts. `ElementsWithin` gives its bounds, and a
/// permutation of `0..n` gives `[0, n - 1]`.
pub fn value_bounds(stated: &[Stated], asserted: impl Fn(StmtId) -> bool) -> Vec<Fact> {
    let mut bounds = Vec::new();
    for s in stated.iter().filter(|s| s.basis.iter().all(|&c| asserted(c))) {
        let (list, lo, hi) = match s.fact.claim {
            Claim::ElementsWithin { list, lo, hi } => (list, lo, hi),
            Claim::Permutation(p, q) if range_len(q).is_some() => (p, Some(0), Some(p.len() as i64 - 1)),
            _ => continue,
        };
        bounds.extend(list.stmts().into_iter().map(|x| everywhere(Claim::Within { x, lo, hi })));
    }
    bounds
}

/// The facts the graph's `IR::ArrayFact` statements state (none when array
/// claims are off).
pub fn stated(graph: &IRGraph) -> Vec<Stated> {
    if !enabled() {
        return Vec::new();
    }
    graph
        .stmts
        .iter()
        .filter_map(|s| match &s.ir {
            IR::ArrayFact(spec) => {
                let len = spec.len as usize;
                let list = ListId::of_stmts(&s.arguments[..len]);
                Some(Stated { fact: everywhere(spec.claim(list)), basis: s.arguments[len..].to_vec() })
            }
            _ => None,
        })
        .collect()
}
