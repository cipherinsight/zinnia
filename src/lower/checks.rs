//! Checks: the program's assertions read as facts, the cost of a check on
//! each target, and the emission of a check for a fact.

use std::collections::{HashMap, HashSet};

use crate::builder::IRBuilder;
use crate::ir::IRGraph;
use crate::ir_defs::IR;
use crate::types::{StmtId, Value};

use super::construction::Target;
use super::graph::{Bound, Claim, Fact, Scope};

/// An assertion of the program and the facts it establishes. The first
/// form is the fact on the asserted operand itself, on every path; a check
/// the IR generator emitted under an execution predicate `p` (on the
/// guarded operand `select(p, x, safe)`, or as `assert(select(p, c, true))`)
/// also has the form "the fact on `x` where `p` holds", which is
/// equivalent: where `p` is false the guarded operand is the safe value,
/// which passes.
///
/// `consequences` are facts the check implies but that do not imply it
/// back (so they never stand for the check as a requirement); they only
/// let the check make later checks redundant (`dominated`).
#[derive(Clone, Debug)]
pub struct ProgramCheck {
    pub stmt: StmtId,
    pub forms: Vec<Fact>,
    pub consequences: Vec<Fact>,
}

/// The program's assertions the layer understands, in statement order.
pub fn program_checks(graph: &IRGraph) -> Vec<ProgramCheck> {
    let bits: HashSet<StmtId> = graph
        .stmts
        .iter()
        .filter(|s| matches!(s.ir, IR::AssertBit))
        .filter_map(|s| s.arguments.first().copied())
        .collect();
    graph
        .stmts
        .iter()
        .filter_map(|stmt| {
            let forms = check_forms(graph, &stmt.ir, &stmt.arguments);
            let consequences = check_consequences(graph, &bits, &stmt.ir, &stmt.arguments);
            (!forms.is_empty()).then_some(ProgramCheck { stmt: stmt.stmt_id, forms, consequences })
        })
        .collect()
}

/// Facts a check implies that are weaker than it. One kind: NumPy's index
/// wrap `w = select(s, x + k, x)` (`helpers/static_array/base.rs`) checked
/// as `0 <= w < h`, with `s` asserted to be a bit and `k >= 0`, implies
/// `-k <= x < h`: for `s = 1`, `x = w - k` lies in `[-k, h - k)`; for
/// `s = 0`, `x = w` lies in `[0, h)`. (It does not imply the check back:
/// that also pins `s`.) The same index's write window `-n <= x < n`
/// (`AssertU128Lt(x + n, 2n)`) is then redundant after the read's check.
fn check_consequences(graph: &IRGraph, bits: &HashSet<StmtId>, ir: &IR, args: &[StmtId]) -> Vec<Fact> {
    let stmt = |s: StmtId| &graph.stmts[s as usize];
    let constant = |s: StmtId| match stmt(s).ir {
        IR::ConstantInt { value } => Some(value),
        _ => None,
    };
    let (IR::AssertU128Lt, &[w, hi]) = (ir, args) else { return Vec::new() };
    let (Some(h), (IR::SelectI, &[s, shifted, x])) = (constant(hi), (&stmt(w).ir, stmt(w).arguments.as_slice())) else {
        return Vec::new();
    };
    let offset = match (&stmt(shifted).ir, stmt(shifted).arguments.as_slice()) {
        (IR::AddI, &[y, k]) if y == x => constant(k),
        (IR::AddI, &[k, y]) if y == x => constant(k),
        _ => None,
    };
    match offset {
        Some(k) if k >= 0 && bits.contains(&s) => {
            vec![Fact::everywhere(Claim::InRange { x, lo: -k, hi: Bound::Const(h) })]
        }
        _ => Vec::new(),
    }
}

fn check_forms(graph: &IRGraph, ir: &IR, args: &[StmtId]) -> Vec<Fact> {
    let constant = |s: StmtId| match graph.stmts[s as usize].ir {
        IR::ConstantInt { value } => Some(value),
        IR::ConstantBool { value } => Some(i64::from(value)),
        _ => None,
    };
    // `select(p, x, k)` with a constant `k`.
    let guarded = |s: StmtId| match (&graph.stmts[s as usize].ir, graph.stmts[s as usize].arguments.as_slice()) {
        (IR::SelectI | IR::SelectB, &[p, x, k]) => constant(k).map(|k| (p, x, k)),
        _ => None,
    };
    // `x + k` with a constant `k`.
    let shifted = |s: StmtId| match (&graph.stmts[s as usize].ir, graph.stmts[s as usize].arguments.as_slice()) {
        (IR::AddI, &[x, k]) => constant(k).map(|k| (x, k)).or_else(|| constant(x).map(|c| (k, c))),
        _ => None,
    };
    match (ir, args) {
        (IR::AssertU128Lt, &[v, hi]) => {
            let bound = constant(hi).map_or(Bound::Stmt(hi), Bound::Const);
            let mut forms = vec![Fact::everywhere(Claim::InRange { x: v, lo: 0, hi: bound })];
            // `0 <= x + k < h` is `-k <= x < h - k` (NumPy's index window
            // `-n <= i < n` is checked as `AssertU128Lt(i + n, 2n)`).
            if let (Some((x, k)), Bound::Const(h)) = (shifted(v), bound) {
                forms.push(Fact::everywhere(Claim::InRange { x, lo: -k, hi: Bound::Const(h - k) }));
            }
            if let Some((p, x, k)) = guarded(v) {
                if k >= 0 && matches!(bound, Bound::Const(h) if k < h) {
                    forms.push(Fact { claim: Claim::InRange { x, lo: 0, hi: bound }, scope: Scope::When(p) });
                }
            }
            forms
        }
        (IR::AssertNonNegI, &[v]) => {
            let mut forms = vec![Fact::everywhere(Claim::NonNegative(v))];
            if let Some((p, x, k)) = guarded(v) {
                if k >= 0 {
                    forms.push(Fact { claim: Claim::NonNegative(x), scope: Scope::When(p) });
                }
            }
            forms
        }
        (IR::Assert, &[c]) => {
            if let Some((p, c, 1)) = guarded(c) {
                return condition_claim(graph, c, &constant)
                    .map(|claim| vec![Fact { claim, scope: Scope::When(p) }])
                    .unwrap_or_default();
            }
            condition_claim(graph, c, &constant).map(|claim| vec![Fact::everywhere(claim)]).unwrap_or_default()
        }
        _ => Vec::new(),
    }
}

/// The claim a Boolean statement states, when it is one the layer knows.
fn condition_claim(graph: &IRGraph, c: StmtId, constant: &dyn Fn(StmtId) -> Option<i64>) -> Option<Claim> {
    let stmt = &graph.stmts[c as usize];
    let float_zero = |s: StmtId| matches!(graph.stmts[s as usize].ir, IR::ConstantFloat { value } if value == 0.0);
    let zero = |s: StmtId| constant(s) == Some(0) || float_zero(s);
    match (&stmt.ir, stmt.arguments.as_slice()) {
        (IR::GteI | IR::GteF, &[x, z]) | (IR::LteI | IR::LteF, &[z, x]) if zero(z) => Some(Claim::NonNegative(x)),
        (IR::NeI | IR::NeF, &[x, z]) | (IR::NeI | IR::NeF, &[z, x]) if zero(z) => Some(Claim::NonZero(x)),
        (IR::EqI, &[a, b]) => Some(Claim::Equal(a, b)),
        // Integer orders against a constant (an input's int64 window).
        (IR::GteI, &[x, c]) | (IR::LteI, &[c, x]) if constant(c).is_some() => {
            Some(Claim::Within { x, lo: constant(c), hi: None })
        }
        (IR::LteI, &[x, c]) | (IR::GteI, &[c, x]) if constant(c).is_some() => {
            Some(Claim::Within { x, lo: None, hi: constant(c) })
        }
        _ => None,
    }
}

/// The checks an earlier check implies ("range checks ⇒ bounds": a later
/// `0 <= x < 8` after `0 <= x < 4`, or `x >= 0` after it). The IR has no
/// control flow, so every assertion is a constraint of the circuit; an
/// implied one adds nothing. Only earlier checks count, so a backend that
/// reads the bound a check proves for the statements after it (a cheaper
/// gadget for a value proven in range) still finds a bound as early.
pub fn dominated(checks: &[ProgramCheck]) -> HashSet<StmtId> {
    let mut known: HashMap<StmtId, Vec<Fact>> = HashMap::new();
    let mut dominated = HashSet::new();
    for check in checks {
        let implied = check.forms.iter().any(|form| {
            let subject = form.claim.subjects()[0];
            known.get(&subject).is_some_and(|facts| facts.iter().any(|k| k.implies(form)))
        });
        if implied {
            dominated.insert(check.stmt);
            continue;
        }
        for form in check.forms.iter().chain(&check.consequences) {
            known.entry(form.claim.subjects()[0]).or_default().push(*form);
        }
    }
    dominated
}

/// The estimated cost of checking `claim` on `target` (halo2-base cells,
/// Noir gates).
pub fn cost(claim: &Claim, target: Target) -> u64 {
    match (claim, target) {
        (Claim::InRange { .. }, Target::Halo2Base) => 8,
        // The Noir emitter checks `0 <= x < h` for a constant `h` as one
        // range check of `x` when `h` is a power of two, else as range
        // checks of `x` and `h - 1 - x` (measured on 0345 / 0346: 4 gates for
        // h = 6, about 1 for h = 8).
        (&Claim::InRange { hi: Bound::Const(h), .. }, Target::Noir) if h > 0 => {
            if (h as u64).is_power_of_two() {
                1
            } else {
                4
            }
        }
        (Claim::InRange { .. }, Target::Noir) => 5,
        (Claim::NonNegative(_), Target::Halo2Base) => 8,
        (Claim::NonNegative(_), Target::Noir) => 4,
        (Claim::NonZero(_), Target::Halo2Base) => 4,
        (Claim::NonZero(_), Target::Noir) => 3,
        (Claim::Equal(..), _) => 1,
        (Claim::Within { .. }, Target::Halo2Base) => 17,
        (Claim::Within { .. }, Target::Noir) => 3,
        (&Claim::Permutation(a, _), target) => super::permutation::hit_count_cost(a.entries().len() as u64, target, false),
    }
}

/// Emit a check of `fact` where its scope holds (through the conditional
/// checks of `builder/guard.rs`); `lookup` maps input statements to their
/// lowered values.
pub fn emit(b: &mut IRBuilder, fact: &Fact, lookup: &HashMap<StmtId, Value>) {
    match fact.scope {
        Scope::Everywhere => emit_claim(b, &fact.claim, lookup),
        Scope::When(p) => {
            let p = lookup[&p].clone();
            b.under_condition(&p, |b| emit_claim(b, &fact.claim, lookup));
        }
    }
}

fn emit_claim(b: &mut IRBuilder, claim: &Claim, lookup: &HashMap<StmtId, Value>) {
    let v = |s: StmtId| lookup[&s].clone();
    match *claim {
        Claim::InRange { x, lo, hi } => {
            let hi = match hi {
                Bound::Const(h) => b.ir_constant_int(h),
                Bound::Stmt(h) => v(h),
            };
            let (x, hi) = if lo == 0 {
                (v(x), hi)
            } else {
                let lo = b.ir_constant_int(lo);
                (b.ir_sub_i(&v(x), &lo), b.ir_sub_i(&hi, &lo))
            };
            b.ir_assert_u128_lt(&x, &hi);
        }
        Claim::NonNegative(x) => {
            b.ir_assert_non_neg_i(&v(x));
        }
        Claim::NonZero(x) => {
            let zero = b.ir_constant_int(0);
            let nonzero = b.ir_not_equal_i(&v(x), &zero);
            b.ir_assert(&nonzero);
        }
        Claim::Equal(x, y) => {
            let equal = b.ir_equal_i(&v(x), &v(y));
            b.ir_assert(&equal);
        }
        Claim::Permutation(values_list, targets_list) => {
            let mut values = |list: super::ListId| -> Vec<Value> {
                list.entries()
                    .iter()
                    .map(|entry| match *entry {
                        super::ListEntry::Stmt(s) => v(s),
                        super::ListEntry::Int(c) => b.ir_constant_int(c),
                    })
                    .collect()
            };
            let (values, targets) = (values(values_list), values(targets_list));
            super::permutation::emit_hit_counts(b, &values, &targets);
        }
        Claim::Within { x, lo, hi } => {
            if let Some(lo) = lo {
                let lo = b.ir_constant_int(lo);
                let above = b.ir_greater_than_or_equal_i(&v(x), &lo);
                b.ir_assert(&above);
            }
            if let Some(hi) = hi {
                let hi = b.ir_constant_int(hi);
                let below = b.ir_less_than_or_equal_i(&v(x), &hi);
                b.ir_assert(&below);
            }
        }
    }
}
