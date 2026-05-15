//! Fact-derivation and interval-bound primitives.
//!
//! These functions read from the fact stack (`FactStack`) to recover
//! `(min, max)` bounds on `ValueId`s, and use those bounds to construct
//! interval-shaped op-contract facts on binary-op outputs. They are
//! facts-only (no SMT) and intentionally narrow — only the shapes our
//! op-contract content emits today.

/// Scan `facts.per_stmt[ptr]` for `Cmp(SsaPtr(ptr) op LitInt(n))` shapes
/// and synthesize a `(min, max)` pair. Returns `None` if either half is
/// missing. Used by [`super::require_static_or_bounded_int`] as a fallback
/// when the resolver-based bound pass fails.
///
/// Recognized shapes (and the bound each yields):
/// - `Ge(SsaPtr, LitInt(n))` → min = n
/// - `Gt(SsaPtr, LitInt(n))` → min = n + 1
/// - `Le(SsaPtr, LitInt(n))` → max = n
/// - `Lt(SsaPtr, LitInt(n))` → max = n - 1
///
/// LitInt-on-left forms (e.g., `Ge(LitInt(0), SsaPtr(p))`) are also
/// handled by swapping the relation. This is intentionally narrow —
/// only the fact shapes our op-contract content emits today. Richer
/// shapes (arithmetic, predicates) require routing through `prove`.
pub(crate) fn derive_bounds_from_facts(
    facts: &crate::optim::predicates::FactStack,
    vid: crate::types::ValueId,
) -> Option<(i64, i64)> {
    use crate::optim::predicates::formula::{BoolOp, CmpOp, ContractTerm, ContractVar};
    let entries = facts.per_value.get(&vid)?;
    let path_conds: Vec<&ContractTerm> = facts.visible_path_conditions();
    let mut lo: Option<i64> = None;
    let mut hi: Option<i64> = None;
    for raw_term in entries {
        // Look through `Implies(p, body)` (encoded as `Or(Not(p), body)`)
        // when `p` matches an in-scope path condition. The wrapping is
        // how the branch-merge preserves the anchor for arm-only facts;
        // the body is the original value_id-anchored claim.
        let term: &ContractTerm = match raw_term {
            ContractTerm::BoolComb { op: BoolOp::Or, operands } if operands.len() == 2 => {
                if let ContractTerm::Not(inner) = &operands[0] {
                    let antecedent = inner.as_ref();
                    if path_conds.iter().any(|p| **p == *antecedent) {
                        &operands[1]
                    } else {
                        raw_term
                    }
                } else {
                    raw_term
                }
            }
            _ => raw_term,
        };
        if let ContractTerm::Cmp { op, lhs, rhs } = term {
            // Match `Value(vid) <op> LitInt(n)` or `LitInt(n) <op> Value(vid)`.
            let (relation, n) = match (lhs.as_ref(), rhs.as_ref()) {
                (
                    ContractTerm::Var(ContractVar::Value(v)),
                    ContractTerm::LitInt(n),
                ) if *v == vid => (*op, *n),
                (
                    ContractTerm::LitInt(n),
                    ContractTerm::Var(ContractVar::Value(v)),
                ) if *v == vid => {
                    // Swap: `n <op> ptr` becomes `ptr <swap(op)> n`.
                    let swapped = match op {
                        CmpOp::Lt => CmpOp::Gt,
                        CmpOp::Le => CmpOp::Ge,
                        CmpOp::Gt => CmpOp::Lt,
                        CmpOp::Ge => CmpOp::Le,
                        CmpOp::Eq => CmpOp::Eq,
                        CmpOp::Ne => CmpOp::Ne,
                    };
                    (swapped, *n)
                }
                _ => continue,
            };
            match relation {
                CmpOp::Ge => lo = Some(lo.map_or(n, |cur| cur.max(n))),
                CmpOp::Gt => lo = Some(lo.map_or(n + 1, |cur| cur.max(n + 1))),
                CmpOp::Le => hi = Some(hi.map_or(n, |cur| cur.min(n))),
                CmpOp::Lt => hi = Some(hi.map_or(n - 1, |cur| cur.min(n - 1))),
                CmpOp::Eq => {
                    lo = Some(lo.map_or(n, |cur| cur.max(n)));
                    hi = Some(hi.map_or(n, |cur| cur.min(n)));
                }
                CmpOp::Ne => {}
            }
        }
    }
    match (lo, hi) {
        (Some(min), Some(max)) => Some((min, max)),
        _ => None,
    }
}

/// Compute the interval bound on the output of an integer arithmetic op
/// given input intervals, returning a deposit-ready `ContractTerm`.
///
/// Facts-only lookup: uses [`derive_bounds_from_facts`] on each input;
/// does NOT invoke SMT. Returns `None` when either input lacks
/// fact-derived bounds.
///
/// Checked-arithmetic default-deny: any `checked_*` overflow on the
/// interval corners produces `None` (no fact deposited) rather than a
/// possibly-unsound wrapped value.
pub fn interval_fact_for_int_binary(
    facts: &crate::optim::predicates::FactStack,
    op: crate::optim::predicates::formula::ArithOp,
    a_vid: crate::types::ValueId,
    b_vid: crate::types::ValueId,
    out_vid: crate::types::ValueId,
) -> Option<crate::optim::predicates::formula::ContractTerm> {
    use crate::optim::predicates::formula::{
        ArithOp, BoolOp, CmpOp, ContractTerm, ContractVar,
    };

    let (a_lo, a_hi) = derive_bounds_from_facts(facts, a_vid)?;
    let (b_lo, b_hi) = derive_bounds_from_facts(facts, b_vid)?;

    let (out_lo, out_hi) = match op {
        ArithOp::Add => {
            let lo = a_lo.checked_add(b_lo)?;
            let hi = a_hi.checked_add(b_hi)?;
            (lo, hi)
        }
        ArithOp::Sub => {
            let lo = a_lo.checked_sub(b_hi)?;
            let hi = a_hi.checked_sub(b_lo)?;
            (lo, hi)
        }
        ArithOp::Mul => {
            let c1 = a_lo.checked_mul(b_lo)?;
            let c2 = a_lo.checked_mul(b_hi)?;
            let c3 = a_hi.checked_mul(b_lo)?;
            let c4 = a_hi.checked_mul(b_hi)?;
            let lo = c1.min(c2).min(c3).min(c4);
            let hi = c1.max(c2).max(c3).max(c4);
            (lo, hi)
        }
        _ => return None,
    };

    Some(ContractTerm::BoolComb {
        op: BoolOp::And,
        operands: vec![
            ContractTerm::Cmp {
                op: CmpOp::Ge,
                lhs: Box::new(ContractTerm::Var(ContractVar::Value(out_vid))),
                rhs: Box::new(ContractTerm::LitInt(out_lo)),
            },
            ContractTerm::Cmp {
                op: CmpOp::Le,
                lhs: Box::new(ContractTerm::Var(ContractVar::Value(out_vid))),
                rhs: Box::new(ContractTerm::LitInt(out_hi)),
            },
        ],
    })
}

/// Float analog of [`derive_bounds_from_facts`]. Walks the fact stack
/// for `Cmp(Ge | Gt | Le | Lt | Eq, Var(Value(vid)), LitFloat(f))` clauses
/// (or the swapped `LitFloat` on the left), returning `(lo, hi)` if both
/// directions are visible.
///
/// Mirrors the int variant structurally; the only differences are the
/// literal type (`LitFloat(ContractFloat)`), the bound type (`f64`), and
/// the absence of integer `+1 / -1` strictness adjustments — for the
/// float case `Gt(v, f)` and `Lt(v, f)` are treated as non-strict (i.e.
/// the same as `Ge`/`Le`). The widening is sound: a strict bound implies
/// the non-strict one.
pub(crate) fn derive_float_bounds_from_facts(
    facts: &crate::optim::predicates::FactStack,
    vid: crate::types::ValueId,
) -> Option<(f64, f64)> {
    use crate::optim::predicates::formula::{BoolOp, CmpOp, ContractFloat, ContractTerm, ContractVar};
    let entries = facts.per_value.get(&vid)?;
    let path_conds: Vec<&ContractTerm> = facts.visible_path_conditions();
    let mut lo: Option<f64> = None;
    let mut hi: Option<f64> = None;
    for raw_term in entries {
        let term: &ContractTerm = match raw_term {
            ContractTerm::BoolComb { op: BoolOp::Or, operands } if operands.len() == 2 => {
                if let ContractTerm::Not(inner) = &operands[0] {
                    let antecedent = inner.as_ref();
                    if path_conds.iter().any(|p| **p == *antecedent) {
                        &operands[1]
                    } else {
                        raw_term
                    }
                } else {
                    raw_term
                }
            }
            _ => raw_term,
        };
        if let ContractTerm::Cmp { op, lhs, rhs } = term {
            let (relation, n) = match (lhs.as_ref(), rhs.as_ref()) {
                (
                    ContractTerm::Var(ContractVar::Value(v)),
                    ContractTerm::LitFloat(ContractFloat(n)),
                ) if *v == vid => (*op, *n),
                (
                    ContractTerm::LitFloat(ContractFloat(n)),
                    ContractTerm::Var(ContractVar::Value(v)),
                ) if *v == vid => {
                    let swapped = match op {
                        CmpOp::Lt => CmpOp::Gt,
                        CmpOp::Le => CmpOp::Ge,
                        CmpOp::Gt => CmpOp::Lt,
                        CmpOp::Ge => CmpOp::Le,
                        CmpOp::Eq => CmpOp::Eq,
                        CmpOp::Ne => CmpOp::Ne,
                    };
                    (swapped, *n)
                }
                _ => continue,
            };
            match relation {
                CmpOp::Ge | CmpOp::Gt => {
                    lo = Some(lo.map_or(n, |cur| cur.max(n)));
                }
                CmpOp::Le | CmpOp::Lt => {
                    hi = Some(hi.map_or(n, |cur| cur.min(n)));
                }
                CmpOp::Eq => {
                    lo = Some(lo.map_or(n, |cur| cur.max(n)));
                    hi = Some(hi.map_or(n, |cur| cur.min(n)));
                }
                CmpOp::Ne => {}
            }
        }
    }
    match (lo, hi) {
        (Some(min), Some(max)) => Some((min, max)),
        _ => None,
    }
}

/// Compute the interval bound on the output of a float arithmetic op
/// given input intervals, returning a deposit-ready `ContractTerm`.
///
/// Facts-only lookup: uses [`derive_float_bounds_from_facts`] on each
/// input; does NOT invoke SMT. Returns `None` when either input lacks
/// fact-derived bounds.
///
/// f64-arithmetic default-deny: overflow on the interval corners
/// produces `±inf` (or NaN via `inf - inf` etc.); the `is_finite` guard
/// drops any non-finite corner so we never deposit a bound that would
/// lower poorly to the ZK Real fragment.
pub fn interval_fact_for_float_binary(
    facts: &crate::optim::predicates::FactStack,
    op: crate::optim::predicates::formula::ArithOp,
    a_vid: crate::types::ValueId,
    b_vid: crate::types::ValueId,
    out_vid: crate::types::ValueId,
) -> Option<crate::optim::predicates::formula::ContractTerm> {
    use crate::optim::predicates::formula::{
        ArithOp, BoolOp, CmpOp, ContractFloat, ContractTerm, ContractVar,
    };

    let (a_lo, a_hi) = derive_float_bounds_from_facts(facts, a_vid)?;
    let (b_lo, b_hi) = derive_float_bounds_from_facts(facts, b_vid)?;

    let (out_lo, out_hi) = match op {
        ArithOp::Add => (a_lo + b_lo, a_hi + b_hi),
        ArithOp::Sub => (a_lo - b_hi, a_hi - b_lo),
        ArithOp::Mul => {
            let c1 = a_lo * b_lo;
            let c2 = a_lo * b_hi;
            let c3 = a_hi * b_lo;
            let c4 = a_hi * b_hi;
            (c1.min(c2).min(c3).min(c4), c1.max(c2).max(c3).max(c4))
        }
        _ => return None,
    };

    // Skip emit on overflow to ±inf or NaN (e.g. `inf - inf`). Sound:
    // no relayed fact is always preferable to lowering a non-finite
    // literal into the ZK Real fragment.
    if !out_lo.is_finite() || !out_hi.is_finite() {
        return None;
    }

    Some(ContractTerm::BoolComb {
        op: BoolOp::And,
        operands: vec![
            ContractTerm::Cmp {
                op: CmpOp::Ge,
                lhs: Box::new(ContractTerm::Var(ContractVar::Value(out_vid))),
                rhs: Box::new(ContractTerm::LitFloat(ContractFloat(out_lo))),
            },
            ContractTerm::Cmp {
                op: CmpOp::Le,
                lhs: Box::new(ContractTerm::Var(ContractVar::Value(out_vid))),
                rhs: Box::new(ContractTerm::LitFloat(ContractFloat(out_hi))),
            },
        ],
    })
}

/// Aggregate per-element fact-derived bounds into a single `[lo, hi]`
/// union (`min` of lows, `max` of highs). Returns `None` if **any**
/// element lacks fact-derived bounds — soundness gate: a single
/// unbounded element makes the union unbounded too.
///
/// Used by [`super::relay_reduction_output_interval_int`] to summarise the
/// element-wise bound for a composite-array input. Per-element facts
/// live on each element's own `ValueId`; the whole-list `ValueId` does
/// not carry the bound directly.
pub fn aggregate_element_bounds(
    facts: &crate::optim::predicates::FactStack,
    element_vids: &[crate::types::ValueId],
) -> Option<(i64, i64)> {
    if element_vids.is_empty() {
        return None;
    }
    let mut lo: Option<i64> = None;
    let mut hi: Option<i64> = None;
    for vid in element_vids {
        let (e_lo, e_hi) = derive_bounds_from_facts(facts, *vid)?;
        lo = Some(lo.map_or(e_lo, |cur| cur.min(e_lo)));
        hi = Some(hi.map_or(e_hi, |cur| cur.max(e_hi)));
    }
    Some((lo?, hi?))
}
