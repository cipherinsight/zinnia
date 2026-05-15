//! Relay helpers: propagate interval bounds / content facts from inputs to
//! outputs of unary / reduction / shape-preserving / element-union ops.
//!
//! Each `relay_*` function reads facts off the input(s), derives a sound
//! output-bound, and deposits it on `output_vid`. All honour the
//! `ZINNIA_REQ_DISABLE` kill switch via [`super::req_disabled`].

use std::collections::HashMap;

use super::intervals::{aggregate_element_bounds, derive_float_bounds_from_facts};
use super::req_disabled;

/// Relay an interval bound through `sqrt`: given `input ∈ [lo, hi]` on the
/// fact stack, plant `output ∈ [sqrt(lo), sqrt(hi)]` on `output_vid`.
///
/// `sqrt` is monotone on `[0, ∞)` and f64 `sqrt` is correctly-rounded, so
/// the relayed interval is exact (no widening needed). Returns `true`
/// when a bound was deposited.
///
/// Mirrors [`super::interval_fact_for_float_binary`]'s pattern: facts-only
/// lookup, default-deny on non-finite corners.
pub fn relay_sqrt_output_interval(
    b: &mut crate::builder::IRBuilder,
    input_vid: crate::types::ValueId,
    output_vid: crate::types::ValueId,
) -> bool {
    use crate::optim::predicates::formula::{
        BoolOp, CmpOp, ContractFloat, ContractTerm, ContractVar,
    };

    if req_disabled() {
        return false;
    }
    let Some((lo, hi)) = derive_float_bounds_from_facts(&b.facts, input_vid) else {
        return false;
    };
    // Defensive: `sqrt_f`'s `requires(x >= 0.0)` (Group 1's R) prevents
    // negative inputs from reaching here. If a stale fact slips through,
    // skip emit — `lo.sqrt()` would produce NaN and we must never
    // deposit an unsound bound.
    if lo < 0.0 {
        return false;
    }
    let out_lo = lo.sqrt();
    let out_hi = hi.sqrt();
    if !out_lo.is_finite() || !out_hi.is_finite() {
        return false;
    }
    b.facts.insert_for(
        output_vid,
        ContractTerm::BoolComb {
            op: BoolOp::And,
            operands: vec![
                ContractTerm::Cmp {
                    op: CmpOp::Ge,
                    lhs: Box::new(ContractTerm::Var(ContractVar::Value(output_vid))),
                    rhs: Box::new(ContractTerm::LitFloat(ContractFloat(out_lo))),
                },
                ContractTerm::Cmp {
                    op: CmpOp::Le,
                    lhs: Box::new(ContractTerm::Var(ContractVar::Value(output_vid))),
                    rhs: Box::new(ContractTerm::LitFloat(ContractFloat(out_hi))),
                },
            ],
        },
    );
    true
}

/// Relay an interval bound through `exp`: given `input ∈ [lo, hi]` on the
/// fact stack, plant `output ∈ [exp(lo), exp(hi)]` on `output_vid`.
///
/// `exp` is monotone-increasing on all of `R`. f64 `exp(big)` overflows
/// to `+inf`; the `is_finite()` guard then skips emit rather than
/// depositing a non-finite literal. Returns `true` when a bound was
/// deposited.
///
/// Mirrors [`relay_sqrt_output_interval`]: facts-only lookup,
/// default-deny on non-finite corners. No precondition guard (domain is
/// all reals).
pub fn relay_exp_output_interval(
    b: &mut crate::builder::IRBuilder,
    input_vid: crate::types::ValueId,
    output_vid: crate::types::ValueId,
) -> bool {
    use crate::optim::predicates::formula::{
        BoolOp, CmpOp, ContractFloat, ContractTerm, ContractVar,
    };

    if req_disabled() {
        return false;
    }
    let Some((lo, hi)) = derive_float_bounds_from_facts(&b.facts, input_vid) else {
        return false;
    };
    let out_lo = lo.exp();
    let out_hi = hi.exp();
    if !out_lo.is_finite() || !out_hi.is_finite() {
        return false;
    }
    b.facts.insert_for(
        output_vid,
        ContractTerm::BoolComb {
            op: BoolOp::And,
            operands: vec![
                ContractTerm::Cmp {
                    op: CmpOp::Ge,
                    lhs: Box::new(ContractTerm::Var(ContractVar::Value(output_vid))),
                    rhs: Box::new(ContractTerm::LitFloat(ContractFloat(out_lo))),
                },
                ContractTerm::Cmp {
                    op: CmpOp::Le,
                    lhs: Box::new(ContractTerm::Var(ContractVar::Value(output_vid))),
                    rhs: Box::new(ContractTerm::LitFloat(ContractFloat(out_hi))),
                },
            ],
        },
    );
    true
}

/// Relay an interval bound through `log`: given `input ∈ [lo, hi]` on
/// the fact stack with `lo > 0`, plant `output ∈ [log(lo), log(hi)]` on
/// `output_vid`.
///
/// `log` is monotone-increasing on `(0, ∞)`. Returns `true` when a
/// bound was deposited.
///
/// Mirrors [`relay_sqrt_output_interval`]: facts-only lookup,
/// default-deny on non-finite corners.
pub fn relay_log_output_interval(
    b: &mut crate::builder::IRBuilder,
    input_vid: crate::types::ValueId,
    output_vid: crate::types::ValueId,
) -> bool {
    use crate::optim::predicates::formula::{
        BoolOp, CmpOp, ContractFloat, ContractTerm, ContractVar,
    };

    if req_disabled() {
        return false;
    }
    let Some((lo, hi)) = derive_float_bounds_from_facts(&b.facts, input_vid) else {
        return false;
    };
    // `log` is undefined for non-positive inputs: f64 `ln(0.0) = -inf`,
    // `ln(neg) = NaN`. Group 1's R prevents these from reaching here,
    // but if a stale fact slips through we must skip rather than deposit
    // an unsound bound.
    if lo <= 0.0 {
        return false;
    }
    let out_lo = lo.ln();
    let out_hi = hi.ln();
    if !out_lo.is_finite() || !out_hi.is_finite() {
        return false;
    }
    b.facts.insert_for(
        output_vid,
        ContractTerm::BoolComb {
            op: BoolOp::And,
            operands: vec![
                ContractTerm::Cmp {
                    op: CmpOp::Ge,
                    lhs: Box::new(ContractTerm::Var(ContractVar::Value(output_vid))),
                    rhs: Box::new(ContractTerm::LitFloat(ContractFloat(out_lo))),
                },
                ContractTerm::Cmp {
                    op: CmpOp::Le,
                    lhs: Box::new(ContractTerm::Var(ContractVar::Value(output_vid))),
                    rhs: Box::new(ContractTerm::LitFloat(ContractFloat(out_hi))),
                },
            ],
        },
    );
    true
}

/// Relay an interval bound through `arccos`: given `input ∈ [lo, hi]`
/// with `[lo, hi] ⊆ [-1, 1]`, plant `output ∈ [acos(hi), acos(lo)]` on
/// `output_vid`.
///
/// `arccos` is monotone-DECREASING on its domain `[-1, 1]`, so the
/// bounds swap: the output's low corner comes from the input's high
/// corner and vice versa. f64 `acos` is correctly-rounded, so the
/// relayed interval is exact (no widening needed). Returns `true` when
/// a bound was deposited.
///
/// Mirrors [`relay_sqrt_output_interval`]: facts-only lookup,
/// default-deny on non-finite corners.
pub fn relay_arccos_output_interval(
    b: &mut crate::builder::IRBuilder,
    input_vid: crate::types::ValueId,
    output_vid: crate::types::ValueId,
) -> bool {
    use crate::optim::predicates::formula::{
        BoolOp, CmpOp, ContractFloat, ContractTerm, ContractVar,
    };

    if req_disabled() {
        return false;
    }
    let Some((lo, hi)) = derive_float_bounds_from_facts(&b.facts, input_vid) else {
        return false;
    };
    // Defensive: `arccos_f`'s `requires(-1 <= x <= 1)` (Group 1's R)
    // prevents out-of-domain inputs from reaching here. If a stale fact
    // slips through, skip emit — `acos(out_of_domain)` is NaN and we
    // must never deposit an unsound bound.
    if lo < -1.0 || hi > 1.0 {
        return false;
    }
    // WHY swap: arccos is monotone-DECREASING on [-1, 1], so the
    // smallest output comes from the largest input and vice versa.
    let out_lo = hi.acos();
    let out_hi = lo.acos();
    if !out_lo.is_finite() || !out_hi.is_finite() {
        return false;
    }
    b.facts.insert_for(
        output_vid,
        ContractTerm::BoolComb {
            op: BoolOp::And,
            operands: vec![
                ContractTerm::Cmp {
                    op: CmpOp::Ge,
                    lhs: Box::new(ContractTerm::Var(ContractVar::Value(output_vid))),
                    rhs: Box::new(ContractTerm::LitFloat(ContractFloat(out_lo))),
                },
                ContractTerm::Cmp {
                    op: CmpOp::Le,
                    lhs: Box::new(ContractTerm::Var(ContractVar::Value(output_vid))),
                    rhs: Box::new(ContractTerm::LitFloat(ContractFloat(out_hi))),
                },
            ],
        },
    );
    true
}

/// Emit `output ∈ [multiplier * agg_lo, multiplier * agg_hi]` on
/// `output_vid` where `[agg_lo, agg_hi]` is the union of per-element
/// fact-derived bounds across `element_vids`. Returns `true` on emit,
/// `false` on no-op (any element unbounded, no elements, or
/// `checked_mul` overflow).
///
/// Sibling of [`super::interval_fact_for_int_binary`] for the reduction case.
/// For `np.sum`, `multiplier = N` (the element count). For `np.max` /
/// `np.min`, `multiplier = 1` (the output lives inside the element
/// bound by definition).
///
/// Soundness: only emits when every element has a visible bound. No
/// fact ⇒ no relay ⇒ no false claim. Overflow on
/// `multiplier * element_bound` likewise yields a no-op rather than a
/// wrapped (possibly unsound) value.
pub fn relay_reduction_output_interval_int(
    b: &mut crate::builder::IRBuilder,
    element_vids: &[crate::types::ValueId],
    output_vid: crate::types::ValueId,
    multiplier: i64,
) -> bool {
    use crate::optim::predicates::formula::{BoolOp, CmpOp, ContractTerm, ContractVar};

    if req_disabled() {
        return false;
    }
    let Some((agg_lo, agg_hi)) = aggregate_element_bounds(&b.facts, element_vids) else {
        return false;
    };
    // `checked_mul` rejects overflow; the helper returns false rather
    // than depositing a wrapped (and possibly unsound) bound.
    let c1 = match agg_lo.checked_mul(multiplier) {
        Some(v) => v,
        None => return false,
    };
    let c2 = match agg_hi.checked_mul(multiplier) {
        Some(v) => v,
        None => return false,
    };
    // For negative multipliers the corners flip; take min/max so the
    // emitted bound is correct regardless of sign. (Sum/max/min in v1
    // always pass non-negative multipliers, but staying generic costs
    // nothing.)
    let (out_lo, out_hi) = (c1.min(c2), c1.max(c2));

    let fact = ContractTerm::BoolComb {
        op: BoolOp::And,
        operands: vec![
            ContractTerm::Cmp {
                op: CmpOp::Ge,
                lhs: Box::new(ContractTerm::Var(ContractVar::Value(output_vid))),
                rhs: Box::new(ContractTerm::LitInt(out_lo)),
            },
            ContractTerm::Cmp {
                op: CmpOp::Le,
                lhs: Box::new(ContractTerm::Var(ContractVar::Value(output_vid))),
                rhs: Box::new(ContractTerm::LitInt(out_hi)),
            },
        ],
    };
    b.facts.insert_for(output_vid, fact);
    true
}

/// Relay `forall_eq_const(in, k)` for `k ∈ {0, 1}` from `input_vid` to
/// `output_vid` by firing the corresponding `zeros_content` / `ones_content`
/// op contract on the output.
///
/// Used by shape-preserving / element-replicating ops (`tile`, `repeat`, and
/// — once new cards land — `concatenate`, `reshape`, `transpose`, `slice`)
/// to forward the content fact that downstream reductions (`sum`, `prod`,
/// `mean`) need for compile-time specialization.
///
/// Returns `true` if a content fact was relayed, `false` otherwise. The
/// `k = 0` probe is tried first; in the pathological case where both
/// `forall_eq_const(in, 0)` and `forall_eq_const(in, 1)` are simultaneously
/// `Proved` (only possible for empty arrays), `zeros_content` wins by
/// declaration order.
pub fn relay_forall_eq_const_from_input(
    b: &mut crate::builder::IRBuilder,
    input_vid: crate::types::ValueId,
    output_vid: crate::types::ValueId,
) -> bool {
    use crate::optim::predicates::formula::{ContractTerm, ContractVar};
    use crate::optim::prove::ProveOutcome;

    if req_disabled() {
        return false;
    }
    let pred_zero = ContractTerm::PredicateApp {
        kind: "forall_eq_const".to_string(),
        args: vec![
            ContractTerm::Var(ContractVar::Value(input_vid)),
            ContractTerm::LitInt(0),
        ],
    };
    if matches!(b.prove(&pred_zero), ProveOutcome::Proved) {
        b.fire_contract("zeros_content", output_vid, &HashMap::new());
        return true;
    }
    let pred_one = ContractTerm::PredicateApp {
        kind: "forall_eq_const".to_string(),
        args: vec![
            ContractTerm::Var(ContractVar::Value(input_vid)),
            ContractTerm::LitInt(1),
        ],
    };
    if matches!(b.prove(&pred_one), ProveOutcome::Proved) {
        b.fire_contract("ones_content", output_vid, &HashMap::new());
        return true;
    }
    false
}

/// Multi-input sibling of [`relay_forall_eq_const_from_input`]. Emits a
/// content fact on `output_vid` only when *every* `input_vid` provably
/// satisfies `forall_eq_const(in_i, k)` for the same `k ∈ {0, 1}`. Used by
/// element-union ops (`concatenate`, `stack`, `vstack`, `hstack`, `dstack`,
/// `column_stack`) where the output's elements are exactly the union of the
/// inputs' elements — so a uniform-constant output requires uniform-constant
/// inputs at the same value.
///
/// Returns `true` if a content fact was emitted. Empty `input_vids` returns
/// `false` (no inputs, no derivation possible).
pub fn relay_forall_eq_const_from_all_inputs(
    b: &mut crate::builder::IRBuilder,
    input_vids: &[crate::types::ValueId],
    output_vid: crate::types::ValueId,
) -> bool {
    use crate::optim::predicates::formula::{ContractTerm, ContractVar};
    use crate::optim::prove::ProveOutcome;

    if req_disabled() {
        return false;
    }
    if input_vids.is_empty() {
        return false;
    }
    for k in &[0i64, 1i64] {
        let all_match = input_vids.iter().all(|&vid| {
            let term = ContractTerm::PredicateApp {
                kind: "forall_eq_const".to_string(),
                args: vec![
                    ContractTerm::Var(ContractVar::Value(vid)),
                    ContractTerm::LitInt(*k),
                ],
            };
            matches!(b.prove(&term), ProveOutcome::Proved)
        });
        if all_match {
            let contract = if *k == 0 { "zeros_content" } else { "ones_content" };
            b.fire_contract(contract, output_vid, &HashMap::new());
            return true;
        }
    }
    false
}
