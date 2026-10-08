//! Relay helpers: propagate interval bounds / content facts from inputs to
//! outputs of unary / reduction / shape-preserving / element-union ops.
//!
//! Each `relay_*` function reads facts off the input(s), derives a sound
//! output-bound, and deposits it on `output_vid`. All honour the
//! `ZINNIA_REQ_DISABLE` kill switch via [`super::req_disabled`].


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

/// Relay an interval bound through `arctan2`: given visible intervals
/// `y ∈ [y_lo, y_hi]` and `x ∈ [x_lo, x_hi]` on the fact stack, plant
/// a tighter output bound on `output_vid` than the static `[-π - slack,
/// π + slack]` contract.
///
/// **Soundness gate (the only subtle thing in this helper).** `arctan2`
/// is continuous everywhere *except* the negative x-axis, where it
/// jumps from `+π` (approaching from `y → 0⁺`) to `-π` (approaching
/// from `y → 0⁻`). If the input bounding box `[x_lo, x_hi] × [y_lo,
/// y_hi]` straddles that discontinuity, the corner-min-max approach is
/// unsound (it could miss values near `±π` that the function attains in
/// the interior of the box). The gate skips emit in that case.
///
/// Concretely the discontinuity lives on the open ray `(-∞, 0) × {0}`.
/// The bounding box can intersect it only when `x_lo < 0` *and*
/// `y_lo ≤ 0 ≤ y_hi`. So the safe condition is:
///
/// - `x_lo >= 0.0` (whole box is in the right half-plane; no discontinuity), OR
/// - `y_lo > 0.0` (whole box is strictly above the x-axis), OR
/// - `y_hi < 0.0` (whole box is strictly below the x-axis).
///
/// On the safe side, `arctan2` is continuous and monotone in each
/// argument separately over the box, so the extrema are attained at
/// corners and corner-min-max is exact. The card's sketch suggested
/// `x_hi >= 0 ⇒ safe`, which is unsound for the box `x ∈ [-3, 5],
/// y ∈ [-2, 2]` (true range approaches `±π`; corners only see
/// `±(π - atan(2/3))`); the corrected gate uses `x_lo >= 0` instead.
///
/// **No slack widening.** The static `arctan2_f` contract widens by
/// `ZINNIA_TRANSCENDENTAL_BOUND_SLACK_F64` to absorb the f64 round-down
/// of `std::f64::consts::PI`. Our relayed bounds are strictly interior
/// to `(-π, π)` whenever the soundness gate passes (the gate excludes
/// the only configurations that can reach `±π` exactly), so the slack
/// is not needed for soundness. We follow the existing
/// `relay_{sqrt,exp,log,arccos}` precedent and deposit the corner-
/// derived bound directly. f64 `atan2` is correctly-rounded on every
/// platform we target (IEEE-754 recommendation, libm-conformant);
/// 1-ulp drift doesn't push us through the gate.
///
/// Mirrors [`relay_sqrt_output_interval`]: facts-only lookup,
/// default-deny on non-finite corners. Returns `true` when a bound was
/// deposited.
pub fn relay_arctan2_output_interval(
    b: &mut crate::builder::IRBuilder,
    y_vid: crate::types::ValueId,
    x_vid: crate::types::ValueId,
    output_vid: crate::types::ValueId,
) -> bool {
    use crate::optim::predicates::formula::{
        BoolOp, CmpOp, ContractFloat, ContractTerm, ContractVar,
    };

    if req_disabled() {
        return false;
    }
    let Some((y_lo, y_hi)) = derive_float_bounds_from_facts(&b.facts, y_vid) else {
        return false;
    };
    let Some((x_lo, x_hi)) = derive_float_bounds_from_facts(&b.facts, x_vid) else {
        return false;
    };

    // Soundness gate — see doc-comment for the derivation.
    let safe = x_lo >= 0.0 || y_lo > 0.0 || y_hi < 0.0;
    if !safe {
        return false;
    }

    // Corner-min-max. On the safe side `arctan2` is continuous and
    // monotone in each argument, so extrema live at the 4 corners.
    let c1 = y_lo.atan2(x_lo);
    let c2 = y_lo.atan2(x_hi);
    let c3 = y_hi.atan2(x_lo);
    let c4 = y_hi.atan2(x_hi);
    let out_lo = c1.min(c2).min(c3).min(c4);
    let out_hi = c1.max(c2).max(c3).max(c4);

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

/// Relay the tight interval for a whole-array float mean. Given an input
/// aggregate bound `arr ∈ [lo, hi]` and a known static element count `n`,
/// plant `output ∈ [lo / n, hi / n]` on the mean result.
///
/// This is facts-only: it reads the already visible aggregate bound on the
/// input array value and deposits a derived fact on the existing mean output.
/// Missing bounds, non-positive counts, or non-finite arithmetic skip emit.
pub fn relay_mean_tight_interval_float(
    b: &mut crate::builder::IRBuilder,
    input_vid: crate::types::ValueId,
    n: i64,
    output_vid: crate::types::ValueId,
) -> bool {
    use crate::optim::predicates::formula::{
        BoolOp, CmpOp, ContractFloat, ContractTerm, ContractVar,
    };

    if req_disabled() || n <= 0 {
        return false;
    }
    let Some((lo, hi)) = derive_float_bounds_from_facts(&b.facts, input_vid) else {
        return false;
    };
    let n_f = n as f64;
    let out_lo = lo / n_f;
    let out_hi = hi / n_f;
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

fn emit_int_interval_fact(
    b: &mut crate::builder::IRBuilder,
    output_vid: crate::types::ValueId,
    out_lo: i64,
    out_hi: i64,
) {
    use crate::optim::predicates::formula::{BoolOp, CmpOp, ContractTerm, ContractVar};

    b.facts.insert_for(
        output_vid,
        ContractTerm::BoolComb {
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
        },
    );
}

/// Compute the exact static-length integer `prod` interval for elements whose
/// aggregate bound is `[lo, hi]`.
///
/// Preconditions are intentionally explicit: `n` is a statically known element
/// count. `n = 0` yields the product identity `[1, 1]`. Callers with runtime-
/// symbolic or parity-unknown lengths must skip this helper rather than guess,
/// because the all-non-positive case depends on parity.
///
/// Returns `None` for mixed-sign bounds and for any checked arithmetic
/// overflow. This keeps the relay conservative: no emit is always sound.
pub fn prod_interval_int_for_static_len(lo: i64, hi: i64, n: u32) -> Option<(i64, i64)> {
    if n == 0 {
        return Some((1, 1));
    }
    if lo >= 0 {
        return Some((lo.checked_pow(n)?, hi.checked_pow(n)?));
    }
    if hi <= 0 {
        let min_abs = hi.checked_neg()?;
        let max_abs = lo.checked_neg()?;
        let min_abs_pow = min_abs.checked_pow(n)?;
        let max_abs_pow = max_abs.checked_pow(n)?;
        if n % 2 == 0 {
            Some((min_abs_pow, max_abs_pow))
        } else {
            Some((max_abs_pow.checked_neg()?, min_abs_pow.checked_neg()?))
        }
    } else {
        None
    }
}

/// Relay a sign-aware integer interval through static `prod`.
///
/// This is facts-only: every element must have a visible integer interval.
/// Mixed-sign aggregate input bounds skip emit. The length is the static
/// element count (`element_vids.len()`), so the non-positive parity split is
/// well-defined here; future dynamic-length callers must skip when parity is
/// unknown.
pub fn relay_prod_output_interval_int(
    b: &mut crate::builder::IRBuilder,
    element_vids: &[crate::types::ValueId],
    output_vid: crate::types::ValueId,
) -> bool {
    if req_disabled() {
        return false;
    }
    let Some((agg_lo, agg_hi)) = aggregate_element_bounds(&b.facts, element_vids) else {
        return false;
    };
    let Ok(n) = u32::try_from(element_vids.len()) else {
        return false;
    };
    let Some((out_lo, out_hi)) = prod_interval_int_for_static_len(agg_lo, agg_hi, n) else {
        return false;
    };
    emit_int_interval_fact(b, output_vid, out_lo, out_hi);
    true
}

/// Relay `forall_eq_const(in, k)` for `k ∈ {0, 1}` from `input` to
/// `output` by stating the corresponding `zeros_content` / `ones_content`
/// op contract on the output.
///
/// Used by shape-preserving / element-replicating ops (`tile`, `repeat`,
/// `reshape`, `transpose`, slices, ...) to forward the content fact that
/// downstream reductions (`sum`, `prod`, `mean`) need for compile-time
/// specialization. The input's fact is not proved once the input was
/// written since it was stated, and the output's fact is stated with a
/// record of what the output holds now, so it lapses when the output (or an
/// array sharing its storage) is written (`builder/array_facts.rs`).
///
/// Returns `true` if a content fact was relayed, `false` otherwise. The
/// `k = 0` probe is tried first; in the pathological case where both
/// `forall_eq_const(in, 0)` and `forall_eq_const(in, 1)` are simultaneously
/// `Proved` (only possible for empty arrays), `zeros_content` wins by
/// declaration order.
pub fn relay_forall_eq_const_from_input(
    b: &mut crate::builder::IRBuilder,
    input: &crate::types::Value,
    output: &crate::types::Value,
) -> bool {
    relay_forall_eq_const_from_all_inputs(b, std::slice::from_ref(input), output)
}

/// Multi-input sibling of [`relay_forall_eq_const_from_input`]. Emits a
/// content fact on `output` only when *every* input provably satisfies
/// `forall_eq_const(in_i, k)` for the same `k ∈ {0, 1}`. Used by
/// element-union ops (`concatenate`, `stack`, `vstack`, `hstack`, `dstack`,
/// `column_stack`) where the output's elements are exactly the union of the
/// inputs' elements — so a uniform-constant output requires uniform-constant
/// inputs at the same value.
///
/// Returns `true` if a content fact was emitted. No inputs, or an input or
/// output without a value id, returns `false` (no derivation possible).
pub fn relay_forall_eq_const_from_all_inputs(
    b: &mut crate::builder::IRBuilder,
    inputs: &[crate::types::Value],
    output: &crate::types::Value,
) -> bool {
    use crate::optim::predicates::formula::{ContractTerm, ContractVar};
    use crate::optim::prove::ProveOutcome;

    if req_disabled() || output.value_id().is_none() {
        return false;
    }
    let Some(input_vids) = inputs.iter().map(|v| v.value_id()).collect::<Option<Vec<_>>>() else {
        return false;
    };
    if input_vids.is_empty() {
        return false;
    }
    for (k, contract) in [(0i64, "zeros_content"), (1i64, "ones_content")] {
        let all_match = input_vids.iter().all(|&vid| {
            let term = ContractTerm::PredicateApp {
                kind: "forall_eq_const".to_string(),
                args: vec![ContractTerm::Var(ContractVar::Value(vid)), ContractTerm::LitInt(k)],
            };
            matches!(b.prove(&term), ProveOutcome::Proved)
        });
        if all_match {
            b.state_array_fact(output, contract);
            return true;
        }
    }
    false
}
