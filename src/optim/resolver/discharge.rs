//! Index- / slice-bound discharge helpers.
//!
//! These are the load-bearing Phase-E chokepoints for "is this index /
//! slice bound provably in range?". They drop facts onto `b.facts` and
//! emit witness checks via [`crate::builder::IRBuilder::discharge_requires`]
//! when unknown.

use crate::types::Value;

use super::chokepoints::SiteKind;

/// Informational in-bound probe for dynamic array indices.
///
/// Item #4 of the `smt-invocation-load-bearing` card. Asks the resolver
/// whether `idx ∈ [lo, hi)` is provable; records telemetry tagged with
/// `SiteKind::DynamicIndexBound` regardless of outcome. Does NOT panic
/// on failure — soundness is enforced at prove time by the memory-trace
/// permutation argument. The probe exists to surface SMT engagement on
/// production-natural patterns (binary search, modular indexing, etc.)
/// where indices are runtime but bounds are nonetheless decidable.
///
/// Returns `true` if both `resolve_max(idx) < hi` and `resolve_min(idx)
/// >= lo` were proved by some layer (static_val / range / SMT).
pub fn probe_in_range(b: &mut crate::builder::IRBuilder, idx: &Value, lo: i64, hi: i64) -> bool {
    use std::sync::atomic::Ordering;
    let site_name = SiteKind::DynamicIndexBound.short_name();

    // Cheap escape: literal integer.
    if let Some(n) = idx.int_val() {
        let in_range = lo <= n && n < hi;
        // We still record the invocation so telemetry shows the call site
        // exists, but skip the resolver round-trip for the literal case.
        let (resolver, _stmts) = b.split_resolver_and_stmts();
        if let Some(t) = resolver.telemetry_handle() {
            t.record_chokepoint_invocation(site_name);
            if in_range {
                t.record_chokepoint_resolved(site_name);
            }
        }
        return in_range;
    }

    // Resolver pass — capture telemetry counters so we can detect SMT
    // engagement, then drop the resolver borrow before consulting facts.
    let (mut in_range, tel, smt_before) = {
        let (resolver, stmts) = b.split_resolver_and_stmts();
        let tel = resolver.telemetry_handle();
        let smt_before = tel.as_ref().map(|t| {
            t.queries_smt_resolved.load(Ordering::Relaxed)
                + t.queries_smt_unknown.load(Ordering::Relaxed)
        });
        if let Some(t) = tel.as_ref() {
            t.record_chokepoint_invocation(site_name);
        }
        let upper = resolver.resolve_max_with_stmts(idx, stmts);
        let lower = resolver.resolve_min_with_stmts(idx, stmts);
        let in_range = upper.map_or(false, |u| u < hi) && lower.map_or(false, |l| l >= lo);
        (in_range, tel, smt_before)
    };

    // Fact-fallback (compiler.fact-aware-chokepoints-batch): if the
    // resolver couldn't prove `idx ∈ [lo, hi)`, op-contract facts on
    // `idx.stmt_id()` may still pin it. Facts only tighten — we accept the
    // fallback iff both halves of `(min, max)` straddle `[lo, hi)`.
    if !in_range {
        if let Some((min, max)) = b.ask_bounds(idx) {
            if min >= lo && max < hi {
                in_range = true;
            }
        }
    }

    if let (Some(t), Some(before)) = (tel.as_ref(), smt_before) {
        let smt_after = t.queries_smt_resolved.load(Ordering::Relaxed)
            + t.queries_smt_unknown.load(Ordering::Relaxed);
        if smt_after > before {
            t.record_chokepoint_smt_engagement(site_name);
        }
        if in_range {
            t.record_chokepoint_resolved(site_name);
        }
    }
    in_range
}

/// Load-bearing Phase-E discharge of a slice bound (`start` or `stop`).
///
/// Slice semantics differ from scalar indexing: numpy allows `i == len`
/// for `arr[i:j]` (an empty trailing slice), so the valid range is the
/// inclusive `[0, len]`. We forward this to [`discharge_index_in_range`]
/// as `[0, len + 1)`. `None` / `Value::None` bounds (open ends like
/// `arr[:j]`) default to `0` or `len` and don't need a discharge.
///
/// Used by the dynamic-slice helpers in `static_array_read` and
/// `array_ops::indexing` to close the slice-OOB witness-miss gap
/// (compiler.fuzz-finding-v2-slice-oob-witness-miss).
pub fn discharge_slice_bound(
    b: &mut crate::builder::IRBuilder,
    bound: Option<&Value>,
    len: usize,
    op_name: &'static str,
) {
    let Some(v) = bound else {
        return;
    };
    if matches!(v, Value::None) {
        return;
    }
    discharge_index_in_range(b, v, 0, len as i64 + 1, op_name);
}

/// Load-bearing Phase-E-style discharge of `lo <= idx < hi` at an
/// indexing chokepoint (Group 5a).
///
/// Unlike [`probe_in_range`] (informational — records telemetry, returns
/// a bool callers may ignore), this function *enforces* the bound:
///
/// * **Literal idx in range**: no-op (cheap fast path).
/// * **Literal idx out of range**: panic with a diagnostic naming `op_name`,
///   the literal, and `[lo, hi)`.
/// * **Non-literal idx with a `value_id`**: build the term
///   `idx >= lo ∧ idx < hi` and discharge through
///   [`crate::builder::IRBuilder::discharge_requires`] — Phase E policy
///   handles Proved (no-op), Disproved (compile panic), and Unknown
///   (witness emit by default, panic under `ZINNIA_OP_REQUIRES_STRICT=1`).
/// * **Non-literal idx with no `value_id` and no literal value**: no-op.
///   Sound: with no anchor we have nothing to reason about and no SSA
///   wire to constrain; the runtime memory-trace permutation argument
///   still enforces address validity in the prover. Callers requiring
///   tighter compile-time enforcement must thread a `value_id` through.
///
/// The action is purely a side effect; there is no return value.
pub fn discharge_index_in_range(
    b: &mut crate::builder::IRBuilder,
    idx: &Value,
    lo: i64,
    hi: i64,
    op_name: &'static str,
) {
    use crate::optim::predicates::formula::{BoolOp, CmpOp, ContractTerm, ContractVar};
    use crate::optim::prove::ProveOutcome;

    // Literal-index fast path: compile-time decidable, panic if out of
    // range. This catches `arr[15]` on a length-10 array even when no
    // resolver is engaged.
    if let Some(n) = idx.int_val() {
        if !(lo <= n && n < hi) {
            panic!(
                "op `{}`: index {} out of range [{}, {})",
                op_name, n, lo, hi
            );
        }
        return;
    }

    // Non-literal: require a `value_id` to anchor a discharge term.
    // No value_id ⇒ no contract handle ⇒ sound no-op (the prover-side
    // permutation argument still enforces validity at proof time).
    let idx_vid = match idx.value_id() {
        Some(v) => v,
        None => return,
    };

    let term = ContractTerm::BoolComb {
        op: BoolOp::And,
        operands: vec![
            ContractTerm::Cmp {
                op: CmpOp::Ge,
                lhs: Box::new(ContractTerm::Var(ContractVar::Value(idx_vid))),
                rhs: Box::new(ContractTerm::LitInt(lo)),
            },
            ContractTerm::Cmp {
                op: CmpOp::Lt,
                lhs: Box::new(ContractTerm::Var(ContractVar::Value(idx_vid))),
                rhs: Box::new(ContractTerm::LitInt(hi)),
            },
        ],
    };
    match b.prove(&term) {
        ProveOutcome::Proved => {}
        ProveOutcome::Disproved => {
            let value_ids = crate::optim::predicates::collect_value_ids(&term);
            panic!(
                "op `{op_name}` requires precondition disproved by visible facts: \
                 term={term:?}, value_ids={value_ids:?}"
            );
        }
        ProveOutcome::Unknown => {
            // Fact-fallback: `b.prove()` can return Unknown when the fact's
            // upper bound is recorded as `Var(len_vid)` (e.g. from the
            // `dyn_argextremum` contract) but our term has `LitInt(hi)` —
            // the pattern doesn't unify even though `len_vid` constant-folds
            // to `hi`. `ask_bounds` collapses the chain and gives concrete
            // (min, max). Mirrors the same fallback in `is_index_in_range`.
            if let Some((min, max)) = b.ask_bounds(idx) {
                if min >= lo && max < hi {
                    return;
                }
            }
            if crate::builder::op_requires_strict() {
                let value_ids = crate::optim::predicates::collect_value_ids(&term);
                panic!(
                    "op `{op_name}` requires precondition not provable from visible facts \
                     (strict mode): term={term:?}, value_ids={value_ids:?}"
                );
            }
            let lo_v = b.ir_constant_int(lo);
            let hi_v = b.ir_constant_int(hi);
            emit_index_bound_assert(b, idx, &lo_v, &hi_v);
        }
    }
}

/// Load-bearing discharge of `lo <= idx < hi`, where the upper bound is
/// itself an SSA value. This is the dynamic-shape analogue of
/// [`discharge_index_in_range`]: if the fact/prove layer can discharge the
/// bound, no IR is emitted; otherwise lenient `discharge_requires` lowers the
/// comparison to an `IR::Assert` so the backend enforces it at prove time.
pub fn discharge_index_below_value(
    b: &mut crate::builder::IRBuilder,
    idx: &Value,
    lo: i64,
    hi: &Value,
    op_name: &'static str,
) {
    use crate::optim::predicates::formula::{BoolOp, CmpOp, ContractTerm, ContractVar};
    use crate::optim::prove::ProveOutcome;

    if let Some(hi_static) = hi.int_val() {
        discharge_index_in_range(b, idx, lo, hi_static, op_name);
        return;
    }

    let lhs = match idx.value_id() {
        Some(v) => ContractTerm::Var(ContractVar::Value(v)),
        None => match idx.int_val() {
            Some(n) => ContractTerm::LitInt(n),
            None => return,
        },
    };
    let hi_vid = match hi.value_id() {
        Some(v) => v,
        None => return,
    };

    let lo_term = ContractTerm::Cmp {
        op: CmpOp::Ge,
        lhs: Box::new(lhs.clone()),
        rhs: Box::new(ContractTerm::LitInt(lo)),
    };
    let hi_term = ContractTerm::Cmp {
        op: CmpOp::Lt,
        lhs: Box::new(lhs.clone()),
        rhs: Box::new(ContractTerm::Var(ContractVar::Value(hi_vid))),
    };
    let term = ContractTerm::BoolComb {
        op: BoolOp::And,
        operands: vec![lo_term.clone(), hi_term.clone()],
    };
    // Per-conjunct discharge: when the upper half is provable (e.g.
    // `idx < max(...) + 1` holds by construction) emit only the cheap
    // `idx >= lo` residual. The upper half is probed first; an Unknown
    // there falls straight through to the combined path so the common
    // unprovable case costs no extra Z3 invocations vs the old code.
    let p_hi = b.prove(&hi_term);
    if matches!(p_hi, ProveOutcome::Disproved) {
        let value_ids = crate::optim::predicates::collect_value_ids(&term);
        panic!(
            "op `{op_name}` requires precondition disproved by visible facts: \
             term={term:?}, value_ids={value_ids:?}"
        );
    }
    let p_lo = if matches!(p_hi, ProveOutcome::Proved) {
        b.prove(&lo_term)
    } else {
        ProveOutcome::Unknown
    };
    if matches!(p_lo, ProveOutcome::Disproved) {
        let value_ids = crate::optim::predicates::collect_value_ids(&term);
        panic!(
            "op `{op_name}` requires precondition disproved by visible facts: \
             term={term:?}, value_ids={value_ids:?}"
        );
    }
    match (p_lo, p_hi) {
        (ProveOutcome::Proved, ProveOutcome::Proved) => {}
        (ProveOutcome::Unknown, ProveOutcome::Proved) => {
            // Residual: idx >= lo only. lo == 0 gets the dedicated
            // non-negativity assert (a single 63-bit range check in
            // halo2-base, ~15 cells vs ~45 for the comparison form).
            if lo == 0 {
                b.ir_assert_non_neg_i(idx);
            } else {
                let lo_v = b.ir_constant_int(lo);
                let ge = b.ir_greater_than_or_equal_i(idx, &lo_v);
                b.ir_assert(&ge);
            }
        }
        (ProveOutcome::Proved, ProveOutcome::Unknown) => {
            // Residual: idx < hi only.
            let lt = b.ir_less_than_i(idx, hi);
            b.ir_assert(&lt);
        }
        _ => {
            // Upper half unprovable: the combined term cannot fare better
            // (prove decomposes conjunctions per-atom), so emit the full
            // bound assert without further solver calls.
            if std::env::var("ZINNIA_DEBUG_DISCHARGE").is_ok() {
                let vids = crate::optim::predicates::collect_value_ids(&term);
                let nfacts: usize = b.facts.visible_facts().len();
                eprintln!("[discharge-unknown] op={op_name} vids={vids:?} visible_facts={nfacts}");
                if std::env::var("ZINNIA_DEBUG_DISCHARGE").as_deref() == Ok("2") {
                    for f in b.facts.visible_facts() {
                        eprintln!("  fact: {f:?}");
                    }
                }
            }
            if crate::builder::op_requires_strict() {
                let value_ids = crate::optim::predicates::collect_value_ids(&term);
                panic!(
                    "op `{op_name}` requires precondition not provable from visible facts \
                     (strict mode): term={term:?}, value_ids={value_ids:?}"
                );
            }
            let lo_v = b.ir_constant_int(lo);
            emit_index_bound_assert(b, idx, &lo_v, hi);
        }
    }
}

fn emit_index_bound_assert(b: &mut crate::builder::IRBuilder, idx: &Value, lo: &Value, hi: &Value) {
    // Fast path: the overwhelming majority of slice/index discharges have
    // `lo == 0`. Lower as a single u128-cast bound check; backends override
    // it to whatever's cheapest (Noir collapses to one `assert((idx as u128)
    // < (hi as u128))`, saving ~20 ACIR per call vs the 2-cmp chain).
    if lo.int_val() == Some(0) {
        b.ir_assert_u128_lt(idx, hi);
        return;
    }
    // General window `lo <= idx < hi` (e.g. numpy's negative-index window
    // `-n <= idx < n`): rewrite as `0 <= idx - lo < hi - lo` and discharge
    // with a single zero-based bound check. When the span is a small constant
    // the backend lowers it to cheap bit-decomposition range checks (~2*bits
    // gates) instead of the two-comparison chain, whose signed compares each
    // force a ~126-bit `to_signed` projection. This is the Pattern-B fix for
    // permutation / fancy-index bounds (cases 0318/0319). Soundness is exact:
    // `idx - lo < 0` (idx below `lo`) or `>= hi - lo` (idx at/above `hi`) both
    // fail the bound. Falls back to the comparison chain only when a bound is
    // not a compile-time constant.
    if let (Some(lo_n), Some(hi_n)) = (lo.int_val(), hi.int_val()) {
        if hi_n > lo_n {
            let neg_lo = b.ir_constant_int(-lo_n);
            let shifted = b.ir_add_i(idx, &neg_lo);
            let span = b.ir_constant_int(hi_n - lo_n);
            b.ir_assert_u128_lt(&shifted, &span);
            return;
        }
    }
    let ge = b.ir_greater_than_or_equal_i(idx, lo);
    let lt = b.ir_less_than_i(idx, hi);
    let ok = b.ir_logical_and(&ge, &lt);
    b.ir_assert(&ok);
}
