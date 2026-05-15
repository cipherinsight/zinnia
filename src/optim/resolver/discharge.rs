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
pub fn probe_in_range(
    b: &mut crate::builder::IRBuilder,
    idx: &Value,
    lo: i64,
    hi: i64,
) -> bool {
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
    use crate::optim::predicates::formula::{
        BoolOp, CmpOp, ContractTerm, ContractVar,
    };

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
    b.discharge_requires(op_name, &term);
}
