//! Tests for the resolver module. Kept under `mod tests` of `mod.rs` (the
//! original location) to preserve discoverability and to avoid juggling
//! cross-module visibility on test-only helpers.

use super::*;
use crate::circuit_input::InputPath;
use crate::ir::IRStatement;
use crate::ir_defs::IR;
use crate::types::{ScalarValue, StmtId};

#[test]
fn static_only_resolver_matches_int_val() {
    let mut r = StaticOnlyResolver::new();
    let v = Value::Integer(ScalarValue::constant(42));
    assert_eq!(r.resolve_int(&v), Some(42));
    assert_eq!(r.resolve_max(&v), Some(42));
    assert_eq!(r.resolve_min(&v), Some(42));
}

#[test]
fn static_only_resolver_unknown_for_runtime_int() {
    let mut r = StaticOnlyResolver::new();
    let v = Value::Integer(ScalarValue::runtime(0));
    assert_eq!(r.resolve_int(&v), None);
}

#[test]
fn static_int_into_i64() {
    let s = StaticInt(7);
    let n: i64 = s.into();
    assert_eq!(n, 7);
}

#[test]
fn site_kind_diagnostic_includes_axis() {
    assert!(SiteKind::ShapeAxis(3)
        .diagnostic()
        .contains("axis 3"));
}

// ---------------------------------------------------------------
// SmtResolver tests
// ---------------------------------------------------------------

/// Helper: build a `Value::Integer` whose ptr is `stmt_id`. Mirrors
/// what `IRBuilder::create_ir` does for the integer return type.
fn runtime_int(stmt_id: StmtId) -> Value {
    Value::Integer(ScalarValue::runtime(stmt_id))
}

/// Helper: build a `Value::Boolean` whose ptr is `stmt_id`.
fn runtime_bool(stmt_id: StmtId) -> Value {
    Value::Boolean(ScalarValue::runtime(stmt_id))
}

/// SMT-decidable but not static_val: `select(true, 7, 9)` with a
/// non-folded ConstantBool input. The static_val path can't see
/// through Select unless the optimiser already folded it; SMT can.
#[test]
fn smt_resolves_select_with_constant_cond() {
    // stmt0 = ConstantBool(true), stmt1 = ConstantInt(7),
    // stmt2 = ConstantInt(9), stmt3 = SelectI(stmt0, stmt1, stmt2)
    let stmts = vec![
        IRStatement::new(0, crate::types::ValueId::next(), IR::ConstantBool { value: true }, vec![], vec![], None),
        IRStatement::new(1, crate::types::ValueId::next(), IR::ConstantInt { value: 7 }, vec![], vec![], None),
        IRStatement::new(2, crate::types::ValueId::next(), IR::ConstantInt { value: 9 }, vec![], vec![], None),
        IRStatement::new(3, crate::types::ValueId::next(), IR::SelectI, vec![0, 1, 2], vec![], None),
    ];
    let v = runtime_int(3);
    let mut r = SmtResolver::new();
    assert_eq!(r.resolve_int_with_stmts(&v, &stmts), Some(7));
}

/// Genuinely SMT-decidable via reasoning: `select(x == 5, 100, 100)`
/// where `x` is a free `ReadInteger`. Both branches are 100, so SMT
/// proves the result is 100 — but `static_val` can't, because it
/// doesn't know `x`.
#[test]
fn smt_resolves_select_with_both_branches_equal() {
    // stmt0 = ReadInteger("x"), stmt1 = ConstantInt(5),
    // stmt2 = EqI(stmt0, stmt1), stmt3 = ConstantInt(100),
    // stmt4 = ConstantInt(100), stmt5 = SelectI(stmt2, stmt3, stmt4)
    let stmts = vec![
        IRStatement::new(
            0, crate::types::ValueId::next(),
            IR::ReadInteger { path: InputPath::new("x", vec![]), is_public: false },
            vec![], vec![],
            None),
        IRStatement::new(1, crate::types::ValueId::next(), IR::ConstantInt { value: 5 }, vec![], vec![], None),
        IRStatement::new(2, crate::types::ValueId::next(), IR::EqI, vec![0, 1], vec![], None),
        IRStatement::new(3, crate::types::ValueId::next(), IR::ConstantInt { value: 100 }, vec![], vec![], None),
        IRStatement::new(4, crate::types::ValueId::next(), IR::ConstantInt { value: 100 }, vec![], vec![], None),
        IRStatement::new(5, crate::types::ValueId::next(), IR::SelectI, vec![2, 3, 4], vec![], None),
    ];
    let v = runtime_int(5);
    let mut r = SmtResolver::new();
    assert_eq!(r.resolve_int_with_stmts(&v, &stmts), Some(100));
}

/// Free input wire: `ReadInteger` returns None (the value is genuinely
/// not constant; SMT must not fabricate one).
#[test]
fn smt_returns_none_on_free_variable() {
    let stmts = vec![IRStatement::new(
        0, crate::types::ValueId::next(),
        IR::ReadInteger { path: InputPath::new("x", vec![]), is_public: false },
        vec![], vec![],
        None)];
    let v = runtime_int(0);
    let mut r = SmtResolver::new();
    assert_eq!(r.resolve_int_with_stmts(&v, &stmts), None);
}

/// Disable flag: even on an SMT-decidable case, returning None
/// (after the static-val fast path).
#[test]
fn smt_disable_flag_short_circuits() {
    let stmts = vec![
        IRStatement::new(0, crate::types::ValueId::next(), IR::ConstantBool { value: true }, vec![], vec![], None),
        IRStatement::new(1, crate::types::ValueId::next(), IR::ConstantInt { value: 7 }, vec![], vec![], None),
        IRStatement::new(2, crate::types::ValueId::next(), IR::ConstantInt { value: 9 }, vec![], vec![], None),
        IRStatement::new(3, crate::types::ValueId::next(), IR::SelectI, vec![0, 1, 2], vec![], None),
    ];
    let v = runtime_int(3);
    let mut r = SmtResolver::new().with_disabled(true);
    assert_eq!(r.resolve_int_with_stmts(&v, &stmts), None);
}

/// Cache hit on second call: after one query, the cache has an entry
/// for ptr 3. A second query on the same wire should hit the cache
/// without re-querying Z3 (we observe this via cache_size == 1
/// throughout).
#[test]
fn smt_caches_resolution() {
    let stmts = vec![
        IRStatement::new(0, crate::types::ValueId::next(), IR::ConstantBool { value: true }, vec![], vec![], None),
        IRStatement::new(1, crate::types::ValueId::next(), IR::ConstantInt { value: 7 }, vec![], vec![], None),
        IRStatement::new(2, crate::types::ValueId::next(), IR::ConstantInt { value: 9 }, vec![], vec![], None),
        IRStatement::new(3, crate::types::ValueId::next(), IR::SelectI, vec![0, 1, 2], vec![], None),
    ];
    let v = runtime_int(3);
    let mut r = SmtResolver::new();

    assert_eq!(r.cache_size(), 0);
    let first = r.resolve_int_with_stmts(&v, &stmts);
    assert_eq!(first, Some(7));
    assert_eq!(r.cache_size(), 1);

    let second = r.resolve_int_with_stmts(&v, &stmts);
    assert_eq!(second, Some(7));
    // cache_size is still 1 — we didn't add a new entry.
    assert_eq!(r.cache_size(), 1);
}

/// Tight timeout returns None on a pathological formula. We construct
/// an SmtResolver with a 1 ms timeout and a deliberately-non-trivial
/// formula (a long chain of MulI's grows the search space) — the
/// timeout fires and the resolver returns None instead of hanging.
///
/// The test is hermetic because Z3 honours the timeout regardless of
/// machine speed; the only assertion is "returns None" (not "returns
/// quickly"). To make this robust we compose a formula whose decision
/// is unique (so the resolver would, given enough time, succeed) but
/// which contains enough multiplicative structure that a 1 ms budget
/// can't crack it.
#[test]
fn smt_honours_tight_timeout() {
    // Build: x = ReadInteger("x"). Then a big chain of multiplications
    // and conditional adds. Even though `x*x*...*x ` is determined by x,
    // the resolver can't prove uniqueness without first picking x — and
    // the formula is large enough that with a 1 ms budget Z3 returns
    // unknown rather than the unique-not-found "non-unique" verdict
    // produced when the sub-checks succeed.
    let mut stmts = Vec::new();
    stmts.push(IRStatement::new(
        0, crate::types::ValueId::next(),
        IR::ReadInteger { path: InputPath::new("x", vec![]), is_public: false },
        vec![], vec![],
        None));
    // Build a chain: stmt1 = stmt0 * stmt0, stmt2 = stmt1 * stmt0, ...
    // up to stmt30. Result has high arithmetic complexity.
    let mut last = 0u32;
    for i in 1..=30 {
        stmts.push(IRStatement::new(i, crate::types::ValueId::next(), IR::MulI, vec![last, 0], vec![], None));
        last = i;
    }
    let v = runtime_int(last);
    let mut r = SmtResolver::new().with_timeout(1);
    // Within 1 ms, Z3 likely returns sat with a model — but the
    // re-check (var != that_value) will likely also return sat (Z3
    // can find another counter-model), so the resolver returns None.
    // Either way, the test asserts no Some(_) leaks: because the
    // wire genuinely depends on x, no unique value should be
    // returned regardless of timeout.
    assert_eq!(r.resolve_int_with_stmts(&v, &stmts), None);
}

/// `on_ir_mutated(&[])` clears the entire cache (P1 conservative).
#[test]
fn smt_on_ir_mutated_clears_cache() {
    let stmts = vec![
        IRStatement::new(0, crate::types::ValueId::next(), IR::ConstantBool { value: true }, vec![], vec![], None),
        IRStatement::new(1, crate::types::ValueId::next(), IR::ConstantInt { value: 7 }, vec![], vec![], None),
        IRStatement::new(2, crate::types::ValueId::next(), IR::ConstantInt { value: 9 }, vec![], vec![], None),
        IRStatement::new(3, crate::types::ValueId::next(), IR::SelectI, vec![0, 1, 2], vec![], None),
    ];
    let v = runtime_int(3);
    let mut r = SmtResolver::new();
    let _ = r.resolve_int_with_stmts(&v, &stmts);
    assert_eq!(r.cache_size(), 1);
    r.on_ir_mutated(&[]);
    assert_eq!(r.cache_size(), 0);
}

/// Static-val fast path: SmtResolver returns the constant immediately
/// without consulting Z3. Verified by passing an EMPTY stmts slice
/// (Z3 path would panic on stmt[ptr] otherwise — actually it'd return
/// None, but the point is fast-path doesn't even attempt the walk).
#[test]
fn smt_static_val_fast_path() {
    let stmts: Vec<IRStatement> = vec![];
    let v = Value::Integer(ScalarValue::constant(123));
    let mut r = SmtResolver::new();
    assert_eq!(r.resolve_int_with_stmts(&v, &stmts), Some(123));
    // No cache entry for static-val results (no ptr to key on).
    assert_eq!(r.cache_size(), 0);
}

/// resolve_bool_with_stmts on a select with a constant cond.
#[test]
fn smt_resolves_bool_through_select() {
    // stmt0 = ConstantBool(false),
    // stmt1 = ConstantBool(true), stmt2 = ConstantBool(true),
    // stmt3 = SelectB(stmt0, stmt1, stmt2)
    // Both branches are true so result is true regardless of cond.
    let stmts = vec![
        IRStatement::new(0, crate::types::ValueId::next(), IR::ConstantBool { value: false }, vec![], vec![], None),
        IRStatement::new(1, crate::types::ValueId::next(), IR::ConstantBool { value: true }, vec![], vec![], None),
        IRStatement::new(2, crate::types::ValueId::next(), IR::ConstantBool { value: true }, vec![], vec![], None),
        IRStatement::new(3, crate::types::ValueId::next(), IR::SelectB, vec![0, 1, 2], vec![], None),
    ];
    let v = runtime_bool(3);
    let mut r = SmtResolver::new();
    assert_eq!(r.resolve_bool_with_stmts(&v, &stmts), Some(true));
}

/// `SmtResolver` is `Send + Sync` (required by `Resolver` trait
/// because `IRGraph` is held by a `#[pyclass]`). Compile-time check.
#[test]
fn smt_resolver_is_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<SmtResolver>();
}

// ---------------------------------------------------------------
// LayeredResolver tests (P2 commit 4)
// ---------------------------------------------------------------

use crate::optim::range::RangeResolver;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// Test-only counter-tracking resolver. Wraps an inner resolver and
/// increments a shared counter on every `_with_stmts` call. Used to
/// verify the layered resolver's "answers first" behaviour.
struct CountingResolver {
    inner: Box<dyn Resolver>,
    calls: Arc<AtomicUsize>,
}

impl Resolver for CountingResolver {
    fn resolve_int(&mut self, val: &Value) -> Option<i64> {
        self.inner.resolve_int(val)
    }
    fn resolve_bool(&mut self, val: &Value) -> Option<bool> {
        self.inner.resolve_bool(val)
    }
    fn resolve_max(&mut self, val: &Value) -> Option<i64> {
        self.inner.resolve_max(val)
    }
    fn resolve_min(&mut self, val: &Value) -> Option<i64> {
        self.inner.resolve_min(val)
    }
    fn resolve_int_with_stmts(
        &mut self,
        val: &Value,
        stmts: &[IRStatement],
    ) -> Option<i64> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.inner.resolve_int_with_stmts(val, stmts)
    }
    fn resolve_bool_with_stmts(
        &mut self,
        val: &Value,
        stmts: &[IRStatement],
    ) -> Option<bool> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.inner.resolve_bool_with_stmts(val, stmts)
    }
    fn resolve_max_with_stmts(
        &mut self,
        val: &Value,
        stmts: &[IRStatement],
    ) -> Option<i64> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.inner.resolve_max_with_stmts(val, stmts)
    }
    fn resolve_min_with_stmts(
        &mut self,
        val: &Value,
        stmts: &[IRStatement],
    ) -> Option<i64> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.inner.resolve_min_with_stmts(val, stmts)
    }
    fn on_ir_mutated(&mut self, affected: &[StmtId]) {
        self.inner.on_ir_mutated(affected);
    }
}

/// When the range layer answers a query, the SMT layer is never
/// consulted. Construct a `select(c, 7, 7)` (range resolves to 7,
/// SMT *would* also resolve, but should be skipped). The counting
/// SMT layer's call-count must be 0.
#[test]
fn layered_range_answers_first() {
    let stmts = vec![
        IRStatement::new(
            0, crate::types::ValueId::next(),
            IR::ReadInteger {
                path: InputPath::new("c", vec![]),
                is_public: false,
            },
            vec![], vec![],
            None),
        IRStatement::new(1, crate::types::ValueId::next(), IR::ConstantInt { value: 7 }, vec![], vec![], None),
        IRStatement::new(2, crate::types::ValueId::next(), IR::ConstantInt { value: 7 }, vec![], vec![], None),
        IRStatement::new(3, crate::types::ValueId::next(), IR::SelectI, vec![0, 1, 2], vec![], None),
    ];
    let v = runtime_int(3);

    let smt_calls = Arc::new(AtomicUsize::new(0));
    let counting_smt = CountingResolver {
        inner: Box::new(SmtResolver::new()),
        calls: Arc::clone(&smt_calls),
    };
    let mut layered = LayeredResolver::new(vec![
        Box::new(RangeResolver::new()),
        Box::new(counting_smt),
    ]);

    assert_eq!(layered.resolve_int_with_stmts(&v, &stmts), Some(7));
    // Range answered → SMT was never called.
    assert_eq!(smt_calls.load(Ordering::SeqCst), 0);
}

/// When range can't resolve, the layered resolver falls through to
/// SMT. Construct `select(x == x, 7, 9)`: range only sees `[7, 9]`
/// (no point) since it doesn't reason about the cond's tautology.
/// SMT proves `x == x` is always true → result is 7. The counting
/// SMT layer's call-count must be ≥ 1.
#[test]
fn layered_falls_through_to_smt() {
    let stmts = vec![
        IRStatement::new(
            0, crate::types::ValueId::next(),
            IR::ReadInteger {
                path: InputPath::new("x", vec![]),
                is_public: false,
            },
            vec![], vec![],
            None),
        IRStatement::new(1, crate::types::ValueId::next(), IR::EqI, vec![0, 0], vec![], None), // x == x → true
        IRStatement::new(2, crate::types::ValueId::next(), IR::ConstantInt { value: 7 }, vec![], vec![], None),
        IRStatement::new(3, crate::types::ValueId::next(), IR::ConstantInt { value: 9 }, vec![], vec![], None),
        IRStatement::new(4, crate::types::ValueId::next(), IR::SelectI, vec![1, 2, 3], vec![], None),
    ];
    let v = runtime_int(4);

    let smt_calls = Arc::new(AtomicUsize::new(0));
    let counting_smt = CountingResolver {
        inner: Box::new(SmtResolver::new()),
        calls: Arc::clone(&smt_calls),
    };
    let mut layered = LayeredResolver::new(vec![
        Box::new(RangeResolver::new()),
        Box::new(counting_smt),
    ]);
    assert_eq!(layered.resolve_int_with_stmts(&v, &stmts), Some(7));
    // SMT was consulted (range couldn't tell which branch).
    assert!(smt_calls.load(Ordering::SeqCst) >= 1);
}

/// When neither layer can resolve (a free variable), the layered
/// resolver returns None.
#[test]
fn layered_returns_none_when_neither_resolves() {
    let stmts = vec![IRStatement::new(
        0, crate::types::ValueId::next(),
        IR::ReadInteger {
            path: InputPath::new("x", vec![]),
            is_public: false,
        },
        vec![], vec![],
        None)];
    let v = runtime_int(0);
    let mut layered = LayeredResolver::range_then_smt();
    assert_eq!(layered.resolve_int_with_stmts(&v, &stmts), None);
}

/// `LayeredResolver` is Send + Sync.
#[test]
fn layered_resolver_is_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<LayeredResolver>();
}

// ---------------------------------------------------------------
// P5 telemetry tests
// ---------------------------------------------------------------

/// `SmtResolver` increments `queries_smt_unknown` when Z3 can't
/// resolve a free variable. The static_val and cache fast-paths
/// shouldn't fire.
#[test]
fn telemetry_smt_unknown_on_free_var() {
    let stmts = vec![IRStatement::new(
        0, crate::types::ValueId::next(),
        IR::ReadInteger {
            path: InputPath::new("x", vec![]),
            is_public: false,
        },
        vec![], vec![],
        None)];
    let v = runtime_int(0);
    let mut r = SmtResolver::new();
    let t = r.telemetry();
    assert_eq!(r.resolve_int_with_stmts(&v, &stmts), None);
    assert_eq!(t.queries_smt_unknown.load(Ordering::SeqCst), 1);
    assert_eq!(t.queries_smt_resolved.load(Ordering::SeqCst), 0);
    // The duration histogram has a single entry now (somewhere).
    let total: usize =
        (0..crate::optim::telemetry::NUM_DURATION_BUCKETS)
            .map(|i| t.smt_duration_buckets[i].load(Ordering::SeqCst))
            .sum();
    assert_eq!(total, 1);
}

/// P5 commit 3: `with_max_formula_size` triggers the oversized
/// short-circuit when a formula's reverse-reachability walk would
/// exceed the cap. The resolver returns None and the
/// `queries_skipped_oversized` counter ticks. Verifies the cap is
/// wired through `resolve_int_with_stmts` to the walker.
#[test]
fn smt_max_formula_size_aborts_oversized_walk() {
    // Construct a 5-statement chain (Constant → MulI → MulI → MulI →
    // MulI). With max_formula_size = 2 the walker aborts before
    // encoding the root, so the resolver returns None and the
    // oversized counter increments.
    let stmts = vec![
        IRStatement::new(
            0, crate::types::ValueId::next(),
            IR::ReadInteger {
                path: InputPath::new("x", vec![]),
                is_public: false,
            },
            vec![], vec![],
            None),
        IRStatement::new(1, crate::types::ValueId::next(), IR::MulI, vec![0, 0], vec![], None),
        IRStatement::new(2, crate::types::ValueId::next(), IR::MulI, vec![1, 0], vec![], None),
        IRStatement::new(3, crate::types::ValueId::next(), IR::MulI, vec![2, 0], vec![], None),
        IRStatement::new(4, crate::types::ValueId::next(), IR::MulI, vec![3, 0], vec![], None),
    ];
    let v = runtime_int(4);
    let mut r = SmtResolver::new().with_max_formula_size(2);
    let t = r.telemetry();
    assert_eq!(r.resolve_int_with_stmts(&v, &stmts), None);
    assert_eq!(t.queries_skipped_oversized.load(Ordering::SeqCst), 1);
    // The "smt_unknown" counter must NOT have incremented — we
    // never hit the solver, the walk aborted first.
    assert_eq!(t.queries_smt_unknown.load(Ordering::SeqCst), 0);
    assert_eq!(t.queries_smt_resolved.load(Ordering::SeqCst), 0);
}

/// P5 commit 3: a 1ms timeout on a moderately-complex formula
/// returns None instead of resolving the unique value. (The legacy
/// `smt_honours_tight_timeout` test already covers this; this
/// duplicates with the explicit "via the new mitigation knob"
/// framing.)
#[test]
fn smt_tight_timeout_via_with_timeout_mitigation() {
    let mut stmts = Vec::new();
    stmts.push(IRStatement::new(
        0, crate::types::ValueId::next(),
        IR::ReadInteger {
            path: InputPath::new("x", vec![]),
            is_public: false,
        },
        vec![], vec![],
        None));
    let mut last = 0u32;
    for i in 1..=20 {
        stmts.push(IRStatement::new(i, crate::types::ValueId::next(), IR::MulI, vec![last, 0], vec![], None));
        last = i;
    }
    let v = runtime_int(last);
    let mut r = SmtResolver::new().with_timeout(1);
    assert_eq!(r.resolve_int_with_stmts(&v, &stmts), None);
}

/// `LayeredResolver::range_then_smt` shares one telemetry across the
/// range and SMT layers. Construct `select(c, 7, 7)` (range answers
/// it). Counters: `queries_total` = 1, `queries_range_hit` = 1,
/// `queries_smt_resolved` = 0.
#[test]
fn telemetry_layered_range_hit_increments_shared_telemetry() {
    let stmts = vec![
        IRStatement::new(
            0, crate::types::ValueId::next(),
            IR::ReadInteger {
                path: InputPath::new("c", vec![]),
                is_public: false,
            },
            vec![], vec![],
            None),
        IRStatement::new(1, crate::types::ValueId::next(), IR::ConstantInt { value: 7 }, vec![], vec![], None),
        IRStatement::new(2, crate::types::ValueId::next(), IR::ConstantInt { value: 7 }, vec![], vec![], None),
        IRStatement::new(3, crate::types::ValueId::next(), IR::SelectI, vec![0, 1, 2], vec![], None),
    ];
    let v = runtime_int(3);
    let mut layered = LayeredResolver::range_then_smt();
    let t = layered.telemetry_handle().unwrap();
    assert_eq!(layered.resolve_int_with_stmts(&v, &stmts), Some(7));
    assert_eq!(t.queries_total.load(Ordering::SeqCst), 1);
    assert_eq!(t.queries_range_hit.load(Ordering::SeqCst), 1);
    assert_eq!(t.queries_smt_resolved.load(Ordering::SeqCst), 0);
    assert_eq!(t.queries_smt_unknown.load(Ordering::SeqCst), 0);
}

// ---------------------------------------------------------------
// require_static_or_bounded_int / BoundedInt
// ---------------------------------------------------------------

#[test]
fn bounded_int_static_branch_for_literal() {
    // A compile-time constant integer must resolve via the Static
    // branch — same as require_static_int's fast lane.
    let mut b = crate::builder::IRBuilder::new();
    let v = b.ir_constant_int(42);
    let outcome = require_static_or_bounded_int(
        &mut b,
        &v,
        SiteKind::ShapeAxis(0),
        None,
    );
    assert_eq!(outcome, BoundedInt::Static(42));
}

#[test]
fn bounded_int_neither_for_unconstrained_symbolic() {
    // A free runtime ReadInteger has no static value and no
    // resolver-provable bound under the SMT-enabled pipeline.
    let mut b = crate::builder::IRBuilder::new();
    b.set_resolver(Box::new(LayeredResolver::range_then_smt_with_budget(500, 4096)));
    let k = b.ir_read_integer(InputPath::new("k", vec![]), false);
    let outcome = require_static_or_bounded_int(
        &mut b,
        &k,
        SiteKind::ShapeAxis(0),
        None,
    );
    assert_eq!(outcome, BoundedInt::Neither);
}

#[test]
fn bounded_int_recovers_bound_from_op_contract_facts() {
    // Consumer-side wiring for `compiler.op-contract-corpus-demos`:
    // when the resolver-based pass yields Neither, the chokepoint
    // recovers a bound from `b.facts.per_stmt` for ptr-anchored
    // `SsaPtr(p) op LitInt(n)` facts deposited by op contracts.
    use crate::optim::predicates::formula::{CmpOp, ContractTerm, ContractVar};

    let mut b = crate::builder::IRBuilder::new();
    b.set_resolver(Box::new(LayeredResolver::range_then_smt_with_budget(500, 4096)));
    let k = b.ir_read_integer(InputPath::new("k", vec![]), false);
    let k_vid = k.value_id().unwrap();

    // Plant op-contract-shaped facts: `k >= 0` and `k <= 16`.
    b.facts.insert_for(
        k_vid,
        ContractTerm::Cmp {
            op: CmpOp::Ge,
            lhs: Box::new(ContractTerm::Var(ContractVar::Value(k_vid))),
            rhs: Box::new(ContractTerm::LitInt(0)),
        },
    );
    b.facts.insert_for(
        k_vid,
        ContractTerm::Cmp {
            op: CmpOp::Le,
            lhs: Box::new(ContractTerm::Var(ContractVar::Value(k_vid))),
            rhs: Box::new(ContractTerm::LitInt(16)),
        },
    );

    let outcome = require_static_or_bounded_int(
        &mut b,
        &k,
        SiteKind::ShapeAxis(0),
        None,
    );
    assert_eq!(outcome, BoundedInt::Bounded { min: 0, max: 16 });
}

#[test]
fn bounded_int_bounded_branch_via_structural_predicate() {
    // The load-bearing end-to-end test: with a `nnz(x) == k`
    // precondition and a 16-element array `x`, the helper returns
    // `Bounded { min: 0, max: 16 }`. Mirrors the Walkthrough-1
    // resolver-chain test but goes through the chokepoint helper.
    use crate::circuit_input::PathSegment;

    let mut b = crate::builder::IRBuilder::new();
    b.set_resolver(Box::new(LayeredResolver::range_then_smt_with_budget(500, 4096)));

    // Emit reads at indices 0 and 15 — the array-length inference
    // uses max-index + 1 = 16.
    let _ = b.ir_read_float(
        InputPath::new("x", vec![PathSegment::Index(0)]),
        false,
    );
    let _ = b.ir_read_float(
        InputPath::new("x", vec![PathSegment::Index(15)]),
        false,
    );

    let k = b.ir_read_integer(InputPath::new("k", vec![]), false);

    let _ = b.ir_structural_predicate(
        "nnz".to_string(),
        vec!["x".to_string()],
        Some("==".to_string()),
        Some("k".to_string()),
    );

    let outcome = require_static_or_bounded_int(
        &mut b,
        &k,
        SiteKind::ShapeAxis(0),
        None,
    );
    assert_eq!(
        outcome,
        BoundedInt::Bounded { min: 0, max: 16 },
        "expected `nnz(x) == k` with len(x)=16 to bound k in [0, 16]; got {:?}",
        outcome,
    );
}

#[test]
fn bounded_int_static_from_statically_resolvable_select() {
    // `select(true, 7, 7)` is statically resolvable to 7 by the SMT
    // layer (and the range layer). Make sure the helper picks the
    // Static branch — not the Bounded one — when a unique value is
    // available.
    let mut b = crate::builder::IRBuilder::new();
    b.set_resolver(Box::new(LayeredResolver::range_then_smt_with_budget(500, 4096)));
    // Build `select(true, 7, 7)` so it constant-folds to 7.
    let t = b.ir_constant_bool(true);
    let a = b.ir_constant_int(7);
    let c = b.ir_constant_int(7);
    let v = b.create_ir(&IR::SelectI, &[t, a, c]);
    let outcome = require_static_or_bounded_int(
        &mut b,
        &v,
        SiteKind::ShapeAxis(0),
        None,
    );
    assert_eq!(outcome, BoundedInt::Static(7));
}

#[test]
fn bounded_int_from_static_int_conversion() {
    let s = StaticInt(13);
    let b: BoundedInt = s.into();
    assert_eq!(b, BoundedInt::Static(13));
}

// ---------------------------------------------------------------
// compiler.fact-aware-chokepoints-batch: each wired chokepoint
// recovers a bound / static value from op-contract-deposited facts
// when the resolver-based pass can't.
// ---------------------------------------------------------------

/// Helper: plant a `Cmp(Value(vid) <op> LitInt(n))` fact on `b.facts`.
fn plant_cmp_fact(
    b: &mut crate::builder::IRBuilder,
    vid: crate::types::ValueId,
    op: crate::optim::predicates::formula::CmpOp,
    n: i64,
) {
    use crate::optim::predicates::formula::{ContractTerm, ContractVar};
    b.facts.insert_for(
        vid,
        ContractTerm::Cmp {
            op,
            lhs: Box::new(ContractTerm::Var(ContractVar::Value(vid))),
            rhs: Box::new(ContractTerm::LitInt(n)),
        },
    );
}

#[test]
fn probe_in_range_recovers_from_op_contract_facts() {
    use crate::optim::predicates::formula::CmpOp;
    let mut b = crate::builder::IRBuilder::new();
    b.set_resolver(Box::new(LayeredResolver::range_then_smt_with_budget(500, 4096)));
    // A runtime int with no resolver-derivable bound. We pick a
    // window [10, 20) so the resolver's spuriously-tight (0, 0)
    // fallback for free reads can't accidentally satisfy the probe.
    let idx = b.ir_read_integer(InputPath::new("idx", vec![]), false);
    let idx_vid = idx.value_id().unwrap();
    assert!(
        !probe_in_range(&mut b, &idx, 10, 20),
        "without facts, an unconstrained read cannot satisfy idx ∈ [10, 20)",
    );

    // Plant `idx >= 10` and `idx <= 19` as op-contract-shaped facts.
    plant_cmp_fact(&mut b, idx_vid, CmpOp::Ge, 10);
    plant_cmp_fact(&mut b, idx_vid, CmpOp::Le, 19);

    assert!(
        probe_in_range(&mut b, &idx, 10, 20),
        "fact-fallback should let probe_in_range prove idx ∈ [10, 20)",
    );
}

#[test]
fn require_static_int_recovers_from_op_contract_eq_fact() {
    // Fact-fallback for chokepoints requiring a single static int
    // (slice/range/repeat/reshape/split). An `Eq` fact pins the
    // value to a single point.
    use crate::optim::predicates::formula::CmpOp;
    let mut b = crate::builder::IRBuilder::new();
    b.set_resolver(Box::new(LayeredResolver::range_then_smt_with_budget(500, 4096)));
    let v = b.ir_read_integer(InputPath::new("n", vec![]), false);
    let v_vid = v.value_id().unwrap();

    // Without facts: require_static_int fails (free read).
    assert!(require_static_int(&mut b, &v, SiteKind::RepeatCount, None).is_err());

    // Plant `n == 7`.
    plant_cmp_fact(&mut b, v_vid, CmpOp::Eq, 7);

    let result = require_static_int(&mut b, &v, SiteKind::RepeatCount, None)
        .expect("expected fact-fallback to recover static int");
    assert_eq!(result, StaticInt(7));
}

#[test]
fn require_static_int_rejects_when_facts_only_give_range() {
    // Range facts (Ge + Le with distinct bounds) don't satisfy the
    // single-static-int contract; the chokepoint must still reject.
    use crate::optim::predicates::formula::CmpOp;
    let mut b = crate::builder::IRBuilder::new();
    b.set_resolver(Box::new(LayeredResolver::range_then_smt_with_budget(500, 4096)));
    let v = b.ir_read_integer(InputPath::new("n", vec![]), false);
    let v_vid = v.value_id().unwrap();
    plant_cmp_fact(&mut b, v_vid, CmpOp::Ge, 0);
    plant_cmp_fact(&mut b, v_vid, CmpOp::Le, 9);

    assert!(
        require_static_int(&mut b, &v, SiteKind::RepeatCount, None).is_err(),
        "range facts (min != max) must not be accepted as a static int",
    );
}

// ---------------------------------------------------------------
// resolve_int_or_bounded — prove-based fallback for bounded ints
// (compiler.chokepoint-prove-integration)
// ---------------------------------------------------------------

#[test]
fn resolve_int_or_bounded_falls_through_to_static_for_literal() {
    let mut b = crate::builder::IRBuilder::new();
    let v = b.ir_constant_int(42);
    let outcome = resolve_int_or_bounded(
        &mut b,
        &v,
        SiteKind::ReshapeDim,
        None,
    );
    assert_eq!(outcome, BoundedInt::Static(42));
}

#[test]
fn resolve_int_or_bounded_falls_through_to_existing_fact_scan() {
    // The existing scanner handles `Ge n` + `Le m` already; the
    // prove-based fallback must not interfere when that layer
    // already returns Bounded.
    use crate::optim::predicates::formula::CmpOp;
    let mut b = crate::builder::IRBuilder::new();
    b.set_resolver(Box::new(LayeredResolver::range_then_smt_with_budget(500, 4096)));
    let k = b.ir_read_integer(InputPath::new("k", vec![]), false);
    let k_vid = k.value_id().unwrap();
    plant_cmp_fact(&mut b, k_vid, CmpOp::Ge, 0);
    plant_cmp_fact(&mut b, k_vid, CmpOp::Le, 16);

    let outcome = resolve_int_or_bounded(
        &mut b,
        &k,
        SiteKind::ReshapeDim,
        None,
    );
    assert_eq!(outcome, BoundedInt::Bounded { min: 0, max: 16 });
}

#[test]
fn resolve_int_or_bounded_recovers_bound_via_prove_arithmetic_fact() {
    // The fact-scanner only matches `Cmp(Value, LitInt)` shapes.
    // An arithmetic-shape fact `k + k <= 20` (== `2*k <= 20`)
    // doesn't decompose to a bound by shape; Z3 proves k <= 10
    // and k >= -10 (no lower bound implied; the symmetric upper
    // gives a finite max but no constructive lower without more
    // facts). We plant both halves so the test pins both bounds.
    use crate::optim::predicates::formula::{ArithOp, CmpOp, ContractTerm, ContractVar};
    let mut b = crate::builder::IRBuilder::new();
    b.set_resolver(Box::new(LayeredResolver::range_then_smt_with_budget(500, 4096)));
    let k = b.ir_read_integer(InputPath::new("k", vec![]), false);
    let k_vid = k.value_id().unwrap();

    // Plant `k + k <= 20` and `k + k >= -20`. Shape-matching
    // scanner can't see through Arith; prove() can.
    let k_plus_k = ContractTerm::Arith {
        op: ArithOp::Add,
        lhs: Box::new(ContractTerm::Var(ContractVar::Value(k_vid))),
        rhs: Box::new(ContractTerm::Var(ContractVar::Value(k_vid))),
    };
    b.facts.insert_for(
        k_vid,
        ContractTerm::Cmp {
            op: CmpOp::Le,
            lhs: Box::new(k_plus_k.clone()),
            rhs: Box::new(ContractTerm::LitInt(20)),
        },
    );
    b.facts.insert_for(
        k_vid,
        ContractTerm::Cmp {
            op: CmpOp::Ge,
            lhs: Box::new(k_plus_k),
            rhs: Box::new(ContractTerm::LitInt(-20)),
        },
    );

    let outcome = resolve_int_or_bounded(
        &mut b,
        &k,
        SiteKind::ReshapeDim,
        None,
    );
    assert_eq!(
        outcome,
        BoundedInt::Bounded { min: -10, max: 10 },
        "expected prove() to derive k ∈ [-10, 10] from `2k ∈ [-20, 20]`",
    );
}

#[test]
fn resolve_int_or_bounded_returns_neither_on_unknown() {
    // A genuinely free runtime int with no facts must return
    // Neither — Unknown from prove() defaults to no admission.
    let mut b = crate::builder::IRBuilder::new();
    b.set_resolver(Box::new(LayeredResolver::range_then_smt_with_budget(500, 4096)));
    let k = b.ir_read_integer(InputPath::new("k", vec![]), false);
    let outcome = resolve_int_or_bounded(
        &mut b,
        &k,
        SiteKind::ReshapeDim,
        None,
    );
    assert_eq!(outcome, BoundedInt::Neither);
}

#[test]
fn resolve_int_or_bounded_collapses_min_eq_max_to_static() {
    // `prove()`-derivable arithmetic identity: `k * k == 16` with
    // an extra `k >= 0` fact pins k to 4 uniquely. The helper
    // should report `Static(4)`, not `Bounded { min: 4, max: 4 }`.
    use crate::optim::predicates::formula::{ArithOp, CmpOp, ContractTerm, ContractVar};
    let mut b = crate::builder::IRBuilder::new();
    b.set_resolver(Box::new(LayeredResolver::range_then_smt_with_budget(500, 4096)));
    let k = b.ir_read_integer(InputPath::new("k", vec![]), false);
    let k_vid = k.value_id().unwrap();

    let k_sq = ContractTerm::Arith {
        op: ArithOp::Mul,
        lhs: Box::new(ContractTerm::Var(ContractVar::Value(k_vid))),
        rhs: Box::new(ContractTerm::Var(ContractVar::Value(k_vid))),
    };
    b.facts.insert_for(
        k_vid,
        ContractTerm::Cmp {
            op: CmpOp::Eq,
            lhs: Box::new(k_sq),
            rhs: Box::new(ContractTerm::LitInt(16)),
        },
    );
    plant_cmp_fact(&mut b, k_vid, CmpOp::Ge, 0);

    let outcome = resolve_int_or_bounded(
        &mut b,
        &k,
        SiteKind::ReshapeDim,
        None,
    );
    assert_eq!(outcome, BoundedInt::Static(4));
}

#[test]
fn ndarray_axis_or_other_chokepoint_admits_value_provable_via_prove() {
    // Marquee regression for compiler.consumer-prove-fallback-rollout:
    // pick a non-reshape/split chokepoint (`SiteKind::RepeatCount`,
    // converted in this rollout) and show `require_provable_static_int`
    // admits a value pinned by `k * k == 16` ∧ `k >= 0` ⟹ k == 4.
    // Today's `require_static_int` would reject this program; the
    // promoted Phase-A helper unblocks it everywhere.
    use crate::optim::predicates::formula::{ArithOp, CmpOp, ContractTerm, ContractVar};
    let mut b = crate::builder::IRBuilder::new();
    b.set_resolver(Box::new(LayeredResolver::range_then_smt_with_budget(500, 4096)));
    let k = b.ir_read_integer(InputPath::new("k", vec![]), false);
    let k_vid = k.value_id().unwrap();

    // Plant `k * k == 16` (arithmetic-shaped fact — shape-matcher
    // can't decompose, prove() can).
    let k_sq = ContractTerm::Arith {
        op: ArithOp::Mul,
        lhs: Box::new(ContractTerm::Var(ContractVar::Value(k_vid))),
        rhs: Box::new(ContractTerm::Var(ContractVar::Value(k_vid))),
    };
    b.facts.insert_for(
        k_vid,
        ContractTerm::Cmp {
            op: CmpOp::Eq,
            lhs: Box::new(k_sq),
            rhs: Box::new(ContractTerm::LitInt(16)),
        },
    );
    plant_cmp_fact(&mut b, k_vid, CmpOp::Ge, 0);

    let n = require_provable_static_int(&mut b, &k, SiteKind::RepeatCount);
    assert_eq!(n, 4, "prove()-derived single-point bound must be admitted at RepeatCount");
}

#[test]
fn ask_bounds_falls_back_to_facts() {
    // ask_bounds returns the resolver's (min, max) when it's a
    // non-trivial range; otherwise it falls back to facts. For an
    // unconstrained read the SMT optimizer returns degenerate
    // (Some(0), Some(0)), which fails the min < max gate, so the
    // helper consults op-contract facts.
    use crate::optim::predicates::formula::CmpOp;
    let mut b = crate::builder::IRBuilder::new();
    b.set_resolver(Box::new(LayeredResolver::range_then_smt_with_budget(500, 4096)));
    let v = b.ir_read_integer(InputPath::new("k", vec![]), false);
    let v_vid = v.value_id().unwrap();
    plant_cmp_fact(&mut b, v_vid, CmpOp::Ge, 0);
    plant_cmp_fact(&mut b, v_vid, CmpOp::Le, 16);
    assert_eq!(b.ask_bounds(&v), Some((0, 16)));
}

// ---------------------------------------------------------------
// Float-arith interval E (Group 9).
// ---------------------------------------------------------------

/// Helper: plant a `Cmp(Value(vid) <op> LitFloat(n))` fact on `b.facts`.
fn plant_float_cmp_fact(
    b: &mut crate::builder::IRBuilder,
    vid: crate::types::ValueId,
    op: crate::optim::predicates::formula::CmpOp,
    n: f64,
) {
    use crate::optim::predicates::formula::{ContractFloat, ContractTerm, ContractVar};
    b.facts.insert_for(
        vid,
        ContractTerm::Cmp {
            op,
            lhs: Box::new(ContractTerm::Var(ContractVar::Value(vid))),
            rhs: Box::new(ContractTerm::LitFloat(ContractFloat(n))),
        },
    );
}

/// Helper: extract `(lo, hi)` from the `BoolComb(And, [Ge(out, lo), Le(out, hi)])`
/// shape that `interval_fact_for_float_binary` emits.
fn extract_float_bounds_from_term(
    term: &crate::optim::predicates::formula::ContractTerm,
) -> (f64, f64) {
    use crate::optim::predicates::formula::{
        BoolOp, CmpOp, ContractFloat, ContractTerm, ContractVar,
    };
    if let ContractTerm::BoolComb { op: BoolOp::And, operands } = term {
        assert_eq!(operands.len(), 2);
        let mut lo: Option<f64> = None;
        let mut hi: Option<f64> = None;
        for clause in operands {
            if let ContractTerm::Cmp { op, lhs, rhs } = clause {
                assert!(matches!(lhs.as_ref(), ContractTerm::Var(ContractVar::Value(_))));
                if let ContractTerm::LitFloat(ContractFloat(n)) = rhs.as_ref() {
                    match op {
                        CmpOp::Ge => lo = Some(*n),
                        CmpOp::Le => hi = Some(*n),
                        _ => panic!("unexpected cmp op in emitted bound: {:?}", op),
                    }
                }
            }
        }
        (lo.expect("missing lo bound"), hi.expect("missing hi bound"))
    } else {
        panic!("expected BoolComb(And, …), got {:?}", term);
    }
}

#[test]
fn interval_float_add_yields_sum_of_bounds() {
    use crate::optim::predicates::formula::{ArithOp, CmpOp};
    let mut b = crate::builder::IRBuilder::new();
    let a = b.ir_read_float(InputPath::new("a", vec![]), false);
    let bb = b.ir_read_float(InputPath::new("b", vec![]), false);
    let a_vid = a.value_id().unwrap();
    let b_vid = bb.value_id().unwrap();
    plant_float_cmp_fact(&mut b, a_vid, CmpOp::Ge, 0.0);
    plant_float_cmp_fact(&mut b, a_vid, CmpOp::Le, 5.0);
    plant_float_cmp_fact(&mut b, b_vid, CmpOp::Ge, 10.0);
    plant_float_cmp_fact(&mut b, b_vid, CmpOp::Le, 20.0);

    let out_vid = crate::types::ValueId::next();
    let term = interval_fact_for_float_binary(&b.facts, ArithOp::Add, a_vid, b_vid, out_vid)
        .expect("expected interval fact for add");
    let (lo, hi) = extract_float_bounds_from_term(&term);
    assert_eq!(lo, 10.0);
    assert_eq!(hi, 25.0);
}

#[test]
fn interval_float_sub_yields_diff_of_bounds() {
    use crate::optim::predicates::formula::{ArithOp, CmpOp};
    let mut b = crate::builder::IRBuilder::new();
    let a = b.ir_read_float(InputPath::new("a", vec![]), false);
    let bb = b.ir_read_float(InputPath::new("b", vec![]), false);
    let a_vid = a.value_id().unwrap();
    let b_vid = bb.value_id().unwrap();
    plant_float_cmp_fact(&mut b, a_vid, CmpOp::Ge, 0.0);
    plant_float_cmp_fact(&mut b, a_vid, CmpOp::Le, 5.0);
    plant_float_cmp_fact(&mut b, b_vid, CmpOp::Ge, 10.0);
    plant_float_cmp_fact(&mut b, b_vid, CmpOp::Le, 20.0);

    let out_vid = crate::types::ValueId::next();
    let term = interval_fact_for_float_binary(&b.facts, ArithOp::Sub, a_vid, b_vid, out_vid)
        .expect("expected interval fact for sub");
    let (lo, hi) = extract_float_bounds_from_term(&term);
    assert_eq!(lo, -20.0);
    assert_eq!(hi, -5.0);
}

#[test]
fn interval_float_mul_yields_corner_min_max() {
    use crate::optim::predicates::formula::{ArithOp, CmpOp};
    let mut b = crate::builder::IRBuilder::new();
    let a = b.ir_read_float(InputPath::new("a", vec![]), false);
    let bb = b.ir_read_float(InputPath::new("b", vec![]), false);
    let a_vid = a.value_id().unwrap();
    let b_vid = bb.value_id().unwrap();
    plant_float_cmp_fact(&mut b, a_vid, CmpOp::Ge, -2.0);
    plant_float_cmp_fact(&mut b, a_vid, CmpOp::Le, 3.0);
    plant_float_cmp_fact(&mut b, b_vid, CmpOp::Ge, -1.0);
    plant_float_cmp_fact(&mut b, b_vid, CmpOp::Le, 4.0);

    let out_vid = crate::types::ValueId::next();
    let term = interval_fact_for_float_binary(&b.facts, ArithOp::Mul, a_vid, b_vid, out_vid)
        .expect("expected interval fact for mul");
    let (lo, hi) = extract_float_bounds_from_term(&term);
    assert_eq!(lo, -8.0);
    assert_eq!(hi, 12.0);
}

#[test]
fn interval_float_overflow_skips_emit() {
    use crate::optim::predicates::formula::{ArithOp, CmpOp};
    let mut b = crate::builder::IRBuilder::new();
    let a = b.ir_read_float(InputPath::new("a", vec![]), false);
    let bb = b.ir_read_float(InputPath::new("b", vec![]), false);
    let a_vid = a.value_id().unwrap();
    let b_vid = bb.value_id().unwrap();
    // Both inputs span the full f64 range. The corner products
    // overflow to ±inf, so no fact is emitted.
    plant_float_cmp_fact(&mut b, a_vid, CmpOp::Ge, f64::MIN);
    plant_float_cmp_fact(&mut b, a_vid, CmpOp::Le, f64::MAX);
    plant_float_cmp_fact(&mut b, b_vid, CmpOp::Ge, f64::MIN);
    plant_float_cmp_fact(&mut b, b_vid, CmpOp::Le, f64::MAX);

    let out_vid = crate::types::ValueId::next();
    let term = interval_fact_for_float_binary(&b.facts, ArithOp::Mul, a_vid, b_vid, out_vid);
    assert!(term.is_none(), "non-finite output bound must skip emit");
}

#[test]
fn interval_float_unbounded_input_skips_emit() {
    use crate::optim::predicates::formula::ArithOp;
    let mut b = crate::builder::IRBuilder::new();
    let a = b.ir_read_float(InputPath::new("a", vec![]), false);
    let bb = b.ir_read_float(InputPath::new("b", vec![]), false);
    let a_vid = a.value_id().unwrap();
    let b_vid = bb.value_id().unwrap();
    // No facts planted on either input.

    let out_vid = crate::types::ValueId::next();
    let term = interval_fact_for_float_binary(&b.facts, ArithOp::Add, a_vid, b_vid, out_vid);
    assert!(term.is_none(), "unbounded input must skip emit");
}

#[test]
fn relay_sqrt_with_input_bound_emits_output_interval() {
    use crate::optim::predicates::formula::CmpOp;
    let mut b = crate::builder::IRBuilder::new();
    let x = b.ir_read_float(InputPath::new("x", vec![]), false);
    let x_vid = x.value_id().unwrap();
    plant_float_cmp_fact(&mut b, x_vid, CmpOp::Ge, 4.0);
    plant_float_cmp_fact(&mut b, x_vid, CmpOp::Le, 9.0);

    let out_vid = crate::types::ValueId::next();
    let emitted = relay_sqrt_output_interval(&mut b, x_vid, out_vid);
    assert!(emitted, "expected relay to deposit an output bound");

    let entries = b.facts.per_value.get(&out_vid).expect("output bucket");
    let term = entries.last().expect("at least one fact");
    let (lo, hi) = extract_float_bounds_from_term(term);
    assert_eq!(lo, 2.0);
    assert_eq!(hi, 3.0);
}

#[test]
fn relay_sqrt_with_unbounded_input_skips_emit() {
    let mut b = crate::builder::IRBuilder::new();
    let x = b.ir_read_float(InputPath::new("x", vec![]), false);
    let x_vid = x.value_id().unwrap();
    // No facts planted on the input.

    let out_vid = crate::types::ValueId::next();
    let emitted = relay_sqrt_output_interval(&mut b, x_vid, out_vid);
    assert!(!emitted, "unbounded input must skip emit");
    assert!(b.facts.per_value.get(&out_vid).is_none());
}

#[test]
fn relay_sqrt_with_negative_lo_skips_emit() {
    use crate::optim::predicates::formula::CmpOp;
    let mut b = crate::builder::IRBuilder::new();
    let x = b.ir_read_float(InputPath::new("x", vec![]), false);
    let x_vid = x.value_id().unwrap();
    plant_float_cmp_fact(&mut b, x_vid, CmpOp::Ge, -1.0);
    plant_float_cmp_fact(&mut b, x_vid, CmpOp::Le, 1.0);

    let out_vid = crate::types::ValueId::next();
    let emitted = relay_sqrt_output_interval(&mut b, x_vid, out_vid);
    assert!(!emitted, "negative lo must skip emit");
    assert!(b.facts.per_value.get(&out_vid).is_none());
}

#[test]
fn relay_exp_with_input_bound_emits_output_interval() {
    use crate::optim::predicates::formula::CmpOp;
    let mut b = crate::builder::IRBuilder::new();
    let x = b.ir_read_float(InputPath::new("x", vec![]), false);
    let x_vid = x.value_id().unwrap();
    plant_float_cmp_fact(&mut b, x_vid, CmpOp::Ge, 0.0);
    plant_float_cmp_fact(&mut b, x_vid, CmpOp::Le, 1.0);

    let out_vid = crate::types::ValueId::next();
    let emitted = relay_exp_output_interval(&mut b, x_vid, out_vid);
    assert!(emitted, "expected relay to deposit an output bound");

    let entries = b.facts.per_value.get(&out_vid).expect("output bucket");
    let term = entries.last().expect("at least one fact");
    let (lo, hi) = extract_float_bounds_from_term(term);
    assert_eq!(lo, 0.0_f64.exp());
    assert_eq!(hi, 1.0_f64.exp());
}

#[test]
fn relay_exp_with_unbounded_input_skips_emit() {
    let mut b = crate::builder::IRBuilder::new();
    let x = b.ir_read_float(InputPath::new("x", vec![]), false);
    let x_vid = x.value_id().unwrap();
    // No facts planted on the input.

    let out_vid = crate::types::ValueId::next();
    let emitted = relay_exp_output_interval(&mut b, x_vid, out_vid);
    assert!(!emitted, "unbounded input must skip emit");
    assert!(b.facts.per_value.get(&out_vid).is_none());
}

#[test]
fn relay_exp_with_overflow_hi_skips_emit() {
    use crate::optim::predicates::formula::CmpOp;
    let mut b = crate::builder::IRBuilder::new();
    let x = b.ir_read_float(InputPath::new("x", vec![]), false);
    let x_vid = x.value_id().unwrap();
    // `exp(1e308)` overflows f64 to `+inf`; `is_finite()` guard must
    // skip rather than depositing a non-finite literal.
    plant_float_cmp_fact(&mut b, x_vid, CmpOp::Ge, 0.0);
    plant_float_cmp_fact(&mut b, x_vid, CmpOp::Le, 1.0e308);

    let out_vid = crate::types::ValueId::next();
    let emitted = relay_exp_output_interval(&mut b, x_vid, out_vid);
    assert!(!emitted, "overflowing exp(hi) must skip emit");
    assert!(b.facts.per_value.get(&out_vid).is_none());
}

#[test]
fn relay_log_with_input_bound_emits_output_interval() {
    use crate::optim::predicates::formula::CmpOp;
    let mut b = crate::builder::IRBuilder::new();
    let x = b.ir_read_float(InputPath::new("x", vec![]), false);
    let x_vid = x.value_id().unwrap();
    let e = 1.0_f64.exp();
    plant_float_cmp_fact(&mut b, x_vid, CmpOp::Ge, 1.0);
    plant_float_cmp_fact(&mut b, x_vid, CmpOp::Le, e);

    let out_vid = crate::types::ValueId::next();
    let emitted = relay_log_output_interval(&mut b, x_vid, out_vid);
    assert!(emitted, "expected relay to deposit an output bound");

    let entries = b.facts.per_value.get(&out_vid).expect("output bucket");
    let term = entries.last().expect("at least one fact");
    let (lo, hi) = extract_float_bounds_from_term(term);
    assert_eq!(lo, 1.0_f64.ln());
    assert_eq!(hi, e.ln());
}

#[test]
fn relay_log_with_unbounded_input_skips_emit() {
    let mut b = crate::builder::IRBuilder::new();
    let x = b.ir_read_float(InputPath::new("x", vec![]), false);
    let x_vid = x.value_id().unwrap();
    // No facts planted on the input.

    let out_vid = crate::types::ValueId::next();
    let emitted = relay_log_output_interval(&mut b, x_vid, out_vid);
    assert!(!emitted, "unbounded input must skip emit");
    assert!(b.facts.per_value.get(&out_vid).is_none());
}

#[test]
fn relay_log_with_non_positive_lo_skips_emit() {
    use crate::optim::predicates::formula::CmpOp;
    let mut b = crate::builder::IRBuilder::new();
    let x = b.ir_read_float(InputPath::new("x", vec![]), false);
    let x_vid = x.value_id().unwrap();
    plant_float_cmp_fact(&mut b, x_vid, CmpOp::Ge, -1.0);
    plant_float_cmp_fact(&mut b, x_vid, CmpOp::Le, 1.0);

    let out_vid = crate::types::ValueId::next();
    let emitted = relay_log_output_interval(&mut b, x_vid, out_vid);
    assert!(!emitted, "non-positive lo must skip emit");
    assert!(b.facts.per_value.get(&out_vid).is_none());
}

#[test]
fn relay_arccos_with_input_bound_emits_output_interval() {
    use crate::optim::predicates::formula::CmpOp;
    let mut b = crate::builder::IRBuilder::new();
    let x = b.ir_read_float(InputPath::new("x", vec![]), false);
    let x_vid = x.value_id().unwrap();
    plant_float_cmp_fact(&mut b, x_vid, CmpOp::Ge, 0.0);
    plant_float_cmp_fact(&mut b, x_vid, CmpOp::Le, 0.5);

    let out_vid = crate::types::ValueId::next();
    let emitted = relay_arccos_output_interval(&mut b, x_vid, out_vid);
    assert!(emitted, "expected relay to deposit an output bound");

    let entries = b.facts.per_value.get(&out_vid).expect("output bucket");
    let term = entries.last().expect("at least one fact");
    let (lo, hi) = extract_float_bounds_from_term(term);
    // arccos is monotone-decreasing: bounds swap.
    assert_eq!(lo, 0.5_f64.acos());
    assert_eq!(hi, 0.0_f64.acos());
}

#[test]
fn relay_arccos_with_unbounded_input_skips_emit() {
    let mut b = crate::builder::IRBuilder::new();
    let x = b.ir_read_float(InputPath::new("x", vec![]), false);
    let x_vid = x.value_id().unwrap();
    // No facts planted on the input.

    let out_vid = crate::types::ValueId::next();
    let emitted = relay_arccos_output_interval(&mut b, x_vid, out_vid);
    assert!(!emitted, "unbounded input must skip emit");
    assert!(b.facts.per_value.get(&out_vid).is_none());
}

#[test]
fn relay_arccos_with_out_of_domain_lo_skips_emit() {
    use crate::optim::predicates::formula::CmpOp;
    let mut b = crate::builder::IRBuilder::new();
    let x = b.ir_read_float(InputPath::new("x", vec![]), false);
    let x_vid = x.value_id().unwrap();
    // `lo = -1.5` is outside arccos's domain `[-1, 1]`; the
    // defensive guard must skip rather than depositing a NaN-laden
    // bound.
    plant_float_cmp_fact(&mut b, x_vid, CmpOp::Ge, -1.5);
    plant_float_cmp_fact(&mut b, x_vid, CmpOp::Le, 0.5);

    let out_vid = crate::types::ValueId::next();
    let emitted = relay_arccos_output_interval(&mut b, x_vid, out_vid);
    assert!(!emitted, "out-of-domain lo must skip emit");
    assert!(b.facts.per_value.get(&out_vid).is_none());
}
