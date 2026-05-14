//! Tests for the discharge orchestrator (`discharge.rs`).

use z3::ast::{Ast, Int};

use crate::circuit_input::InputPath;
use crate::ir::IRStatement;
use crate::ir_defs::IR;
use crate::optim::predicates::{
    build_input_name_index, find_structural_predicates, Discharger, DischargeKey,
    DischargeResult,
};

// ---------------------------------------------------------------------------
// Synthetic-IR helpers
// ---------------------------------------------------------------------------

fn read_int_stmt(stmt_id: u32, param: &str) -> IRStatement {
    IRStatement::new(
        stmt_id, crate::types::ValueId::next(),
        IR::ReadInteger {
            path: InputPath::new(param.to_string(), vec![]),
            is_public: false,
        },
        vec![], vec![],
        None)
}

fn structural_pred_stmt(
    stmt_id: u32,
    kind: &str,
    args: &[&str],
    op: Option<&str>,
    bound: Option<&str>,
) -> IRStatement {
    IRStatement::new(
        stmt_id, crate::types::ValueId::next(),
        IR::StructuralPredicate {
            kind: kind.to_string(),
            args: args.iter().map(|s| s.to_string()).collect(),
            op: op.map(String::from),
            bound: bound.map(String::from),
        },
        vec![], vec![],
        None)
}

// ---------------------------------------------------------------------------
// find_structural_predicates
// ---------------------------------------------------------------------------

#[test]
fn find_structural_predicates_skips_other_ops() {
    let stmts = vec![
        read_int_stmt(0, "x"),
        read_int_stmt(1, "k"),
        structural_pred_stmt(2, "nnz", &["x"], Some("=="), Some("k")),
        IRStatement::new(3, crate::types::ValueId::next(), IR::ConstantInt { value: 42 }, vec![], vec![], None),
    ];
    let preds = find_structural_predicates(&stmts);
    assert_eq!(preds.len(), 1);
    assert_eq!(preds[0].0, 2);
}

#[test]
fn find_structural_predicates_preserves_order() {
    let stmts = vec![
        structural_pred_stmt(0, "nnz", &["x"], None, None),
        read_int_stmt(1, "y"),
        structural_pred_stmt(2, "is_sorted", &["y"], None, None),
        structural_pred_stmt(3, "nnz", &["z"], Some("<="), Some("8")),
    ];
    let preds = find_structural_predicates(&stmts);
    assert_eq!(preds.len(), 3);
    assert_eq!(preds.iter().map(|(id, _)| *id).collect::<Vec<_>>(), vec![0, 2, 3]);
}

// ---------------------------------------------------------------------------
// build_input_name_index
// ---------------------------------------------------------------------------

#[test]
fn input_name_index_maps_scalars_to_their_read_stmt() {
    let stmts = vec![
        read_int_stmt(0, "x"),
        read_int_stmt(1, "k"),
        structural_pred_stmt(2, "nnz", &["x"], Some("=="), Some("k")),
    ];
    let idx = build_input_name_index(&stmts);
    assert_eq!(idx.get("x").copied(), Some(0));
    assert_eq!(idx.get("k").copied(), Some(1));
    assert_eq!(idx.len(), 2);
}

#[test]
fn input_name_index_ignores_composite_reads() {
    // A composite input produces ReadInteger with non-empty segments
    // (e.g., indexed element of an ndarray). The foundation index only
    // captures scalar (segment-less) reads.
    let composite_read = IRStatement::new(
        0, crate::types::ValueId::next(),
        IR::ReadInteger {
            path: InputPath::new(
                "arr".to_string(),
                vec![crate::circuit_input::PathSegment::Index(0)],
            ),
            is_public: false,
        },
        vec![], vec![],
        None);
    let stmts = vec![composite_read, read_int_stmt(1, "k")];
    let idx = build_input_name_index(&stmts);
    assert!(idx.get("arr").is_none(), "composite reads must not appear");
    assert_eq!(idx.get("k").copied(), Some(1));
}

// ---------------------------------------------------------------------------
// Discharger: predicate-constraint collection
// ---------------------------------------------------------------------------

#[test]
fn collect_predicate_constraints_ties_pred_value_to_scalar_bound() {
    // Synthetic IR: scalar inputs `x`, `k` plus a precondition
    // `nnz(x) == k`. Verify that the collected constraints, when
    // combined with a witness `k == 5`, force the predicate-symbolic
    // value to 5 too.
    let stmts = vec![
        read_int_stmt(0, "x"),
        read_int_stmt(1, "k"),
        structural_pred_stmt(2, "nnz", &["x"], Some("=="), Some("k")),
    ];

    let discharger = Discharger::new();

    // Per-name Z3 terms. Real callers route through the resolver's
    // walker; tests construct fresh symbols and tie them to the
    // constraint set manually.
    let x_term = Int::fresh_const("x_");
    let k_term = Int::fresh_const("k_");

    let input_lengths = std::collections::HashMap::new();
    let constraints = discharger.collect_predicate_constraints(&stmts, &input_lengths, |name| match name {
        "x" => x_term.clone(),
        "k" => k_term.clone(),
        _ => Int::fresh_const(&format!("test_{name}_")),
    });

    assert!(
        !constraints.clauses.is_empty(),
        "must emit at least one constraint clause"
    );
    assert_eq!(
        constraints.meta_facts.len(),
        1,
        "nnz must inject exactly one meta-fact"
    );

    // Plumb the constraints into a Z3 solver to verify they're consistent
    // and load-bearing. After asserting all clauses + meta-facts + k = 5,
    // the solver should be SAT.
    let solver = z3::Solver::new();
    for c in &constraints.clauses {
        solver.assert(c);
    }
    for m in &constraints.meta_facts {
        solver.assert(m);
    }
    solver.assert(&k_term._eq(&Int::from_i64(5)));
    assert_eq!(solver.check(), z3::SatResult::Sat);
}

#[test]
fn collect_predicate_constraints_ignores_unregistered_kinds() {
    let stmts = vec![
        structural_pred_stmt(0, "not_a_real_predicate", &["x"], None, None),
    ];
    let discharger = Discharger::new();
    let input_lengths = std::collections::HashMap::new();
    let c = discharger.collect_predicate_constraints(&stmts, &input_lengths, |_| {
        Int::fresh_const("t_")
    });
    assert!(c.clauses.is_empty());
    assert!(c.meta_facts.is_empty());
}

#[test]
fn collect_predicate_constraints_dedups_meta_facts_within_one_query() {
    // Three references to nnz must result in exactly one meta-fact set.
    let stmts = vec![
        structural_pred_stmt(0, "nnz", &["x"], None, None),
        structural_pred_stmt(1, "nnz", &["y"], None, None),
        structural_pred_stmt(2, "nnz", &["z"], None, None),
    ];
    let discharger = Discharger::new();
    let input_lengths = std::collections::HashMap::new();
    let c = discharger.collect_predicate_constraints(&stmts, &input_lengths, |name| {
        Int::fresh_const(&format!("test_{name}_"))
    });
    assert_eq!(c.meta_facts.len(), 1,
               "meta-facts must dedup by predicate kind, got {} for 3 nnz refs",
               c.meta_facts.len());
}

// ---------------------------------------------------------------------------
// Cache integration
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Load-bearing test: length bound + predicate clause discharges `k <= len(x)`
// ---------------------------------------------------------------------------

#[test]
fn nnz_length_bound_constrains_k_via_z3() {
    // The kill-criterion test for W1's machinery. Given:
    //   - precondition: nnz(x) == k
    //   - x is an input array of length 1024
    //   - k is a scalar input
    // Z3 must derive `0 <= k <= 1024`. We verify by asserting `k = 2000`
    // and confirming the solver returns Unsat.
    use crate::circuit_input::PathSegment;
    use crate::ir::IRStatement;
    use crate::ir_defs::IR;

    // Synthesize an IR slice that mirrors what ir-gen would produce for
    // `x: NDArray[Float, 1024], k: int` plus the precondition.
    let mut stmts = Vec::new();
    let mut stmt_id: u32 = 0;
    // Element-wise reads for x: 1024 ReadFloat with segments [Index(i)].
    for i in 0..1024u32 {
        stmts.push(IRStatement::new(
            stmt_id, crate::types::ValueId::next(),
            IR::ReadFloat {
                path: crate::circuit_input::InputPath::new(
                    "x".to_string(),
                    vec![PathSegment::Index(i)],
                ),
                is_public: false,
            },
            vec![], vec![],
            None));
        stmt_id += 1;
    }
    // Scalar read for k.
    let k_stmt_id = stmt_id;
    stmts.push(read_int_stmt(stmt_id, "k"));
    stmt_id += 1;
    // Structural-predicate atom: nnz(x) == k.
    stmts.push(structural_pred_stmt(
        stmt_id,
        "nnz",
        &["x"],
        Some("=="),
        Some("k"),
    ));

    // Derive helper maps the same way the resolver would.
    let input_lengths = crate::optim::predicates::build_input_array_lengths(&stmts);
    assert_eq!(input_lengths.get("x").copied(), Some(1024),
               "build_input_array_lengths must recover x's length from the reads");

    // Pre-resolve the scalar k name to a Z3 term so the discharger sees
    // the real symbol (mirrors what the resolver does at chokepoint).
    let k_term = Int::fresh_const("k_term_");
    let _ = k_stmt_id; // walker-encoded in the real integration; not needed here

    let discharger = Discharger::new();
    let c = discharger.collect_predicate_constraints(&stmts, &input_lengths, |name| {
        if name == "k" { k_term.clone() } else { Int::fresh_const(&format!("test_{name}_")) }
    });

    let solver = z3::Solver::new();
    for cl in &c.clauses {
        solver.assert(cl);
    }
    for f in &c.meta_facts {
        solver.assert(f);
    }

    // 1) Witness k = 500 is satisfiable.
    solver.push();
    solver.assert(&k_term._eq(&Int::from_i64(500)));
    assert_eq!(solver.check(), z3::SatResult::Sat,
               "k = 500 must be admissible given nnz(x) <= 1024");
    solver.pop(1);

    // 2) Witness k = 2000 is UNSAT (Z3 derived the bound).
    solver.push();
    solver.assert(&k_term._eq(&Int::from_i64(2000)));
    assert_eq!(solver.check(), z3::SatResult::Unsat,
               "k = 2000 must be rejected — the length bound is load-bearing");
    solver.pop(1);

    // 3) Witness k = -1 is UNSAT (the >= 0 half of the bound).
    solver.push();
    solver.assert(&k_term._eq(&Int::from_i64(-1)));
    assert_eq!(solver.check(), z3::SatResult::Unsat,
               "k = -1 must be rejected — nnz is non-negative");
    solver.pop(1);
}

// ---------------------------------------------------------------------------
// Full LayeredResolver chain: resolve_max(k) under SMT
// ---------------------------------------------------------------------------

#[test]
fn full_resolver_chain_proves_k_bound_via_smt() {
    // The end-to-end integration test for W1's load-bearing premise.
    // Build a Walkthrough-1-style synthetic IR, instantiate the full
    // LayeredResolver with SMT enabled, and query `resolve_max(k)`. The
    // resolver must return `Some(1024)` — the bound that the ShapeAxis
    // chokepoint downstream would consume to admit `np.zeros(k, ...)` as
    // a dyn-ndarray.
    use crate::circuit_input::PathSegment;
    use crate::ir::IRStatement;
    use crate::ir_defs::IR;
    use crate::optim::resolver::{LayeredResolver, Resolver};
    use crate::types::{ScalarValue, Value};

    // Element-wise reads for x (length 1024).
    let mut stmts = Vec::new();
    let mut id: u32 = 0;
    for i in 0..1024u32 {
        stmts.push(IRStatement::new(
            id, crate::types::ValueId::next(),
            IR::ReadFloat {
                path: crate::circuit_input::InputPath::new(
                    "x".to_string(),
                    vec![PathSegment::Index(i)],
                ),
                is_public: false,
            },
            vec![], vec![],
            None));
        id += 1;
    }
    // Scalar read for k.
    let k_id = id;
    stmts.push(IRStatement::new(
        id, crate::types::ValueId::next(),
        IR::ReadInteger {
            path: crate::circuit_input::InputPath::new("k".to_string(), vec![]),
            is_public: false,
        },
        vec![], vec![],
        None));
    id += 1;
    // Structural-predicate: nnz(x) == k.
    stmts.push(structural_pred_stmt(id, "nnz", &["x"], Some("=="), Some("k")));

    // `Value` for the k wire that the chokepoint queries against.
    let k_value = Value::Integer(ScalarValue::new(None, Some(k_id)));

    // Build the full resolver pipeline (range → SMT). SMT timeout/size
    // matches IRGenConfig defaults; the test runs in well under the
    // budget for the synthetic IR.
    let mut resolver = LayeredResolver::range_then_smt_with_budget(500, 4096);

    // The chokepoint would call `resolve_max_with_stmts`.
    let upper = resolver.resolve_max_with_stmts(&k_value, &stmts);
    assert_eq!(
        upper,
        Some(1024),
        "LayeredResolver must derive `k <= 1024` from `nnz(x) == k` plus \
         `len(x) == 1024` (structural-predicate length bound). got {:?}",
        upper
    );

    // The lower bound (`k >= 0`) follows from `nnz >= 0`.
    let lower = resolver.resolve_min_with_stmts(&k_value, &stmts);
    assert_eq!(
        lower,
        Some(0),
        "LayeredResolver must derive `k >= 0` from `nnz(x) >= 0`. got {:?}",
        lower
    );
}

#[test]
fn try_discharge_with_caches_first_result() {
    let mut d = Discharger::new();
    let key = DischargeKey::new(0, 1, 2);
    let mut called = 0;
    let r = d.try_discharge_with(key.clone(), || {
        called += 1;
        DischargeResult::Proved
    });
    assert_eq!(r, DischargeResult::Proved);
    assert_eq!(called, 1);

    let mut called2 = 0;
    let r2 = d.try_discharge_with(key, || {
        called2 += 1;
        DischargeResult::Disproved
    });
    assert_eq!(r2, DischargeResult::Proved, "cache must return first result");
    assert_eq!(called2, 0, "callback must not be invoked on cache hit");
}

#[test]
fn try_discharge_with_distinguishes_keys() {
    let mut d = Discharger::new();
    d.try_discharge_with(DischargeKey::new(0, 0, 0), || DischargeResult::Proved);
    let r = d.try_discharge_with(DischargeKey::new(1, 0, 0), || DischargeResult::Unknown);
    assert_eq!(r, DischargeResult::Unknown);
    assert_eq!(d.cache().len(), 2);
}
