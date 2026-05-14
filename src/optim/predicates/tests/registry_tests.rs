//! Tests for the predicate-kind registry (`registry.rs`).

use crate::ir_defs::IR;
use crate::optim::predicates::{
    registry, smt_encode_structural_predicate,
};
use crate::optim::smt_encoding::{IROp, SmtEncodingCtx, Z3Term};

#[test]
fn registry_contains_nnz_stub() {
    let r = registry();
    let entry = r.get("nnz").expect("`nnz` must be registered");
    assert_eq!(entry.kind, "nnz");
    assert_eq!(entry.arity, 1);
}

#[test]
fn registry_returns_none_for_unknown_kind() {
    assert!(registry().get("not_a_real_predicate").is_none());
}

#[test]
fn smt_encode_known_predicate_returns_bool_term() {
    let mut ctx = SmtEncodingCtx::new();
    let ir = IR::StructuralPredicate {
        kind: "nnz".to_string(),
        args: vec!["x".to_string()],
        op: Some("==".to_string()),
        bound: Some("k".to_string()),
    };
    let term = smt_encode_structural_predicate(&ir, &mut ctx);
    assert!(matches!(term, Z3Term::Bool(_)));
}

#[test]
fn smt_encode_unregistered_predicate_falls_back_gracefully() {
    let mut ctx = SmtEncodingCtx::new();
    let ir = IR::StructuralPredicate {
        kind: "not_a_real_predicate".to_string(),
        args: vec!["x".to_string()],
        op: None,
        bound: None,
    };
    let term = smt_encode_structural_predicate(&ir, &mut ctx);
    // Unregistered kind → fresh_unconstrained Int.
    assert!(matches!(term, Z3Term::Int(_)));
}

#[test]
fn smt_encode_via_ir_op_trait_routes_to_predicate_encoder() {
    let mut ctx = SmtEncodingCtx::new();
    let ir = IR::StructuralPredicate {
        kind: "nnz".to_string(),
        args: vec!["x".to_string()],
        op: None,
        bound: None,
    };
    let term = <IR as IROp>::smt_encode(&ir, &mut ctx, &[]);
    assert!(matches!(term, Z3Term::Bool(_)));
}

#[test]
fn meta_facts_injected_on_first_reference() {
    let mut ctx = SmtEncodingCtx::new();
    assert!(!ctx.has_injected("nnz"));
    let ir = IR::StructuralPredicate {
        kind: "nnz".to_string(),
        args: vec!["x".to_string()],
        op: None,
        bound: None,
    };
    let _ = smt_encode_structural_predicate(&ir, &mut ctx);
    assert!(ctx.has_injected("nnz"));
    assert_eq!(ctx.meta_facts.len(), 1);
}

#[test]
fn meta_facts_deduplicated_across_references() {
    let mut ctx = SmtEncodingCtx::new();
    let ir = IR::StructuralPredicate {
        kind: "nnz".to_string(),
        args: vec!["x".to_string()],
        op: None,
        bound: None,
    };
    let _ = smt_encode_structural_predicate(&ir, &mut ctx);
    let _ = smt_encode_structural_predicate(&ir, &mut ctx);
    let _ = smt_encode_structural_predicate(&ir, &mut ctx);
    assert_eq!(ctx.meta_facts.len(), 1);
}

#[test]
fn arity_mismatch_falls_back_gracefully() {
    // `nnz` has arity 1; supplying 2 args must NOT panic.
    let mut ctx = SmtEncodingCtx::new();
    let ir = IR::StructuralPredicate {
        kind: "nnz".to_string(),
        args: vec!["x".to_string(), "y".to_string()],
        op: None,
        bound: None,
    };
    let term = smt_encode_structural_predicate(&ir, &mut ctx);
    assert!(matches!(term, Z3Term::Int(_)));
}
