//! Tests for the `ContractTerm` AST + Z3 lowering (`formula.rs`).

use z3::ast::{Ast, Int};

use crate::optim::predicates::{
    lower_bool, ArithOp, CmpOp, ContractTerm, ContractVar, LowerError, Substitution,
};

// ---------------------------------------------------------------------------
// Lowering: simple comparisons
// ---------------------------------------------------------------------------

#[test]
fn lowers_input_equality_to_z3() {
    // template: `Input("k") == 5`
    let template = ContractTerm::eq(ContractTerm::var_in("k"), ContractTerm::lit(5));

    let k_term = Int::fresh_const("k_");
    let subst = Substitution::new().with_input("k", k_term.clone());

    let out = lower_bool(&template, &subst).expect("must lower");

    // Sanity-check via a solver: the only model of `out` must have k = 5.
    let solver = z3::Solver::new();
    solver.assert(&out.term);
    assert_eq!(solver.check(), z3::SatResult::Sat);
    let model = solver.get_model().unwrap();
    let v = model.eval(&k_term, true).unwrap().as_i64().unwrap();
    assert_eq!(v, 5);
}

#[test]
fn lowers_le_chain_via_and() {
    // template: `0 <= k AND k <= 1024`
    let template = ContractTerm::and(vec![
        ContractTerm::le(ContractTerm::lit(0), ContractTerm::var_in("k")),
        ContractTerm::le(ContractTerm::var_in("k"), ContractTerm::lit(1024)),
    ]);

    let k_term = Int::fresh_const("k_");
    let subst = Substitution::new().with_input("k", k_term.clone());

    let out = lower_bool(&template, &subst).unwrap();
    let solver = z3::Solver::new();
    solver.assert(&out.term);

    // Reject k = -1.
    solver.push();
    solver.assert(&k_term._eq(&Int::from_i64(-1)));
    assert_eq!(solver.check(), z3::SatResult::Unsat);
    solver.pop(1);

    // Reject k = 2000.
    solver.push();
    solver.assert(&k_term._eq(&Int::from_i64(2000)));
    assert_eq!(solver.check(), z3::SatResult::Unsat);
    solver.pop(1);

    // Accept k = 500.
    solver.push();
    solver.assert(&k_term._eq(&Int::from_i64(500)));
    assert_eq!(solver.check(), z3::SatResult::Sat);
    solver.pop(1);
}

#[test]
fn lowers_arithmetic_inside_comparison() {
    // template: `Input("a") + Input("b") == 7`
    let template = ContractTerm::eq(
        ContractTerm::Arith {
            op: ArithOp::Add,
            lhs: Box::new(ContractTerm::var_in("a")),
            rhs: Box::new(ContractTerm::var_in("b")),
        },
        ContractTerm::lit(7),
    );

    let a = Int::fresh_const("a_");
    let b = Int::fresh_const("b_");
    let subst = Substitution::new()
        .with_input("a", a.clone())
        .with_input("b", b.clone());

    let out = lower_bool(&template, &subst).unwrap();

    let solver = z3::Solver::new();
    solver.assert(&out.term);
    solver.assert(&a._eq(&Int::from_i64(3)));
    assert_eq!(solver.check(), z3::SatResult::Sat);
    let model = solver.get_model().unwrap();
    assert_eq!(model.eval(&b, true).unwrap().as_i64().unwrap(), 4);
}

// ---------------------------------------------------------------------------
// Lowering: predicate applications
// ---------------------------------------------------------------------------

#[test]
fn lowers_predicate_application_and_records_meta_facts() {
    // template: `nnz(Input("x"))`
    let template = ContractTerm::pred("nnz", vec![ContractTerm::var_in("x")]);

    let x = Int::fresh_const("x_");
    let subst = Substitution::new().with_input("x", x);

    let out = lower_bool(&template, &subst).expect("must lower");

    // The stub nnz registration injects one meta-fact (`nnz(v) >= 0`).
    assert_eq!(out.meta_fact_sets.len(), 1);
    let (kind, facts) = &out.meta_fact_sets[0];
    assert_eq!(kind, "nnz");
    assert_eq!(facts.len(), 1);
}

#[test]
fn rejects_unknown_predicate_with_specific_error() {
    let template = ContractTerm::pred("not_a_real_predicate", vec![ContractTerm::var_in("x")]);
    let subst = Substitution::new().with_input("x", Int::fresh_const("x_"));
    match lower_bool(&template, &subst) {
        Err(LowerError::UnknownPredicate(k)) => assert_eq!(k, "not_a_real_predicate"),
        other => panic!("expected UnknownPredicate, got {:?}", other),
    }
}

#[test]
fn rejects_predicate_with_arity_mismatch() {
    let template = ContractTerm::pred("nnz", vec![
        ContractTerm::var_in("x"),
        ContractTerm::var_in("y"),
    ]);
    let subst = Substitution::new()
        .with_input("x", Int::fresh_const("x_"))
        .with_input("y", Int::fresh_const("y_"));
    match lower_bool(&template, &subst) {
        Err(LowerError::ArityMismatch { kind, expected, got }) => {
            assert_eq!(kind, "nnz");
            assert_eq!(expected, 1);
            assert_eq!(got, 2);
        }
        other => panic!("expected ArityMismatch, got {:?}", other),
    }
}

// ---------------------------------------------------------------------------
// Lowering: error paths
// ---------------------------------------------------------------------------

#[test]
fn unbound_input_is_a_clear_error() {
    let template = ContractTerm::eq(ContractTerm::var_in("z"), ContractTerm::lit(0));
    let subst = Substitution::new(); // no bindings
    match lower_bool(&template, &subst) {
        Err(LowerError::UnboundInput(name)) => assert_eq!(name, "z"),
        other => panic!("expected UnboundInput, got {:?}", other),
    }
}

#[test]
fn unbound_output_is_a_clear_error() {
    let template = ContractTerm::eq(ContractTerm::var_out(), ContractTerm::lit(0));
    let subst = Substitution::new();
    match lower_bool(&template, &subst) {
        Err(LowerError::UnboundOutput) => (),
        other => panic!("expected UnboundOutput, got {:?}", other),
    }
}

#[test]
fn sort_mismatch_int_at_bool_position() {
    // template: just `Input("k")` — Int, not Bool, at the top.
    let template = ContractTerm::var_in("k");
    let subst = Substitution::new().with_input("k", Int::fresh_const("k_"));
    match lower_bool(&template, &subst) {
        Err(LowerError::SortMismatch(_)) => (),
        other => panic!("expected SortMismatch, got {:?}", other),
    }
}

#[test]
fn sort_mismatch_bool_at_int_position() {
    // template: `LitBool(true) + 1` — Bool inside Add (Int slot).
    let template = ContractTerm::Arith {
        op: ArithOp::Add,
        lhs: Box::new(ContractTerm::LitBool(true)),
        rhs: Box::new(ContractTerm::lit(1)),
    };
    // Wrap in eq so the top is Bool.
    let wrapper = ContractTerm::eq(template, ContractTerm::lit(2));
    let subst = Substitution::new();
    match lower_bool(&wrapper, &subst) {
        Err(LowerError::SortMismatch(_)) => (),
        other => panic!("expected SortMismatch, got {:?}", other),
    }
}

// ---------------------------------------------------------------------------
// Builders
// ---------------------------------------------------------------------------

#[test]
fn builder_methods_produce_expected_variants() {
    let t = ContractTerm::and(vec![
        ContractTerm::le(ContractTerm::var_in("a"), ContractTerm::var_in("b")),
        ContractTerm::eq(ContractTerm::var_out(), ContractTerm::lit(0)),
    ]);
    match t {
        ContractTerm::BoolComb { op, operands } => {
            assert_eq!(op, crate::optim::predicates::BoolOp::And);
            assert_eq!(operands.len(), 2);
            assert!(matches!(operands[0], ContractTerm::Cmp { op: CmpOp::Le, .. }));
            assert!(matches!(operands[1], ContractTerm::Cmp { op: CmpOp::Eq, .. }));
        }
        other => panic!("expected BoolComb, got {:?}", other),
    }
}

#[test]
fn contract_var_eq_and_hash() {
    use std::collections::HashSet;
    let mut s = HashSet::new();
    s.insert(ContractVar::Input("x".to_string()));
    s.insert(ContractVar::Input("x".to_string())); // duplicate
    s.insert(ContractVar::Input("y".to_string()));
    s.insert(ContractVar::Output);
    assert_eq!(s.len(), 3);
}
