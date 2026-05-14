//! Tests for the per-IR-kind contract registry (`contracts.rs`).

use crate::ir_defs::IR;
use crate::optim::predicates::{op_contract_for, FrameCondition, OpContract};

#[test]
fn default_op_contract_is_empty_for_assert() {
    let c = op_contract_for(&IR::Assert);
    assert!(c.requires.is_empty());
    assert!(c.ensures.is_empty());
    assert_eq!(c.frame, FrameCondition::Pure);
}

#[test]
fn default_op_contract_is_empty_for_structural_predicate() {
    let c = op_contract_for(&IR::StructuralPredicate {
        kind: "nnz".to_string(),
        args: vec!["x".to_string()],
        op: None,
        bound: None,
    });
    assert!(c.requires.is_empty());
    assert!(c.ensures.is_empty());
}

#[test]
fn default_op_contract_constructor() {
    let c = OpContract::default_contract();
    assert!(c.requires.is_empty());
    assert!(c.ensures.is_empty());
    assert_eq!(c.frame, FrameCondition::Pure);
    assert!(c.is_default());
}

#[test]
fn pure_constructor_with_explicit_clauses_marks_non_default() {
    use crate::optim::predicates::{ContractFormula, ContractTerm};
    let c = OpContract::pure(
        vec![ContractFormula::new(ContractTerm::LitBool(true))],
        vec![],
    );
    assert!(!c.is_default());
    assert_eq!(c.frame, FrameCondition::Pure);
}

#[test]
fn op_contract_by_name_returns_registered_contract_for_dyn_fill() {
    use crate::optim::predicates::{op_contract_by_name, ContractTerm};
    let c = op_contract_by_name("dyn_fill_with_active");
    assert!(!c.is_default());
    // Two ensures: `Output >= 0` and `Output == Formal("active")`.
    assert_eq!(c.ensures.len(), 2);
    // Both templates' top level is a Bool Cmp.
    assert!(matches!(c.ensures[0].term, ContractTerm::Cmp { .. }));
    assert!(matches!(c.ensures[1].term, ContractTerm::Cmp { .. }));
}

#[test]
fn op_contract_registry_has_dyn_filter_entry() {
    use crate::optim::predicates::{op_contract_by_name, ContractTerm};
    let c = op_contract_by_name("dyn_filter");
    assert!(!c.is_default(), "dyn_filter should have a registered contract");
    assert_eq!(c.ensures.len(), 1);
    assert!(matches!(c.ensures[0].term, ContractTerm::Cmp { .. }));
}

#[test]
fn op_contract_registry_has_dyn_concatenate_entry() {
    use crate::optim::predicates::{op_contract_by_name, ContractTerm};
    let c = op_contract_by_name("dyn_concatenate");
    assert!(!c.is_default());
    assert_eq!(c.ensures.len(), 1);
    assert!(matches!(c.ensures[0].term, ContractTerm::Cmp { .. }));
}

#[test]
fn op_contract_by_name_returns_default_for_unknown_op() {
    use crate::optim::predicates::op_contract_by_name;
    let c = op_contract_by_name("totally_not_a_real_op");
    assert!(c.is_default());
}

#[test]
fn op_contract_registry_has_dyn_argextremum_entry() {
    // compiler.op-fact-group-3a-reductions-static-ensures: argmax /
    // argmin on dyn arrays deposits both `Output >= 0` (pure template)
    // and `Output < Formal("len_arr")` (multi-formal). Callers must bind
    // `len_arr` at the fire site.
    use crate::optim::predicates::{op_contract_by_name, ContractTerm};
    let c = op_contract_by_name("dyn_argextremum");
    assert!(
        !c.is_default(),
        "dyn_argextremum should have a registered contract",
    );
    assert_eq!(c.ensures.len(), 2);
    assert!(matches!(c.ensures[0].term, ContractTerm::Cmp { .. }));
    assert!(matches!(c.ensures[1].term, ContractTerm::Cmp { .. }));
}

#[test]
fn registered_dyn_fill_template_instantiates_to_value_anchored_fact() {
    use crate::optim::predicates::{
        instantiate_contract, op_contract_by_name,
    };
    use crate::optim::predicates::formula::{CmpOp, ContractTerm, ContractVar};
    use crate::types::ValueId;
    use std::collections::HashMap;

    let c = op_contract_by_name("dyn_fill_with_active");
    let formals: HashMap<String, ValueId> = HashMap::new();
    let target = ValueId(42);
    let inst = instantiate_contract(&c.ensures[0].term, Some(target), &formals);

    let expected = ContractTerm::Cmp {
        op: CmpOp::Ge,
        lhs: Box::new(ContractTerm::Var(ContractVar::Value(target))),
        rhs: Box::new(ContractTerm::LitInt(0)),
    };
    assert_eq!(inst, expected);
}

#[test]
fn fire_contract_deposits_registry_facts_on_output_value() {
    use crate::circuit_input::InputPath;
    use crate::optim::predicates::formula::{CmpOp, ContractTerm, ContractVar};
    use std::collections::HashMap;

    let mut b = crate::builder::IRBuilder::new();
    let v = b.ir_read_integer(InputPath::new("k", vec![]), false);
    let v_vid = v.value_id().unwrap();

    assert!(b.facts.per_value.get(&v_vid).map_or(true, |s: &Vec<_>| s.is_empty()));

    b.fire_contract("dyn_fill_with_active", v_vid, &HashMap::new());

    let facts = b
        .facts
        .per_value
        .get(&v_vid)
        .expect("fire_contract should have deposited at least one fact");
    let expected = ContractTerm::Cmp {
        op: CmpOp::Ge,
        lhs: Box::new(ContractTerm::Var(ContractVar::Value(v_vid))),
        rhs: Box::new(ContractTerm::LitInt(0)),
    };
    assert!(
        facts.iter().any(|f| *f == expected),
        "expected value_id-anchored Output >= 0 fact after fire_contract; got {:?}",
        facts,
    );
}

#[test]
fn op_contract_registry_has_abs_i_entry() {
    use crate::optim::predicates::{op_contract_by_name, ContractTerm};
    let c = op_contract_by_name("abs_i");
    assert!(!c.is_default(), "abs_i should have a registered contract");
    assert_eq!(c.ensures.len(), 1);
    assert!(matches!(c.ensures[0].term, ContractTerm::Cmp { .. }));
}

#[test]
fn op_contract_registry_has_abs_f_entry() {
    use crate::optim::predicates::{op_contract_by_name, ContractTerm};
    let c = op_contract_by_name("abs_f");
    assert!(!c.is_default(), "abs_f should have a registered contract");
    assert_eq!(c.ensures.len(), 1);
    // Float ensures still surface as a top-level Cmp; the rhs is a LitFloat.
    match &c.ensures[0].term {
        ContractTerm::Cmp { rhs, .. } => {
            assert!(matches!(**rhs, ContractTerm::LitFloat(_)));
        }
        other => panic!("expected Cmp template, got {:?}", other),
    }
}

#[test]
fn abs_i_op_fires_nonneg_fact_on_output_value() {
    use crate::circuit_input::InputPath;
    use crate::optim::predicates::formula::{CmpOp, ContractTerm, ContractVar};

    let mut b = crate::builder::IRBuilder::new();
    let x = b.ir_read_integer(InputPath::new("x", vec![]), false);
    let out = b.ir_abs_i(&x);
    let out_vid = out.value_id().expect("ir_abs_i output must have a ValueId");

    let facts = b
        .facts
        .per_value
        .get(&out_vid)
        .expect("ir_abs_i should have deposited at least one fact");
    let expected = ContractTerm::Cmp {
        op: CmpOp::Ge,
        lhs: Box::new(ContractTerm::Var(ContractVar::Value(out_vid))),
        rhs: Box::new(ContractTerm::LitInt(0)),
    };
    assert!(
        facts.iter().any(|f| *f == expected),
        "expected value_id-anchored Output >= 0 fact after ir_abs_i; got {:?}",
        facts,
    );
}

#[test]
fn abs_f_op_fires_nonneg_fact_on_output_value() {
    use crate::circuit_input::InputPath;
    use crate::optim::predicates::formula::{CmpOp, ContractFloat, ContractTerm, ContractVar};

    let mut b = crate::builder::IRBuilder::new();
    let x = b.ir_read_float(InputPath::new("x", vec![]), false);
    let out = b.ir_abs_f(&x);
    let out_vid = out.value_id().expect("ir_abs_f output must have a ValueId");

    let facts = b
        .facts
        .per_value
        .get(&out_vid)
        .expect("ir_abs_f should have deposited at least one fact");
    let expected = ContractTerm::Cmp {
        op: CmpOp::Ge,
        lhs: Box::new(ContractTerm::Var(ContractVar::Value(out_vid))),
        rhs: Box::new(ContractTerm::LitFloat(ContractFloat(0.0))),
    };
    assert!(
        facts.iter().any(|f| *f == expected),
        "expected value_id-anchored Output >= 0.0 fact after ir_abs_f; got {:?}",
        facts,
    );
}

#[test]
fn sqrt_f_op_fires_nonneg_fact_on_output_value() {
    use crate::circuit_input::InputPath;
    use crate::optim::predicates::formula::{CmpOp, ContractFloat, ContractTerm, ContractVar};

    let mut b = crate::builder::IRBuilder::new();
    let x = b.ir_read_float(InputPath::new("x", vec![]), false);
    let out = b.ir_sqrt_f(&x);
    let out_vid = out.value_id().expect("ir_sqrt_f output must have a ValueId");

    let facts = b
        .facts
        .per_value
        .get(&out_vid)
        .expect("ir_sqrt_f should have deposited at least one fact");
    let expected = ContractTerm::Cmp {
        op: CmpOp::Ge,
        lhs: Box::new(ContractTerm::Var(ContractVar::Value(out_vid))),
        rhs: Box::new(ContractTerm::LitFloat(ContractFloat(0.0))),
    };
    assert!(
        facts.iter().any(|f| *f == expected),
        "expected value_id-anchored Output >= 0.0 fact after ir_sqrt_f; got {:?}",
        facts,
    );
}

#[test]
fn exp_f_op_fires_nonneg_fact_on_output_value() {
    use crate::circuit_input::InputPath;
    use crate::optim::predicates::formula::{CmpOp, ContractFloat, ContractTerm, ContractVar};

    let mut b = crate::builder::IRBuilder::new();
    let x = b.ir_read_float(InputPath::new("x", vec![]), false);
    let out = b.ir_exp_f(&x);
    let out_vid = out.value_id().expect("ir_exp_f output must have a ValueId");

    let facts = b
        .facts
        .per_value
        .get(&out_vid)
        .expect("ir_exp_f should have deposited at least one fact");
    let expected = ContractTerm::Cmp {
        op: CmpOp::Ge,
        lhs: Box::new(ContractTerm::Var(ContractVar::Value(out_vid))),
        rhs: Box::new(ContractTerm::LitFloat(ContractFloat(0.0))),
    };
    assert!(
        facts.iter().any(|f| *f == expected),
        "expected value_id-anchored Output >= 0.0 fact after ir_exp_f; got {:?}",
        facts,
    );
}

#[test]
fn sign_i_op_fires_range_fact_on_output_value() {
    use crate::circuit_input::InputPath;
    use crate::optim::predicates::formula::{BoolOp, CmpOp, ContractTerm, ContractVar};

    let mut b = crate::builder::IRBuilder::new();
    let x = b.ir_read_integer(InputPath::new("x", vec![]), false);
    let out = b.ir_sign_i(&x);
    let out_vid = out.value_id().expect("ir_sign_i output must have a ValueId");

    let facts = b
        .facts
        .per_value
        .get(&out_vid)
        .expect("ir_sign_i should have deposited at least one fact");
    let expected = ContractTerm::BoolComb {
        op: BoolOp::And,
        operands: vec![
            ContractTerm::Cmp {
                op: CmpOp::Ge,
                lhs: Box::new(ContractTerm::Var(ContractVar::Value(out_vid))),
                rhs: Box::new(ContractTerm::LitInt(-1)),
            },
            ContractTerm::Cmp {
                op: CmpOp::Le,
                lhs: Box::new(ContractTerm::Var(ContractVar::Value(out_vid))),
                rhs: Box::new(ContractTerm::LitInt(1)),
            },
        ],
    };
    assert!(
        facts.iter().any(|f| *f == expected),
        "expected value_id-anchored -1 <= Output <= 1 fact after ir_sign_i; got {:?}",
        facts,
    );
}

#[test]
fn sign_f_op_fires_range_fact_on_output_value() {
    use crate::circuit_input::InputPath;
    use crate::optim::predicates::formula::{
        BoolOp, CmpOp, ContractFloat, ContractTerm, ContractVar,
    };

    let mut b = crate::builder::IRBuilder::new();
    let x = b.ir_read_float(InputPath::new("x", vec![]), false);
    let out = b.ir_sign_f(&x);
    let out_vid = out.value_id().expect("ir_sign_f output must have a ValueId");

    let facts = b
        .facts
        .per_value
        .get(&out_vid)
        .expect("ir_sign_f should have deposited at least one fact");
    let expected = ContractTerm::BoolComb {
        op: BoolOp::And,
        operands: vec![
            ContractTerm::Cmp {
                op: CmpOp::Ge,
                lhs: Box::new(ContractTerm::Var(ContractVar::Value(out_vid))),
                rhs: Box::new(ContractTerm::LitFloat(ContractFloat(-1.0))),
            },
            ContractTerm::Cmp {
                op: CmpOp::Le,
                lhs: Box::new(ContractTerm::Var(ContractVar::Value(out_vid))),
                rhs: Box::new(ContractTerm::LitFloat(ContractFloat(1.0))),
            },
        ],
    };
    assert!(
        facts.iter().any(|f| *f == expected),
        "expected value_id-anchored -1.0 <= Output <= 1.0 fact after ir_sign_f; got {:?}",
        facts,
    );
}

#[test]
fn sin_f_op_fires_range_fact_on_output_value() {
    use crate::circuit_input::InputPath;
    use crate::optim::predicates::formula::{
        BoolOp, CmpOp, ContractFloat, ContractTerm, ContractVar,
    };

    let mut b = crate::builder::IRBuilder::new();
    let x = b.ir_read_float(InputPath::new("x", vec![]), false);
    let out = b.ir_sin_f(&x);
    let out_vid = out.value_id().expect("ir_sin_f output must have a ValueId");

    let facts = b
        .facts
        .per_value
        .get(&out_vid)
        .expect("ir_sin_f should have deposited at least one fact");
    let expected = ContractTerm::BoolComb {
        op: BoolOp::And,
        operands: vec![
            ContractTerm::Cmp {
                op: CmpOp::Ge,
                lhs: Box::new(ContractTerm::Var(ContractVar::Value(out_vid))),
                rhs: Box::new(ContractTerm::LitFloat(ContractFloat(-1.0))),
            },
            ContractTerm::Cmp {
                op: CmpOp::Le,
                lhs: Box::new(ContractTerm::Var(ContractVar::Value(out_vid))),
                rhs: Box::new(ContractTerm::LitFloat(ContractFloat(1.0))),
            },
        ],
    };
    assert!(
        facts.iter().any(|f| *f == expected),
        "expected value_id-anchored -1.0 <= Output <= 1.0 fact after ir_sin_f; got {:?}",
        facts,
    );
}

#[test]
fn cos_f_op_fires_range_fact_on_output_value() {
    use crate::circuit_input::InputPath;
    use crate::optim::predicates::formula::{
        BoolOp, CmpOp, ContractFloat, ContractTerm, ContractVar,
    };

    let mut b = crate::builder::IRBuilder::new();
    let x = b.ir_read_float(InputPath::new("x", vec![]), false);
    let out = b.ir_cos_f(&x);
    let out_vid = out.value_id().expect("ir_cos_f output must have a ValueId");

    let facts = b
        .facts
        .per_value
        .get(&out_vid)
        .expect("ir_cos_f should have deposited at least one fact");
    let expected = ContractTerm::BoolComb {
        op: BoolOp::And,
        operands: vec![
            ContractTerm::Cmp {
                op: CmpOp::Ge,
                lhs: Box::new(ContractTerm::Var(ContractVar::Value(out_vid))),
                rhs: Box::new(ContractTerm::LitFloat(ContractFloat(-1.0))),
            },
            ContractTerm::Cmp {
                op: CmpOp::Le,
                lhs: Box::new(ContractTerm::Var(ContractVar::Value(out_vid))),
                rhs: Box::new(ContractTerm::LitFloat(ContractFloat(1.0))),
            },
        ],
    };
    assert!(
        facts.iter().any(|f| *f == expected),
        "expected value_id-anchored -1.0 <= Output <= 1.0 fact after ir_cos_f; got {:?}",
        facts,
    );
}

#[test]
fn tanh_f_op_fires_range_fact_on_output_value() {
    use crate::circuit_input::InputPath;
    use crate::optim::predicates::formula::{
        BoolOp, CmpOp, ContractFloat, ContractTerm, ContractVar,
    };

    let mut b = crate::builder::IRBuilder::new();
    let x = b.ir_read_float(InputPath::new("x", vec![]), false);
    let out = b.ir_tanh_f(&x);
    let out_vid = out.value_id().expect("ir_tanh_f output must have a ValueId");

    let facts = b
        .facts
        .per_value
        .get(&out_vid)
        .expect("ir_tanh_f should have deposited at least one fact");
    let expected = ContractTerm::BoolComb {
        op: BoolOp::And,
        operands: vec![
            ContractTerm::Cmp {
                op: CmpOp::Ge,
                lhs: Box::new(ContractTerm::Var(ContractVar::Value(out_vid))),
                rhs: Box::new(ContractTerm::LitFloat(ContractFloat(-1.0))),
            },
            ContractTerm::Cmp {
                op: CmpOp::Le,
                lhs: Box::new(ContractTerm::Var(ContractVar::Value(out_vid))),
                rhs: Box::new(ContractTerm::LitFloat(ContractFloat(1.0))),
            },
        ],
    };
    assert!(
        facts.iter().any(|f| *f == expected),
        "expected value_id-anchored -1.0 <= Output <= 1.0 fact after ir_tanh_f; got {:?}",
        facts,
    );
}

#[test]
fn cosh_f_op_fires_lower_bound_fact_on_output_value() {
    use crate::circuit_input::InputPath;
    use crate::optim::predicates::formula::{CmpOp, ContractFloat, ContractTerm, ContractVar};

    let mut b = crate::builder::IRBuilder::new();
    let x = b.ir_read_float(InputPath::new("x", vec![]), false);
    let out = b.ir_cosh_f(&x);
    let out_vid = out.value_id().expect("ir_cosh_f output must have a ValueId");

    let facts = b
        .facts
        .per_value
        .get(&out_vid)
        .expect("ir_cosh_f should have deposited at least one fact");
    let expected = ContractTerm::Cmp {
        op: CmpOp::Ge,
        lhs: Box::new(ContractTerm::Var(ContractVar::Value(out_vid))),
        rhs: Box::new(ContractTerm::LitFloat(ContractFloat(1.0))),
    };
    assert!(
        facts.iter().any(|f| *f == expected),
        "expected value_id-anchored Output >= 1.0 fact after ir_cosh_f; got {:?}",
        facts,
    );
}

#[test]
fn fire_contract_with_unknown_name_is_noop() {
    use crate::circuit_input::InputPath;
    use std::collections::HashMap;
    let mut b = crate::builder::IRBuilder::new();
    let v = b.ir_read_integer(InputPath::new("k", vec![]), false);
    let v_vid = v.value_id().unwrap();
    b.fire_contract("totally_not_a_real_op", v_vid, &HashMap::new());
    assert!(b.facts.per_value.get(&v_vid).map_or(true, |s: &Vec<_>| s.is_empty()));
}

#[test]
fn ir_equal_i_fires_bool_bounds_fact_on_output_value() {
    use crate::circuit_input::InputPath;
    use crate::optim::predicates::formula::{BoolOp, CmpOp, ContractTerm, ContractVar};

    let mut b = crate::builder::IRBuilder::new();
    let x = b.ir_read_integer(InputPath::new("x", vec![]), false);
    let y = b.ir_read_integer(InputPath::new("y", vec![]), false);
    let out = b.ir_equal_i(&x, &y);
    let out_vid = out.value_id().expect("ir_equal_i output must have a ValueId");

    let facts = b
        .facts
        .per_value
        .get(&out_vid)
        .expect("ir_equal_i should have deposited at least one fact");
    let expected = ContractTerm::BoolComb {
        op: BoolOp::And,
        operands: vec![
            ContractTerm::Cmp {
                op: CmpOp::Ge,
                lhs: Box::new(ContractTerm::Var(ContractVar::Value(out_vid))),
                rhs: Box::new(ContractTerm::LitInt(0)),
            },
            ContractTerm::Cmp {
                op: CmpOp::Le,
                lhs: Box::new(ContractTerm::Var(ContractVar::Value(out_vid))),
                rhs: Box::new(ContractTerm::LitInt(1)),
            },
        ],
    };
    assert!(
        facts.iter().any(|f| *f == expected),
        "expected value_id-anchored 0 <= Output <= 1 fact after ir_equal_i; got {:?}",
        facts,
    );
}

#[test]
fn ir_less_than_f_fires_bool_bounds_fact_on_output_value() {
    use crate::circuit_input::InputPath;
    use crate::optim::predicates::formula::{BoolOp, CmpOp, ContractTerm, ContractVar};

    let mut b = crate::builder::IRBuilder::new();
    let x = b.ir_read_float(InputPath::new("x", vec![]), false);
    let y = b.ir_read_float(InputPath::new("y", vec![]), false);
    let out = b.ir_less_than_f(&x, &y);
    let out_vid = out
        .value_id()
        .expect("ir_less_than_f output must have a ValueId");

    let facts = b
        .facts
        .per_value
        .get(&out_vid)
        .expect("ir_less_than_f should have deposited at least one fact");
    let expected = ContractTerm::BoolComb {
        op: BoolOp::And,
        operands: vec![
            ContractTerm::Cmp {
                op: CmpOp::Ge,
                lhs: Box::new(ContractTerm::Var(ContractVar::Value(out_vid))),
                rhs: Box::new(ContractTerm::LitInt(0)),
            },
            ContractTerm::Cmp {
                op: CmpOp::Le,
                lhs: Box::new(ContractTerm::Var(ContractVar::Value(out_vid))),
                rhs: Box::new(ContractTerm::LitInt(1)),
            },
        ],
    };
    assert!(
        facts.iter().any(|f| *f == expected),
        "expected value_id-anchored 0 <= Output <= 1 fact after ir_less_than_f; got {:?}",
        facts,
    );
}

#[test]
fn ir_logical_and_fires_bool_bounds_fact_on_output_value() {
    use crate::circuit_input::InputPath;
    use crate::optim::predicates::formula::{BoolOp, CmpOp, ContractTerm, ContractVar};

    let mut b = crate::builder::IRBuilder::new();
    let x = b.ir_read_integer(InputPath::new("x", vec![]), false);
    let y = b.ir_read_integer(InputPath::new("y", vec![]), false);
    // Synthesize Bool inputs via `ir_equal_i` so the logical-and is
    // well-typed; both operands carry the bool-bounds fact too.
    let bx = b.ir_equal_i(&x, &y);
    let by = b.ir_equal_i(&y, &x);
    let out = b.ir_logical_and(&bx, &by);
    let out_vid = out
        .value_id()
        .expect("ir_logical_and output must have a ValueId");

    let facts = b
        .facts
        .per_value
        .get(&out_vid)
        .expect("ir_logical_and should have deposited at least one fact");
    let expected = ContractTerm::BoolComb {
        op: BoolOp::And,
        operands: vec![
            ContractTerm::Cmp {
                op: CmpOp::Ge,
                lhs: Box::new(ContractTerm::Var(ContractVar::Value(out_vid))),
                rhs: Box::new(ContractTerm::LitInt(0)),
            },
            ContractTerm::Cmp {
                op: CmpOp::Le,
                lhs: Box::new(ContractTerm::Var(ContractVar::Value(out_vid))),
                rhs: Box::new(ContractTerm::LitInt(1)),
            },
        ],
    };
    assert!(
        facts.iter().any(|f| *f == expected),
        "expected value_id-anchored 0 <= Output <= 1 fact after ir_logical_and; got {:?}",
        facts,
    );
}

#[test]
fn ir_bool_cast_fires_bool_bounds_fact_on_output_value() {
    use crate::circuit_input::InputPath;
    use crate::optim::predicates::formula::{BoolOp, CmpOp, ContractTerm, ContractVar};

    let mut b = crate::builder::IRBuilder::new();
    let x = b.ir_read_integer(InputPath::new("x", vec![]), false);
    let out = b.ir_bool_cast(&x);
    let out_vid = out
        .value_id()
        .expect("ir_bool_cast output must have a ValueId");

    let facts = b
        .facts
        .per_value
        .get(&out_vid)
        .expect("ir_bool_cast should have deposited at least one fact");
    let expected = ContractTerm::BoolComb {
        op: BoolOp::And,
        operands: vec![
            ContractTerm::Cmp {
                op: CmpOp::Ge,
                lhs: Box::new(ContractTerm::Var(ContractVar::Value(out_vid))),
                rhs: Box::new(ContractTerm::LitInt(0)),
            },
            ContractTerm::Cmp {
                op: CmpOp::Le,
                lhs: Box::new(ContractTerm::Var(ContractVar::Value(out_vid))),
                rhs: Box::new(ContractTerm::LitInt(1)),
            },
        ],
    };
    assert!(
        facts.iter().any(|f| *f == expected),
        "expected value_id-anchored 0 <= Output <= 1 fact after ir_bool_cast; got {:?}",
        facts,
    );
}

#[test]
fn ir_bit_and_i_fires_bool_bounds_fact_when_inputs_are_bool() {
    use crate::optim::predicates::formula::{BoolOp, CmpOp, ContractTerm, ContractVar};

    let mut b = crate::builder::IRBuilder::new();
    let x = b.ir_constant_bool(true);
    let y = b.ir_constant_bool(false);
    let out = b.ir_bit_and_i(&x, &y);
    let out_vid = out
        .value_id()
        .expect("ir_bit_and_i output must have a ValueId");

    let facts = b
        .facts
        .per_value
        .get(&out_vid)
        .expect("ir_bit_and_i on Boolean inputs should have deposited at least one fact");
    let expected = ContractTerm::BoolComb {
        op: BoolOp::And,
        operands: vec![
            ContractTerm::Cmp {
                op: CmpOp::Ge,
                lhs: Box::new(ContractTerm::Var(ContractVar::Value(out_vid))),
                rhs: Box::new(ContractTerm::LitInt(0)),
            },
            ContractTerm::Cmp {
                op: CmpOp::Le,
                lhs: Box::new(ContractTerm::Var(ContractVar::Value(out_vid))),
                rhs: Box::new(ContractTerm::LitInt(1)),
            },
        ],
    };
    assert!(
        facts.iter().any(|f| *f == expected),
        "expected value_id-anchored 0 <= Output <= 1 fact after ir_bit_and_i on Boolean inputs; got {:?}",
        facts,
    );
}

#[test]
fn ir_bit_and_i_does_not_fire_when_inputs_are_int() {
    use crate::circuit_input::InputPath;

    let mut b = crate::builder::IRBuilder::new();
    let x = b.ir_read_integer(InputPath::new("x", vec![]), false);
    let y = b.ir_read_integer(InputPath::new("y", vec![]), false);
    let out = b.ir_bit_and_i(&x, &y);
    let out_vid = out
        .value_id()
        .expect("ir_bit_and_i output must have a ValueId");

    let bucket = b.facts.per_value.get(&out_vid);
    assert!(
        bucket.map_or(true, |s| s.is_empty()),
        "expected no facts on ir_bit_and_i output when inputs are Integer; got {:?}",
        bucket,
    );
}

#[test]
fn ir_int_cast_fires_bool_bounds_fact_when_input_is_bool() {
    use crate::optim::predicates::formula::{BoolOp, CmpOp, ContractTerm, ContractVar};

    let mut b = crate::builder::IRBuilder::new();
    let x = b.ir_constant_bool(true);
    let out = b.ir_int_cast(&x);
    let out_vid = out
        .value_id()
        .expect("ir_int_cast output must have a ValueId");

    let facts = b
        .facts
        .per_value
        .get(&out_vid)
        .expect("ir_int_cast on Boolean input should have deposited at least one fact");
    let expected = ContractTerm::BoolComb {
        op: BoolOp::And,
        operands: vec![
            ContractTerm::Cmp {
                op: CmpOp::Ge,
                lhs: Box::new(ContractTerm::Var(ContractVar::Value(out_vid))),
                rhs: Box::new(ContractTerm::LitInt(0)),
            },
            ContractTerm::Cmp {
                op: CmpOp::Le,
                lhs: Box::new(ContractTerm::Var(ContractVar::Value(out_vid))),
                rhs: Box::new(ContractTerm::LitInt(1)),
            },
        ],
    };
    assert!(
        facts.iter().any(|f| *f == expected),
        "expected value_id-anchored 0 <= Output <= 1 fact after ir_int_cast on Boolean input; got {:?}",
        facts,
    );
}

#[test]
fn ir_int_cast_does_not_fire_when_input_is_int() {
    use crate::circuit_input::InputPath;

    let mut b = crate::builder::IRBuilder::new();
    let x = b.ir_read_integer(InputPath::new("x", vec![]), false);
    let out = b.ir_int_cast(&x);
    let out_vid = out
        .value_id()
        .expect("ir_int_cast output must have a ValueId");

    let bucket = b.facts.per_value.get(&out_vid);
    assert!(
        bucket.map_or(true, |s| s.is_empty()),
        "expected no facts on ir_int_cast output when input is Integer; got {:?}",
        bucket,
    );
}

/// Plant a `Cmp(Value(vid) op LitInt(n))` fact on `vid`'s bucket.
/// Mirrors `plant_cmp_fact` in `optim::resolver` tests.
#[cfg(test)]
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
fn ir_add_i_emits_interval_fact_when_both_inputs_have_facts() {
    use crate::circuit_input::InputPath;
    use crate::optim::predicates::formula::{BoolOp, CmpOp, ContractTerm, ContractVar};

    let mut b = crate::builder::IRBuilder::new();
    let a = b.ir_read_integer(InputPath::new("a", vec![]), false);
    let bv = b.ir_read_integer(InputPath::new("b", vec![]), false);
    let a_vid = a.value_id().unwrap();
    let b_vid = bv.value_id().unwrap();

    // a ∈ [0, 10], b ∈ [0, 5]
    plant_cmp_fact(&mut b, a_vid, CmpOp::Ge, 0);
    plant_cmp_fact(&mut b, a_vid, CmpOp::Le, 10);
    plant_cmp_fact(&mut b, b_vid, CmpOp::Ge, 0);
    plant_cmp_fact(&mut b, b_vid, CmpOp::Le, 5);

    let out = b.ir_add_i(&a, &bv);
    let out_vid = out.value_id().expect("ir_add_i output must have a ValueId");

    let facts = b
        .facts
        .per_value
        .get(&out_vid)
        .expect("ir_add_i should have deposited at least one fact");
    let expected = ContractTerm::BoolComb {
        op: BoolOp::And,
        operands: vec![
            ContractTerm::Cmp {
                op: CmpOp::Ge,
                lhs: Box::new(ContractTerm::Var(ContractVar::Value(out_vid))),
                rhs: Box::new(ContractTerm::LitInt(0)),
            },
            ContractTerm::Cmp {
                op: CmpOp::Le,
                lhs: Box::new(ContractTerm::Var(ContractVar::Value(out_vid))),
                rhs: Box::new(ContractTerm::LitInt(15)),
            },
        ],
    };
    assert!(
        facts.iter().any(|f| *f == expected),
        "expected [0, 15] interval fact on ir_add_i output; got {:?}",
        facts,
    );
}

#[test]
fn ir_add_i_no_fact_when_input_lacks_bounds() {
    use crate::circuit_input::InputPath;

    let mut b = crate::builder::IRBuilder::new();
    let a = b.ir_read_integer(InputPath::new("a", vec![]), false);
    let bv = b.ir_read_integer(InputPath::new("b", vec![]), false);

    let out = b.ir_add_i(&a, &bv);
    let out_vid = out.value_id().expect("ir_add_i output must have a ValueId");

    let bucket = b.facts.per_value.get(&out_vid);
    assert!(
        bucket.map_or(true, |s| s.is_empty()),
        "expected no facts on ir_add_i output when inputs lack bounds; got {:?}",
        bucket,
    );
}

#[test]
fn ir_add_i_no_fact_on_overflow() {
    use crate::circuit_input::InputPath;
    use crate::optim::predicates::formula::CmpOp;

    let mut b = crate::builder::IRBuilder::new();
    let a = b.ir_read_integer(InputPath::new("a", vec![]), false);
    let bv = b.ir_read_integer(InputPath::new("b", vec![]), false);
    let a_vid = a.value_id().unwrap();
    let b_vid = bv.value_id().unwrap();

    // a ∈ [i64::MAX, i64::MAX], b ∈ [1, 1] — checked_add overflows.
    plant_cmp_fact(&mut b, a_vid, CmpOp::Ge, i64::MAX);
    plant_cmp_fact(&mut b, a_vid, CmpOp::Le, i64::MAX);
    plant_cmp_fact(&mut b, b_vid, CmpOp::Ge, 1);
    plant_cmp_fact(&mut b, b_vid, CmpOp::Le, 1);

    let out = b.ir_add_i(&a, &bv);
    let out_vid = out.value_id().expect("ir_add_i output must have a ValueId");

    let bucket = b.facts.per_value.get(&out_vid);
    assert!(
        bucket.map_or(true, |s| s.is_empty()),
        "expected no facts on ir_add_i output when checked_add overflows; got {:?}",
        bucket,
    );
}

// ── relay_reduction_output_interval_int (Group 3b) ─────────────────────

/// Construct `N` integer inputs each with the planted bound `[lo, hi]`
/// and return their value_ids alongside the builder. Helper for the
/// reduction-relay tests below.
#[cfg(test)]
fn planted_int_inputs(
    n: usize,
    lo: i64,
    hi: i64,
) -> (crate::builder::IRBuilder, Vec<crate::types::ValueId>) {
    use crate::circuit_input::InputPath;
    use crate::optim::predicates::formula::CmpOp;

    let mut b = crate::builder::IRBuilder::new();
    let mut vids = Vec::with_capacity(n);
    for i in 0..n {
        let name = format!("arr_{}", i);
        let v = b.ir_read_integer(InputPath::new(&name, vec![]), false);
        let vid = v.value_id().unwrap();
        plant_cmp_fact(&mut b, vid, CmpOp::Ge, lo);
        plant_cmp_fact(&mut b, vid, CmpOp::Le, hi);
        vids.push(vid);
    }
    (b, vids)
}

/// Mint a fresh output value_id by reading another input — the relay
/// helper takes an output_vid that doesn't have to be the real
/// reduction sink; for tests we just need a distinct vid.
#[cfg(test)]
fn fresh_output_vid(b: &mut crate::builder::IRBuilder) -> crate::types::ValueId {
    use crate::circuit_input::InputPath;
    let v = b.ir_read_integer(InputPath::new("output", vec![]), false);
    v.value_id().unwrap()
}

#[test]
fn relay_sum_static_array_yields_n_times_input_interval() {
    use crate::optim::predicates::formula::{BoolOp, CmpOp, ContractTerm, ContractVar};

    // 4 elements each in [0, 5]. Multiplier = N = 4 ⇒ output ∈ [0, 20].
    let (mut b, vids) = planted_int_inputs(4, 0, 5);
    let out_vid = fresh_output_vid(&mut b);

    let emitted = crate::optim::resolver::relay_reduction_output_interval_int(
        &mut b, &vids, out_vid, 4,
    );
    assert!(emitted, "relay should emit when all elements have bounds");

    let facts = b
        .facts
        .per_value
        .get(&out_vid)
        .expect("relay should have deposited a fact bucket");
    let expected = ContractTerm::BoolComb {
        op: BoolOp::And,
        operands: vec![
            ContractTerm::Cmp {
                op: CmpOp::Ge,
                lhs: Box::new(ContractTerm::Var(ContractVar::Value(out_vid))),
                rhs: Box::new(ContractTerm::LitInt(0)),
            },
            ContractTerm::Cmp {
                op: CmpOp::Le,
                lhs: Box::new(ContractTerm::Var(ContractVar::Value(out_vid))),
                rhs: Box::new(ContractTerm::LitInt(20)),
            },
        ],
    };
    assert!(
        facts.iter().any(|f| *f == expected),
        "expected [0, 20] interval fact on sum-relay output; got {:?}",
        facts,
    );
}

#[test]
fn relay_max_static_array_yields_input_interval() {
    use crate::optim::predicates::formula::{BoolOp, CmpOp, ContractTerm, ContractVar};

    // 4 elements each in [-3, 7]. Multiplier = 1 ⇒ output ∈ [-3, 7].
    let (mut b, vids) = planted_int_inputs(4, -3, 7);
    let out_vid = fresh_output_vid(&mut b);

    let emitted = crate::optim::resolver::relay_reduction_output_interval_int(
        &mut b, &vids, out_vid, 1,
    );
    assert!(emitted, "relay should emit when all elements have bounds");

    let facts = b
        .facts
        .per_value
        .get(&out_vid)
        .expect("relay should have deposited a fact bucket");
    let expected = ContractTerm::BoolComb {
        op: BoolOp::And,
        operands: vec![
            ContractTerm::Cmp {
                op: CmpOp::Ge,
                lhs: Box::new(ContractTerm::Var(ContractVar::Value(out_vid))),
                rhs: Box::new(ContractTerm::LitInt(-3)),
            },
            ContractTerm::Cmp {
                op: CmpOp::Le,
                lhs: Box::new(ContractTerm::Var(ContractVar::Value(out_vid))),
                rhs: Box::new(ContractTerm::LitInt(7)),
            },
        ],
    };
    assert!(
        facts.iter().any(|f| *f == expected),
        "expected [-3, 7] interval fact on max-relay output; got {:?}",
        facts,
    );
}

#[test]
fn relay_min_static_array_yields_input_interval() {
    use crate::optim::predicates::formula::{BoolOp, CmpOp, ContractTerm, ContractVar};

    // 3 elements each in [2, 9]. Multiplier = 1 ⇒ output ∈ [2, 9].
    let (mut b, vids) = planted_int_inputs(3, 2, 9);
    let out_vid = fresh_output_vid(&mut b);

    let emitted = crate::optim::resolver::relay_reduction_output_interval_int(
        &mut b, &vids, out_vid, 1,
    );
    assert!(emitted, "relay should emit when all elements have bounds");

    let facts = b
        .facts
        .per_value
        .get(&out_vid)
        .expect("relay should have deposited a fact bucket");
    let expected = ContractTerm::BoolComb {
        op: BoolOp::And,
        operands: vec![
            ContractTerm::Cmp {
                op: CmpOp::Ge,
                lhs: Box::new(ContractTerm::Var(ContractVar::Value(out_vid))),
                rhs: Box::new(ContractTerm::LitInt(2)),
            },
            ContractTerm::Cmp {
                op: CmpOp::Le,
                lhs: Box::new(ContractTerm::Var(ContractVar::Value(out_vid))),
                rhs: Box::new(ContractTerm::LitInt(9)),
            },
        ],
    };
    assert!(
        facts.iter().any(|f| *f == expected),
        "expected [2, 9] interval fact on min-relay output; got {:?}",
        facts,
    );
}

#[test]
fn relay_sum_with_unbounded_input_emits_nothing() {
    use crate::circuit_input::InputPath;

    // Two elements but neither has any planted bound — relay should
    // no-op and leave the output bucket empty.
    let mut b = crate::builder::IRBuilder::new();
    let a = b.ir_read_integer(InputPath::new("a", vec![]), false);
    let bv = b.ir_read_integer(InputPath::new("b", vec![]), false);
    let vids = vec![a.value_id().unwrap(), bv.value_id().unwrap()];
    let out_vid = fresh_output_vid(&mut b);

    let emitted = crate::optim::resolver::relay_reduction_output_interval_int(
        &mut b, &vids, out_vid, 2,
    );
    assert!(!emitted, "relay should refuse to emit when elements unbounded");

    let bucket = b.facts.per_value.get(&out_vid);
    assert!(
        bucket.map_or(true, |s| s.is_empty()),
        "expected no facts on relay output when inputs lack bounds; got {:?}",
        bucket,
    );
}

#[test]
fn relay_sum_with_overflowing_multiplier_emits_nothing() {
    // Element in [1, i64::MAX]; multiplier = 2 ⇒ i64::MAX * 2 overflows.
    let (mut b, vids) = planted_int_inputs(1, 1, i64::MAX);
    let out_vid = fresh_output_vid(&mut b);

    let emitted = crate::optim::resolver::relay_reduction_output_interval_int(
        &mut b, &vids, out_vid, 2,
    );
    assert!(!emitted, "relay should refuse to emit on multiplier overflow");

    let bucket = b.facts.per_value.get(&out_vid);
    assert!(
        bucket.map_or(true, |s| s.is_empty()),
        "expected no facts on relay output on overflow; got {:?}",
        bucket,
    );
}

#[test]
fn relay_aggregates_union_of_heterogeneous_element_bounds() {
    use crate::circuit_input::InputPath;
    use crate::optim::predicates::formula::{BoolOp, CmpOp, ContractTerm, ContractVar};

    // Element a ∈ [0, 5], element b ∈ [-2, 3]; union = [-2, 5].
    // Multiplier = 1 (max-shape relay) ⇒ output ∈ [-2, 5].
    let mut b = crate::builder::IRBuilder::new();
    let a = b.ir_read_integer(InputPath::new("a", vec![]), false);
    let bv = b.ir_read_integer(InputPath::new("b", vec![]), false);
    let a_vid = a.value_id().unwrap();
    let b_vid = bv.value_id().unwrap();
    use crate::optim::predicates::formula::CmpOp as Cmp;
    plant_cmp_fact(&mut b, a_vid, Cmp::Ge, 0);
    plant_cmp_fact(&mut b, a_vid, Cmp::Le, 5);
    plant_cmp_fact(&mut b, b_vid, Cmp::Ge, -2);
    plant_cmp_fact(&mut b, b_vid, Cmp::Le, 3);
    let out_vid = fresh_output_vid(&mut b);

    let emitted = crate::optim::resolver::relay_reduction_output_interval_int(
        &mut b,
        &[a_vid, b_vid],
        out_vid,
        1,
    );
    assert!(emitted, "relay should emit when all elements have bounds");

    let facts = b
        .facts
        .per_value
        .get(&out_vid)
        .expect("relay should have deposited a fact bucket");
    let expected = ContractTerm::BoolComb {
        op: BoolOp::And,
        operands: vec![
            ContractTerm::Cmp {
                op: CmpOp::Ge,
                lhs: Box::new(ContractTerm::Var(ContractVar::Value(out_vid))),
                rhs: Box::new(ContractTerm::LitInt(-2)),
            },
            ContractTerm::Cmp {
                op: CmpOp::Le,
                lhs: Box::new(ContractTerm::Var(ContractVar::Value(out_vid))),
                rhs: Box::new(ContractTerm::LitInt(5)),
            },
        ],
    };
    assert!(
        facts.iter().any(|f| *f == expected),
        "expected [-2, 5] interval fact (union) on relay output; got {:?}",
        facts,
    );
}

#[test]
fn dyn_fill_with_active_emits_runtime_length_eq_active_fact() {
    use crate::circuit_input::InputPath;
    use crate::optim::predicates::formula::{CmpOp, ContractTerm, ContractVar};
    use std::collections::HashMap;

    let mut b = crate::builder::IRBuilder::new();
    let out = b.ir_read_integer(InputPath::new("out_len", vec![]), false);
    let active = b.ir_read_integer(InputPath::new("active", vec![]), false);
    let out_vid = out.value_id().unwrap();
    let active_vid = active.value_id().unwrap();

    let mut formals: HashMap<String, crate::types::ValueId> = HashMap::new();
    formals.insert("active".to_string(), active_vid);
    b.fire_contract("dyn_fill_with_active", out_vid, &formals);

    let facts = b
        .facts
        .per_value
        .get(&out_vid)
        .expect("fire_contract should have deposited at least one fact");

    // (1) Existing Output >= 0 ensures.
    let expected_nonneg = ContractTerm::Cmp {
        op: CmpOp::Ge,
        lhs: Box::new(ContractTerm::Var(ContractVar::Value(out_vid))),
        rhs: Box::new(ContractTerm::LitInt(0)),
    };
    assert!(
        facts.iter().any(|f| *f == expected_nonneg),
        "expected Output >= 0 fact on dyn_fill_with_active output; got {:?}",
        facts,
    );

    // (2) New Output == Formal("active") ensures, post-instantiation.
    let expected_eq = ContractTerm::Cmp {
        op: CmpOp::Eq,
        lhs: Box::new(ContractTerm::Var(ContractVar::Value(out_vid))),
        rhs: Box::new(ContractTerm::Var(ContractVar::Value(active_vid))),
    };
    assert!(
        facts.iter().any(|f| *f == expected_eq),
        "expected Output == active fact on dyn_fill_with_active output; got {:?}",
        facts,
    );
}

#[test]
fn dyn_arange_bounded_emits_length_eq_fact() {
    use crate::circuit_input::InputPath;
    use crate::ops::static_ndarray_ops::np_arange;
    use crate::optim::predicates::formula::{
        ArithOp, CmpOp, ContractTerm, ContractVar,
    };
    use crate::optim::resolver::LayeredResolver;
    use crate::types::Value;

    let mut b = crate::builder::IRBuilder::new();
    b.set_resolver(Box::new(LayeredResolver::range_then_smt_with_budget(500, 4096)));

    let n = b.ir_read_integer(InputPath::new("n", vec![]), false);
    let n_vid = n.value_id().unwrap();
    // n ∈ [0, 10]: planted via `n + n <= 20` and `n + n >= 0` so the
    // outward-doubling probe in `resolve_int_or_bounded` admits it.
    let n_plus_n = ContractTerm::Arith {
        op: ArithOp::Add,
        lhs: Box::new(ContractTerm::Var(ContractVar::Value(n_vid))),
        rhs: Box::new(ContractTerm::Var(ContractVar::Value(n_vid))),
    };
    b.facts.insert_for(
        n_vid,
        ContractTerm::Cmp {
            op: CmpOp::Le,
            lhs: Box::new(n_plus_n.clone()),
            rhs: Box::new(ContractTerm::LitInt(20)),
        },
    );
    b.facts.insert_for(
        n_vid,
        ContractTerm::Cmp {
            op: CmpOp::Ge,
            lhs: Box::new(n_plus_n),
            rhs: Box::new(ContractTerm::LitInt(0)),
        },
    );

    let out = np_arange(&mut b, &[n.clone()]);
    let runtime_length_vid = match &out {
        Value::DynamicNDArray(d) => d.meta.runtime_length.value_id,
        _ => panic!("expected DynamicNDArray, got {:?}", out),
    };

    let facts = b
        .facts
        .per_value
        .get(&runtime_length_vid)
        .expect("dyn_arange should have deposited facts on the runtime_length value");

    // Expected: runtime_length == stop - start. For the 1-arg form,
    // start binds to the ValueId of the literal 0. The output is
    // anchored at the runtime_length ValueId.
    let start_constant_vid = {
        // The bounded path materialised `ir_constant_int(0)` at the
        // firing site. We don't know its ValueId directly; find the
        // single `Sub` fact in the bucket and read its RHS leaves.
        let mut found = None;
        for fact in facts {
            if let ContractTerm::Cmp { op: CmpOp::Eq, rhs, .. } = fact {
                if let ContractTerm::Arith { op: ArithOp::Sub, rhs: r, .. } = rhs.as_ref() {
                    if let ContractTerm::Var(ContractVar::Value(vid)) = r.as_ref() {
                        found = Some(*vid);
                    }
                }
            }
        }
        found.expect("expected an Eq(Sub) fact on runtime_length")
    };
    let expected_eq = ContractTerm::Cmp {
        op: CmpOp::Eq,
        lhs: Box::new(ContractTerm::Var(ContractVar::Value(runtime_length_vid))),
        rhs: Box::new(ContractTerm::Arith {
            op: ArithOp::Sub,
            lhs: Box::new(ContractTerm::Var(ContractVar::Value(n_vid))),
            rhs: Box::new(ContractTerm::Var(ContractVar::Value(start_constant_vid))),
        }),
    };
    assert!(
        facts.iter().any(|f| *f == expected_eq),
        "expected runtime_length == stop - start fact; got {:?}",
        facts,
    );
}

#[test]
fn dyn_tile_bounded_emits_length_eq_fact() {
    use crate::circuit_input::InputPath;
    use crate::ops::static_ndarray_ops::np_tile;
    use crate::optim::predicates::formula::{
        ArithOp, CmpOp, ContractTerm, ContractVar,
    };
    use crate::optim::resolver::LayeredResolver;
    use crate::types::{CompositeData, Value, ValueId, ZinniaType};

    let mut b = crate::builder::IRBuilder::new();
    b.set_resolver(Box::new(LayeredResolver::range_then_smt_with_budget(500, 4096)));

    let k = b.ir_read_integer(InputPath::new("k", vec![]), false);
    let k_vid = k.value_id().unwrap();
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
            rhs: Box::new(ContractTerm::LitInt(0)),
        },
    );

    // 3-element static array.
    let vals: Vec<Value> = (0..3).map(|i| b.ir_constant_float(i as f64)).collect();
    let arr = Value::List(CompositeData {
        elements_type: vec![ZinniaType::Float; 3],
        values: vals,
    
        value_id: ValueId::next(),
    });

    let out = np_tile(&mut b, &[arr, k.clone()]);
    let runtime_length_vid = match &out {
        Value::DynamicNDArray(d) => d.meta.runtime_length.value_id,
        _ => panic!("expected DynamicNDArray, got {:?}", out),
    };

    let facts = b
        .facts
        .per_value
        .get(&runtime_length_vid)
        .expect("dyn_tile should have deposited facts on runtime_length");

    let len_arr_vid = {
        let mut found = None;
        for fact in facts {
            if let ContractTerm::Cmp { op: CmpOp::Eq, rhs, .. } = fact {
                if let ContractTerm::Arith { op: ArithOp::Mul, lhs: l, .. } = rhs.as_ref() {
                    if let ContractTerm::Var(ContractVar::Value(vid)) = l.as_ref() {
                        found = Some(*vid);
                    }
                }
            }
        }
        found.expect("expected an Eq(Mul) fact on runtime_length")
    };
    let expected_eq = ContractTerm::Cmp {
        op: CmpOp::Eq,
        lhs: Box::new(ContractTerm::Var(ContractVar::Value(runtime_length_vid))),
        rhs: Box::new(ContractTerm::Arith {
            op: ArithOp::Mul,
            lhs: Box::new(ContractTerm::Var(ContractVar::Value(len_arr_vid))),
            rhs: Box::new(ContractTerm::Var(ContractVar::Value(k_vid))),
        }),
    };
    assert!(
        facts.iter().any(|f| *f == expected_eq),
        "expected runtime_length == len_arr * k fact; got {:?}",
        facts,
    );
}

#[test]
fn dyn_repeat_bounded_emits_length_eq_fact() {
    use crate::circuit_input::InputPath;
    use crate::ops::static_ndarray_ops::ndarray_repeat;
    use crate::optim::predicates::formula::{
        ArithOp, CmpOp, ContractTerm, ContractVar,
    };
    use crate::optim::resolver::LayeredResolver;
    use crate::types::{CompositeData, Value, ValueId, ZinniaType};

    let mut b = crate::builder::IRBuilder::new();
    b.set_resolver(Box::new(LayeredResolver::range_then_smt_with_budget(500, 4096)));

    let k = b.ir_read_integer(InputPath::new("k", vec![]), false);
    let k_vid = k.value_id().unwrap();
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
            rhs: Box::new(ContractTerm::LitInt(0)),
        },
    );

    let vals: Vec<Value> = (0..3).map(|i| b.ir_constant_float(i as f64)).collect();
    let arr = Value::List(CompositeData {
        elements_type: vec![ZinniaType::Float; 3],
        values: vals,
    
        value_id: ValueId::next(),
    });

    let kwargs = std::collections::HashMap::new();
    let out = ndarray_repeat(&mut b, &arr, &[k.clone()], &kwargs);
    let runtime_length_vid = match &out {
        Value::DynamicNDArray(d) => d.meta.runtime_length.value_id,
        _ => panic!("expected DynamicNDArray, got {:?}", out),
    };

    let facts = b
        .facts
        .per_value
        .get(&runtime_length_vid)
        .expect("dyn_repeat should have deposited facts on runtime_length");

    let len_arr_vid = {
        let mut found = None;
        for fact in facts {
            if let ContractTerm::Cmp { op: CmpOp::Eq, rhs, .. } = fact {
                if let ContractTerm::Arith { op: ArithOp::Mul, lhs: l, .. } = rhs.as_ref() {
                    if let ContractTerm::Var(ContractVar::Value(vid)) = l.as_ref() {
                        found = Some(*vid);
                    }
                }
            }
        }
        found.expect("expected an Eq(Mul) fact on runtime_length")
    };
    let expected_eq = ContractTerm::Cmp {
        op: CmpOp::Eq,
        lhs: Box::new(ContractTerm::Var(ContractVar::Value(runtime_length_vid))),
        rhs: Box::new(ContractTerm::Arith {
            op: ArithOp::Mul,
            lhs: Box::new(ContractTerm::Var(ContractVar::Value(len_arr_vid))),
            rhs: Box::new(ContractTerm::Var(ContractVar::Value(k_vid))),
        }),
    };
    assert!(
        facts.iter().any(|f| *f == expected_eq),
        "expected runtime_length == len_arr * k fact; got {:?}",
        facts,
    );
}

#[test]
fn dyn_linspace_bounded_emits_length_eq_fact() {
    use crate::circuit_input::InputPath;
    use crate::ops::static_ndarray_ops::np_linspace;
    use crate::optim::predicates::formula::{
        ArithOp, CmpOp, ContractTerm, ContractVar,
    };
    use crate::optim::resolver::LayeredResolver;
    use crate::types::{ScalarValue, Value};

    let mut b = crate::builder::IRBuilder::new();
    b.set_resolver(Box::new(LayeredResolver::range_then_smt_with_budget(500, 4096)));

    // num ∈ [2, 10] — bounded admission requires num >= 2 for endpoint.
    let num = b.ir_read_integer(InputPath::new("num", vec![]), false);
    let num_vid = num.value_id().unwrap();
    let num_plus_num = ContractTerm::Arith {
        op: ArithOp::Add,
        lhs: Box::new(ContractTerm::Var(ContractVar::Value(num_vid))),
        rhs: Box::new(ContractTerm::Var(ContractVar::Value(num_vid))),
    };
    b.facts.insert_for(
        num_vid,
        ContractTerm::Cmp {
            op: CmpOp::Le,
            lhs: Box::new(num_plus_num),
            rhs: Box::new(ContractTerm::LitInt(20)),
        },
    );
    b.facts.insert_for(
        num_vid,
        ContractTerm::Cmp {
            op: CmpOp::Ge,
            lhs: Box::new(ContractTerm::Var(ContractVar::Value(num_vid))),
            rhs: Box::new(ContractTerm::LitInt(2)),
        },
    );

    let start = Value::Float(ScalarValue::constant(0.0));
    let stop = Value::Float(ScalarValue::constant(1.0));
    let kwargs = std::collections::HashMap::new();
    let out = np_linspace(&mut b, &[start, stop, num.clone()], &kwargs);
    let runtime_length_vid = match &out {
        Value::DynamicNDArray(d) => d.meta.runtime_length.value_id,
        _ => panic!("expected DynamicNDArray, got {:?}", out),
    };

    let facts = b
        .facts
        .per_value
        .get(&runtime_length_vid)
        .expect("dyn_linspace should have deposited facts on runtime_length");

    let expected_eq = ContractTerm::Cmp {
        op: CmpOp::Eq,
        lhs: Box::new(ContractTerm::Var(ContractVar::Value(runtime_length_vid))),
        rhs: Box::new(ContractTerm::Var(ContractVar::Value(num_vid))),
    };
    assert!(
        facts.iter().any(|f| *f == expected_eq),
        "expected runtime_length == num fact; got {:?}",
        facts,
    );
}

#[test]
fn dyn_identity_bounded_emits_length_eq_fact() {
    use crate::circuit_input::InputPath;
    use crate::ops::static_ndarray_ops::np_identity;
    use crate::optim::predicates::formula::{
        ArithOp, CmpOp, ContractTerm, ContractVar,
    };
    use crate::optim::resolver::LayeredResolver;
    use crate::types::Value;

    let mut b = crate::builder::IRBuilder::new();
    b.set_resolver(Box::new(LayeredResolver::range_then_smt_with_budget(500, 4096)));

    let n = b.ir_read_integer(InputPath::new("n", vec![]), false);
    let n_vid = n.value_id().unwrap();
    let n_plus_n = ContractTerm::Arith {
        op: ArithOp::Add,
        lhs: Box::new(ContractTerm::Var(ContractVar::Value(n_vid))),
        rhs: Box::new(ContractTerm::Var(ContractVar::Value(n_vid))),
    };
    b.facts.insert_for(
        n_vid,
        ContractTerm::Cmp {
            op: CmpOp::Le,
            lhs: Box::new(n_plus_n.clone()),
            rhs: Box::new(ContractTerm::LitInt(20)),
        },
    );
    b.facts.insert_for(
        n_vid,
        ContractTerm::Cmp {
            op: CmpOp::Ge,
            lhs: Box::new(n_plus_n),
            rhs: Box::new(ContractTerm::LitInt(0)),
        },
    );

    let out = np_identity(&mut b, &[n.clone()]);
    let runtime_length_vid = match &out {
        Value::DynamicNDArray(d) => d.meta.runtime_length.value_id,
        _ => panic!("expected DynamicNDArray, got {:?}", out),
    };

    let facts = b
        .facts
        .per_value
        .get(&runtime_length_vid)
        .expect("dyn_identity should have deposited facts on runtime_length");

    let expected_eq = ContractTerm::Cmp {
        op: CmpOp::Eq,
        lhs: Box::new(ContractTerm::Var(ContractVar::Value(runtime_length_vid))),
        rhs: Box::new(ContractTerm::Arith {
            op: ArithOp::Mul,
            lhs: Box::new(ContractTerm::Var(ContractVar::Value(n_vid))),
            rhs: Box::new(ContractTerm::Var(ContractVar::Value(n_vid))),
        }),
    };
    assert!(
        facts.iter().any(|f| *f == expected_eq),
        "expected runtime_length == N * N fact; got {:?}",
        facts,
    );
}

#[test]
fn every_known_ir_kind_resolves_to_at_least_default_contract() {
    // Smoke check: a sample of IR variants spanning all categories must
    // not panic when queried, and (for now) returns the default.
    let samples = [
        IR::Assert,
        IR::AddI,
        IR::ConstantInt { value: 1 },
        IR::ExposePublicI,
        IR::StructuralPredicate {
            kind: "nnz".to_string(),
            args: vec!["x".to_string()],
            op: None,
            bound: None,
        },
    ];
    for ir in &samples {
        let c = op_contract_for(ir);
        assert!(c.is_default(),
                "expected default contract for {:?}; got {:?}", ir, c);
    }
}

#[test]
fn ir_arccos_f_fires_range_fact_on_output_value() {
    use crate::circuit_input::InputPath;
    use crate::optim::predicates::formula::{
        BoolOp, CmpOp, ContractFloat, ContractTerm, ContractVar,
    };

    // Default slack at test time (env var unset).
    let slack: f64 = 0.001;
    let pi = std::f64::consts::PI;

    let mut b = crate::builder::IRBuilder::new();
    let x = b.ir_read_float(InputPath::new("x", vec![]), false);
    let out = b.ir_arccos_f(&x);
    let out_vid = out.value_id().expect("ir_arccos_f output must have a ValueId");

    let facts = b
        .facts
        .per_value
        .get(&out_vid)
        .expect("ir_arccos_f should have deposited at least one fact");
    let expected = ContractTerm::BoolComb {
        op: BoolOp::And,
        operands: vec![
            ContractTerm::Cmp {
                op: CmpOp::Ge,
                lhs: Box::new(ContractTerm::Var(ContractVar::Value(out_vid))),
                rhs: Box::new(ContractTerm::LitFloat(ContractFloat(0.0 - slack))),
            },
            ContractTerm::Cmp {
                op: CmpOp::Le,
                lhs: Box::new(ContractTerm::Var(ContractVar::Value(out_vid))),
                rhs: Box::new(ContractTerm::LitFloat(ContractFloat(pi + slack))),
            },
        ],
    };
    assert!(
        facts.iter().any(|f| *f == expected),
        "expected value_id-anchored [0 - slack, PI + slack] fact after ir_arccos_f; got {:?}",
        facts,
    );
}

#[test]
fn ir_arctan2_f_fires_range_fact_on_output_value() {
    use crate::circuit_input::InputPath;
    use crate::optim::predicates::formula::{
        BoolOp, CmpOp, ContractFloat, ContractTerm, ContractVar,
    };

    let slack: f64 = 0.001;
    let pi = std::f64::consts::PI;

    let mut b = crate::builder::IRBuilder::new();
    let y = b.ir_read_float(InputPath::new("y", vec![]), false);
    let x = b.ir_read_float(InputPath::new("x", vec![]), false);
    let out = b.ir_arctan2_f(&y, &x);
    let out_vid = out.value_id().expect("ir_arctan2_f output must have a ValueId");

    let facts = b
        .facts
        .per_value
        .get(&out_vid)
        .expect("ir_arctan2_f should have deposited at least one fact");
    let expected = ContractTerm::BoolComb {
        op: BoolOp::And,
        operands: vec![
            ContractTerm::Cmp {
                op: CmpOp::Ge,
                lhs: Box::new(ContractTerm::Var(ContractVar::Value(out_vid))),
                rhs: Box::new(ContractTerm::LitFloat(ContractFloat(-pi - slack))),
            },
            ContractTerm::Cmp {
                op: CmpOp::Le,
                lhs: Box::new(ContractTerm::Var(ContractVar::Value(out_vid))),
                rhs: Box::new(ContractTerm::LitFloat(ContractFloat(pi + slack))),
            },
        ],
    };
    assert!(
        facts.iter().any(|f| *f == expected),
        "expected value_id-anchored [-PI - slack, PI + slack] fact after ir_arctan2_f; got {:?}",
        facts,
    );
}

// ---------------------------------------------------------------------------
// Op-build-time `requires` discharge (compiler.op-build-time-requires-discharge)
// ---------------------------------------------------------------------------
//
// Exercises `IRBuilder::discharge_requires` directly with synthetic terms,
// because no entry in the live registry today carries a `requires` clause.

/// Shared mutex serializing env-var-mutating tests for
/// `ZINNIA_OP_REQUIRES_STRICT`. Mirrors the pattern in
/// `helpers/array_ops/indexing.rs` (`STRICT_ENV_LOCK`). Without this,
/// concurrent strict-mode tests race on the env var.
#[cfg(test)]
static REQUIRES_STRICT_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(test)]
struct ScopedRequiresStrict {
    previous: Option<String>,
    _lock: std::sync::MutexGuard<'static, ()>,
}

#[cfg(test)]
impl ScopedRequiresStrict {
    fn set(value: &str) -> Self {
        let lock = REQUIRES_STRICT_ENV_LOCK
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        let previous = std::env::var("ZINNIA_OP_REQUIRES_STRICT").ok();
        std::env::set_var("ZINNIA_OP_REQUIRES_STRICT", value);
        Self { previous, _lock: lock }
    }
}

#[cfg(test)]
impl Drop for ScopedRequiresStrict {
    fn drop(&mut self) {
        match &self.previous {
            Some(v) => std::env::set_var("ZINNIA_OP_REQUIRES_STRICT", v),
            None => std::env::remove_var("ZINNIA_OP_REQUIRES_STRICT"),
        }
    }
}

#[test]
fn discharge_requires_proved_path_does_not_panic_and_emits_no_witness() {
    use crate::circuit_input::InputPath;
    use crate::optim::predicates::formula::{CmpOp, ContractTerm, ContractVar};

    let mut b = crate::builder::IRBuilder::new();
    let v = b.ir_read_integer(InputPath::new("x", vec![]), false);
    let v_vid = v.value_id().unwrap();
    // Plant `v >= 0` so `v >= 0` is trivially provable.
    b.facts.insert_for(
        v_vid,
        ContractTerm::Cmp {
            op: CmpOp::Ge,
            lhs: Box::new(ContractTerm::Var(ContractVar::Value(v_vid))),
            rhs: Box::new(ContractTerm::LitInt(0)),
        },
    );
    let term = ContractTerm::Cmp {
        op: CmpOp::Ge,
        lhs: Box::new(ContractTerm::Var(ContractVar::Value(v_vid))),
        rhs: Box::new(ContractTerm::LitInt(0)),
    };
    let stmt_count_before = b.stmts.len();
    b.discharge_requires("synth_op", &term);
    assert_eq!(
        b.stmts.len(),
        stmt_count_before,
        "Proved path must not emit a witness constraint",
    );
}

#[test]
#[should_panic(expected = "requires precondition disproved")]
fn discharge_requires_disproved_path_panics() {
    use crate::circuit_input::InputPath;
    use crate::optim::predicates::formula::{CmpOp, ContractTerm, ContractVar};

    let mut b = crate::builder::IRBuilder::new();
    let v = b.ir_read_integer(InputPath::new("x", vec![]), false);
    let v_vid = v.value_id().unwrap();
    // Plant `v < 0` (encoded as `v <= -1`) so `v >= 0` is refutable.
    b.facts.insert_for(
        v_vid,
        ContractTerm::Cmp {
            op: CmpOp::Le,
            lhs: Box::new(ContractTerm::Var(ContractVar::Value(v_vid))),
            rhs: Box::new(ContractTerm::LitInt(-1)),
        },
    );
    let term = ContractTerm::Cmp {
        op: CmpOp::Ge,
        lhs: Box::new(ContractTerm::Var(ContractVar::Value(v_vid))),
        rhs: Box::new(ContractTerm::LitInt(0)),
    };
    b.discharge_requires("synth_op", &term);
}

#[test]
fn discharge_requires_unknown_lenient_path_emits_witness_assert() {
    use crate::circuit_input::InputPath;
    use crate::ir_defs::IR;
    use crate::optim::predicates::formula::{CmpOp, ContractTerm, ContractVar};

    // Ensure strict mode is off for this test (default).
    let _guard = ScopedRequiresStrict::set("0");

    let mut b = crate::builder::IRBuilder::new();
    let v = b.ir_read_integer(InputPath::new("x", vec![]), false);
    let v_vid = v.value_id().unwrap();
    // No fact planted on v_vid → `v >= 0` is Unknown.
    let term = ContractTerm::Cmp {
        op: CmpOp::Ge,
        lhs: Box::new(ContractTerm::Var(ContractVar::Value(v_vid))),
        rhs: Box::new(ContractTerm::LitInt(0)),
    };
    let stmt_count_before = b.stmts.len();
    b.discharge_requires("synth_op", &term);
    let stmt_count_after = b.stmts.len();
    assert!(
        stmt_count_after > stmt_count_before,
        "Unknown lenient path must emit witness constraint(s)",
    );
    // The last new statement must be an IR::Assert (the witness
    // emit ends with constraining the lowered Bool to 1).
    let last = b.stmts.last().expect("at least one stmt");
    assert!(
        matches!(last.ir, IR::Assert),
        "last emitted IR must be IR::Assert; got {:?}",
        last.ir,
    );
}

#[test]
#[should_panic(expected = "requires precondition not provable")]
fn discharge_requires_unknown_strict_path_panics() {
    use crate::circuit_input::InputPath;
    use crate::optim::predicates::formula::{CmpOp, ContractTerm, ContractVar};

    let _guard = ScopedRequiresStrict::set("1");

    let mut b = crate::builder::IRBuilder::new();
    let v = b.ir_read_integer(InputPath::new("x", vec![]), false);
    let v_vid = v.value_id().unwrap();
    let term = ContractTerm::Cmp {
        op: CmpOp::Ge,
        lhs: Box::new(ContractTerm::Var(ContractVar::Value(v_vid))),
        rhs: Box::new(ContractTerm::LitInt(0)),
    };
    b.discharge_requires("synth_op", &term);
}

#[test]
#[should_panic(expected = "could not be lowered to a witness constraint")]
fn discharge_requires_unknown_unwitnessable_predicate_panics() {
    // Unknown-lenient + term whose generic lowering returns None
    // (a `PredicateApp` with no per-predicate emitter and no planted
    // fact) must panic with the soundness-floor diagnostic, not
    // silently drop the precondition. Closes the soundness gap from
    // `compiler.op-requires-predicate-witness-soundness`.
    use crate::circuit_input::InputPath;
    use crate::optim::predicates::formula::{ContractTerm, ContractVar};

    // Ensure strict mode is off so we exercise the lenient branch.
    let _guard = ScopedRequiresStrict::set("0");

    let mut b = crate::builder::IRBuilder::new();
    let v = b.ir_read_integer(InputPath::new("x", vec![]), false);
    let v_vid = v.value_id().unwrap();
    // No fact planted on v_vid → `is_sorted(v)` is Unknown.
    // The generic `emit_term_as_bool_value` lowering returns None
    // for any `PredicateApp` (no registered runtime witness emitter
    // for `is_sorted` exists), so the lenient branch must panic
    // rather than continue.
    let term = ContractTerm::PredicateApp {
        kind: "is_sorted".to_string(),
        args: vec![ContractTerm::Var(ContractVar::Value(v_vid))],
    };
    b.discharge_requires("synth_op", &term);
}

#[test]
fn discharge_requires_unknown_witnessable_succeeds() {
    // Regression check: a `Cmp` term still lowers cleanly under
    // Unknown-lenient mode (it produces a Bool SSA wire that's
    // constrained via `IR::Assert`). The new compile-error guard
    // for unemittable terms must not trip on terms the generic
    // emitter handles.
    use crate::circuit_input::InputPath;
    use crate::ir_defs::IR;
    use crate::optim::predicates::formula::{CmpOp, ContractTerm, ContractVar};

    let _guard = ScopedRequiresStrict::set("0");

    let mut b = crate::builder::IRBuilder::new();
    let v = b.ir_read_integer(InputPath::new("x", vec![]), false);
    let v_vid = v.value_id().unwrap();
    // No fact planted on v_vid → `v >= 0` is Unknown.
    let term = ContractTerm::Cmp {
        op: CmpOp::Ge,
        lhs: Box::new(ContractTerm::Var(ContractVar::Value(v_vid))),
        rhs: Box::new(ContractTerm::LitInt(0)),
    };
    let stmt_count_before = b.stmts.len();
    // Should not panic — generic emitter handles `Cmp` cleanly.
    b.discharge_requires("synth_op", &term);
    let stmt_count_after = b.stmts.len();
    assert!(
        stmt_count_after > stmt_count_before,
        "witnessable Unknown lenient path must still emit witness IR",
    );
    let last = b.stmts.last().expect("at least one stmt");
    assert!(
        matches!(last.ir, IR::Assert),
        "last emitted IR must be IR::Assert; got {:?}",
        last.ir,
    );
}

// ---------------------------------------------------------------------------
// Group 5a — `discharge_index_in_range` (Phase E for the indexing chokepoints)
// ---------------------------------------------------------------------------

#[test]
fn discharge_index_in_range_literal_in_bounds_no_panic() {
    // Literal-index fast path: 3 ∈ [0, 10) ⇒ silent no-op, no IR emitted.
    use crate::optim::resolver::discharge_index_in_range;
    use crate::types::ScalarValue;

    let mut b = crate::builder::IRBuilder::new();
    let idx = crate::types::Value::Integer(ScalarValue::constant(3));
    let stmt_count_before = b.stmts.len();
    discharge_index_in_range(&mut b, &idx, 0, 10, "test_op");
    assert_eq!(
        b.stmts.len(),
        stmt_count_before,
        "literal-in-bounds path must not emit any IR",
    );
}

#[test]
#[should_panic(expected = "out of range")]
fn discharge_index_in_range_literal_out_of_bounds_panics() {
    // Literal-index fast path: 15 ∉ [0, 10) ⇒ compile panic.
    use crate::optim::resolver::discharge_index_in_range;
    use crate::types::ScalarValue;

    let mut b = crate::builder::IRBuilder::new();
    let idx = crate::types::Value::Integer(ScalarValue::constant(15));
    discharge_index_in_range(&mut b, &idx, 0, 10, "test_op");
}

#[test]
fn discharge_index_in_range_proved_with_fact_no_witness_emit() {
    // Non-literal idx with planted facts `idx >= 2` and `idx <= 4`
    // makes `0 <= idx < 10` Proved; no witness IR should be emitted.
    use crate::circuit_input::InputPath;
    use crate::optim::predicates::formula::{CmpOp, ContractTerm, ContractVar};
    use crate::optim::resolver::discharge_index_in_range;

    let mut b = crate::builder::IRBuilder::new();
    let idx = b.ir_read_integer(InputPath::new("idx", vec![]), false);
    let idx_vid = idx.value_id().unwrap();
    b.facts.insert_for(
        idx_vid,
        ContractTerm::Cmp {
            op: CmpOp::Ge,
            lhs: Box::new(ContractTerm::Var(ContractVar::Value(idx_vid))),
            rhs: Box::new(ContractTerm::LitInt(2)),
        },
    );
    b.facts.insert_for(
        idx_vid,
        ContractTerm::Cmp {
            op: CmpOp::Le,
            lhs: Box::new(ContractTerm::Var(ContractVar::Value(idx_vid))),
            rhs: Box::new(ContractTerm::LitInt(4)),
        },
    );
    let stmt_count_before = b.stmts.len();
    discharge_index_in_range(&mut b, &idx, 0, 10, "test_op");
    assert_eq!(
        b.stmts.len(),
        stmt_count_before,
        "Proved path must not emit witness IR",
    );
}

#[test]
fn discharge_index_in_range_unknown_lenient_emits_witness() {
    // No facts ⇒ Unknown ⇒ lenient witness emit. The last new IR statement
    // must be IR::Assert, mirroring the discharge_requires emit shape.
    use crate::circuit_input::InputPath;
    use crate::ir_defs::IR;
    use crate::optim::resolver::discharge_index_in_range;

    let _guard = ScopedRequiresStrict::set("0");

    let mut b = crate::builder::IRBuilder::new();
    let idx = b.ir_read_integer(InputPath::new("idx", vec![]), false);
    let stmt_count_before = b.stmts.len();
    discharge_index_in_range(&mut b, &idx, 0, 10, "test_op");
    let stmt_count_after = b.stmts.len();
    assert!(
        stmt_count_after > stmt_count_before,
        "Unknown lenient path must emit witness constraint(s)",
    );
    let last = b.stmts.last().expect("at least one stmt");
    assert!(
        matches!(last.ir, IR::Assert),
        "last emitted IR must be IR::Assert; got {:?}",
        last.ir,
    );
}

#[test]
#[should_panic(expected = "requires precondition disproved")]
fn discharge_index_in_range_disproved_with_fact_panics() {
    // Plant `idx >= 20` so `idx < 10` is refutable ⇒ Disproved path panics
    // through `discharge_requires`.
    use crate::circuit_input::InputPath;
    use crate::optim::predicates::formula::{CmpOp, ContractTerm, ContractVar};
    use crate::optim::resolver::discharge_index_in_range;

    let mut b = crate::builder::IRBuilder::new();
    let idx = b.ir_read_integer(InputPath::new("idx", vec![]), false);
    let idx_vid = idx.value_id().unwrap();
    b.facts.insert_for(
        idx_vid,
        ContractTerm::Cmp {
            op: CmpOp::Ge,
            lhs: Box::new(ContractTerm::Var(ContractVar::Value(idx_vid))),
            rhs: Box::new(ContractTerm::LitInt(20)),
        },
    );
    b.facts.insert_for(
        idx_vid,
        ContractTerm::Cmp {
            op: CmpOp::Le,
            lhs: Box::new(ContractTerm::Var(ContractVar::Value(idx_vid))),
            rhs: Box::new(ContractTerm::LitInt(30)),
        },
    );
    discharge_index_in_range(&mut b, &idx, 0, 10, "test_op");
}

#[test]
fn fire_contract_with_requires_proved_then_publishes_ensures() {
    // End-to-end through `fire_contract`: when an op contract carries a
    // requires whose discharge is Proved, the corresponding ensures
    // facts must still be published on the output value's bucket.
    use crate::circuit_input::InputPath;
    use crate::optim::predicates::formula::{CmpOp, ContractTerm, ContractVar};

    let mut b = crate::builder::IRBuilder::new();
    let out = b.ir_read_integer(InputPath::new("out", vec![]), false);
    let active = b.ir_read_integer(InputPath::new("active", vec![]), false);
    let out_vid = out.value_id().unwrap();
    let active_vid = active.value_id().unwrap();

    let mut formals: std::collections::HashMap<String, crate::types::ValueId> =
        std::collections::HashMap::new();
    formals.insert("active".to_string(), active_vid);
    // The registered `dyn_fill_with_active` contract has an empty
    // `requires`, so we're really exercising the loop's no-op branch
    // and asserting the ensures still publish.
    b.fire_contract("dyn_fill_with_active", out_vid, &formals);

    let facts = b
        .facts
        .per_value
        .get(&out_vid)
        .expect("fire_contract should have deposited at least one ensures fact");
    let expected_nonneg = ContractTerm::Cmp {
        op: CmpOp::Ge,
        lhs: Box::new(ContractTerm::Var(ContractVar::Value(out_vid))),
        rhs: Box::new(ContractTerm::LitInt(0)),
    };
    assert!(facts.iter().any(|f| *f == expected_nonneg));
}

// ---------------------------------------------------------------------------
// Group 1 — sqrt / log / arccos requires discharge through fire_contract.
// ---------------------------------------------------------------------------
//
// Plant a fact on the input ValueId, fire the op contract with that ValueId
// bound as formal `"x"`, and assert: Proved path deposits the ensures (if
// any) without panic; Disproved path panics with the discharge diagnostic.

#[test]
fn fire_contract_sqrt_f_with_satisfying_input_publishes_ensures() {
    use crate::circuit_input::InputPath;
    use crate::optim::predicates::formula::{CmpOp, ContractFloat, ContractTerm, ContractVar};

    let mut b = crate::builder::IRBuilder::new();
    let x = b.ir_read_float(InputPath::new("x", vec![]), false);
    let x_vid = x.value_id().unwrap();
    b.facts.insert_for(
        x_vid,
        ContractTerm::Cmp {
            op: CmpOp::Ge,
            lhs: Box::new(ContractTerm::Var(ContractVar::Value(x_vid))),
            rhs: Box::new(ContractTerm::LitFloat(ContractFloat(0.0))),
        },
    );

    let out = b.ir_sqrt_f(&x);
    let out_vid = out.value_id().expect("ir_sqrt_f output must have a ValueId");

    let facts = b
        .facts
        .per_value
        .get(&out_vid)
        .expect("ir_sqrt_f should have deposited at least one ensures fact");
    let expected_ensures = ContractTerm::Cmp {
        op: CmpOp::Ge,
        lhs: Box::new(ContractTerm::Var(ContractVar::Value(out_vid))),
        rhs: Box::new(ContractTerm::LitFloat(ContractFloat(0.0))),
    };
    assert!(
        facts.iter().any(|f| *f == expected_ensures),
        "expected Output >= 0.0 ensures fact after ir_sqrt_f; got {:?}",
        facts,
    );
}

#[test]
#[should_panic(expected = "requires precondition disproved")]
fn fire_contract_sqrt_f_with_contradicting_input_panics() {
    use crate::circuit_input::InputPath;
    use crate::optim::predicates::formula::{CmpOp, ContractFloat, ContractTerm, ContractVar};

    let mut b = crate::builder::IRBuilder::new();
    let x = b.ir_read_float(InputPath::new("x", vec![]), false);
    let x_vid = x.value_id().unwrap();
    b.facts.insert_for(
        x_vid,
        ContractTerm::Cmp {
            op: CmpOp::Le,
            lhs: Box::new(ContractTerm::Var(ContractVar::Value(x_vid))),
            rhs: Box::new(ContractTerm::LitFloat(ContractFloat(-1.0))),
        },
    );
    let _ = b.ir_sqrt_f(&x);
}

#[test]
fn fire_contract_log_f_with_satisfying_input_does_not_panic() {
    use crate::circuit_input::InputPath;
    use crate::optim::predicates::formula::{CmpOp, ContractFloat, ContractTerm, ContractVar};

    let mut b = crate::builder::IRBuilder::new();
    let x = b.ir_read_float(InputPath::new("x", vec![]), false);
    let x_vid = x.value_id().unwrap();
    // Plant `x >= 1.0`; transitively proves the strict `x > 0.0` requires.
    b.facts.insert_for(
        x_vid,
        ContractTerm::Cmp {
            op: CmpOp::Ge,
            lhs: Box::new(ContractTerm::Var(ContractVar::Value(x_vid))),
            rhs: Box::new(ContractTerm::LitFloat(ContractFloat(1.0))),
        },
    );
    let out = b.ir_log_f(&x);
    assert!(out.value_id().is_some(), "ir_log_f output must have a ValueId");
}

#[test]
#[should_panic(expected = "requires precondition disproved")]
fn fire_contract_log_f_with_contradicting_input_panics() {
    use crate::circuit_input::InputPath;
    use crate::optim::predicates::formula::{CmpOp, ContractFloat, ContractTerm, ContractVar};

    let mut b = crate::builder::IRBuilder::new();
    let x = b.ir_read_float(InputPath::new("x", vec![]), false);
    let x_vid = x.value_id().unwrap();
    // Plant `x <= -1.0`; refutes the strict-positive requires.
    b.facts.insert_for(
        x_vid,
        ContractTerm::Cmp {
            op: CmpOp::Le,
            lhs: Box::new(ContractTerm::Var(ContractVar::Value(x_vid))),
            rhs: Box::new(ContractTerm::LitFloat(ContractFloat(-1.0))),
        },
    );
    let _ = b.ir_log_f(&x);
}

#[test]
fn fire_contract_arccos_f_with_satisfying_input_publishes_ensures() {
    use crate::circuit_input::InputPath;
    use crate::optim::predicates::formula::{
        BoolOp, CmpOp, ContractFloat, ContractTerm, ContractVar,
    };

    let mut b = crate::builder::IRBuilder::new();
    let x = b.ir_read_float(InputPath::new("x", vec![]), false);
    let x_vid = x.value_id().unwrap();
    // Plant `-1 <= x <= 1` as a conjunction so the BoolComb requires is
    // discharged Proved.
    b.facts.insert_for(
        x_vid,
        ContractTerm::BoolComb {
            op: BoolOp::And,
            operands: vec![
                ContractTerm::Cmp {
                    op: CmpOp::Ge,
                    lhs: Box::new(ContractTerm::Var(ContractVar::Value(x_vid))),
                    rhs: Box::new(ContractTerm::LitFloat(ContractFloat(-1.0))),
                },
                ContractTerm::Cmp {
                    op: CmpOp::Le,
                    lhs: Box::new(ContractTerm::Var(ContractVar::Value(x_vid))),
                    rhs: Box::new(ContractTerm::LitFloat(ContractFloat(1.0))),
                },
            ],
        },
    );

    let out = b.ir_arccos_f(&x);
    let out_vid = out
        .value_id()
        .expect("ir_arccos_f output must have a ValueId");

    let facts = b
        .facts
        .per_value
        .get(&out_vid)
        .expect("ir_arccos_f should have deposited at least one ensures fact");
    // The arccos ensures is the existing `[0 - slack, π + slack]` range.
    // Don't pin the slack; just confirm the BoolComb range fact landed.
    assert!(
        facts
            .iter()
            .any(|f| matches!(f, ContractTerm::BoolComb { .. })),
        "expected output-range ensures fact after ir_arccos_f; got {:?}",
        facts,
    );
}

#[test]
#[should_panic(expected = "requires precondition disproved")]
fn fire_contract_arccos_f_with_contradicting_input_panics() {
    use crate::circuit_input::InputPath;
    use crate::optim::predicates::formula::{CmpOp, ContractFloat, ContractTerm, ContractVar};

    let mut b = crate::builder::IRBuilder::new();
    let x = b.ir_read_float(InputPath::new("x", vec![]), false);
    let x_vid = x.value_id().unwrap();
    // Plant `x >= 2.0`; the closed `x <= 1.0` half of the requires is
    // refuted, so the conjunction is Disproved.
    b.facts.insert_for(
        x_vid,
        ContractTerm::Cmp {
            op: CmpOp::Ge,
            lhs: Box::new(ContractTerm::Var(ContractVar::Value(x_vid))),
            rhs: Box::new(ContractTerm::LitFloat(ContractFloat(2.0))),
        },
    );
    let _ = b.ir_arccos_f(&x);
}

// ---------------------------------------------------------------------------
// div / floor_div / mod — divisor-nonzero requires discharge.
// ---------------------------------------------------------------------------

#[test]
fn fire_contract_div_i_with_nonzero_rhs_no_panic() {
    use crate::circuit_input::InputPath;
    use crate::optim::predicates::formula::{CmpOp, ContractTerm, ContractVar};

    let mut b = crate::builder::IRBuilder::new();
    let a = b.ir_read_integer(InputPath::new("a", vec![]), false);
    let r = b.ir_read_integer(InputPath::new("b", vec![]), false);
    let r_vid = r.value_id().unwrap();
    // Plant `b >= 1`; transitively proves `b != 0`.
    b.facts.insert_for(
        r_vid,
        ContractTerm::Cmp {
            op: CmpOp::Ge,
            lhs: Box::new(ContractTerm::Var(ContractVar::Value(r_vid))),
            rhs: Box::new(ContractTerm::LitInt(1)),
        },
    );
    let out = b.ir_div_i(&a, &r);
    assert!(out.value_id().is_some(), "ir_div_i output must have a ValueId");
}

#[test]
#[should_panic(expected = "requires precondition disproved")]
fn fire_contract_div_i_with_zero_rhs_panics() {
    use crate::circuit_input::InputPath;
    use crate::optim::predicates::formula::{CmpOp, ContractTerm, ContractVar};

    let mut b = crate::builder::IRBuilder::new();
    let a = b.ir_read_integer(InputPath::new("a", vec![]), false);
    let r = b.ir_read_integer(InputPath::new("b", vec![]), false);
    let r_vid = r.value_id().unwrap();
    // Plant `b == 0`; refutes the `b != 0` requires.
    b.facts.insert_for(
        r_vid,
        ContractTerm::Cmp {
            op: CmpOp::Eq,
            lhs: Box::new(ContractTerm::Var(ContractVar::Value(r_vid))),
            rhs: Box::new(ContractTerm::LitInt(0)),
        },
    );
    let _ = b.ir_div_i(&a, &r);
}

#[test]
fn fire_contract_div_f_with_nonzero_rhs_no_panic() {
    use crate::circuit_input::InputPath;
    use crate::optim::predicates::formula::{CmpOp, ContractFloat, ContractTerm, ContractVar};

    let mut b = crate::builder::IRBuilder::new();
    let a = b.ir_read_float(InputPath::new("a", vec![]), false);
    let r = b.ir_read_float(InputPath::new("b", vec![]), false);
    let r_vid = r.value_id().unwrap();
    // Plant `b >= 1.0`; transitively proves `b != 0.0`.
    b.facts.insert_for(
        r_vid,
        ContractTerm::Cmp {
            op: CmpOp::Ge,
            lhs: Box::new(ContractTerm::Var(ContractVar::Value(r_vid))),
            rhs: Box::new(ContractTerm::LitFloat(ContractFloat(1.0))),
        },
    );
    let out = b.ir_div_f(&a, &r);
    assert!(out.value_id().is_some(), "ir_div_f output must have a ValueId");
}

#[test]
fn fire_contract_mod_i_emits_requires() {
    // Sanity: the mod_i registry entry carries a single `rhs != 0` requires.
    use crate::optim::predicates::{op_contract_by_name, ContractTerm};
    let c = op_contract_by_name("mod_i");
    assert!(!c.is_default());
    assert_eq!(c.requires.len(), 1);
    assert!(c.ensures.is_empty());
    assert!(matches!(c.requires[0].term, ContractTerm::Cmp { .. }));
}

#[test]
fn fire_contract_inv_i_with_nonzero_no_panic() {
    use crate::circuit_input::InputPath;
    use crate::optim::predicates::formula::{CmpOp, ContractTerm, ContractVar};

    let mut b = crate::builder::IRBuilder::new();
    let x = b.ir_read_integer(InputPath::new("x", vec![]), false);
    let x_vid = x.value_id().unwrap();
    b.facts.insert_for(
        x_vid,
        ContractTerm::Cmp {
            op: CmpOp::Ge,
            lhs: Box::new(ContractTerm::Var(ContractVar::Value(x_vid))),
            rhs: Box::new(ContractTerm::LitInt(1)),
        },
    );
    let out = b.ir_inv_i(&x);
    assert!(out.value_id().is_some(), "ir_inv_i output must have a ValueId");
}

#[test]
#[should_panic(expected = "requires precondition disproved")]
fn fire_contract_inv_i_with_zero_panics() {
    use crate::circuit_input::InputPath;
    use crate::optim::predicates::formula::{CmpOp, ContractTerm, ContractVar};

    let mut b = crate::builder::IRBuilder::new();
    let x = b.ir_read_integer(InputPath::new("x", vec![]), false);
    let x_vid = x.value_id().unwrap();
    b.facts.insert_for(
        x_vid,
        ContractTerm::Cmp {
            op: CmpOp::Eq,
            lhs: Box::new(ContractTerm::Var(ContractVar::Value(x_vid))),
            rhs: Box::new(ContractTerm::LitInt(0)),
        },
    );
    let _ = b.ir_inv_i(&x);
}

// ---------------------------------------------------------------------------
// pow_i / pow_f — `base != 0 OR exp >= 0` domain requires discharge.
// ---------------------------------------------------------------------------

#[test]
fn fire_contract_pow_i_with_nonzero_base_no_panic() {
    use crate::circuit_input::InputPath;
    use crate::optim::predicates::formula::{CmpOp, ContractTerm, ContractVar};

    let mut b = crate::builder::IRBuilder::new();
    let base = b.ir_read_integer(InputPath::new("b", vec![]), false);
    let exp = b.ir_read_integer(InputPath::new("e", vec![]), false);
    let base_vid = base.value_id().unwrap();
    // Plant `b >= 1`; transitively proves `b != 0`, satisfying the first
    // branch of the OR-form requires.
    b.facts.insert_for(
        base_vid,
        ContractTerm::Cmp {
            op: CmpOp::Ge,
            lhs: Box::new(ContractTerm::Var(ContractVar::Value(base_vid))),
            rhs: Box::new(ContractTerm::LitInt(1)),
        },
    );
    let out = b.ir_pow_i(&base, &exp);
    assert!(out.value_id().is_some(), "ir_pow_i output must have a ValueId");
}

#[test]
fn fire_contract_pow_i_with_nonneg_exp_no_panic() {
    use crate::circuit_input::InputPath;
    use crate::optim::predicates::formula::{CmpOp, ContractTerm, ContractVar};

    let mut b = crate::builder::IRBuilder::new();
    let base = b.ir_read_integer(InputPath::new("b", vec![]), false);
    let exp = b.ir_read_integer(InputPath::new("e", vec![]), false);
    let exp_vid = exp.value_id().unwrap();
    // Plant `e >= 0`; satisfies the second branch of the OR-form requires.
    b.facts.insert_for(
        exp_vid,
        ContractTerm::Cmp {
            op: CmpOp::Ge,
            lhs: Box::new(ContractTerm::Var(ContractVar::Value(exp_vid))),
            rhs: Box::new(ContractTerm::LitInt(0)),
        },
    );
    let out = b.ir_pow_i(&base, &exp);
    assert!(out.value_id().is_some(), "ir_pow_i output must have a ValueId");
}

#[test]
#[should_panic(expected = "requires precondition disproved")]
fn fire_contract_pow_i_with_zero_base_negative_exp_panics() {
    use crate::circuit_input::InputPath;
    use crate::optim::predicates::formula::{CmpOp, ContractTerm, ContractVar};

    let mut b = crate::builder::IRBuilder::new();
    let base = b.ir_read_integer(InputPath::new("b", vec![]), false);
    let exp = b.ir_read_integer(InputPath::new("e", vec![]), false);
    let base_vid = base.value_id().unwrap();
    let exp_vid = exp.value_id().unwrap();
    // Plant `b == 0` AND `e <= -1`; refutes both branches of the OR.
    b.facts.insert_for(
        base_vid,
        ContractTerm::Cmp {
            op: CmpOp::Eq,
            lhs: Box::new(ContractTerm::Var(ContractVar::Value(base_vid))),
            rhs: Box::new(ContractTerm::LitInt(0)),
        },
    );
    b.facts.insert_for(
        exp_vid,
        ContractTerm::Cmp {
            op: CmpOp::Le,
            lhs: Box::new(ContractTerm::Var(ContractVar::Value(exp_vid))),
            rhs: Box::new(ContractTerm::LitInt(-1)),
        },
    );
    let _ = b.ir_pow_i(&base, &exp);
}

#[test]
fn fire_contract_pow_f_with_nonzero_base_no_panic() {
    use crate::circuit_input::InputPath;
    use crate::optim::predicates::formula::{CmpOp, ContractFloat, ContractTerm, ContractVar};

    let mut b = crate::builder::IRBuilder::new();
    let base = b.ir_read_float(InputPath::new("b", vec![]), false);
    let exp = b.ir_read_float(InputPath::new("e", vec![]), false);
    let base_vid = base.value_id().unwrap();
    // Plant `b >= 1.0`; transitively proves `b != 0.0`.
    b.facts.insert_for(
        base_vid,
        ContractTerm::Cmp {
            op: CmpOp::Ge,
            lhs: Box::new(ContractTerm::Var(ContractVar::Value(base_vid))),
            rhs: Box::new(ContractTerm::LitFloat(ContractFloat(1.0))),
        },
    );
    let out = b.ir_pow_f(&base, &exp);
    assert!(out.value_id().is_some(), "ir_pow_f output must have a ValueId");
}

#[test]
fn fire_contract_pow_f_with_nonneg_exp_no_panic() {
    use crate::circuit_input::InputPath;
    use crate::optim::predicates::formula::{CmpOp, ContractFloat, ContractTerm, ContractVar};

    let mut b = crate::builder::IRBuilder::new();
    let base = b.ir_read_float(InputPath::new("b", vec![]), false);
    let exp = b.ir_read_float(InputPath::new("e", vec![]), false);
    let exp_vid = exp.value_id().unwrap();
    // Plant `e >= 0.0`; satisfies the second branch of the OR-form requires.
    b.facts.insert_for(
        exp_vid,
        ContractTerm::Cmp {
            op: CmpOp::Ge,
            lhs: Box::new(ContractTerm::Var(ContractVar::Value(exp_vid))),
            rhs: Box::new(ContractTerm::LitFloat(ContractFloat(0.0))),
        },
    );
    let out = b.ir_pow_f(&base, &exp);
    assert!(out.value_id().is_some(), "ir_pow_f output must have a ValueId");
}

#[test]
fn fire_contract_all_yields_output_bool_facts() {
    // Firing "all" on a synthetic output value should plant both halves
    // of the bool-range fact (`Output >= 0` and `Output <= 1`) on the
    // output's fact bucket.
    use crate::circuit_input::InputPath;
    use crate::optim::predicates::formula::{CmpOp, ContractTerm, ContractVar};
    use std::collections::HashMap;

    let mut b = crate::builder::IRBuilder::new();
    let v = b.ir_read_integer(InputPath::new("out", vec![]), false);
    let v_vid = v.value_id().unwrap();

    b.fire_contract("all", v_vid, &HashMap::new());

    let facts = b
        .facts
        .per_value
        .get(&v_vid)
        .expect("fire_contract(all) should have deposited at least one fact");
    let ge_zero = ContractTerm::Cmp {
        op: CmpOp::Ge,
        lhs: Box::new(ContractTerm::Var(ContractVar::Value(v_vid))),
        rhs: Box::new(ContractTerm::LitInt(0)),
    };
    let le_one = ContractTerm::Cmp {
        op: CmpOp::Le,
        lhs: Box::new(ContractTerm::Var(ContractVar::Value(v_vid))),
        rhs: Box::new(ContractTerm::LitInt(1)),
    };
    let has_ge = facts.iter().any(|f| match f {
        ContractTerm::BoolComb { operands, .. } => operands.iter().any(|o| *o == ge_zero),
        f => *f == ge_zero,
    });
    let has_le = facts.iter().any(|f| match f {
        ContractTerm::BoolComb { operands, .. } => operands.iter().any(|o| *o == le_one),
        f => *f == le_one,
    });
    assert!(has_ge, "expected Output >= 0 fact after fire_contract(all); got {:?}", facts);
    assert!(has_le, "expected Output <= 1 fact after fire_contract(all); got {:?}", facts);
}

#[test]
fn fire_contract_any_yields_output_bool_facts() {
    use crate::circuit_input::InputPath;
    use crate::optim::predicates::formula::{CmpOp, ContractTerm, ContractVar};
    use std::collections::HashMap;

    let mut b = crate::builder::IRBuilder::new();
    let v = b.ir_read_integer(InputPath::new("out", vec![]), false);
    let v_vid = v.value_id().unwrap();

    b.fire_contract("any", v_vid, &HashMap::new());

    let facts = b
        .facts
        .per_value
        .get(&v_vid)
        .expect("fire_contract(any) should have deposited at least one fact");
    let ge_zero = ContractTerm::Cmp {
        op: CmpOp::Ge,
        lhs: Box::new(ContractTerm::Var(ContractVar::Value(v_vid))),
        rhs: Box::new(ContractTerm::LitInt(0)),
    };
    let le_one = ContractTerm::Cmp {
        op: CmpOp::Le,
        lhs: Box::new(ContractTerm::Var(ContractVar::Value(v_vid))),
        rhs: Box::new(ContractTerm::LitInt(1)),
    };
    let has_ge = facts.iter().any(|f| match f {
        ContractTerm::BoolComb { operands, .. } => operands.iter().any(|o| *o == ge_zero),
        f => *f == ge_zero,
    });
    let has_le = facts.iter().any(|f| match f {
        ContractTerm::BoolComb { operands, .. } => operands.iter().any(|o| *o == le_one),
        f => *f == le_one,
    });
    assert!(has_ge, "expected Output >= 0 fact after fire_contract(any); got {:?}", facts);
    assert!(has_le, "expected Output <= 1 fact after fire_contract(any); got {:?}", facts);
}

#[test]
fn fire_contract_argextremum_includes_lt_len_upper_bound() {
    // Firing `dyn_argextremum` with both Output and `len_arr` bound
    // should plant the lower bound (`Output >= 0`) and the new symbolic
    // upper bound (`Output < len_arr_vid`).
    use crate::circuit_input::InputPath;
    use crate::optim::predicates::formula::{CmpOp, ContractTerm, ContractVar};
    use std::collections::HashMap;

    let mut b = crate::builder::IRBuilder::new();
    let idx = b.ir_read_integer(InputPath::new("idx", vec![]), false);
    let idx_vid = idx.value_id().unwrap();
    let len = b.ir_read_integer(InputPath::new("len", vec![]), false);
    let len_vid = len.value_id().unwrap();

    let mut formals = HashMap::new();
    formals.insert("len_arr".to_string(), len_vid);
    b.fire_contract("dyn_argextremum", idx_vid, &formals);

    let facts = b
        .facts
        .per_value
        .get(&idx_vid)
        .expect("fire_contract(dyn_argextremum) should have deposited facts");

    let ge_zero = ContractTerm::Cmp {
        op: CmpOp::Ge,
        lhs: Box::new(ContractTerm::Var(ContractVar::Value(idx_vid))),
        rhs: Box::new(ContractTerm::LitInt(0)),
    };
    let lt_len = ContractTerm::Cmp {
        op: CmpOp::Lt,
        lhs: Box::new(ContractTerm::Var(ContractVar::Value(idx_vid))),
        rhs: Box::new(ContractTerm::Var(ContractVar::Value(len_vid))),
    };
    assert!(
        facts.iter().any(|f| *f == ge_zero),
        "expected Output >= 0 lower bound; got {:?}",
        facts,
    );
    assert!(
        facts.iter().any(|f| *f == lt_len),
        "expected Output < len_arr_vid upper bound; got {:?}",
        facts,
    );
}

// ── Group 4a (compiler.op-fact-group-4a-fill-constructors-forall-eq-const) ──

#[test]
fn fire_contract_zeros_content_deposits_forall_eq_const_zero() {
    use crate::circuit_input::InputPath;
    use crate::optim::predicates::formula::{ContractTerm, ContractVar};
    use crate::optim::prove::ProveOutcome;
    use std::collections::HashMap;

    let mut b = crate::builder::IRBuilder::new();
    let out = b.ir_read_integer(InputPath::new("out", vec![]), false);
    let out_vid = out.value_id().unwrap();

    b.fire_contract("zeros_content", out_vid, &HashMap::new());

    let query = ContractTerm::PredicateApp {
        kind: "forall_eq_const".to_string(),
        args: vec![
            ContractTerm::Var(ContractVar::Value(out_vid)),
            ContractTerm::LitInt(0),
        ],
    };
    assert_eq!(b.prove(&query), ProveOutcome::Proved);
}

#[test]
fn fire_contract_ones_content_deposits_forall_eq_const_one() {
    use crate::circuit_input::InputPath;
    use crate::optim::predicates::formula::{ContractTerm, ContractVar};
    use crate::optim::prove::ProveOutcome;
    use std::collections::HashMap;

    let mut b = crate::builder::IRBuilder::new();
    let out = b.ir_read_integer(InputPath::new("out", vec![]), false);
    let out_vid = out.value_id().unwrap();

    b.fire_contract("ones_content", out_vid, &HashMap::new());

    let query = ContractTerm::PredicateApp {
        kind: "forall_eq_const".to_string(),
        args: vec![
            ContractTerm::Var(ContractVar::Value(out_vid)),
            ContractTerm::LitInt(1),
        ],
    };
    assert_eq!(b.prove(&query), ProveOutcome::Proved);
}

#[test]
fn fire_contract_zeros_content_does_not_imply_forall_eq_one() {
    // Soundness: planting `forall_eq_const(out, 0)` must not let the
    // resolver conclude `forall_eq_const(out, 1)`. The predicate is
    // encoded as a cached uninterpreted Bool keyed by (kind, args), so
    // the two queries lower to different symbols.
    use crate::circuit_input::InputPath;
    use crate::optim::predicates::formula::{ContractTerm, ContractVar};
    use crate::optim::prove::ProveOutcome;
    use std::collections::HashMap;

    let mut b = crate::builder::IRBuilder::new();
    let out = b.ir_read_integer(InputPath::new("out", vec![]), false);
    let out_vid = out.value_id().unwrap();

    b.fire_contract("zeros_content", out_vid, &HashMap::new());

    let query_one = ContractTerm::PredicateApp {
        kind: "forall_eq_const".to_string(),
        args: vec![
            ContractTerm::Var(ContractVar::Value(out_vid)),
            ContractTerm::LitInt(1),
        ],
    };
    assert_eq!(b.prove(&query_one), ProveOutcome::Unknown);
}

// ── Group 8b (compiler.op-fact-group-8b-matmul-identity-short-circuit) ──

#[test]
fn fire_contract_identity_content_emits_is_identity_predicate() {
    // Firing `identity_content` plants `is_identity(Output)`. The cached
    // uninterpreted Bool encoding means `prove(is_identity(out_vid))`
    // returns `Proved` only when the matching fact is in scope.
    use crate::circuit_input::InputPath;
    use crate::optim::predicates::formula::{ContractTerm, ContractVar};
    use crate::optim::prove::ProveOutcome;
    use std::collections::HashMap;

    let mut b = crate::builder::IRBuilder::new();
    let out = b.ir_read_integer(InputPath::new("out", vec![]), false);
    let out_vid = out.value_id().unwrap();

    b.fire_contract("identity_content", out_vid, &HashMap::new());

    let query = ContractTerm::PredicateApp {
        kind: "is_identity".to_string(),
        args: vec![ContractTerm::Var(ContractVar::Value(out_vid))],
    };
    assert_eq!(b.prove(&query), ProveOutcome::Proved);
}

#[test]
fn fire_contract_identity_content_does_not_imply_other_arrays() {
    // Soundness: planting `is_identity(out_a)` must not let the
    // resolver conclude `is_identity(out_b)` for a different value.
    use crate::circuit_input::InputPath;
    use crate::optim::predicates::formula::{ContractTerm, ContractVar};
    use crate::optim::prove::ProveOutcome;
    use std::collections::HashMap;

    let mut b = crate::builder::IRBuilder::new();
    let out_a = b.ir_read_integer(InputPath::new("a", vec![]), false);
    let out_b = b.ir_read_integer(InputPath::new("b", vec![]), false);
    let a_vid = out_a.value_id().unwrap();
    let b_vid = out_b.value_id().unwrap();

    b.fire_contract("identity_content", a_vid, &HashMap::new());

    let query_b = ContractTerm::PredicateApp {
        kind: "is_identity".to_string(),
        args: vec![ContractTerm::Var(ContractVar::Value(b_vid))],
    };
    assert_eq!(b.prove(&query_b), ProveOutcome::Unknown);
}

#[test]
fn np_identity_static_path_fires_identity_content() {
    // Integration: `np_identity` static path produces a StaticArray
    // value carrying `is_identity(out)` as a planted fact.
    use crate::optim::predicates::formula::{ContractTerm, ContractVar};
    use crate::optim::prove::ProveOutcome;
    use crate::types::Value;

    let mut b = crate::builder::IRBuilder::new();
    let n = b.ir_constant_int(4);
    let out = crate::ops::static_ndarray_ops::np_identity(&mut b, &[n]);
    let out_vid = match &out {
        Value::StaticArray { value_id, .. } => *value_id,
        other => panic!("expected StaticArray from static np_identity, got {:?}", other.zinnia_type()),
    };

    let query = ContractTerm::PredicateApp {
        kind: "is_identity".to_string(),
        args: vec![ContractTerm::Var(ContractVar::Value(out_vid))],
    };
    assert_eq!(b.prove(&query), ProveOutcome::Proved);
}

#[test]
fn np_fill_call_site_fires_zeros_for_zero_fill_value() {
    // Integration: exercise `np_fill` end-to-end. The bounded-axis
    // path produces a `DynamicNDArray` whose `value_id` we use to query
    // `forall_eq_const`. Plant `n + n <= 20` / `n + n >= 0` so the
    // outward-doubling probe in `resolve_int_or_bounded` admits `n` as
    // bounded — this routes through `dyn_fill_with_active`.
    use crate::circuit_input::InputPath;
    use crate::ops::static_ndarray_ops::np_fill;
    use crate::optim::predicates::formula::{
        ArithOp, CmpOp, ContractTerm, ContractVar,
    };
    use crate::optim::prove::ProveOutcome;
    use crate::optim::resolver::LayeredResolver;
    use crate::types::Value;
    use std::collections::HashMap;

    let mut b = crate::builder::IRBuilder::new();
    b.set_resolver(Box::new(LayeredResolver::range_then_smt_with_budget(500, 4096)));

    let n = b.ir_read_integer(InputPath::new("n", vec![]), false);
    let n_vid = n.value_id().unwrap();
    let n_plus_n = ContractTerm::Arith {
        op: ArithOp::Add,
        lhs: Box::new(ContractTerm::Var(ContractVar::Value(n_vid))),
        rhs: Box::new(ContractTerm::Var(ContractVar::Value(n_vid))),
    };
    b.facts.insert_for(
        n_vid,
        ContractTerm::Cmp {
            op: CmpOp::Le,
            lhs: Box::new(n_plus_n.clone()),
            rhs: Box::new(ContractTerm::LitInt(20)),
        },
    );
    b.facts.insert_for(
        n_vid,
        ContractTerm::Cmp {
            op: CmpOp::Ge,
            lhs: Box::new(n_plus_n),
            rhs: Box::new(ContractTerm::LitInt(0)),
        },
    );

    let out = np_fill(&mut b, &[n.clone()], &HashMap::new(), 0);
    let out_vid = match &out {
        Value::DynamicNDArray(d) => d.value_id,
        other => panic!("expected dyn-ndarray from bounded np_fill, got {:?}", other.zinnia_type()),
    };

    let zeros_query = ContractTerm::PredicateApp {
        kind: "forall_eq_const".to_string(),
        args: vec![
            ContractTerm::Var(ContractVar::Value(out_vid)),
            ContractTerm::LitInt(0),
        ],
    };
    assert_eq!(
        b.prove(&zeros_query),
        ProveOutcome::Proved,
        "np_fill(n, fill_value=0) should fire zeros_content on the dyn output",
    );

    // Sanity: ones-fill at the same site should fire ones_content, not zeros.
    let mut b2 = crate::builder::IRBuilder::new();
    b2.set_resolver(Box::new(LayeredResolver::range_then_smt_with_budget(500, 4096)));
    let n2 = b2.ir_read_integer(InputPath::new("n", vec![]), false);
    let n2_vid = n2.value_id().unwrap();
    let n2_plus_n2 = ContractTerm::Arith {
        op: ArithOp::Add,
        lhs: Box::new(ContractTerm::Var(ContractVar::Value(n2_vid))),
        rhs: Box::new(ContractTerm::Var(ContractVar::Value(n2_vid))),
    };
    b2.facts.insert_for(
        n2_vid,
        ContractTerm::Cmp {
            op: CmpOp::Le,
            lhs: Box::new(n2_plus_n2.clone()),
            rhs: Box::new(ContractTerm::LitInt(20)),
        },
    );
    b2.facts.insert_for(
        n2_vid,
        ContractTerm::Cmp {
            op: CmpOp::Ge,
            lhs: Box::new(n2_plus_n2),
            rhs: Box::new(ContractTerm::LitInt(0)),
        },
    );
    let out_ones = np_fill(&mut b2, &[n2.clone()], &HashMap::new(), 1);
    let out_ones_vid = match &out_ones {
        Value::DynamicNDArray(d) => d.value_id,
        other => panic!("expected dyn-ndarray, got {:?}", other.zinnia_type()),
    };
    let ones_query = ContractTerm::PredicateApp {
        kind: "forall_eq_const".to_string(),
        args: vec![
            ContractTerm::Var(ContractVar::Value(out_ones_vid)),
            ContractTerm::LitInt(1),
        ],
    };
    assert_eq!(b2.prove(&ones_query), ProveOutcome::Proved);
}

// ── Group 4b: range constructors emit is_sorted ───────────────────────
//
// `np.arange(stop)` and `np.linspace(a, b, N)` produce strictly /
// non-strictly ascending arrays when the direction check holds
// (step > 0 / a <= b). The constructor fires `is_sorted(out)` on the
// array's value_id so downstream Phase-F consumers (boundary-read on
// max/min/argmax/argmin) can specialize. Descending or unknown-direction
// forms simply skip the fire — no false claim.

#[test]
fn fire_contract_arange_is_sorted_deposits_is_sorted() {
    use crate::circuit_input::InputPath;
    use crate::optim::predicates::formula::{ContractTerm, ContractVar};
    use crate::optim::prove::ProveOutcome;
    use std::collections::HashMap;

    let mut b = crate::builder::IRBuilder::new();
    let out = b.ir_read_integer(InputPath::new("out", vec![]), false);
    let out_vid = out.value_id().unwrap();

    b.fire_contract("arange_is_sorted", out_vid, &HashMap::new());

    let query = ContractTerm::PredicateApp {
        kind: "is_sorted".to_string(),
        args: vec![ContractTerm::Var(ContractVar::Value(out_vid))],
    };
    assert_eq!(b.prove(&query), ProveOutcome::Proved);
}

#[test]
fn fire_contract_linspace_is_sorted_deposits_is_sorted() {
    use crate::circuit_input::InputPath;
    use crate::optim::predicates::formula::{ContractTerm, ContractVar};
    use crate::optim::prove::ProveOutcome;
    use std::collections::HashMap;

    let mut b = crate::builder::IRBuilder::new();
    let out = b.ir_read_integer(InputPath::new("out", vec![]), false);
    let out_vid = out.value_id().unwrap();

    b.fire_contract("linspace_is_sorted", out_vid, &HashMap::new());

    let query = ContractTerm::PredicateApp {
        kind: "is_sorted".to_string(),
        args: vec![ContractTerm::Var(ContractVar::Value(out_vid))],
    };
    assert_eq!(b.prove(&query), ProveOutcome::Proved);
}

#[test]
fn np_arange_bounded_call_site_fires_is_sorted_on_output() {
    // Integration: exercise `np_arange` end-to-end via the bounded 1-arg
    // form. The result is a `DynamicNDArray`; the call site should fire
    // `arange_is_sorted` on the array's `value_id` (not the length-bearing
    // scalar) — so `prove(is_sorted(out_vid))` returns `Proved`.
    use crate::circuit_input::InputPath;
    use crate::ops::static_ndarray_ops::np_arange;
    use crate::optim::predicates::formula::{
        ArithOp, CmpOp, ContractTerm, ContractVar,
    };
    use crate::optim::prove::ProveOutcome;
    use crate::optim::resolver::LayeredResolver;
    use crate::types::Value;

    let mut b = crate::builder::IRBuilder::new();
    b.set_resolver(Box::new(LayeredResolver::range_then_smt_with_budget(500, 4096)));

    let n = b.ir_read_integer(InputPath::new("n", vec![]), false);
    let n_vid = n.value_id().unwrap();
    let n_plus_n = ContractTerm::Arith {
        op: ArithOp::Add,
        lhs: Box::new(ContractTerm::Var(ContractVar::Value(n_vid))),
        rhs: Box::new(ContractTerm::Var(ContractVar::Value(n_vid))),
    };
    b.facts.insert_for(
        n_vid,
        ContractTerm::Cmp {
            op: CmpOp::Le,
            lhs: Box::new(n_plus_n.clone()),
            rhs: Box::new(ContractTerm::LitInt(20)),
        },
    );
    b.facts.insert_for(
        n_vid,
        ContractTerm::Cmp {
            op: CmpOp::Ge,
            lhs: Box::new(n_plus_n),
            rhs: Box::new(ContractTerm::LitInt(0)),
        },
    );

    let out = np_arange(&mut b, &[n.clone()]);
    let out_vid = match &out {
        Value::DynamicNDArray(d) => d.value_id,
        other => panic!("expected DynamicNDArray, got {:?}", other.zinnia_type()),
    };

    let query = ContractTerm::PredicateApp {
        kind: "is_sorted".to_string(),
        args: vec![ContractTerm::Var(ContractVar::Value(out_vid))],
    };
    assert_eq!(
        b.prove(&query),
        ProveOutcome::Proved,
        "np.arange(n) should fire is_sorted on the dyn-ndarray output",
    );
}

#[test]
fn np_linspace_bounded_call_site_fires_is_sorted_when_ascending() {
    // Integration: exercise `np_linspace` end-to-end via the bounded form
    // with `start <= stop`. The result is a `DynamicNDArray`; the call
    // site should fire `linspace_is_sorted` on the array's `value_id`.
    use crate::circuit_input::InputPath;
    use crate::ops::static_ndarray_ops::np_linspace;
    use crate::optim::predicates::formula::{
        ArithOp, CmpOp, ContractTerm, ContractVar,
    };
    use crate::optim::prove::ProveOutcome;
    use crate::optim::resolver::LayeredResolver;
    use crate::types::Value;
    use std::collections::HashMap;

    let mut b = crate::builder::IRBuilder::new();
    b.set_resolver(Box::new(LayeredResolver::range_then_smt_with_budget(500, 4096)));

    let num = b.ir_read_integer(InputPath::new("num", vec![]), false);
    let num_vid = num.value_id().unwrap();
    // num ∈ [2, 20]: plant `num + num >= 4` and `num + num <= 40` so the
    // outward-doubling probe admits it as bounded. The static
    // soundness-guard inside `np_linspace` requires `num >= 2` for the
    // default `endpoint=True` path.
    let np = ContractTerm::Arith {
        op: ArithOp::Add,
        lhs: Box::new(ContractTerm::Var(ContractVar::Value(num_vid))),
        rhs: Box::new(ContractTerm::Var(ContractVar::Value(num_vid))),
    };
    b.facts.insert_for(
        num_vid,
        ContractTerm::Cmp {
            op: CmpOp::Ge,
            lhs: Box::new(np.clone()),
            rhs: Box::new(ContractTerm::LitInt(4)),
        },
    );
    b.facts.insert_for(
        num_vid,
        ContractTerm::Cmp {
            op: CmpOp::Le,
            lhs: Box::new(np),
            rhs: Box::new(ContractTerm::LitInt(40)),
        },
    );

    let start = b.ir_constant_int(0);
    let stop = b.ir_constant_int(1);
    let kwargs = HashMap::new();
    let out = np_linspace(&mut b, &[start, stop, num.clone()], &kwargs);
    let out_vid = match &out {
        Value::DynamicNDArray(d) => d.value_id,
        other => panic!("expected DynamicNDArray, got {:?}", other.zinnia_type()),
    };

    let query = ContractTerm::PredicateApp {
        kind: "is_sorted".to_string(),
        args: vec![ContractTerm::Var(ContractVar::Value(out_vid))],
    };
    assert_eq!(
        b.prove(&query),
        ProveOutcome::Proved,
        "np.linspace(0, 1, num) should fire is_sorted on the dyn-ndarray output",
    );
}

#[test]
fn np_arange_call_site_does_not_emit_for_negative_step() {
    // Soundness: `np.arange(10, 0, -1)` produces a descending sequence.
    // The is_sorted (ascending) fact MUST NOT be deposited. The
    // call-site direction check in `arange_static` gates on `step > 0`,
    // so we exercise that branch and confirm no `is_sorted` fact
    // surfaces anywhere in the FactStack. (We can't anchor on the
    // output's value_id because the fully-static descending path returns
    // a `Value::StaticArray` with no value_id — so we instead scan all
    // deposited facts and assert none match the `is_sorted` predicate
    // shape.)
    use crate::ops::static_ndarray_ops::np_arange;
    use crate::optim::predicates::formula::ContractTerm;

    let mut b = crate::builder::IRBuilder::new();
    let start = b.ir_constant_int(10);
    let stop = b.ir_constant_int(0);
    let step = b.ir_constant_int(-1);
    let _out = np_arange(&mut b, &[start, stop, step]);

    let any_is_sorted = b.facts.per_value.values().any(|bucket| {
        bucket.iter().any(|fact| {
            matches!(
                fact,
                ContractTerm::PredicateApp { kind, .. } if kind == "is_sorted"
            )
        })
    });
    assert!(
        !any_is_sorted,
        "np.arange(10, 0, -1) should NOT deposit an is_sorted fact (descending), \
         but one was found on the FactStack",
    );
}

// ── Group 4c (compiler.op-fact-group-4c-tile-repeat-content-relay) ──

#[test]
fn relay_forall_eq_const_zero_through_tile() {
    // Plant `forall_eq_const(in, 0)` as a fact on a synthetic input vid,
    // call the relay, and assert the output carries the same fact.
    use crate::circuit_input::InputPath;
    use crate::optim::predicates::formula::{ContractTerm, ContractVar};
    use crate::optim::prove::ProveOutcome;
    use crate::optim::resolver::relay_forall_eq_const_from_input;

    let mut b = crate::builder::IRBuilder::new();
    let in_val = b.ir_read_integer(InputPath::new("inp", vec![]), false);
    let out_val = b.ir_read_integer(InputPath::new("out", vec![]), false);
    let in_vid = in_val.value_id().unwrap();
    let out_vid = out_val.value_id().unwrap();

    b.facts.insert_for(
        in_vid,
        ContractTerm::PredicateApp {
            kind: "forall_eq_const".to_string(),
            args: vec![
                ContractTerm::Var(ContractVar::Value(in_vid)),
                ContractTerm::LitInt(0),
            ],
        },
    );

    let fired = relay_forall_eq_const_from_input(&mut b, in_vid, out_vid);
    assert!(fired, "relay should fire when forall_eq_const(in, 0) holds");

    let query = ContractTerm::PredicateApp {
        kind: "forall_eq_const".to_string(),
        args: vec![
            ContractTerm::Var(ContractVar::Value(out_vid)),
            ContractTerm::LitInt(0),
        ],
    };
    assert_eq!(b.prove(&query), ProveOutcome::Proved);
}

#[test]
fn relay_forall_eq_const_one() {
    use crate::circuit_input::InputPath;
    use crate::optim::predicates::formula::{ContractTerm, ContractVar};
    use crate::optim::prove::ProveOutcome;
    use crate::optim::resolver::relay_forall_eq_const_from_input;

    let mut b = crate::builder::IRBuilder::new();
    let in_val = b.ir_read_integer(InputPath::new("inp", vec![]), false);
    let out_val = b.ir_read_integer(InputPath::new("out", vec![]), false);
    let in_vid = in_val.value_id().unwrap();
    let out_vid = out_val.value_id().unwrap();

    b.facts.insert_for(
        in_vid,
        ContractTerm::PredicateApp {
            kind: "forall_eq_const".to_string(),
            args: vec![
                ContractTerm::Var(ContractVar::Value(in_vid)),
                ContractTerm::LitInt(1),
            ],
        },
    );

    let fired = relay_forall_eq_const_from_input(&mut b, in_vid, out_vid);
    assert!(fired, "relay should fire when forall_eq_const(in, 1) holds");

    let query = ContractTerm::PredicateApp {
        kind: "forall_eq_const".to_string(),
        args: vec![
            ContractTerm::Var(ContractVar::Value(out_vid)),
            ContractTerm::LitInt(1),
        ],
    };
    assert_eq!(b.prove(&query), ProveOutcome::Proved);
}

#[test]
fn relay_forall_eq_const_skips_when_no_fact() {
    use crate::circuit_input::InputPath;
    use crate::optim::predicates::formula::{ContractTerm, ContractVar};
    use crate::optim::prove::ProveOutcome;
    use crate::optim::resolver::relay_forall_eq_const_from_input;

    let mut b = crate::builder::IRBuilder::new();
    let in_val = b.ir_read_integer(InputPath::new("inp", vec![]), false);
    let out_val = b.ir_read_integer(InputPath::new("out", vec![]), false);
    let in_vid = in_val.value_id().unwrap();
    let out_vid = out_val.value_id().unwrap();

    let fired = relay_forall_eq_const_from_input(&mut b, in_vid, out_vid);
    assert!(!fired, "relay should not fire when no content fact is visible");

    for k in [0i64, 1i64] {
        let query = ContractTerm::PredicateApp {
            kind: "forall_eq_const".to_string(),
            args: vec![
                ContractTerm::Var(ContractVar::Value(out_vid)),
                ContractTerm::LitInt(k),
            ],
        };
        assert_eq!(b.prove(&query), ProveOutcome::Unknown);
    }
}

#[test]
fn relay_forall_eq_const_zero_wins_when_both_proved() {
    // Pathological case: both `forall_eq_const(in, 0)` and
    // `forall_eq_const(in, 1)` are simultaneously Proved. Semantically this
    // implies the array is empty (no element can equal both 0 and 1), but
    // the relay sees both facts as proven via the per_value fact bucket.
    // Declared order picks `zeros_content`: relay fires `zeros_content`,
    // not `ones_content`. Output carries `forall_eq_const(out, 0)` only;
    // `forall_eq_const(out, 1)` remains Unknown.
    use crate::circuit_input::InputPath;
    use crate::optim::predicates::formula::{ContractTerm, ContractVar};
    use crate::optim::prove::ProveOutcome;
    use crate::optim::resolver::relay_forall_eq_const_from_input;

    let mut b = crate::builder::IRBuilder::new();
    let in_val = b.ir_read_integer(InputPath::new("inp", vec![]), false);
    let out_val = b.ir_read_integer(InputPath::new("out", vec![]), false);
    let in_vid = in_val.value_id().unwrap();
    let out_vid = out_val.value_id().unwrap();

    for k in [0i64, 1i64] {
        b.facts.insert_for(
            in_vid,
            ContractTerm::PredicateApp {
                kind: "forall_eq_const".to_string(),
                args: vec![
                    ContractTerm::Var(ContractVar::Value(in_vid)),
                    ContractTerm::LitInt(k),
                ],
            },
        );
    }

    let fired = relay_forall_eq_const_from_input(&mut b, in_vid, out_vid);
    assert!(fired);

    let zero_query = ContractTerm::PredicateApp {
        kind: "forall_eq_const".to_string(),
        args: vec![
            ContractTerm::Var(ContractVar::Value(out_vid)),
            ContractTerm::LitInt(0),
        ],
    };
    let one_query = ContractTerm::PredicateApp {
        kind: "forall_eq_const".to_string(),
        args: vec![
            ContractTerm::Var(ContractVar::Value(out_vid)),
            ContractTerm::LitInt(1),
        ],
    };
    assert_eq!(b.prove(&zero_query), ProveOutcome::Proved);
    assert_eq!(b.prove(&one_query), ProveOutcome::Unknown);
}

// ── Group 6 (compiler.op-fact-group-6-shape-preserving-relay) ──

#[cfg(test)]
mod group_6_shape_preserving {
    use crate::builder::IRBuilder;
    use crate::optim::predicates::formula::{ContractTerm, ContractVar};
    use crate::optim::prove::ProveOutcome;
    use crate::types::{CompositeData, NumberType, Value, ValueId};
    use std::collections::HashMap;

    /// Build a 2-D StaticArray of integers from row-major literals.
    fn make_2d_int(b: &mut IRBuilder, rows: &[&[i64]]) -> Value {
        let mut flat: Vec<Value> = Vec::new();
        for r in rows {
            for &n in *r {
                flat.push(b.ir_constant_int(n));
            }
        }
        crate::helpers::static_array::build_static_array_from_flat(
            b,
            flat,
            vec![rows.len(), rows[0].len()],
            NumberType::Integer,
        )
    }

    /// Build a 1-D StaticArray of integers.
    fn make_1d_int(b: &mut IRBuilder, vals: &[i64]) -> Value {
        let flat: Vec<Value> = vals.iter().map(|n| b.ir_constant_int(*n)).collect();
        crate::helpers::static_array::build_static_array_from_flat(
            b,
            flat,
            vec![vals.len()],
            NumberType::Integer,
        )
    }

    fn plant_forall_eq_const(b: &mut IRBuilder, vid: ValueId, k: i64) {
        b.facts.insert_for(
            vid,
            ContractTerm::PredicateApp {
                kind: "forall_eq_const".to_string(),
                args: vec![
                    ContractTerm::Var(ContractVar::Value(vid)),
                    ContractTerm::LitInt(k),
                ],
            },
        );
    }

    fn assert_forall_eq_const_proved(b: &mut IRBuilder, vid: ValueId, k: i64) {
        let query = ContractTerm::PredicateApp {
            kind: "forall_eq_const".to_string(),
            args: vec![
                ContractTerm::Var(ContractVar::Value(vid)),
                ContractTerm::LitInt(k),
            ],
        };
        assert_eq!(
            b.prove(&query),
            ProveOutcome::Proved,
            "expected forall_eq_const(out, {}) to prove",
            k
        );
    }

    #[test]
    fn relay_forall_eq_const_through_transpose() {
        let mut b = IRBuilder::new();
        let a = make_2d_int(&mut b, &[&[0, 0, 0], &[0, 0, 0]]);
        let in_vid = a.value_id().unwrap();
        plant_forall_eq_const(&mut b, in_vid, 0);

        let out = crate::helpers::array_ops::transpose(&mut b, &a, &[]);
        let out_vid = out.value_id().expect("transpose output should have value_id");
        assert_forall_eq_const_proved(&mut b, out_vid, 0);
    }

    #[test]
    fn relay_forall_eq_const_through_reshape() {
        let mut b = IRBuilder::new();
        let a = make_1d_int(&mut b, &[1, 1, 1, 1, 1, 1]);
        let in_vid = a.value_id().unwrap();
        plant_forall_eq_const(&mut b, in_vid, 1);

        let two = b.ir_constant_int(2);
        let three = b.ir_constant_int(3);
        let out = crate::helpers::array_ops::reshape(&mut b, &a, &[two, three]);
        let out_vid = out.value_id().expect("reshape output should have value_id");
        assert_forall_eq_const_proved(&mut b, out_vid, 1);
    }

    #[test]
    fn relay_forall_eq_const_through_squeeze() {
        let mut b = IRBuilder::new();
        // Shape (1, 3) — squeezable to (3,).
        let a_2d = make_2d_int(&mut b, &[&[0, 0, 0]]);
        let in_vid = a_2d.value_id().unwrap();
        plant_forall_eq_const(&mut b, in_vid, 0);

        let kwargs = HashMap::new();
        let out = crate::ops::static_ndarray_ops::np_squeeze(&mut b, &[a_2d], &kwargs);
        let out_vid = out.value_id().expect("squeeze output should have value_id");
        assert_forall_eq_const_proved(&mut b, out_vid, 0);
    }

    #[test]
    fn relay_forall_eq_const_through_expand_dims() {
        let mut b = IRBuilder::new();
        let a = make_1d_int(&mut b, &[0, 0, 0]);
        let in_vid = a.value_id().unwrap();
        plant_forall_eq_const(&mut b, in_vid, 0);

        let axis = b.ir_constant_int(0);
        let out = crate::ops::static_ndarray_ops::np_expand_dims(&mut b, &[a, axis]);
        let out_vid = out.value_id().expect("expand_dims output should have value_id");
        assert_forall_eq_const_proved(&mut b, out_vid, 0);
    }

    #[test]
    fn relay_forall_eq_const_through_broadcast_to() {
        let mut b = IRBuilder::new();
        // `np_broadcast_to` walks the legacy `flatten_composite` path which
        // expects a nested `Value::List` of leaves, so build one directly
        // rather than starting from a StaticArray.
        let leaves: Vec<Value> = (0..3).map(|_| b.ir_constant_int(1)).collect();
        let a = Value::List(CompositeData {
            elements_type: leaves.iter().map(|v| v.zinnia_type()).collect(),
            values: leaves,
            value_id: ValueId::next(),
        });
        let in_vid = a.value_id().unwrap();
        plant_forall_eq_const(&mut b, in_vid, 1);

        // Broadcast (3,) → (2, 3).
        let two = b.ir_constant_int(2);
        let three = b.ir_constant_int(3);
        let shape = Value::Tuple(CompositeData {
            elements_type: vec![crate::types::ZinniaType::Integer; 2],
            values: vec![two, three],
            value_id: ValueId::next(),
        });
        let out = crate::ops::static_ndarray_ops::np_broadcast_to(&mut b, &[a, shape]);
        let out_vid = out.value_id().expect("broadcast_to output should have value_id");
        assert_forall_eq_const_proved(&mut b, out_vid, 1);
    }

    #[test]
    fn relay_forall_eq_const_through_transpose_no_fact_no_relay() {
        let mut b = IRBuilder::new();
        let a = make_2d_int(&mut b, &[&[1, 2, 3], &[4, 5, 6]]);
        // No fact planted on `a`.
        let out = crate::helpers::array_ops::transpose(&mut b, &a, &[]);
        let out_vid = out.value_id().expect("transpose output should have value_id");
        for k in [0i64, 1i64] {
            let query = ContractTerm::PredicateApp {
                kind: "forall_eq_const".to_string(),
                args: vec![
                    ContractTerm::Var(ContractVar::Value(out_vid)),
                    ContractTerm::LitInt(k),
                ],
            };
            assert_eq!(
                b.prove(&query),
                ProveOutcome::Unknown,
                "no source fact ⇒ output should remain Unknown for k={}",
                k
            );
        }
    }
}

// ── Group 7 (compiler.op-fact-group-7-concat-stack-relay) ──

#[test]
fn relay_concat_all_zeros_emits_zeros() {
    use crate::circuit_input::InputPath;
    use crate::optim::predicates::formula::{ContractTerm, ContractVar};
    use crate::optim::prove::ProveOutcome;
    use crate::optim::resolver::relay_forall_eq_const_from_all_inputs;

    let mut b = crate::builder::IRBuilder::new();
    let in1 = b.ir_read_integer(InputPath::new("in1", vec![]), false);
    let in2 = b.ir_read_integer(InputPath::new("in2", vec![]), false);
    let out_val = b.ir_read_integer(InputPath::new("out", vec![]), false);
    let in1_vid = in1.value_id().unwrap();
    let in2_vid = in2.value_id().unwrap();
    let out_vid = out_val.value_id().unwrap();

    for vid in [in1_vid, in2_vid] {
        b.facts.insert_for(
            vid,
            ContractTerm::PredicateApp {
                kind: "forall_eq_const".to_string(),
                args: vec![
                    ContractTerm::Var(ContractVar::Value(vid)),
                    ContractTerm::LitInt(0),
                ],
            },
        );
    }

    let fired = relay_forall_eq_const_from_all_inputs(&mut b, &[in1_vid, in2_vid], out_vid);
    assert!(fired);

    let query = ContractTerm::PredicateApp {
        kind: "forall_eq_const".to_string(),
        args: vec![
            ContractTerm::Var(ContractVar::Value(out_vid)),
            ContractTerm::LitInt(0),
        ],
    };
    assert_eq!(b.prove(&query), ProveOutcome::Proved);
}

#[test]
fn relay_concat_all_ones_emits_ones() {
    use crate::circuit_input::InputPath;
    use crate::optim::predicates::formula::{ContractTerm, ContractVar};
    use crate::optim::prove::ProveOutcome;
    use crate::optim::resolver::relay_forall_eq_const_from_all_inputs;

    let mut b = crate::builder::IRBuilder::new();
    let in1 = b.ir_read_integer(InputPath::new("in1", vec![]), false);
    let in2 = b.ir_read_integer(InputPath::new("in2", vec![]), false);
    let in3 = b.ir_read_integer(InputPath::new("in3", vec![]), false);
    let out_val = b.ir_read_integer(InputPath::new("out", vec![]), false);
    let in1_vid = in1.value_id().unwrap();
    let in2_vid = in2.value_id().unwrap();
    let in3_vid = in3.value_id().unwrap();
    let out_vid = out_val.value_id().unwrap();

    for vid in [in1_vid, in2_vid, in3_vid] {
        b.facts.insert_for(
            vid,
            ContractTerm::PredicateApp {
                kind: "forall_eq_const".to_string(),
                args: vec![
                    ContractTerm::Var(ContractVar::Value(vid)),
                    ContractTerm::LitInt(1),
                ],
            },
        );
    }

    let fired =
        relay_forall_eq_const_from_all_inputs(&mut b, &[in1_vid, in2_vid, in3_vid], out_vid);
    assert!(fired);

    let query = ContractTerm::PredicateApp {
        kind: "forall_eq_const".to_string(),
        args: vec![
            ContractTerm::Var(ContractVar::Value(out_vid)),
            ContractTerm::LitInt(1),
        ],
    };
    assert_eq!(b.prove(&query), ProveOutcome::Proved);
}

#[test]
fn relay_concat_mixed_zero_one_emits_nothing() {
    use crate::circuit_input::InputPath;
    use crate::optim::predicates::formula::{ContractTerm, ContractVar};
    use crate::optim::prove::ProveOutcome;
    use crate::optim::resolver::relay_forall_eq_const_from_all_inputs;

    let mut b = crate::builder::IRBuilder::new();
    let in1 = b.ir_read_integer(InputPath::new("in1", vec![]), false);
    let in2 = b.ir_read_integer(InputPath::new("in2", vec![]), false);
    let out_val = b.ir_read_integer(InputPath::new("out", vec![]), false);
    let in1_vid = in1.value_id().unwrap();
    let in2_vid = in2.value_id().unwrap();
    let out_vid = out_val.value_id().unwrap();

    // in1 is zeros, in2 is ones — no single k satisfies both.
    b.facts.insert_for(
        in1_vid,
        ContractTerm::PredicateApp {
            kind: "forall_eq_const".to_string(),
            args: vec![
                ContractTerm::Var(ContractVar::Value(in1_vid)),
                ContractTerm::LitInt(0),
            ],
        },
    );
    b.facts.insert_for(
        in2_vid,
        ContractTerm::PredicateApp {
            kind: "forall_eq_const".to_string(),
            args: vec![
                ContractTerm::Var(ContractVar::Value(in2_vid)),
                ContractTerm::LitInt(1),
            ],
        },
    );

    let fired = relay_forall_eq_const_from_all_inputs(&mut b, &[in1_vid, in2_vid], out_vid);
    assert!(!fired);

    for k in [0i64, 1i64] {
        let query = ContractTerm::PredicateApp {
            kind: "forall_eq_const".to_string(),
            args: vec![
                ContractTerm::Var(ContractVar::Value(out_vid)),
                ContractTerm::LitInt(k),
            ],
        };
        assert_eq!(b.prove(&query), ProveOutcome::Unknown);
    }
}

#[test]
fn relay_concat_one_input_no_fact_skips() {
    use crate::circuit_input::InputPath;
    use crate::optim::predicates::formula::{ContractTerm, ContractVar};
    use crate::optim::prove::ProveOutcome;
    use crate::optim::resolver::relay_forall_eq_const_from_all_inputs;

    let mut b = crate::builder::IRBuilder::new();
    let in1 = b.ir_read_integer(InputPath::new("in1", vec![]), false);
    let in2 = b.ir_read_integer(InputPath::new("in2", vec![]), false);
    let out_val = b.ir_read_integer(InputPath::new("out", vec![]), false);
    let in1_vid = in1.value_id().unwrap();
    let in2_vid = in2.value_id().unwrap();
    let out_vid = out_val.value_id().unwrap();

    // Only in1 has a fact; in2 has none.
    b.facts.insert_for(
        in1_vid,
        ContractTerm::PredicateApp {
            kind: "forall_eq_const".to_string(),
            args: vec![
                ContractTerm::Var(ContractVar::Value(in1_vid)),
                ContractTerm::LitInt(0),
            ],
        },
    );

    let fired = relay_forall_eq_const_from_all_inputs(&mut b, &[in1_vid, in2_vid], out_vid);
    assert!(!fired);

    for k in [0i64, 1i64] {
        let query = ContractTerm::PredicateApp {
            kind: "forall_eq_const".to_string(),
            args: vec![
                ContractTerm::Var(ContractVar::Value(out_vid)),
                ContractTerm::LitInt(k),
            ],
        };
        assert_eq!(b.prove(&query), ProveOutcome::Unknown);
    }
}

#[test]
fn relay_concat_empty_input_list_returns_false() {
    use crate::circuit_input::InputPath;
    use crate::optim::resolver::relay_forall_eq_const_from_all_inputs;

    let mut b = crate::builder::IRBuilder::new();
    let out_val = b.ir_read_integer(InputPath::new("out", vec![]), false);
    let out_vid = out_val.value_id().unwrap();

    let fired = relay_forall_eq_const_from_all_inputs(&mut b, &[], out_vid);
    assert!(!fired);
}

// ── Group 3e (compiler.op-fact-group-3e-var-std) ─────────────────────
//
// `var` and `std` carry a multi-formal requires `len_arr >= 1` and a
// single ensures `Output >= 0.0`. Plant a fact on the `len_arr` formal,
// fire, and check both the proved-discharge and disproved-discharge
// branches behave as expected.

#[test]
fn fire_contract_var_proved_when_len_ge_1() {
    use crate::circuit_input::InputPath;
    use crate::optim::predicates::formula::{CmpOp, ContractFloat, ContractTerm, ContractVar};
    use std::collections::HashMap;

    let mut b = crate::builder::IRBuilder::new();
    let out = b.ir_read_float(InputPath::new("out", vec![]), false);
    let out_vid = out.value_id().unwrap();
    let len = b.ir_read_integer(InputPath::new("len", vec![]), false);
    let len_vid = len.value_id().unwrap();
    b.facts.insert_for(
        len_vid,
        ContractTerm::Cmp {
            op: CmpOp::Ge,
            lhs: Box::new(ContractTerm::Var(ContractVar::Value(len_vid))),
            rhs: Box::new(ContractTerm::LitInt(1)),
        },
    );

    let mut formals = HashMap::new();
    formals.insert("len_arr".to_string(), len_vid);
    b.fire_contract("var", out_vid, &formals);

    let facts = b
        .facts
        .per_value
        .get(&out_vid)
        .expect("fire_contract(var) should have deposited at least one ensures fact");
    let expected = ContractTerm::Cmp {
        op: CmpOp::Ge,
        lhs: Box::new(ContractTerm::Var(ContractVar::Value(out_vid))),
        rhs: Box::new(ContractTerm::LitFloat(ContractFloat(0.0))),
    };
    assert!(
        facts.iter().any(|f| *f == expected),
        "expected Output >= 0.0 ensures fact after fire_contract(var); got {:?}",
        facts,
    );
}

#[test]
#[should_panic(expected = "requires precondition disproved")]
fn fire_contract_var_disproved_when_len_eq_0() {
    use crate::circuit_input::InputPath;
    use crate::optim::predicates::formula::{CmpOp, ContractTerm, ContractVar};
    use std::collections::HashMap;

    let mut b = crate::builder::IRBuilder::new();
    let out = b.ir_read_float(InputPath::new("out", vec![]), false);
    let out_vid = out.value_id().unwrap();
    let len = b.ir_read_integer(InputPath::new("len", vec![]), false);
    let len_vid = len.value_id().unwrap();
    // Plant `len <= 0`; refutes `len >= 1`.
    b.facts.insert_for(
        len_vid,
        ContractTerm::Cmp {
            op: CmpOp::Le,
            lhs: Box::new(ContractTerm::Var(ContractVar::Value(len_vid))),
            rhs: Box::new(ContractTerm::LitInt(0)),
        },
    );

    let mut formals = HashMap::new();
    formals.insert("len_arr".to_string(), len_vid);
    b.fire_contract("var", out_vid, &formals);
}

#[test]
fn fire_contract_std_emits_output_nonneg() {
    use crate::circuit_input::InputPath;
    use crate::optim::predicates::formula::{CmpOp, ContractFloat, ContractTerm, ContractVar};
    use std::collections::HashMap;

    let mut b = crate::builder::IRBuilder::new();
    let out = b.ir_read_float(InputPath::new("out", vec![]), false);
    let out_vid = out.value_id().unwrap();
    let len = b.ir_read_integer(InputPath::new("len", vec![]), false);
    let len_vid = len.value_id().unwrap();
    b.facts.insert_for(
        len_vid,
        ContractTerm::Cmp {
            op: CmpOp::Ge,
            lhs: Box::new(ContractTerm::Var(ContractVar::Value(len_vid))),
            rhs: Box::new(ContractTerm::LitInt(1)),
        },
    );

    let mut formals = HashMap::new();
    formals.insert("len_arr".to_string(), len_vid);
    b.fire_contract("std", out_vid, &formals);

    let facts = b
        .facts
        .per_value
        .get(&out_vid)
        .expect("fire_contract(std) should have deposited at least one ensures fact");
    let expected = ContractTerm::Cmp {
        op: CmpOp::Ge,
        lhs: Box::new(ContractTerm::Var(ContractVar::Value(out_vid))),
        rhs: Box::new(ContractTerm::LitFloat(ContractFloat(0.0))),
    };
    assert!(
        facts.iter().any(|f| *f == expected),
        "expected Output >= 0.0 ensures fact after fire_contract(std); got {:?}",
        facts,
    );
}

// ── Group 5b (compiler.op-fact-group-5b-slice-content-relay) ──

#[cfg(test)]
mod group_5b_slice_content {
    use crate::builder::IRBuilder;
    use crate::optim::predicates::formula::{ContractTerm, ContractVar};
    use crate::optim::prove::ProveOutcome;
    use crate::types::{NumberType, SliceIndex, Value, ValueId};

    /// Build a 1-D StaticArray of integers.
    fn make_1d_int(b: &mut IRBuilder, vals: &[i64]) -> Value {
        let flat: Vec<Value> = vals.iter().map(|n| b.ir_constant_int(*n)).collect();
        crate::helpers::static_array::build_static_array_from_flat(
            b,
            flat,
            vec![vals.len()],
            NumberType::Integer,
        )
    }

    fn plant_forall_eq_const(b: &mut IRBuilder, vid: ValueId, k: i64) {
        b.facts.insert_for(
            vid,
            ContractTerm::PredicateApp {
                kind: "forall_eq_const".to_string(),
                args: vec![
                    ContractTerm::Var(ContractVar::Value(vid)),
                    ContractTerm::LitInt(k),
                ],
            },
        );
    }

    fn assert_forall_eq_const_proved(b: &mut IRBuilder, vid: ValueId, k: i64) {
        let query = ContractTerm::PredicateApp {
            kind: "forall_eq_const".to_string(),
            args: vec![
                ContractTerm::Var(ContractVar::Value(vid)),
                ContractTerm::LitInt(k),
            ],
        };
        assert_eq!(
            b.prove(&query),
            ProveOutcome::Proved,
            "expected forall_eq_const(out, {}) to prove",
            k
        );
    }

    /// Build a static-bound 1-D range slice `[start:stop]`.
    fn static_range(b: &mut IRBuilder, start: i64, stop: i64) -> SliceIndex {
        SliceIndex::Range(
            Some(b.ir_constant_int(start)),
            Some(b.ir_constant_int(stop)),
            None,
        )
    }

    #[test]
    fn relay_forall_eq_const_zero_through_slice() {
        let mut b = IRBuilder::new();
        let a = make_1d_int(&mut b, &[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        let in_vid = a.value_id().unwrap();
        plant_forall_eq_const(&mut b, in_vid, 0);

        let idx = static_range(&mut b, 2, 5);
        let out = crate::helpers::static_array_read::static_array_subscript(&mut b, &a, &[idx]);
        let out_vid = out.value_id().expect("slice output should have value_id");
        assert_forall_eq_const_proved(&mut b, out_vid, 0);
    }

    #[test]
    fn relay_forall_eq_const_one_through_slice() {
        let mut b = IRBuilder::new();
        let a = make_1d_int(&mut b, &[1, 1, 1, 1, 1, 1]);
        let in_vid = a.value_id().unwrap();
        plant_forall_eq_const(&mut b, in_vid, 1);

        let idx = static_range(&mut b, 1, 4);
        let out = crate::helpers::static_array_read::static_array_subscript(&mut b, &a, &[idx]);
        let out_vid = out.value_id().expect("slice output should have value_id");
        assert_forall_eq_const_proved(&mut b, out_vid, 1);
    }

    #[test]
    fn relay_forall_eq_const_skips_on_unbounded_slice() {
        // No fact planted on the source — slice output must remain Unknown
        // for both k=0 and k=1.
        let mut b = IRBuilder::new();
        let a = make_1d_int(&mut b, &[1, 2, 3, 4, 5, 6]);
        let idx = static_range(&mut b, 1, 4);
        let out = crate::helpers::static_array_read::static_array_subscript(&mut b, &a, &[idx]);
        let out_vid = out.value_id().expect("slice output should have value_id");
        for k in [0i64, 1i64] {
            let query = ContractTerm::PredicateApp {
                kind: "forall_eq_const".to_string(),
                args: vec![
                    ContractTerm::Var(ContractVar::Value(out_vid)),
                    ContractTerm::LitInt(k),
                ],
            };
            assert_eq!(
                b.prove(&query),
                ProveOutcome::Unknown,
                "no source fact ⇒ slice output should remain Unknown for k={}",
                k
            );
        }
    }
}

// ── Group 5c (compiler.op-fact-group-5c-gather-content-relay) ──

#[cfg(test)]
mod group_5c_gather_content {
    use crate::builder::IRBuilder;
    use crate::optim::predicates::formula::{ContractTerm, ContractVar};
    use crate::optim::prove::ProveOutcome;
    use crate::types::{CompositeData, Value, ValueId};

    /// Build a `Value::List` of integer leaves with a fresh `value_id`.
    fn make_1d_list_int(b: &mut IRBuilder, vals: &[i64]) -> Value {
        let values: Vec<Value> = vals.iter().map(|n| b.ir_constant_int(*n)).collect();
        let elements_type = values.iter().map(|v| v.zinnia_type()).collect();
        Value::List(CompositeData { elements_type, values, value_id: ValueId::next() })
    }

    /// Build a Python-list-like index array `[i, j, k]` of integer literals.
    fn make_index_list(b: &mut IRBuilder, idxs: &[i64]) -> Value {
        let values: Vec<Value> = idxs.iter().map(|n| b.ir_constant_int(*n)).collect();
        let elements_type = values.iter().map(|v| v.zinnia_type()).collect();
        Value::List(CompositeData { elements_type, values, value_id: ValueId::next() })
    }

    fn plant_forall_eq_const(b: &mut IRBuilder, vid: ValueId, k: i64) {
        b.facts.insert_for(
            vid,
            ContractTerm::PredicateApp {
                kind: "forall_eq_const".to_string(),
                args: vec![
                    ContractTerm::Var(ContractVar::Value(vid)),
                    ContractTerm::LitInt(k),
                ],
            },
        );
    }

    fn assert_forall_eq_const_proved(b: &mut IRBuilder, vid: ValueId, k: i64) {
        let query = ContractTerm::PredicateApp {
            kind: "forall_eq_const".to_string(),
            args: vec![
                ContractTerm::Var(ContractVar::Value(vid)),
                ContractTerm::LitInt(k),
            ],
        };
        assert_eq!(
            b.prove(&query),
            ProveOutcome::Proved,
            "expected forall_eq_const(out, {}) to prove",
            k
        );
    }

    #[test]
    fn relay_forall_eq_const_zero_through_fancy_index_static() {
        // Plant `forall_eq_const(in, 0)` on a List input, fancy-index it, and
        // fire the gather-site relay (mirroring the visitors.rs wiring).
        let mut b = IRBuilder::new();
        let arr = make_1d_list_int(&mut b, &[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        let in_vid = arr.value_id().unwrap();
        plant_forall_eq_const(&mut b, in_vid, 0);

        let idx = make_index_list(&mut b, &[2, 5, 8]);
        let data = match &arr {
            Value::List(d) => d.clone(),
            _ => unreachable!(),
        };
        let out = crate::helpers::ndarray::fancy_index_static(&data, &idx)
            .expect("fancy_index_static should succeed on int literals");
        let out_vid = out.value_id().expect("fancy_index_static output should have a value_id");

        let fired = crate::optim::resolver::relay_forall_eq_const_from_input(
            &mut b, in_vid, out_vid,
        );
        assert!(fired, "relay should fire when forall_eq_const(in, 0) holds");
        assert_forall_eq_const_proved(&mut b, out_vid, 0);
    }

    #[test]
    fn relay_forall_eq_const_one_through_take() {
        // Same shape as the fancy-index case but driving the relay via the
        // np.take dispatch (which lowers to fancy_index_static under the hood).
        let mut b = IRBuilder::new();
        let arr = make_1d_list_int(&mut b, &[1, 1, 1, 1, 1, 1]);
        let in_vid = arr.value_id().unwrap();
        plant_forall_eq_const(&mut b, in_vid, 1);

        let idx = make_index_list(&mut b, &[0, 2, 4]);
        let data = match &arr {
            Value::List(d) => d.clone(),
            _ => unreachable!(),
        };
        let out = crate::helpers::ndarray::fancy_index_static(&data, &idx)
            .expect("fancy_index_static should succeed on int literals");
        let out_vid = out.value_id().expect("output should have a value_id");

        let fired = crate::optim::resolver::relay_forall_eq_const_from_input(
            &mut b, in_vid, out_vid,
        );
        assert!(fired, "relay should fire when forall_eq_const(in, 1) holds");
        assert_forall_eq_const_proved(&mut b, out_vid, 1);
    }

    #[test]
    fn relay_forall_eq_const_skips_when_no_fact() {
        // No fact planted on the source — gather output must remain Unknown
        // for both k=0 and k=1 even after invoking the relay.
        let mut b = IRBuilder::new();
        let arr = make_1d_list_int(&mut b, &[1, 2, 3, 4, 5, 6]);
        let in_vid = arr.value_id().unwrap();
        let idx = make_index_list(&mut b, &[0, 2, 4]);
        let data = match &arr {
            Value::List(d) => d.clone(),
            _ => unreachable!(),
        };
        let out = crate::helpers::ndarray::fancy_index_static(&data, &idx)
            .expect("fancy_index_static should succeed on int literals");
        let out_vid = out.value_id().expect("output should have a value_id");

        let fired = crate::optim::resolver::relay_forall_eq_const_from_input(
            &mut b, in_vid, out_vid,
        );
        assert!(!fired, "relay should NOT fire without a source fact");
        for k in [0i64, 1i64] {
            let query = ContractTerm::PredicateApp {
                kind: "forall_eq_const".to_string(),
                args: vec![
                    ContractTerm::Var(ContractVar::Value(out_vid)),
                    ContractTerm::LitInt(k),
                ],
            };
            assert_eq!(
                b.prove(&query),
                ProveOutcome::Unknown,
                "no source fact ⇒ gather output should remain Unknown for k={}",
                k
            );
        }
    }
}
