// Note: This file is included as `#[cfg(test)] mod tests;` from mod.rs,
// so it's already test-only. No inner `mod tests` wrapper needed.

use std::sync::atomic::Ordering;

use super::*;
use crate::ir_defs::IR;
use crate::optim::predicates::formula::ContractFloat;
use crate::optim::predicates::{CmpOp, ContractTerm, ContractVar};
use crate::optim::prove::{prove_with_facts, ProveOutcome};
use crate::types::ValueId;
use super::input_domain::is_domain_check_stmt as is_input_check;

/// Whether `stmts[index]` is the backstop of a `while` unroll
/// (`assert_while_exhausted`): `assert(select(running, not guard, true))`.
fn is_while_backstop(stmts: &[crate::ir::IRStatement], index: usize) -> bool {
    let stmt = &stmts[index];
    let IR::Assert = stmt.ir else { return false };
    let mut cond = &stmts[stmt.arguments[0] as usize];
    if matches!(cond.ir, IR::SelectI) {
        cond = &stmts[cond.arguments[1] as usize];
    }
    matches!(cond.ir, IR::LogicalNot)
}

#[test]
fn test_generate_simple_constants() {
    let json = r#"{
            "__class__": "ASTCircuit",
            "block": [
                {
                    "__class__": "ASTAssignStatement",
                    "targets": [{"__class__": "ASTNameAssignTarget", "name": "x"}],
                    "value": {"__class__": "ASTConstantInteger", "value": 42}
                }
            ],
            "inputs": []
        }"#;

    let graph = IRGenerator::generate_from_json(IRGenConfig::default(), json).unwrap();
    assert!(!graph.is_empty()); // At least the constant + type registrations
}

#[test]
fn symbolic_shape_descriptor_fails_before_ir_generation() {
    let json = r#"{
            "__class__": "ASTCircuit",
            "inputs": [
                {
                    "__class__": "ASTCircuitInput",
                    "name": "x",
                    "annotation": {
                        "__class__": "ASTAnnotation",
                        "kind": "Private",
                        "dt": {"NDArray": {"shape": ["N", 2], "dtype": "Float"}}
                    }
                }
            ],
            "block": []
        }"#;

    let err = std::panic::catch_unwind(|| {
        let _ = IRGenerator::generate_from_json(IRGenConfig::default(), json);
    });
    assert!(
        err.is_err(),
        "symbolic shape descriptors must not silently reach IR generation"
    );
}

#[test]
fn bounded_dynamic_input_metadata_uses_runtime_shape_witness() {
    let json = r#"{
            "__class__": "ASTCircuit",
            "inputs": [
                {
                    "__class__": "ASTCircuitInput",
                    "name": "x",
                    "annotation": {
                        "__class__": "ASTAnnotation",
                        "kind": "Private",
                        "dt": {"DynamicNDArray": {"dtype": "Float", "max_length": 4, "max_rank": 1}}
                    }
                }
            ],
            "block": []
        }"#;

    let node: ASTNode = serde_json::from_str(json).unwrap();
    let ASTNode::ASTCircuit(circuit) = node else {
        panic!("expected circuit");
    };
    let mut gen = IRGenerator::new(IRGenConfig::default());
    gen.visit_circuit(&circuit);
    let x = gen.ctx.get("x").expect("x should be bound");
    let Value::DynamicNDArray(data) = x else {
        panic!("expected DynamicNDArray input");
    };
    assert_eq!(data.envelope.total_bound, 4);
    assert_eq!(
        data.envelope.dims.iter().map(|d| d.max).collect::<Vec<_>>(),
        vec![4]
    );
    assert_eq!(data.meta.logical_shape, vec![4]);
    assert_eq!(data.meta.runtime_length.static_val, None);
    assert!(data.meta.runtime_length.stmt_id.is_some());
    assert_eq!(data.meta.runtime_rank.static_val, Some(1));
    assert_eq!(data.meta.runtime_shape[0].static_val, None);
    assert!(data.meta.runtime_shape[0].stmt_id.is_some());
    assert!(
        gen.builder.stmts.iter().any(|stmt| matches!(
            stmt.ir,
            IR::WitnessDynamicNDArrayMeta {
                array_id: _,
                max_rank: 1,
                ref param_name,
            } if param_name == "x"
        )),
        "dynamic input lowering should tie metadata to the named witness parameter"
    );
}

#[test]
fn dynamic_input_metadata_preserves_per_axis_bounds() {
    let json = r#"{
            "__class__": "ASTCircuit",
            "inputs": [
                {
                    "__class__": "ASTCircuitInput",
                    "name": "x",
                    "annotation": {
                        "__class__": "ASTAnnotation",
                        "kind": "Private",
                        "dt": {"DynamicNDArray": {"dtype": "Float", "max_length": 12, "max_rank": 2, "max_shape": [4, 3]}}
                    }
                }
            ],
            "block": []
        }"#;

    let node: ASTNode = serde_json::from_str(json).unwrap();
    let ASTNode::ASTCircuit(circuit) = node else {
        panic!("expected circuit");
    };
    let mut gen = IRGenerator::new(IRGenConfig::default());
    gen.visit_circuit(&circuit);
    let x = gen.ctx.get("x").expect("x should be bound");
    let Value::DynamicNDArray(data) = x else {
        panic!("expected DynamicNDArray input");
    };
    assert_eq!(data.envelope.total_bound, 12);
    assert_eq!(
        data.envelope.dims.iter().map(|d| d.max).collect::<Vec<_>>(),
        vec![4, 3]
    );
    assert_eq!(data.meta.logical_shape, vec![4, 3]);
    assert!(
        gen.builder.stmts.iter().any(|stmt| matches!(
            stmt.ir,
            IR::AllocateDynamicNDArrayMeta {
                array_id: _,
                max_length: 12,
                max_rank: 2,
                ref max_shape,
                ..
            } if max_shape == &vec![4, 3]
        )),
        "dynamic input allocation should carry per-axis max_shape"
    );
}

#[test]
fn test_generate_binary_op() {
    let json = r#"{
            "__class__": "ASTCircuit",
            "block": [
                {
                    "__class__": "ASTAssignStatement",
                    "targets": [{"__class__": "ASTNameAssignTarget", "name": "z"}],
                    "value": {
                        "__class__": "ASTBinaryOperator",
                        "operator": "add",
                        "lhs": {"__class__": "ASTConstantInteger", "value": 10},
                        "rhs": {"__class__": "ASTConstantInteger", "value": 20}
                    }
                }
            ],
            "inputs": []
        }"#;

    let graph = IRGenerator::generate_from_json(IRGenConfig::default(), json).unwrap();
    // Should have: constant(10), constant(20), add_i
    assert!(graph.len() >= 3);
}

#[test]
fn test_generate_if_else() {
    let json = r#"{
            "__class__": "ASTCircuit",
            "block": [
                {
                    "__class__": "ASTAssignStatement",
                    "targets": [{"__class__": "ASTNameAssignTarget", "name": "cond"}],
                    "value": {"__class__": "ASTConstantBoolean", "value": true}
                },
                {
                    "__class__": "ASTCondStatement",
                    "cond": {"__class__": "ASTLoad", "name": "cond"},
                    "t_block": [
                        {
                            "__class__": "ASTAssignStatement",
                            "targets": [{"__class__": "ASTNameAssignTarget", "name": "x"}],
                            "value": {"__class__": "ASTConstantInteger", "value": 1}
                        }
                    ],
                    "f_block": [
                        {
                            "__class__": "ASTAssignStatement",
                            "targets": [{"__class__": "ASTNameAssignTarget", "name": "x"}],
                            "value": {"__class__": "ASTConstantInteger", "value": 2}
                        }
                    ]
                }
            ],
            "inputs": []
        }"#;

    let graph = IRGenerator::generate_from_json(IRGenConfig::default(), json).unwrap();
    assert!(graph.len() >= 3);
}

#[test]
fn statement_branch_condition_emits_guarded_path_facts_for_both_arms() {
    let json = r#"{
            "__class__": "ASTCircuit",
            "inputs": [
                {
                    "__class__": "ASTCircuitInput",
                    "name": "x",
                    "annotation": {
                        "__class__": "ASTAnnotation",
                        "kind": "Private",
                        "dt": "Integer"
                    }
                }
            ],
            "block": [
                {
                    "__class__": "ASTCondStatement",
                    "cond": {
                        "__class__": "ASTBinaryOperator",
                        "operator": "gt",
                        "lhs": {"__class__": "ASTLoad", "name": "x"},
                        "rhs": {"__class__": "ASTConstantInteger", "value": 0}
                    },
                    "t_block": [
                        {
                            "__class__": "ASTAssignStatement",
                            "targets": [{"__class__": "ASTNameAssignTarget", "name": "y"}],
                            "value": {
                                "__class__": "ASTBinaryOperator",
                                "operator": "eq",
                                "lhs": {"__class__": "ASTLoad", "name": "x"},
                                "rhs": {"__class__": "ASTConstantInteger", "value": 1}
                            }
                        }
                    ],
                    "f_block": [
                        {
                            "__class__": "ASTAssignStatement",
                            "targets": [{"__class__": "ASTNameAssignTarget", "name": "y"}],
                            "value": {
                                "__class__": "ASTBinaryOperator",
                                "operator": "ne",
                                "lhs": {"__class__": "ASTLoad", "name": "x"},
                                "rhs": {"__class__": "ASTConstantInteger", "value": 1}
                            }
                        }
                    ]
                }
            ]
        }"#;

    let node: ASTNode = serde_json::from_str(json).unwrap();
    let ASTNode::ASTCircuit(circuit) = node else {
        panic!("expected circuit");
    };
    let mut gen = IRGenerator::new(IRGenConfig::default());
    gen.visit_circuit(&circuit);

    let cond_stmt = gen
        .builder
        .stmts
        .iter()
        .find(|stmt| matches!(stmt.ir, IR::GtI))
        .expect("condition comparison should emit GtI");
    let true_cond_vid = cond_stmt.value_id;

    let then_path = crate::optim::predicates::ContractTerm::Cmp {
        op: crate::optim::predicates::CmpOp::Eq,
        lhs: Box::new(crate::optim::predicates::ContractTerm::Var(
            crate::optim::predicates::ContractVar::Value(true_cond_vid),
        )),
        rhs: Box::new(crate::optim::predicates::ContractTerm::LitInt(1)),
    };
    let else_path = crate::optim::predicates::ContractTerm::Cmp {
        op: crate::optim::predicates::CmpOp::Eq,
        lhs: Box::new(crate::optim::predicates::ContractTerm::Var(
            crate::optim::predicates::ContractVar::Value(true_cond_vid),
        )),
        rhs: Box::new(crate::optim::predicates::ContractTerm::LitInt(0)),
    };

    let joined_facts: Vec<_> = gen.builder.facts.visible_facts();
    assert!(
        joined_facts.iter().any(|fact| match fact {
            crate::optim::predicates::ContractTerm::BoolComb {
                op: crate::optim::predicates::BoolOp::Or,
                operands,
            } if operands.len() == 2 => {
                matches!(&operands[0], crate::optim::predicates::ContractTerm::Not(inner) if inner.as_ref() == &then_path)
            }
            _ => false,
        }),
        "then-arm facts should be guarded by cond == 1; got {:?}",
        joined_facts
    );
    assert!(
        joined_facts.iter().any(|fact| match fact {
            crate::optim::predicates::ContractTerm::BoolComb {
                op: crate::optim::predicates::BoolOp::Or,
                operands,
            } if operands.len() == 2 => {
                matches!(&operands[0], crate::optim::predicates::ContractTerm::Not(inner) if inner.as_ref() == &else_path)
            }
            _ => false,
        }),
        "else-arm facts should be guarded by cond == 0; got {:?}",
        joined_facts
    );
}

#[test]
fn to_scalar_bool_keeps_boolean_values_without_redundant_bool_cast() {
    let mut builder = crate::builder::IRBuilder::new();
    let lhs = builder.ir_read_integer(crate::circuit_input::InputPath::new("x", vec![]), true);
    let zero = builder.ir_constant_int(0);
    let cmp = builder.ir_greater_than_i(&lhs, &zero);
    let before = builder.stmts.len();
    let out = crate::helpers::value_ops::to_scalar_bool(&mut builder, &cmp);

    assert_eq!(out.value_id(), cmp.value_id());
    assert_eq!(
        builder.stmts.len(),
        before,
        "Boolean input should not emit a BoolCastIR"
    );
}

#[test]
fn to_scalar_bool_still_casts_integer_values() {
    let mut builder = crate::builder::IRBuilder::new();
    let x = builder.ir_read_integer(crate::circuit_input::InputPath::new("x", vec![]), true);
    let before = builder.stmts.len();
    let out = crate::helpers::value_ops::to_scalar_bool(&mut builder, &x);

    assert_ne!(out.value_id(), x.value_id());
    assert_eq!(builder.stmts.len(), before + 1);
    assert!(matches!(builder.stmts.last().unwrap().ir, IR::BoolCast));
}

fn float_input(name: &str) -> serde_json::Value {
    serde_json::json!({
        "__class__": "ASTCircuitInput",
        "name": name,
        "annotation": {
            "__class__": "ASTAnnotation",
            "kind": "Private",
            "dt": "Float"
        }
    })
}

fn int_input(name: &str) -> serde_json::Value {
    serde_json::json!({
        "__class__": "ASTCircuitInput",
        "name": name,
        "annotation": {
            "__class__": "ASTAnnotation",
            "kind": "Private",
            "dt": "Integer"
        }
    })
}

fn assert_cmp(operator: &str, lhs: serde_json::Value, rhs: serde_json::Value) -> serde_json::Value {
    serde_json::json!({
        "__class__": "ASTAssertStatement",
        "expr": {
            "__class__": "ASTBinaryOperator",
            "operator": operator,
            "lhs": lhs,
            "rhs": rhs,
        }
    })
}

fn and(lhs: serde_json::Value, rhs: serde_json::Value) -> serde_json::Value {
    serde_json::json!({
        "__class__": "ASTBinaryOperator",
        "operator": "and",
        "lhs": lhs,
        "rhs": rhs,
    })
}

fn cmp_expr(operator: &str, lhs: serde_json::Value, rhs: serde_json::Value) -> serde_json::Value {
    serde_json::json!({
        "__class__": "ASTBinaryOperator",
        "operator": operator,
        "lhs": lhs,
        "rhs": rhs,
    })
}

fn assert_expr(expr: serde_json::Value) -> serde_json::Value {
    serde_json::json!({
        "__class__": "ASTAssertStatement",
        "expr": expr
    })
}

fn load(name: &str) -> serde_json::Value {
    serde_json::json!({"__class__": "ASTLoad", "name": name})
}

fn float_const(value: f64) -> serde_json::Value {
    serde_json::json!({"__class__": "ASTConstantFloat", "value": value})
}

fn int_const(value: i64) -> serde_json::Value {
    serde_json::json!({"__class__": "ASTConstantInteger", "value": value})
}

fn generate_with_builder(circuit_json: serde_json::Value) -> IRGenerator {
    let node: ASTNode = serde_json::from_value(circuit_json).unwrap();
    let ASTNode::ASTCircuit(circuit) = node else {
        panic!("expected circuit");
    };
    let mut gen = IRGenerator::new(IRGenConfig::default());
    gen.visit_circuit(&circuit);
    gen
}

fn input_vid(gen: &IRGenerator, name: &str) -> ValueId {
    gen.builder
        .stmts
        .iter()
        .find_map(|stmt| match &stmt.ir {
            IR::ReadInteger { path, .. } | IR::ReadFloat { path, .. } if path.param == name => {
                Some(stmt.value_id)
            }
            _ => None,
        })
        .unwrap_or_else(|| panic!("missing input {name}"))
}

#[test]
fn assert_scalar_fact_discharges_later_sqrt_requires_without_extra_witness() {
    let circuit = serde_json::json!({
        "__class__": "ASTCircuit",
        "inputs": [float_input("x")],
        "block": [
            assert_cmp("gte", load("x"), float_const(0.0)),
            {
                "__class__": "ASTAssignStatement",
                "targets": [{"__class__": "ASTNameAssignTarget", "name": "y"}],
                "value": {
                    "__class__": "ASTNamedAttribute",
                    "target": "np",
                    "member": "sqrt",
                    "args": [load("x")],
                    "kwargs": {}
                }
            }
        ]
    });
    let gen = generate_with_builder(circuit);
    let assert_count = gen
        .builder
        .stmts
        .iter()
        .enumerate()
        .filter(|(i, stmt)| matches!(stmt.ir, IR::Assert) && !is_input_check(&gen.builder.stmts, *i))
        .count();
    assert_eq!(
        assert_count, 1,
        "sqrt requires should be Proved from the prior assert and emit no extra witness assert"
    );

    let x_vid = input_vid(&gen, "x");
    let expected = ContractTerm::Cmp {
        op: CmpOp::Ge,
        lhs: Box::new(ContractTerm::Var(ContractVar::Value(x_vid))),
        rhs: Box::new(ContractTerm::LitFloat(ContractFloat(0.0))),
    };
    let facts = gen.builder.facts.per_value.get(&x_vid).unwrap();
    assert!(facts.iter().any(|fact| *fact == expected));
}

#[test]
fn assert_conjunction_fact_discharges_later_sqrt_requires_without_extra_witness() {
    let circuit = serde_json::json!({
        "__class__": "ASTCircuit",
        "inputs": [float_input("x"), int_input("y")],
        "block": [
            assert_expr(and(
                cmp_expr("gte", load("x"), float_const(0.0)),
                cmp_expr("lt", load("y"), int_const(10))
            )),
            {
                "__class__": "ASTAssignStatement",
                "targets": [{"__class__": "ASTNameAssignTarget", "name": "z"}],
                "value": {
                    "__class__": "ASTNamedAttribute",
                    "target": "np",
                    "member": "sqrt",
                    "args": [load("x")],
                    "kwargs": {}
                }
            }
        ]
    });
    let gen = generate_with_builder(circuit);
    let assert_count = gen
        .builder
        .stmts
        .iter()
        .enumerate()
        .filter(|(i, stmt)| matches!(stmt.ir, IR::Assert) && !is_input_check(&gen.builder.stmts, *i))
        .count();
    assert_eq!(
        assert_count, 1,
        "sqrt requires should be Proved from the supported assert conjunct and emit no extra witness assert"
    );

    let x_vid = input_vid(&gen, "x");
    let expected = ContractTerm::Cmp {
        op: CmpOp::Ge,
        lhs: Box::new(ContractTerm::Var(ContractVar::Value(x_vid))),
        rhs: Box::new(ContractTerm::LitFloat(ContractFloat(0.0))),
    };
    let facts = gen.builder.facts.per_value.get(&x_vid).unwrap();
    assert!(facts.iter().any(|fact| *fact == expected));
}

#[test]
fn assert_conjunction_publishes_each_supported_scalar_fact() {
    let circuit = serde_json::json!({
        "__class__": "ASTCircuit",
        "inputs": [float_input("x"), int_input("y")],
        "block": [
            assert_expr(and(
                cmp_expr("gte", load("x"), float_const(0.0)),
                cmp_expr("lt", load("y"), int_const(10))
            ))
        ]
    });
    let gen = generate_with_builder(circuit);
    let x_vid = input_vid(&gen, "x");
    let y_vid = input_vid(&gen, "y");
    let x_fact = ContractTerm::Cmp {
        op: CmpOp::Ge,
        lhs: Box::new(ContractTerm::Var(ContractVar::Value(x_vid))),
        rhs: Box::new(ContractTerm::LitFloat(ContractFloat(0.0))),
    };
    let y_fact = ContractTerm::Cmp {
        op: CmpOp::Lt,
        lhs: Box::new(ContractTerm::Var(ContractVar::Value(y_vid))),
        rhs: Box::new(ContractTerm::LitInt(10)),
    };

    assert!(
        gen.builder
            .facts
            .per_value
            .get(&x_vid)
            .is_some_and(|facts| facts.iter().any(|fact| *fact == x_fact)),
        "first supported conjunct should be indexed under x"
    );
    assert!(
        gen.builder
            .facts
            .per_value
            .get(&y_vid)
            .is_some_and(|facts| facts.iter().any(|fact| *fact == y_fact)),
        "second supported conjunct should be indexed under y"
    );
}

#[test]
fn assert_scalar_fact_is_guarded_after_branch_join() {
    let circuit = serde_json::json!({
        "__class__": "ASTCircuit",
        "inputs": [float_input("x"), int_input("flag")],
        "block": [
            {
                "__class__": "ASTCondStatement",
                "cond": load("flag"),
                "t_block": [assert_cmp("gte", load("x"), float_const(0.0))],
                "f_block": []
            }
        ]
    });
    let gen = generate_with_builder(circuit);
    let x_vid = input_vid(&gen, "x");
    let assert_fact = ContractTerm::Cmp {
        op: CmpOp::Ge,
        lhs: Box::new(ContractTerm::Var(ContractVar::Value(x_vid))),
        rhs: Box::new(ContractTerm::LitFloat(ContractFloat(0.0))),
    };
    let facts: Vec<_> = gen
        .builder
        .facts
        .visible_facts()
        .into_iter()
        .cloned()
        .collect();
    assert_eq!(
        prove_with_facts(&assert_fact, &facts, &[]),
        ProveOutcome::Unknown,
        "post-join assert fact must remain guarded without the branch path"
    );
    // The fact is kept under the branch's execution guard (usable again
    // by code that runs under the same condition), not unconditionally.
    assert!(gen
        .builder
        .facts
        .guarded_facts()
        .any(|(atoms, fact)| !atoms.is_empty() && *fact == assert_fact));
}

#[test]
fn assert_scalar_fact_inside_loop_does_not_escape() {
    let circuit = serde_json::json!({
        "__class__": "ASTCircuit",
        "inputs": [float_input("x")],
        "block": [
            {
                "__class__": "ASTForInStatement",
                "target": {"__class__": "ASTNameAssignTarget", "name": "i"},
                "iter_expr": {
                    "__class__": "ASTSquareBrackets",
                    "values": [{"__class__": "ASTConstantInteger", "value": 1}]
                },
                "block": [assert_cmp("gte", load("x"), float_const(0.0))],
                "orelse": []
            }
        ]
    });
    let gen = generate_with_builder(circuit);
    let x_vid = input_vid(&gen, "x");
    assert!(
        gen.builder.facts.per_value.get(&x_vid).is_none(),
        "loop-local assert fact must be removed from per_value on loop exit"
    );
}

#[test]
fn unsupported_assert_shape_still_emits_assert_without_fact() {
    let circuit = serde_json::json!({
        "__class__": "ASTCircuit",
        "inputs": [float_input("x")],
        "block": [
            {
                "__class__": "ASTAssertStatement",
                "expr": load("x")
            }
        ]
    });
    let gen = generate_with_builder(circuit);
    assert!(
        gen.builder
            .stmts
            .iter()
            .any(|stmt| matches!(stmt.ir, IR::Assert)),
        "unsupported assert truthiness should still emit witness enforcement"
    );
    let x_vid = input_vid(&gen, "x");
    assert!(
        gen.builder.facts.per_value.get(&x_vid).is_none(),
        "unsupported assert shape must publish no compile-time fact"
    );
}

#[test]
fn composite_assert_comparison_still_emits_assert_without_scalar_fact() {
    let circuit = serde_json::json!({
        "__class__": "ASTCircuit",
        "inputs": [],
        "block": [
            {
                "__class__": "ASTAssignStatement",
                "targets": [{"__class__": "ASTNameAssignTarget", "name": "arr"}],
                "value": {
                    "__class__": "ASTSquareBrackets",
                    "values": [int_const(1), int_const(2)]
                }
            },
            assert_cmp("eq", load("arr"), load("arr"))
        ]
    });
    let gen = generate_with_builder(circuit);
    assert!(
        gen.builder
            .stmts
            .iter()
            .any(|stmt| matches!(stmt.ir, IR::Assert)),
        "array equality assertion should still emit witness enforcement"
    );

    let arr_vid = gen
        .ctx
        .get("arr")
        .and_then(|v| v.value_id())
        .expect("array binding should carry a composite ValueId");
    assert!(
        gen.builder.facts.per_value.get(&arr_vid).is_none(),
        "composite assert comparison must not publish a scalar-looking fact for the array ValueId"
    );
}

#[test]
fn mixed_assert_conjunction_publishes_only_supported_scalar_siblings() {
    let circuit = serde_json::json!({
        "__class__": "ASTCircuit",
        "inputs": [float_input("x")],
        "block": [
            {
                "__class__": "ASTAssignStatement",
                "targets": [{"__class__": "ASTNameAssignTarget", "name": "arr"}],
                "value": {
                    "__class__": "ASTSquareBrackets",
                    "values": [int_const(1), int_const(2)]
                }
            },
            assert_expr(and(
                cmp_expr("gte", load("x"), float_const(0.0)),
                cmp_expr("eq", load("arr"), load("arr"))
            ))
        ]
    });
    let gen = generate_with_builder(circuit);
    assert!(
        gen.builder
            .stmts
            .iter()
            .any(|stmt| matches!(stmt.ir, IR::Assert)),
        "full mixed conjunction should still emit witness enforcement"
    );

    let x_vid = input_vid(&gen, "x");
    let x_fact = ContractTerm::Cmp {
        op: CmpOp::Ge,
        lhs: Box::new(ContractTerm::Var(ContractVar::Value(x_vid))),
        rhs: Box::new(ContractTerm::LitFloat(ContractFloat(0.0))),
    };
    assert!(
        gen.builder
            .facts
            .per_value
            .get(&x_vid)
            .is_some_and(|facts| facts.iter().any(|fact| *fact == x_fact)),
        "supported scalar sibling conjunct should still publish"
    );

    let arr_vid = gen
        .ctx
        .get("arr")
        .and_then(|v| v.value_id())
        .expect("array binding should carry a composite ValueId");
    assert!(
        gen.builder.facts.per_value.get(&arr_vid).is_none(),
        "unsupported composite sibling must not publish a scalar-looking fact"
    );
}

/// P4 round 1 — while-loop early termination via the resolver.
///
/// A `while i < 5: i += 1` loop with starting `i = 0` must terminate
/// after 5 unrolled iterations, well before `loop_limit` (default 256).
/// Today this works through the static-val fast-path inside
/// `bool_val()` / `int_val()`, but the test pins the behaviour: any
/// regression in the resolver routing in `visit_while` (e.g. wrong
/// pre-/post-iteration env capture) would make the loop unroll to its
/// full 256-iteration cap and emit thousands of guarded statements.
///
/// We pick a tight `loop_limit = 32`: if the early-exit fires after
/// 5 iterations the resulting graph is small (~30 stmts); if it
/// regresses to "always unroll", the graph balloons to 32+
/// guarded body emissions.
#[test]
fn test_while_static_guard_early_termination() {
    // while i < 5: i = i + 1   (start i = 0)
    let json = r#"{
            "__class__": "ASTCircuit",
            "block": [
                {
                    "__class__": "ASTAssignStatement",
                    "targets": [{"__class__": "ASTNameAssignTarget", "name": "i"}],
                    "value": {"__class__": "ASTConstantInteger", "value": 0}
                },
                {
                    "__class__": "ASTWhileStatement",
                    "test_expr": {
                        "__class__": "ASTBinaryOperator",
                        "operator": "lt",
                        "lhs": {"__class__": "ASTLoad", "name": "i"},
                        "rhs": {"__class__": "ASTConstantInteger", "value": 5}
                    },
                    "block": [
                        {
                            "__class__": "ASTAssignStatement",
                            "targets": [{"__class__": "ASTNameAssignTarget", "name": "i"}],
                            "value": {
                                "__class__": "ASTBinaryOperator",
                                "operator": "add",
                                "lhs": {"__class__": "ASTLoad", "name": "i"},
                                "rhs": {"__class__": "ASTConstantInteger", "value": 1}
                            }
                        }
                    ]
                }
            ],
            "inputs": []
        }"#;

    let mut cfg = IRGenConfig::default();
    cfg.loop_limit = 32;
    let graph = IRGenerator::generate_from_json(cfg, json).unwrap();

    // The early-exit path should have fired at iteration 5
    // (when `i` first reaches 5 and the static guard becomes false).
    // Each iteration produces ~5 IR statements (load i, const 5, lt,
    // load i, const 1, add). 5 iterations × ~6 stmts ≈ 30 stmts.
    // A regression to full unrolling would yield 32 iterations of
    // guarded bodies — ≥ 100+ stmts.
    assert!(
        graph.len() < 80,
        "expected early termination, got {} stmts (regression?)",
        graph.len()
    );
}

#[test]
fn while_guard_throttle_skips_repeated_unresolved_resolver_queries() {
    let json = r#"{
            "__class__": "ASTCircuit",
            "inputs": [
                {
                    "__class__": "ASTCircuitInput",
                    "name": "x",
                    "annotation": {
                        "__class__": "ASTAnnotation",
                        "kind": "Private",
                        "dt": "Integer"
                    }
                }
            ],
            "block": [
                {
                    "__class__": "ASTWhileStatement",
                    "test_expr": {
                        "__class__": "ASTBinaryOperator",
                        "operator": "gt",
                        "lhs": {"__class__": "ASTLoad", "name": "x"},
                        "rhs": {"__class__": "ASTConstantInteger", "value": 0}
                    },
                    "block": [
                        {
                            "__class__": "ASTAssignStatement",
                            "targets": [{"__class__": "ASTNameAssignTarget", "name": "y"}],
                            "value": {"__class__": "ASTConstantInteger", "value": 1}
                        }
                    ],
                    "orelse": []
                }
            ]
        }"#;

    let node: ASTNode = serde_json::from_str(json).unwrap();
    let ASTNode::ASTCircuit(circuit) = node else {
        panic!("expected circuit");
    };
    let mut cfg = IRGenConfig::default();
    cfg.smt_enable = true;
    cfg.smt_query_rlimit = 1;
    cfg.loop_limit = 16;
    let mut gen = IRGenerator::new(cfg);
    let telemetry = gen
        .builder
        .resolver_telemetry()
        .expect("SMT-enabled generator should have telemetry");
    // `while x > 0` never changes `x`: the trip count has no bound, so the
    // loop is rejected once the quota is used up (it is no longer truncated).
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        gen.visit_circuit(&circuit)
    }));
    assert!(outcome.is_err(), "an unbounded while loop must be rejected");

    assert!(
        telemetry
            .queries_skipped_while_throttle
            .load(Ordering::SeqCst)
            > 0,
        "repeated unresolved while guards should be skipped by the local throttle"
    );
    assert!(
        telemetry.queries_total.load(Ordering::SeqCst) <= 8,
        "throttle should cap resolver attempts for one unproductive while loop"
    );
}

fn static_for_while_cap_fixture(init: &str, decrement_rhs: &str) -> String {
    static_for_while_cap_fixture_with_iter_and_guard(
        init,
        decrement_rhs,
        &[
            r#"{"__class__": "ASTConstantInteger", "value": 1}"#,
            r#"{"__class__": "ASTConstantInteger", "value": 2}"#,
            r#"{"__class__": "ASTConstantInteger", "value": 3}"#,
            r#"{"__class__": "ASTConstantInteger", "value": 4}"#,
        ],
        "gt",
        r#"{"__class__": "ASTLoad", "name": "j"}"#,
        r#"{"__class__": "ASTConstantInteger", "value": 0}"#,
    )
}

fn static_for_while_cap_fixture_with_iter_and_guard(
    init: &str,
    decrement_rhs: &str,
    iter_values: &[&str],
    guard_operator: &str,
    guard_lhs: &str,
    guard_rhs: &str,
) -> String {
    static_for_while_cap_fixture_full(
        init,
        decrement_rhs,
        iter_values,
        guard_operator,
        guard_lhs,
        guard_rhs,
        "j",
    )
}

fn static_for_while_cap_fixture_full(
    init: &str,
    decrement_rhs: &str,
    iter_values: &[&str],
    guard_operator: &str,
    guard_lhs: &str,
    guard_rhs: &str,
    decrement_target: &str,
) -> String {
    let iter_values = iter_values.join(",\n                            ");
    let decrement_stmt = format!(
            r#"{{
                                    "__class__": "ASTAssignStatement",
                                    "targets": [{{"__class__": "ASTNameAssignTarget", "name": "{decrement_target}"}}],
                                    "value": {{
                                        "__class__": "ASTBinaryOperator",
                                        "operator": "sub",
                                        "lhs": {{"__class__": "ASTLoad", "name": "{decrement_target}"}},
                                        "rhs": {decrement_rhs}
                                    }}
                                }}"#
        );
    format!(
        r#"{{
            "__class__": "ASTCircuit",
            "inputs": [
                {{
                    "__class__": "ASTCircuitInput",
                    "name": "x",
                    "annotation": {{
                        "__class__": "ASTAnnotation",
                        "kind": "Private",
                        "dt": "Integer"
                    }}
                }}
            ],
            "block": [
                {{
                    "__class__": "ASTForInStatement",
                    "target": {{"__class__": "ASTNameAssignTarget", "name": "i"}},
                    "iter_expr": {{
                        "__class__": "ASTSquareBrackets",
                        "values": [
                            {iter_values}
                        ]
                    }},
                    "block": [
                        {{
                            "__class__": "ASTAssignStatement",
                            "targets": [{{"__class__": "ASTNameAssignTarget", "name": "j"}}],
                            "value": {init}
                        }},
                        {{
                            "__class__": "ASTWhileStatement",
                            "test_expr": {{
                                "__class__": "ASTBinaryOperator",
                                "operator": "and",
                                "lhs": {{
                                    "__class__": "ASTBinaryOperator",
                                    "operator": "{guard_operator}",
                                    "lhs": {guard_lhs},
                                    "rhs": {guard_rhs}
                                }},
                                "rhs": {{
                                    "__class__": "ASTBinaryOperator",
                                    "operator": "gt",
                                    "lhs": {{"__class__": "ASTLoad", "name": "x"}},
                                    "rhs": {{"__class__": "ASTConstantInteger", "value": 0}}
                                }}
                            }},
                            "block": [
                                {{
                                    "__class__": "ASTAssignStatement",
                                    "targets": [{{"__class__": "ASTNameAssignTarget", "name": "sink"}}],
                                    "value": {{"__class__": "ASTLoad", "name": "x"}}
                                }},
                                {decrement_stmt}
                            ],
                            "orelse": []
                        }}
                    ],
                    "orelse": []
                }}
            ]
        }}"#
    )
}

fn static_for_while_prewrite_fixture() -> String {
    let init = r#"{"__class__": "ASTLoad", "name": "i"}"#;
    let decrement_rhs = r#"{"__class__": "ASTConstantInteger", "value": 1}"#;
    let mut json = static_for_while_cap_fixture(init, decrement_rhs);
    let marker = r#"{
                                    "__class__": "ASTAssignStatement",
                                    "targets": [{"__class__": "ASTNameAssignTarget", "name": "sink"}],
                                    "value": {"__class__": "ASTLoad", "name": "x"}
                                }"#;
    let prewrite = r#"{
                                    "__class__": "ASTAssignStatement",
                                    "targets": [{"__class__": "ASTNameAssignTarget", "name": "j"}],
                                    "value": {"__class__": "ASTLoad", "name": "x"}
                                },
                                {
                                    "__class__": "ASTAssignStatement",
                                    "targets": [{"__class__": "ASTNameAssignTarget", "name": "sink"}],
                                    "value": {"__class__": "ASTLoad", "name": "x"}
                                }"#;
    json = json.replacen(marker, prewrite, 1);
    json
}

fn static_for_static_array_read_fixture(
    while_body_value: &str,
    after_loop_value: Option<&str>,
    init: &str,
    decrement_rhs: &str,
) -> String {
    let after_loop_stmt = after_loop_value.map_or(String::new(), |value| {
        format!(
            r#",
                        {{
                            "__class__": "ASTAssignStatement",
                            "targets": [{{"__class__": "ASTNameAssignTarget", "name": "after"}}],
                            "value": {value}
                        }}"#
        )
    });
    format!(
        r#"{{
            "__class__": "ASTCircuit",
            "inputs": [
                {{
                    "__class__": "ASTCircuitInput",
                    "name": "arr",
                    "annotation": {{
                        "__class__": "ASTAnnotation",
                        "kind": "Private",
                        "dt": {{"NDArray": {{"shape": [4], "dtype": "Float"}}}}
                    }}
                }},
                {{
                    "__class__": "ASTCircuitInput",
                    "name": "x",
                    "annotation": {{
                        "__class__": "ASTAnnotation",
                        "kind": "Private",
                        "dt": "Integer"
                    }}
                }}
            ],
            "block": [
                {{
                    "__class__": "ASTForInStatement",
                    "target": {{"__class__": "ASTNameAssignTarget", "name": "i"}},
                    "iter_expr": {{
                        "__class__": "ASTSquareBrackets",
                        "values": [
                            {{"__class__": "ASTConstantInteger", "value": 1}},
                            {{"__class__": "ASTConstantInteger", "value": 2}},
                            {{"__class__": "ASTConstantInteger", "value": 3}}
                        ]
                    }},
                    "block": [
                        {{
                            "__class__": "ASTAssignStatement",
                            "targets": [{{"__class__": "ASTNameAssignTarget", "name": "j"}}],
                            "value": {init}
                        }},
                        {{
                            "__class__": "ASTWhileStatement",
                            "test_expr": {{
                                "__class__": "ASTBinaryOperator",
                                "operator": "and",
                                "lhs": {{
                                    "__class__": "ASTBinaryOperator",
                                    "operator": "gt",
                                    "lhs": {{"__class__": "ASTLoad", "name": "j"}},
                                    "rhs": {{"__class__": "ASTConstantInteger", "value": 0}}
                                }},
                                "rhs": {{
                                    "__class__": "ASTBinaryOperator",
                                    "operator": "gt",
                                    "lhs": {{"__class__": "ASTLoad", "name": "x"}},
                                    "rhs": {{"__class__": "ASTConstantInteger", "value": 0}}
                                }}
                            }},
                            "block": [
                                {{
                                    "__class__": "ASTAssignStatement",
                                    "targets": [{{"__class__": "ASTNameAssignTarget", "name": "sink"}}],
                                    "value": {while_body_value}
                                }},
                                {{
                                    "__class__": "ASTAssignStatement",
                                    "targets": [{{"__class__": "ASTNameAssignTarget", "name": "j"}}],
                                    "value": {{
                                        "__class__": "ASTBinaryOperator",
                                        "operator": "sub",
                                        "lhs": {{"__class__": "ASTLoad", "name": "j"}},
                                        "rhs": {decrement_rhs}
                                    }}
                                }}
                            ],
                            "orelse": []
                        }}{after_loop_stmt}
                    ],
                    "orelse": []
                }}
            ]
        }}"#
    )
}

#[test]
fn masked_last_source_reuses_scan_and_keeps_index_checks() {
    use crate::helpers::static_array::{base::to_static_array, mask::try_apply_boolean_mask_read};
    let mut b = crate::builder::IRBuilder::new();
    let sources: Vec<_> = [10, 20, 30].into_iter().map(|x| b.ir_constant_int(x)).collect();
    let zero = b.ir_constant_int(0);
    let mask: Vec<_> = (0..3).map(|i| {
        let input = b.ir_read_integer(crate::circuit_input::InputPath::new(&format!("m{i}"), vec![]), false);
        b.ir_greater_than_i(&input, &zero)
    }).collect();
    let as_list = |values: Vec<Value>| Value::List(crate::types::CompositeData {
        elements_type: values.iter().map(Value::zinnia_type).collect(), values,
        value_id: ValueId::next(),
    });
    let arr = to_static_array(&mut b, &as_list(sources.clone())).unwrap();
    let mask_arr = to_static_array(&mut b, &as_list(mask.clone())).unwrap();
    let Value::DynamicNDArray(data) = try_apply_boolean_mask_read(&mut b, &arr, &mask_arr).unwrap() else {
        panic!("runtime mask must produce a dynamic array");
    };
    let minus_one = b.ir_constant_int(-1);
    let indices = [SliceIndex::Single(minus_one)];
    let _original = crate::helpers::array_ops::dyn_subscript(&mut b, &data, &indices);
    let before = b.stmts.len();
    let selected = super::visitors::last_masked_source(&mut b, &data, &indices).unwrap();
    let last = &b.stmts[selected.stmt_id().unwrap() as usize];
    assert!(matches!(last.ir, IR::SelectI));
    assert_eq!(last.arguments[0], mask[2].stmt_id().unwrap());
    assert_eq!(last.arguments[1], sources[2].stmt_id().unwrap());
    assert_eq!(b.stmts.len(), before + 2, "only two selects are appended; original bounds and empty-mask checks remain intact");
    let mut sliced = data.clone();
    sliced.meta.logical_offset = 1;
    assert!(super::visitors::last_masked_source(&mut b, &sliced, &indices).is_none());
    let mut reversed = data.clone();
    reversed.meta.logical_strides = vec![2];
    assert!(super::visitors::last_masked_source(&mut b, &reversed, &indices).is_none());
    let mut resized = data.clone();
    resized.meta.runtime_length.stmt_id = zero.stmt_id();
    assert!(super::visitors::last_masked_source(&mut b, &resized, &indices).is_none());
    let other_index = [SliceIndex::Single(zero)];
    assert!(super::visitors::last_masked_source(&mut b, &data, &other_index).is_none());
}

#[test]
fn active_while_counter_load_unwraps_only_on_proven_path() {
    let mut gen = IRGenerator::new(IRGenConfig::default());
    gen.ctx.circuit_enter(crate::types::ZinniaType::None);
    let x = gen.builder.ir_read_integer(crate::circuit_input::InputPath::new("x", vec![]), false);
    let zero = gen.builder.ir_constant_int(0);
    let guard = gen.builder.ir_greater_than_i(&x, &zero);
    let old = gen.builder.ir_constant_int(4);
    let one = gen.builder.ir_constant_int(1);
    let next = gen.builder.ir_sub_i(&old, &one);
    let merged = gen.builder.ir_select_i(&guard, &next, &old);
    gen.ctx.set("j", merged.clone());
    gen.loop_index_bound_contexts.push(LoopIndexBoundContext {
        var_name: "j".into(), max_bound: 4, current_bound: 4,
        positive_guard: true, guard_condition: None, record_guard_read: false,
    });
    let load = ASTLoad { name: "j".into(), dbg: None };
    assert_eq!(gen.visit_load(&load).stmt_id(), merged.stmt_id(), "an unrelated path cannot unwrap the exit binding");
    gen.ctx.loop_enter();
    let not_guard = gen.builder.ir_logical_not(&guard);
    gen.ctx.loop_break(Some(not_guard), &mut gen.builder);
    // loop_break constructs not(not(guard)); use the shared structural
    // implication checker rather than requiring condition node identity.
    assert_eq!(gen.visit_load(&load).int_val(), Some(3));
    assert_eq!(gen.ctx.get("j").unwrap().stmt_id(), merged.stmt_id());
    gen.ctx.loop_leave();
    gen.loop_index_bound_contexts.pop();
    assert_eq!(gen.visit_load(&load).stmt_id(), merged.stmt_id(), "the post-loop value must still include the inactive path");
}

#[test]
fn static_for_alias_unit_decrement_caps_symbolic_while_unroll() {
    let init = r#"{"__class__": "ASTLoad", "name": "i"}"#;
    let decrement = r#"{"__class__": "ASTConstantInteger", "value": 1}"#;
    let mut cfg = IRGenConfig::default();
    cfg.loop_limit = 32;
    let graph =
        IRGenerator::generate_from_json(cfg, &static_for_while_cap_fixture(init, decrement))
            .unwrap();

    assert!(
        graph.len() < 320,
        "expected static-for structural cap to avoid full loop_limit unroll, got {} stmts",
        graph.len()
    );
}

#[test]
fn loop_derived_static_array_read_skips_index_bound_witness_bundle() {
    let body_value = r#"{
                                    "__class__": "ASTSubscriptExp",
                                    "val": {"__class__": "ASTLoad", "name": "arr"},
                                    "slicing": {
                                        "__class__": "ASTSlice",
                                        "data": [
                                            {
                                                "__class__": "ASTBinaryOperator",
                                                "operator": "sub",
                                                "lhs": {"__class__": "ASTLoad", "name": "j"},
                                                "rhs": {"__class__": "ASTConstantInteger", "value": 1}
                                            }
                                        ]
                                    }
                                }"#;
    let json = static_for_static_array_read_fixture(
        body_value,
        None,
        r#"{"__class__": "ASTLoad", "name": "i"}"#,
        r#"{"__class__": "ASTConstantInteger", "value": 1}"#,
    );
    let mut cfg = IRGenConfig::default();
    cfg.loop_limit = 32;
    let graph = IRGenerator::generate_from_json(cfg, &json).unwrap();

    let assert_count = graph
        .stmts
        .iter()
        .enumerate()
        .filter(|(i, stmt)| matches!(stmt.ir, IR::Assert) && !is_input_check(&graph.stmts, *i) && !is_while_backstop(&graph.stmts, *i))
        .count();
    let gte_count = graph
        .stmts
        .iter()
        .enumerate()
        .filter(|(i, stmt)| matches!(stmt.ir, IR::GteI) && !is_input_check(&graph.stmts, *i))
        .count();
    let lt_count = graph
        .stmts
        .iter()
        .enumerate()
        .filter(|(i, stmt)| matches!(stmt.ir, IR::LtI) && !is_input_check(&graph.stmts, *i))
        .count();

    assert_eq!(
        assert_count, 0,
        "preproved loop-derived `j - 1` static_array_read should not emit bound Asserts"
    );
    assert_eq!(
        gte_count, 0,
        "preproved loop-derived `j - 1` static_array_read should not emit lower-bound checks"
    );
    assert_eq!(
        lt_count, 0,
        "preproved loop-derived `j - 1` static_array_read should not emit upper-bound checks"
    );
}

#[test]
fn same_index_outside_matched_while_keeps_static_array_read_witness_bundle() {
    let body_value = r#"{"__class__": "ASTLoad", "name": "x"}"#;
    let after_value = r#"{
                                    "__class__": "ASTSubscriptExp",
                                    "val": {"__class__": "ASTLoad", "name": "arr"},
                                    "slicing": {
                                        "__class__": "ASTSlice",
                                        "data": [
                                            {
                                                "__class__": "ASTBinaryOperator",
                                                "operator": "sub",
                                                "lhs": {"__class__": "ASTLoad", "name": "j"},
                                                "rhs": {"__class__": "ASTConstantInteger", "value": 1}
                                            }
                                        ]
                                    }
                                }"#;
    let json = static_for_static_array_read_fixture(
        body_value,
        Some(after_value),
        r#"{"__class__": "ASTLoad", "name": "i"}"#,
        r#"{"__class__": "ASTConstantInteger", "value": 1}"#,
    );
    let mut cfg = IRGenConfig::default();
    cfg.loop_limit = 32;
    let graph = IRGenerator::generate_from_json(cfg, &json).unwrap();

    let assert_count = graph
        .stmts
        .iter()
        .filter(|stmt| matches!(stmt.ir, IR::Assert))
        .count();

    assert!(
        assert_count > 0,
        "same-looking `j - 1` read outside active matched while must keep fallback witness path"
    );
}

#[test]
fn non_unit_decrement_keeps_static_array_read_witness_bundle() {
    let body_value = r#"{
                                    "__class__": "ASTSubscriptExp",
                                    "val": {"__class__": "ASTLoad", "name": "arr"},
                                    "slicing": {
                                        "__class__": "ASTSlice",
                                        "data": [
                                            {
                                                "__class__": "ASTBinaryOperator",
                                                "operator": "sub",
                                                "lhs": {"__class__": "ASTLoad", "name": "j"},
                                                "rhs": {"__class__": "ASTConstantInteger", "value": 1}
                                            }
                                        ]
                                    }
                                }"#;
    let json = static_for_static_array_read_fixture(
        body_value,
        None,
        r#"{"__class__": "ASTLoad", "name": "i"}"#,
        r#"{"__class__": "ASTConstantInteger", "value": 2}"#,
    );
    // Test fallback read checks independently of the rejection of loops
    // whose termination cannot be proved. The non-unit decrement still
    // prevents the unit-decrement index proof from matching.
    let json = break_after_tested_while_body(&json);
    let mut cfg = IRGenConfig::default();
    cfg.loop_limit = 8;
    let graph = IRGenerator::generate_from_json(cfg, &json).unwrap();

    let assert_count = graph
        .stmts
        .iter()
        .filter(|stmt| matches!(stmt.ir, IR::Assert))
        .count();

    assert!(
        assert_count > 0,
        "non-unit decrement must not use the preproved index-bound path"
    );
}

fn break_after_tested_while_body(json: &str) -> String {
    let mut ast: serde_json::Value = serde_json::from_str(json).unwrap();
    let body = ast["block"][0]["block"][1]["block"].as_array_mut().unwrap();
    body.push(serde_json::json!({"__class__": "ASTBreakStatement"}));
    serde_json::to_string(&ast).unwrap()
}

fn replace_guard_rhs_with_static_array_read(json: &mut String) {
    let old_guard_rhs = r#"{
                                    "__class__": "ASTBinaryOperator",
                                    "operator": "gt",
                                    "lhs": {"__class__": "ASTLoad", "name": "x"},
                                    "rhs": {"__class__": "ASTConstantInteger", "value": 0}
                                }"#;
    let new_guard_rhs = r#"{
                                    "__class__": "ASTBinaryOperator",
                                    "operator": "gt",
                                    "lhs": {
                                        "__class__": "ASTSubscriptExp",
                                        "val": {"__class__": "ASTLoad", "name": "arr"},
                                        "slicing": {
                                            "__class__": "ASTSlice",
                                            "data": [
                                                {
                                                    "__class__": "ASTBinaryOperator",
                                                    "operator": "sub",
                                                    "lhs": {"__class__": "ASTLoad", "name": "j"},
                                                    "rhs": {"__class__": "ASTConstantInteger", "value": 1}
                                                }
                                            ]
                                        }
                                    },
                                    "rhs": {"__class__": "ASTConstantFloat", "value": 0.0}
                                }"#;
    *json = json.replacen(old_guard_rhs, new_guard_rhs, 1);
}

fn static_array_read_stmt_count(graph: &crate::ir::IRGraph) -> usize {
    graph
        .stmts
        .iter()
        .filter(|stmt| matches!(stmt.ir, IR::ArrayRead(_)))
        .count()
}

fn forall_eq_const_term(vid: ValueId, k: i64) -> ContractTerm {
    ContractTerm::PredicateApp {
        kind: "forall_eq_const".to_string(),
        args: vec![
            ContractTerm::Var(ContractVar::Value(vid)),
            ContractTerm::LitInt(k),
        ],
    }
}

fn plant_forall_eq_const_fact(gen: &mut IRGenerator, name: &str, k: i64) {
    let vid = gen
        .ctx
        .get(name)
        .and_then(|v| v.value_id())
        .expect("input should carry a value_id");
    gen.builder
        .facts
        .insert_for(vid, forall_eq_const_term(vid, k));
}

fn assert_forall_eq_const_outcome(
    gen: &mut IRGenerator,
    value: &Value,
    k: i64,
    expected: ProveOutcome,
) {
    let vid = value.value_id().expect("value should carry a value_id");
    assert_eq!(gen.builder.prove(&forall_eq_const_term(vid, k)), expected);
}

#[test]
fn scalar_dynamic_static_array_read_skips_forall_eq_const_relay() {
    let mut gen = IRGenerator::new(IRGenConfig::default());
    let ast: ASTCircuit = serde_json::from_str(&static_for_static_array_read_fixture(
        r#"{"__class__": "ASTLoad", "name": "x"}"#,
        None,
        r#"{"__class__": "ASTLoad", "name": "i"}"#,
        r#"{"__class__": "ASTConstantInteger", "value": 1}"#,
    ))
    .unwrap();
    gen.visit_circuit(&ast);
    plant_forall_eq_const_fact(&mut gen, "arr", 0);

    let read = gen.visit(&ASTNode::ASTSubscriptExp(ASTSubscriptExp {
        val: Box::new(ASTNode::ASTLoad(ASTLoad {
            name: "arr".to_string(),
            dbg: None,
        })),
        slicing: ASTSlice {
            data: vec![serde_json::json!({"__class__": "ASTLoad", "name": "x"})],
            dbg: None,
        },
        dbg: None,
    }));

    assert_forall_eq_const_outcome(&mut gen, &read, 0, ProveOutcome::Unknown);
}

#[test]
fn shaped_static_array_slice_keeps_forall_eq_const_relay() {
    let mut gen = IRGenerator::new(IRGenConfig::default());
    let ast: ASTCircuit = serde_json::from_str(&static_for_static_array_read_fixture(
        r#"{"__class__": "ASTLoad", "name": "x"}"#,
        None,
        r#"{"__class__": "ASTLoad", "name": "i"}"#,
        r#"{"__class__": "ASTConstantInteger", "value": 1}"#,
    ))
    .unwrap();
    gen.visit_circuit(&ast);
    plant_forall_eq_const_fact(&mut gen, "arr", 0);

    let slice = gen.visit(&ASTNode::ASTSubscriptExp(ASTSubscriptExp {
        val: Box::new(ASTNode::ASTLoad(ASTLoad {
            name: "arr".to_string(),
            dbg: None,
        })),
        slicing: ASTSlice {
            data: vec![serde_json::json!([
                {"__class__": "ASTConstantInteger", "value": 1},
                {"__class__": "ASTConstantInteger", "value": 3},
                null
            ])],
            dbg: None,
        },
        dbg: None,
    }));

    assert_forall_eq_const_outcome(&mut gen, &slice, 0, ProveOutcome::Proved);
}

fn replace_body_sink_load_with_static_array_assignment_read(json: &mut String, source_name: &str) {
    let old = r#"{
                                    "__class__": "ASTAssignStatement",
                                    "targets": [{"__class__": "ASTNameAssignTarget", "name": "sink"}],
                                    "value": {"__class__": "ASTLoad", "name": "x"}
                                }"#;
    let new = format!(
        r#"{{
                                    "__class__": "ASTAssignStatement",
                                    "targets": [
                                        {{
                                            "__class__": "ASTSubscriptAssignTarget",
                                            "target": {{"__class__": "ASTLoad", "name": "arr"}},
                                            "slicing": {{
                                                "__class__": "ASTSlice",
                                                "data": [
                                                    {{"__class__": "ASTLoad", "name": "j"}}
                                                ]
                                            }}
                                        }}
                                    ],
                                    "value": {{
                                        "__class__": "ASTSubscriptExp",
                                        "val": {{"__class__": "ASTLoad", "name": "{source_name}"}},
                                        "slicing": {{
                                            "__class__": "ASTSlice",
                                            "data": [
                                                {{
                                                    "__class__": "ASTBinaryOperator",
                                                    "operator": "sub",
                                                    "lhs": {{"__class__": "ASTLoad", "name": "j"}},
                                                    "rhs": {{"__class__": "ASTConstantInteger", "value": 1}}
                                                }}
                                            ]
                                        }}
                                    }}
                                }}"#
    );
    *json = json.replacen(old, &new, 1);
}

fn add_second_static_array_input(json: &mut String, name: &str) {
    let second_input = format!(
        r#"                {{
                    "__class__": "ASTCircuitInput",
                    "name": "{name}",
                    "annotation": {{
                        "__class__": "ASTAnnotation",
                        "kind": "Private",
                        "dt": {{"NDArray": {{"shape": [4], "dtype": "Float"}}}}
                    }}
                }},
"#
    );
    *json = json.replacen(
        r#"                {
                    "__class__": "ASTCircuitInput",
                    "name": "arr",
                    "annotation": {
                        "__class__": "ASTAnnotation",
                        "kind": "Private",
                        "dt": {"NDArray": {"shape": [4], "dtype": "Float"}}
                    }
                },
"#,
        &format!(
            r#"                {{
                    "__class__": "ASTCircuitInput",
                    "name": "arr",
                    "annotation": {{
                        "__class__": "ASTAnnotation",
                        "kind": "Private",
                        "dt": {{"NDArray": {{"shape": [4], "dtype": "Float"}}}}
                    }}
                }},
{second_input}"#
        ),
        1,
    );
}

#[test]
fn matching_guard_rhs_static_array_read_uses_guarded_safe_fallback_address() {
    let body_value = r#"{"__class__": "ASTLoad", "name": "x"}"#;
    let mut json = static_for_static_array_read_fixture(
        body_value,
        None,
        r#"{"__class__": "ASTLoad", "name": "i"}"#,
        r#"{"__class__": "ASTConstantInteger", "value": 1}"#,
    );
    replace_guard_rhs_with_static_array_read(&mut json);
    let mut cfg = IRGenConfig::default();
    cfg.loop_limit = 32;
    let graph = IRGenerator::generate_from_json(cfg, &json).unwrap();

    let assert_count = graph
        .stmts
        .iter()
        .enumerate()
        .filter(|(i, stmt)| matches!(stmt.ir, IR::Assert) && !is_input_check(&graph.stmts, *i) && !is_while_backstop(&graph.stmts, *i))
        .count();

    assert_eq!(
        assert_count, 0,
        "matching guard RHS `j - 1` read should use the path-conditioned preproved path"
    );

    let has_guarded_safe_read = graph.stmts.iter().any(|stmt| {
        if !matches!(stmt.ir, IR::ArrayRead(_)) || stmt.arguments.len() != 2 {
            return false;
        }
        let addr_stmt = &graph.stmts[stmt.arguments[1] as usize];
        if !matches!(addr_stmt.ir, IR::SelectI) || addr_stmt.arguments.len() != 3 {
            return false;
        }
        matches!(
            graph.stmts[addr_stmt.arguments[2] as usize].ir,
            IR::ConstantInt { value: 0 }
        )
    });
    assert!(
        has_guarded_safe_read,
        "guard RHS preproved read must select a known-valid fallback address when lhs is false"
    );
}

#[test]
fn nonmatching_guard_lhs_rhs_static_array_read_keeps_witness_bundle() {
    let body_value = r#"{"__class__": "ASTLoad", "name": "x"}"#;
    let mut json = static_for_static_array_read_fixture(
        body_value,
        None,
        r#"{"__class__": "ASTLoad", "name": "i"}"#,
        r#"{"__class__": "ASTConstantInteger", "value": 1}"#,
    );
    replace_guard_rhs_with_static_array_read(&mut json);
    let old_guard_lhs = r#"{
                                    "__class__": "ASTBinaryOperator",
                                    "operator": "gt",
                                    "lhs": {"__class__": "ASTLoad", "name": "j"},
                                    "rhs": {"__class__": "ASTConstantInteger", "value": 0}
                                }"#;
    let new_guard_lhs = r#"{
                                    "__class__": "ASTBinaryOperator",
                                    "operator": "gt",
                                    "lhs": {"__class__": "ASTLoad", "name": "x"},
                                    "rhs": {"__class__": "ASTConstantInteger", "value": 0}
                                }"#;
    json = json.replacen(old_guard_lhs, new_guard_lhs, 1);
    // The x-guard provides no decreasing-counter termination proof.
    // Bound this fixture explicitly so it tests fallback indexing rather
    // than the separate rejection of an unbounded while.
    json = break_after_tested_while_body(&json);
    let mut cfg = IRGenConfig::default();
    cfg.loop_limit = 8;
    let graph = IRGenerator::generate_from_json(cfg, &json).unwrap();

    let assert_count = graph
        .stmts
        .iter()
        .filter(|stmt| matches!(stmt.ir, IR::Assert))
        .count();

    assert!(
        assert_count > 0,
        "same RHS `j - 1` read under a non-matching left conjunct must keep fallback witness path"
    );
}

#[test]
fn guard_rhs_read_is_reused_by_first_matching_body_read_once() {
    let body_value = r#"{"__class__": "ASTLoad", "name": "x"}"#;
    let mut json = static_for_static_array_read_fixture(
        body_value,
        None,
        r#"{"__class__": "ASTLoad", "name": "i"}"#,
        r#"{"__class__": "ASTConstantInteger", "value": 1}"#,
    );
    replace_guard_rhs_with_static_array_read(&mut json);
    replace_body_sink_load_with_static_array_assignment_read(&mut json, "arr");
    let mut cfg = IRGenConfig::default();
    cfg.loop_limit = 32;
    let graph = IRGenerator::generate_from_json(cfg, &json).unwrap();

    let reused_count = static_array_read_stmt_count(&graph);

    let mut different_array_json = static_for_static_array_read_fixture(
        body_value,
        None,
        r#"{"__class__": "ASTLoad", "name": "i"}"#,
        r#"{"__class__": "ASTConstantInteger", "value": 1}"#,
    );
    replace_guard_rhs_with_static_array_read(&mut different_array_json);
    replace_body_sink_load_with_static_array_assignment_read(&mut different_array_json, "arr2");
    add_second_static_array_input(&mut different_array_json, "arr2");
    let graph_without_reuse = IRGenerator::generate_from_json(
        IRGenConfig {
            loop_limit: 32,
            ..Default::default()
        },
        &different_array_json,
    )
    .unwrap();
    let fallback_count = static_array_read_stmt_count(&graph_without_reuse);

    assert!(
        reused_count < fallback_count,
        "guard RHS read should be reused by the first identical body read; reused={reused_count}, fallback={fallback_count}"
    );
}

#[test]
fn guard_rhs_read_reuse_is_one_shot() {
    let body_value = r#"{"__class__": "ASTLoad", "name": "x"}"#;
    let mut json = static_for_static_array_read_fixture(
        body_value,
        None,
        r#"{"__class__": "ASTLoad", "name": "i"}"#,
        r#"{"__class__": "ASTConstantInteger", "value": 1}"#,
    );
    replace_guard_rhs_with_static_array_read(&mut json);
    replace_body_sink_load_with_static_array_assignment_read(&mut json, "arr");
    let first_body_read = r#"{
                                    "__class__": "ASTAssignStatement",
                                    "targets": [
                                        {
                                            "__class__": "ASTSubscriptAssignTarget",
                                            "target": {"__class__": "ASTLoad", "name": "arr"},
                                            "slicing": {
                                                "__class__": "ASTSlice",
                                                "data": [
                                                    {"__class__": "ASTLoad", "name": "j"}
                                                ]
                                            }
                                        }
                                    ],
                                    "value": {
                                        "__class__": "ASTSubscriptExp",
                                        "val": {"__class__": "ASTLoad", "name": "arr"},
                                        "slicing": {
                                            "__class__": "ASTSlice",
                                            "data": [
                                                {
                                                    "__class__": "ASTBinaryOperator",
                                                    "operator": "sub",
                                                    "lhs": {"__class__": "ASTLoad", "name": "j"},
                                                    "rhs": {"__class__": "ASTConstantInteger", "value": 1}
                                                }
                                            ]
                                        }
                                    }
                                }"#;
    let two_reads =
        format!("{first_body_read},\n                                {first_body_read}");
    json = json.replacen(first_body_read, &two_reads, 1);
    let mut cfg = IRGenConfig::default();
    cfg.loop_limit = 32;
    let graph = IRGenerator::generate_from_json(cfg, &json).unwrap();

    let two_read_count = static_array_read_stmt_count(&graph);

    let mut single_read_json = static_for_static_array_read_fixture(
        body_value,
        None,
        r#"{"__class__": "ASTLoad", "name": "i"}"#,
        r#"{"__class__": "ASTConstantInteger", "value": 1}"#,
    );
    replace_guard_rhs_with_static_array_read(&mut single_read_json);
    replace_body_sink_load_with_static_array_assignment_read(&mut single_read_json, "arr");
    let single_read_graph = IRGenerator::generate_from_json(
        IRGenConfig {
            loop_limit: 32,
            ..Default::default()
        },
        &single_read_json,
    )
    .unwrap();
    let single_read_count = static_array_read_stmt_count(&single_read_graph);

    assert!(
        two_read_count > single_read_count,
        "reuse must be one-shot; second identical body read emits its own memory read; single={single_read_count}, two={two_read_count}"
    );
}

#[test]
fn unsupported_body_prefix_statement_blocks_guard_rhs_read_reuse() {
    let body_value = r#"{"__class__": "ASTLoad", "name": "x"}"#;
    let mut json = static_for_static_array_read_fixture(
        body_value,
        None,
        r#"{"__class__": "ASTLoad", "name": "i"}"#,
        r#"{"__class__": "ASTConstantInteger", "value": 1}"#,
    );
    replace_guard_rhs_with_static_array_read(&mut json);
    replace_body_sink_load_with_static_array_assignment_read(&mut json, "arr");
    let first_body_read = r#"{
                                    "__class__": "ASTAssignStatement",
                                    "targets": [
                                        {
                                            "__class__": "ASTSubscriptAssignTarget",
                                            "target": {"__class__": "ASTLoad", "name": "arr"},
                                            "slicing": {
                                                "__class__": "ASTSlice",
                                                "data": [
                                                    {"__class__": "ASTLoad", "name": "j"}
                                                ]
                                            }
                                        }
                                    ],
                                    "value": {
                                        "__class__": "ASTSubscriptExp",
                                        "val": {"__class__": "ASTLoad", "name": "arr"},
                                        "slicing": {
                                            "__class__": "ASTSlice",
                                            "data": [
                                                {
                                                    "__class__": "ASTBinaryOperator",
                                                    "operator": "sub",
                                                    "lhs": {"__class__": "ASTLoad", "name": "j"},
                                                    "rhs": {"__class__": "ASTConstantInteger", "value": 1}
                                                }
                                            ]
                                        }
                                    }
                                }"#;
    let unsupported_prefix = format!(
        r#"{{
                                    "__class__": "ASTAssignStatement",
                                    "targets": [{{"__class__": "ASTNameAssignTarget", "name": "tmp"}}],
                                    "value": {{"__class__": "ASTLoad", "name": "x"}}
                                }},
                                {first_body_read}"#
    );
    json = json.replacen(first_body_read, &unsupported_prefix, 1);
    let mut cfg = IRGenConfig::default();
    cfg.loop_limit = 32;
    let graph = IRGenerator::generate_from_json(cfg, &json).unwrap();

    let blocked_count = static_array_read_stmt_count(&graph);

    let mut matched_json = static_for_static_array_read_fixture(
        body_value,
        None,
        r#"{"__class__": "ASTLoad", "name": "i"}"#,
        r#"{"__class__": "ASTConstantInteger", "value": 1}"#,
    );
    replace_guard_rhs_with_static_array_read(&mut matched_json);
    replace_body_sink_load_with_static_array_assignment_read(&mut matched_json, "arr");
    let matched_graph = IRGenerator::generate_from_json(
        IRGenConfig {
            loop_limit: 32,
            ..Default::default()
        },
        &matched_json,
    )
    .unwrap();
    let matched_count = static_array_read_stmt_count(&matched_graph);

    assert!(
        blocked_count > matched_count,
        "unsupported body-prefix statements must block guard/body read reuse; matched={matched_count}, blocked={blocked_count}"
    );
}

#[test]
fn same_index_different_static_array_identity_does_not_reuse_guard_rhs_read() {
    let body_value = r#"{"__class__": "ASTLoad", "name": "x"}"#;
    let mut json = static_for_static_array_read_fixture(
        body_value,
        None,
        r#"{"__class__": "ASTLoad", "name": "i"}"#,
        r#"{"__class__": "ASTConstantInteger", "value": 1}"#,
    );
    replace_guard_rhs_with_static_array_read(&mut json);
    replace_body_sink_load_with_static_array_assignment_read(&mut json, "arr2");
    add_second_static_array_input(&mut json, "arr2");
    let mut cfg = IRGenConfig::default();
    cfg.loop_limit = 32;
    let graph = IRGenerator::generate_from_json(cfg, &json).unwrap();

    let mut matched_json = static_for_static_array_read_fixture(
        body_value,
        None,
        r#"{"__class__": "ASTLoad", "name": "i"}"#,
        r#"{"__class__": "ASTConstantInteger", "value": 1}"#,
    );
    replace_guard_rhs_with_static_array_read(&mut matched_json);
    replace_body_sink_load_with_static_array_assignment_read(&mut matched_json, "arr");
    let matched_graph = IRGenerator::generate_from_json(
        IRGenConfig {
            loop_limit: 32,
            ..Default::default()
        },
        &matched_json,
    )
    .unwrap();
    let different_identity_count = static_array_read_stmt_count(&graph);
    let matched_count = static_array_read_stmt_count(&matched_graph);

    assert!(
        different_identity_count > matched_count,
        "same index over a different StaticArray identity must not reuse the guard value; matched={matched_count}, different_identity={different_identity_count}"
    );
}

#[test]
#[should_panic(expected = "loop bound depends on a private value with no provable maximum")]
/// Without a structural cap or provable bound, a private-guard `while` must be rejected, never truncated at loop_limit (it used to "fail open" to loop_limit iterations).
fn unrelated_origin_without_bound_is_rejected() {
    let init = r#"{"__class__": "ASTLoad", "name": "x"}"#;
    let decrement = r#"{"__class__": "ASTConstantInteger", "value": 1}"#;
    let mut cfg = IRGenConfig::default();
    cfg.loop_limit = 32;
    let _graph =
        IRGenerator::generate_from_json(cfg, &static_for_while_cap_fixture(init, decrement))
            .unwrap();
}

#[test]
#[should_panic(expected = "loop bound depends on a private value with no provable maximum")]
/// A non-unit decrement takes no structural cap; with no other bound the loop is rejected.
fn non_unit_decrement_without_bound_is_rejected() {
    let init = r#"{"__class__": "ASTLoad", "name": "i"}"#;
    let decrement = r#"{"__class__": "ASTConstantInteger", "value": 2}"#;
    let mut cfg = IRGenConfig::default();
    cfg.loop_limit = 32;
    let _graph =
        IRGenerator::generate_from_json(cfg, &static_for_while_cap_fixture(init, decrement))
            .unwrap();
}

#[test]
#[should_panic(expected = "loop bound depends on a private value with no provable maximum")]
/// A rebind before the decrement takes no structural cap; with no other bound the loop is rejected.
fn pre_decrement_rebind_without_bound_is_rejected() {
    let mut cfg = IRGenConfig::default();
    cfg.loop_limit = 32;
    let _graph = IRGenerator::generate_from_json(cfg, &static_for_while_prewrite_fixture()).unwrap();
}

#[test]
#[should_panic(expected = "loop bound depends on a private value with no provable maximum")]
/// `0 > j` is not a positive guard, so no structural cap applies and the loop is rejected.
fn reversed_positive_guard_without_bound_is_rejected() {
    let init = r#"{"__class__": "ASTLoad", "name": "i"}"#;
    let decrement = r#"{"__class__": "ASTConstantInteger", "value": 1}"#;
    let json = static_for_while_cap_fixture_with_iter_and_guard(
        init,
        decrement,
        &[
            r#"{"__class__": "ASTConstantInteger", "value": -1}"#,
            r#"{"__class__": "ASTConstantInteger", "value": 1}"#,
        ],
        "gt",
        r#"{"__class__": "ASTConstantInteger", "value": 0}"#,
        r#"{"__class__": "ASTLoad", "name": "j"}"#,
    );
    let mut cfg = IRGenConfig::default();
    cfg.loop_limit = 32;
    let _graph = IRGenerator::generate_from_json(cfg, &json).unwrap();
}

/// The backstop of a `while` loop the compiler stops on a fact: with the
/// fact's own check removed (a fact that does not hold), the guard is still
/// true after the last unrolled iteration, and the circuit has no witness
/// instead of proving a truncated loop.
#[test]
fn while_loop_stopped_by_a_fact_is_backstopped() {
    use crate::circuit_input::{CircuitInputs, InputNode, InputParam, ResolvedWitness};
    use crate::prove::mock::MockProverBackend;
    use crate::prove::traits::ProverBackend;
    use crate::prove::types::ProvingParams;

    // assert 0 <= n; assert n <= 3; while n * 2 > 10: n = n - 1
    // (not a counted loop: the guard's left side is no counter). The facts
    // make the guard false before the first iteration.
    let circuit = serde_json::json!({
        "__class__": "ASTCircuit",
        "inputs": [int_input("n")],
        "block": [
            assert_cmp("gte", load("n"), int_const(0)),
            assert_cmp("lte", load("n"), int_const(3)),
            {"__class__": "ASTWhileStatement",
             "test_expr": cmp_expr("gt",
                 serde_json::json!({"__class__": "ASTBinaryOperator", "operator": "mul", "lhs": load("n"), "rhs": int_const(2)}),
                 int_const(10)),
             "block": [{"__class__": "ASTAssignStatement",
                        "targets": [{"__class__": "ASTNameAssignTarget", "name": "n"}],
                        "value": {"__class__": "ASTBinaryOperator", "operator": "sub",
                                  "lhs": load("n"), "rhs": int_const(1)}}],
             "orelse": []}
        ]
    });
    let mut cfg = IRGenConfig::default();
    cfg.loop_limit = 64;
    let graph = IRGenerator::generate_from_json(cfg, &circuit.to_string()).unwrap();

    let satisfied = |graph: &crate::ir::IRGraph, n: i64| {
        let inputs = CircuitInputs {
            params: vec![InputParam {
                name: "n".into(),
                is_public: false,
                dtype: serde_json::json!("Integer"),
                value: InputNode::Int(n),
            }],
        };
        let params = ProvingParams::default();
        let witness = ResolvedWitness::new(inputs, params.precision_bits);
        MockProverBackend.prove(graph, &witness, &params).unwrap().proof_bytes == "mock_satisfied"
    };
    assert!(satisfied(&graph, 2));
    assert!(!satisfied(&graph, 5));

    // Drop the user's `n <= 3` check: the fact the loop stopped on no longer
    // holds for n = 5, and only the backstop rejects it.
    let upper_check: Vec<crate::types::StmtId> = graph
        .stmts
        .iter()
        .filter(|s| matches!(s.ir, IR::Assert))
        .filter(|s| {
            // `assert_value` asserts `select(execution, cmp, true)`.
            let mut cond = &graph.stmts[s.arguments[0] as usize];
            if matches!(cond.ir, IR::SelectI) {
                cond = &graph.stmts[cond.arguments[1] as usize];
            }
            matches!(cond.ir, IR::LteI)
                && cond.arguments.get(1).is_some_and(|&c| matches!(graph.stmts[c as usize].ir, IR::ConstantInt { value: 3 }))
        })
        .map(|s| s.stmt_id)
        .collect();
    assert_eq!(upper_check.len(), 1, "the user's `n <= 3` check");
    let mut without_fact_check = graph.clone();
    without_fact_check.remove_stmt_bunch(&upper_check);
    assert!(satisfied(&without_fact_check, 3));
    // n = 7 enters the loop (Python runs it twice); skipping it must have no
    // witness.
    assert!(!satisfied(&without_fact_check, 7), "a truncated loop must have no witness");
}
