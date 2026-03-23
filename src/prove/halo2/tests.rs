//! Tests for the halo2 proving backend.
//! Each test builds a small IR circuit and verifies constraints pass the halo2 MockProver.

use crate::ir::{IRGraph, IRStatement};
use crate::ir_defs::IR;
use crate::prove::halo2::mock_prove;
use crate::prove::types::{ProvingParams, Value, WitnessInput};

fn make_graph(stmts: Vec<(IR, Vec<u32>)>) -> IRGraph {
    let ir_stmts: Vec<IRStatement> = stmts
        .into_iter()
        .enumerate()
        .map(|(i, (ir, args))| IRStatement::new(i as u32, ir, args, None))
        .collect();
    IRGraph::new(ir_stmts)
}

#[test]
fn test_constant_add() {
    let ir = make_graph(vec![
        (IR::ConstantInt { value: 3 }, vec![]),
        (IR::ConstantInt { value: 5 }, vec![]),
        (IR::AddI, vec![0, 1]),
    ]);
    mock_prove(&ir, &WitnessInput::new(), &ProvingParams { k: 5, ..Default::default() }, vec![]).unwrap();
}

#[test]
fn test_mul() {
    let ir = make_graph(vec![
        (IR::ConstantInt { value: 4 }, vec![]),
        (IR::ConstantInt { value: 7 }, vec![]),
        (IR::MulI, vec![0, 1]),
    ]);
    mock_prove(&ir, &WitnessInput::new(), &ProvingParams { k: 5, ..Default::default() }, vec![]).unwrap();
}

#[test]
fn test_sub() {
    let ir = make_graph(vec![
        (IR::ConstantInt { value: 10 }, vec![]),
        (IR::ConstantInt { value: 3 }, vec![]),
        (IR::SubI, vec![0, 1]),
    ]);
    mock_prove(&ir, &WitnessInput::new(), &ProvingParams { k: 5, ..Default::default() }, vec![]).unwrap();
}

#[test]
fn test_boolean_logic() {
    let ir = make_graph(vec![
        (IR::ConstantBool { value: true }, vec![]),
        (IR::ConstantBool { value: false }, vec![]),
        (IR::LogicalAnd, vec![0, 1]),
        (IR::LogicalNot, vec![1]),
        (IR::LogicalOr, vec![0, 1]),
    ]);
    mock_prove(&ir, &WitnessInput::new(), &ProvingParams { k: 5, ..Default::default() }, vec![]).unwrap();
}

#[test]
fn test_select() {
    let ir = make_graph(vec![
        (IR::ConstantBool { value: true }, vec![]),
        (IR::ConstantInt { value: 10 }, vec![]),
        (IR::ConstantInt { value: 20 }, vec![]),
        (IR::SelectI, vec![0, 1, 2]),
    ]);
    mock_prove(&ir, &WitnessInput::new(), &ProvingParams { k: 5, ..Default::default() }, vec![]).unwrap();
}

#[test]
fn test_with_input() {
    let ir = make_graph(vec![
        (IR::ReadInteger { indices: vec![0, 0], is_public: false }, vec![]),
        (IR::ConstantInt { value: 5 }, vec![]),
        (IR::AddI, vec![0, 1]),
    ]);
    let mut witness = WitnessInput::new();
    witness.entries.push(("0_0".to_string(), Value::Integer(42)));
    mock_prove(&ir, &witness, &ProvingParams { k: 8, ..Default::default() }, vec![]).unwrap();
}

#[test]
fn test_div() {
    let ir = make_graph(vec![
        (IR::ConstantInt { value: 12 }, vec![]),
        (IR::ConstantInt { value: 4 }, vec![]),
        (IR::DivI, vec![0, 1]),
    ]);
    mock_prove(&ir, &WitnessInput::new(), &ProvingParams { k: 5, ..Default::default() }, vec![]).unwrap();
}

#[test]
fn test_inv() {
    let ir = make_graph(vec![
        (IR::ConstantInt { value: 5 }, vec![]),
        (IR::InvI, vec![0]),
        (IR::MulI, vec![0, 1]),
    ]);
    mock_prove(&ir, &WitnessInput::new(), &ProvingParams { k: 5, ..Default::default() }, vec![]).unwrap();
}

#[test]
fn test_equality() {
    let ir = make_graph(vec![
        (IR::ConstantInt { value: 3 }, vec![]),
        (IR::ConstantInt { value: 3 }, vec![]),
        (IR::EqI, vec![0, 1]),
    ]);
    mock_prove(&ir, &WitnessInput::new(), &ProvingParams { k: 6, ..Default::default() }, vec![]).unwrap();
}

#[test]
fn test_inequality() {
    let ir = make_graph(vec![
        (IR::ConstantInt { value: 3 }, vec![]),
        (IR::ConstantInt { value: 5 }, vec![]),
        (IR::NeI, vec![0, 1]),
    ]);
    mock_prove(&ir, &WitnessInput::new(), &ProvingParams { k: 6, ..Default::default() }, vec![]).unwrap();
}

#[test]
fn test_floor_div() {
    let ir = make_graph(vec![
        (IR::ConstantInt { value: 7 }, vec![]),
        (IR::ConstantInt { value: 3 }, vec![]),
        (IR::FloorDivI, vec![0, 1]),
    ]);
    mock_prove(&ir, &WitnessInput::new(), &ProvingParams { k: 5, ..Default::default() }, vec![]).unwrap();
}

#[test]
fn test_mod() {
    let ir = make_graph(vec![
        (IR::ConstantInt { value: 7 }, vec![]),
        (IR::ConstantInt { value: 3 }, vec![]),
        (IR::ModI, vec![0, 1]),
    ]);
    mock_prove(&ir, &WitnessInput::new(), &ProvingParams { k: 5, ..Default::default() }, vec![]).unwrap();
}

#[test]
fn test_pow() {
    let ir = make_graph(vec![
        (IR::ConstantInt { value: 2 }, vec![]),
        (IR::ConstantInt { value: 3 }, vec![]),
        (IR::PowI, vec![0, 1]),
    ]);
    mock_prove(&ir, &WitnessInput::new(), &ProvingParams { k: 8, ..Default::default() }, vec![]).unwrap();
}

#[test]
fn test_assert_with_eq() {
    let ir = make_graph(vec![
        (IR::ConstantInt { value: 3 }, vec![]),
        (IR::ConstantInt { value: 3 }, vec![]),
        (IR::EqI, vec![0, 1]),
        (IR::Assert, vec![2]),
    ]);
    mock_prove(&ir, &WitnessInput::new(), &ProvingParams { k: 6, ..Default::default() }, vec![]).unwrap();
}

#[test]
fn test_memory() {
    let ir = make_graph(vec![
        (IR::AllocateMemory { segment_id: 0, size: 4, init_value: 0 }, vec![]),
        (IR::ConstantInt { value: 0 }, vec![]),
        (IR::ConstantInt { value: 42 }, vec![]),
        (IR::WriteMemory { segment_id: 0 }, vec![1, 2]),
        (IR::ReadMemory { segment_id: 0 }, vec![1]),
    ]);
    mock_prove(&ir, &WitnessInput::new(), &ProvingParams { k: 5, ..Default::default() }, vec![]).unwrap();
}

#[test]
fn test_lt() {
    let ir = make_graph(vec![
        (IR::ConstantInt { value: 3 }, vec![]),
        (IR::ConstantInt { value: 5 }, vec![]),
        (IR::LtI, vec![0, 1]),
    ]);
    mock_prove(&ir, &WitnessInput::new(), &ProvingParams { k: 7, ..Default::default() }, vec![]).unwrap();
}

#[test]
fn test_sign() {
    let ir = make_graph(vec![
        (IR::ConstantInt { value: 5 }, vec![]),
        (IR::SignI, vec![0]),
    ]);
    mock_prove(&ir, &WitnessInput::new(), &ProvingParams { k: 8, ..Default::default() }, vec![]).unwrap();
}

#[test]
fn test_abs() {
    let ir = make_graph(vec![
        (IR::ConstantInt { value: -7 }, vec![]),
        (IR::AbsI, vec![0]),
    ]);
    mock_prove(&ir, &WitnessInput::new(), &ProvingParams { k: 6, ..Default::default() }, vec![]).unwrap();
}

#[test]
fn test_exp_constrained() {
    let ir = make_graph(vec![
        (IR::ConstantFloat { value: 1.0 }, vec![]),
        (IR::ExpF, vec![0]),
    ]);
    mock_prove(&ir, &WitnessInput::new(), &ProvingParams { k: 12, ..Default::default() }, vec![]).unwrap();
}

#[test]
fn test_sin_constrained() {
    let ir = make_graph(vec![
        (IR::ConstantFloat { value: 0.5 }, vec![]),
        (IR::SinF, vec![0]),
    ]);
    mock_prove(&ir, &WitnessInput::new(), &ProvingParams { k: 14, ..Default::default() }, vec![]).unwrap();
}

#[test]
fn test_poseidon_constrained() {
    let ir = make_graph(vec![
        (IR::ConstantInt { value: 1 }, vec![]),
        (IR::ConstantInt { value: 2 }, vec![]),
        (IR::PoseidonHash, vec![0, 1]),
    ]);
    mock_prove(&ir, &WitnessInput::new(), &ProvingParams { k: 12, ..Default::default() }, vec![]).unwrap();
}

#[test]
fn test_bool_cast() {
    let ir = make_graph(vec![
        (IR::ConstantInt { value: 5 }, vec![]),
        (IR::BoolCast, vec![0]),
    ]);
    mock_prove(&ir, &WitnessInput::new(), &ProvingParams { k: 6, ..Default::default() }, vec![]).unwrap();
}

#[test]
fn test_float_mul_constrained() {
    let ir = make_graph(vec![
        (IR::ConstantFloat { value: 1.5 }, vec![]),
        (IR::ConstantFloat { value: 2.0 }, vec![]),
        (IR::MulF, vec![0, 1]),
    ]);
    mock_prove(&ir, &WitnessInput::new(), &ProvingParams { k: 6, ..Default::default() }, vec![]).unwrap();
}
