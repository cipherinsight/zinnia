//! Tests for the mock proving backend.

use crate::ir::{IRGraph, IRStatement};
use crate::ir_defs::IR;
use crate::prove::mock::MockProverBackend;
use crate::prove::traits::ProverBackend;
use crate::prove::types::ProvingParams;
use crate::circuit_input::{CircuitInputs, InputNode, InputParam, ResolvedWitness, InputPath, PathSegment};

fn empty_resolved(prec: u32) -> ResolvedWitness {
    ResolvedWitness::new(CircuitInputs::new(), prec)
}

fn make_graph(stmts: Vec<(IR, Vec<u32>)>) -> IRGraph {
    let ir_stmts: Vec<IRStatement> = stmts
        .into_iter()
        .enumerate()
        .map(|(i, (ir, args))| IRStatement::new(i as u32, ir, args, None))
        .collect();
    IRGraph::new(ir_stmts)
}

#[test]
fn test_mock_backend_name() {
    let backend = MockProverBackend;
    assert_eq!(backend.name(), "mock");
}

#[test]
fn test_mock_prove_simple_add() {
    let ir = make_graph(vec![
        (IR::ConstantInt { value: 3 }, vec![]),
        (IR::ConstantInt { value: 5 }, vec![]),
        (IR::AddI, vec![0, 1]),
    ]);
    let backend = MockProverBackend;
    let params = ProvingParams::default();
    let artifact = backend.prove(&ir, &empty_resolved(params.precision_bits), &params).unwrap();
    assert_eq!(artifact.backend, "mock");
    assert_eq!(artifact.proof_bytes, "mock_satisfied");
}

#[test]
fn test_mock_prove_assert_pass() {
    let ir = make_graph(vec![
        (IR::ConstantInt { value: 3 }, vec![]),
        (IR::ConstantInt { value: 3 }, vec![]),
        (IR::EqI, vec![0, 1]),
        (IR::Assert, vec![2]),
    ]);
    let backend = MockProverBackend;
    let params = ProvingParams::default();
    let artifact = backend.prove(&ir, &empty_resolved(params.precision_bits), &params).unwrap();
    assert_eq!(artifact.proof_bytes, "mock_satisfied");

    // Verify returns ok
    let result = backend.verify(&artifact).unwrap();
    assert!(result.valid);
}

#[test]
fn test_mock_prove_assert_fail() {
    let ir = make_graph(vec![
        (IR::ConstantInt { value: 3 }, vec![]),
        (IR::ConstantInt { value: 5 }, vec![]),
        (IR::EqI, vec![0, 1]),
        (IR::Assert, vec![2]),
    ]);
    let backend = MockProverBackend;
    let params = ProvingParams::default();
    let artifact = backend.prove(&ir, &empty_resolved(params.precision_bits), &params).unwrap();
    assert!(artifact.proof_bytes.starts_with("mock_unsatisfied"));

    // Verify returns invalid
    let result = backend.verify(&artifact).unwrap();
    assert!(!result.valid);
}

#[test]
fn test_mock_prove_with_witness() {
    let ir = make_graph(vec![
        (IR::ReadInteger { path: InputPath::new("x", vec![]), is_public: false }, vec![]),
        (IR::ConstantInt { value: 5 }, vec![]),
        (IR::AddI, vec![0, 1]),
    ]);
    let witness = CircuitInputs {
        params: vec![InputParam {
            name: "x".to_string(),
            is_public: false,
            dtype: serde_json::json!("Integer"),
            value: InputNode::Int(42),
        }],
    };
    let backend = MockProverBackend;
    let params = ProvingParams::default();
    let resolved = ResolvedWitness::new(witness, params.precision_bits);
    let artifact = backend.prove(&ir, &resolved, &params).unwrap();
    assert_eq!(artifact.proof_bytes, "mock_satisfied");
}

#[test]
fn test_mock_prove_memory() {
    let ir = make_graph(vec![
        (IR::AllocateMemory { segment_id: 0, size: 4, init_value: 0 }, vec![]),
        (IR::ConstantInt { value: 0 }, vec![]),
        (IR::ConstantInt { value: 42 }, vec![]),
        (IR::WriteMemory { segment_id: 0 }, vec![1, 2]),
        (IR::ReadMemory { segment_id: 0 }, vec![1]),
    ]);
    let backend = MockProverBackend;
    let params = ProvingParams::default();
    let artifact = backend.prove(&ir, &empty_resolved(params.precision_bits), &params).unwrap();
    assert_eq!(artifact.proof_bytes, "mock_satisfied");
}

/// Test that mock and halo2 produce the same result for a simple circuit.
#[test]
fn test_mock_halo2_consistency_add() {
    use crate::prove::halo2;

    let ir = make_graph(vec![
        (IR::ConstantInt { value: 7 }, vec![]),
        (IR::ConstantInt { value: 11 }, vec![]),
        (IR::AddI, vec![0, 1]),
        (IR::ConstantInt { value: 18 }, vec![]),
        (IR::EqI, vec![2, 3]),
        (IR::Assert, vec![4]),
    ]);
    let params = ProvingParams { k: 8, ..Default::default() };
    let resolved = empty_resolved(params.precision_bits);

    // Mock backend
    let mock = MockProverBackend;
    let mock_artifact = mock.prove(&ir, &resolved, &params).unwrap();
    assert_eq!(mock_artifact.proof_bytes, "mock_satisfied");

    // Halo2 MockProver (built-in constraint checker)
    halo2::mock_prove(&ir, &resolved, &params, vec![]).unwrap();
}
