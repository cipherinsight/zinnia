//! Tests for the mock proving backend.

use crate::circuit_input::{CircuitInputs, InputNode, InputParam, InputPath, ResolvedWitness};
use crate::ir::{IRGraph, IRStatement};
use crate::ir_defs::IR;
use crate::prove::kernel;
use crate::prove::mock::MockProverBackend;
use crate::prove::traits::ProverBackend;
use crate::prove::types::ProvingParams;

fn empty_resolved(prec: u32) -> ResolvedWitness {
    ResolvedWitness::new(CircuitInputs::new(), prec)
}

fn make_graph(stmts: Vec<(IR, Vec<u32>)>) -> IRGraph {
    let ir_stmts: Vec<IRStatement> = stmts
        .into_iter()
        .enumerate()
        .map(|(i, (ir, args))| {
            IRStatement::new(
                i as u32,
                crate::types::ValueId::next(),
                ir,
                args,
                vec![],
                None,
            )
        })
        .collect();
    IRGraph::new(ir_stmts)
}

/// ProductSumF operands are adjacent float pairs:
/// `[lhs0, rhs0, lhs1, rhs1, ...]`. The `len` payload is the number of
/// pairs, not the number of scalar operands. Raw products are accumulated
/// first, then the total is rescaled once, matching halo2-graph
/// `FixedPointChip::inner_product` and NumPy-style real arithmetic more
/// closely than per-product rounding.
fn product_sum_expected(values: &[(f64, f64)], precision_bits: u32) -> f64 {
    let pairs: Vec<_> = values
        .iter()
        .map(|(lhs, rhs)| {
            (
                kernel::quantize_to_fp(*lhs, precision_bits),
                kernel::quantize_to_fp(*rhs, precision_bits),
            )
        })
        .collect();
    let (result, _) = kernel::fp_product_sum_rescale(&pairs, precision_bits);
    kernel::dequantize_from_fp(result, precision_bits)
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
    let artifact = backend
        .prove(&ir, &empty_resolved(params.precision_bits), &params)
        .unwrap();
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
    let artifact = backend
        .prove(&ir, &empty_resolved(params.precision_bits), &params)
        .unwrap();
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
    let artifact = backend
        .prove(&ir, &empty_resolved(params.precision_bits), &params)
        .unwrap();
    assert!(artifact.proof_bytes.starts_with("mock_unsatisfied"));

    // Verify returns invalid
    let result = backend.verify(&artifact).unwrap();
    assert!(!result.valid);
}

#[test]
fn test_mock_prove_with_witness() {
    let ir = make_graph(vec![
        (
            IR::ReadInteger {
                path: InputPath::new("x", vec![]),
                is_public: false,
            },
            vec![],
        ),
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
fn test_mock_product_sum_f_matches_scalar_path() {
    let params = ProvingParams {
        precision_bits: 16,
        ..Default::default()
    };
    let pairs = [(1.25, 2.0), (-0.5, 3.5), (0.125, -4.0)];
    let expected = product_sum_expected(&pairs, params.precision_bits);
    let ir = make_graph(vec![
        (IR::ConstantFloat { value: pairs[0].0 }, vec![]),
        (IR::ConstantFloat { value: pairs[0].1 }, vec![]),
        (IR::ConstantFloat { value: pairs[1].0 }, vec![]),
        (IR::ConstantFloat { value: pairs[1].1 }, vec![]),
        (IR::ConstantFloat { value: pairs[2].0 }, vec![]),
        (IR::ConstantFloat { value: pairs[2].1 }, vec![]),
        (IR::ProductSumF { len: 3 }, vec![0, 1, 2, 3, 4, 5]),
        (IR::ConstantFloat { value: expected }, vec![]),
        (IR::EqF, vec![6, 7]),
        (IR::Assert, vec![8]),
    ]);
    let backend = MockProverBackend;
    let artifact = backend
        .prove(&ir, &empty_resolved(params.precision_bits), &params)
        .unwrap();
    assert_eq!(artifact.proof_bytes, "mock_satisfied");
}

#[test]
fn test_mock_product_sum_f_rejects_invalid_operand_shape() {
    let params = ProvingParams::default();
    let ir = make_graph(vec![
        (IR::ConstantFloat { value: 1.0 }, vec![]),
        (IR::ConstantFloat { value: 2.0 }, vec![]),
        (IR::ConstantFloat { value: 3.0 }, vec![]),
        (IR::ProductSumF { len: 2 }, vec![0, 1, 2]),
    ]);
    let backend = MockProverBackend;
    let err = backend
        .prove(&ir, &empty_resolved(params.precision_bits), &params)
        .unwrap_err();
    assert!(
        err.to_string()
            .contains("ProductSumF expected 4 operands for 2 pairs, got 3"),
        "unexpected error: {err}"
    );
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
    let params = ProvingParams {
        k: 8,
        ..Default::default()
    };
    let resolved = empty_resolved(params.precision_bits);

    // Mock backend
    let mock = MockProverBackend;
    let mock_artifact = mock.prove(&ir, &resolved, &params).unwrap();
    assert_eq!(mock_artifact.proof_bytes, "mock_satisfied");

    // Halo2 MockProver (built-in constraint checker)
    halo2::mock_prove(&ir, &resolved, &params, vec![]).unwrap();
}
