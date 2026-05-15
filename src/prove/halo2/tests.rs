//! Tests for the halo2 proving backend.
//! Each test builds a small IR circuit and verifies constraints pass the halo2 MockProver.

use crate::ir::{IRGraph, IRStatement};
use crate::ir_defs::IR;
use crate::prove::halo2::mock_prove;
use crate::prove::types::ProvingParams;
use crate::circuit_input::{CircuitInputs, InputNode, InputParam, ResolvedWitness, InputPath};

fn make_graph(stmts: Vec<(IR, Vec<u32>)>) -> IRGraph {
    let ir_stmts: Vec<IRStatement> = stmts
        .into_iter()
        .enumerate()
        .map(|(i, (ir, args))| IRStatement::new(i as u32, crate::types::ValueId::next(), ir, args, vec![], None))
        .collect();
    IRGraph::new(ir_stmts)
}

fn empty_resolved(params: &ProvingParams) -> ResolvedWitness {
    ResolvedWitness::new(CircuitInputs::new(), params.precision_bits)
}

#[test]
fn test_constant_add() {
    let ir = make_graph(vec![
        (IR::ConstantInt { value: 3 }, vec![]),
        (IR::ConstantInt { value: 5 }, vec![]),
        (IR::AddI, vec![0, 1]),
    ]);
    let params = ProvingParams { k: 5, ..Default::default() };
    mock_prove(&ir, &empty_resolved(&params), &params, vec![]).unwrap();
}

#[test]
fn test_mul() {
    let ir = make_graph(vec![
        (IR::ConstantInt { value: 4 }, vec![]),
        (IR::ConstantInt { value: 7 }, vec![]),
        (IR::MulI, vec![0, 1]),
    ]);
    let params = ProvingParams { k: 5, ..Default::default() };
    mock_prove(&ir, &empty_resolved(&params), &params, vec![]).unwrap();
}

#[test]
fn test_sub() {
    let ir = make_graph(vec![
        (IR::ConstantInt { value: 10 }, vec![]),
        (IR::ConstantInt { value: 3 }, vec![]),
        (IR::SubI, vec![0, 1]),
    ]);
    let params = ProvingParams { k: 5, ..Default::default() };
    mock_prove(&ir, &empty_resolved(&params), &params, vec![]).unwrap();
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
    let params = ProvingParams { k: 5, ..Default::default() };
    mock_prove(&ir, &empty_resolved(&params), &params, vec![]).unwrap();
}

#[test]
fn test_select() {
    let ir = make_graph(vec![
        (IR::ConstantBool { value: true }, vec![]),
        (IR::ConstantInt { value: 10 }, vec![]),
        (IR::ConstantInt { value: 20 }, vec![]),
        (IR::SelectI, vec![0, 1, 2]),
    ]);
    let params = ProvingParams { k: 5, ..Default::default() };
    mock_prove(&ir, &empty_resolved(&params), &params, vec![]).unwrap();
}

#[test]
fn test_with_input() {
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
    let params = ProvingParams { k: 8, ..Default::default() };
    let resolved = ResolvedWitness::new(witness, params.precision_bits);
    mock_prove(&ir, &resolved, &params, vec![]).unwrap();
}

#[test]
fn test_div() {
    let ir = make_graph(vec![
        (IR::ConstantInt { value: 12 }, vec![]),
        (IR::ConstantInt { value: 4 }, vec![]),
        (IR::DivI, vec![0, 1]),
    ]);
    let params = ProvingParams { k: 5, ..Default::default() };
    mock_prove(&ir, &empty_resolved(&params), &params, vec![]).unwrap();
}

#[test]
fn test_inv() {
    let ir = make_graph(vec![
        (IR::ConstantInt { value: 5 }, vec![]),
        (IR::InvI, vec![0]),
        (IR::MulI, vec![0, 1]),
    ]);
    let params = ProvingParams { k: 5, ..Default::default() };
    mock_prove(&ir, &empty_resolved(&params), &params, vec![]).unwrap();
}

#[test]
fn test_equality() {
    let ir = make_graph(vec![
        (IR::ConstantInt { value: 3 }, vec![]),
        (IR::ConstantInt { value: 3 }, vec![]),
        (IR::EqI, vec![0, 1]),
    ]);
    let params = ProvingParams { k: 6, ..Default::default() };
    mock_prove(&ir, &empty_resolved(&params), &params, vec![]).unwrap();
}

#[test]
fn test_inequality() {
    let ir = make_graph(vec![
        (IR::ConstantInt { value: 3 }, vec![]),
        (IR::ConstantInt { value: 5 }, vec![]),
        (IR::NeI, vec![0, 1]),
    ]);
    let params = ProvingParams { k: 6, ..Default::default() };
    mock_prove(&ir, &empty_resolved(&params), &params, vec![]).unwrap();
}

#[test]
fn test_floor_div() {
    let ir = make_graph(vec![
        (IR::ConstantInt { value: 7 }, vec![]),
        (IR::ConstantInt { value: 3 }, vec![]),
        (IR::FloorDivI, vec![0, 1]),
    ]);
    let params = ProvingParams { k: 5, ..Default::default() };
    mock_prove(&ir, &empty_resolved(&params), &params, vec![]).unwrap();
}

#[test]
fn test_mod() {
    let ir = make_graph(vec![
        (IR::ConstantInt { value: 7 }, vec![]),
        (IR::ConstantInt { value: 3 }, vec![]),
        (IR::ModI, vec![0, 1]),
    ]);
    let params = ProvingParams { k: 5, ..Default::default() };
    mock_prove(&ir, &empty_resolved(&params), &params, vec![]).unwrap();
}

#[test]
fn test_pow() {
    let ir = make_graph(vec![
        (IR::ConstantInt { value: 2 }, vec![]),
        (IR::ConstantInt { value: 3 }, vec![]),
        (IR::PowI, vec![0, 1]),
    ]);
    let params = ProvingParams { k: 8, ..Default::default() };
    mock_prove(&ir, &empty_resolved(&params), &params, vec![]).unwrap();
}

#[test]
fn test_assert_with_eq() {
    let ir = make_graph(vec![
        (IR::ConstantInt { value: 3 }, vec![]),
        (IR::ConstantInt { value: 3 }, vec![]),
        (IR::EqI, vec![0, 1]),
        (IR::Assert, vec![2]),
    ]);
    let params = ProvingParams { k: 6, ..Default::default() };
    mock_prove(&ir, &empty_resolved(&params), &params, vec![]).unwrap();
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
    let params = ProvingParams { k: 5, ..Default::default() };
    mock_prove(&ir, &empty_resolved(&params), &params, vec![]).unwrap();
}

#[test]
fn test_lt() {
    let ir = make_graph(vec![
        (IR::ConstantInt { value: 3 }, vec![]),
        (IR::ConstantInt { value: 5 }, vec![]),
        (IR::LtI, vec![0, 1]),
    ]);
    let params = ProvingParams { k: 7, ..Default::default() };
    mock_prove(&ir, &empty_resolved(&params), &params, vec![]).unwrap();
}

#[test]
fn test_sign() {
    let ir = make_graph(vec![
        (IR::ConstantInt { value: 5 }, vec![]),
        (IR::SignI, vec![0]),
    ]);
    let params = ProvingParams { k: 8, ..Default::default() };
    mock_prove(&ir, &empty_resolved(&params), &params, vec![]).unwrap();
}

#[test]
fn test_abs() {
    let ir = make_graph(vec![
        (IR::ConstantInt { value: -7 }, vec![]),
        (IR::AbsI, vec![0]),
    ]);
    let params = ProvingParams { k: 6, ..Default::default() };
    mock_prove(&ir, &empty_resolved(&params), &params, vec![]).unwrap();
}

#[test]
fn test_exp_constrained() {
    let ir = make_graph(vec![
        (IR::ConstantFloat { value: 1.0 }, vec![]),
        (IR::ExpF, vec![0]),
    ]);
    let params = ProvingParams { k: 12, ..Default::default() };
    mock_prove(&ir, &empty_resolved(&params), &params, vec![]).unwrap();
}

#[test]
fn test_sin_constrained() {
    let ir = make_graph(vec![
        (IR::ConstantFloat { value: 0.5 }, vec![]),
        (IR::SinF, vec![0]),
    ]);
    let params = ProvingParams { k: 14, ..Default::default() };
    mock_prove(&ir, &empty_resolved(&params), &params, vec![]).unwrap();
}

#[test]
fn test_poseidon_constrained() {
    let ir = make_graph(vec![
        (IR::ConstantInt { value: 1 }, vec![]),
        (IR::ConstantInt { value: 2 }, vec![]),
        (IR::PoseidonHash, vec![0, 1]),
    ]);
    let params = ProvingParams { k: 12, ..Default::default() };
    mock_prove(&ir, &empty_resolved(&params), &params, vec![]).unwrap();
}

#[test]
fn test_bool_cast() {
    let ir = make_graph(vec![
        (IR::ConstantInt { value: 5 }, vec![]),
        (IR::BoolCast, vec![0]),
    ]);
    let params = ProvingParams { k: 6, ..Default::default() };
    mock_prove(&ir, &empty_resolved(&params), &params, vec![]).unwrap();
}

#[test]
fn test_float_mul_constrained() {
    let ir = make_graph(vec![
        (IR::ConstantFloat { value: 1.5 }, vec![]),
        (IR::ConstantFloat { value: 2.0 }, vec![]),
        (IR::MulF, vec![0, 1]),
    ]);
    // MulF now emits a precision_bits-wide range-check on the div_mod remainder,
    // which expands the row count beyond k=6.
    let params = ProvingParams { k: 8, ..Default::default() };
    mock_prove(&ir, &empty_resolved(&params), &params, vec![]).unwrap();
}

// ── Bitwise ops (constrained via 64-bit two's-complement decomposition) ──

#[test]
fn test_bit_and_const() {
    let ir = make_graph(vec![
        (IR::ConstantInt { value: 0b1100 }, vec![]),
        (IR::ConstantInt { value: 0b1010 }, vec![]),
        (IR::BitAndI, vec![0, 1]),
        (IR::ConstantInt { value: 0b1000 }, vec![]),
        (IR::EqI, vec![2, 3]),
        (IR::Assert, vec![4]),
    ]);
    let params = ProvingParams { k: 14, ..Default::default() };
    mock_prove(&ir, &empty_resolved(&params), &params, vec![]).unwrap();
}

#[test]
fn test_bit_or_const() {
    let ir = make_graph(vec![
        (IR::ConstantInt { value: 0b1100 }, vec![]),
        (IR::ConstantInt { value: 0b1010 }, vec![]),
        (IR::BitOrI, vec![0, 1]),
        (IR::ConstantInt { value: 0b1110 }, vec![]),
        (IR::EqI, vec![2, 3]),
        (IR::Assert, vec![4]),
    ]);
    let params = ProvingParams { k: 14, ..Default::default() };
    mock_prove(&ir, &empty_resolved(&params), &params, vec![]).unwrap();
}

#[test]
fn test_bit_xor_const() {
    let ir = make_graph(vec![
        (IR::ConstantInt { value: 0b1100 }, vec![]),
        (IR::ConstantInt { value: 0b1010 }, vec![]),
        (IR::BitXorI, vec![0, 1]),
        (IR::ConstantInt { value: 0b0110 }, vec![]),
        (IR::EqI, vec![2, 3]),
        (IR::Assert, vec![4]),
    ]);
    let params = ProvingParams { k: 14, ..Default::default() };
    mock_prove(&ir, &empty_resolved(&params), &params, vec![]).unwrap();
}

#[test]
fn test_bit_not_const() {
    // ~5 = -6 under two's-complement i64 semantics.
    let ir = make_graph(vec![
        (IR::ConstantInt { value: 5 }, vec![]),
        (IR::BitNotI, vec![0]),
        (IR::ConstantInt { value: -6 }, vec![]),
        (IR::EqI, vec![1, 2]),
        (IR::Assert, vec![3]),
    ]);
    let params = ProvingParams { k: 14, ..Default::default() };
    mock_prove(&ir, &empty_resolved(&params), &params, vec![]).unwrap();
}

#[test]
fn test_shl_const() {
    let ir = make_graph(vec![
        (IR::ConstantInt { value: 1 }, vec![]),
        (IR::ConstantInt { value: 4 }, vec![]),
        (IR::ShlI, vec![0, 1]),
        (IR::ConstantInt { value: 16 }, vec![]),
        (IR::EqI, vec![2, 3]),
        (IR::Assert, vec![4]),
    ]);
    let params = ProvingParams { k: 14, ..Default::default() };
    mock_prove(&ir, &empty_resolved(&params), &params, vec![]).unwrap();
}

#[test]
fn test_shr_const() {
    let ir = make_graph(vec![
        (IR::ConstantInt { value: 16 }, vec![]),
        (IR::ConstantInt { value: 2 }, vec![]),
        (IR::ShrI, vec![0, 1]),
        (IR::ConstantInt { value: 4 }, vec![]),
        (IR::EqI, vec![2, 3]),
        (IR::Assert, vec![4]),
    ]);
    let params = ProvingParams { k: 14, ..Default::default() };
    mock_prove(&ir, &empty_resolved(&params), &params, vec![]).unwrap();
}

// ── Transcendental value correctness ───────────────────────────────────
//
// These tests are the load-bearing regression guard for the layered defects
// fixed in this card:
//   1. `poly_eval_horner` constant-first iteration on leading-first coefs.
//   2. No range reduction (polynomial evaluated outside its fit domain).
//   3. Q32 wrap on deep Horner accumulator.
//
// Each test brackets the halo2-computed value against the numpy/f64 reference
// using the (lo, hi) gate pattern from the diagnosis card. A bracket fail
// means a value-level regression that MockProver will surface on the assert
// gate; running these under `cargo test` exercises every layer end-to-end.

/// Build an IR of the form:
///   ConstantFloat(x) → OP → ConstantFloat(lo), ConstantFloat(hi)
///                              ↓                ↓
///                              GtF              LtF
///                                  ↘            ↙
///                                   LogicalAnd
///                                       ↓
///                                     Assert
///
/// Asserts that `OP(x) ∈ (lo, hi)`. Fails MockProver verification if not.
fn assert_op_in_bracket(op: IR, x: f64, lo: f64, hi: f64, k: u32) {
    let ir = make_graph(vec![
        (IR::ConstantFloat { value: x }, vec![]),
        (op, vec![0]),
        (IR::ConstantFloat { value: lo }, vec![]),
        (IR::ConstantFloat { value: hi }, vec![]),
        (IR::GtF, vec![1, 2]),
        (IR::LtF, vec![1, 3]),
        (IR::LogicalAnd, vec![4, 5]),
        (IR::Assert, vec![6]),
    ]);
    let params = ProvingParams { k, ..Default::default() };
    mock_prove(&ir, &empty_resolved(&params), &params, vec![]).unwrap();
}

#[test]
fn halo2_log_value_correctness() {
    // For inputs where the mantissa `m = x * 2^{-k}` lands at exactly 2.0
    // (i.e., x is itself a power of two), the polynomial evaluates at the
    // left endpoint of its fit domain and is accurate to ~1e-6. For
    // arbitrary x (e.g. 6.77 → m ≈ 3.385), Q32 truncation accumulates
    // across the 14-step Horner chain to ~3e-3.
    //
    // Bracket widths reflect these observed bounds:
    //   log(2.0)  ≈ 0.6931 → (0.692, 0.694)   tight, m=2 exactly
    //   log(8.0)  ≈ 2.0794 → (2.078, 2.080)   tight, m=2 exactly
    //   log(6.77) ≈ 1.9125 → (1.907, 1.913)   loose, m=3.385
    assert_op_in_bracket(IR::LogF, 2.0, 0.692, 0.694, 14);
    assert_op_in_bracket(IR::LogF, 8.0, 2.078, 2.080, 14);
    assert_op_in_bracket(IR::LogF, 6.77, 1.907, 1.913, 14);
}

#[test]
fn halo2_exp_value_correctness() {
    // exp2 polynomial on [0, 1) preserves precision well in Q32 (coefs do
    // not have a tiny leading term — the smallest is 3.6e-11 in scaled
    // arithmetic but the accumulator magnitudes stay close to 1). Errors
    // are dominated by the ln(2) Q-rescaling and 2^i lift, ~1e-6 overall.
    //   exp(0.5) ≈ 1.6487 → (1.6485, 1.6489)
    //   exp(1.5) ≈ 4.4817 → (4.4815, 4.4819)
    assert_op_in_bracket(IR::ExpF, 0.5, 1.6485, 1.6489, 14);
    assert_op_in_bracket(IR::ExpF, 1.5, 4.4815, 4.4819, 14);
}

#[test]
fn halo2_sqrt_value_correctness() {
    // sqrt is composed as exp(0.5 * log(x)). For x a power of two, log(x)
    // is computed at m=2 exactly so the chain is precise; sqrt error is
    // dominated by exp's Q-rescaling, ~1e-5.
    //   sqrt(2.0) ≈ 1.4142 → (1.4140, 1.4144)
    //   sqrt(8.0) ≈ 2.8284 → (2.8283, 2.8286)
    assert_op_in_bracket(IR::SqrtF, 2.0, 1.4140, 1.4144, 14);
    assert_op_in_bracket(IR::SqrtF, 8.0, 2.8283, 2.8286, 14);
}

#[test]
fn test_bitwise_with_negative() {
    // -1 & 0xF = 0xF (the low 4 bits of -1 are all set).
    let ir = make_graph(vec![
        (IR::ConstantInt { value: -1 }, vec![]),
        (IR::ConstantInt { value: 0xF }, vec![]),
        (IR::BitAndI, vec![0, 1]),
        (IR::ConstantInt { value: 0xF }, vec![]),
        (IR::EqI, vec![2, 3]),
        (IR::Assert, vec![4]),
    ]);
    let params = ProvingParams { k: 14, ..Default::default() };
    mock_prove(&ir, &empty_resolved(&params), &params, vec![]).unwrap();
}

// ── End-to-end prove() / verify() coverage ─────────────────────────────
//
// These tests exercise `Halo2ProverBackend` (not just `mock_prove`)
// to guard the gaps documented in
// `kanban/cards/compiler/halo2-prove-and-verify-no-op-gaps/`:
//
//   1. prove() must reject witnesses that don't satisfy the circuit.
//   2. prove() → verify() must round-trip a valid proof to Ok.
//   3. verify() must reject a tampered proof.

use crate::prove::error::ProvingError;
use crate::prove::halo2::Halo2ProverBackend;
use crate::prove::traits::ProverBackend;

#[test]
fn halo2_prove_rejects_unsat_witness() {
    // assert(false) — a constant-false predicate flowing into Assert.
    // Without the MockProver validation step inside prove(), this would
    // produce a valid-looking proof artifact for an unsatisfiable
    // circuit.
    let ir = make_graph(vec![
        (IR::ConstantBool { value: false }, vec![]),
        (IR::Assert, vec![0]),
    ]);
    let params = ProvingParams { k: 5, ..Default::default() };
    let backend = Halo2ProverBackend;
    let result = backend.prove(&ir, &empty_resolved(&params), &params);
    match result {
        Err(ProvingError::ProvingFailed { detail }) => {
            assert!(
                detail.contains("Witness does not satisfy")
                    || detail.contains("ConstraintNotSatisfied"),
                "expected constraint-not-satisfied error, got: {}",
                detail
            );
        }
        Err(other) => panic!("expected ProvingFailed, got: {:?}", other),
        Ok(_) => panic!("expected prove() to reject the unsatisfiable witness"),
    }
}

#[test]
fn halo2_prove_then_verify_in_process_roundtrip() {
    // A trivially satisfiable circuit: assert(3 == 3).
    let ir = make_graph(vec![
        (IR::ConstantInt { value: 3 }, vec![]),
        (IR::ConstantInt { value: 3 }, vec![]),
        (IR::EqI, vec![0, 1]),
        (IR::Assert, vec![2]),
    ]);
    let params = ProvingParams { k: 6, ..Default::default() };
    let backend = Halo2ProverBackend;

    let artifact = backend
        .prove(&ir, &empty_resolved(&params), &params)
        .expect("prove() should succeed on a satisfiable circuit");
    assert_eq!(artifact.backend, "halo2-ipa");
    assert!(!artifact.proof_bytes.is_empty());
    assert!(!artifact.vk_bytes.is_empty(), "vk_bytes should carry the in-process VK handle");

    let result = backend
        .verify(&artifact)
        .expect("verify() should not error on a well-formed artifact");
    assert!(
        result.valid,
        "expected verify() to accept a freshly-produced proof, got: {:?}",
        result.error
    );
}

#[test]
fn halo2_verify_rejects_tampered_proof() {
    let ir = make_graph(vec![
        (IR::ConstantInt { value: 3 }, vec![]),
        (IR::ConstantInt { value: 3 }, vec![]),
        (IR::EqI, vec![0, 1]),
        (IR::Assert, vec![2]),
    ]);
    let params = ProvingParams { k: 6, ..Default::default() };
    let backend = Halo2ProverBackend;

    let mut artifact = backend
        .prove(&ir, &empty_resolved(&params), &params)
        .expect("prove() should succeed");

    // Flip a byte in the middle of the transcript. We pick a byte well
    // past the start of the hex string to avoid touching length-coded
    // prefixes (if any) — the goal is to corrupt a commitment in the
    // body of the proof so verify_proof rejects it.
    let mut bytes = hex_decode_for_test(&artifact.proof_bytes);
    let mid = bytes.len() / 2;
    bytes[mid] ^= 0xFF;
    artifact.proof_bytes = hex_encode_for_test(&bytes);

    let result = backend
        .verify(&artifact)
        .expect("verify() should return Ok(VerifyResult), not error");
    assert!(
        !result.valid,
        "expected tampered proof to be rejected, got valid=true"
    );
}

// Local hex helpers for tests (the ones in mod.rs are private to the module).
fn hex_decode_for_test(s: &str) -> Vec<u8> {
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).expect("test hex"))
        .collect()
}

fn hex_encode_for_test(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}
