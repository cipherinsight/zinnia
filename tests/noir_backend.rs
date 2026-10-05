//! Integration tests for the Noir source-to-source backend.
//!
//! Lives under `tests/` (separate crate) so it does NOT pull in the
//! pre-existing in-lib tests, which currently fail to compile against
//! halo2 0.3.x for reasons unrelated to this work.
//!
//! Gated on the `noir-backend` feature. If `nargo`/`bb` are missing the
//! end-to-end test prints a skip notice and exits.

#![cfg(feature = "noir-backend")]

use _zinnia_core::circuit_input::{
    CircuitInputs, InputNode, InputParam, InputPath, PathSegment, ResolvedWitness,
};
use _zinnia_core::ir::{IRGraph, IRStatement};
use _zinnia_core::ir_defs::IR;
use _zinnia_core::prove::error::ProvingError;
use _zinnia_core::prove::noir::emit::emit_circuit;
use _zinnia_core::prove::noir::NoirProverBackend;
use _zinnia_core::prove::traits::ProverBackend;
use _zinnia_core::prove::types::ProvingParams;
use _zinnia_core::types::ValueId;

fn make_three_float_add_ir() -> IRGraph {
    let path = |i: u32| InputPath::new("a", vec![PathSegment::Index(i)]);
    let v = [
        ValueId::next(),
        ValueId::next(),
        ValueId::next(),
        ValueId::next(),
        ValueId::next(),
    ];
    let stmts = vec![
        IRStatement::new(
            0,
            v[0],
            IR::ReadFloat {
                path: path(0),
                is_public: false,
            },
            vec![],
            vec![],
            None,
        ),
        IRStatement::new(
            1,
            v[1],
            IR::ReadFloat {
                path: path(1),
                is_public: false,
            },
            vec![],
            vec![],
            None,
        ),
        IRStatement::new(
            2,
            v[2],
            IR::ReadFloat {
                path: path(2),
                is_public: false,
            },
            vec![],
            vec![],
            None,
        ),
        IRStatement::new(3, v[3], IR::AddF, vec![0, 1], vec![v[0], v[1]], None),
        IRStatement::new(4, v[4], IR::AddF, vec![3, 2], vec![v[3], v[2]], None),
        IRStatement::new(
            5,
            ValueId::next(),
            IR::ExposePublicF,
            vec![4],
            vec![v[4]],
            None,
        ),
    ];
    IRGraph::new(stmts)
}

fn three_float_witness() -> ResolvedWitness {
    let mut inputs = CircuitInputs::new();
    inputs.params.push(InputParam {
        name: "a".to_string(),
        is_public: false,
        dtype: serde_json::Value::Null,
        value: InputNode::Array {
            shape: vec![3],
            elements: vec![
                InputNode::Float(1.0),
                InputNode::Float(2.0),
                InputNode::Float(3.0),
            ],
            runtime_shape: None,
        },
    });
    ResolvedWitness::new(inputs, 32)
}

#[test]
fn test_emit_three_float_add_text_shape() {
    let ir = make_three_float_add_ir();
    let w = three_float_witness();
    let params = ProvingParams::default();
    let out = emit_circuit(&ir, &w, &params).expect("emit");
    eprintln!("--- main.nr ---\n{}", out.main_nr);
    eprintln!("--- Prover.toml ---\n{}", out.prover_toml);
    assert!(out.main_nr.contains("fn main(a: [Field; 3])"));
    assert!(out.main_nr.contains("qadd"));
    assert!(out.main_nr.contains("-> pub Field"));
    assert!(out.prover_toml.contains("a = ["));
    assert_eq!(out.return_count, 1);
    assert_eq!(out.public_input_count, 0);
    // 6.0 * 2^32 = 25_769_803_776 = 0x600000000
    if let Some(vals) = execute_if_toolchain(&ir, &w) {
        assert_eq!(vals, vec![format!("0x{:064x}", 25_769_803_776u64)]);
    }
}

/// `nargo execute` the circuit when the toolchain is installed.
fn execute_if_toolchain(ir: &IRGraph, w: &ResolvedWitness) -> Option<Vec<String>> {
    use _zinnia_core::prove::noir::driver::DriverPaths;
    if !DriverPaths::default().nargo.exists() {
        eprintln!("skip execution: nargo missing");
        return None;
    }
    Some(
        NoirProverBackend
            .execute(ir, w, &ProvingParams::default())
            .expect("nargo execute"),
    )
}

fn field_hex(v: u64) -> String {
    format!("0x{:064x}", v)
}

// ── C16a: precision_bits != 32 is rejected honestly (Q32 is hardcoded) ──
#[test]
fn test_c16a_non32_precision_bits_rejected() {
    let ir = make_three_float_add_ir();
    let w = three_float_witness();
    let mut params = ProvingParams::default();
    params.precision_bits = 16;
    let err = emit_circuit(&ir, &w, &params)
        .err()
        .expect("non-32 precision must reject");
    let msg = err.to_string();
    assert!(
        msg.contains("precision_bits") && msg.contains("32"),
        "error must name the precision_bits == 32 limitation, got: {msg}"
    );
    // The 32-bit default still succeeds.
    let mut ok_params = ProvingParams::default();
    ok_params.precision_bits = 32;
    emit_circuit(&ir, &w, &ok_params).expect("precision_bits == 32 still emits");
}

// ── C16b: a genuine multi-axis input read errors instead of computing a
//          silently-wrong flat index ──
#[test]
fn test_c16b_multi_axis_input_index_errors() {
    // Read `a[1][2]` from a 2-axis array param. IR-gen normally flattens
    // such reads to a single Index segment; an UNFLATTENED multi-segment
    // path used to take the old `flat += i` fallback (== sum of segments,
    // wrong). It must now error loud.
    let path = InputPath::new("a", vec![PathSegment::Index(1), PathSegment::Index(2)]);
    let v0 = ValueId::next();
    let stmts = vec![
        IRStatement::new(
            0,
            v0,
            IR::ReadInteger { path, is_public: false },
            vec![],
            vec![],
            None,
        ),
        IRStatement::new(1, ValueId::next(), IR::ExposePublicI, vec![0], vec![v0], None),
    ];
    let ir = IRGraph::new(stmts);

    let mut inputs = CircuitInputs::new();
    inputs.params.push(InputParam {
        name: "a".to_string(),
        is_public: false,
        dtype: serde_json::Value::Null,
        value: InputNode::Array {
            shape: vec![2, 3],
            elements: vec![
                InputNode::Array {
                    shape: vec![3],
                    elements: vec![InputNode::Int(0), InputNode::Int(1), InputNode::Int(2)],
                    runtime_shape: None,
                },
                InputNode::Array {
                    shape: vec![3],
                    elements: vec![InputNode::Int(3), InputNode::Int(4), InputNode::Int(5)],
                    runtime_shape: None,
                },
            ],
            runtime_shape: None,
        },
    });
    let w = ResolvedWitness::new(inputs, 32);
    let params = ProvingParams::default();

    let err = emit_circuit(&ir, &w, &params)
        .err()
        .expect("multi-axis read must error");
    match err {
        ProvingError::UnsupportedIR { op_name } => {
            assert!(
                op_name.contains("multi-axis"),
                "expected multi-axis UnsupportedIR, got op_name={op_name}"
            );
        }
        other => panic!("expected UnsupportedIR for multi-axis input, got {other:?}"),
    }
}

// ── C16c: a scalar bool input proves with a Field-typed param, so its
//          Prover.toml value must be `"1"`/`"0"`, not bare `true`/`false`
//          (a nargo type error against the `Field` declaration) ──
#[test]
fn test_c16c_scalar_bool_param_toml_is_field_literal() {
    // main(flag: Field) { expose(flag) }, witness flag = True.
    let path = InputPath::new("flag", vec![]);
    let v0 = ValueId::next();
    let stmts = vec![
        IRStatement::new(
            0,
            v0,
            IR::ReadInteger { path, is_public: false },
            vec![],
            vec![],
            None,
        ),
        IRStatement::new(1, ValueId::next(), IR::ExposePublicI, vec![0], vec![v0], None),
    ];
    let ir = IRGraph::new(stmts);

    let mut inputs = CircuitInputs::new();
    inputs.params.push(InputParam {
        name: "flag".to_string(),
        is_public: false,
        dtype: serde_json::Value::Null,
        value: InputNode::Bool(true),
    });
    let w = ResolvedWitness::new(inputs, 32);
    let params = ProvingParams::default();

    let out = emit_circuit(&ir, &w, &params).expect("emit scalar-bool kernel");
    eprintln!("--- Prover.toml ---\n{}", out.prover_toml);
    // The parameter is declared `Field` in the signature ...
    assert!(out.main_nr.contains("flag: Field"));
    // ... so the TOML must give it a string field literal, never bare bool.
    assert!(
        out.prover_toml.contains("flag = \"1\""),
        "scalar bool must serialize as a Field literal \"1\", got: {}",
        out.prover_toml
    );
    assert!(
        !out.prover_toml.contains("flag = true") && !out.prover_toml.contains("flag = false"),
        "bare true/false is a nargo type error against a Field param"
    );
}

/// IR: read an integer input `x`, `bool(x)`, expose. Used by C16d.
fn make_bool_cast_ir() -> IRGraph {
    let path = InputPath::new("x", vec![]);
    let vread = ValueId::next();
    let vcast = ValueId::next();
    let stmts = vec![
        IRStatement::new(
            0,
            vread,
            IR::ReadInteger { path, is_public: false },
            vec![],
            vec![],
            None,
        ),
        IRStatement::new(1, vcast, IR::BoolCast, vec![0], vec![vread], None),
        IRStatement::new(2, ValueId::next(), IR::ExposePublicI, vec![1], vec![vcast], None),
    ];
    IRGraph::new(stmts)
}

fn scalar_int_witness(name: &str, value: i64) -> ResolvedWitness {
    let mut inputs = CircuitInputs::new();
    inputs.params.push(InputParam {
        name: name.to_string(),
        is_public: false,
        dtype: serde_json::Value::Null,
        value: InputNode::Int(value),
    });
    ResolvedWitness::new(inputs, 32)
}

// ── C16d: bool(x) for a non-boolean integer must produce a real 0/1, not
//          identity. Cross-backend value: bool(7) == 1 (mock returns 1). ──
#[test]
fn test_c16d_bool_cast_nonboolean_emits_gadget_and_folds_to_one() {
    let ir = make_bool_cast_ir();
    let w = scalar_int_witness("x", 7);
    let params = ProvingParams::default();
    let out = emit_circuit(&ir, &w, &params).expect("emit bool(7)");
    eprintln!("--- main.nr ---\n{}", out.main_nr);
    // A genuine truthiness gadget appears (not a bare identity passthrough).
    assert!(
        out.main_nr.contains("== 0 { 0 } else { 1 }"),
        "expected truthiness gadget for non-boolean input, got:\n{}",
        out.main_nr
    );
    // Value agreement with mock/Python: bool(7) == 1.
    if let Some(vals) = execute_if_toolchain(&ir, &w) {
        assert_eq!(vals, vec![field_hex(1)]);
    }
}

// bool(0) must fold to 0 (still via the gadget, since the input bound is
// unknown / not provably boolean).
#[test]
fn test_c16d_bool_cast_zero_folds_to_zero() {
    let ir = make_bool_cast_ir();
    let w = scalar_int_witness("x", 0);
    let params = ProvingParams::default();
    let out = emit_circuit(&ir, &w, &params).expect("emit bool(0)");
    assert!(out.main_nr.contains("== 0 { 0 } else { 1 }"));
    if let Some(vals) = execute_if_toolchain(&ir, &w) {
        assert_eq!(vals, vec![field_hex(0)]);
    }
}

// Fast path: when the operand is provably already boolean (a prior BoolCast
// result is bound <= 1), the second bool() is identity — NO extra gadget.
#[test]
fn test_c16d_bool_cast_idempotent_uses_fast_path() {
    let path = InputPath::new("x", vec![]);
    let vread = ValueId::next();
    let vcast1 = ValueId::next();
    let vcast2 = ValueId::next();
    let stmts = vec![
        IRStatement::new(
            0,
            vread,
            IR::ReadInteger { path, is_public: false },
            vec![],
            vec![],
            None,
        ),
        IRStatement::new(1, vcast1, IR::BoolCast, vec![0], vec![vread], None),
        // bool(bool(x)): operand is already bound <= 1.
        IRStatement::new(2, vcast2, IR::BoolCast, vec![1], vec![vcast1], None),
        IRStatement::new(3, ValueId::next(), IR::ExposePublicI, vec![2], vec![vcast2], None),
    ];
    let ir = IRGraph::new(stmts);
    let w = scalar_int_witness("x", 7);
    let params = ProvingParams::default();
    let out = emit_circuit(&ir, &w, &params).expect("emit bool(bool(7))");
    eprintln!("--- main.nr ---\n{}", out.main_nr);
    // Exactly one gadget: the inner bool(x). The outer bool() takes the
    // identity fast path because its operand is provably <= 1.
    let gadget_count = out.main_nr.matches("== 0 { 0 } else { 1 }").count();
    assert_eq!(
        gadget_count, 1,
        "outer bool() of a provably-boolean cell must NOT emit a second gadget:\n{}",
        out.main_nr
    );
    if let Some(vals) = execute_if_toolchain(&ir, &w) {
        assert_eq!(vals, vec![field_hex(1)]);
    }
}

#[test]
fn test_end_to_end_pipeline_if_toolchain_present() {
    use _zinnia_core::prove::noir::driver::DriverPaths;
    let paths = DriverPaths::default();
    if !paths.nargo.exists() || !paths.bb.exists() {
        eprintln!(
            "skip: nargo or bb missing (nargo={}, bb={})",
            paths.nargo.display(),
            paths.bb.display()
        );
        return;
    }

    let ir = make_three_float_add_ir();
    let w = three_float_witness();
    let params = ProvingParams::default();
    let artifact = NoirProverBackend.prove(&ir, &w, &params).expect("prove");
    eprintln!("noir prove_seconds = {:.3}", artifact.prove_seconds);
    assert!(artifact.prove_seconds > 0.0);
    assert!(!artifact.proof_bytes.is_empty());
    assert_eq!(artifact.public_values.len(), 1);
}

/// C13 regression: a bb-rejected proof must not count as valid. Prove a
/// circuit, corrupt the proof bytes, and check that `verify` (which runs
/// `bb verify` on the artifact) rejects it. Requires nargo and bb.
#[test]
fn test_corrupted_proof_is_rejected_by_verify() {
    use _zinnia_core::prove::noir::driver::DriverPaths;
    let paths = DriverPaths::default();
    if !paths.nargo.exists() || !paths.bb.exists() {
        eprintln!("skip: nargo or bb missing");
        return;
    }
    let ir = make_three_float_add_ir();
    let w = three_float_witness();
    let artifact = NoirProverBackend
        .prove(&ir, &w, &ProvingParams::default())
        .expect("prove");
    assert!(NoirProverBackend.verify(&artifact).unwrap().valid);
    // Flip bytes deep in the proof (hex-encoded: 2 chars per byte).
    let mut corrupted = artifact.clone();
    let flipped: String = corrupted.proof_bytes[128..192]
        .chars()
        .map(|c| if c == 'f' { '0' } else { 'f' })
        .collect();
    corrupted.proof_bytes.replace_range(128..192, &flipped);
    assert!(!NoirProverBackend.verify(&corrupted).unwrap().valid);
}
