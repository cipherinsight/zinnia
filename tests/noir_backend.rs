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
    assert_eq!(out.expose_count, 1);
    // 6.0 * 2^32 = 25_769_803_776
    assert_eq!(out.public_values_decimal[0], "25769803776");
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
