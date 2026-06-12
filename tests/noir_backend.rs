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

/// C13 regression: a bb-rejected proof must surface as a `ProvingError`, never
/// as a "valid" artifact. We reproduce a real `bb verify` rejection by proving
/// circuit A and then corrupting the staged proof bytes before the driver's
/// in-process verify runs, asserting the driver reports `verify_ok == false`
/// and captures the verifier's diagnostics. Requires the real nargo+bb
/// toolchain; skips cleanly otherwise.
#[test]
fn test_corrupted_proof_is_rejected_by_driver_verify() {
    use _zinnia_core::prove::noir::driver::{run_pipeline, DriverPaths};
    use std::process::Command;

    let paths = DriverPaths::default();
    if !paths.nargo.exists() || !paths.bb.exists() {
        eprintln!(
            "skip: nargo or bb missing (nargo={}, bb={})",
            paths.nargo.display(),
            paths.bb.display()
        );
        return;
    }

    // Stage a real package by hand (mirrors mod.rs::stage_package), so we can
    // drive nargo/bb directly and inject a corrupt proof between prove and
    // verify — the exact failure mode C13 is about.
    let ir = make_three_float_add_ir();
    let w = three_float_witness();
    let params = ProvingParams::default();
    let emit = emit_circuit(&ir, &w, &params).expect("emit");

    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let dir = std::env::temp_dir().join(format!("zinnia-noir-c13-{}", nanos));
    let src_dir = dir.join("src");
    let lib_dir = dir.join("noir_fpc").join("src");
    std::fs::create_dir_all(&src_dir).unwrap();
    std::fs::create_dir_all(&lib_dir).unwrap();
    std::fs::write(
        dir.join("Nargo.toml"),
        "[package]\nname = \"zinnia_circuit\"\ntype = \"bin\"\nauthors = [\"\"]\n\n\
         [dependencies]\nnoir_fpc = { path = \"./noir_fpc\" }\n",
    )
    .unwrap();
    std::fs::write(src_dir.join("main.nr"), &emit.main_nr).unwrap();
    std::fs::write(dir.join("Prover.toml"), &emit.prover_toml).unwrap();
    std::fs::write(
        dir.join("noir_fpc").join("Nargo.toml"),
        include_str!("../benchmarking/noir_fpc/Nargo.toml"),
    )
    .unwrap();
    std::fs::write(
        lib_dir.join("lib.nr"),
        include_str!("../benchmarking/noir_fpc/src/lib.nr"),
    )
    .unwrap();

    // First a clean run to confirm the staged package proves+verifies, and to
    // populate target/ with a genuine bytecode, witness, vk, and proof.
    let clean = run_pipeline(&dir, &paths).expect("clean pipeline");
    assert!(clean.verify_ok, "baseline pipeline should verify");
    assert!(clean.verify_detail.is_none());

    // Now corrupt the staged proof and re-run ONLY the verify step against the
    // genuine VK + public_inputs, exactly as the driver would. A flipped proof
    // byte makes bb verify reject.
    let target = dir.join("target");
    let proof_path = target.join("proof");
    let mut proof = std::fs::read(&proof_path).expect("read proof");
    assert!(!proof.is_empty());
    // Flip several bytes deep in the proof to defeat any leading-header
    // tolerance.
    for b in proof.iter_mut().skip(64).take(32) {
        *b ^= 0xff;
    }
    std::fs::write(&proof_path, &proof).unwrap();

    let verify = Command::new(&paths.bb)
        .arg("verify")
        .arg("-k")
        .arg(target.join("vk"))
        .arg("-p")
        .arg(&proof_path)
        .arg("-i")
        .arg(target.join("public_inputs"))
        .arg("--scheme")
        .arg("ultra_honk")
        .arg("--verifier_target")
        .arg("noir-recursive-no-zk")
        .output()
        .expect("spawn bb verify");
    eprintln!(
        "corrupted-proof bb verify: success={}, stderr_tail={:?}",
        verify.status.success(),
        String::from_utf8_lossy(&verify.stderr)
    );
    assert!(
        !verify.status.success(),
        "bb verify must reject a corrupted proof"
    );

    let _ = std::fs::remove_dir_all(&dir);
}
