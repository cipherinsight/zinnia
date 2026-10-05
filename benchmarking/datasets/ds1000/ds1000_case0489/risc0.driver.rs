// DS-1000 case 0489 — Risc0 host driver.

use methods::{GUEST_CODE_FOR_ZK_PROOF_ELF, GUEST_CODE_FOR_ZK_PROOF_ID};
use risc0_zkvm::{default_prover, ExecutorEnv};
use std::time::Instant;

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::filter::EnvFilter::from_default_env())
        .init();
    let arr: [f64; 8] = [-20.0, -5.0, 0.0, 12.0, 14.9, 15.0, 20.0, 100.0];
    let expect: [f64; 8] = [0.0, 0.0, 5.0, 17.0, 19.9, 30.0, 30.0, 30.0];
    let mut builder = ExecutorEnv::builder();
    for &v in &arr { builder.write(&v).unwrap(); }
    for &v in &expect { builder.write(&v).unwrap(); }
    let env = builder.build().unwrap();
    let prover = default_prover();
    let start = Instant::now();
    let prove_info = prover.prove(env, GUEST_CODE_FOR_ZK_PROOF_ELF).unwrap();
    println!("Prove time (zk-STARK) (ms): {:?}", start.elapsed().as_millis());
    let start = Instant::now();
    prove_info.receipt.verify(GUEST_CODE_FOR_ZK_PROOF_ID).unwrap();
    println!("Verify time (zk-STARK) (ms): {:?}", start.elapsed().as_millis());
}
