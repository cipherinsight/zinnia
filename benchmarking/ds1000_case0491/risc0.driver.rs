// DS-1000 case 491 — Risc0 host driver.

use methods::{GUEST_CODE_FOR_ZK_PROOF_ELF, GUEST_CODE_FOR_ZK_PROOF_ID};
use risc0_zkvm::{default_prover, ExecutorEnv};
use std::time::Instant;

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::filter::EnvFilter::from_default_env())
        .init();
    let s1: [f64; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let s2: [f64; 6] = [1.0, 2.5, 3.0, 4.1, 5.0, 6.000001];
    let claimed: i64 = 2;

    let mut builder = ExecutorEnv::builder();
    for &v in s1.iter() { builder.write(&v).unwrap(); }
    for &v in s2.iter() { builder.write(&v).unwrap(); }
    builder.write(&claimed).unwrap();
    let env = builder.build().unwrap();

    let prover = default_prover();
    let start = Instant::now();
    let prove_info = prover.prove(env, GUEST_CODE_FOR_ZK_PROOF_ELF).unwrap();
    println!("Prove time (zk-STARK) (ms): {:?}", start.elapsed().as_millis());
    let receipt = prove_info.receipt;
    let start = Instant::now();
    receipt.verify(GUEST_CODE_FOR_ZK_PROOF_ID).unwrap();
    println!("Verify time (zk-STARK) (ms): {:?}", start.elapsed().as_millis());
}
