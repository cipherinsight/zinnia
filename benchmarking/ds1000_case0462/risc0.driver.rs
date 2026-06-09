// DS-1000 case 462 — Risc0 host driver.

use methods::{GUEST_CODE_FOR_ZK_PROOF_ELF, GUEST_CODE_FOR_ZK_PROOF_ID};
use risc0_zkvm::{default_prover, ExecutorEnv};
use std::time::Instant;

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::filter::EnvFilter::from_default_env())
        .init();

    let a: [i64; 20] = [0, 0, 1, 1, 2, 3, 3, 0, 4, 4, 5, 0, 6, 7, 7, 8, 0, 9, 9, 0];
    let claimed: [i64; 9] = [1, 2, 3, 4, 5, 6, 7, 8, 9];
    let claimed_len: u64 = claimed.len() as u64;

    let mut builder = ExecutorEnv::builder();
    for v in a.iter() { builder.write(v).unwrap(); }
    builder.write(&claimed_len).unwrap();
    for v in claimed.iter() { builder.write(v).unwrap(); }
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
