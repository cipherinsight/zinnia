// DS-1000 case 367 — Risc0 host driver.

use methods::{GUEST_CODE_FOR_ZK_PROOF_ELF, GUEST_CODE_FOR_ZK_PROOF_ID};
use risc0_zkvm::{default_prover, ExecutorEnv};
use std::time::Instant;

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::filter::EnvFilter::from_default_env())
        .init();

    let pairs: [[i64; 2]; 3] = [[1, 2], [3, 4], [5, 6]];
    let array_of_arrays: [i64; 3] = [7, 8, 9];
    let expected: [i64; 3] = [7, 8, 9];

    let mut builder = ExecutorEnv::builder();
    for row in pairs.iter() {
        for &v in row.iter() { builder.write(&v).unwrap(); }
    }
    for &v in array_of_arrays.iter() { builder.write(&v).unwrap(); }
    for &v in expected.iter() { builder.write(&v).unwrap(); }
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
