// DS-1000 case 0354 — Risc0 host driver.

use methods::{GUEST_CODE_FOR_ZK_PROOF_ELF, GUEST_CODE_FOR_ZK_PROOF_ID};
use risc0_zkvm::{default_prover, ExecutorEnv};
use std::time::Instant;

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::filter::EnvFilter::from_default_env())
        .init();
    let a: [[i64; 3]; 2] = [[1, 2, 0], [3, 1, 2]];
    let b: [[i64; 3]; 7] = [
        [1, 2, 0], [0, 0, 0], [5, 5, 5], [4, 4, 4],
        [9, 9, 9], [8, 8, 8], [7, 7, 7],
    ];
    let mut builder = ExecutorEnv::builder();
    for row in &a { for &v in row { builder.write(&v).unwrap(); } }
    for row in &b { for &v in row { builder.write(&v).unwrap(); } }
    let claim_count: u32 = 1;
    builder.write(&claim_count).unwrap();
    let result: [[i64; 3]; 1] = [[3, 1, 2]];
    for row in &result { for &v in row { builder.write(&v).unwrap(); } }
    let env = builder.build().unwrap();
    let prover = default_prover();
    let start = Instant::now();
    let prove_info = prover.prove(env, GUEST_CODE_FOR_ZK_PROOF_ELF).unwrap();
    println!("Prove time (zk-STARK) (ms): {:?}", start.elapsed().as_millis());
    let start = Instant::now();
    prove_info.receipt.verify(GUEST_CODE_FOR_ZK_PROOF_ID).unwrap();
    println!("Verify time (zk-STARK) (ms): {:?}", start.elapsed().as_millis());
}
