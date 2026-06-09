// DS-1000 case 481 — Risc0 host driver.

use methods::{GUEST_CODE_FOR_ZK_PROOF_ELF, GUEST_CODE_FOR_ZK_PROOF_ID};
use risc0_zkvm::{default_prover, ExecutorEnv};
use std::time::Instant;

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::filter::EnvFilter::from_default_env())
        .init();
    let x: [i64; 9] = [1, 5, 5, 2, 5, 5, 3, 5, 4];
    let y: [i64; 9] = [2, 7, 7, 3, 8, 7, 4, 7, 5];
    let a: i64 = 5;
    let b: i64 = 7;
    let claimed_len: u64 = 4;
    let mut claimed: [u64; 9] = [0; 9];
    let expected: [u64; 4] = [1, 2, 5, 7];
    for k in 0..4 { claimed[k] = expected[k]; }

    let mut builder = ExecutorEnv::builder();
    for &v in x.iter() { builder.write(&v).unwrap(); }
    for &v in y.iter() { builder.write(&v).unwrap(); }
    builder.write(&a).unwrap();
    builder.write(&b).unwrap();
    builder.write(&claimed_len).unwrap();
    for &v in claimed.iter() { builder.write(&v).unwrap(); }
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
