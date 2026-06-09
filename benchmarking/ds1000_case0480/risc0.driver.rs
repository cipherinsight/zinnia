// DS-1000 case 480 — Risc0 host driver.

use methods::{GUEST_CODE_FOR_ZK_PROOF_ELF, GUEST_CODE_FOR_ZK_PROOF_ID};
use risc0_zkvm::{default_prover, ExecutorEnv};
use std::time::Instant;

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::filter::EnvFilter::from_default_env())
        .init();
    let x: [i64; 8] = [10, 20, 30, 40, 50, 60, 70, 80];
    let y: [i64; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
    let a: i64 = 50;
    let b: i64 = 5;
    let claimed: i64 = 4;

    let mut builder = ExecutorEnv::builder();
    for &v in x.iter() { builder.write(&v).unwrap(); }
    for &v in y.iter() { builder.write(&v).unwrap(); }
    builder.write(&a).unwrap();
    builder.write(&b).unwrap();
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
