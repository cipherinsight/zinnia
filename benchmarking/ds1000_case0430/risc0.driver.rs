// DS-1000 case 430 — Risc0 host driver.

use methods::{GUEST_CODE_FOR_ZK_PROOF_ELF, GUEST_CODE_FOR_ZK_PROOF_ID};
use risc0_zkvm::{default_prover, ExecutorEnv};
use std::time::Instant;

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::filter::EnvFilter::from_default_env())
        .init();

    let mut a: [f64; 30] = [0.0; 30];
    for i in 0..30 { a[i] = (i + 1) as f64; }
    let claimed_lo: f64 = -10.46632434519757;
    let claimed_hi: f64 =  41.46632434519757;

    let mut builder = ExecutorEnv::builder();
    for &v in a.iter() { builder.write(&v).unwrap(); }
    builder.write(&claimed_lo).unwrap();
    builder.write(&claimed_hi).unwrap();
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
