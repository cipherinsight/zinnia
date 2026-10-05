// DS-1000 case 361 — Risc0 host driver.

use methods::{GUEST_CODE_FOR_ZK_PROOF_ELF, GUEST_CODE_FOR_ZK_PROOF_ID};
use risc0_zkvm::{default_prover, ExecutorEnv};
use std::time::Instant;

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::filter::EnvFilter::from_default_env())
        .init();
    let mut a: [[f64; 6]; 6] = [[0.0; 6]; 6];
    for i in 0..6 { for j in 0..6 { a[i][j] = (i as f64) * 10.0 + (j as f64) + 0.1; } }
    let kept: [usize; 4] = [1, 3, 4, 5];
    let mut expected: [[f64; 4]; 6] = [[0.0; 4]; 6];
    for i in 0..6 { for (jo, &j) in kept.iter().enumerate() { expected[i][jo] = a[i][j]; } }

    let mut builder = ExecutorEnv::builder();
    for r in a.iter() { for &v in r.iter() { builder.write(&v).unwrap(); } }
    for r in expected.iter() { for &v in r.iter() { builder.write(&v).unwrap(); } }
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
