// DS-1000 case 363 — Risc0 host driver.

use methods::{GUEST_CODE_FOR_ZK_PROOF_ELF, GUEST_CODE_FOR_ZK_PROOF_ID};
use risc0_zkvm::{default_prover, ExecutorEnv};
use std::time::Instant;

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::filter::EnvFilter::from_default_env())
        .init();
    let mut a = [0.0f64; 100];
    for i in 0..100 { a[i] = (i as f64) * 0.1; }
    let pos: i64 = 7;
    let element: i64 = 999;
    let mut expected = [0.0f64; 101];
    for i in 0..101 {
        if i < pos as usize { expected[i] = a[i]; }
        else if i == pos as usize { expected[i] = element as f64; }
        else { expected[i] = a[i - 1]; }
    }

    let mut builder = ExecutorEnv::builder();
    for &v in a.iter() { builder.write(&v).unwrap(); }
    builder.write(&pos).unwrap();
    builder.write(&element).unwrap();
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
