// DS-1000 case 364 — Risc0 host driver.

use methods::{GUEST_CODE_FOR_ZK_PROOF_ELF, GUEST_CODE_FOR_ZK_PROOF_ID};
use risc0_zkvm::{default_prover, ExecutorEnv};
use std::time::Instant;

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::filter::EnvFilter::from_default_env())
        .init();
    let mut a = [[0.0f64; 10]; 100];
    for i in 0..100 { for j in 0..10 { a[i][j] = (i as f64) + (j as f64) * 0.01; } }
    let pos: i64 = 5;
    let mut element = [0.0f64; 10];
    for j in 0..10 { element[j] = 999.0 + (j as f64); }
    let mut expected = [[0.0f64; 10]; 101];
    for i in 0..101 {
        if i < pos as usize {
            for j in 0..10 { expected[i][j] = a[i][j]; }
        } else if i == pos as usize {
            for j in 0..10 { expected[i][j] = element[j]; }
        } else {
            for j in 0..10 { expected[i][j] = a[i - 1][j]; }
        }
    }

    let mut builder = ExecutorEnv::builder();
    for r in a.iter() { for &v in r.iter() { builder.write(&v).unwrap(); } }
    builder.write(&pos).unwrap();
    for &v in element.iter() { builder.write(&v).unwrap(); }
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
