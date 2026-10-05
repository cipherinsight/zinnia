// DS-1000 case 0499 — Risc0 host driver.

use methods::{GUEST_CODE_FOR_ZK_PROOF_ELF, GUEST_CODE_FOR_ZK_PROOF_ID};
use risc0_zkvm::{default_prover, ExecutorEnv};
use std::time::Instant;

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::filter::EnvFilter::from_default_env())
        .init();
    const R: usize = 42;
    const C: usize = 10;
    let new_rows: u32 = 45;
    let new_cols: u32 = 12;
    let mut a = [[0.0f64; C]; R];
    for i in 0..R { for j in 0..C { a[i][j] = (i as f64 + 1.0) * 0.1 + (j as f64 + 1.0) * 0.01; } }
    let mut builder = ExecutorEnv::builder();
    builder.write(&new_rows).unwrap();
    builder.write(&new_cols).unwrap();
    for row in &a { for &v in row { builder.write(&v).unwrap(); } }
    for i in 0..new_rows as usize {
        for j in 0..new_cols as usize {
            let v: f64 = if i < R && j < C { a[i][j] } else { 0.0 };
            builder.write(&v).unwrap();
        }
    }
    let env = builder.build().unwrap();
    let prover = default_prover();
    let start = Instant::now();
    let prove_info = prover.prove(env, GUEST_CODE_FOR_ZK_PROOF_ELF).unwrap();
    println!("Prove time (zk-STARK) (ms): {:?}", start.elapsed().as_millis());
    let start = Instant::now();
    prove_info.receipt.verify(GUEST_CODE_FOR_ZK_PROOF_ID).unwrap();
    println!("Verify time (zk-STARK) (ms): {:?}", start.elapsed().as_millis());
}
