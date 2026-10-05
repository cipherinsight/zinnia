// DS-1000 case 0411 — Risc0 host driver.

use methods::{GUEST_CODE_FOR_ZK_PROOF_ELF, GUEST_CODE_FOR_ZK_PROOF_ID};
use risc0_zkvm::{default_prover, ExecutorEnv};
use std::time::Instant;

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::filter::EnvFilter::from_default_env())
        .init();
    let low: i64 = 2;
    let high: i64 = 5;
    let mut a = [[0.0f64; 5]; 5];
    for i in 0..5 { for j in 0..5 { a[i][j] = (i as f64) * 5.0 + (j as f64) + 1.0; } }
    let mut builder = ExecutorEnv::builder();
    builder.write(&low).unwrap();
    builder.write(&high).unwrap();
    for row in &a { for &v in row { builder.write(&v).unwrap(); } }
    let h = (high - low) as usize;
    for i in 0..h {
        for j in 0..h {
            let si = low + i as i64;
            let sj = low + j as i64;
            let v: f64 = if si >= 0 && si < 5 && sj >= 0 && sj < 5 {
                a[si as usize][sj as usize]
            } else { 0.0 };
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
