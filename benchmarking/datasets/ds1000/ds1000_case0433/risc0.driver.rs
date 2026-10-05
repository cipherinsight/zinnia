// DS-1000 case 433 — Risc0 host driver.

use methods::{GUEST_CODE_FOR_ZK_PROOF_ELF, GUEST_CODE_FOR_ZK_PROOF_ID};
use risc0_zkvm::{default_prover, ExecutorEnv};
use std::time::Instant;

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::filter::EnvFilter::from_default_env())
        .init();

    let mut a: [i64; 16] = [0; 16];
    for i in 0..4 { for j in 0..4 { a[i * 4 + j] = (i * 4 + j + 1) as i64; } }
    let zero_rows: i64 = 2;
    let zero_cols: i64 = 1;
    let mut expected: [i64; 16] = [0; 16];
    for i in 0..4 {
        for j in 0..4 {
            let row_match = (i as i64) == zero_rows;
            let col_match = (j as i64) == zero_cols;
            expected[i * 4 + j] = if row_match || col_match { 0 } else { a[i * 4 + j] };
        }
    }

    let mut builder = ExecutorEnv::builder();
    for &v in a.iter() { builder.write(&v).unwrap(); }
    builder.write(&zero_rows).unwrap();
    builder.write(&zero_cols).unwrap();
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
