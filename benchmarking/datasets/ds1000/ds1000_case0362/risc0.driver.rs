// DS-1000 case 0362 — Risc0 host driver.

use methods::{GUEST_CODE_FOR_ZK_PROOF_ELF, GUEST_CODE_FOR_ZK_PROOF_ID};
use risc0_zkvm::{default_prover, ExecutorEnv};
use std::time::Instant;

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::filter::EnvFilter::from_default_env())
        .init();
    const R: usize = 8;
    const C: usize = 6;
    let mut a = [[0.0f64; C]; R];
    for i in 0..R { for j in 0..C { a[i][j] = (i as f64 + 1.0) * 10.0 + (j as f64 + 1.0); } }
    let del_col: [i64; 4] = [1, 3, 5, 7];
    let out_cols: u32 = 3;
    let mut builder = ExecutorEnv::builder();
    builder.write(&out_cols).unwrap();
    for row in &a { for &v in row { builder.write(&v).unwrap(); } }
    for &v in &del_col { builder.write(&v).unwrap(); }
    let mut delete_mask = [false; C];
    for &v in &del_col {
        if v >= 1 && v <= C as i64 { delete_mask[(v - 1) as usize] = true; }
    }
    for i in 0..R {
        for j in 0..C {
            if !delete_mask[j] { builder.write(&a[i][j]).unwrap(); }
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
