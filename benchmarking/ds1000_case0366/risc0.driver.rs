// DS-1000 case 0366 — Risc0 host driver.

use methods::{GUEST_CODE_FOR_ZK_PROOF_ELF, GUEST_CODE_FOR_ZK_PROOF_ID};
use risc0_zkvm::{default_prover, ExecutorEnv};
use std::time::Instant;

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::filter::EnvFilter::from_default_env())
        .init();
    const R: usize = 100;
    const C: usize = 10;
    const E: usize = 5;
    let mut a = vec![vec![0.0f64; C]; R];
    for i in 0..R { for j in 0..C { a[i][j] = (i as f64 + 1.0) * 0.1 + (j as f64 + 1.0) * 0.01; } }
    let pos: [i64; E] = [10, 20, 30, 40, 50];
    let mut element = [[0.0f64; C]; E];
    for k in 0..E { for j in 0..C { element[k][j] = 1000.0 + (k as f64) * 10.0 + (j as f64); } }
    let mut builder = ExecutorEnv::builder();
    for row in &a { for &v in row { builder.write(&v).unwrap(); } }
    for &v in &pos { builder.write(&v).unwrap(); }
    for row in &element { for &v in row { builder.write(&v).unwrap(); } }
    let adjusted: [i64; E] = std::array::from_fn(|k| pos[k] - (k as i64));
    let mut out: Vec<[f64; C]> = Vec::with_capacity(R + E);
    let mut k = 0usize;
    for src in 0..R {
        while k < E && adjusted[k] as usize == src { out.push(element[k]); k += 1; }
        let mut row = [0.0f64; C];
        for j in 0..C { row[j] = a[src][j]; }
        out.push(row);
    }
    while k < E { out.push(element[k]); k += 1; }
    for row in &out { for &v in row { builder.write(&v).unwrap(); } }
    let env = builder.build().unwrap();
    let prover = default_prover();
    let start = Instant::now();
    let prove_info = prover.prove(env, GUEST_CODE_FOR_ZK_PROOF_ELF).unwrap();
    println!("Prove time (zk-STARK) (ms): {:?}", start.elapsed().as_millis());
    let start = Instant::now();
    prove_info.receipt.verify(GUEST_CODE_FOR_ZK_PROOF_ID).unwrap();
    println!("Verify time (zk-STARK) (ms): {:?}", start.elapsed().as_millis());
}
