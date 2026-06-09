// DS-1000 case 387 — Risc0 host driver.

use methods::{GUEST_CODE_FOR_ZK_PROOF_ELF, GUEST_CODE_FOR_ZK_PROOF_ID};
use risc0_zkvm::{default_prover, ExecutorEnv};
use std::time::Instant;

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::filter::EnvFilter::from_default_env())
        .init();

    const H: usize = 100;
    const W: usize = 200;

    let mut a: Vec<Vec<f64>> = Vec::with_capacity(H);
    for i in 0..H {
        let mut row: Vec<f64> = Vec::with_capacity(W);
        for j in 0..W { row.push(i as f64 * 0.001 + j as f64 * 0.0001); }
        a.push(row);
    }
    let mut expected: Vec<f64> = Vec::with_capacity(20000);
    for r_block in 0..50 {
        for c_block in 0..100 {
            for p in 0..2 {
                for q in 0..2 {
                    expected.push(a[r_block * 2 + p][c_block * 2 + q]);
                }
            }
        }
    }

    let mut builder = ExecutorEnv::builder();
    for row in a.iter() { for &v in row.iter() { builder.write(&v).unwrap(); } }
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
