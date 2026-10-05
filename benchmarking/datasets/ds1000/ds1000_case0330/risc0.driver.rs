// DS-1000 case 330 — Risc0 host driver.

use methods::{GUEST_CODE_FOR_ZK_PROOF_ELF, GUEST_CODE_FOR_ZK_PROOF_ID};
use risc0_zkvm::{default_prover, ExecutorEnv};
use std::time::Instant;

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::filter::EnvFilter::from_default_env())
        .init();

    let mut a: [[f64; 6]; 8] = [[0.0; 6]; 8];
    for i in 0..8 {
        for j in 0..6 {
            a[i][j] = (i as f64 + 1.0) + 0.1 * (j as f64 + 1.0);
        }
    }
    let power: u64 = 3;

    let mut claimed: [[f64; 6]; 8] = [[0.0; 6]; 8];
    for i in 0..8 {
        for j in 0..6 {
            let mut acc: f64 = 1.0;
            for _ in 0..power {
                acc *= a[i][j];
            }
            claimed[i][j] = acc;
        }
    }

    let mut builder = ExecutorEnv::builder();
    for row in a.iter() {
        for &v in row.iter() {
            builder.write(&v).unwrap();
        }
    }
    builder.write(&power).unwrap();
    for row in claimed.iter() {
        for &v in row.iter() {
            builder.write(&v).unwrap();
        }
    }
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
