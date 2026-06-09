// DS-1000 case 340 — Risc0 host driver.

use methods::{GUEST_CODE_FOR_ZK_PROOF_ELF, GUEST_CODE_FOR_ZK_PROOF_ID};
use risc0_zkvm::{default_prover, ExecutorEnv};
use std::time::Instant;

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::filter::EnvFilter::from_default_env())
        .init();

    let mut x = [[0i64; 6]; 5];
    for i in 0..5 {
        for j in 0..6 {
            x[i][j] = (i as i64 + 1) * 10 + (j as i64 + 1);
        }
    }
    let mut claimed = [0i64; 30];
    let mut k = 0;
    for i in 0..5 {
        for j in 0..6 {
            claimed[k] = x[i][j];
            k += 1;
        }
    }

    let mut builder = ExecutorEnv::builder();
    for row in x.iter() {
        for &v in row.iter() { builder.write(&v).unwrap(); }
    }
    for &v in claimed.iter() { builder.write(&v).unwrap(); }
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
