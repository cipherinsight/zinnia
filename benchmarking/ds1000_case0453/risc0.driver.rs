// DS-1000 case 453 — Risc0 host driver.

use methods::{GUEST_CODE_FOR_ZK_PROOF_ELF, GUEST_CODE_FOR_ZK_PROOF_ID};
use risc0_zkvm::{default_prover, ExecutorEnv};
use std::time::Instant;

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::filter::EnvFilter::from_default_env())
        .init();

    let x: [[i64; 4]; 5] = [
        [3, 4, 0, 0],
        [0, 3, 4, 0],
        [0, 0, 3, 4],
        [4, 0, 0, 3],
        [3, 0, 4, 0],
    ];
    let mut claimed: [[f64; 4]; 5] = [[0.0; 4]; 5];
    for i in 0..5 {
        let mut ss: f64 = 0.0;
        for j in 0..4 { ss += (x[i][j] as f64) * (x[i][j] as f64); }
        let l2 = ss.sqrt();
        for j in 0..4 { claimed[i][j] = (x[i][j] as f64) / l2; }
    }

    let mut builder = ExecutorEnv::builder();
    for row in x.iter() { for v in row.iter() { builder.write(v).unwrap(); } }
    for row in claimed.iter() { for v in row.iter() { builder.write(v).unwrap(); } }
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
