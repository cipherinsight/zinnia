// DS-1000 case 349 — Risc0 host driver.

use methods::{GUEST_CODE_FOR_ZK_PROOF_ELF, GUEST_CODE_FOR_ZK_PROOF_ID};
use risc0_zkvm::{default_prover, ExecutorEnv};
use std::time::Instant;

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::filter::EnvFilter::from_default_env())
        .init();
    let data: [[f64; 6]; 8] = [
        [0.11, 0.12, 0.13, 0.14, 0.15, 0.16],
        [0.21, 0.22, 0.23, 0.24, 0.25, 0.26],
        [0.31, 0.32, 0.33, 0.34, 0.35, 0.36],
        [0.41, 0.42, 0.43, 0.44, 0.45, 0.46],
        [0.51, 0.52, 0.53, 0.54, 0.55, 0.56],
        [0.61, 0.62, 0.63, 0.64, 0.65, 0.66],
        [0.71, 0.72, 0.73, 0.74, 0.75, 0.76],
        [0.81, 0.82, 0.83, 0.84, 0.85, 0.86],
    ];
    let expected: u64 = 6;
    let mut builder = ExecutorEnv::builder();
    for row in data.iter() {
        for &v in row.iter() {
            builder.write(&v).unwrap();
        }
    }
    builder.write(&expected).unwrap();
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
