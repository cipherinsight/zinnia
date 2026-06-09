// DS-1000 case 320 — Risc0 host driver.

use methods::{GUEST_CODE_FOR_ZK_PROOF_ELF, GUEST_CODE_FOR_ZK_PROOF_ID};
use risc0_zkvm::{default_prover, ExecutorEnv};
use std::time::Instant;

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::filter::EnvFilter::from_default_env())
        .init();

    let data: [[f64; 6]; 5] = [
        [0.11, 0.12, 0.13, 0.14, 0.15, 0.16],
        [0.21, 0.22, 0.23, 0.24, 0.25, 0.26],
        [0.31, 0.32, 0.33, 0.34, -99.0, 0.36],
        [0.41, 0.42, 0.43, 0.44, 0.45, 0.46],
        [0.51, 0.52, 0.53, 0.54, 0.55, 0.56],
    ];
    let claimed_row: u64 = 2;
    let claimed_col: u64 = 4;

    let mut builder = ExecutorEnv::builder();
    for row in data.iter() {
        for &v in row.iter() {
            builder.write(&v).unwrap();
        }
    }
    builder.write(&claimed_row).unwrap();
    builder.write(&claimed_col).unwrap();
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
