// DS-1000 case 346 — Risc0 host driver.

use methods::{GUEST_CODE_FOR_ZK_PROOF_ELF, GUEST_CODE_FOR_ZK_PROOF_ID};
use risc0_zkvm::{default_prover, ExecutorEnv};
use std::time::Instant;

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::filter::EnvFilter::from_default_env())
        .init();
    let data: [[f64; 6]; 8] = [
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    ];
    let row: i64 = 3;
    let multiply_number: i64 = 2;
    let expected: [f64; 6] = [2.0, 6.0, 12.0, 20.0, 30.0, 42.0];

    let mut builder = ExecutorEnv::builder();
    for r in data.iter() {
        for &v in r.iter() {
            builder.write(&v).unwrap();
        }
    }
    builder.write(&row).unwrap();
    builder.write(&multiply_number).unwrap();
    for &v in expected.iter() {
        builder.write(&v).unwrap();
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
