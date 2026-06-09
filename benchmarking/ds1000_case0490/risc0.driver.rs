// DS-1000 case 490 — Risc0 host driver.

use methods::{GUEST_CODE_FOR_ZK_PROOF_ELF, GUEST_CODE_FOR_ZK_PROOF_ID};
use risc0_zkvm::{default_prover, ExecutorEnv};
use std::time::Instant;

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::filter::EnvFilter::from_default_env())
        .init();
    let mut arr = [[[0.0f64; 4]; 4]; 5];
    for i in 0..5 {
        for j in 0..4 {
            for k in 0..4 {
                arr[i][j][k] = ((i * 16 + j * 4 + k) % 30) as f64;
            }
        }
    }
    let n1: [i64; 5] = [5, 10, 15, 0, 8];
    let n2: [i64; 5] = [20, 25, 22, 30, 18];
    let expected: [f64; 80] = [
        0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
        21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 30.0, 30.0, 30.0, 30.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 20.0, 21.0, 22.0,
        23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 5.0, 6.0, 7.0, 8.0,
        0.0, 0.0, 0.0, 0.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 30.0, 30.0,
    ];

    let mut builder = ExecutorEnv::builder();
    for i in 0..5 {
        for j in 0..4 {
            for k in 0..4 {
                builder.write(&arr[i][j][k]).unwrap();
            }
        }
    }
    for &v in n1.iter() { builder.write(&v).unwrap(); }
    for &v in n2.iter() { builder.write(&v).unwrap(); }
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
