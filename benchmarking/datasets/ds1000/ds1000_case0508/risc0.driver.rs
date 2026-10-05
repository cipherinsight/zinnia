// DS-1000 case 508 — Risc0 host driver.

use methods::{GUEST_CODE_FOR_ZK_PROOF_ELF, GUEST_CODE_FOR_ZK_PROOF_ID};
use risc0_zkvm::{default_prover, ExecutorEnv};
use std::time::Instant;

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::filter::EnvFilter::from_default_env())
        .init();

    let mut a = [[0i64; 10]; 10];
    let block: [[i64; 4]; 4] = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
    ];
    for di in 0..4 {
        for dj in 0..4 {
            a[2 + di][3 + dj] = block[di][dj];
        }
    }
    let ystart: u64 = 2;
    let ystop: u64 = 6;
    let xstart: u64 = 3;
    let xstop: u64 = 7;
    let expected: [i64; 16] = [
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16,
    ];

    let mut builder = ExecutorEnv::builder();
    for row in a.iter() {
        for &v in row.iter() {
            builder.write(&v).unwrap();
        }
    }
    builder.write(&ystart).unwrap();
    builder.write(&ystop).unwrap();
    builder.write(&xstart).unwrap();
    builder.write(&xstop).unwrap();
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
