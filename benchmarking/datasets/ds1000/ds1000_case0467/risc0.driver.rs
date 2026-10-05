// DS-1000 case 467 — Risc0 host driver.

use methods::{GUEST_CODE_FOR_ZK_PROOF_ELF, GUEST_CODE_FOR_ZK_PROOF_ID};
use risc0_zkvm::{default_prover, ExecutorEnv};
use std::time::Instant;

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::filter::EnvFilter::from_default_env())
        .init();

    let a: [[i64; 4]; 4] = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
    ];
    let size: [i64; 2] = [3, 3];
    let counts: [u64; 16] = [4, 6, 6, 4, 6, 9, 9, 6, 6, 9, 9, 6, 4, 6, 6, 4];
    let total: u64 = 100;
    let flat: [i64; 100] = [
        1, 2, 5, 6,
        1, 2, 3, 5, 6, 7,
        2, 3, 4, 6, 7, 8,
        3, 4, 7, 8,
        1, 2, 5, 6, 9, 10,
        1, 2, 3, 5, 6, 7, 9, 10, 11,
        2, 3, 4, 6, 7, 8, 10, 11, 12,
        3, 4, 7, 8, 11, 12,
        5, 6, 9, 10, 13, 14,
        5, 6, 7, 9, 10, 11, 13, 14, 15,
        6, 7, 8, 10, 11, 12, 14, 15, 16,
        7, 8, 11, 12, 15, 16,
        9, 10, 13, 14,
        9, 10, 11, 13, 14, 15,
        10, 11, 12, 14, 15, 16,
        11, 12, 15, 16,
    ];

    let mut builder = ExecutorEnv::builder();
    for row in a.iter() {
        for &v in row.iter() {
            builder.write(&v).unwrap();
        }
    }
    builder.write(&size[0]).unwrap();
    builder.write(&size[1]).unwrap();
    for &c in counts.iter() {
        builder.write(&c).unwrap();
    }
    builder.write(&total).unwrap();
    for &v in flat.iter() {
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
