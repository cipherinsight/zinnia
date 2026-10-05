// DS-1000 case 507 — Risc0 host driver.

use methods::{GUEST_CODE_FOR_ZK_PROOF_ELF, GUEST_CODE_FOR_ZK_PROOF_ID};
use risc0_zkvm::{default_prover, ExecutorEnv};
use std::time::Instant;

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::filter::EnvFilter::from_default_env())
        .init();

    let im: [[i64; 6]; 5] = [
        [0, 0, 0, 0, 0, 0],
        [0, 0, 1, 2, 3, 0],
        [0, 0, 4, 5, 6, 0],
        [0, 0, 7, 8, 9, 0],
        [0, 0, 0, 0, 0, 0],
    ];
    let rmin: u64 = 1;
    let rmax: u64 = 3;
    let cmin: u64 = 2;
    let cmax: u64 = 4;
    let expected: [i64; 9] = [1, 2, 3, 4, 5, 6, 7, 8, 9];

    let mut builder = ExecutorEnv::builder();
    for row in im.iter() {
        for &v in row.iter() {
            builder.write(&v).unwrap();
        }
    }
    builder.write(&rmin).unwrap();
    builder.write(&rmax).unwrap();
    builder.write(&cmin).unwrap();
    builder.write(&cmax).unwrap();
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
