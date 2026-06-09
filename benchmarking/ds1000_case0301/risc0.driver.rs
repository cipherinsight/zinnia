// DS-1000 case 301 — Risc0 host driver.

use methods::{GUEST_CODE_FOR_ZK_PROOF_ELF, GUEST_CODE_FOR_ZK_PROOF_ID};
use risc0_zkvm::{default_prover, ExecutorEnv};
use std::time::Instant;

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::filter::EnvFilter::from_default_env())
        .init();

    let mut a: [f64; 20] = [0.0; 20];
    for i in 0..20 {
        a[i] = (i as f64) + 1.0;
    }
    let ncol: i64 = 4;
    let claim_rows: i64 = 5;
    let claim_ncol: i64 = 4;
    let claim_total: i64 = 20;
    let claim_rank: i64 = 2;

    let mut builder = ExecutorEnv::builder();
    for &v in a.iter() {
        builder.write(&v).unwrap();
    }
    builder.write(&ncol).unwrap();
    builder.write(&claim_rows).unwrap();
    builder.write(&claim_ncol).unwrap();
    for &v in a.iter() {
        builder.write(&v).unwrap();
    }
    builder.write(&claim_total).unwrap();
    builder.write(&claim_rank).unwrap();
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
