// DS-1000 case 427 — Risc0 host driver.

use methods::{GUEST_CODE_FOR_ZK_PROOF_ELF, GUEST_CODE_FOR_ZK_PROOF_ID};
use risc0_zkvm::{default_prover, ExecutorEnv};
use std::time::Instant;

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::filter::EnvFilter::from_default_env())
        .init();
    let m: u32 = 5;
    let a: Vec<u64> = (1u64..=20).collect();
    let mut builder = ExecutorEnv::builder();
    builder.write(&m).unwrap();
    for &v in &a { builder.write(&v).unwrap(); }
    let mut res: u64 = 0;
    for v in &a { res ^= v; }
    for j in 0..m as usize {
        let shift = (m as u32) - 1 - (j as u32);
        let bit: u64 = if (res & (1u64 << shift)) > 0 { 1 } else { 0 };
        builder.write(&bit).unwrap();
    }
    let env = builder.build().unwrap();
    let prover = default_prover();
    let start = Instant::now();
    let prove_info = prover.prove(env, GUEST_CODE_FOR_ZK_PROOF_ELF).unwrap();
    println!("Prove time (zk-STARK) (ms): {:?}", start.elapsed().as_millis());
    let start = Instant::now();
    prove_info.receipt.verify(GUEST_CODE_FOR_ZK_PROOF_ID).unwrap();
    println!("Verify time (zk-STARK) (ms): {:?}", start.elapsed().as_millis());
}
