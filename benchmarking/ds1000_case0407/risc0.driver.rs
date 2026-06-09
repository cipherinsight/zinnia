// DS-1000 case 0407 — Risc0 host driver.

use methods::{GUEST_CODE_FOR_ZK_PROOF_ELF, GUEST_CODE_FOR_ZK_PROOF_ID};
use risc0_zkvm::{default_prover, ExecutorEnv};
use std::time::Instant;

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::filter::EnvFilter::from_default_env())
        .init();
    let a: Vec<i64> = (1i64..=100).collect();
    let accmap: Vec<i64> = (0i64..100).map(|i| (i % 5) - 2).collect();
    let out_len: u32 = 3;
    let expect: [i64; 3] = [1010, 2000, 2040];
    let mut builder = ExecutorEnv::builder();
    for &v in &a { builder.write(&v).unwrap(); }
    for &v in &accmap { builder.write(&v).unwrap(); }
    builder.write(&out_len).unwrap();
    for &v in &expect { builder.write(&v).unwrap(); }
    let env = builder.build().unwrap();
    let prover = default_prover();
    let start = Instant::now();
    let prove_info = prover.prove(env, GUEST_CODE_FOR_ZK_PROOF_ELF).unwrap();
    println!("Prove time (zk-STARK) (ms): {:?}", start.elapsed().as_millis());
    let start = Instant::now();
    prove_info.receipt.verify(GUEST_CODE_FOR_ZK_PROOF_ID).unwrap();
    println!("Verify time (zk-STARK) (ms): {:?}", start.elapsed().as_millis());
}
