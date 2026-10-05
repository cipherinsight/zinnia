// DS-1000 case 334 — Risc0 host driver.

use methods::{GUEST_CODE_FOR_ZK_PROOF_ELF, GUEST_CODE_FOR_ZK_PROOF_ID};
use risc0_zkvm::{default_prover, ExecutorEnv};
use std::time::Instant;

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::filter::EnvFilter::from_default_env())
        .init();

    let mut a = [0.0f64; 50];
    let mut b = [0.0f64; 50];
    let mut c = [0.0f64; 50];
    let mut result = [0.0f64; 50];
    for i in 0..50 {
        a[i] = 0.30 + 0.01 * (i as f64);
        b[i] = 0.30 + 0.02 * (i as f64);
        c[i] = 0.30 + 0.03 * (i as f64);
        result[i] = 0.30 + 0.02 * (i as f64);
    }

    let mut builder = ExecutorEnv::builder();
    for v in a.iter() { builder.write(v).unwrap(); }
    for v in b.iter() { builder.write(v).unwrap(); }
    for v in c.iter() { builder.write(v).unwrap(); }
    for v in result.iter() { builder.write(v).unwrap(); }
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
