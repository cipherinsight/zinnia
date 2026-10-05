// DS-1000 case 472 — Risc0 host driver.

use methods::{GUEST_CODE_FOR_ZK_PROOF_ELF, GUEST_CODE_FOR_ZK_PROOF_ID};
use risc0_zkvm::{default_prover, ExecutorEnv};
use std::time::Instant;

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::filter::EnvFilter::from_default_env())
        .init();
    const N: usize = 9072;
    const SLAB: usize = 1512;
    let mut a: Vec<f64> = Vec::with_capacity(N);
    for k in 0..N {
        a.push((k as f64) * 0.01);
    }
    let start = 5 * SLAB;

    let mut builder = ExecutorEnv::builder();
    for &v in a.iter() {
        builder.write(&v).unwrap();
    }
    for k in 0..SLAB {
        let v = a[start + k];
        builder.write(&v).unwrap();
    }
    let env = builder.build().unwrap();

    let prover = default_prover();
    let start_t = Instant::now();
    let prove_info = prover.prove(env, GUEST_CODE_FOR_ZK_PROOF_ELF).unwrap();
    println!("Prove time (zk-STARK) (ms): {:?}", start_t.elapsed().as_millis());
    let receipt = prove_info.receipt;
    let start_t = Instant::now();
    receipt.verify(GUEST_CODE_FOR_ZK_PROOF_ID).unwrap();
    println!("Verify time (zk-STARK) (ms): {:?}", start_t.elapsed().as_millis());
}
