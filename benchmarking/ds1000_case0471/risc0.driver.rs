// DS-1000 case 0471 — Risc0 host driver.

use methods::{GUEST_CODE_FOR_ZK_PROOF_ELF, GUEST_CODE_FOR_ZK_PROOF_ID};
use risc0_zkvm::{default_prover, ExecutorEnv};
use std::time::Instant;

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::filter::EnvFilter::from_default_env())
        .init();
    const N: usize = 2;
    const M: usize = 3;
    const K: usize = 4;
    let mut z = [[[0.0f64; K]; M]; N];
    for i in 0..N { for j in 0..M { for k in 0..K {
        z[i][j][k] = (i as f64) * 100.0 + (j as f64) * 10.0 + (k as f64);
    } } }
    let mut builder = ExecutorEnv::builder();
    for row in &z { for col in row { for &v in col { builder.write(&v).unwrap(); } } }
    for i in 0..N { for j in 0..M {
        let v: f64 = z[i][j][K - 1];
        builder.write(&v).unwrap();
    } }
    let env = builder.build().unwrap();
    let prover = default_prover();
    let start = Instant::now();
    let prove_info = prover.prove(env, GUEST_CODE_FOR_ZK_PROOF_ELF).unwrap();
    println!("Prove time (zk-STARK) (ms): {:?}", start.elapsed().as_millis());
    let start = Instant::now();
    prove_info.receipt.verify(GUEST_CODE_FOR_ZK_PROOF_ID).unwrap();
    println!("Verify time (zk-STARK) (ms): {:?}", start.elapsed().as_millis());
}
