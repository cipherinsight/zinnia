// DS-1000 case 0498 — Risc0 host driver.

use methods::{GUEST_CODE_FOR_ZK_PROOF_ELF, GUEST_CODE_FOR_ZK_PROOF_ID};
use risc0_zkvm::{default_prover, ExecutorEnv};
use std::time::Instant;

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::filter::EnvFilter::from_default_env())
        .init();
    const N: usize = 2;
    const M: usize = 3;
    const K: usize = 2;
    let new_n: u32 = 4;
    let new_m: u32 = 5;
    let mut arr = [[[0.0f64; K]; M]; N];
    for i in 0..N { for j in 0..M { for k in 0..K {
        arr[i][j][k] = (i as f64) * 100.0 + (j as f64) * 10.0 + (k as f64);
    } } }
    let mut builder = ExecutorEnv::builder();
    builder.write(&new_n).unwrap();
    builder.write(&new_m).unwrap();
    for plane in &arr { for row in plane { for &v in row { builder.write(&v).unwrap(); } } }
    for i in 0..new_n as usize {
        for j in 0..new_m as usize {
            for k in 0..K {
                let v: f64 = if i < N && j < M { arr[i][j][k] } else { 0.0 };
                builder.write(&v).unwrap();
            }
        }
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
