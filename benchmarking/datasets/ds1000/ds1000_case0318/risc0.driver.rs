// DS-1000 case 318 — Risc0 host driver.

use methods::{GUEST_CODE_FOR_ZK_PROOF_ELF, GUEST_CODE_FOR_ZK_PROOF_ID};
use risc0_zkvm::{default_prover, ExecutorEnv};
use std::time::Instant;

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::filter::EnvFilter::from_default_env())
        .init();

    let mut a: [[f64; 6]; 8] = [[0.0; 6]; 8];
    for i in 0..8 {
        for j in 0..6 {
            a[i][j] = ((i + 1) * 10 + (j + 1)) as f64;
        }
    }
    let permutation: [i64; 6] = [2, 0, 4, 1, 5, 3];
    let c: [i64; 6] = [1, 3, 0, 5, 2, 4];

    let mut out: [[f64; 6]; 8] = [[0.0; 6]; 8];
    for i in 0..8 {
        for j in 0..6 {
            out[i][j] = a[i][c[j] as usize];
        }
    }

    let mut builder = ExecutorEnv::builder();
    for row in a.iter() {
        for &v in row.iter() {
            builder.write(&v).unwrap();
        }
    }
    for &p in permutation.iter() {
        builder.write(&p).unwrap();
    }
    for row in out.iter() {
        for &v in row.iter() {
            builder.write(&v).unwrap();
        }
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
