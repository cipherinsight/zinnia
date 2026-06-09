// DS-1000 case 505 — Risc0 host driver.

use methods::{GUEST_CODE_FOR_ZK_PROOF_ELF, GUEST_CODE_FOR_ZK_PROOF_ID};
use risc0_zkvm::{default_prover, ExecutorEnv};
use std::time::Instant;

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::filter::EnvFilter::from_default_env())
        .init();

    let mut a = [[[0f64; 5]; 13]; 13];
    for i in 0..13 {
        for j in 0..13 {
            for k in 0..5 {
                a[i][j][k] = ((i * 13 * 5 + j * 5 + k + 1) as f64) / 100.0;
            }
        }
    }
    let mut b = [[0u64; 13]; 13];
    for i in 0..13 {
        for j in 0..13 {
            b[i][j] = ((i + j) % 5) as u64;
        }
    }
    let mut sum_a: f64 = 0.0;
    for i in 0..13 {
        for j in 0..13 {
            for k in 0..5 {
                sum_a += a[i][j][k];
            }
        }
    }
    let mut sum_arr: f64 = 0.0;
    for i in 0..13 {
        for j in 0..13 {
            sum_arr += a[i][j][b[i][j] as usize];
        }
    }
    let expected: f64 = sum_a - sum_arr;

    let mut builder = ExecutorEnv::builder();
    for plane in a.iter() {
        for row in plane.iter() {
            for &v in row.iter() {
                builder.write(&v).unwrap();
            }
        }
    }
    for row in b.iter() {
        for &v in row.iter() {
            builder.write(&v).unwrap();
        }
    }
    builder.write(&expected).unwrap();
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
