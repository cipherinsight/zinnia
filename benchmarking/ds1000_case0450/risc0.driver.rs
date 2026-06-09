// DS-1000 case 450 — Risc0 host driver.

use methods::{GUEST_CODE_FOR_ZK_PROOF_ELF, GUEST_CODE_FOR_ZK_PROOF_ID};
use risc0_zkvm::{default_prover, ExecutorEnv};
use std::time::Instant;

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::filter::EnvFilter::from_default_env())
        .init();

    let mut a: [[[f64; 9]; 8]; 7] = [[[0.0; 9]; 8]; 7];
    for i in 0..7 {
        for j in 0..8 {
            for k in 0..9 {
                a[i][j][k] = (i * 100 + j * 10 + k) as f64;
            }
        }
    }
    let second: [i64; 2] = [2, 5];
    let third: [i64; 2] = [3, 7];
    let mut claimed: [[[f64; 2]; 2]; 7] = [[[0.0; 2]; 2]; 7];
    for i in 0..7 {
        for p in 0..2 {
            for q in 0..2 {
                let j = second[p] as usize;
                let k = third[q] as usize;
                claimed[i][p][q] = a[i][j][k];
            }
        }
    }

    let mut builder = ExecutorEnv::builder();
    for i in 0..7 {
        for j in 0..8 {
            for k in 0..9 {
                builder.write(&a[i][j][k]).unwrap();
            }
        }
    }
    for v in second.iter() { builder.write(v).unwrap(); }
    for v in third.iter() { builder.write(v).unwrap(); }
    for plane in claimed.iter() {
        for row in plane.iter() {
            for &v in row.iter() { builder.write(&v).unwrap(); }
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
