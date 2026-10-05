// DS-1000 case 389 — Risc0 host driver.

use methods::{GUEST_CODE_FOR_ZK_PROOF_ELF, GUEST_CODE_FOR_ZK_PROOF_ID};
use risc0_zkvm::{default_prover, ExecutorEnv};
use std::time::Instant;

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::filter::EnvFilter::from_default_env())
        .init();

    let h: u64 = 6;
    let w: u64 = 12;
    let mut a: [[[f64; 4]; 3]; 6] = [[[0.0; 4]; 3]; 6];
    for i in 0..6 {
        for j in 0..3 {
            for k in 0..4 {
                a[i][j][k] = (i * 12 + j * 4 + k) as f64;
            }
        }
    }
    let hu = h as usize;
    let wu = w as usize;
    let mut expected: Vec<f64> = Vec::with_capacity(hu * wu);
    for out_i in 0..hu {
        for out_j in 0..wu {
            let out_idx = out_i * wu + out_j;
            let i0 = out_idx / 36;
            let i1 = (out_idx / 12) % 3;
            let i2 = (out_idx / 4) % 3;
            let i3 = out_idx % 4;
            expected.push(a[i0 * 3 + i2][i1][i3]);
        }
    }

    let mut builder = ExecutorEnv::builder();
    for plane in a.iter() {
        for row in plane.iter() {
            for &v in row.iter() { builder.write(&v).unwrap(); }
        }
    }
    builder.write(&h).unwrap();
    builder.write(&w).unwrap();
    for &v in expected.iter() { builder.write(&v).unwrap(); }
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
