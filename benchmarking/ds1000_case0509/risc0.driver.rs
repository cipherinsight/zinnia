// DS-1000 case 509 — Risc0 host driver.

use methods::{GUEST_CODE_FOR_ZK_PROOF_ELF, GUEST_CODE_FOR_ZK_PROOF_ID};
use risc0_zkvm::{default_prover, ExecutorEnv};
use std::time::Instant;

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::filter::EnvFilter::from_default_env())
        .init();

    let zeros: [(usize, usize); 4] = [(1, 2), (1, 6), (4, 2), (4, 6)];
    let mut im = [[0f64; 10]; 10];
    for i in 0..10 {
        for j in 0..10 {
            im[i][j] = ((i * 10 + j + 1) as f64) / 10.0;
        }
    }
    for &(i, j) in zeros.iter() {
        im[i][j] = 0.0;
    }

    let rmin: u64 = 1;
    let rmax: u64 = 4;
    let cmin: u64 = 2;
    let cmax: u64 = 6;
    let mut expected: Vec<f64> = Vec::with_capacity(20);
    for i in (rmin as usize)..=(rmax as usize) {
        for j in (cmin as usize)..=(cmax as usize) {
            expected.push(im[i][j]);
        }
    }

    let mut builder = ExecutorEnv::builder();
    for row in im.iter() {
        for &v in row.iter() {
            builder.write(&v).unwrap();
        }
    }
    builder.write(&rmin).unwrap();
    builder.write(&rmax).unwrap();
    builder.write(&cmin).unwrap();
    builder.write(&cmax).unwrap();
    for &v in expected.iter() {
        builder.write(&v).unwrap();
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
