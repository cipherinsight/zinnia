// DS-1000 case 358 — Risc0 host driver.

use methods::{GUEST_CODE_FOR_ZK_PROOF_ELF, GUEST_CODE_FOR_ZK_PROOF_ID};
use risc0_zkvm::{default_prover, ExecutorEnv};
use std::time::Instant;

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::filter::EnvFilter::from_default_env())
        .init();
    let mut a: [[[f64; 3]; 3]; 3] = [[[0.0; 3]; 3]; 3];
    for j in 0..3 { for k in 0..3 {
        a[0][j][k] = 3.0;
        a[1][j][k] = 1.0;
        a[2][j][k] = 2.0;
    } }
    let mut b: [[[i64; 3]; 3]; 3] = [[[0; 3]; 3]; 3];
    for i in 0..3 { for j in 0..3 { for k in 0..3 { b[i][j][k] = (100 * i + 10 * j + k) as i64; } } }
    let perm = [1usize, 2, 0];
    let mut expected: [[[i64; 3]; 3]; 3] = [[[0; 3]; 3]; 3];
    for i in 0..3 { for j in 0..3 { for k in 0..3 { expected[i][j][k] = b[perm[i]][j][k]; } } }

    let mut builder = ExecutorEnv::builder();
    for i in 0..3 { for j in 0..3 { for k in 0..3 { builder.write(&a[i][j][k]).unwrap(); } } }
    for i in 0..3 { for j in 0..3 { for k in 0..3 { builder.write(&b[i][j][k]).unwrap(); } } }
    for i in 0..3 { for j in 0..3 { for k in 0..3 { builder.write(&expected[i][j][k]).unwrap(); } } }
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
