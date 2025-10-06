// Risc 0 driver code
use methods::{GUEST_CODE_FOR_ZK_PROOF_ELF, GUEST_CODE_FOR_ZK_PROOF_ID};
use risc0_zkvm::{default_prover, ExecutorEnv};
use std::time::Instant;

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::filter::EnvFilter::from_default_env())
        .init();

    let mut builder = ExecutorEnv::builder();

    // Input matrix X[5][4]
    let x: [[i32; 4]; 5] = [
        [1, -2, 3, 6],
        [4, 5, -6, 5],
        [-1, 2, 5, 5],
        [4, 5, 10, -25],
        [5, -2, 10, 25],
    ];

    for i in 0..5 {
        for j in 0..4 {
            builder.write(&x[i][j]).unwrap();
        }
    }

    // Compute expected L1-normalized result
    let mut result: [[f32; 4]; 5] = [[0.0; 4]; 5];
    let mut l1: [f32; 5] = [0.0; 5];

    for i in 0..5 {
        let mut s: f32 = 0.0;
        for j in 0..4 {
            let val = x[i][j] as f32;
            s += if val >= 0.0 { val } else { -val };
        }
        l1[i] = s;
    }

    for i in 0..5 {
        for j in 0..4 {
            result[i][j] = (x[i][j] as f32) / l1[i];
            builder.write(&result[i][j]).unwrap();
        }
    }

    let env = builder.build().unwrap();
    let prover = default_prover();

    let start = Instant::now();
    let prove_info = prover.prove(env, GUEST_CODE_FOR_ZK_PROOF_ELF).unwrap();
    let duration = start.elapsed();
    println!("Prove time (zk-STARK) (ms): {:?}", duration.as_millis());

    let receipt = prove_info.receipt;

    // let _output: u32 = receipt.journal.decode().unwrap();

    let start = Instant::now();
    receipt.verify(GUEST_CODE_FOR_ZK_PROOF_ID).unwrap();
    let duration = start.elapsed();
    println!("Verify time (zk-STARK) (ms): {:?}", duration.as_millis());
}
