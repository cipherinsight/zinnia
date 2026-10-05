// Risc 0 driver code
use methods::{GUEST_CODE_FOR_ZK_PROOF_ELF, GUEST_CODE_FOR_ZK_PROOF_ID};
use risc0_zkvm::{default_prover, ExecutorEnv};
use std::time::Instant;

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::filter::EnvFilter::from_default_env())
        .init();

    let mut builder = ExecutorEnv::builder();

    // Input matrix a
    let a: [[i32; 4]; 4] = [
        [0, 3, 1, 3],
        [3, 0, 0, 0],
        [1, 0, 0, 0],
        [3, 0, 0, 0],
    ];
    for i in 0..4 {
        for j in 0..4 {
            builder.write(&a[i][j]).unwrap();
        }
    }

    // Expected result
    let mut modified = a;
    let row = 1usize;
    let col = 0usize;
    for j in 0..4 {
        modified[row][j] = 0;
    }
    for i in 0..4 {
        modified[i][col] = 0;
    }

    for i in 0..4 {
        for j in 0..4 {
            builder.write(&modified[i][j]).unwrap();
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
