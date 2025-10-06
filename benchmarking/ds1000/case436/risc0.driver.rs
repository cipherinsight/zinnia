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
    let a: [[i32; 2]; 3] = [
        [0, 1],
        [2, 1],
        [4, 8],
    ];
    for i in 0..3 {
        for j in 0..2 {
            builder.write(&a[i][j]).unwrap();
        }
    }

    // Expected mask
    let mut mask: [[i32; 2]; 3] = [[0; 2]; 3];
    for i in 0..3 {
        let mut row_max = a[i][0];
        for j in 1..2 {
            if a[i][j] > row_max {
                row_max = a[i][j];
            }
        }
        for j in 0..2 {
            mask[i][j] = if a[i][j] == row_max { 1 } else { 0 };
        }
    }

    for i in 0..3 {
        for j in 0..2 {
            builder.write(&mask[i][j]).unwrap();
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
