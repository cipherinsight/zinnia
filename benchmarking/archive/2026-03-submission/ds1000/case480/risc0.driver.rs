// Risc 0 driver code
use methods::{GUEST_CODE_FOR_ZK_PROOF_ELF, GUEST_CODE_FOR_ZK_PROOF_ID};
use risc0_zkvm::{default_prover, ExecutorEnv};
use std::time::Instant;

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::filter::EnvFilter::from_default_env())
        .init();

    let mut builder = ExecutorEnv::builder();

    // Inputs
    let x: [i32; 9] = [0, 1, 1, 1, 3, 1, 5, 5, 5];
    let y: [i32; 9] = [0, 2, 3, 4, 2, 4, 3, 4, 5];
    let a: i32 = 1;
    let b: i32 = 4;

    for i in 0..9 {
        builder.write(&x[i]).unwrap();
    }
    for i in 0..9 {
        builder.write(&y[i]).unwrap();
    }
    builder.write(&a).unwrap();
    builder.write(&b).unwrap();

    // Compute expected result
    let mut found_index: i32 = -1;
    for i in 0..9 {
        if x[i] == a && y[i] == b && found_index == -1 {
            found_index = i as i32;
        }
    }
    let result = found_index;
    builder.write(&result).unwrap();

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
