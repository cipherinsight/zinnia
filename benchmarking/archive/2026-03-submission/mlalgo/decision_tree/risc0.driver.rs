// Risc 0 driver code
use methods::{
    GUEST_CODE_FOR_ZK_PROOF_ELF, GUEST_CODE_FOR_ZK_PROOF_ID
};
use risc0_zkvm::{default_prover, ExecutorEnv};
use std::time::Instant;

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::filter::EnvFilter::from_default_env())
        .init();

    let mut builder = ExecutorEnv::builder();

    // training_x (10x2)
    for data in [
        0.2, 0.1,
        0.5, -1.0,
        1.2, 0.0,
        1.8, -0.3,
        2.2, 1.0,
        0.7, 0.5,
        1.1, -0.2,
        2.8, 0.9,
        0.3, -0.4,
        1.6, 1.2
    ] {
        builder.write(&data).unwrap();
    }

    // training_y (10)
    for data in [0, 0, 0, 0, 1, 0, 0, 1, 0, 1] {
        let val: i32 = data;
        builder.write(&val).unwrap();
    }

    // testing_x (2x2)
    for data in [1.4, 0.0, 0.4, 2.0] {
        builder.write(&data).unwrap();
    }

    // testing_y (2)
    for data in [0, 0] {
        let val: i32 = data;
        builder.write(&val).unwrap();
    }

    let env = builder.build().unwrap();
    let prover = default_prover();

    let start = Instant::now();
    let prove_info = prover
        .prove(env, GUEST_CODE_FOR_ZK_PROOF_ELF)
        .unwrap();
    let duration = start.elapsed();
    println!("Prove time (zk-STARK) (ms): {:?}", duration.as_millis());

    let receipt = prove_info.receipt;

    // let _output: u32 = receipt.journal.decode().unwrap();

    let start = Instant::now();
    receipt.verify(GUEST_CODE_FOR_ZK_PROOF_ID).unwrap();
    let duration = start.elapsed();
    println!("Verify time (zk-STARK) (ms): {:?}", duration.as_millis());
}
