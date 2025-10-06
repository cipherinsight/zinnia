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

    // Input data:
    // x = [-2, -1.4, -1.1, 0, 1.2, 2.2, 3.1, 4.4, 8.3, 9.9, 10, 14, 16.2]
    // result = [0, 1.2, 2.2, 3.1, 4.4, 8.3, 9.9, 10, 14, 16.2]
    for data in [
        -2.0, -1.4, -1.1, 0.0, 1.2, 2.2, 3.1, 4.4, 8.3, 9.9, 10.0, 14.0, 16.2
    ] {
        let tmp: f32 = data as f32;
        builder.write(&tmp).unwrap();
    }
    for data in [
        0.0, 1.2, 2.2, 3.1, 4.4, 8.3, 9.9, 10.0, 14.0, 16.2
    ] {
        let tmp: f32 = data as f32;
        builder.write(&tmp).unwrap();
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
