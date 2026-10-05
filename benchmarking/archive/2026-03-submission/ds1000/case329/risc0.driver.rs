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
    // a = [[0,1],[2,3]]
    // power = 5
    // result = [[0,1],[32,243]]
    for data in [
        0, 1,
        2, 3
    ] {
        let tmp: i32 = data as i32;
        builder.write(&tmp).unwrap();
    }
    let power: i32 = 5;
    builder.write(&power).unwrap();
    for data in [
        0, 1,
        32, 243
    ] {
        let tmp: i32 = data as i32;
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
