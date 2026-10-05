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
    // a = [10,20,30]
    // b = [30,20,20]
    // c = [50,20,40]
    // result = [30.0,20.0,30.0]
    for data in [10.0, 20.0, 30.0] {
        builder.write(&data).unwrap();
    }
    for data in [30.0, 20.0, 20.0] {
        builder.write(&data).unwrap();
    }
    for data in [50.0, 20.0, 40.0] {
        builder.write(&data).unwrap();
    }
    for data in [30.0, 20.0, 30.0] {
        builder.write(&data).unwrap();
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
