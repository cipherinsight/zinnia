// Risc 0 driver code
use methods::{
    GUEST_CODE_FOR_ZK_PROOF_ELF, GUEST_CODE_FOR_ZK_PROOF_ID
};
use risc0_zkvm::{default_prover, ExecutorEnv};
use std::time::Instant;

fn main() {
    // Initialize tracing. In order to view logs, run `RUST_LOG=info cargo run`
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::filter::EnvFilter::from_default_env())
        .init();

    let mut builder = ExecutorEnv::builder();

    // grades (27)
    for data in [
        60.8_f64, 61.0, 65.5, 69.0, 76.0, 76.0, 78.0, 78.0, 82.0,
        86.0, 87.5, 89.5, 91.0, 91.5, 92.3, 92.5, 92.8, 93.0,
        93.5, 93.5, 94.5, 94.5, 95.0, 95.5, 98.0, 98.5, 99.5
    ] {
        let tmp: f64 = data as f64;
        builder.write(&tmp).unwrap();
    }

    // threshold (scalar)
    let threshold: f64 = 0.5_f64;
    builder.write(&threshold).unwrap();

    // low, high (scalars)
    let low: f64 = 60.8_f64;
    builder.write(&low).unwrap();
    let high: f64 = 91.5_f64;
    builder.write(&high).unwrap();

    let env = builder
        .build()
        .unwrap();

    // Obtain the default prover.
    let prover = default_prover();

    // Proof information by proving the specified ELF binary.
    // This struct contains the receipt along with statistics about execution of the guest
    let start = Instant::now();
    let prove_info = prover
        .prove(env, GUEST_CODE_FOR_ZK_PROOF_ELF)
        .unwrap();
    let duration = start.elapsed();
    println!("Prove time (zk-STARK) (ms): {:?}", duration.as_millis());

    // extract the receipt.
    let receipt = prove_info.receipt;

    // TODO: Implement code for retrieving receipt journal here.
    // let _output: u32 = receipt.journal.decode().unwrap();

    // The receipt was verified at the end of proving, but the below code is an
    // example of how someone else could verify this receipt.
    let start = Instant::now();
    receipt
        .verify(GUEST_CODE_FOR_ZK_PROOF_ID)
        .unwrap();
    let duration = start.elapsed();
    println!("Verify time (zk-STARK) (ms): {:?}", duration.as_millis());
}
