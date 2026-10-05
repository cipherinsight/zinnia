use methods::{
    GUEST_CODE_FOR_ZK_PROOF_ELF, GUEST_CODE_FOR_ZK_PROOF_ID
};
use risc0_zkvm::{default_prover, ExecutorEnv};
use std::time::Instant;
use ethereum_types::U512;

fn main() {
    // Initialize tracing. In order to view logs, run `RUST_LOG=info cargo run`
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::filter::EnvFilter::from_default_env())
        .init();

    // An executor environment describes the configurations for the zkVM
    // including program inputs.
    // A default ExecutorEnv can be created like so:
    // `let env = ExecutorEnv::builder().build().unwrap();`
    // However, this `env` does not have any inputs.
    //
    // To add guest input to the executor environment, use
    // ExecutorEnvBuilder::write().
    // To access this method, you'll need to use ExecutorEnv::builder(), which
    // creates an ExecutorEnvBuilder. When you're done adding input, call
    // ExecutorEnvBuilder::build().

    // For example:
    let mut builder = ExecutorEnv::builder();

    let x1 = U512::from_str_radix("995203441582195749578291179787384436505546430278305826713579947235728471134", 10).unwrap();
    for i in 0..8 {
        builder.write(&x1.0[i]);
    }
    let y1 = U512::from_str_radix("5472060717959818805561601436314318772137091100104008585924551046643952123905", 10).unwrap();
    for i in 0..8 {
        builder.write(&y1.0[i]);
    }
    let x2 = U512::from_str_radix("5299619240641551281634865583518297030282874472190772894086521144482721001553", 10).unwrap();
    for i in 0..8 {
        builder.write(&x2.0[i]);
    }
    let y2 = U512::from_str_radix("16950150798460657717958625567821834550301663161624707787222815936182638968203", 10).unwrap();
    for i in 0..8 {
        builder.write(&y2.0[i]);
    }
    let x3 = U512::from_str_radix("14805543388578810117460687107379140748822348273316260688573060998934016770136", 10).unwrap();
    for i in 0..8 {
        builder.write(&x3.0[i]);
    }
    let y3 = U512::from_str_radix("13589798946988221969763682225123791336245855044059976312385135587934609470572", 10).unwrap();
    for i in 0..8 {
        builder.write(&y3.0[i]);
    }

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

    // For example:
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