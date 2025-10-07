// Risc 0 driver code
use methods::{GUEST_CODE_FOR_ZK_PROOF_ELF, GUEST_CODE_FOR_ZK_PROOF_ID};
use risc0_zkvm::{default_prover, ExecutorEnv};
use std::time::Instant;

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::filter::EnvFilter::from_default_env())
        .init();

    let mut builder = ExecutorEnv::builder();

    // === Inputs ===
    // training_x: 10x2
    let training_x: [[f64; 2]; 10] = [
        [-2.0, -1.0],
        [-2.0,  1.0],
        [-1.0, -1.0],
        [-1.0,  1.0],
        [ 0.0,  0.0],
        [ 1.0, -1.0],
        [ 1.0,  1.0],
        [ 2.0, -1.0],
        [ 2.0,  1.0],
        [ 2.0,  0.0],
    ];
    let mut training_y: [f64; 10] = [0.0; 10];
    for i in 0..10 {
        let x1 = training_x[i][0];
        let x2 = training_x[i][1];
        let t = x1 + 0.5 * x2 + 1.0;
        training_y[i] = t * t + 0.5;
    }

    // testing_x: 2x2
    let testing_x: [[f64; 2]; 2] = [
        [-1.0, 0.0],
        [ 1.0, 2.0],
    ];
    let mut testing_y: [f64; 2] = [0.0; 2];
    for i in 0..2 {
        let x1 = testing_x[i][0];
        let x2 = testing_x[i][1];
        let t = x1 + 0.5 * x2 + 1.0;
        testing_y[i] = t * t + 0.5;
    }

    // Write all data into executor environment
    for i in 0..10 {
        builder.write(&training_x[i][0]).unwrap();
        builder.write(&training_x[i][1]).unwrap();
    }
    for i in 0..10 {
        builder.write(&training_y[i]).unwrap();
    }
    for i in 0..2 {
        builder.write(&testing_x[i][0]).unwrap();
        builder.write(&testing_x[i][1]).unwrap();
    }
    for i in 0..2 {
        builder.write(&testing_y[i]).unwrap();
    }

    let env = builder.build().unwrap();
    let prover = default_prover();

    let start = Instant::now();
    let prove_info = prover.prove(env, GUEST_CODE_FOR_ZK_PROOF_ELF).unwrap();
    let duration = start.elapsed();
    println!("Prove time (zk-STARK) (ms): {:?}", duration.as_millis());

    let receipt = prove_info.receipt;

    let start = Instant::now();
    receipt.verify(GUEST_CODE_FOR_ZK_PROOF_ID).unwrap();
    let duration = start.elapsed();
    println!("Verify time (zk-STARK) (ms): {:?}", duration.as_millis());
}
