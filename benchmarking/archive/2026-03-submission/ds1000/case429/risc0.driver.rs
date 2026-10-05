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

    // Input array:
    // a = [0,1,2,5,6,7,8,8,8,10,29,32,45]
    let a: [f32; 13] = [0.0, 1.0, 2.0, 5.0, 6.0, 7.0, 8.0, 8.0, 8.0, 10.0, 29.0, 32.0, 45.0];
    for v in a {
        builder.write(&v).unwrap();
    }

    // Compute expected interval
    let n: f32 = 13.0;
    let mut sum = 0.0;
    for v in a {
        sum += v;
    }
    let mean_val = sum / n;
    let mut var_sum = 0.0;
    for v in a {
        var_sum += (v - mean_val) * (v - mean_val);
    }
    let variance = var_sum / n;
    let std_val = variance.sqrt();
    let lower = mean_val - 2.0 * std_val;
    let upper = mean_val + 2.0 * std_val;

    builder.write(&lower).unwrap();
    builder.write(&upper).unwrap();

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
