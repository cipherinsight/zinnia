// Risc 0 driver code
use methods::{GUEST_CODE_FOR_ZK_PROOF_ELF, GUEST_CODE_FOR_ZK_PROOF_ID};
use risc0_zkvm::{default_prover, ExecutorEnv};
use std::time::Instant;

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::filter::EnvFilter::from_default_env())
        .init();

    let mut builder = ExecutorEnv::builder();

    // Input arrays
    let post: [f32; 4] = [2.0, 5.0, 6.0, 10.0];
    let distance: [f32; 4] = [50.0, 100.0, 500.0, 1000.0];

    for i in 0..4 {
        builder.write(&post[i]).unwrap();
    }
    for i in 0..4 {
        builder.write(&distance[i]).unwrap();
    }

    // Compute expected Pearson correlation manually
    let n: f32 = 4.0;
    let mean_post: f32 = (post[0] + post[1] + post[2] + post[3]) / n;
    let mean_distance: f32 = (distance[0] + distance[1] + distance[2] + distance[3]) / n;

    let mut cov: f32 = 0.0;
    for i in 0..4 {
        cov += (post[i] - mean_post) * (distance[i] - mean_distance);
    }
    cov /= n;

    let mut var_post: f32 = 0.0;
    let mut var_distance: f32 = 0.0;
    for i in 0..4 {
        var_post += (post[i] - mean_post).powf(2.0);
        var_distance += (distance[i] - mean_distance).powf(2.0);
    }
    var_post /= n;
    var_distance /= n;

    let std_post = var_post.sqrt();
    let std_distance = var_distance.sqrt();

    let pearson_r = cov / (std_post * std_distance);

    builder.write(&pearson_r).unwrap();

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
