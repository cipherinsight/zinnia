// Risc 0 driver code
use methods::{GUEST_CODE_FOR_ZK_PROOF_ELF, GUEST_CODE_FOR_ZK_PROOF_ID};
use risc0_zkvm::{default_prover, ExecutorEnv};
use std::time::Instant;

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::filter::EnvFilter::from_default_env())
        .init();

    let mut builder = ExecutorEnv::builder();

    // Input tensor Y[4][3][3]
    let y: [[[f32; 3]; 3]; 4] = [
        [[81.0, 63.0, 63.0], [63.0, 49.0, 49.0], [63.0, 49.0, 49.0]],
        [[4.0, 12.0, 8.0], [12.0, 36.0, 24.0], [8.0, 24.0, 16.0]],
        [[25.0, 35.0, 25.0], [35.0, 49.0, 35.0], [25.0, 35.0, 25.0]],
        [[25.0, 30.0, 10.0], [30.0, 36.0, 12.0], [10.0, 12.0, 4.0]],
    ];
    for i in 0..4 {
        for j in 0..3 {
            for k in 0..3 {
                builder.write(&y[i][j][k]).unwrap();
            }
        }
    }

    // Expected X[3][4] = sqrt(diag(Y[i]))
    let mut x: [[f32; 4]; 3] = [[0.0; 4]; 3];
    for i in 0..4 {
        for j in 0..3 {
            x[j][i] = y[i][j][j].sqrt();
        }
    }

    for i in 0..3 {
        for j in 0..4 {
            builder.write(&x[i][j]).unwrap();
        }
    }

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
