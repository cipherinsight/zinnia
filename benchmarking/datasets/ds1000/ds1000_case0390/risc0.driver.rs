// DS-1000 case 390 — Risc0 host driver. Patch reshape over Float[100,200].

use methods::{GUEST_CODE_FOR_ZK_PROOF_ELF, GUEST_CODE_FOR_ZK_PROOF_ID};
use risc0_zkvm::{default_prover, ExecutorEnv};
use std::time::Instant;

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::filter::EnvFilter::from_default_env())
        .init();

    const H: usize = 100;
    const W: usize = 200;
    const PS: usize = 10;
    let h_blocks = H / PS;
    let w_blocks = W / PS;
    let n_patches = h_blocks * w_blocks;

    let mut a: Vec<Vec<f64>> = Vec::with_capacity(H);
    for i in 0..H {
        let mut row: Vec<f64> = Vec::with_capacity(W);
        for j in 0..W { row.push((i * 200 + j) as f64); }
        a.push(row);
    }
    let patch_size: u32 = PS as u32;

    let mut builder = ExecutorEnv::builder();
    for row in a.iter() {
        for &v in row.iter() { builder.write(&v).unwrap(); }
    }
    builder.write(&patch_size).unwrap();
    for p in 0..n_patches {
        let block_col = p / h_blocks;
        let block_row = p % h_blocks;
        for r in 0..PS {
            for c in 0..PS {
                let v = ((block_row * PS + r) * 200 + (block_col * PS + c)) as f64;
                builder.write(&v).unwrap();
            }
        }
    }
    let env = builder.build().unwrap();

    let prover = default_prover();
    let start = Instant::now();
    let prove_info = prover.prove(env, GUEST_CODE_FOR_ZK_PROOF_ELF).unwrap();
    println!("Prove time (zk-STARK) (ms): {:?}", start.elapsed().as_millis());
    let receipt = prove_info.receipt;
    let start = Instant::now();
    receipt.verify(GUEST_CODE_FOR_ZK_PROOF_ID).unwrap();
    println!("Verify time (zk-STARK) (ms): {:?}", start.elapsed().as_millis());
}
