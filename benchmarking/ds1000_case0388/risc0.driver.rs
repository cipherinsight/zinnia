// DS-1000 case 388 — Risc0 host driver.

use methods::{GUEST_CODE_FOR_ZK_PROOF_ELF, GUEST_CODE_FOR_ZK_PROOF_ID};
use risc0_zkvm::{default_prover, ExecutorEnv};
use std::time::Instant;

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::filter::EnvFilter::from_default_env())
        .init();

    const H: usize = 100;
    const W: usize = 200;
    let p: u64 = 4;
    let pu: usize = p as usize;

    let mut a: Vec<Vec<f64>> = Vec::with_capacity(H);
    for i in 0..H {
        let mut row: Vec<f64> = Vec::with_capacity(W);
        for j in 0..W { row.push(i as f64 * 0.001 + j as f64 * 0.0001); }
        a.push(row);
    }
    let h_eff = (H / pu) * pu;
    let w_eff = (W / pu) * pu;
    let r_blocks = h_eff / pu;
    let c_blocks = w_eff / pu;
    let mut expected: Vec<f64> = Vec::with_capacity(r_blocks * c_blocks * pu * pu);
    for r_block in 0..r_blocks {
        for c_block in 0..c_blocks {
            for pp in 0..pu {
                for qq in 0..pu {
                    expected.push(a[r_block * pu + pp][c_block * pu + qq]);
                }
            }
        }
    }

    let mut builder = ExecutorEnv::builder();
    for row in a.iter() { for &v in row.iter() { builder.write(&v).unwrap(); } }
    builder.write(&p).unwrap();
    for &v in expected.iter() { builder.write(&v).unwrap(); }
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
