use methods::{GUEST_CODE_FOR_ZK_PROOF_ELF, GUEST_CODE_FOR_ZK_PROOF_ID};
use risc0_zkvm::{default_prover, ExecutorEnv};
use std::time::Instant;
use ethereum_types::U512;

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::filter::EnvFilter::from_default_env())
        .init();

    let mut builder = ExecutorEnv::builder();

    let g = U512::from(5u64);
    let sk = U512::from_str_radix("123456789123456789123456789123456789", 10).unwrap();
    let r  = U512::from_str_radix("987654321987654321987654321987654321", 10).unwrap();
    let msg = U512::from_str_radix("42424242424242424242", 10).unwrap();

    builder.write(&g).unwrap();
    builder.write(&sk).unwrap();
    builder.write(&r).unwrap();
    builder.write(&msg).unwrap();

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
