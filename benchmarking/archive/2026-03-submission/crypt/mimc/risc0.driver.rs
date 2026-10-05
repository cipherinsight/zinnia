use risc0_zkvm::{default_prover, ExecutorEnv};
use methods::{GUEST_CODE_FOR_ZK_PROOF_ELF, GUEST_CODE_FOR_ZK_PROOF_ID};
use std::time::Instant;
use ethereum_types::U512;

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::filter::EnvFilter::from_default_env())
        .init();

    let mut builder = ExecutorEnv::builder();

    // msg = [1, 2, 3]
    let msg = [
        U512::from(1u64),
        U512::from(2u64),
        U512::from(3u64),
    ];
    let expected = U512::from_str_radix(
        "13282693387779170360280659014090582903649482011954396102989514311726011132212", 10
    ).unwrap();

    for v in &msg {
        builder.write(v).unwrap();
    }
    builder.write(&expected).unwrap();

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
