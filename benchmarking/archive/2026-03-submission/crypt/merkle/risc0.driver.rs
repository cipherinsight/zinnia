use methods::{GUEST_CODE_FOR_ZK_PROOF_ELF, GUEST_CODE_FOR_ZK_PROOF_ID};
use risc0_zkvm::{default_prover, ExecutorEnv};
use std::time::Instant;
use ethereum_types::U512;

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::filter::EnvFilter::from_default_env())
        .init();

    let mut builder = ExecutorEnv::builder();

    let leaves = [
        11u64, 22, 33, 44, 55, 66, 77, 88
    ].map(U512::from);
    let leaf_idx = U512::from(5u64);
    let path = [
        U512::from(55u64),
        U512::from_str_radix("17601130510839997314447930637874687829291919088912867269545606748119523760824", 10).unwrap(),
        U512::from_str_radix("16113907914567631108494203727747775918365688705733099625555899303940350166804", 10).unwrap(),
    ];
    let bits = [U512::from(1u64), U512::from(0u64), U512::from(1u64)];

    for v in &leaves { builder.write(v).unwrap(); }
    builder.write(&leaf_idx).unwrap();
    for v in &path { builder.write(v).unwrap(); }
    for v in &bits { builder.write(v).unwrap(); }

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
