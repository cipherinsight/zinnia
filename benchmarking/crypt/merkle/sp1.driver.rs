use sp1_sdk::{include_elf, ProverClient, SP1Stdin};
use std::time::Instant;
use ethereum_types::U512;

pub const MERKLE_ELF: &[u8] = include_elf!("merkle_program");

fn main() {
    sp1_sdk::utils::setup_logger();
    dotenv::dotenv().ok();
    let client = ProverClient::from_env();
    let mut stdin = SP1Stdin::new();

    let leaves = [
        11u64, 22, 33, 44, 55, 66, 77, 88
    ].map(U512::from);
    let leaf_idx = U512::from(5u64);
    let path = [
        U512::from(55u64),
        U512::from_str_radix("11523261815481333160108258185116327645151901011750749413828919435813935579027", 10).unwrap(),
        U512::from_str_radix("21212311445138905926880864250103604663610259329539511211264150430276934290235", 10).unwrap(),
    ];
    let bits = [U512::from(1u64), U512::from(0u64), U512::from(1u64)];

    for v in &leaves { stdin.write(v); }
    stdin.write(&leaf_idx);
    for v in &path { stdin.write(v); }
    for v in &bits { stdin.write(v); }

    let (pk, vk) = client.setup(MERKLE_ELF);

    let start = Instant::now();
    let proof = client.prove(&pk, &stdin)
        .run()
        .expect("failed to generate proof");
    println!("Prove time (zk-STARK) (ms): {:?}", start.elapsed().as_millis());

    proof.save("proof-with-pis-stark.bin").unwrap();

    let start = Instant::now();
    client.verify(&proof, &vk).unwrap();
    println!("Verify time (zk-STARK) (ms): {:?}", start.elapsed().as_millis());
}
