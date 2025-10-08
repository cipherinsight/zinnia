use sp1_sdk::{include_elf, ProverClient, SP1Stdin};
use std::time::Instant;
use ethereum_types::U512;

pub const MIMC_ELF: &[u8] = include_elf!("mimc3_program");

fn main() {
    sp1_sdk::utils::setup_logger();
    dotenv::dotenv().ok();
    let client = ProverClient::from_env();
    let mut stdin = SP1Stdin::new();

    // Inputs: msg = [1, 2, 3]
    let msg = [
        U512::from(1u64),
        U512::from(2u64),
        U512::from(3u64)
    ];
    let expected = U512::from_str_radix(
        "13282693387779170360280659014090582903649482011954396102989514311726011132212", 10
    ).unwrap();

    for v in &msg {
        stdin.write(v);
    }
    stdin.write(&expected);

    let (pk, vk) = client.setup(MIMC_ELF);

    let start = Instant::now();
    let proof = client.prove(&pk, &stdin)
        .run()
        .expect("failed to generate proof");
    println!("Prove time (ms): {:?}", start.elapsed().as_millis());

    proof.save("proof-with-pis-stark.bin").unwrap();

    let start = Instant::now();
    client.verify(&proof, &vk).unwrap();
    println!("Verify time (ms): {:?}", start.elapsed().as_millis());
}
