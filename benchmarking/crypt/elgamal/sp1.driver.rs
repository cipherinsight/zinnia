use sp1_sdk::{include_elf, ProverClient, SP1Stdin};
use std::time::Instant;
use ethereum_types::U512;

pub const ELGAMAL_ELF: &[u8] = include_elf!("elgamal_program");

fn main() {
    sp1_sdk::utils::setup_logger();
    dotenv::dotenv().ok();
    let client = ProverClient::from_env();
    let mut stdin = SP1Stdin::new();

    let g = U512::from(5u64);
    let sk = U512::from_str_radix("123456789123456789123456789123456789", 10).unwrap();
    let r  = U512::from_str_radix("987654321987654321987654321987654321", 10).unwrap();
    let msg = U512::from_str_radix("42424242424242424242", 10).unwrap();

    stdin.write(&g);
    stdin.write(&sk);
    stdin.write(&r);
    stdin.write(&msg);

    let (pk, vk) = client.setup(ELGAMAL_ELF);

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
