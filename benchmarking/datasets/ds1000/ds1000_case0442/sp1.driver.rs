// DS-1000 case 442 — SP1 host driver.
// A = [0..20), B = [3, 7, 13]. Expected: A without {3, 7, 13} (17 elements).

use clap::Parser;
use sp1_sdk::{include_elf, ProverClient, SP1Stdin};
use std::time::Instant;

pub const IN1D_NEG_ELF: &[u8] = include_elf!("in1d-neg-program");

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    #[clap(long)] execute: bool,
    #[clap(long)] prove: bool,
}

fn main() {
    sp1_sdk::utils::setup_logger();
    dotenv::dotenv().ok();
    let args = Args::parse();
    if args.execute == args.prove {
        eprintln!("Error: specify either --execute or --prove");
        std::process::exit(1);
    }
    let client = ProverClient::from_env();
    let mut stdin = SP1Stdin::new();

    let a: [i64; 20] = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19];
    let b: [i64; 3] = [3, 7, 13];
    let claimed: [i64; 17] = [0,1,2,4,5,6,8,9,10,11,12,14,15,16,17,18,19];
    let claimed_len: u64 = claimed.len() as u64;

    for v in a.iter() { stdin.write(v); }
    for v in b.iter() { stdin.write(v); }
    stdin.write(&claimed_len);
    for v in claimed.iter() { stdin.write(v); }

    if args.execute {
        panic!("Execution not supported in this environment.");
    } else {
        let (pk, vk) = client.setup(IN1D_NEG_ELF);
        let start = Instant::now();
        let proof = client.prove(&pk, &stdin).run().expect("failed to generate proof");
        println!("Prove time (zk-STARK) (ms): {:?}", start.elapsed().as_millis());
        proof.save("proof-with-pis-stark.bin").expect("saving proof failed");
        let start = Instant::now();
        client.verify(&proof, &vk).expect("failed to verify proof");
        println!("Verify time (zk-STARK) (ms): {:?}", start.elapsed().as_millis());

        let start = Instant::now();
        let proof = client.prove(&pk, &stdin).plonk().run().expect("failed to generate proof");
        println!("Prove time (zk-SNARK) (ms): {:?}", start.elapsed().as_millis());
        proof.save("proof-with-pis.bin").expect("saving proof failed");
        let start = Instant::now();
        client.verify(&proof, &vk).expect("failed to verify proof");
        println!("Verify time (zk-SNARK) (ms): {:?}", start.elapsed().as_millis());
    }
}
