// DS-1000 case 481 — SP1 host driver.
// x = [1, 5, 5, 2, 5, 5, 3, 5, 4]; y = [2, 7, 7, 3, 8, 7, 4, 7, 5].
// (a, b) = (5, 7) → matches at indices 1, 2, 5, 7. Expected length 4.

use clap::Parser;
use sp1_sdk::{include_elf, ProverClient, SP1Stdin};
use std::time::Instant;

pub const NONZERO_AND_ELF: &[u8] = include_elf!("nonzero-and-program");

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

    let x: [i64; 9] = [1, 5, 5, 2, 5, 5, 3, 5, 4];
    let y: [i64; 9] = [2, 7, 7, 3, 8, 7, 4, 7, 5];
    let a: i64 = 5;
    let b: i64 = 7;
    let claimed_len: u64 = 4;
    let mut claimed: [u64; 9] = [0; 9];
    let expected: [u64; 4] = [1, 2, 5, 7];
    for k in 0..4 { claimed[k] = expected[k]; }

    for &v in x.iter() { stdin.write(&v); }
    for &v in y.iter() { stdin.write(&v); }
    stdin.write(&a);
    stdin.write(&b);
    stdin.write(&claimed_len);
    for &v in claimed.iter() { stdin.write(&v); }

    if args.execute {
        let (_o, report) = client.execute(NONZERO_AND_ELF, &stdin).run().unwrap();
        println!("executed; cycles: {}", report.total_instruction_count());
        return;
    }
    let (pk, vk) = client.setup(NONZERO_AND_ELF);
    let start = Instant::now();
    let proof = client.prove(&pk, &stdin).run().unwrap();
    println!("Prove time (zk-STARK) (ms): {:?}", start.elapsed().as_millis());
    proof.save("proof-with-pis-stark.bin").unwrap();
    let start = Instant::now();
    client.verify(&proof, &vk).unwrap();
    println!("Verify time (zk-STARK) (ms): {:?}", start.elapsed().as_millis());

    let start = Instant::now();
    let proof = client.prove(&pk, &stdin).plonk().run().unwrap();
    println!("Prove time (zk-SNARK) (ms): {:?}", start.elapsed().as_millis());
    proof.save("proof-with-pis.bin").unwrap();
    let start = Instant::now();
    client.verify(&proof, &vk).unwrap();
    println!("Verify time (zk-SNARK) (ms): {:?}", start.elapsed().as_millis());
}
