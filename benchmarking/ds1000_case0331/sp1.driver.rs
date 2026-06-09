// DS-1000 case 331 — SP1 host driver.
// Inputs: numerator=24, denominator=18; gcd=6; expected (4, 3).

use clap::Parser;
use sp1_sdk::{include_elf, ProverClient, SP1Stdin};
use std::time::Instant;

pub const GCD_ELF: &[u8] = include_elf!("gcd-program");

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
    let num: u64 = 24;
    let den: u64 = 18;
    let claim_n: u64 = 4;
    let claim_d: u64 = 3;
    stdin.write(&num);
    stdin.write(&den);
    stdin.write(&claim_n);
    stdin.write(&claim_d);

    if args.execute {
        let (_o, report) = client.execute(GCD_ELF, &stdin).run().unwrap();
        println!("executed; cycles: {}", report.total_instruction_count());
        return;
    }
    let (pk, vk) = client.setup(GCD_ELF);
    let start = Instant::now();
    let proof = client.prove(&pk, &stdin).run().unwrap();
    println!("Prove time (ms): {:?}", start.elapsed().as_millis());
    let start = Instant::now();
    client.verify(&proof, &vk).unwrap();
    println!("Verify time (ms): {:?}", start.elapsed().as_millis());
}
