// DS-1000 case 0489 — SP1 host driver.

use clap::Parser;
use sp1_sdk::{include_elf, ProverClient, SP1Stdin};
use std::time::Instant;

pub const WHERE_MUT_ELF: &[u8] = include_elf!("where-mutate-program");

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
    if args.execute == args.prove { eprintln!("Error"); std::process::exit(1); }
    let client = ProverClient::from_env();
    let mut stdin = SP1Stdin::new();
    let arr: [f64; 8] = [-20.0, -5.0, 0.0, 12.0, 14.9, 15.0, 20.0, 100.0];
    let expect: [f64; 8] = [0.0, 0.0, 5.0, 17.0, 19.9, 30.0, 30.0, 30.0];
    for &v in &arr { stdin.write(&v); }
    for &v in &expect { stdin.write(&v); }
    if args.execute {
        let (_o, report) = client.execute(WHERE_MUT_ELF, &stdin).run().unwrap();
        println!("executed; cycles: {}", report.total_instruction_count());
        return;
    }
    let (pk, vk) = client.setup(WHERE_MUT_ELF);
    let start = Instant::now();
    let proof = client.prove(&pk, &stdin).run().unwrap();
    println!("Prove time (ms): {:?}", start.elapsed().as_millis());
    let start = Instant::now();
    client.verify(&proof, &vk).unwrap();
    println!("Verify time (ms): {:?}", start.elapsed().as_millis());
}
