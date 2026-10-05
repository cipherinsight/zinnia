// DS-1000 case 426 — SP1 host driver.
// a = [0..20), m = 5. Output is 20×5 bit matrix.

use clap::Parser;
use sp1_sdk::{include_elf, ProverClient, SP1Stdin};
use std::time::Instant;

pub const BITS_ELF: &[u8] = include_elf!("bitdecomp-program");

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
    let m: u32 = 5;
    let a: Vec<u64> = (0u64..20).collect();
    stdin.write(&m);
    for &v in &a { stdin.write(&v); }
    // expected
    for i in 0..20usize {
        for j in 0..m as usize {
            let shift = (m as u32) - 1 - (j as u32);
            let bit: u64 = if (a[i] & (1u64 << shift)) > 0 { 1 } else { 0 };
            stdin.write(&bit);
        }
    }
    if args.execute {
        let (_o, report) = client.execute(BITS_ELF, &stdin).run().unwrap();
        println!("executed; cycles: {}", report.total_instruction_count());
        return;
    }
    let (pk, vk) = client.setup(BITS_ELF);
    let start = Instant::now();
    let proof = client.prove(&pk, &stdin).run().unwrap();
    println!("Prove time (ms): {:?}", start.elapsed().as_millis());
    let start = Instant::now();
    client.verify(&proof, &vk).unwrap();
    println!("Verify time (ms): {:?}", start.elapsed().as_millis());
}
