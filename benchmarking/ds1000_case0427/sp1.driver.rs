// DS-1000 case 427 — SP1 host driver.
// a = [1..21], m = 5. XOR(1..20) = 20 = 0b10100. Output: [1,0,1,0,0].

use clap::Parser;
use sp1_sdk::{include_elf, ProverClient, SP1Stdin};
use std::time::Instant;

pub const XOR_ELF: &[u8] = include_elf!("xor-bits-program");

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
    let a: Vec<u64> = (1u64..=20).collect();
    stdin.write(&m);
    for &v in &a { stdin.write(&v); }
    let mut res: u64 = 0;
    for v in &a { res ^= v; }
    for j in 0..m as usize {
        let shift = (m as u32) - 1 - (j as u32);
        let bit: u64 = if (res & (1u64 << shift)) > 0 { 1 } else { 0 };
        stdin.write(&bit);
    }
    if args.execute {
        let (_o, report) = client.execute(XOR_ELF, &stdin).run().unwrap();
        println!("executed; cycles: {}", report.total_instruction_count());
        return;
    }
    let (pk, vk) = client.setup(XOR_ELF);
    let start = Instant::now();
    let proof = client.prove(&pk, &stdin).run().unwrap();
    println!("Prove time (ms): {:?}", start.elapsed().as_millis());
    let start = Instant::now();
    client.verify(&proof, &vk).unwrap();
    println!("Verify time (ms): {:?}", start.elapsed().as_millis());
}
