// DS-1000 case 327 — SP1 host driver.
// A = [0.1..1.0], length = 15 → pad 5 zeros.

use clap::Parser;
use sp1_sdk::{include_elf, ProverClient, SP1Stdin};
use std::time::Instant;

pub const PAD_ELF: &[u8] = include_elf!("pad-runtime-program");

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
    let a: Vec<f64> = (0..10).map(|i| (i as f64 + 1.0) * 0.1).collect();
    let length: u32 = 15;
    stdin.write(&length);
    for &v in &a { stdin.write(&v); }
    for i in 0..length as usize {
        let v: f64 = if i < a.len() { a[i] } else { 0.0 };
        stdin.write(&v);
    }
    if args.execute {
        let (_o, report) = client.execute(PAD_ELF, &stdin).run().unwrap();
        println!("executed; cycles: {}", report.total_instruction_count());
        return;
    }
    let (pk, vk) = client.setup(PAD_ELF);
    let start = Instant::now();
    let proof = client.prove(&pk, &stdin).run().unwrap();
    println!("Prove time (ms): {:?}", start.elapsed().as_millis());
    let start = Instant::now();
    client.verify(&proof, &vk).unwrap();
    println!("Verify time (ms): {:?}", start.elapsed().as_millis());
}
