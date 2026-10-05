// DS-1000 case 0296 — SP1 host driver.
// a = [i%4 for i in 0..50]. max=3 → output 50×4 one-hot.

use clap::Parser;
use sp1_sdk::{include_elf, ProverClient, SP1Stdin};
use std::time::Instant;

pub const ONEHOT_ELF: &[u8] = include_elf!("onehot-program");

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
    let a: Vec<u64> = (0u64..50).map(|i| i % 4).collect();
    let cols: u32 = 4;
    stdin.write(&cols);
    for &v in &a { stdin.write(&v); }
    for (i, _) in a.iter().enumerate() {
        for c in 0..cols as usize {
            let v: u64 = if c == a[i] as usize { 1 } else { 0 };
            stdin.write(&v);
        }
    }
    if args.execute {
        let (_o, report) = client.execute(ONEHOT_ELF, &stdin).run().unwrap();
        println!("executed; cycles: {}", report.total_instruction_count());
        return;
    }
    let (pk, vk) = client.setup(ONEHOT_ELF);
    let start = Instant::now();
    let proof = client.prove(&pk, &stdin).run().unwrap();
    println!("Prove time (ms): {:?}", start.elapsed().as_millis());
    let start = Instant::now();
    client.verify(&proof, &vk).unwrap();
    println!("Verify time (ms): {:?}", start.elapsed().as_millis());
}
