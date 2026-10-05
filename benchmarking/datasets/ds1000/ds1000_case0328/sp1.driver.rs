// DS-1000 case 0328 — SP1 host driver.
// A = [1..6], length = 8 → output [1,2,3,4,5,0,0,0].

use clap::Parser;
use sp1_sdk::{include_elf, ProverClient, SP1Stdin};
use std::time::Instant;

pub const PAD_TRUNC_ELF: &[u8] = include_elf!("pad-trunc-program");

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
    let a: [i64; 5] = [1, 2, 3, 4, 5];
    let length: u32 = 8;
    let expect: [i64; 8] = [1, 2, 3, 4, 5, 0, 0, 0];
    stdin.write(&length);
    for &v in &a { stdin.write(&v); }
    for &v in &expect { stdin.write(&v); }
    if args.execute {
        let (_o, report) = client.execute(PAD_TRUNC_ELF, &stdin).run().unwrap();
        println!("executed; cycles: {}", report.total_instruction_count());
        return;
    }
    let (pk, vk) = client.setup(PAD_TRUNC_ELF);
    let start = Instant::now();
    let proof = client.prove(&pk, &stdin).run().unwrap();
    println!("Prove time (ms): {:?}", start.elapsed().as_millis());
    let start = Instant::now();
    client.verify(&proof, &vk).unwrap();
    println!("Verify time (ms): {:?}", start.elapsed().as_millis());
}
