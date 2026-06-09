// DS-1000 case 0354 — SP1 host driver.
// A = [[1,2,0],[3,1,2]]; B has A[0] present, A[1] not. Result rows = [[3,1,2]].

use clap::Parser;
use sp1_sdk::{include_elf, ProverClient, SP1Stdin};
use std::time::Instant;

pub const SETDIFF_ELF: &[u8] = include_elf!("setdiff-program");

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
    let a: [[i64; 3]; 2] = [[1, 2, 0], [3, 1, 2]];
    let b: [[i64; 3]; 7] = [
        [1, 2, 0], [0, 0, 0], [5, 5, 5], [4, 4, 4],
        [9, 9, 9], [8, 8, 8], [7, 7, 7],
    ];
    for row in &a { for &v in row { stdin.write(&v); } }
    for row in &b { for &v in row { stdin.write(&v); } }
    let claim_count: u32 = 1;
    stdin.write(&claim_count);
    let result: [[i64; 3]; 1] = [[3, 1, 2]];
    for row in &result { for &v in row { stdin.write(&v); } }
    if args.execute {
        let (_o, report) = client.execute(SETDIFF_ELF, &stdin).run().unwrap();
        println!("executed; cycles: {}", report.total_instruction_count());
        return;
    }
    let (pk, vk) = client.setup(SETDIFF_ELF);
    let start = Instant::now();
    let proof = client.prove(&pk, &stdin).run().unwrap();
    println!("Prove time (ms): {:?}", start.elapsed().as_millis());
    let start = Instant::now();
    client.verify(&proof, &vk).unwrap();
    println!("Verify time (ms): {:?}", start.elapsed().as_millis());
}
