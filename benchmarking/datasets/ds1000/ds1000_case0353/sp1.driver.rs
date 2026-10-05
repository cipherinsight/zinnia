// DS-1000 case 353 — SP1 host driver.

use clap::Parser;
use sp1_sdk::{include_elf, ProverClient, SP1Stdin};
use std::time::Instant;

pub const SETDIFF_ELF: &[u8] = include_elf!("setdiff-rows-program");

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

    // Row 0 of A appears in B (matches B[2]); row 1 of A is unique -> kept.
    let a: [[i64; 3]; 2] = [
        [1, 2, 3],
        [4, 5, 6],
    ];
    let b: [[i64; 3]; 7] = [
        [0, 0, 0],
        [1, 1, 1],
        [1, 2, 3],
        [2, 2, 2],
        [3, 3, 3],
        [7, 8, 9],
        [9, 9, 9],
    ];
    for r in a.iter() { for &v in r.iter() { stdin.write(&v); } }
    for r in b.iter() { for &v in r.iter() { stdin.write(&v); } }
    let claim_n: u64 = 1;
    stdin.write(&claim_n);
    let expected_rows: [[i64; 3]; 1] = [[4, 5, 6]];
    for r in expected_rows.iter() { for &v in r.iter() { stdin.write(&v); } }

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
