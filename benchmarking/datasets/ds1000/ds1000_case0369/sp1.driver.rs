// DS-1000 case 369 — SP1 host driver.
// Inputs: not all rows have all-equal columns → false. Non-trivial example.

use clap::Parser;
use sp1_sdk::{include_elf, ProverClient, SP1Stdin};
use std::time::Instant;

pub const ISCLOSE_ELF: &[u8] = include_elf!("isclose-col-program");

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

    // Row 0 is constant; row 1 j=1 differs; row 2 constant → result false.
    let a: [[i64; 3]; 3] = [[5, 5, 5], [7, 8, 7], [9, 9, 9]];
    let expected_bool: u8 = 0;

    for row in a.iter() {
        for &v in row.iter() { stdin.write(&v); }
    }
    stdin.write(&expected_bool);

    if args.execute {
        let (_o, report) = client.execute(ISCLOSE_ELF, &stdin).run().unwrap();
        println!("executed; cycles: {}", report.total_instruction_count());
        return;
    }
    let (pk, vk) = client.setup(ISCLOSE_ELF);
    let start = Instant::now();
    let proof = client.prove(&pk, &stdin).run().unwrap();
    println!("Prove time (zk-STARK) (ms): {:?}", start.elapsed().as_millis());
    let start = Instant::now();
    client.verify(&proof, &vk).unwrap();
    println!("Verify time (zk-STARK) (ms): {:?}", start.elapsed().as_millis());
}
