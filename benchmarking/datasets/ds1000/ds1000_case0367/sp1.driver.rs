// DS-1000 case 367 — SP1 host driver.
// Inputs: pairs = [[1,2],[3,4],[5,6]] (unused), array_of_arrays = [7, 8, 9].
// Expected deepcopy = [7, 8, 9].

use clap::Parser;
use sp1_sdk::{include_elf, ProverClient, SP1Stdin};
use std::time::Instant;

pub const DEEPCOPY_ELF: &[u8] = include_elf!("deepcopy-program");

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

    let pairs: [[i64; 2]; 3] = [[1, 2], [3, 4], [5, 6]];
    let array_of_arrays: [i64; 3] = [7, 8, 9];
    let expected: [i64; 3] = [7, 8, 9];

    for row in pairs.iter() {
        for &v in row.iter() { stdin.write(&v); }
    }
    for &v in array_of_arrays.iter() { stdin.write(&v); }
    for &v in expected.iter() { stdin.write(&v); }

    if args.execute {
        let (_o, report) = client.execute(DEEPCOPY_ELF, &stdin).run().unwrap();
        println!("executed; cycles: {}", report.total_instruction_count());
        return;
    }
    let (pk, vk) = client.setup(DEEPCOPY_ELF);
    let start = Instant::now();
    let proof = client.prove(&pk, &stdin).run().unwrap();
    println!("Prove time (zk-STARK) (ms): {:?}", start.elapsed().as_millis());
    let start = Instant::now();
    client.verify(&proof, &vk).unwrap();
    println!("Verify time (zk-STARK) (ms): {:?}", start.elapsed().as_millis());
}
