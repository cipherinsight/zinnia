// DS-1000 case 373 — SP1 host driver.
// Grades: deliberately shuffled values; expected ys = i/50 for i in 1..=50.

use clap::Parser;
use sp1_sdk::{include_elf, ProverClient, SP1Stdin};
use std::time::Instant;

pub const ECDF_ELF: &[u8] = include_elf!("ecdf-y-program");

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

    // 50 grades: (50 - i) * 1.5 — a strictly decreasing permutation so sort actually does work.
    let mut grades: [f64; 50] = [0.0; 50];
    for i in 0..50 { grades[i] = (50 - i) as f64 * 1.5; }

    let mut expected: [f64; 50] = [0.0; 50];
    for i in 0..50 { expected[i] = (i as f64 + 1.0) / 50.0; }

    for &v in grades.iter() { stdin.write(&v); }
    for &v in expected.iter() { stdin.write(&v); }

    if args.execute {
        let (_o, report) = client.execute(ECDF_ELF, &stdin).run().unwrap();
        println!("executed; cycles: {}", report.total_instruction_count());
        return;
    }
    let (pk, vk) = client.setup(ECDF_ELF);
    let start = Instant::now();
    let proof = client.prove(&pk, &stdin).run().unwrap();
    println!("Prove time (zk-STARK) (ms): {:?}", start.elapsed().as_millis());
    let start = Instant::now();
    client.verify(&proof, &vk).unwrap();
    println!("Verify time (zk-STARK) (ms): {:?}", start.elapsed().as_millis());
}
