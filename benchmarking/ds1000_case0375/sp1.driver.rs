// DS-1000 case 375 — SP1 host driver.
// grades = 1..=50; threshold = 0.5. low/high computed but discarded; expected = grades.

use clap::Parser;
use sp1_sdk::{include_elf, ProverClient, SP1Stdin};
use std::time::Instant;

pub const ECDF_ELF: &[u8] = include_elf!("ecdf-bounds-program");

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

    let mut grades: [f64; 50] = [0.0; 50];
    for i in 0..50 { grades[i] = (i + 1) as f64; }
    let threshold: f64 = 0.5;
    let expected: [f64; 50] = grades;

    for &v in grades.iter() { stdin.write(&v); }
    stdin.write(&threshold);
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
