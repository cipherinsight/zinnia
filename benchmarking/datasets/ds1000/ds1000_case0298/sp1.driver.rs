// DS-1000 case 298 — SP1 host driver.
// Input: a = [1.5, -0.4, 1.3, 1.5, 1.3]; vals = [-0.4, 1.3, 1.5], idx = [2, 0, 1, 2, 1].

use clap::Parser;
use sp1_sdk::{include_elf, ProverClient, SP1Stdin};
use std::time::Instant;

pub const UNIQUE_ELF: &[u8] = include_elf!("unique-onehot-program");

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

    let a: [f64; 5] = [1.5, -0.4, 1.3, 1.5, 1.3];
    for &v in a.iter() {
        stdin.write(&v);
    }
    let n_unique: u64 = 3;
    stdin.write(&n_unique);
    let b: [[f64; 3]; 5] = [
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
    ];
    for row in b.iter() {
        for &v in row.iter() {
            stdin.write(&v);
        }
    }

    if args.execute {
        let (_o, report) = client.execute(UNIQUE_ELF, &stdin).run().unwrap();
        println!("executed; cycles: {}", report.total_instruction_count());
        return;
    }
    let (pk, vk) = client.setup(UNIQUE_ELF);
    let start = Instant::now();
    let proof = client.prove(&pk, &stdin).run().unwrap();
    println!("Prove time (ms): {:?}", start.elapsed().as_millis());
    let start = Instant::now();
    client.verify(&proof, &vk).unwrap();
    println!("Verify time (ms): {:?}", start.elapsed().as_millis());
}
