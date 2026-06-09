// DS-1000 case 360 — SP1 host driver.

use clap::Parser;
use sp1_sdk::{include_elf, ProverClient, SP1Stdin};
use std::time::Instant;

pub const DELROW_ELF: &[u8] = include_elf!("delete-row-4x4-program");

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

    let a: [[f64; 4]; 4] = [
        [1.1, 1.2, 1.3, 1.4],
        [2.1, 2.2, 2.3, 2.4],
        [3.1, 3.2, 3.3, 3.4],
        [4.1, 4.2, 4.3, 4.4],
    ];
    let expected: [[f64; 4]; 3] = [
        [1.1, 1.2, 1.3, 1.4],
        [2.1, 2.2, 2.3, 2.4],
        [4.1, 4.2, 4.3, 4.4],
    ];

    for r in a.iter() { for &v in r.iter() { stdin.write(&v); } }
    for r in expected.iter() { for &v in r.iter() { stdin.write(&v); } }

    if args.execute {
        let (_o, report) = client.execute(DELROW_ELF, &stdin).run().unwrap();
        println!("executed; cycles: {}", report.total_instruction_count());
        return;
    }
    let (pk, vk) = client.setup(DELROW_ELF);
    let start = Instant::now();
    let proof = client.prove(&pk, &stdin).run().unwrap();
    println!("Prove time (ms): {:?}", start.elapsed().as_millis());
    let start = Instant::now();
    client.verify(&proof, &vk).unwrap();
    println!("Verify time (ms): {:?}", start.elapsed().as_millis());
}
