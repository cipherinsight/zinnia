// DS-1000 case 291 — SP1 host driver.
// Feeds 100 integer cells and the claimed shape tuple (2, 5, 5, 2).

use clap::Parser;
use sp1_sdk::{include_elf, ProverClient, SP1Stdin};
use std::time::Instant;

pub const SHAPE_ELF: &[u8] = include_elf!("shape-program");

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

    // Deterministic 100-cell tensor (2×5×5×2). Cell value = linear index + 1.
    for i in 0i64..100 {
        let v: i64 = i + 1;
        stdin.write(&v);
    }
    let d0: u64 = 2;
    let d1: u64 = 5;
    let d2: u64 = 5;
    let d3: u64 = 2;
    stdin.write(&d0);
    stdin.write(&d1);
    stdin.write(&d2);
    stdin.write(&d3);

    if args.execute {
        let (_o, report) = client.execute(SHAPE_ELF, &stdin).run().unwrap();
        println!("executed; cycles: {}", report.total_instruction_count());
        return;
    }
    let (pk, vk) = client.setup(SHAPE_ELF);
    let start = Instant::now();
    let proof = client.prove(&pk, &stdin).run().unwrap();
    println!("Prove time (ms): {:?}", start.elapsed().as_millis());
    let start = Instant::now();
    client.verify(&proof, &vk).unwrap();
    println!("Verify time (ms): {:?}", start.elapsed().as_millis());
}
